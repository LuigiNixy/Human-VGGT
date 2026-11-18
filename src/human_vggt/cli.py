import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from human_vggt.config import Config, load_config
from human_vggt.data.io import (
    prepare_sample,
    read_calibrated_sample,
    read_rgb_sample,
    sample_manifest,
)
from human_vggt.data.synthetic import make_sample
from human_vggt.evaluation.correspondence import evaluate_pairs
from human_vggt.evaluation.reconstruction import (
    evaluate_reconstruction,
    export_prediction,
)
from human_vggt.models.fixture import fixture_features, fixture_prediction
from human_vggt.models.upstream import model_manifest
from human_vggt.provenance import (
    Run,
    environment,
    git,
    resolve_device,
    seed_models,
    write_json,
)


def execute(config: Config, command: str, argv: list[str]) -> None:
    use_models = command != "prepare" and not config.mock
    with Run(config, argv) as run:
        if command != "prepare":
            config = config.resolved(
                resolve_device(config.device, config.precision, use_models)
            )
            run.config = config
            run.details["resolved_device"] = config.device
            import yaml

            (run.path / "config.yaml").write_text(
                yaml.safe_dump(config.dictionary(), sort_keys=True)
            )
        if config.dataset == "synthetic":
            sample = make_sample(config.num_views, config.seed)
        elif config.dataset == "calibrated":
            sample = read_calibrated_sample(Path(config.rgb_dir), config.num_views)
        else:
            sample = read_rgb_sample(Path(config.rgb_dir), config.num_views)
        images = prepare_sample(sample, run.path / "fixture")
        manifest = run.path / "dataset_manifest.json"
        write_json(manifest, sample_manifest(sample, images))
        run.details["dataset_manifest_sha256"] = hashlib.sha256(
            manifest.read_bytes()
        ).hexdigest()
        if command == "prepare":
            write_json(
                run.path / "metrics.json", {"status": "prepared", "views": len(images)}
            )
            return

        tracks = {"fixture": config.mock, "dataset": config.dataset}
        prediction = None
        parity = None
        if use_models:
            from human_vggt.models.vggt_adapter import VGGTAdapter

            checkpoints = model_manifest()["checkpoints"]
            run.details["checkpoints"] = {"vggt": checkpoints["vggt"]}
            seed_models(config.seed)
            model = VGGTAdapter(Path(config.weights), config.device, config.precision)
            prediction = model.predict(images, config.feature_sites)
            grids = prediction.features
            run.details["vggt"] = prediction.model_info
            if config.task == "baseline":
                parity = model.check_parity(
                    images, prediction, config.parity_atol, config.parity_rtol
                )
            del model
        else:
            grids = fixture_features(sample) if config.task == "correspondence" else ()
            if config.task == "baseline":
                prediction = fixture_prediction(sample)

        if config.sapiens:
            from human_vggt.models.sapiens_adapter import SapiensAdapter

            run.details["checkpoints"]["sapiens2"] = checkpoints["sapiens2"]
            teacher = SapiensAdapter(
                Path(config.weights), config.device, config.precision
            )
            grids += teacher.extract(images, config.sapiens_layers)
            run.details["sapiens2"] = {"parameter_count": teacher.parameter_count}
        run.details["features"] = [
            {
                "site": grid.site,
                "layer": grid.layer,
                "shape": list(grid.values.shape),
                "patch_size": grid.patch_size,
                "transforms": [asdict(t) for t in grid.transforms],
            }
            for grid in grids
        ]
        if config.task == "baseline":
            export_prediction(run.path / "prediction.npz", prediction)
            metrics, rows = evaluate_reconstruction(
                sample, prediction, known_metric=config.mock
            )
            metrics = tracks | metrics | {"official_parity": parity}
        else:
            from human_vggt.evaluation.plots import plot_correspondence

            rows = evaluate_pairs(
                sample,
                grids,
                config.samples_per_pair,
                config.occlusion_tolerance_m,
                config.seed,
            )
            plot_correspondence(rows, run.path / "figures" / "recall_by_angle.png")
            metrics = tracks | {
                "status": "ok"
                if all(row["status"] == "ok" for row in rows)
                else "incomplete",
                "pairs_by_site": rows,
            }
        run.write_rows(rows)
        write_json(run.path / "metrics.json", metrics)
        if parity is not None and not parity["passed"]:
            raise RuntimeError("Official VGGT parity failed; see metrics.json")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m human_vggt.cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor")
    for command in ("prepare", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument(
            "--set", action="append", default=[], metavar="KEY=VALUE"
        )
    args = parser.parse_args(argv)
    if args.command == "doctor":
        report = environment() | {
            "git_sha": git("rev-parse", "HEAD"),
            "upstream": git("submodule", "status", "--recursive"),
        }
        print(json.dumps(report, indent=2))
    else:
        config = load_config(args.config, args.set)
        command_line = [sys.executable, "-m", "human_vggt.cli", *(argv or sys.argv[1:])]
        execute(config, args.command, command_line)


if __name__ == "__main__":
    main()
