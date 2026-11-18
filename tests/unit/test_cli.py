import csv
import json
from pathlib import Path

import numpy as np
import pytest

from human_vggt.cli import main
from human_vggt.config import load_config


def test_config_and_environment(tmp_path: Path, monkeypatch):
    file = tmp_path / "config.yaml"
    file.write_text("experiment: test\ntask: baseline\nruns: ${RUN_TEST_ROOT}\n")
    monkeypatch.setenv("RUN_TEST_ROOT", str(tmp_path))
    assert load_config(file, ["num_views=2"]).runs == str(tmp_path)
    with pytest.raises(ValueError, match="Unknown"):
        load_config(file, ["typo=true"])
    with pytest.raises(ValueError, match="Invalid type"):
        load_config(file, ["seed=true"])
    with pytest.raises(ValueError, match="Invalid list"):
        load_config(file, ["feature_sites=[1]"])


def test_e00_command_writes_unique_runs_and_distinct_arrays(tmp_path: Path):
    argv = [
        "--config",
        "configs/experiment/e00_synthetic.yaml",
        "--set",
        f"runs={tmp_path}",
    ]
    main(["prepare", *argv])
    main(["evaluate", *argv])
    runs = list((tmp_path / "e00_synthetic").iterdir())
    assert len(runs) == 2
    evaluated = next(run for run in runs if (run / "prediction.npz").exists())
    metrics = json.loads((evaluated / "metrics.json").read_text())
    assert metrics["fixture"] and metrics["official_parity"] is None
    manifest = json.loads((evaluated / "provenance.json").read_text())
    assert (
        manifest["status"] == "complete"
        and len(manifest["dataset_manifest_sha256"]) == 64
    )
    with np.load(evaluated / "prediction.npz") as saved:
        assert saved["T_cw"].shape == (3, 4, 4)
        assert saved["direct_world_points"].shape == saved["depth_world_points"].shape
        assert saved["direct_world_points"].shape[-1] == 3
    with (evaluated / "per_sample.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 6


def test_e03_command_emits_pair_results_and_plot(tmp_path: Path):
    main(
        [
            "evaluate",
            "--config",
            "configs/experiment/e03_correspondence.yaml",
            "--set",
            f"runs={tmp_path}",
        ]
    )
    (run,) = (tmp_path / "e03_correspondence_fixture").iterdir()
    scores = json.loads((run / "metrics.json").read_text())
    assert scores["fixture"] and len(scores["pairs_by_site"]) == 6
    assert (run / "figures/recall_by_angle.png").stat().st_size > 0


def test_failed_run_records_provenance(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        main(
            [
                "evaluate",
                "--config",
                "configs/experiment/e00_synthetic.yaml",
                "--set",
                "mock=false",
                "--set",
                f"runs={tmp_path}",
                "--set",
                f"weights={tmp_path / 'missing'}",
            ]
        )
    (run,) = (tmp_path / "e00_synthetic").iterdir()
    provenance = json.loads((run / "provenance.json").read_text())
    assert provenance["status"] == "failed"
    assert provenance["checkpoints"]["vggt"]["repo_id"] == "facebook/VGGT-1B"
    assert "Missing checkpoint" in (run / "logs.txt").read_text()
    assert json.loads((run / "metrics.json").read_text())["status"] == "failed"
