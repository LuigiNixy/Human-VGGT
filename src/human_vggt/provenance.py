import csv
import json
import platform
import subprocess
import traceback
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import yaml

from human_vggt.config import Config
from human_vggt.models.upstream import ROOT


def git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True
    )
    return result.stdout.rstrip() if result.returncode == 0 else None


def environment() -> dict:
    versions = {}
    for package in (
        "human-vggt",
        "numpy",
        "scipy",
        "Pillow",
        "torch",
        "torchvision",
        "safetensors",
    ):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    hardware = {}
    if versions["torch"] is not None:
        import torch

        hardware = {
            "cuda": torch.cuda.is_available(),
            "torch_cuda": torch.version.cuda,
            "mps": torch.backends.mps.is_available(),
        }
        if hardware["cuda"]:
            free, total = torch.cuda.mem_get_info()
            hardware |= {
                "gpu": torch.cuda.get_device_name(),
                "gpu_free_bytes": free,
                "gpu_total_bytes": total,
                "gpu_capability": torch.cuda.get_device_capability(),
            }
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    text=True,
                )
                hardware["driver"] = (
                    result.stdout.splitlines()[0] if result.returncode == 0 else None
                )
            except FileNotFoundError:
                hardware["driver"] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        **hardware,
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


class Run:
    def __init__(self, config: Config, command: list[str]) -> None:
        run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
        self.path = Path(config.runs) / config.experiment / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.command = command
        self.details = {"seed": config.seed}
        self.start = perf_counter()

    def __enter__(self) -> "Run":
        (self.path / "config.yaml").write_text(
            yaml.safe_dump(self.config.dictionary(), sort_keys=True)
        )
        self.details.update(
            git_sha=git("rev-parse", "HEAD"),
            git_status=git("status", "--porcelain"),
            upstream=git("submodule", "status", "--recursive"),
            command=self.command,
            environment=environment(),
        )
        print(self.path, flush=True)
        return self

    def __exit__(self, exc_type, error, tb) -> None:
        self.details["elapsed_seconds"] = perf_counter() - self.start
        self.details["status"] = "complete" if error is None else "failed"
        if self.details.get("resolved_device") == "cuda":
            import torch

            self.details["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(self.path / "provenance.json", self.details)
        log = (
            "Run complete.\n"
            if error is None
            else "".join(traceback.format_exception(error))
        )
        (self.path / "logs.txt").write_text(log)
        if error is not None and not (self.path / "metrics.json").exists():
            write_json(
                self.path / "metrics.json", {"status": "failed", "reason": str(error)}
            )
        if not (self.path / "per_sample.csv").exists():
            self.write_rows([])

    def write_rows(self, rows: list[dict]) -> None:
        columns = list(dict.fromkeys(key for row in rows for key in row)) or [
            "sample_id"
        ]
        with (self.path / "per_sample.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def resolve_device(requested: str, precision: str, use_models: bool) -> str:
    if requested == "cpu" and precision == "fp32" and not use_models:
        return "cpu"
    import torch

    if requested == "auto":
        requested = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested, but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested, but is unavailable")
    if precision == "bf16" and (
        requested != "cuda" or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("bf16 requires a CUDA device with bf16 support")
    if requested == "cuda":
        torch.cuda.reset_peak_memory_stats()
    return requested


def seed_models(seed: int) -> None:
    import torch

    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
