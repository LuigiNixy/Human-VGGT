import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def model_manifest() -> dict:
    return yaml.safe_load((ROOT / "manifests/upstream/revisions.yaml").read_text())


def source_root(family: str) -> Path:
    path = ROOT / "third_party" / family
    if not (path / ".git").exists():
        raise FileNotFoundError(
            f"Initialize {path.name} via git submodule update --init"
        )
    revision = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    )
    expected = model_manifest()["repositories"][family]["commit"]
    if revision.strip() != expected:
        raise ValueError(f"{family} checkout differs from pinned revision {expected}")
    return path


def checkpoint_path(family: str, weights: Path) -> Path:
    record = model_manifest()["checkpoints"][family]
    path = weights / family / record["filename"]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint {path}. See README.md for the pinned checkpoint command."
        )
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != record["sha256"]:
        raise ValueError(f"SHA256 mismatch for checkpoint {path}")
    return path
