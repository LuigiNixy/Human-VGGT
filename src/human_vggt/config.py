import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from string import Template

import yaml
from yaml.constructor import ConstructorError


class UniqueKeys(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = super().construct_mapping(node, deep)
        if len(mapping) != len(node.value):
            raise ConstructorError(None, None, "Duplicate config key", node.start_mark)
        return mapping


@dataclass(frozen=True)
class Config:
    experiment: str
    task: str
    dataset: str = "synthetic"
    seed: int = 42
    num_views: int = 3
    device: str = "cpu"
    precision: str = "fp32"
    mock: bool = False
    rgb_dir: str = ""
    runs: str = "outputs"
    weights: str = "checkpoints"
    samples_per_pair: int = 256
    occlusion_tolerance_m: float = 0.01
    parity_atol: float = 1e-5
    parity_rtol: float = 1e-5
    sapiens: bool = False
    feature_sites: tuple[str, ...] = ("patch_embedding", "aggregation_last")
    sapiens_layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.experiment or not self.experiment.replace("_", "").isalnum():
            raise ValueError("experiment must be an alphanumeric name with underscores")
        if self.task not in {"baseline", "correspondence"}:
            raise ValueError("task must be baseline or correspondence")
        if self.dataset not in {"synthetic", "rgb", "calibrated"}:
            raise ValueError("dataset must be synthetic, rgb or calibrated")
        if self.dataset != "synthetic" and not self.rgb_dir:
            raise ValueError("rgb_dir is required for external inputs")
        if self.mock and (
            self.dataset != "synthetic"
            or self.device != "cpu"
            or self.precision != "fp32"
            or self.sapiens
        ):
            raise ValueError("Fixture mode requires synthetic data, CPU and no teacher")
        if self.device not in {"cpu", "mps", "cuda", "auto"}:
            raise ValueError("Unsupported device")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or bf16")
        if self.seed < 0 or not 2 <= self.num_views <= 4 or self.samples_per_pair < 6:
            raise ValueError(
                "Require nonnegative seed, 2–4 views, and at least 6 candidates"
            )
        if not (0 < self.occlusion_tolerance_m < float("inf")):
            raise ValueError("occlusion_tolerance_m must be positive and finite")
        if any(not 0 <= v < float("inf") for v in (self.parity_atol, self.parity_rtol)):
            raise ValueError("Parity tolerances must be nonnegative and finite")
        if self.task == "correspondence" and not (
            self.mock or self.feature_sites or self.sapiens
        ):
            raise ValueError(
                "Correspondence requires at least one encoder feature site"
            )
        if self.dataset == "rgb" and self.task == "correspondence":
            raise ValueError("Correspondence requires calibrated cameras and GT depth")
        valid_sites = {
            "patch_embedding",
            "aggregation_last",
            "aggregation_4",
            "aggregation_11",
            "aggregation_17",
            "aggregation_23",
        }
        if set(self.feature_sites) - valid_sites or len(set(self.feature_sites)) != len(
            self.feature_sites
        ):
            raise ValueError("feature_sites must contain unique supported VGGT sites")
        if any(i < 0 for i in self.sapiens_layers) or len(
            set(self.sapiens_layers)
        ) != len(self.sapiens_layers):
            raise ValueError("sapiens_layers must contain unique zero-based indices")

    def resolved(self, device: str) -> "Config":
        return replace(self, device=device)

    def dictionary(self) -> dict:
        return asdict(self)


def load_config(path: Path, overrides: list[str]) -> Config:
    values = yaml.load(path.read_text(), Loader=UniqueKeys)
    if not isinstance(values, dict) or any(not isinstance(key, str) for key in values):
        raise ValueError("Config must be a YAML mapping")
    for override in overrides:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError("Use --set KEY=VALUE")
        values[key] = yaml.safe_load(value)
    known = {field.name: field.type for field in fields(Config)}
    if values.keys() - known.keys():
        raise ValueError(
            f"Unknown config fields: {sorted(values.keys() - known.keys())}"
        )
    for key, value in values.items():
        kind = known[key]
        if kind is float and type(value) is int:
            value = float(value)
        if key in {"feature_sites", "sapiens_layers"}:
            item_type = str if key == "feature_sites" else int
            if not isinstance(value, (list, tuple)) or any(
                type(v) is not item_type for v in value
            ):
                raise ValueError(f"Invalid list elements for {key}")
            values[key] = tuple(value)
        elif type(value) is not kind:
            raise ValueError(f"Invalid type for {key}: expected {kind.__name__}")
        else:
            values[key] = (
                Template(value).substitute(os.environ) if kind is str else value
            )
    if not {"experiment", "task"} <= values.keys():
        raise ValueError("Config requires experiment and task")
    return Config(**values)
