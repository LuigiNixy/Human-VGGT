from dataclasses import dataclass
from pathlib import Path

from numpy.typing import NDArray

from human_vggt.geometry.camera import Camera


@dataclass(frozen=True)
class View:
    view_id: str
    rgb: NDArray  # uint8 [H,W,3]
    camera: Camera | None = None
    mask: NDArray | None = None  # bool [H,W]
    depth_z: NDArray | None = None  # float [H,W], optical-axis meters
    surface_ids: NDArray | None = None  # int [H,W], zero is background
    world_points: NDArray | None = None  # float [H,W,3]
    rgb_path: Path | None = None
    mask_path: Path | None = None
    depth_path: Path | None = None
    valid_mask: NDArray | None = None

    @property
    def capabilities(self) -> dict[str, bool]:
        return {
            name: getattr(self, name) is not None
            for name in (
                "camera",
                "mask",
                "depth_z",
                "surface_ids",
                "world_points",
                "valid_mask",
            )
        }


@dataclass(frozen=True)
class Sample:
    sample_id: str
    subject_id: str | None
    scene_id: str
    split: str
    views: tuple[View, ...]
    provenance: str
    time_id: str | None = None
    mesh_path: Path | None = None
