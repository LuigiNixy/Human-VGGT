from dataclasses import dataclass

from numpy.typing import NDArray

from human_vggt.data.transforms import PixelTransform
from human_vggt.geometry.camera import Camera
from human_vggt.models.features import FeatureGrid


@dataclass(frozen=True)
class Prediction:
    transforms: tuple[PixelTransform, ...]
    pose_encoding: NDArray
    cameras: tuple[Camera, ...]
    depth: NDArray  # [S,H,W]
    depth_confidence: NDArray
    direct_points: NDArray  # [S,H,W,3]
    point_confidence: NDArray
    depth_points: NDArray
    features: tuple[FeatureGrid, ...]
    model_info: dict
