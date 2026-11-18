import numpy as np
from numpy.typing import NDArray

from human_vggt.geometry.camera import Camera


def visible(
    camera: Camera, points: NDArray, depth_z: NDArray, mask: NDArray, tolerance_m: float
) -> tuple[NDArray, NDArray]:
    """Compare projected Z against the nearest GT z-buffer pixel."""
    if depth_z.shape != camera.hw or mask.shape != camera.hw or tolerance_m <= 0:
        raise ValueError(
            "Visibility requires matching image shapes and a positive tolerance"
        )
    xy, z = camera.project(points)
    h, w = camera.hw
    inside = np.isfinite(xy).all(axis=-1) & (z > 0)
    inside &= (xy[..., 0] >= 0) & (xy[..., 0] <= w - 1)
    inside &= (xy[..., 1] >= 0) & (xy[..., 1] <= h - 1)
    x, y = np.rint(np.where(inside[..., None], xy, 0)).astype(int).T
    reference = depth_z[y, x]
    keep = (
        inside & mask[y, x] & (reference > 0) & (np.abs(z - reference) <= tolerance_m)
    )
    return xy, keep
