"""OpenCV cameras, integer pixel centers, and optical-axis depth in meters."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Camera:
    K: NDArray
    T_cw: NDArray
    hw: tuple[int, int]

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3) or self.T_cw.shape != (4, 4):
            raise ValueError("Camera requires K [3,3] and T_cw [4,4]")
        if len(self.hw) != 2 or min(self.hw) <= 0:
            raise ValueError("Camera requires positive image dimensions")

    @property
    def center(self) -> NDArray:
        rotation = self.T_cw[:3, :3]
        return -rotation.T @ self.T_cw[:3, 3]

    def project(self, world: NDArray) -> tuple[NDArray, NDArray]:
        xyz = world @ self.T_cw[:3, :3].T + self.T_cw[:3, 3]
        homogeneous = xyz @ self.K.T
        with np.errstate(divide="ignore", invalid="ignore"):
            return homogeneous[..., :2] / homogeneous[..., 2:], xyz[..., 2]

    def unproject(self, pixels: NDArray, depth_z: NDArray) -> NDArray:
        ones = np.ones((*pixels.shape[:-1], 1))
        rays = np.concatenate((pixels, ones), axis=-1) @ np.linalg.inv(self.K).T
        xyz = rays * depth_z[..., None]
        return (xyz - self.T_cw[:3, 3]) @ self.T_cw[:3, :3]

    def unproject_image(self, depth_z: NDArray) -> NDArray:
        if depth_z.shape != self.hw:
            raise ValueError("Depth image dimensions must match the camera")
        return self.unproject(pixel_grid(self.hw), depth_z)


def pixel_grid(hw: tuple[int, int]) -> NDArray:
    y, x = np.indices(hw)
    return np.stack((x, y), axis=-1)


def look_at(
    center: NDArray, target: NDArray, K: NDArray, hw: tuple[int, int]
) -> Camera:
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    transform = np.eye(4)
    transform[:3, :3] = [right, down, forward]
    transform[:3, 3] = -transform[:3, :3] @ center
    return Camera(K, transform, hw)
