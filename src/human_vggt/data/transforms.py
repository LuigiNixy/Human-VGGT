"""Mappings between original RGB, network pixels and patch centers."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from human_vggt.geometry.camera import Camera


@dataclass(frozen=True)
class PixelTransform:
    original_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    output_hw: tuple[int, int]
    offset_xy: tuple[int, int] = (0, 0)  # positive padding, negative crop

    @property
    def matrix(self) -> NDArray:
        sy, sx = np.divide(self.resized_hw, self.original_hw)
        dx, dy = self.offset_xy
        return np.array(
            [[sx, 0, (sx - 1) / 2 + dx], [0, sy, (sy - 1) / 2 + dy], [0, 0, 1]]
        )

    def apply(self, pixels: NDArray) -> NDArray:
        matrix = self.matrix
        return pixels @ matrix[:2, :2].T + matrix[:2, 2]

    def camera(self, camera: Camera) -> Camera:
        if camera.hw != self.original_hw:
            raise ValueError("Camera dimensions must match the original image")
        return Camera(self.matrix @ camera.K, camera.T_cw, self.output_hw)

    def patch_coordinates(self, pixels: NDArray, patch_size: int) -> NDArray:
        return (self.apply(pixels) + 0.5) / patch_size - 0.5
