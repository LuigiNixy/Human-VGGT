"""A ray-traced asymmetric assembly of ellipsoids with world-fixed texture."""

from dataclasses import replace

import numpy as np

from human_vggt.data.schema import Sample, View
from human_vggt.geometry.camera import Camera, look_at, pixel_grid

RENDERER_REVISION = "ellipsoids-v1"


def render(camera: Camera, texture_phase: float) -> View:
    grid = pixel_grid(camera.hw)
    rays = camera.unproject(grid, np.ones(camera.hw)) - camera.center
    shapes = (
        ([0.0, 0.02, 0.0], [0.44, 0.68, 0.29]),
        ([0.24, 0.41, 0.14], [0.30, 0.19, 0.22]),
        ([-0.35, -0.20, -0.11], [0.25, 0.28, 0.19]),
    )
    depth = np.full(camera.hw, np.inf)
    surface_ids = np.zeros(camera.hw, dtype=np.int32)
    for index, (center, radii) in enumerate(shapes, start=1):
        origin = (camera.center - center) / radii
        direction = rays / radii
        a = np.sum(direction**2, axis=-1)
        b = direction @ origin
        c = origin @ origin - 1
        discriminant = b**2 - a * c
        hit_z = (-b - np.sqrt(np.maximum(discriminant, 0))) / a
        hit = (discriminant >= 0) & (hit_z > 0) & (hit_z < depth)
        depth[hit], surface_ids[hit] = hit_z[hit], index

    mask = surface_ids > 0
    depth[~mask] = 0
    points = camera.unproject(grid, depth)
    phase = points @ np.array([[12, 3, 10], [4, 16, 7], [9, 5, 14]])
    rgb = np.rint(125 + 95 * np.sin(phase + texture_phase)).astype(np.uint8)
    rgb[~mask] = [22, 28, 36]
    points[~mask] = np.nan
    return View("", rgb, camera, mask, depth, surface_ids, points, valid_mask=mask)


def make_sample(num_views: int = 3, seed: int = 42) -> Sample:
    if not 2 <= num_views <= 4:
        raise ValueError("Fixture supports two to four views")
    hw = (96, 112)
    K = np.array([[108, 0, 55.5], [0, 108, 47.5], [0, 0, 1]])
    texture_phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
    views = []
    for index, angle in enumerate(np.linspace(-0.35, 0.35, num_views)):
        center = np.array(
            [3 * np.sin(angle), 0.12 + 0.12 * np.cos(3 * angle), -3 * np.cos(angle)]
        )
        camera = look_at(center, np.zeros(3), K, hw)
        views.append(replace(render(camera, texture_phase), view_id=f"v{index:02d}"))
    return Sample(
        "synthetic-000", None, "ellipsoids", "test", tuple(views), RENDERER_REVISION
    )
