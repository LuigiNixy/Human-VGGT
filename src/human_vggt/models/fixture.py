"""Known fixture signals for testing the harness end to end."""

import numpy as np

from human_vggt.data.schema import Sample
from human_vggt.data.transforms import PixelTransform
from human_vggt.models.features import FeatureGrid
from human_vggt.models.prediction import Prediction


def fixture_prediction(sample: Sample) -> Prediction:
    transforms = tuple(
        PixelTransform(v.camera.hw, v.camera.hw, v.camera.hw) for v in sample.views
    )
    cameras = tuple(v.camera for v in sample.views)
    depth = np.stack([v.depth_z for v in sample.views])
    masks = np.stack([v.mask for v in sample.views])
    direct = np.stack([v.world_points for v in sample.views])
    unprojected = np.stack([v.camera.unproject_image(v.depth_z) for v in sample.views])
    return Prediction(
        transforms,
        np.empty((len(cameras), 0)),
        cameras,
        depth,
        masks,
        direct,
        masks,
        unprojected,
        (),
        {"fixture": True},
    )


def fixture_features(sample: Sample) -> tuple[FeatureGrid, ...]:
    grids = []
    for patch in (4, 8):
        values = []
        transforms = []
        for view in sample.views:
            h, w = view.rgb.shape[:2]
            pooled = view.rgb.reshape(h // patch, patch, w // patch, patch, 3).mean(
                axis=(1, 3)
            )
            values.append(pooled.astype(np.float32) / 255)
            transforms.append(PixelTransform((h, w), (h, w), (h, w)))
        grids.append(
            FeatureGrid(
                np.stack(values)[None],
                tuple(transforms),
                patch,
                f"fixture_rgb_patch_{patch}",
            )
        )
    return tuple(grids)
