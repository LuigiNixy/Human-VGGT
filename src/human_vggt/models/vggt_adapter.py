"""Pinned VGGT quick-start inference with RGB-only inputs."""

import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from safetensors.torch import load_file

from human_vggt.data.transforms import PixelTransform
from human_vggt.geometry.camera import Camera
from human_vggt.models.features import FeatureGrid, spatial_tokens
from human_vggt.models.prediction import Prediction
from human_vggt.models.upstream import checkpoint_path, model_manifest, source_root


def crop_transforms(sizes: tuple[tuple[int, int], ...]) -> tuple[PixelTransform, ...]:
    resized_heights = [round(h * (518 / w) / 14) * 14 for h, w in sizes]
    cropped_heights = [min(518, h) for h in resized_heights]
    output_hw = (max(cropped_heights), 518)
    if min(cropped_heights) <= 0:
        raise ValueError(
            "VGGT official crop preprocessing rounds a resized height to zero"
        )
    return tuple(
        PixelTransform(
            hw,
            (resized, 518),
            output_hw,
            (0, (output_hw[0] - cropped) // 2 - (resized - cropped) // 2),
        )
        for hw, resized, cropped in zip(
            sizes, resized_heights, cropped_heights, strict=True
        )
    )


class VGGTAdapter:
    def __init__(self, weights: Path, device: str, precision: str) -> None:
        source = source_root("vggt")
        sys.path.insert(0, str(source))
        from vggt.models.vggt import VGGT

        if not Path(sys.modules[VGGT.__module__].__file__).is_relative_to(source):
            raise ImportError("VGGT must be imported from the pinned submodule")
        path = checkpoint_path("vggt", weights)
        args = model_manifest()["checkpoints"]["vggt"]["constructor"]
        self.model = VGGT(**args)
        self.model.load_state_dict(load_file(path), strict=True, assign=True)
        self.model.requires_grad_(False).eval().to(device)
        self.device = device
        self.precision = precision

    def _forward(self, images: torch.Tensor) -> dict:
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            return self.model(images)

    def predict(self, images: tuple[Path, ...], sites: tuple[str, ...]) -> Prediction:
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        sizes = []
        for path in images:
            with Image.open(path) as image:
                sizes.append((image.height, image.width))
        transforms = crop_transforms(tuple(sizes))
        inputs = load_and_preprocess_images(images).to(self.device)
        height, width = inputs.shape[-2:]
        if (height, width) != transforms[0].output_hw:
            raise ValueError(
                "VGGT preprocessing disagrees with recorded pixel transforms"
            )
        patch = self.model.aggregator.patch_size
        grid_hw = height // patch, width // patch
        grids = []

        def grid(tokens, start, site, layer):
            values = spatial_tokens(tokens, start, grid_hw)
            grids.append(
                FeatureGrid(
                    values.float().cpu().numpy(), transforms, patch, site, layer
                )
            )

        def embedding_hook(_module, _args, output):
            dense = output["x_norm_patchtokens"] if isinstance(output, dict) else output
            grid(dense.unsqueeze(0), 0, "patch_embedding", None)

        def aggregator_hook(_module, _args, output):
            layers, patch_start = output
            for site in sites:
                if site.startswith("aggregation_"):
                    index = (
                        len(layers) - 1
                        if site.endswith("_last")
                        else int(site.split("_")[1])
                    )
                    if index >= len(layers) or layers[index] is None:
                        raise ValueError(
                            f"VGGT does not expose aggregation slot {index}"
                        )
                    grid(layers[index], patch_start, site, index)

        aggregator = self.model.aggregator
        hooks = []
        if "patch_embedding" in sites:
            hooks.append(aggregator.patch_embed.register_forward_hook(embedding_hook))
        if any(site.startswith("aggregation_") for site in sites):
            hooks.append(aggregator.register_forward_hook(aggregator_hook))
        try:
            output = self._forward(inputs)
        finally:
            for hook in hooks:
                hook.remove()

        with torch.inference_mode():
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                output["pose_enc"], (height, width)
            )
        Ks, T_cw = array(intrinsic), array(extrinsic)
        cameras = []
        for k, t in zip(Ks, T_cw, strict=True):
            transform = np.eye(4)
            transform[:3] = t
            cameras.append(Camera(k, transform, (height, width)))
        depth = array(output["depth"])[..., 0]
        depth_points = np.stack(
            [c.unproject_image(z) for c, z in zip(cameras, depth, strict=True)]
        )
        return Prediction(
            transforms,
            array(output["pose_enc"]),
            tuple(cameras),
            depth,
            array(output["depth_conf"]),
            array(output["world_points"]),
            array(output["world_points_conf"]),
            depth_points,
            tuple(grids),
            {
                "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "input_shape": list(inputs.shape),
            },
        )

    def check_parity(
        self, images: tuple[Path, ...], prediction: Prediction, atol: float, rtol: float
    ) -> dict:
        from vggt.utils.geometry import unproject_depth_map_to_point_map
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        inputs = load_and_preprocess_images(images).to(self.device)
        output = self._forward(inputs)
        with torch.inference_mode():
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                output["pose_enc"], inputs.shape[-2:]
            )
        extrinsic, intrinsic = array(extrinsic), array(intrinsic)
        pairs = {
            "pose_encoding": (prediction.pose_encoding, array(output["pose_enc"])),
            "extrinsic": (
                np.stack([c.T_cw[:3] for c in prediction.cameras]),
                extrinsic,
            ),
            "intrinsic": (np.stack([c.K for c in prediction.cameras]), intrinsic),
            "depth": (prediction.depth, array(output["depth"])[..., 0]),
            "depth_confidence": (
                prediction.depth_confidence,
                array(output["depth_conf"]),
            ),
            "direct_points": (prediction.direct_points, array(output["world_points"])),
            "point_confidence": (
                prediction.point_confidence,
                array(output["world_points_conf"]),
            ),
            "depth_points": (
                prediction.depth_points,
                unproject_depth_map_to_point_map(
                    array(output["depth"]), extrinsic, intrinsic
                ),
            ),
        }
        checks = {}
        for name, (actual, expected) in pairs.items():
            difference = np.abs(actual - expected)
            checks[name] = {
                "passed": bool(np.allclose(actual, expected, atol=atol, rtol=rtol)),
                "max_absolute_error": float(difference.max())
                if np.isfinite(difference).all()
                else None,
                "shape": list(actual.shape),
            }
        return {
            "passed": all(item["passed"] for item in checks.values()),
            "checks": checks,
            "atol": atol,
            "rtol": rtol,
        }


def array(value: torch.Tensor) -> NDArray:
    return value[0].float().cpu().numpy()
