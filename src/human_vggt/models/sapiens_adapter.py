"""Frozen 0.1B teacher via the pinned standalone Sapiens 2 implementation."""

from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model

from human_vggt.data.transforms import PixelTransform
from human_vggt.models.features import FeatureGrid, spatial_tokens
from human_vggt.models.upstream import checkpoint_path, model_manifest, source_root

INPUT_HW = (1024, 768)
PATCH_SIZE = 16


class SapiensAdapter:
    def __init__(self, weights: Path, device: str, precision: str) -> None:
        root = source_root("sapiens2")
        path = checkpoint_path("sapiens2", weights)
        # Import only the self-contained backbone. sapiens/__init__.py imports task stacks.
        file = root / "sapiens/backbones/standalone/sapiens2.py"
        spec = spec_from_file_location("_human_vggt_sapiens2_standalone", file)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        args = model_manifest()["checkpoints"]["sapiens2"]["constructor"]
        self.model = module.Sapiens2(**args)
        load_model(self.model, path, strict=True)
        self.model.requires_grad_(False).eval().to(device)
        self.device = device
        self.precision = precision

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.model.parameters())

    def extract(
        self, images: tuple[Path, ...], layers: tuple[int, ...]
    ) -> tuple[FeatureGrid, ...]:
        selected = sorted(layers or (self.model.num_layers - 1,))
        if selected[-1] >= self.model.num_layers:
            raise ValueError(f"Sapiens 2 has {self.model.num_layers} layers")
        self.model.out_indices = selected
        height, width = INPUT_HW
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        transforms, views = [], []
        for path in images:
            with Image.open(path) as image:
                rgb = image.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
                transforms.append(
                    PixelTransform((image.height, image.width), INPUT_HW, INPUT_HW)
                )
                tensor = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).float()
                views.append(((tensor / 255 - mean) / std).to(self.device))
        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.precision == "bf16"
            else nullcontext()
        )
        by_view = []
        with torch.inference_mode(), autocast:
            for rgb in views:
                tokens = self.model(rgb[None])
                by_view.append(
                    [
                        spatial_tokens(
                            t, self.model.num_extra_tokens, (height // 16, width // 16)
                        )
                        .float()
                        .cpu()
                        .numpy()
                        for t in tokens
                    ]
                )
        return tuple(
            FeatureGrid(
                np.stack(values, axis=1),
                tuple(transforms),
                PATCH_SIZE,
                f"sapiens2_layer_{index}",
                index,
            )
            for index, values in zip(selected, zip(*by_view, strict=True), strict=True)
        )
