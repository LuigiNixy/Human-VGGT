import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from human_vggt.data.schema import Sample, View
from human_vggt.geometry.camera import Camera


def _rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode == "RGBA":
            image = Image.alpha_composite(Image.new("RGBA", image.size, "white"), image)
        return np.array(image.convert("RGB"))


def read_rgb_sample(directory: Path, num_views: int) -> Sample:
    files = tuple(
        sorted(
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpeg", ".jpg"}
        )
    )
    if len(files) != num_views:
        raise ValueError(
            f"Require exactly {num_views} RGBs in {directory}, found {len(files)}"
        )
    views = tuple(View(path.stem, _rgb(path), rgb_path=path) for path in files)
    return Sample(
        directory.name, None, directory.name, "unspecified", views, "rgb-directory"
    )


def read_calibrated_sample(directory: Path, num_views: int) -> Sample:
    path = directory / "sample.json"
    manifest = json.loads(path.read_text())
    if manifest["depth_unit"] != "m" or manifest["depth_type"] != "z":
        raise ValueError("Calibrated samples require optical-axis depth in meters")
    if len(manifest["views"]) != num_views:
        raise ValueError(f"Calibrated sample must specify exactly {num_views} views")
    ids = [record["view_id"] for record in manifest["views"]]
    if len(set(ids)) != len(ids):
        raise ValueError("Calibrated sample view IDs must be unique")
    views = []
    for record in manifest["views"]:
        rgb_path = directory / record["rgb_path"]
        mask_path = directory / record["mask_path"]
        depth_path = directory / record["depth_path"]
        rgb = _rgb(rgb_path)
        hw = rgb.shape[:2]
        with Image.open(mask_path) as image:
            mask = np.array(image.convert("L")) > 0
        depth = np.load(depth_path, allow_pickle=False)
        if depth.shape != hw or mask.shape != hw or not np.isfinite(depth[mask]).all():
            raise ValueError(f"Invalid mask or depth for {record['view_id']}")
        camera = Camera(np.array(record["K"]), np.array(record["T_cw"]), hw)
        views.append(
            View(
                record["view_id"],
                rgb,
                camera,
                mask,
                depth,
                rgb_path=rgb_path,
                mask_path=mask_path,
                depth_path=depth_path,
                valid_mask=mask & (depth > 0),
            )
        )
    return Sample(
        manifest["sample_id"],
        manifest.get("subject_id"),
        manifest["scene_id"],
        manifest["split"],
        tuple(views),
        manifest["provenance"],
        manifest.get("time_id"),
        directory / manifest["mesh_path"] if manifest.get("mesh_path") else None,
    )


def sample_manifest(sample: Sample, images: tuple[Path, ...]) -> dict[str, Any]:
    views = []
    for view, path in zip(sample.views, images, strict=True):
        record = {
            "view_id": view.view_id,
            "hw": list(view.rgb.shape[:2]),
            "capabilities": view.capabilities,
            "rgb_filename": path.name,
        }
        if view.camera is not None:
            record.update(
                K=view.camera.K.tolist(),
                T_cw=view.camera.T_cw.tolist(),
                depth_type="z",
                depth_unit="m",
            )
        for name in (
            "rgb",
            "mask",
            "depth",
            "surface_ids",
            "world_points",
            "valid_mask",
        ):
            array = getattr(view, "depth_z" if name == "depth" else name)
            if array is not None:
                digest = hashlib.sha256()
                digest.update(str((array.shape, array.dtype.str)).encode())
                digest.update(np.ascontiguousarray(array).view(np.uint8))
                record[f"{name}_array_sha256"] = digest.hexdigest()
        with path.open("rb") as stream:
            record["rgb_file_sha256"] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
        views.append(record)
    return {
        "sample_id": sample.sample_id,
        "subject_id": sample.subject_id,
        "scene_id": sample.scene_id,
        "time_id": sample.time_id,
        "split": sample.split,
        "provenance": sample.provenance,
        "views": views,
    }


def prepare_sample(sample: Sample, output: Path) -> tuple[Path, ...]:
    """Materialize generated RGB and numeric GT. External RGB stays at its source path."""
    images = []
    for index, view in enumerate(sample.views):
        path = view.rgb_path
        if path is None:
            output.mkdir(exist_ok=True)
            path = output / f"view_{index:02d}.png"
            Image.fromarray(view.rgb).save(path)
            np.savez_compressed(
                path.with_suffix(".npz"),
                depth_z=view.depth_z,
                mask=view.mask,
                surface_ids=view.surface_ids,
                world_points=view.world_points,
                K=view.camera.K,
                T_cw=view.camera.T_cw,
            )
        images.append(path)
    return tuple(images)
