from pathlib import Path

import numpy as np

from human_vggt.data.schema import Sample
from human_vggt.evaluation.metrics import depth_metrics, surface_metrics
from human_vggt.geometry.alignment import Similarity, fit_similarity
from human_vggt.geometry.camera import pixel_grid
from human_vggt.models.prediction import Prediction


def export_prediction(path: Path, prediction: Prediction) -> None:
    np.savez_compressed(
        path,
        pose_encoding=prediction.pose_encoding,
        K=np.stack([cam.K for cam in prediction.cameras]),
        T_cw=np.stack([cam.T_cw for cam in prediction.cameras]),
        pixel_transform=np.stack([t.matrix for t in prediction.transforms]),
        depth_z=prediction.depth,
        direct_world_points=prediction.direct_points,
        depth_world_points=prediction.depth_points,
        depth_confidence=prediction.depth_confidence,
        point_confidence=prediction.point_confidence,
    )


def evaluate_reconstruction(
    sample: Sample, prediction: Prediction, *, known_metric: bool
) -> tuple[dict, list[dict]]:
    if any(
        v.camera is None or v.depth_z is None or v.mask is None for v in sample.views
    ):
        return {"status": "no_metric_ground_truth"}, []
    if known_metric:
        alignment = Similarity(1.0, np.eye(3), np.zeros(3))
        protocol = "synthetic_native_metric"
    else:
        source = np.stack([camera.center for camera in prediction.cameras])
        target = np.stack([view.camera.center for view in sample.views])
        try:
            alignment = fit_similarity(source, target)
        except ValueError as error:
            return {"status": "alignment_unavailable", "reason": str(error)}, []
        protocol = "sequence_camera_centers_sim3"
    rows = []
    predicted_clouds: dict[str, list] = {"direct": [], "depth_unprojected": []}
    gt_clouds = []
    for i, (view, transform) in enumerate(
        zip(sample.views, prediction.transforms, strict=True)
    ):
        points = view.camera.unproject_image(view.depth_z)
        # Evaluate at network pixel centers mapped to the closest original GT pixels.
        inverse = np.linalg.inv(transform.matrix)
        grid = pixel_grid(transform.output_hw)
        xy = grid @ inverse[:2, :2].T + inverse[:2, 2]
        x, y = np.rint(xy).astype(int).transpose(2, 0, 1)
        h, w = view.camera.hw
        content = (x >= 0) & (x < w) & (y >= 0) & (y < h)
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        mask = content & view.mask[y, x] & (view.depth_z[y, x] > 0)
        z_gt = view.depth_z[y, x]
        gt = points[y, x][mask]
        gt_clouds.append(gt)
        stats = depth_metrics(prediction.depth[i] * alignment.scale, z_gt, mask)
        for name, maps in (
            ("direct", prediction.direct_points),
            ("depth_unprojected", prediction.depth_points),
        ):
            pred = maps[i]
            valid = mask & np.isfinite(pred).all(axis=-1)
            if name == "depth_unprojected":
                valid &= np.isfinite(prediction.depth[i]) & (prediction.depth[i] > 0)
            pred = alignment.apply(pred[valid])
            predicted_clouds[name].append(pred)
            row = {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "scene_id": sample.scene_id,
                "view_id": view.view_id,
                "output": name,
            }
            row.update(surface_metrics(pred, gt))
            if name == "depth_unprojected":
                row.update({f"depth_{key}": value for key, value in stats.items()})
            rows.append(row)
    return {
        "status": "ok",
        "ground_truth": "visible_surface_point_samples",
        "alignment": protocol,
        "scale": alignment.scale,
        "rotation": alignment.rotation.tolist(),
        "translation": alignment.translation.tolist(),
        "sequence_surface": {
            name: surface_metrics(np.concatenate(clouds), np.concatenate(gt_clouds))
            for name, clouds in predicted_clouds.items()
        },
    }, rows
