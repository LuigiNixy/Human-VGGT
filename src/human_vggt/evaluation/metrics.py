import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree


def depth_metrics(
    pred: NDArray, gt: NDArray, mask: NDArray
) -> dict[str, float | int | None]:
    if pred.shape != gt.shape or gt.shape != mask.shape:
        raise ValueError("Depth and mask shapes must match")
    eligible = mask & np.isfinite(gt) & (gt > 0)
    valid = eligible & np.isfinite(pred) & (pred > 0)
    count = int(valid.sum())
    error = pred[valid] - gt[valid]
    return {
        "valid": count,
        "eligible": int(eligible.sum()),
        "abs_rel": float(np.mean(np.abs(error) / gt[valid])) if count else None,
        "rmse": float(np.sqrt(np.mean(error**2))) if count else None,
    }


def surface_metrics(pred: NDArray, gt: NDArray) -> dict[str, float | int | None]:
    """Point-sampled distances in meters. Caller selects one sequence-wide gauge."""
    if any(points.ndim != 2 or points.shape[1] != 3 for points in (pred, gt)):
        raise ValueError("Surface samples must be [N,3]")
    if not np.isfinite(pred).all() or not np.isfinite(gt).all():
        raise ValueError("Surface samples must be finite")
    scores = {"pred_count": len(pred), "gt_count": len(gt)}
    if not len(pred) or not len(gt):
        return scores | dict.fromkeys(
            (
                "accuracy_m",
                "completeness_m",
                "chamfer_l1_m",
                "f1_1cm",
                "f1_2cm",
                "f1_5cm",
            )
        )
    forward = cKDTree(gt).query(pred)[0]
    backward = cKDTree(pred).query(gt)[0]
    scores |= {
        "accuracy_m": float(forward.mean()),
        "completeness_m": float(backward.mean()),
        "chamfer_l1_m": float((forward.mean() + backward.mean()) / 2),
    }
    for cm in (1, 2, 5):
        precision, recall = np.mean(forward <= cm / 100), np.mean(backward <= cm / 100)
        scores[f"f1_{cm}cm"] = float(
            2 * precision * recall / (precision + recall) if precision + recall else 0
        )
    return scores
