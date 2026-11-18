"""Paired GT-visible point retrieval with common sampling across feature sites."""

from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from human_vggt.data.schema import Sample
from human_vggt.geometry.visibility import visible
from human_vggt.models.features import FeatureGrid


def within_grid(hw: tuple[int, int], xy: NDArray) -> NDArray:
    h, w = hw
    x, y = xy.T
    return (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)


def bilinear(grid: NDArray, xy: NDArray) -> tuple[NDArray, NDArray]:
    h, w = grid.shape[:2]
    x, y = xy.T
    valid = within_grid((h, w), xy)
    x, y = np.where(valid, x, 0), np.where(valid, y, 0)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    dx, dy = (x - x0)[:, None], (y - y0)[:, None]
    values = ((1 - dx) * grid[y0, x0] + dx * grid[y0, x1]) * (1 - dy)
    values += ((1 - dx) * grid[y1, x0] + dx * grid[y1, x1]) * dy
    return values.astype(grid.dtype, copy=False), valid


def retrieval(
    query: NDArray, target: NDArray, xy: NDArray, world: NDArray, scene_scale: float
) -> dict:
    if len(query) < 2 or query.shape != target.shape or scene_scale <= 0:
        raise ValueError(
            "Retrieval requires paired descriptors and a positive scene scale"
        )
    qa = query / np.maximum(np.linalg.norm(query, axis=-1, keepdims=True), 1e-12)
    ta = target / np.maximum(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12)
    similarities = qa @ ta.T
    order = np.argsort(-similarities, axis=1, kind="stable")
    correct = np.arange(len(query))
    best = order[:, 0]
    matched = similarities.diagonal().copy()
    negative_mean = (
        similarities.sum(dtype=np.float64) - matched.sum(dtype=np.float64)
    ) / (similarities.size - len(matched))
    np.fill_diagonal(similarities, -np.inf)
    hardest = similarities.max(axis=1)
    pixel_error = np.linalg.norm(xy[best] - xy, axis=1)
    distance = np.linalg.norm(world[best] - world, axis=1)
    singular = np.linalg.svd(ta - ta.mean(axis=0), compute_uv=False)
    total = singular.sum()
    mass = singular[singular > 0] / total if total else singular[:0]
    rank = float(np.exp(-np.sum(mass * np.log(mass)))) if total else 0.0
    metrics = {
        "matched_cosine": float(matched.mean()),
        "mean_negative_cosine": float(negative_mean),
        "hardest_negative_cosine": float(hardest.mean()),
        "cosine_gap": float((matched - hardest).mean()),
        "recall_at_1": float(np.mean(best == correct)),
        "recall_at_5": float(np.mean(np.any(order[:, :5] == correct[:, None], axis=1))),
        "normalized_surface_distance": float(distance.mean() / scene_scale),
        "target_effective_rank": rank,
    }
    return metrics | {
        f"pck_{px}px": float(np.mean(pixel_error <= px)) for px in (2, 5, 10)
    }


def evaluate_pairs(
    sample: Sample,
    grids: tuple[FeatureGrid, ...],
    limit: int,
    tolerance_m: float,
    seed: int,
) -> list[dict]:
    if not grids:
        raise ValueError("Correspondence requires at least one feature site")
    for view in sample.views:
        if any(value is None for value in (view.camera, view.depth_z, view.mask)):
            raise ValueError(
                "Correspondence requires GT cameras, masks and optical-axis depth"
            )
    clouds = [
        view.camera.unproject_image(view.depth_z)[
            view.mask & np.isfinite(view.depth_z) & (view.depth_z > 0)
        ]
        for view in sample.views
    ]
    cloud = np.concatenate(clouds)
    scene_scale = np.linalg.norm(np.ptp(cloud, axis=0)) if len(cloud) else 0
    if not scene_scale:
        raise ValueError(
            "Correspondence requires a non-degenerate visible reference surface"
        )
    center = cloud.mean(axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for first, second in combinations(range(len(sample.views)), 2):
        a, b = sample.views[first], sample.views[second]
        points = clouds[first]
        xy_a, valid_a = visible(a.camera, points, a.depth_z, a.mask, tolerance_m)
        xy_b, valid_b = visible(b.camera, points, b.depth_z, b.mask, tolerance_m)
        visible_ids = np.flatnonzero(valid_a & valid_b)
        points, xy_a, xy_b = points[visible_ids], xy_a[visible_ids], xy_b[visible_ids]
        keep = np.ones(len(points), dtype=bool)
        sampled = []
        for grid in grids:
            xy0 = grid.transforms[first].patch_coordinates(xy_a, grid.patch_size)
            xy1 = grid.transforms[second].patch_coordinates(xy_b, grid.patch_size)
            keep &= within_grid(grid.values.shape[2:4], xy0)
            keep &= within_grid(grid.values.shape[2:4], xy1)
            sampled.append((xy0, xy1))
        eligible = np.flatnonzero(keep)
        # A target pixel must appear once to avoid competing identical pixel matches.
        _, unique = np.unique(
            np.rint(xy_b[eligible]).astype(int), axis=0, return_index=True
        )
        eligible = eligible[unique]
        chosen = rng.choice(eligible, min(limit, len(eligible)), replace=False)
        features = []
        finite = np.ones(len(chosen), dtype=bool)
        for grid, (xy0, xy1) in zip(grids, sampled, strict=True):
            fv0, _ = bilinear(grid.values[0, first], xy0[chosen])
            fv1, _ = bilinear(grid.values[0, second], xy1[chosen])
            finite &= np.isfinite(fv0).all(axis=1) & np.isfinite(fv1).all(axis=1)
            features.append((fv0, fv1))
        selected = chosen[finite]
        ray_a = center - a.camera.center
        ray_b = center - b.camera.center
        cos_angle = (ray_a @ ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b))
        for grid, (fv0, fv1) in zip(grids, features, strict=True):
            row = {
                "sample_id": sample.sample_id,
                "subject_id": sample.subject_id,
                "scene_id": sample.scene_id,
                "source": a.view_id,
                "target": b.view_id,
                "site": grid.site,
                "angle_deg": float(np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))),
                "gt_visible": len(visible_ids),
                "eligible_unique": len(eligible),
                "candidates": len(selected),
                "nonfinite_candidates": int((~finite).sum()),
                "status": "ok" if len(selected) >= 6 else "insufficient_candidates",
            }
            if len(selected) >= 6:
                row.update(
                    retrieval(
                        fv0[finite],
                        fv1[finite],
                        xy_b[selected],
                        points[selected],
                        float(scene_scale),
                    )
                )
            rows.append(row)
    return rows
