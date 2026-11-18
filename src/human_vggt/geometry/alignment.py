from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: NDArray
    translation: NDArray

    def apply(self, points: NDArray) -> NDArray:
        return self.scale * (points @ self.rotation.T) + self.translation


def fit_similarity(source: NDArray, target: NDArray) -> Similarity:
    """Fit one proper Sim(3) from paired, non-collinear sequence anchors."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Similarity fitting requires matched [N,3] anchors")
    if (
        len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise ValueError("Similarity fitting requires at least three finite anchors")
    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_mean, target - target_mean
    covariance = y.T @ x
    if np.linalg.matrix_rank(covariance) < 2:
        raise ValueError("Similarity anchors are degenerate")
    u, singular_values, vh = np.linalg.svd(covariance)
    signs = np.array([1.0, 1.0, np.linalg.det(u @ vh)])
    rotation = (u * signs) @ vh
    scale = float(singular_values @ signs / np.sum(x**2))
    return Similarity(scale, rotation, target_mean - scale * (rotation @ source_mean))
