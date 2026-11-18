import numpy as np
import pytest

from human_vggt.data.synthetic import make_sample
from human_vggt.evaluation.correspondence import bilinear, evaluate_pairs, retrieval
from human_vggt.evaluation.metrics import depth_metrics, surface_metrics
from human_vggt.evaluation.reconstruction import evaluate_reconstruction
from human_vggt.models.features import spatial_tokens
from human_vggt.models.fixture import fixture_features, fixture_prediction


def test_spatial_token_slicing_uses_reported_prefix():
    tokens = np.arange(2 * 3 * 9 * 5).reshape(2, 3, 9, 5)
    dense = spatial_tokens(tokens, 3, (2, 3))
    assert dense.shape == (2, 3, 2, 3, 5)
    np.testing.assert_array_equal(dense.reshape(2, 3, 6, 5), tokens[..., 3:, :])
    with pytest.raises(ValueError, match="Token count"):
        spatial_tokens(tokens, 2, (2, 3))


def test_bilinear_features_reject_outside_and_preserve_centers():
    grid = np.array([[[0.0], [1.0]], [[10.0], [11.0]]])
    values, keep = bilinear(
        grid, np.array([[0.5, 0.25], [1, 1], [-0.1, 0], [np.nan, 0]])
    )
    np.testing.assert_allclose(values[:2, 0], [3, 11])
    np.testing.assert_array_equal(keep, [True, True, False, False])


def test_masked_depth_and_noisy_surface_metrics():
    z = np.array([[2.0, 2.0, 2.0, 0.0]])
    pred = np.array([[3.0, np.nan, 9.0, 3.0]])
    score = depth_metrics(pred, z, np.array([[True, True, False, True]]))
    assert score == {"valid": 1, "eligible": 2, "abs_rel": 0.5, "rmse": 1}
    cloud = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    assert surface_metrics(cloud, cloud)["f1_1cm"] == 1
    noisy = surface_metrics(cloud + [0, 0.03, 0], cloud)
    assert noisy["chamfer_l1_m"] == pytest.approx(0.03)
    assert noisy["f1_1cm"] == 0 and noisy["f1_5cm"] == 1
    assert surface_metrics(cloud[:0], cloud)["chamfer_l1_m"] is None


def test_retrieval_perfect_and_shuffled():
    descriptors = np.eye(6)
    xy = np.column_stack((np.arange(6) * 8, np.zeros(6)))
    points = np.column_stack((xy, np.ones(6)))
    perfect = retrieval(descriptors, descriptors, xy, points, 40)
    shifted = retrieval(descriptors, np.roll(descriptors, 1, axis=0), xy, points, 40)
    assert perfect["recall_at_1"] == 1 and perfect["pck_2px"] == 1
    assert shifted["recall_at_1"] == 0 and shifted["normalized_surface_distance"] > 0


def test_fixture_paths_share_candidates_and_score_both_outputs():
    sample = make_sample()
    rows = evaluate_pairs(sample, fixture_features(sample), 64, 0.01, 42)
    assert len(rows) == 6
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert first["candidates"] == second["candidates"] == 64
        assert first["status"] == second["status"] == "ok"
    metrics, per_view = evaluate_reconstruction(
        sample, fixture_prediction(sample), known_metric=True
    )
    assert metrics["alignment"] == "synthetic_native_metric"
    assert len(per_view) == 6
    assert all(row["chamfer_l1_m"] == 0 for row in per_view)
