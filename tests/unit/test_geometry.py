import numpy as np
import pytest

from human_vggt.data.synthetic import make_sample
from human_vggt.data.transforms import PixelTransform
from human_vggt.geometry.alignment import fit_similarity
from human_vggt.geometry.camera import pixel_grid
from human_vggt.geometry.visibility import visible


def test_fixture_axes_reprojection_and_repeatability():
    sample = make_sample()
    repeated = make_sample()
    for a, b in zip(sample.views, repeated.views, strict=True):
        np.testing.assert_array_equal(a.rgb, b.rgb)
        np.testing.assert_array_equal(a.surface_ids, b.surface_ids)
        np.testing.assert_allclose(a.world_points, b.world_points)
        xyz = a.world_points[a.mask]
        grid, z = a.camera.project(xyz)
        np.testing.assert_allclose(grid, pixel_grid(a.camera.hw)[a.mask], atol=1e-12)
        np.testing.assert_allclose(z, a.depth_z[a.mask], atol=1e-12)
        np.testing.assert_allclose(a.camera.unproject(grid, z), xyz, atol=1e-12)
        np.testing.assert_allclose(np.linalg.det(a.camera.T_cw[:3, :3]), 1)
        center, forward_z = a.camera.project(np.zeros((1, 3)))
        np.testing.assert_allclose(center[0], a.camera.K[:2, 2])
        assert forward_z[0] > 0
        assert a.mask.any() and not a.mask.all()


def test_transform_intrinsics_resize_crop_and_patch_centers():
    view = make_sample().views[0]
    transform = PixelTransform(view.camera.hw, (192, 224), (168, 252), (14, -12))
    xy = np.array([[0.0, 0.0], [30.0, 35.0]])
    np.testing.assert_allclose(transform.apply(xy), [[14.5, -11.5], [74.5, 58.5]])
    world = view.camera.unproject(xy, np.array([2.0, 3.0]))
    mapped, _ = transform.camera(view.camera).project(world)
    np.testing.assert_allclose(mapped, transform.apply(xy))
    identity = PixelTransform((16, 16), (16, 16), (16, 16))
    np.testing.assert_allclose(
        identity.patch_coordinates(np.array([[3.5, 11.5]]), 8), [[0, 1]]
    )


def test_visibility_rejects_occluded_outside_and_behind():
    view = make_sample().views[0]
    y, x = np.argwhere(view.mask)[len(np.argwhere(view.mask)) // 2]
    z = view.depth_z[y, x]
    points = view.camera.unproject(
        np.array([[x, y], [x, y], [-4, y], [x, y]]), np.array([z, z + 0.1, z, -z])
    )
    _, keep = visible(view.camera, points, view.depth_z, view.mask, 0.01)
    np.testing.assert_array_equal(keep, [True, False, False, False])


def test_sequence_similarity_and_degenerate_anchors():
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [1, 1, 1]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    target = 2 * source @ rotation.T + [10, -2, 3]
    transform = fit_similarity(source, target)
    np.testing.assert_allclose(transform.apply(source), target, atol=1e-12)
    assert transform.scale == pytest.approx(2)
    with pytest.raises(ValueError, match="degenerate"):
        fit_similarity(np.tile([1, 2, 3], (3, 1)), target[:3])
