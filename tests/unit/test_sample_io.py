import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from human_vggt.data.io import read_calibrated_sample
from human_vggt.data.synthetic import make_sample


def test_calibrated_sample_keeps_view_order_and_requires_optical_depth(tmp_path: Path):
    original = make_sample(2)
    records = []
    for view in reversed(original.views):
        stem = view.view_id
        Image.fromarray(view.rgb).save(tmp_path / f"{stem}.png")
        Image.fromarray(view.mask.astype(np.uint8) * 255).save(
            tmp_path / f"{stem}_mask.png"
        )
        np.save(tmp_path / f"{stem}.npy", view.depth_z)
        records.append(
            {
                "view_id": stem,
                "rgb_path": f"{stem}.png",
                "mask_path": f"{stem}_mask.png",
                "depth_path": f"{stem}.npy",
                "K": view.camera.K.tolist(),
                "T_cw": view.camera.T_cw.tolist(),
            }
        )
    manifest = {
        "sample_id": "sample",
        "subject_id": "subject",
        "scene_id": "scene",
        "split": "test",
        "provenance": "unit-test",
        "depth_unit": "m",
        "depth_type": "z",
        "views": records,
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(manifest))
    sample = read_calibrated_sample(tmp_path, 2)
    assert sample.subject_id == "subject"
    assert tuple(view.view_id for view in sample.views) == ("v01", "v00")
    np.testing.assert_array_equal(sample.views[0].mask, original.views[1].mask)
    np.testing.assert_array_equal(sample.views[0].depth_z, original.views[1].depth_z)
    manifest["depth_type"] = "ray"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="optical-axis"):
        read_calibrated_sample(tmp_path, 2)
