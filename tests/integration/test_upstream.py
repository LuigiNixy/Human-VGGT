import os
from pathlib import Path

import pytest

from human_vggt.data.io import prepare_sample
from human_vggt.data.synthetic import make_sample

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        os.getenv("HUMAN_VGGT_TEST_UPSTREAM") != "1",
        reason="Set HUMAN_VGGT_TEST_UPSTREAM=1 with licensed local weights",
    ),
]


def test_vggt_quickstart_parity(tmp_path: Path):
    from human_vggt.models.vggt_adapter import VGGTAdapter
    from human_vggt.provenance import resolve_device

    device = resolve_device(
        os.environ.get("HUMAN_VGGT_TEST_DEVICE", "cuda"), "fp32", True
    )
    images = prepare_sample(make_sample(2), tmp_path / "fixture")
    model = VGGTAdapter(Path(os.environ["HUMAN_VGGT_WEIGHTS"]), device, "fp32")
    prediction = model.predict(images, ("patch_embedding", "aggregation_last"))
    assert prediction.features[0].values.shape == (1, 2, 32, 37, 1024)
    assert prediction.features[1].values.shape == (1, 2, 32, 37, 2048)
    parity = model.check_parity(images, prediction, 1e-5, 1e-5)
    assert parity["passed"], parity


def test_sapiens_native_teacher(tmp_path: Path):
    if os.getenv("HUMAN_VGGT_TEST_SAPIENS") != "1":
        pytest.skip("Set HUMAN_VGGT_TEST_SAPIENS=1 to include standalone teacher")
    from human_vggt.models.sapiens_adapter import SapiensAdapter
    from human_vggt.provenance import resolve_device

    device = resolve_device(
        os.environ.get("HUMAN_VGGT_TEST_DEVICE", "cuda"), "fp32", True
    )
    images = prepare_sample(make_sample(2), tmp_path / "fixture")
    model = SapiensAdapter(Path(os.environ["HUMAN_VGGT_WEIGHTS"]), device, "fp32")
    features = model.extract(images, (2, 11))
    assert [grid.layer for grid in features] == [2, 11]
    assert all(grid.values.shape == (1, 2, 64, 48, 768) for grid in features)
    assert not model.model.training
    assert all(not p.requires_grad and p.grad is None for p in model.model.parameters())
