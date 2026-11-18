import numpy as np

from human_vggt.models.vggt_adapter import crop_transforms


def test_vggt_crop_and_mixed_aspect_padding():
    wide, portrait = crop_transforms(((96, 112), (144, 96)))
    assert wide.resized_hw == (448, 518) and wide.offset_xy == (0, 35)
    assert portrait.resized_hw == (784, 518) and portrait.offset_xy == (0, -133)
    assert wide.output_hw == portrait.output_hw == (518, 518)
    for transform in (wide, portrait):
        h, w = transform.original_hw
        np.testing.assert_allclose(
            transform.apply(np.array([[(w - 1) / 2, (h - 1) / 2]])), [[258.5, 258.5]]
        )
