from dataclasses import dataclass

from numpy.typing import NDArray

from human_vggt.data.transforms import PixelTransform


@dataclass(frozen=True)
class FeatureGrid:
    values: NDArray  # [B,S,h,w,C], in float32
    transforms: tuple[PixelTransform, ...]
    patch_size: int
    site: str
    layer: int | None = None


def spatial_tokens(tokens, patch_start: int, grid_hw: tuple[int, int]):
    """Keep the upstream-delimited spatial block for [..., P, C]."""
    height, width = grid_hw
    if patch_start < 0 or tokens.shape[-2] - patch_start != height * width:
        raise ValueError(
            f"Token count disagrees with patch_start={patch_start}, grid={grid_hw}"
        )
    return tokens[..., patch_start:, :].reshape(
        *tokens.shape[:-2], height, width, tokens.shape[-1]
    )
