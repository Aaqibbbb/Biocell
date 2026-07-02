"""Grid tiling of images into (optionally tissue-filtered) patches.

Whole-slide images are far too large to process at once, so they are cut into
fixed-size tiles. :func:`tile_image` performs a simple regular grid tiling with
optional overlap and an optional tissue-mask filter so that empty background
tiles are dropped. Only complete tiles (fully inside the image) are emitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Tile", "tile_image"]


@dataclass
class Tile:
    """A single image tile and its position within the source image.

    Attributes
    ----------
    row, col:
        Grid indices of the tile (``0``-based).
    y, x:
        Pixel coordinates of the tile's top-left corner in the source image.
    size:
        Tile side length in pixels.
    image:
        ``(size, size)`` or ``(size, size, C)`` array with the tile pixels.
    """

    row: int
    col: int
    y: int
    x: int
    size: int
    image: np.ndarray


def tile_image(
    image: np.ndarray,
    tile_size: int,
    *,
    overlap: int = 0,
    tissue_mask: np.ndarray | None = None,
    min_tissue_fraction: float = 0.0,
) -> list[Tile]:
    """Cut ``image`` into a regular grid of square tiles.

    Parameters
    ----------
    image:
        ``(H, W)`` or ``(H, W, C)`` array.
    tile_size:
        Side length of each (square) tile in pixels.
    overlap:
        Number of pixels neighbouring tiles overlap. Must satisfy
        ``0 <= overlap < tile_size``; the grid step is ``tile_size - overlap``.
    tissue_mask:
        Optional ``(H, W)`` boolean array. When given, a tile is only kept when
        its tissue fraction is ``>= min_tissue_fraction``.
    min_tissue_fraction:
        Minimum fraction of tissue pixels (in ``[0, 1]``) required to keep a tile.

    Returns
    -------
    tiles:
        List of :class:`Tile`, row-major over the grid.
    """
    image = np.asarray(image)
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D or 3-D, got shape {image.shape}")
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")

    height, width = image.shape[:2]
    if tissue_mask is not None:
        tissue_mask = np.asarray(tissue_mask, dtype=bool)
        if tissue_mask.shape != (height, width):
            raise ValueError(
                f"tissue_mask must have shape {(height, width)}, got {tissue_mask.shape}"
            )

    step = tile_size - overlap
    tiles: list[Tile] = []
    for row, y in enumerate(range(0, height - tile_size + 1, step)):
        for col, x in enumerate(range(0, width - tile_size + 1, step)):
            if tissue_mask is not None:
                sub = tissue_mask[y : y + tile_size, x : x + tile_size]
                if float(np.count_nonzero(sub) / sub.size) < min_tissue_fraction:
                    continue
            patch = image[y : y + tile_size, x : x + tile_size]
            tiles.append(Tile(row=row, col=col, y=y, x=x, size=tile_size, image=patch))
    return tiles
