"""Tests for image tiling."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.wsi import Tile, tile_image


def test_tile_grid_no_overlap():
    image = np.arange(100 * 100).reshape(100, 100)
    tiles = tile_image(image, tile_size=50)
    assert len(tiles) == 4
    assert all(isinstance(t, Tile) for t in tiles)
    coords = {(t.y, t.x) for t in tiles}
    assert coords == {(0, 0), (0, 50), (50, 0), (50, 50)}
    grid = {(t.row, t.col) for t in tiles}
    assert grid == {(0, 0), (0, 1), (1, 0), (1, 1)}
    for t in tiles:
        assert t.size == 50
        assert t.image.shape == (50, 50)
    # tile content matches the source slice
    tl = next(t for t in tiles if (t.row, t.col) == (0, 0))
    np.testing.assert_array_equal(tl.image, image[0:50, 0:50])


def test_tile_multichannel():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    tiles = tile_image(image, tile_size=50)
    assert len(tiles) == 4
    assert all(t.image.shape == (50, 50, 3) for t in tiles)


def test_tile_overlap():
    image = np.zeros((100, 100))
    tiles = tile_image(image, tile_size=50, overlap=25)  # step 25 -> starts 0,25,50
    assert len(tiles) == 9
    xs = sorted({t.x for t in tiles})
    assert xs == [0, 25, 50]


def test_tile_tissue_filter():
    image = np.zeros((100, 100))
    mask = np.zeros((100, 100), dtype=bool)
    mask[0:50, 0:50] = True  # only the top-left tile is tissue
    tiles = tile_image(image, tile_size=50, tissue_mask=mask, min_tissue_fraction=0.5)
    assert len(tiles) == 1
    assert (tiles[0].row, tiles[0].col) == (0, 0)


def test_tile_image_smaller_than_tile():
    assert tile_image(np.zeros((30, 30)), tile_size=50) == []


def test_tile_errors():
    image = np.zeros((100, 100))
    with pytest.raises(ValueError):
        tile_image(image, tile_size=0)
    with pytest.raises(ValueError):
        tile_image(image, tile_size=50, overlap=50)
    with pytest.raises(ValueError):
        tile_image(image, tile_size=50, overlap=-1)
    with pytest.raises(ValueError):
        tile_image(np.zeros((10,)), tile_size=5)
    with pytest.raises(ValueError):
        tile_image(image, tile_size=50, tissue_mask=np.zeros((10, 10), dtype=bool))
