"""Tests for whole-slide tissue detection."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.wsi import detect_tissue, tissue_fraction, tissue_regions


def _white_rgb_with_square(size=20, box=(5, 11)):
    """White (unstained) RGB thumbnail with one saturated colour square."""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    a, b = box
    img[a:b, a:b] = (200, 50, 50)  # saturated red -> high saturation
    return img


def test_detect_tissue_rgb_otsu():
    img = _white_rgb_with_square()
    mask = detect_tissue(img)
    assert mask.dtype == bool
    assert mask.shape == (20, 20)
    # the saturated square is tissue, the white background is not
    assert mask[5:11, 5:11].all()
    assert not mask[:5, :5].any()
    assert tissue_fraction(mask) == pytest.approx(36 / 400)


def test_detect_tissue_explicit_threshold():
    img = _white_rgb_with_square()
    mask = detect_tissue(img, sat_threshold=0.5)
    assert mask[5:11, 5:11].all()
    assert not mask[:5, :5].any()


def test_detect_tissue_grayscale():
    # bright background (250) with a dark tissue square (20)
    img = np.full((20, 20), 250, dtype=np.uint8)
    img[6:12, 6:12] = 20
    mask = detect_tissue(img)
    assert mask[6:12, 6:12].all()
    assert tissue_fraction(mask) == pytest.approx(36 / 400)


def test_detect_tissue_rgba():
    img = np.full((8, 8, 4), 255, dtype=np.uint8)
    img[2:5, 2:5, :3] = (30, 180, 60)
    mask = detect_tissue(img)
    assert mask.shape == (8, 8)
    assert mask[2:5, 2:5].all()


def test_tissue_fraction_bounds_and_empty():
    assert tissue_fraction(np.ones((4, 4), dtype=bool)) == 1.0
    assert tissue_fraction(np.zeros((4, 4), dtype=bool)) == 0.0
    assert tissue_fraction(np.empty((0, 0), dtype=bool)) == 0.0


def test_tissue_regions_two_components_and_min_area():
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:5, 2:5] = True  # 9-pixel component
    mask[10:16, 10:16] = True  # 36-pixel component
    boxes = tissue_regions(mask)
    assert (2, 2, 5, 5) in boxes
    assert (10, 10, 16, 16) in boxes
    assert len(boxes) == 2

    # filtering by area drops the small component
    big_only = tissue_regions(mask, min_area=20)
    assert big_only == [(10, 10, 16, 16)]


def test_tissue_regions_empty():
    assert tissue_regions(np.zeros((5, 5), dtype=bool)) == []


def test_detect_tissue_bad_shape():
    with pytest.raises(ValueError):
        detect_tissue(np.zeros((5,)))
    with pytest.raises(ValueError):
        detect_tissue(np.zeros((5, 5, 5)))


def test_tissue_regions_bad_shape():
    with pytest.raises(ValueError):
        tissue_regions(np.zeros((2, 2, 2), dtype=bool))
