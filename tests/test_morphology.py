"""Tests for morphology feature extraction (Dimension 1)."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.features import FEATURE_NAMES, extract_morphology


def test_two_cells_area_and_centroid():
    mask = np.zeros((12, 12), dtype=int)
    mask[1:4, 1:3] = 1  # 3x2 block -> area 6, centroid (2, 1.5)
    mask[6:10, 6:10] = 2  # 4x4 block -> area 16, centroid (7.5, 7.5)

    morph = extract_morphology(mask)
    assert morph.num_cells == 2
    assert morph.feature_names == FEATURE_NAMES

    area = morph.features[:, 0]
    assert area.tolist() == [6.0, 16.0]
    np.testing.assert_allclose(morph.centroids[0], [2.0, 1.5])
    np.testing.assert_allclose(morph.centroids[1], [7.5, 7.5])


def test_elongation_increases_eccentricity():
    line = np.zeros((10, 10), dtype=int)
    line[5, 1:8] = 1
    block = np.zeros((10, 10), dtype=int)
    block[3:6, 3:6] = 1

    ecc_idx = FEATURE_NAMES.index("eccentricity")
    line_ecc = extract_morphology(line).features[0, ecc_idx]
    block_ecc = extract_morphology(block).features[0, ecc_idx]
    assert line_ecc > block_ecc
    assert line_ecc > 0.9
    assert block_ecc < 1e-6


def test_mean_intensity():
    mask = np.zeros((6, 6), dtype=int)
    mask[1:3, 1:3] = 1
    intensity = np.zeros((6, 6))
    intensity[1:3, 1:3] = 5.0
    morph = extract_morphology(mask, intensity=intensity)
    mean_idx = FEATURE_NAMES.index("mean_intensity")
    assert morph.features[0, mean_idx] == pytest.approx(5.0)


def test_empty_mask():
    morph = extract_morphology(np.zeros((5, 5), dtype=int))
    assert morph.num_cells == 0
    assert morph.centroids.shape == (0, 2)


def test_errors():
    with pytest.raises(ValueError):
        extract_morphology(np.zeros((2, 2, 2), dtype=int))
    with pytest.raises(ValueError):
        extract_morphology(np.zeros((4, 4), dtype=int), intensity=np.zeros((3, 3)))
