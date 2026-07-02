"""Tests for IHC colour deconvolution and biomarker scoring."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.biomarkers import (
    HE_MATRIX,
    HED_MATRIX,
    h_score,
    positivity_index,
    separate_stains,
)


def test_stain_matrices_are_row_normalised():
    np.testing.assert_allclose(np.linalg.norm(HED_MATRIX, axis=1), 1.0)
    np.testing.assert_allclose(np.linalg.norm(HE_MATRIX, axis=1), 1.0)
    assert HED_MATRIX.shape == (3, 3)
    assert HE_MATRIX.shape == (2, 3)


def test_separate_stains_shapes():
    rgb = np.full((6, 5, 3), 200, dtype=np.uint8)
    assert separate_stains(rgb).shape == (6, 5, 3)
    assert separate_stains(rgb, stain_matrix=HE_MATRIX).shape == (6, 5, 2)


def test_separate_stains_white_is_zero():
    white = np.full((4, 4, 3), 255, dtype=np.uint8)
    conc = separate_stains(white)
    np.testing.assert_allclose(conc, 0.0, atol=1e-8)


def test_separate_stains_recovers_pure_dab():
    # build an image whose optical density is exactly one unit of the DAB vector
    dab_od = HED_MATRIX[2]
    intensity = 256.0 * 10.0 ** (-dab_od) - 1.0
    rgb = np.tile(intensity, (3, 3, 1))
    conc = separate_stains(rgb)
    np.testing.assert_allclose(conc[..., 2], 1.0, atol=1e-6)  # DAB channel ~ 1
    np.testing.assert_allclose(conc[..., :2], 0.0, atol=1e-6)  # H and E ~ 0


def test_separate_stains_errors():
    with pytest.raises(ValueError):
        separate_stains(np.zeros((4, 4)))  # missing channel axis
    with pytest.raises(ValueError):
        separate_stains(np.zeros((4, 4, 3)), stain_matrix=np.zeros((3, 2)))


def _four_nuclei():
    """Return (labels, dab) with 4 nuclei of mean DAB 0.1, 0.3, 0.6, 0.9."""
    labels = np.zeros((10, 10), dtype=int)
    dab = np.zeros((10, 10), dtype=float)
    spots = {
        1: (slice(0, 2), slice(0, 2), 0.1),
        2: (slice(0, 2), slice(4, 6), 0.3),
        3: (slice(4, 6), slice(0, 2), 0.6),
        4: (slice(4, 6), slice(4, 6), 0.9),
    }
    for label, (rs, cs, value) in spots.items():
        labels[rs, cs] = label
        dab[rs, cs] = value
    return labels, dab


def test_positivity_index():
    labels, dab = _four_nuclei()
    assert positivity_index(labels, dab, threshold=0.5) == pytest.approx(0.5)  # 0.6, 0.9
    assert positivity_index(labels, dab, threshold=0.05) == pytest.approx(1.0)
    assert positivity_index(labels, dab, threshold=1.0) == pytest.approx(0.0)


def test_positivity_index_all_positive():
    labels = np.zeros((6, 6), dtype=int)
    labels[1:3, 1:3] = 1
    labels[3:5, 3:5] = 2
    dab = np.full((6, 6), 2.0)
    assert positivity_index(labels, dab, threshold=1.0) == pytest.approx(1.0)


def test_positivity_index_empty():
    labels = np.zeros((5, 5), dtype=int)
    dab = np.zeros((5, 5))
    assert positivity_index(labels, dab, threshold=0.5) == 0.0


def test_h_score():
    labels, dab = _four_nuclei()
    # bins: 0.1->neg, 0.3->1+, 0.6->2+, 0.9->3+  => 1*25 + 2*25 + 3*25 = 150
    assert h_score(labels, dab, thresholds=(0.2, 0.5, 0.75)) == pytest.approx(150.0)


def test_h_score_extremes():
    labels = np.zeros((6, 6), dtype=int)
    labels[0:2, 0:2] = 1
    labels[3:5, 3:5] = 2
    strong = np.full((6, 6), 5.0)
    assert h_score(labels, strong, thresholds=(0.2, 0.5, 0.75)) == pytest.approx(300.0)
    weak = np.zeros((6, 6))
    assert h_score(labels, weak, thresholds=(0.2, 0.5, 0.75)) == pytest.approx(0.0)


def test_h_score_empty_and_bad_thresholds():
    empty = np.zeros((5, 5), dtype=int)
    assert h_score(empty, np.zeros((5, 5))) == 0.0
    with pytest.raises(ValueError):
        h_score(empty, np.zeros((5, 5)), thresholds=(0.1, 0.2))


def test_scoring_shape_mismatch():
    with pytest.raises(ValueError):
        positivity_index(np.zeros((5, 5), dtype=int), np.zeros((4, 4)), threshold=0.5)
