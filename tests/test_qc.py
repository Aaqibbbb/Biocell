"""Tests for image quality control."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from cellgraphfm.qc import QCReport, assess_quality


def _checkerboard(n=32, block=4, hi=255.0, lo=0.0):
    idx = np.arange(n) // block
    pattern = (idx[:, None] + idx[None, :]) % 2
    return np.where(pattern == 1, hi, lo).astype(np.float64)


def test_sharp_high_contrast_passes():
    report = assess_quality(_checkerboard())
    assert isinstance(report, QCReport)
    assert not report.blurry
    assert report.passed
    assert report.contrast == pytest.approx(127.5)
    assert report.brightness == pytest.approx(127.5)
    assert report.sharpness > 100.0


def test_uniform_image_is_blurry_and_fails():
    report = assess_quality(np.full((32, 32), 128.0))
    assert report.sharpness == pytest.approx(0.0)
    assert report.contrast == pytest.approx(0.0)
    assert report.blurry
    assert not report.passed


def test_blurring_reduces_sharpness():
    sharp = assess_quality(_checkerboard()).sharpness
    blurred = assess_quality(ndimage.uniform_filter(_checkerboard(), size=5)).sharpness
    assert sharp > blurred


def test_custom_thresholds():
    board = _checkerboard()
    # an absurdly high sharpness bar flags even a crisp checkerboard as blurry
    report = assess_quality(board, min_sharpness=1e12)
    assert report.blurry
    assert not report.passed


def test_float01_input_is_rescaled():
    board01 = _checkerboard(hi=1.0, lo=0.0)
    report = assess_quality(board01)
    assert report.contrast == pytest.approx(127.5)
    assert report.passed


def test_color_input():
    board = _checkerboard()
    rgb = np.stack([board, board, board], axis=-1)
    report = assess_quality(rgb)
    assert report.passed
    assert report.contrast > 10.0


def test_as_dict():
    report = assess_quality(_checkerboard())
    d = report.as_dict()
    assert set(d) == {"sharpness", "contrast", "brightness", "blurry", "passed"}
    assert d["passed"] is True


def test_bad_shape():
    with pytest.raises(ValueError):
        assess_quality(np.zeros((5,)))
    with pytest.raises(ValueError):
        assess_quality(np.zeros((5, 5, 5, 5)))
