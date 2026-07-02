"""Tests for focus measurement and focus stacking."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from cellgraphfm.wsi import focus_measure, focus_stack


def _stripes(height, width):
    """Vertical 0/255 stripes (high-frequency, in focus)."""
    return np.tile(np.array([0, 255], dtype=np.uint8), (height, width // 2))


def test_focus_measure_sharp_beats_blurred():
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 256, size=(40, 40)).astype(np.float64)
    blurred = ndimage.uniform_filter(sharp, size=5)
    assert focus_measure(sharp) > focus_measure(blurred)


def test_focus_measure_uniform_is_zero():
    assert focus_measure(np.full((10, 10), 128.0)) == pytest.approx(0.0)


def test_focus_measure_requires_2d():
    with pytest.raises(ValueError):
        focus_measure(np.zeros((3, 4, 5)))


def test_focus_stack_selects_sharp_region():
    height = width = 20
    stripes = _stripes(height, width)
    # slice 0 is sharp on the left, flat on the right; slice 1 the opposite
    s0 = np.full((height, width), 128, dtype=np.uint8)
    s0[:, :10] = stripes[:, :10]
    s1 = np.full((height, width), 128, dtype=np.uint8)
    s1[:, 10:] = stripes[:, 10:]
    zstack = np.stack([s0, s1])

    composite = focus_stack(zstack)
    assert composite.shape == (height, width)
    assert composite.dtype == np.uint8
    # left interior should come from slice 0, right interior from slice 1
    np.testing.assert_array_equal(composite[:, :6], s0[:, :6])
    np.testing.assert_array_equal(composite[:, 14:], s1[:, 14:])
    # and the fused result is sharper than either flawed input slice
    assert focus_measure(composite.astype(float)) > focus_measure(s0.astype(float))
    assert focus_measure(composite.astype(float)) > focus_measure(s1.astype(float))


def test_focus_stack_color_shape():
    zstack = np.zeros((3, 12, 12, 3), dtype=np.uint8)
    zstack[1, 4:8, 4:8] = 255  # a sharp block in slice 1
    composite = focus_stack(zstack)
    assert composite.shape == (12, 12, 3)
    assert composite.dtype == np.uint8


def test_focus_stack_errors():
    with pytest.raises(ValueError):
        focus_stack(np.zeros((10, 10)))  # not a stack
    with pytest.raises(ValueError):
        focus_stack(np.zeros((0, 10, 10)))  # empty stack
    with pytest.raises(ValueError):
        focus_stack(np.zeros((2, 10, 10)), window=0)
