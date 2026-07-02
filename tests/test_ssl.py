"""Tests for self-supervised learning utilities (Dimension 5)."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.ssl import mask_features, random_node_mask


def test_random_node_mask_count():
    mask = random_node_mask(100, 0.3, seed=0)
    assert mask.dtype == bool
    assert mask.sum() == 30


def test_random_node_mask_edges():
    assert random_node_mask(50, 0.0).sum() == 0
    assert random_node_mask(50, 1.0).sum() == 50
    assert random_node_mask(0, 0.5).sum() == 0
    # non-zero rate always masks at least one node
    assert random_node_mask(3, 0.01).sum() == 1


def test_random_node_mask_invalid():
    with pytest.raises(ValueError):
        random_node_mask(10, 1.5)


def test_mask_features_zero():
    feats = np.ones((4, 3))
    mask = np.array([True, False, True, False])
    out = mask_features(feats, mask, strategy="zero")
    assert np.all(out[mask] == 0.0)
    assert np.all(out[~mask] == 1.0)


def test_mask_features_mean():
    feats = np.array([[0.0], [2.0], [4.0], [6.0]])
    mask = np.array([True, False, False, False])
    out = mask_features(feats, mask, strategy="mean")
    # mean of the unmasked rows (2, 4, 6) == 4
    assert out[0, 0] == pytest.approx(4.0)


def test_mask_features_errors():
    with pytest.raises(ValueError):
        mask_features(np.zeros((3, 2)), np.array([True, False, True]), strategy="bad")
    with pytest.raises(ValueError):
        mask_features(np.zeros((3, 2)), np.array([True, False]))


def test_sce_loss_and_reconstruction():
    torch = pytest.importorskip("torch")
    from cellgraphfm.ssl import masked_reconstruction_loss, sce_loss

    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert sce_loss(x, x.clone()).item() == pytest.approx(0.0, abs=1e-6)

    orthogonal = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    assert sce_loss(x, orthogonal).item() == pytest.approx(1.0, abs=1e-6)

    mask = torch.tensor([False, False])
    assert masked_reconstruction_loss(x, orthogonal, mask).item() == pytest.approx(0.0)

    mask = torch.tensor([True, True])
    assert masked_reconstruction_loss(x, orthogonal, mask).item() == pytest.approx(1.0, abs=1e-6)
