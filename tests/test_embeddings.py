"""Tests for cell embedders and the foundation-model registry (Dimensions 1 & 8)."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.features import (
    IdentityEmbedder,
    RandomProjectionEmbedder,
    describe_foundation_model,
    get_embedder,
    list_foundation_models,
)


def test_registry():
    models = list_foundation_models()
    assert "uni" in models and "virchow" in models
    assert describe_foundation_model("UNI").embed_dim == 1024
    with pytest.raises(KeyError):
        describe_foundation_model("does-not-exist")


def test_identity_embedder():
    emb = IdentityEmbedder(embed_dim=12)
    out = emb.embed(np.zeros((5, 3, 4)))
    assert out.shape == (5, 12)
    with pytest.raises(ValueError):
        emb.embed(np.zeros((5, 10)))


def test_random_projection_deterministic_and_normalized():
    patches = np.random.default_rng(0).normal(size=(7, 48))
    a = RandomProjectionEmbedder(input_dim=48, embed_dim=32, seed=3)
    b = RandomProjectionEmbedder(input_dim=48, embed_dim=32, seed=3)
    out_a = a.embed(patches)
    out_b = b.embed(patches)
    assert out_a.shape == (7, 32)
    np.testing.assert_allclose(out_a, out_b)
    np.testing.assert_allclose(np.linalg.norm(out_a, axis=1), 1.0, atol=1e-5)


def test_random_projection_wrong_input_dim():
    emb = RandomProjectionEmbedder(input_dim=16, embed_dim=8)
    with pytest.raises(ValueError):
        emb.embed(np.zeros((3, 20)))


def test_get_embedder():
    assert isinstance(get_embedder("identity", embed_dim=4), IdentityEmbedder)
    assert isinstance(get_embedder("random", input_dim=4, embed_dim=8), RandomProjectionEmbedder)
    with pytest.raises(NotImplementedError):
        get_embedder("uni")
    with pytest.raises(ValueError):
        get_embedder("banana")


def test_embedder_is_callable():
    emb = get_embedder("random", input_dim=4, embed_dim=8)
    np.testing.assert_allclose(emb(np.ones((2, 4))), emb.embed(np.ones((2, 4))))
