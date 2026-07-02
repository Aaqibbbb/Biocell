"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.data import TissueGraph
from cellgraphfm.pipeline import build_tissue_graph, synthetic_tissue


@pytest.fixture
def synthetic():
    """A clustered synthetic tissue: (coords, features, cell_types)."""
    return synthetic_tissue(n_cells=120, n_types=4, seed=1)


@pytest.fixture
def graph(synthetic):
    """A knn TissueGraph built from the synthetic tissue."""
    coords, features, cell_types = synthetic
    return build_tissue_graph(coords, features, cell_types=cell_types, method="knn", k=6)


@pytest.fixture
def tiny_graph():
    """A 4-node chain with types [0, 0, 1, 1] and known undirected edges."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    features = np.arange(4, dtype=float).reshape(4, 1)
    edge_index = np.array([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    cell_types = np.array([0, 0, 1, 1])
    return TissueGraph(coords, features, edge_index, cell_types=cell_types)
