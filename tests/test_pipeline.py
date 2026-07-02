"""Tests for the end-to-end pipeline helpers."""

from __future__ import annotations

import numpy as np

from cellgraphfm.data import TissueGraph
from cellgraphfm.pipeline import (
    build_tissue_graph,
    demo_graph,
    graph_from_mask,
    synthetic_tissue,
)


def test_synthetic_tissue_shapes():
    coords, features, cell_types = synthetic_tissue(n_cells=80, n_types=3, n_features=10, seed=2)
    assert coords.shape == (80, 2)
    assert features.shape == (80, 10)
    assert cell_types.shape == (80,)
    assert set(np.unique(cell_types).tolist()).issubset({0, 1, 2})


def test_build_tissue_graph_edge_attr():
    coords, features, cell_types = synthetic_tissue(n_cells=60, seed=0)
    g = build_tissue_graph(coords, features, cell_types=cell_types, method="knn", k=5)
    assert isinstance(g, TissueGraph)
    assert g.edge_attr is not None
    assert g.edge_attr.shape == (g.num_edges, 1)
    assert g.metadata["graph_method"] == "knn"


def test_graph_from_mask():
    mask = np.zeros((20, 20), dtype=int)
    mask[2:4, 2:4] = 1
    mask[2:4, 15:17] = 2
    mask[15:17, 2:4] = 3
    mask[15:17, 15:17] = 4
    g = graph_from_mask(mask, method="knn", k=2)
    assert g.num_nodes == 4
    assert "feature_names" in g.metadata
    assert g.metadata["source"] == "mask"


def test_demo_graph():
    g = demo_graph(n_cells=50, method="knn", k=4, seed=0)
    assert g.num_nodes == 50
    assert g.cell_types is not None
    assert g.num_edges > 0
