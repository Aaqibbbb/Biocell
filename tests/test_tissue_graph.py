"""Tests for the TissueGraph data structure."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.data import TissueGraph


def test_basic_properties(tiny_graph):
    assert tiny_graph.num_nodes == 4
    assert tiny_graph.num_edges == 6
    assert tiny_graph.num_node_features == 1
    assert not tiny_graph.is_empty
    assert "nodes=4" in repr(tiny_graph)


def test_degree_and_neighbors(tiny_graph):
    deg = tiny_graph.degree()
    assert deg.tolist() == [1, 2, 2, 1]
    assert set(tiny_graph.neighbors(1).tolist()) == {0, 2}


def test_empty_edges():
    g = TissueGraph(
        coords=np.zeros((3, 2)),
        node_features=np.zeros((3, 2)),
        edge_index=np.empty((2, 0), dtype=np.int64),
    )
    assert g.num_edges == 0
    assert g.degree().tolist() == [0, 0, 0]


def test_subgraph(tiny_graph):
    sub = tiny_graph.subgraph([1, 2, 3])
    assert sub.num_nodes == 3
    # original edges (1,2),(2,1),(2,3),(3,2) survive and are relabeled to 0,1,2
    assert sub.num_edges == 4
    assert sub.cell_types.tolist() == [0, 1, 1]


def test_validation_coords():
    with pytest.raises(ValueError):
        TissueGraph(np.zeros((3, 5)), np.zeros((3, 2)), np.empty((2, 0)))


def test_validation_feature_rows():
    with pytest.raises(ValueError):
        TissueGraph(np.zeros((3, 2)), np.zeros((2, 2)), np.empty((2, 0)))


def test_validation_edge_range():
    with pytest.raises(ValueError):
        TissueGraph(np.zeros((3, 2)), np.zeros((3, 2)), np.array([[0, 9], [1, 2]]))


def test_validation_cell_types():
    with pytest.raises(ValueError):
        TissueGraph(
            np.zeros((3, 2)),
            np.zeros((3, 2)),
            np.empty((2, 0)),
            cell_types=np.array([0, 1]),
        )


def test_to_pyg(tiny_graph):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    data = tiny_graph.to_pyg()
    assert data.x.shape == (4, 1)
    assert data.edge_index.shape == (2, 6)
    assert data.y.shape == (4,)


def test_to_networkx(tiny_graph):
    pytest.importorskip("networkx")
    nx_graph = tiny_graph.to_networkx()
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3  # undirected collapses the 6 directed
