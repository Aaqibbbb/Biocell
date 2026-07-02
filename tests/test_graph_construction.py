"""Tests for graph construction (Dimension 2)."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.graph import (
    add_self_loops,
    build_graph,
    contact_graph,
    delaunay_graph,
    edge_lengths,
    knn_graph,
    radius_graph,
    to_undirected,
)


def _edge_set(edge_index):
    return {tuple(edge_index[:, e].tolist()) for e in range(edge_index.shape[1])}


def _is_undirected(edge_index):
    edges = _edge_set(edge_index)
    return all((j, i) in edges for (i, j) in edges)


@pytest.fixture
def points():
    rng = np.random.default_rng(0)
    return rng.uniform(0, 100, size=(50, 2))


def test_knn_shape_and_validity(points):
    edge_index = knn_graph(points, k=5)
    assert edge_index.shape[0] == 2
    assert edge_index.dtype == np.int64
    assert edge_index.min() >= 0
    assert edge_index.max() < points.shape[0]
    # no self loops, symmetric
    assert not any(i == j for i, j in _edge_set(edge_index))
    assert _is_undirected(edge_index)


def test_knn_directed_option(points):
    directed = knn_graph(points, k=4, symmetric=False)
    # each node has exactly 4 out-edges when directed and no self loops
    counts = np.bincount(directed[0], minlength=points.shape[0])
    assert np.all(counts == 4)


def test_knn_include_self(points):
    edge_index = knn_graph(points, k=3, include_self=True, symmetric=False)
    assert any(i == j for i, j in _edge_set(edge_index))


def test_knn_invalid_k(points):
    with pytest.raises(ValueError):
        knn_graph(points, k=0)


def test_radius_graph(points):
    edge_index = radius_graph(points, radius=20.0)
    lengths = edge_lengths(points, edge_index)
    assert np.all(lengths <= 20.0 + 1e-9)
    assert _is_undirected(edge_index)


def test_radius_graph_max_neighbors(points):
    edge_index = radius_graph(points, radius=40.0, max_neighbors=3)
    counts = np.bincount(edge_index[0], minlength=points.shape[0])
    assert counts.max() <= 3


def test_delaunay_graph():
    xs, ys = np.meshgrid(np.arange(5), np.arange(5))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    edge_index = delaunay_graph(coords)
    assert edge_index.shape[1] > 0
    assert _is_undirected(edge_index)
    assert edge_index.max() < coords.shape[0]


def test_delaunay_too_few_points():
    with pytest.raises(ValueError):
        delaunay_graph(np.array([[0.0, 0.0], [1.0, 1.0]]))


def test_contact_graph():
    coords = np.array([[0.0, 0.0], [5.0, 0.0], [50.0, 0.0]])
    edge_index = contact_graph(coords, cell_radii=3.0)
    edges = _edge_set(edge_index)
    assert (0, 1) in edges and (1, 0) in edges  # 3 + 3 >= 5
    assert (0, 2) not in edges  # far apart


def test_contact_graph_per_cell_radii():
    coords = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert contact_graph(coords, cell_radii=np.array([2.0, 2.0])).shape[1] == 0
    assert contact_graph(coords, cell_radii=np.array([4.0, 7.0])).shape[1] == 2


def test_build_graph_dispatch(points):
    assert build_graph(points, method="knn", k=4).shape[0] == 2
    with pytest.raises(ValueError):
        build_graph(points, method="nonexistent")


def test_helpers():
    edge_index = np.array([[0, 1], [1, 2]])
    und = to_undirected(edge_index)
    assert _is_undirected(und)
    looped = add_self_loops(und, num_nodes=3)
    assert all((i, i) in _edge_set(looped) for i in range(3))


def test_bad_coords_shape():
    with pytest.raises(ValueError):
        knn_graph(np.zeros((5, 4)), k=2)
