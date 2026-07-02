"""Tests for tumour/immune spatial scoring over cell graphs."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.data import TissueGraph
from cellgraphfm.scoring import (
    immune_infiltration,
    mixing_score,
    til_density,
    tumor_cellularity,
)


def _make_graph(cell_types, undirected_pairs):
    """Build a small undirected TissueGraph from cell types and edge pairs."""
    n = len(cell_types)
    coords = np.arange(n * 2, dtype=float).reshape(n, 2)
    features = np.zeros((n, 1))
    edges: list[tuple[int, int]] = []
    for a, b in undirected_pairs:
        edges.append((a, b))
        edges.append((b, a))
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.empty((2, 0), dtype=np.int64)
    return TissueGraph(coords, features, edge_index, cell_types=np.array(cell_types))


def test_tumor_cellularity_and_til_density():
    graph = _make_graph([0, 1, 1, 3], [(0, 1)])
    assert tumor_cellularity(graph, tumor_type=0) == pytest.approx(0.25)
    assert til_density(graph, immune_type=1) == pytest.approx(0.5)


def test_immune_infiltration_partial():
    # tumour(0) touches immune(1) and other(2): half of its edges are immune
    graph = _make_graph([0, 1, 3], [(0, 1), (0, 2)])
    assert immune_infiltration(graph, tumor_type=0, immune_type=1) == pytest.approx(0.5)


def test_immune_infiltration_full():
    graph = _make_graph([0, 1, 0, 1], [(0, 1), (2, 3)])
    assert immune_infiltration(graph, tumor_type=0, immune_type=1) == pytest.approx(1.0)


def test_immune_infiltration_no_edges():
    graph = _make_graph([0, 1, 3], [])
    assert immune_infiltration(graph, tumor_type=0, immune_type=1) == 0.0


def test_mixing_score_segregated_vs_mixed():
    segregated = _make_graph([0, 0, 1, 1], [(0, 1), (2, 3)])  # tumour|tumour, immune|immune
    assert mixing_score(segregated, tumor_type=0, immune_type=1) == pytest.approx(0.0)

    mixed = _make_graph([0, 0, 1, 1], [(0, 2), (0, 3), (1, 2), (1, 3)])  # all cross-type
    assert mixing_score(mixed, tumor_type=0, immune_type=1) == pytest.approx(2.0)


def test_mixing_score_missing_type_and_no_edges():
    graph = _make_graph([0, 0, 1, 1], [(0, 2), (1, 3)])
    assert mixing_score(graph, tumor_type=0, immune_type=5) == 0.0  # immune type absent
    empty = _make_graph([0, 1], [])
    assert mixing_score(empty, tumor_type=0, immune_type=1) == 0.0


def test_empty_cell_type_fractions():
    empty = TissueGraph(
        np.empty((0, 2)),
        np.empty((0, 1)),
        np.empty((2, 0)),
        cell_types=np.empty((0,), dtype=int),
    )
    assert tumor_cellularity(empty, tumor_type=0) == 0.0
    assert til_density(empty, immune_type=1) == 0.0


def test_requires_cell_types():
    g = TissueGraph(np.zeros((3, 2)), np.zeros((3, 1)), np.empty((2, 0)))
    with pytest.raises(ValueError):
        tumor_cellularity(g, tumor_type=0)
    with pytest.raises(ValueError):
        til_density(g, immune_type=1)
    with pytest.raises(ValueError):
        immune_infiltration(g, tumor_type=0, immune_type=1)
    with pytest.raises(ValueError):
        mixing_score(g, tumor_type=0, immune_type=1)
