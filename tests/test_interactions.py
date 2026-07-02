"""Tests for entity-based explanations (Dimension 7)."""

from __future__ import annotations

import numpy as np
import pytest

from cellgraphfm.data import TissueGraph
from cellgraphfm.explain import (
    find_interaction_paths,
    interaction_matrix,
    neighborhood_enrichment,
    top_interactions,
)


def test_interaction_matrix(tiny_graph):
    counts, type_ids = interaction_matrix(tiny_graph)
    assert type_ids.tolist() == [0, 1]
    np.testing.assert_array_equal(counts, [[2, 1], [1, 2]])


def test_top_interactions(tiny_graph):
    tops = top_interactions(tiny_graph, k=3)
    as_pairs = {(a, b): c for a, b, c in tops}
    assert as_pairs[(0, 0)] == 2
    assert as_pairs[(1, 1)] == 2
    assert as_pairs[(0, 1)] == 2  # folded 1 + 1


def test_find_interaction_paths(tiny_graph):
    paths = find_interaction_paths(tiny_graph, [0, 1])
    assert [1, 2] in paths
    assert all(len(p) == 2 for p in paths)

    longer = find_interaction_paths(tiny_graph, [0, 1, 1])
    assert [1, 2, 3] in longer


def test_find_interaction_paths_max(tiny_graph):
    paths = find_interaction_paths(tiny_graph, [0, 0], max_paths=1)
    assert len(paths) == 1


def test_requires_cell_types():
    g = TissueGraph(np.zeros((3, 2)), np.zeros((3, 1)), np.empty((2, 0)))
    with pytest.raises(ValueError):
        interaction_matrix(g)
    with pytest.raises(ValueError):
        find_interaction_paths(g, [0])


def test_neighborhood_enrichment_positive_diagonal(graph):
    zscore, type_ids = neighborhood_enrichment(graph, n_permutations=200, seed=0)
    assert zscore.shape == (type_ids.size, type_ids.size)
    # clustered tissue => same-type contacts are enriched
    assert np.mean(np.diag(zscore)) > 0.0


def test_neighborhood_enrichment_bad_permutations(graph):
    with pytest.raises(ValueError):
        neighborhood_enrichment(graph, n_permutations=0)
