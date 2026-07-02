"""Entity-based explanations over cell graphs (Research Dimension 7).

Instead of pixel heatmaps, CellGraphFM explains tissue in the language a
pathologist uses: *which cell types interact, how much, and along which paths*.
All functions here are pure ``numpy`` and operate on a
:class:`~cellgraphfm.data.tissue_graph.TissueGraph` that carries ``cell_types``.
"""

from __future__ import annotations

import numpy as np

from cellgraphfm.data.tissue_graph import TissueGraph


def _require_types(graph: TissueGraph) -> np.ndarray:
    if graph.cell_types is None:
        raise ValueError("graph.cell_types is required for interaction analysis")
    return graph.cell_types


def _mapped_types(graph: TissueGraph) -> tuple[np.ndarray, np.ndarray]:
    """Return (type_ids, mapped) where mapped[i] is the index of node i's type."""
    types = _require_types(graph)
    type_ids = np.unique(types)
    mapped = np.searchsorted(type_ids, types)
    return type_ids, mapped


def _count_matrix(mapped: np.ndarray, edge_index: np.ndarray, t: int) -> np.ndarray:
    matrix = np.zeros((t, t), dtype=np.float64)
    if edge_index.size:
        np.add.at(matrix, (mapped[edge_index[0]], mapped[edge_index[1]]), 1.0)
    return matrix


def interaction_matrix(graph: TissueGraph) -> tuple[np.ndarray, np.ndarray]:
    """Count edges between every ordered pair of cell types.

    Returns
    -------
    counts:
        ``(T, T)`` matrix; ``counts[a, b]`` is the number of directed edges from a
        cell of type ``type_ids[a]`` to one of type ``type_ids[b]``.
    type_ids:
        ``(T,)`` sorted array of the distinct cell-type labels.
    """
    type_ids, mapped = _mapped_types(graph)
    counts = _count_matrix(mapped, graph.edge_index, type_ids.size)
    return counts, type_ids


def neighborhood_enrichment(
    graph: TissueGraph,
    n_permutations: int = 100,
    *,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Permutation-test enrichment of cell-type adjacency.

    For each cell-type pair this compares the observed number of contacts against
    a null distribution built by shuffling the cell-type labels across nodes
    (the standard neighbourhood-enrichment test). Positive z-scores mean the two
    types are found next to each other more than chance; negative means avoidance.

    Returns ``(zscore, type_ids)`` where ``zscore`` is ``(T, T)``.
    """
    if n_permutations < 1:
        raise ValueError("n_permutations must be >= 1")
    type_ids, mapped = _mapped_types(graph)
    t = type_ids.size
    edge_index = graph.edge_index

    observed = _count_matrix(mapped, edge_index, t)

    rng = np.random.default_rng(seed)
    perm_counts = np.empty((n_permutations, t, t), dtype=np.float64)
    for i in range(n_permutations):
        permuted = rng.permutation(mapped)
        perm_counts[i] = _count_matrix(permuted, edge_index, t)

    mean = perm_counts.mean(axis=0)
    std = perm_counts.std(axis=0)
    zscore = np.zeros((t, t), dtype=np.float64)
    nonzero = std > 0
    zscore[nonzero] = (observed[nonzero] - mean[nonzero]) / std[nonzero]
    return zscore, type_ids


def top_interactions(
    graph: TissueGraph,
    k: int = 5,
) -> list[tuple[int, int, int]]:
    """Return the ``k`` most frequent unordered cell-type interactions.

    Each entry is ``(type_a, type_b, count)`` with ``type_a <= type_b``.
    """
    counts, type_ids = interaction_matrix(graph)
    # Fold the directed matrix into unordered pairs.
    folded = np.triu(counts) + np.triu(counts.T, 1)
    pairs: list[tuple[int, int, int]] = []
    for a in range(folded.shape[0]):
        for b in range(a, folded.shape[1]):
            c = int(folded[a, b])
            if c > 0:
                pairs.append((int(type_ids[a]), int(type_ids[b]), c))
    pairs.sort(key=lambda item: item[2], reverse=True)
    return pairs[:k]


def _adjacency(graph: TissueGraph) -> list[np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(graph.num_nodes)]
    src, dst = graph.edge_index
    for s, d in zip(src.tolist(), dst.tolist(), strict=False):
        adj[s].append(d)
    return [np.array(sorted(set(neigh)), dtype=np.int64) for neigh in adj]


def find_interaction_paths(
    graph: TissueGraph,
    type_sequence: list[int],
    *,
    max_paths: int | None = None,
) -> list[list[int]]:
    """Find simple paths whose node cell-types match ``type_sequence`` in order.

    This surfaces interpretable multi-cell motifs such as a tumour cell adjacent
    to a fibroblast adjacent to a macrophage adjacent to a T cell — the kind of
    high-order interaction that carries prognostic meaning.

    Parameters
    ----------
    type_sequence:
        Ordered list of cell-type labels the path must follow.
    max_paths:
        Optional cap on the number of returned paths.

    Returns a list of paths, each a list of node indices of length
    ``len(type_sequence)``.
    """
    types = _require_types(graph)
    seq = list(type_sequence)
    if len(seq) < 1:
        raise ValueError("type_sequence must contain at least one type")

    adj = _adjacency(graph)
    results: list[list[int]] = []
    starts = np.where(types == seq[0])[0]

    def dfs(path: list[int], visited: set[int]) -> None:
        if max_paths is not None and len(results) >= max_paths:
            return
        depth = len(path)
        if depth == len(seq):
            results.append(list(path))
            return
        want = seq[depth]
        for nxt in adj[path[-1]].tolist():
            if nxt in visited:
                continue
            if types[nxt] == want:
                visited.add(nxt)
                path.append(nxt)
                dfs(path, visited)
                path.pop()
                visited.remove(nxt)
            if max_paths is not None and len(results) >= max_paths:
                return

    for start in starts.tolist():
        if max_paths is not None and len(results) >= max_paths:
            break
        dfs([start], {start})
    return results
