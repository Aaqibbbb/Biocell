"""Graph construction from cell coordinates (Research Dimension 2).

Every function returns an ``edge_index`` array of shape ``(2, E)`` with dtype
``int64`` describing directed edges. Undirected constructions return both
``(i, j)`` and ``(j, i)``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay, cKDTree

__all__ = [
    "build_graph",
    "knn_graph",
    "radius_graph",
    "delaunay_graph",
    "contact_graph",
    "to_undirected",
    "add_self_loops",
    "edge_lengths",
]


def _as_coords(coords: np.ndarray) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError(f"coords must have shape (N, 2) or (N, 3), got {coords.shape}")
    return coords


def _pairs_to_edge_index(pairs: np.ndarray) -> np.ndarray:
    """Turn an ``(E, 2)`` array of ordered pairs into a ``(2, E)`` edge_index."""
    if pairs.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    return np.ascontiguousarray(pairs.T.astype(np.int64))


def to_undirected(edge_index: np.ndarray) -> np.ndarray:
    """Symmetrize ``edge_index`` and drop duplicate directed edges."""
    if edge_index.size == 0:
        return np.empty((2, 0), dtype=np.int64)
    both = np.concatenate([edge_index, edge_index[::-1]], axis=1)
    uniq = np.unique(both.T, axis=0)
    return np.ascontiguousarray(uniq.T.astype(np.int64))


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    """Add ``(i, i)`` self-loops for every node, de-duplicating existing ones."""
    loops = np.repeat(np.arange(num_nodes, dtype=np.int64)[None, :], 2, axis=0)
    combined = np.concatenate([edge_index, loops], axis=1) if edge_index.size else loops
    uniq = np.unique(combined.T, axis=0)
    return np.ascontiguousarray(uniq.T.astype(np.int64))


def knn_graph(
    coords: np.ndarray,
    k: int = 6,
    *,
    symmetric: bool = True,
    include_self: bool = False,
) -> np.ndarray:
    """Connect each cell to its ``k`` nearest neighbours.

    Parameters
    ----------
    coords:
        ``(N, D)`` cell coordinates.
    k:
        Number of neighbours per node.
    symmetric:
        If ``True`` (default) the result is made undirected.
    include_self:
        Whether a node counts as its own neighbour.
    """
    coords = _as_coords(coords)
    n = coords.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    query_k = min(n, k + (0 if include_self else 1))
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=query_k)
    idx = np.atleast_2d(idx)
    if idx.shape[0] == 1 and n != 1:
        idx = idx.T  # query returns 1-D when query_k == 1

    sources = np.repeat(np.arange(n), idx.shape[1])
    targets = idx.reshape(-1)
    pairs = np.stack([sources, targets], axis=1)
    if not include_self:
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    edge_index = _pairs_to_edge_index(pairs)
    return to_undirected(edge_index) if symmetric else edge_index


def radius_graph(
    coords: np.ndarray,
    radius: float,
    *,
    include_self: bool = False,
    max_neighbors: int | None = None,
) -> np.ndarray:
    """Connect every pair of cells within Euclidean ``radius`` (undirected)."""
    coords = _as_coords(coords)
    n = coords.shape[0]
    if radius <= 0:
        raise ValueError("radius must be > 0")
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    tree = cKDTree(coords)
    pairs = np.array(sorted(tree.query_pairs(radius)), dtype=np.int64)
    if pairs.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    else:
        edge_index = to_undirected(_pairs_to_edge_index(pairs))

    if include_self:
        edge_index = add_self_loops(edge_index, n)
    if max_neighbors is not None:
        edge_index = _cap_degree(edge_index, coords, max_neighbors)
    return edge_index


def delaunay_graph(coords: np.ndarray) -> np.ndarray:
    """Build a Delaunay triangulation graph (undirected).

    Delaunay edges capture natural cell adjacency without a distance threshold.
    Requires at least 3 non-collinear points.
    """
    coords = _as_coords(coords)
    n = coords.shape[0]
    if n < 3:
        raise ValueError("Delaunay graph requires at least 3 points")

    try:
        tri = Delaunay(coords)
    except Exception as exc:  # pragma: no cover - degenerate input
        raise ValueError(
            "Delaunay triangulation failed (points may be collinear/degenerate)"
        ) from exc

    simplices = tri.simplices
    # Every unordered pair of vertices within a simplex is an edge.
    edges = set()
    for simplex in simplices:
        for a_idx in range(len(simplex)):
            for b_idx in range(a_idx + 1, len(simplex)):
                a, b = int(simplex[a_idx]), int(simplex[b_idx])
                edges.add((a, b) if a < b else (b, a))
    pairs = np.array(sorted(edges), dtype=np.int64)
    return to_undirected(_pairs_to_edge_index(pairs))


def contact_graph(
    coords: np.ndarray,
    cell_radii: np.ndarray | float,
    *,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Connect cells whose bodies touch, given per-cell radii.

    Two cells ``i`` and ``j`` are connected when
    ``dist(i, j) <= r_i + r_j + tolerance``. This approximates membrane contact
    from segmentation without needing the full masks.
    """
    coords = _as_coords(coords)
    n = coords.shape[0]
    if n == 0:
        return np.empty((2, 0), dtype=np.int64)

    if np.isscalar(cell_radii):
        radii = np.full(n, float(cell_radii))
    else:
        radii = np.asarray(cell_radii, dtype=np.float64)
        if radii.shape != (n,):
            raise ValueError(f"cell_radii must be scalar or shape (N,)={n}")
    if np.any(radii < 0):
        raise ValueError("cell_radii must be non-negative")

    tree = cKDTree(coords)
    max_reach = float(radii.max() * 2 + tolerance)
    candidate_pairs = tree.query_pairs(max_reach)

    kept = []
    for i, j in candidate_pairs:
        dist = float(np.linalg.norm(coords[i] - coords[j]))
        if dist <= radii[i] + radii[j] + tolerance:
            kept.append((i, j))
    if not kept:
        return np.empty((2, 0), dtype=np.int64)
    pairs = np.array(sorted(kept), dtype=np.int64)
    return to_undirected(_pairs_to_edge_index(pairs))


def edge_lengths(coords: np.ndarray, edge_index: np.ndarray) -> np.ndarray:
    """Return the Euclidean length of every edge as an ``(E,)`` array."""
    coords = _as_coords(coords)
    if edge_index.size == 0:
        return np.empty((0,), dtype=np.float64)
    src, dst = edge_index
    return np.linalg.norm(coords[src] - coords[dst], axis=1)


def _cap_degree(edge_index: np.ndarray, coords: np.ndarray, max_neighbors: int) -> np.ndarray:
    """Keep only the ``max_neighbors`` closest edges per source node."""
    if edge_index.size == 0:
        return edge_index
    lengths = edge_lengths(coords, edge_index)
    order = np.lexsort((lengths, edge_index[0]))
    src_sorted = edge_index[0, order]
    kept_mask = np.zeros(edge_index.shape[1], dtype=bool)
    counts: dict[int, int] = {}
    for pos in range(order.size):
        s = int(src_sorted[pos])
        if counts.get(s, 0) < max_neighbors:
            kept_mask[order[pos]] = True
            counts[s] = counts.get(s, 0) + 1
    return np.ascontiguousarray(edge_index[:, kept_mask])


_BUILDERS = {
    "knn": knn_graph,
    "radius": radius_graph,
    "delaunay": delaunay_graph,
    "contact": contact_graph,
}


def build_graph(coords: np.ndarray, method: str = "knn", **kwargs) -> np.ndarray:
    """Dispatch to a named graph constructor.

    Parameters
    ----------
    coords:
        ``(N, D)`` cell coordinates.
    method:
        One of ``"knn"``, ``"radius"``, ``"delaunay"``, ``"contact"``.
    **kwargs:
        Passed through to the chosen constructor.
    """
    key = method.lower()
    if key not in _BUILDERS:
        raise ValueError(f"unknown method '{method}'. Choose from {sorted(_BUILDERS)}")
    return _BUILDERS[key](coords, **kwargs)
