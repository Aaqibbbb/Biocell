"""End-to-end helpers: cells -> graph (mask -> cells -> graph)."""

from __future__ import annotations

from typing import Any

import numpy as np

from cellgraphfm.data.tissue_graph import TissueGraph
from cellgraphfm.features.morphology import extract_morphology
from cellgraphfm.graph.construction import build_graph, edge_lengths


def build_tissue_graph(
    coords: np.ndarray,
    node_features: np.ndarray,
    cell_types: np.ndarray | None = None,
    *,
    method: str = "knn",
    metadata: dict[str, Any] | None = None,
    with_edge_length: bool = True,
    **kwargs,
) -> TissueGraph:
    """Build a :class:`TissueGraph` from coordinates and node features.

    Extra keyword arguments are forwarded to the graph constructor
    (e.g. ``k`` for ``method="knn"`` or ``radius`` for ``method="radius"``).
    """
    edge_index = build_graph(coords, method=method, **kwargs)
    edge_attr = None
    if with_edge_length and edge_index.size:
        edge_attr = edge_lengths(coords, edge_index)[:, None]
    meta = {"graph_method": method}
    if metadata:
        meta.update(metadata)
    return TissueGraph(
        coords=coords,
        node_features=node_features,
        edge_index=edge_index,
        cell_types=cell_types,
        edge_attr=edge_attr,
        metadata=meta,
    )


def graph_from_mask(
    mask: np.ndarray,
    intensity: np.ndarray | None = None,
    *,
    method: str = "delaunay",
    cell_types: np.ndarray | None = None,
    **kwargs,
) -> TissueGraph:
    """Segmentation mask -> morphology features -> cell graph.

    Cell centroids become node coordinates and morphology descriptors become node
    features. The morphology feature names are stored in ``graph.metadata``.
    """
    morph = extract_morphology(mask, intensity)
    metadata = {"feature_names": morph.feature_names, "source": "mask"}
    return build_tissue_graph(
        morph.centroids,
        morph.features,
        cell_types=cell_types,
        method=method,
        metadata=metadata,
        **kwargs,
    )


def synthetic_tissue(
    n_cells: int = 200,
    n_types: int = 4,
    *,
    size: float = 1000.0,
    n_features: int = 16,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a spatially-clustered synthetic tissue section.

    Cells are drawn around ``n_types`` spatial cluster centres so that same-type
    neighbourhood enrichment is meaningfully positive — handy for demos and tests.

    Returns ``(coords, features, cell_types)``.
    """
    if n_cells < 1 or n_types < 1:
        raise ValueError("n_cells and n_types must be >= 1")
    rng = np.random.default_rng(seed)

    centers = rng.uniform(0.15 * size, 0.85 * size, size=(n_types, 2))
    cell_types = rng.integers(0, n_types, size=n_cells)
    spread = 0.12 * size
    coords = centers[cell_types] + rng.normal(0.0, spread, size=(n_cells, 2))
    coords = np.clip(coords, 0.0, size)

    # Type-correlated features + noise.
    type_signatures = rng.normal(size=(n_types, n_features))
    features = type_signatures[cell_types] + rng.normal(0.0, 0.5, size=(n_cells, n_features))

    return coords, features, cell_types


def demo_graph(
    n_cells: int = 200,
    *,
    method: str = "knn",
    seed: int = 0,
    **kwargs,
) -> TissueGraph:
    """Build a ready-to-use synthetic :class:`TissueGraph` for demos and tests."""
    coords, features, cell_types = synthetic_tissue(n_cells=n_cells, seed=seed)
    return build_tissue_graph(coords, features, cell_types=cell_types, method=method, **kwargs)
