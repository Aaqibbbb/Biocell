"""Tumour and immune scoring over cell graphs (spatial biology analog).

Spatial read-outs used in immuno-oncology, computed directly from a
:class:`~cellgraphfm.data.tissue_graph.TissueGraph` that carries ``cell_types``:
tumour cellularity, tumour-infiltrating-lymphocyte (TIL) density, immune
infiltration of the tumour compartment, and a mixing score that compares
observed tumour-immune contacts against a random-mixing null. Pure ``numpy``.
"""

from __future__ import annotations

import numpy as np

from cellgraphfm.data.tissue_graph import TissueGraph
from cellgraphfm.explain.interactions import interaction_matrix

__all__ = [
    "immune_infiltration",
    "mixing_score",
    "til_density",
    "tumor_cellularity",
]


def _require_types(graph: TissueGraph) -> np.ndarray:
    if graph.cell_types is None:
        raise ValueError("graph.cell_types is required for tumour/immune scoring")
    return graph.cell_types


def tumor_cellularity(graph: TissueGraph, *, tumor_type: int) -> float:
    """Return the fraction of cells that are tumour cells.

    Parameters
    ----------
    graph:
        A :class:`TissueGraph` carrying ``cell_types``.
    tumor_type:
        Cell-type label denoting tumour cells.
    """
    types = _require_types(graph)
    if types.size == 0:
        return 0.0
    return float(np.mean(types == tumor_type))


def til_density(graph: TissueGraph, *, immune_type: int) -> float:
    """Return the fraction of cells that are the immune (lymphocyte) type.

    Parameters
    ----------
    graph:
        A :class:`TissueGraph` carrying ``cell_types``.
    immune_type:
        Cell-type label denoting immune cells.
    """
    types = _require_types(graph)
    if types.size == 0:
        return 0.0
    return float(np.mean(types == immune_type))


def immune_infiltration(graph: TissueGraph, *, tumor_type: int, immune_type: int) -> float:
    """Fraction of tumour-immune edges among all edges incident to tumour cells.

    A high value means tumour cells are largely in contact with immune cells (an
    infiltrated / "hot" tumour); a low value means tumour cells mostly contact
    other cell types.

    Parameters
    ----------
    graph:
        A :class:`TissueGraph` carrying ``cell_types``.
    tumor_type, immune_type:
        Cell-type labels for tumour and immune cells.

    Returns
    -------
    fraction:
        Value in ``[0, 1]`` (``0.0`` when no tumour-incident edges exist).
    """
    types = _require_types(graph)
    src, dst = graph.edge_index
    if src.size == 0:
        return 0.0

    src_tumor = types[src] == tumor_type
    dst_tumor = types[dst] == tumor_type
    incident = src_tumor | dst_tumor
    n_incident = int(np.count_nonzero(incident))
    if n_incident == 0:
        return 0.0

    src_immune = types[src] == immune_type
    dst_immune = types[dst] == immune_type
    tumor_immune = (src_tumor & dst_immune) | (src_immune & dst_tumor)
    return float(np.count_nonzero(tumor_immune) / n_incident)


def mixing_score(graph: TissueGraph, *, tumor_type: int, immune_type: int) -> float:
    """Ratio of observed tumour-immune contacts to those expected at random.

    Observed tumour-immune (directed) edges are counted from the cell-type
    interaction matrix. The expectation under random label mixing is
    ``E * 2 * p_tumor * p_immune`` where ``E`` is the edge count and ``p_*`` the
    cell-type proportions. A score ``> 1`` means the two compartments intermix
    more than chance; ``< 1`` means they segregate.

    Parameters
    ----------
    graph:
        A :class:`TissueGraph` carrying ``cell_types``.
    tumor_type, immune_type:
        Cell-type labels for tumour and immune cells.

    Returns
    -------
    score:
        Non-negative mixing ratio (``0.0`` when the expectation is zero, e.g. a
        type is absent or there are no edges).
    """
    types = _require_types(graph)
    total_edges = graph.num_edges
    if total_edges == 0:
        return 0.0

    counts, type_ids = interaction_matrix(graph)
    tumor_idx = np.where(type_ids == tumor_type)[0]
    immune_idx = np.where(type_ids == immune_type)[0]
    if tumor_idx.size == 0 or immune_idx.size == 0:
        return 0.0
    ti, ii = int(tumor_idx[0]), int(immune_idx[0])
    observed = float(counts[ti, ii] + counts[ii, ti])

    p_tumor = float(np.mean(types == tumor_type))
    p_immune = float(np.mean(types == immune_type))
    expected = total_edges * 2.0 * p_tumor * p_immune
    if expected <= 0.0:
        return 0.0
    return observed / expected
