"""Entity-based explainability for cell graphs (Research Dimension 7)."""

from __future__ import annotations

from cellgraphfm.explain.interactions import (
    find_interaction_paths,
    interaction_matrix,
    neighborhood_enrichment,
    top_interactions,
)

__all__ = [
    "find_interaction_paths",
    "interaction_matrix",
    "neighborhood_enrichment",
    "top_interactions",
]
