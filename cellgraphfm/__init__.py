"""CellGraphFM — cellular graph intelligence for computational pathology.

The package is organised around the CellGraphFM research dimensions:

* :mod:`cellgraphfm.data` — the :class:`~cellgraphfm.data.tissue_graph.TissueGraph`
  data structure.
* :mod:`cellgraphfm.graph` — graph construction from cell coordinates.
* :mod:`cellgraphfm.features` — cell representation (morphology + embeddings).
* :mod:`cellgraphfm.models` — GNN encoders (optional ``torch`` extra).
* :mod:`cellgraphfm.ssl` — self-supervised objectives.
* :mod:`cellgraphfm.explain` — entity-based interaction explanations.
* :mod:`cellgraphfm.pipeline` — end-to-end helpers.

Only ``numpy`` and ``scipy`` are required for the core; deep-learning features
are lazily imported from the optional ``torch`` extra.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("cellgraphfm")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

from cellgraphfm.data import TissueGraph
from cellgraphfm.graph import (
    build_graph,
    contact_graph,
    delaunay_graph,
    knn_graph,
    radius_graph,
)

__all__ = [
    "TissueGraph",
    "__version__",
    "build_graph",
    "contact_graph",
    "delaunay_graph",
    "knn_graph",
    "radius_graph",
]
