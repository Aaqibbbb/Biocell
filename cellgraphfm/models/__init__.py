"""GNN encoders (Research Dimension 4).

Only the factory helpers are exported at import time; the concrete
:class:`~cellgraphfm.models.gnn.GNNEncoder` (which imports ``torch``) is loaded
lazily by :func:`build_gnn`.
"""

from __future__ import annotations

from cellgraphfm.models.factory import (
    AVAILABLE_ARCHITECTURES,
    build_gnn,
    list_architectures,
)

__all__ = ["AVAILABLE_ARCHITECTURES", "build_gnn", "list_architectures"]
