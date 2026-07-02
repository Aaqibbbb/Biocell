"""Factory for GNN encoders (Research Dimension 4).

The factory itself has no heavy imports — ``torch``/``torch_geometric`` are only
imported when a model is actually built, so ``import cellgraphfm.models`` works
without the deep-learning stack installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from cellgraphfm.models.gnn import GNNEncoder

AVAILABLE_ARCHITECTURES: tuple[str, ...] = (
    "gcn",
    "graphsage",
    "gat",
    "gatv2",
    "gin",
    "graph_transformer",
)


def list_architectures() -> list[str]:
    """Return the names of the supported GNN architectures."""
    return list(AVAILABLE_ARCHITECTURES)


def build_gnn(
    name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int | None = None,
    num_layers: int = 2,
    **kwargs,
) -> GNNEncoder:
    """Build a :class:`~cellgraphfm.models.gnn.GNNEncoder` by architecture name.

    Parameters
    ----------
    name:
        One of :data:`AVAILABLE_ARCHITECTURES`.
    in_channels, hidden_channels, out_channels, num_layers:
        Standard encoder dimensions.
    **kwargs:
        Forwarded to :class:`~cellgraphfm.models.gnn.GNNEncoder`
        (``heads``, ``dropout``, ``pool``).
    """
    key = name.lower()
    if key not in AVAILABLE_ARCHITECTURES:
        raise ValueError(
            f"unknown architecture '{name}'. Choose from {list(AVAILABLE_ARCHITECTURES)}"
        )
    from cellgraphfm.models.gnn import GNNEncoder

    return GNNEncoder(
        architecture=key,
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        num_layers=num_layers,
        **kwargs,
    )
