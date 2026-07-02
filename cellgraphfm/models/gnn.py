"""GNN encoder architectures (Research Dimension 4).

Requires the optional ``torch`` extra (``pip install "cellgraphfm[torch]"``).
A single :class:`GNNEncoder` builds any of the supported message-passing
architectures so they can be compared on equal footing.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    SAGEConv,
    TransformerConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)

_ATTENTION = {"gat", "gatv2", "graph_transformer"}
_POOLS = {
    "mean": global_mean_pool,
    "max": global_max_pool,
    "sum": global_add_pool,
}


class GNNEncoder(nn.Module):
    """A configurable stack of message-passing layers.

    Parameters
    ----------
    architecture:
        One of ``gcn``, ``graphsage``, ``gat``, ``gatv2``, ``gin``,
        ``graph_transformer``.
    in_channels:
        Input node-feature dimensionality.
    hidden_channels:
        Width of the hidden layers.
    out_channels:
        Output dimensionality (defaults to ``hidden_channels``).
    num_layers:
        Number of message-passing layers (``>= 1``).
    heads:
        Attention heads for attention-based architectures.
    dropout:
        Dropout probability applied between layers.
    pool:
        Optional graph-level readout: ``"mean"``, ``"max"``, or ``"sum"``. When
        set, :meth:`forward` returns one vector per graph; otherwise it returns
        per-node embeddings.
    """

    def __init__(
        self,
        architecture: str,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 2,
        *,
        heads: int = 4,
        dropout: float = 0.0,
        pool: str | None = None,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if pool is not None and pool not in _POOLS:
            raise ValueError(f"pool must be one of {sorted(_POOLS)} or None")

        self.architecture = architecture.lower()
        self.dropout = dropout
        self.pool = pool
        self.out_channels = out_channels or hidden_channels

        widths = [in_channels] + [hidden_channels] * (num_layers - 1) + [self.out_channels]
        self.convs = nn.ModuleList(
            self._make_conv(self.architecture, widths[i], widths[i + 1], heads)
            for i in range(num_layers)
        )

    def _make_conv(self, arch: str, cin: int, cout: int, heads: int) -> nn.Module:
        if arch == "gcn":
            return GCNConv(cin, cout)
        if arch == "graphsage":
            return SAGEConv(cin, cout)
        if arch == "gin":
            mlp = nn.Sequential(nn.Linear(cin, cout), nn.ReLU(), nn.Linear(cout, cout))
            return GINConv(mlp)
        if arch in _ATTENTION:
            h = heads if cout % heads == 0 else 1
            per_head = cout // h
            if arch == "gat":
                return GATConv(cin, per_head, heads=h, concat=True, dropout=self.dropout)
            if arch == "gatv2":
                return GATv2Conv(cin, per_head, heads=h, concat=True, dropout=self.dropout)
            return TransformerConv(cin, per_head, heads=h, concat=True)
        raise ValueError(f"unknown architecture '{arch}'")

    def _global_pool(self, x: torch.Tensor, batch: torch.Tensor | None) -> torch.Tensor:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        return _POOLS[self.pool](x, batch)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.pool is not None:
            x = self._global_pool(x, batch)
        return x
