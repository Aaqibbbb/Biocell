"""The :class:`TissueGraph` — CellGraphFM's core cell-graph data structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from cellgraphfm.utils.deps import require_dependency

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx


@dataclass
class TissueGraph:
    """A graph of cells extracted from a tissue region.

    Parameters
    ----------
    coords:
        ``(N, 2)`` or ``(N, 3)`` array of cell centroid coordinates.
    node_features:
        ``(N, F)`` array of per-cell feature vectors (morphology, foundation-model
        embeddings, cell-type probabilities, ...).
    edge_index:
        ``(2, E)`` integer array of directed edges. Undirected graphs are stored
        with both ``(i, j)`` and ``(j, i)`` present.
    cell_types:
        Optional ``(N,)`` integer array of cell-type labels.
    edge_attr:
        Optional ``(E, D)`` array of per-edge features (e.g. distance).
    metadata:
        Free-form dictionary for provenance, scale level, slide id, etc.
    """

    coords: np.ndarray
    node_features: np.ndarray
    edge_index: np.ndarray
    cell_types: np.ndarray | None = None
    edge_attr: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.coords = np.asarray(self.coords)
        self.node_features = np.asarray(self.node_features)
        self.edge_index = np.asarray(self.edge_index)

        if self.coords.ndim != 2 or self.coords.shape[1] not in (2, 3):
            raise ValueError(f"coords must have shape (N, 2) or (N, 3), got {self.coords.shape}")
        n = self.coords.shape[0]

        if self.node_features.ndim != 2 or self.node_features.shape[0] != n:
            raise ValueError(
                f"node_features must have shape (N, F) with N={n}, got {self.node_features.shape}"
            )

        if self.edge_index.size == 0:
            self.edge_index = np.empty((2, 0), dtype=np.int64)
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape (2, E), got {self.edge_index.shape}")
        self.edge_index = self.edge_index.astype(np.int64, copy=False)
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index contains out-of-range node indices")

        if self.cell_types is not None:
            self.cell_types = np.asarray(self.cell_types)
            if self.cell_types.shape != (n,):
                raise ValueError(
                    f"cell_types must have shape (N,) with N={n}, got {self.cell_types.shape}"
                )

        if self.edge_attr is not None:
            self.edge_attr = np.asarray(self.edge_attr)
            if self.edge_attr.shape[0] != self.edge_index.shape[1]:
                raise ValueError(
                    "edge_attr must have one row per edge "
                    f"({self.edge_index.shape[1]}), got {self.edge_attr.shape[0]}"
                )

    # -- basic properties -------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return int(self.coords.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_node_features(self) -> int:
        return int(self.node_features.shape[1])

    @property
    def is_empty(self) -> bool:
        return self.num_nodes == 0

    def degree(self) -> np.ndarray:
        """Return the out-degree of every node as an ``(N,)`` array."""
        if self.num_edges == 0:
            return np.zeros(self.num_nodes, dtype=np.int64)
        return np.bincount(self.edge_index[0], minlength=self.num_nodes)

    def neighbors(self, node: int) -> np.ndarray:
        """Return the destination nodes of edges originating at ``node``."""
        mask = self.edge_index[0] == node
        return self.edge_index[1, mask]

    # -- transforms -------------------------------------------------------

    def subgraph(self, nodes: np.ndarray) -> TissueGraph:
        """Return the induced subgraph over ``nodes`` (relabeled from 0)."""
        nodes = np.unique(np.asarray(nodes, dtype=np.int64))
        if nodes.size and (nodes.min() < 0 or nodes.max() >= self.num_nodes):
            raise ValueError("node indices out of range")

        relabel = -np.ones(self.num_nodes, dtype=np.int64)
        relabel[nodes] = np.arange(nodes.size)

        keep = np.isin(self.edge_index[0], nodes) & np.isin(self.edge_index[1], nodes)
        new_edges = relabel[self.edge_index[:, keep]] if self.num_edges else self.edge_index

        return TissueGraph(
            coords=self.coords[nodes],
            node_features=self.node_features[nodes],
            edge_index=new_edges,
            cell_types=None if self.cell_types is None else self.cell_types[nodes],
            edge_attr=None if self.edge_attr is None else self.edge_attr[keep],
            metadata=dict(self.metadata),
        )

    def to_pyg(self):  # noqa: ANN201 - torch_geometric.data.Data (optional dep)
        """Convert to a :class:`torch_geometric.data.Data` object."""
        torch = require_dependency("torch")
        geo_data = require_dependency("torch_geometric.data")

        data = geo_data.Data(
            x=torch.as_tensor(self.node_features, dtype=torch.float32),
            edge_index=torch.as_tensor(self.edge_index, dtype=torch.long),
            pos=torch.as_tensor(self.coords, dtype=torch.float32),
        )
        if self.cell_types is not None:
            data.y = torch.as_tensor(self.cell_types, dtype=torch.long)
        if self.edge_attr is not None:
            data.edge_attr = torch.as_tensor(self.edge_attr, dtype=torch.float32)
        return data

    def to_networkx(self) -> nx.Graph:
        """Convert to an undirected :class:`networkx.Graph`."""
        nx = require_dependency("networkx")
        graph = nx.Graph()
        for i in range(self.num_nodes):
            attrs: dict[str, Any] = {"pos": tuple(self.coords[i].tolist())}
            if self.cell_types is not None:
                attrs["cell_type"] = int(self.cell_types[i])
            graph.add_node(i, **attrs)
        for e in range(self.num_edges):
            graph.add_edge(int(self.edge_index[0, e]), int(self.edge_index[1, e]))
        return graph

    def __repr__(self) -> str:
        n_types = None if self.cell_types is None else int(np.unique(self.cell_types).size)
        return (
            f"TissueGraph(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"features={self.num_node_features}, cell_types={n_types})"
        )
