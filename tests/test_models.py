"""Tests for the GNN model factory (Dimension 4). Requires the torch extra."""

from __future__ import annotations

import pytest

from cellgraphfm.models import build_gnn, list_architectures

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


ARCHS = list_architectures()


def _tiny_graph():
    x = torch.randn(5, 8)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long
    )
    return x, edge_index


def test_list_architectures():
    assert set(ARCHS) == {"gcn", "graphsage", "gat", "gatv2", "gin", "graph_transformer"}


@pytest.mark.parametrize("arch", ARCHS)
def test_node_level_forward(arch):
    x, edge_index = _tiny_graph()
    model = build_gnn(arch, in_channels=8, hidden_channels=16, out_channels=4, num_layers=2)
    out = model(x, edge_index)
    assert out.shape == (5, 4)


@pytest.mark.parametrize("arch", ARCHS)
def test_graph_level_pool(arch):
    x, edge_index = _tiny_graph()
    model = build_gnn(
        arch, in_channels=8, hidden_channels=16, out_channels=4, num_layers=3, pool="mean"
    )
    out = model(x, edge_index)
    assert out.shape == (1, 4)


def test_single_layer():
    x, edge_index = _tiny_graph()
    model = build_gnn("gcn", in_channels=8, hidden_channels=16, out_channels=6, num_layers=1)
    assert model(x, edge_index).shape == (5, 6)


def test_unknown_architecture():
    with pytest.raises(ValueError):
        build_gnn("transformer_xl", in_channels=8)


def test_bad_pool():
    with pytest.raises(ValueError):
        build_gnn("gcn", in_channels=8, pool="median")
