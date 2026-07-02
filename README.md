<div align="center">

# CellGraphFM

**A Foundation Model toolkit for Cellular Graph Intelligence in Computational Pathology**

*Represent tissue as interacting biological entities — not just pixels.*

[![CI](https://github.com/Aaqibbbb/Biocell/actions/workflows/ci.yml/badge.svg)](https://github.com/Aaqibbbb/Biocell/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)

</div>

---

## Vision

Traditional computer vision treats a pathology slide as a grid of pixels. Human
pathologists do not — they reason about **cells**, **tissue organization**,
**immune infiltration**, **tumor architecture**, and the **spatial context** in
which biological interactions happen.

**CellGraphFM** is an open-source, modular, and explainable toolkit for turning
histopathology into *biological graphs* and learning universal representations
over them. The pipeline it scaffolds is:

```
Whole Slide → Cell Detection → Cell Typing → Biological Graph
            → Graph Foundation Model → Reasoning → Clinical Prediction → Discovery
```

The goal is not merely better classification accuracy, but a **reusable
biological representation** that supports diagnosis, prognosis, biomarker
discovery, and scientific hypothesis generation.

## Core research question

> Can tissue be represented as an evolving biological graph rather than an image,
> and can graph foundation models learn universal biological representations that
> generalize across cancers, organs, imaging modalities, and downstream tasks?

## What's in this repository

This repository is the **engineering foundation** for that research program. It
provides clean, tested, dependency-light building blocks so that experiments can
be run reproducibly today, without waiting on a monolithic training stack.

| Module | Purpose |
| --- | --- |
| `cellgraphfm.data` | `TissueGraph` — the core cell-graph data structure with conversions to PyG / NetworkX. |
| `cellgraphfm.graph` | Graph construction: KNN, radius, Delaunay, and contact graphs from cell coordinates. |
| `cellgraphfm.features` | Morphology feature extraction from segmentation masks; foundation-model embedding interfaces (UNI, Virchow, Prov-GigaPath, …). |
| `cellgraphfm.models` | GNN encoder factory: GCN, GraphSAGE, GAT, GATv2, GIN, and Graph Transformer (PyTorch Geometric). |
| `cellgraphfm.ssl` | Self-supervised objectives: node/feature masking and a GraphMAE-style reconstruction loss. |
| `cellgraphfm.explain` | Entity-based explanations: cell-type interaction matrices, neighborhood enrichment, and interaction-path mining. |
| `cellgraphfm.pipeline` | End-to-end helpers: mask → cells → graph. |
| `cellgraphfm.cli` | `cellgraphfm` command-line entry point. |

### Design principles

- **Lightweight core.** The core (`numpy` + `scipy`) always imports and always
  runs in CI. Heavy dependencies (`torch`, `torch-geometric`) live behind
  optional extras and lazy imports, so the graph and reasoning layers are usable
  even without a deep-learning stack.
- **Explainable by construction.** Explanations are expressed in the language a
  pathologist uses — *cell types and their interactions* — not opaque heatmaps.
- **Modular.** Every research dimension (representation, construction, multi-scale
  graphs, architectures, SSL, reasoning, explainability, foundation-model
  integration, multimodality, discovery) maps to a swappable component.

## Installation

```bash
# core (numpy + scipy only)
pip install -e .

# with the PyTorch Geometric model layer
pip install -e ".[torch]"

# everything, including visualization + dev tooling
pip install -e ".[torch,viz,dev]"
```

## Quick start

```python
import numpy as np
from cellgraphfm import TissueGraph, build_graph

# 200 cells scattered in a tissue section, each with a feature vector
rng = np.random.default_rng(0)
coords = rng.uniform(0, 1000, size=(200, 2))
features = rng.normal(size=(200, 16))
cell_types = rng.integers(0, 4, size=200)

edge_index = build_graph(coords, method="knn", k=6)
graph = TissueGraph(coords=coords, node_features=features,
                    edge_index=edge_index, cell_types=cell_types)

print(graph)                       # TissueGraph(nodes=200, edges=..., features=16)
print(graph.degree().mean())       # average node degree
```

Explain the tissue as biological interactions:

```python
from cellgraphfm.explain import interaction_matrix, neighborhood_enrichment

counts, type_ids = interaction_matrix(graph)             # who touches whom
zscore, type_ids = neighborhood_enrichment(graph, 100)   # is it more than chance?
```

Build a GNN encoder (requires the `torch` extra):

```python
from cellgraphfm.models import build_gnn, list_architectures

print(list_architectures())  # ['gcn', 'graphsage', 'gat', 'gatv2', 'gin', 'graph_transformer']
model = build_gnn("gatv2", in_channels=16, hidden_channels=64, out_channels=32, num_layers=3)
```

Or explore from the command line:

```bash
cellgraphfm info          # list architectures & supported foundation models
cellgraphfm demo --cells 300 --method delaunay
```

## Research dimensions

CellGraphFM is organized around ten research dimensions. Each maps to code:

1. **Cellular representation** — `features.morphology`, `features.embeddings`
2. **Graph construction** — `graph.construction`
3. **Multi-scale graphs** — planned; `TissueGraph` metadata carries scale levels
4. **Graph neural architectures** — `models.factory`
5. **Self-supervised learning** — `ssl.masking`
6. **Biological reasoning** — downstream heads (roadmap)
7. **Explainability** — `explain.interactions`
8. **Foundation-model integration** — `features.embeddings`
9. **Multimodal learning** — roadmap
10. **Scientific discovery** — `explain` enrichment / motif mining

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for the full plan and status.

## Datasets

CellGraphFM targets public benchmarks and ships **no** patient data. Cell
segmentation: PanNuke, MoNuSeg, CoNSeP, NuCLS. Whole-slide: TCGA, CPTAC,
CAMELYON16, BRACS, BreakHis. Spatial biology: 10x Genomics public spatial
transcriptomics, Human Cell Atlas.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Every PR runs
lint (`ruff`) and the test suite (`pytest`) across Python 3.10–3.12 via GitHub
Actions.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use CellGraphFM in your research, please cite it — see
[`CITATION.cff`](CITATION.cff).
