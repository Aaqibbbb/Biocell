# CellGraphFM Roadmap

CellGraphFM is organised around ten research dimensions. This document tracks
what exists today and what is planned. ✅ implemented · 🚧 partial · 🔭 planned.

## Dimension 1 — Cellular representation learning
- ✅ Morphology features from segmentation masks (`features.morphology`).
- ✅ Embedding interface + foundation-model registry (`features.embeddings`).
- 🔭 Direct loaders for UNI/Virchow/GigaPath weights (kept optional, out-of-tree).
- 🔭 Learnable cell tokenizer combining morphology + FM embeddings.

## Dimension 2 — Graph construction
- ✅ KNN, radius, Delaunay, and contact graphs (`graph.construction`).
- 🔭 Ligand–receptor / communication-prior edges.
- 🔭 Adaptive / learned graph construction.

## Dimension 3 — Multi-scale graphs
- 🚧 `TissueGraph.metadata` can carry a scale level; hierarchy helpers pending.
- 🔭 Cell → microenvironment → region → tissue → slide graph-of-graphs, connecting
  levels (extending the HACT-Net idea).

## Dimension 4 — Graph neural architectures
- ✅ GCN, GraphSAGE, GAT, GATv2, GIN, Graph Transformer via one factory
  (`models.factory` / `models.gnn`).
- 🔭 GraphGPS, heterogeneous and hierarchical GNNs.

## Dimension 5 — Self-supervised learning
- ✅ Node/feature masking + GraphMAE-style scaled-cosine reconstruction loss
  (`ssl.masking`).
- 🔭 Contrastive (GraphCL), DINO-like distillation, neighbour prediction.

## Dimension 6 — Biological reasoning
- 🔭 Downstream heads: subtype, survival, mutation, immune phenotype,
  drug-response prediction.

## Dimension 7 — Explainability
- ✅ Cell-type interaction matrix, permutation-based neighbourhood enrichment,
  and interaction-path mining (`explain.interactions`).
- 🔭 Attention-attribution and subgraph explanations tied to predictions.

## Dimension 8 — Foundation-model integration
- ✅ Registry + embedder interface (`features.embeddings`).
- 🔭 Cached tile-embedding extraction pipeline over whole-slide images.

## Dimension 9 — Multimodal learning
- 🔭 Joint embedding of image graph + spatial transcriptomics + clinical text.

## Dimension 10 — Scientific discovery
- 🚧 Enrichment / motif mining surfaces candidate interactions today.
- 🔭 Biomarker and novel-microenvironment discovery workflows.

## Engineering
- ✅ Dependency-light core (numpy + scipy); torch behind an optional extra.
- ✅ CI: lint + tests across Python 3.10–3.12, plus a torch/PyG job.
- 🔭 Docs site, benchmark harness on public datasets (PanNuke, TCGA, CAMELYON16).
