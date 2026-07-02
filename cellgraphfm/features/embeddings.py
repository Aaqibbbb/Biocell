"""Cell embedding interfaces and foundation-model integration (Dimensions 1 & 8).

Pathology foundation models (UNI, Virchow, Prov-GigaPath, CONCH, ...) are strong
patch encoders whose features transfer far better than ImageNet features and are
increasingly used as *node embeddings* for cellular graphs. This module defines a
small, stable interface for turning per-cell image patches into node features,
plus a registry describing the supported foundation models.

Model weights are **not** bundled (they have their own licenses and are large).
:class:`RandomProjectionEmbedder` provides a deterministic, dependency-light
stand-in so the rest of the pipeline can be developed and tested end-to-end.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FoundationModelSpec:
    """Reference metadata for a pathology foundation model."""

    name: str
    embed_dim: int
    architecture: str
    reference: str
    note: str = ""


# Typical published patch/tile embedding dimensions. These describe the interface
# (output width) — they are not the weights.
FOUNDATION_MODELS: dict[str, FoundationModelSpec] = {
    "uni": FoundationModelSpec(
        "UNI",
        1024,
        "ViT-L/16 (DINOv2)",
        "Chen et al., Nat. Med. 2024",
    ),
    "uni2": FoundationModelSpec(
        "UNI2-h",
        1536,
        "ViT-H/14 (DINOv2)",
        "MahmoodLab, 2024",
    ),
    "virchow": FoundationModelSpec(
        "Virchow",
        1280,
        "ViT-H/14 (DINOv2)",
        "Vorontsov et al., 2024",
        "CLS token; 2560 with mean-pool concat",
    ),
    "virchow2": FoundationModelSpec(
        "Virchow2",
        1280,
        "ViT-H/14 (DINOv2)",
        "Zimmermann et al., 2024",
    ),
    "gigapath": FoundationModelSpec(
        "Prov-GigaPath",
        1536,
        "ViT-G/14 tile encoder",
        "Xu et al., Nature 2024",
    ),
    "conch": FoundationModelSpec(
        "CONCH",
        512,
        "ViT-B/16 vision-language",
        "Lu et al., Nat. Med. 2024",
    ),
    "biomedclip": FoundationModelSpec(
        "BiomedCLIP",
        512,
        "ViT-B/16 vision-language",
        "Zhang et al., 2023",
    ),
    "ctranspath": FoundationModelSpec(
        "CTransPath",
        768,
        "Swin Transformer (SRCL)",
        "Wang et al., MedIA 2022",
    ),
    "dinov3": FoundationModelSpec(
        "DINOv3",
        1024,
        "ViT (self-supervised)",
        "Meta AI, 2025",
    ),
}


def list_foundation_models() -> list[str]:
    """Return the registry keys of the supported foundation models."""
    return sorted(FOUNDATION_MODELS)


def describe_foundation_model(name: str) -> FoundationModelSpec:
    """Return the :class:`FoundationModelSpec` for ``name`` (case-insensitive)."""
    key = name.lower()
    if key not in FOUNDATION_MODELS:
        raise KeyError(f"unknown foundation model '{name}'. Known: {list_foundation_models()}")
    return FOUNDATION_MODELS[key]


class CellEmbedder(ABC):
    """Abstract interface: image patches -> ``(N, embed_dim)`` node features."""

    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Dimensionality of the produced embeddings."""

    @abstractmethod
    def embed(self, patches: np.ndarray) -> np.ndarray:
        """Embed a batch of patches of shape ``(N, ...)`` into ``(N, embed_dim)``."""

    def __call__(self, patches: np.ndarray) -> np.ndarray:
        return self.embed(patches)


class IdentityEmbedder(CellEmbedder):
    """Flatten each patch into a feature vector (useful for precomputed features)."""

    def __init__(self, embed_dim: int) -> None:
        self._embed_dim = int(embed_dim)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, patches: np.ndarray) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[1] != self._embed_dim:
            raise ValueError(f"expected flattened dim {self._embed_dim}, got {flat.shape[1]}")
        return flat


class RandomProjectionEmbedder(CellEmbedder):
    """Deterministic random projection standing in for a frozen FM encoder.

    Flattens each patch and applies a fixed (seeded) Gaussian random projection
    to ``embed_dim``, optionally L2-normalising the output the way real
    foundation-model features usually are. This is a *stand-in* for wiring up the
    pipeline — not a substitute for real pretrained weights.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 1024,
        *,
        seed: int = 0,
        normalize: bool = True,
    ) -> None:
        self.input_dim = int(input_dim)
        self._embed_dim = int(embed_dim)
        self.normalize = normalize
        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(self._embed_dim)
        self._projection = rng.normal(scale=scale, size=(self.input_dim, self._embed_dim)).astype(
            np.float32
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def embed(self, patches: np.ndarray) -> np.ndarray:
        arr = np.asarray(patches, dtype=np.float32)
        flat = arr.reshape(arr.shape[0], -1)
        if flat.shape[1] != self.input_dim:
            raise ValueError(f"expected flattened input dim {self.input_dim}, got {flat.shape[1]}")
        out = flat @ self._projection
        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.clip(norms, 1e-8, None)
        return out


def get_embedder(name: str, **kwargs) -> CellEmbedder:
    """Factory for the built-in, dependency-light embedders.

    ``"identity"`` -> :class:`IdentityEmbedder`,
    ``"random"`` -> :class:`RandomProjectionEmbedder`.

    Real foundation-model encoders are wired in by subclassing
    :class:`CellEmbedder` around the loaded model; see the registry via
    :func:`list_foundation_models`.
    """
    key = name.lower()
    if key == "identity":
        return IdentityEmbedder(**kwargs)
    if key == "random":
        return RandomProjectionEmbedder(**kwargs)
    if key in FOUNDATION_MODELS:
        raise NotImplementedError(
            f"'{name}' is a registered foundation model but its weights are not "
            "bundled. Load the model yourself and wrap it in a CellEmbedder "
            "subclass. See describe_foundation_model() for its embed_dim."
        )
    raise ValueError(f"unknown embedder '{name}'. Use 'identity' or 'random'.")
