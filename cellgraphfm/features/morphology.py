"""Morphology feature extraction from segmentation masks (Research Dimension 1).

Given a labelled instance mask (``0`` = background, ``k`` = the ``k``-th cell),
:func:`extract_morphology` computes interpretable per-cell shape descriptors
using only ``numpy`` and ``scipy``. These make natural node features for a
cellular graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

FEATURE_NAMES: tuple[str, ...] = (
    "area",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "orientation",
    "extent",
    "mean_intensity",
)


@dataclass
class MorphologyFeatures:
    """Per-cell morphology descriptors extracted from a mask.

    Attributes
    ----------
    labels:
        ``(N,)`` array of the source instance labels.
    centroids:
        ``(N, 2)`` array of ``(row, col)`` centroid coordinates.
    features:
        ``(N, F)`` array of features, columns given by :attr:`feature_names`.
    feature_names:
        Names of the ``F`` feature columns.
    """

    labels: np.ndarray
    centroids: np.ndarray
    features: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES

    @property
    def num_cells(self) -> int:
        return int(self.labels.shape[0])

    def __repr__(self) -> str:
        return f"MorphologyFeatures(cells={self.num_cells}, features={len(self.feature_names)})"


def _region_descriptors(
    sub_mask: np.ndarray,
) -> tuple[float, float, float, float, float, float, float]:
    """Compute (cy, cx, major, minor, ecc, orientation, extent) for one region."""
    ys, xs = np.nonzero(sub_mask)
    area = float(ys.size)
    cy = float(ys.mean())
    cx = float(xs.mean())

    dy = ys - cy
    dx = xs - cx
    mu20 = float(np.mean(dx * dx))
    mu02 = float(np.mean(dy * dy))
    mu11 = float(np.mean(dx * dy))

    cov = np.array([[mu20, mu11], [mu11, mu02]], dtype=np.float64)
    eigvals = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
    l2, l1 = float(eigvals[0]), float(eigvals[1])  # eigvalsh returns ascending

    major = 4.0 * np.sqrt(l1)
    minor = 4.0 * np.sqrt(l2)
    eccentricity = float(np.sqrt(1.0 - l2 / l1)) if l1 > 0 else 0.0
    orientation = 0.5 * float(np.arctan2(2.0 * mu11, mu20 - mu02))

    bbox_area = float(sub_mask.shape[0] * sub_mask.shape[1])
    extent = area / bbox_area if bbox_area > 0 else 0.0
    return cy, cx, major, minor, eccentricity, orientation, extent


def extract_morphology(
    mask: np.ndarray,
    intensity: np.ndarray | None = None,
) -> MorphologyFeatures:
    """Extract morphology features from a labelled instance mask.

    Parameters
    ----------
    mask:
        ``(H, W)`` integer array. ``0`` is background; each positive integer is a
        distinct cell instance.
    intensity:
        Optional ``(H, W)`` intensity image (e.g. haematoxylin channel) used to
        compute ``mean_intensity``. When omitted that column is ``0``.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    if intensity is not None:
        intensity = np.asarray(intensity, dtype=np.float64)
        if intensity.shape != mask.shape:
            raise ValueError("intensity must have the same shape as mask")

    labels = np.unique(mask)
    labels = labels[labels != 0]
    n = labels.size

    centroids = np.zeros((n, 2), dtype=np.float64)
    features = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float64)
    if n == 0:
        return MorphologyFeatures(labels=labels, centroids=centroids, features=features)

    objects = ndimage.find_objects(mask.astype(np.intp))
    for row, label in enumerate(labels):
        slc = objects[int(label) - 1]
        if slc is None:  # pragma: no cover - only if labels are non-contiguous holes
            continue
        sub = mask[slc] == label
        cy, cx, major, minor, ecc, orient, extent = _region_descriptors(sub)
        offset_y, offset_x = slc[0].start, slc[1].start
        centroids[row] = (cy + offset_y, cx + offset_x)

        mean_int = 0.0
        if intensity is not None:
            mean_int = float(intensity[slc][sub].mean())

        features[row] = (
            sub.sum(),
            major,
            minor,
            ecc,
            orient,
            extent,
            mean_int,
        )

    return MorphologyFeatures(labels=labels, centroids=centroids, features=features)
