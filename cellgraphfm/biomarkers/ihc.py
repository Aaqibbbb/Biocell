"""IHC quantification via color deconvolution (biomarker scoring analog).

Immunohistochemistry stains are separated into their component dyes using
optical-density colour deconvolution with the Ruifrok--Johnston reference stain
vectors. From the DAB (chromogen) channel we derive standard biomarker read-outs
used for Ki-67 / ER / PR / HER2 scoring: a labelling / positivity index and an
H-score. Pure ``numpy`` + ``scipy``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "HED_MATRIX",
    "HE_MATRIX",
    "h_score",
    "positivity_index",
    "separate_stains",
]


def _normalize_rows(matrix: list[list[float]]) -> np.ndarray:
    """Return ``matrix`` with each row scaled to unit L2 norm (``float64``)."""
    arr = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


# Ruifrok & Johnston (2001) reference optical-density vectors, row-normalised.
# Rows are stains, columns are the (R, G, B) absorbances.
HED_MATRIX: np.ndarray = _normalize_rows(
    [
        [0.65, 0.70, 0.29],  # Haematoxylin
        [0.07, 0.99, 0.11],  # Eosin
        [0.27, 0.57, 0.78],  # DAB
    ]
)
HE_MATRIX: np.ndarray = _normalize_rows(
    [
        [0.65, 0.70, 0.29],  # Haematoxylin
        [0.07, 0.99, 0.11],  # Eosin
    ]
)


def separate_stains(rgb: np.ndarray, stain_matrix: np.ndarray = HED_MATRIX) -> np.ndarray:
    """Deconvolve an RGB image into per-stain concentrations.

    Converts the image to optical density ``OD = -log10((I + 1) / 256)`` and
    projects it onto the (pseudo-)inverse of the stain matrix.

    Parameters
    ----------
    rgb:
        ``(H, W, 3)`` RGB image. Byte (0-255) and float (0-1) ranges are both
        accepted; float images in ``[0, 1]`` are rescaled to ``0-255``.
    stain_matrix:
        ``(n_stains, 3)`` matrix of row-normalised reference OD vectors. Defaults
        to :data:`HED_MATRIX`.

    Returns
    -------
    concentrations:
        ``(H, W, n_stains)`` array of per-pixel stain concentrations.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {rgb.shape}")
    stain_matrix = np.asarray(stain_matrix, dtype=np.float64)
    if stain_matrix.ndim != 2 or stain_matrix.shape[1] != 3:
        raise ValueError(f"stain_matrix must have shape (n_stains, 3), got {stain_matrix.shape}")

    if rgb.size and rgb.max() <= 1.0:
        rgb = rgb * 255.0
    optical_density = -np.log10((rgb + 1.0) / 256.0)
    return optical_density @ np.linalg.pinv(stain_matrix)


def _mean_per_nucleus(nuclei_labels: np.ndarray, concentration: np.ndarray) -> np.ndarray:
    """Return the mean ``concentration`` within each labelled nucleus."""
    labels = np.asarray(nuclei_labels)
    conc = np.asarray(concentration, dtype=np.float64)
    if labels.shape != conc.shape:
        raise ValueError(f"nuclei_labels {labels.shape} and concentration {conc.shape} must match")
    ids = np.unique(labels)
    ids = ids[ids != 0]
    if ids.size == 0:
        return np.empty((0,), dtype=np.float64)
    means = ndimage.mean(conc, labels=labels, index=ids)
    return np.atleast_1d(np.asarray(means, dtype=np.float64))


def positivity_index(
    nuclei_labels: np.ndarray,
    dab_concentration: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Fraction of nuclei whose mean DAB concentration exceeds ``threshold``.

    This is the Ki-67 / labelling-index analog: the proportion of positively
    stained nuclei among all detected nuclei.

    Parameters
    ----------
    nuclei_labels:
        ``(H, W)`` integer label image (``0`` background, ``k`` the ``k``-th
        nucleus), as produced by nucleus segmentation.
    dab_concentration:
        ``(H, W)`` DAB concentration image (e.g. from :func:`separate_stains`).
    threshold:
        A nucleus is positive when its mean DAB concentration is ``> threshold``.

    Returns
    -------
    index:
        Fraction of positive nuclei in ``[0, 1]`` (``0.0`` when no nuclei exist).
    """
    means = _mean_per_nucleus(nuclei_labels, dab_concentration)
    if means.size == 0:
        return 0.0
    return float(np.mean(means > threshold))


def h_score(
    nuclei_labels: np.ndarray,
    dab_concentration: np.ndarray,
    *,
    thresholds: tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> float:
    """Compute the IHC H-score in ``[0, 300]``.

    Each nucleus is binned by its mean DAB concentration into negative (``0``),
    weak (``1+``), moderate (``2+``) or strong (``3+``) using the three ascending
    ``thresholds``. The score is ``1 * %1+ + 2 * %2+ + 3 * %3+`` where each
    ``%`` is the percentage of nuclei in that bin.

    Parameters
    ----------
    nuclei_labels:
        ``(H, W)`` integer nucleus label image.
    dab_concentration:
        ``(H, W)`` DAB concentration image.
    thresholds:
        Ascending ``(t1, t2, t3)`` cut-points separating the intensity bins.

    Returns
    -------
    score:
        H-score in ``[0, 300]`` (``0.0`` when no nuclei exist).
    """
    if len(thresholds) != 3:
        raise ValueError("thresholds must be a 3-tuple (t1, t2, t3)")
    t1, t2, t3 = (float(t) for t in thresholds)

    means = _mean_per_nucleus(nuclei_labels, dab_concentration)
    if means.size == 0:
        return 0.0

    bins = np.zeros(means.shape, dtype=np.int64)
    bins[means >= t1] = 1
    bins[means >= t2] = 2
    bins[means >= t3] = 3

    pct_1 = 100.0 * np.mean(bins == 1)
    pct_2 = 100.0 * np.mean(bins == 2)
    pct_3 = 100.0 * np.mean(bins == 3)
    return float(1.0 * pct_1 + 2.0 * pct_2 + 3.0 * pct_3)
