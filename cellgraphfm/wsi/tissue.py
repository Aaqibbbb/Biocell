"""Tissue detection on whole-slide thumbnails (OptraScan-inspired analog).

A cheap, dependency-light way to find where tissue sits on an otherwise mostly
empty slide: convert the RGB thumbnail to HSV, and threshold the saturation
channel. Glass/background is near-white and therefore weakly saturated, while
stained tissue is strongly saturated. When no threshold is supplied an Otsu
threshold is computed from a 256-bin histogram. All functions are pure
``numpy`` + ``scipy``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = [
    "detect_tissue",
    "tissue_fraction",
    "tissue_regions",
]


def _to_unit_range(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as ``float64`` in ``[0, 1]``.

    Integer/byte images (values ``> 1``) are divided by ``255``; already
    normalised float images are passed through and clipped.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.size and img.max() > 1.0:
        img = img / 255.0
    return np.clip(img, 0.0, 1.0)


def _otsu_threshold(values: np.ndarray) -> int:
    """Otsu threshold of ``values`` (``uint8``/0-255) from a 256-bin histogram.

    Returns the integer bin ``t`` maximising the between-class variance; pixels
    with value ``> t`` are foreground.
    """
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0
    prob = hist / total
    omega = np.cumsum(prob)
    levels = np.arange(256, dtype=np.float64)
    mu = np.cumsum(prob * levels)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    numer = (mu_total * omega - mu) ** 2
    sigma_b = np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)
    return int(np.argmax(sigma_b))


def _tissue_channel(image: np.ndarray) -> np.ndarray:
    """Return an ``(H, W)`` ``[0, 1]`` channel that is high on tissue.

    For RGB(A) input this is the HSV saturation channel. For a 2-D grayscale
    image it is ``1 - intensity`` (tissue is darker than bright background).
    """
    img = np.asarray(image)
    if img.ndim == 2:
        return 1.0 - _to_unit_range(img)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        rgb = _to_unit_range(img[..., :3])
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        channel = np.zeros_like(maxc)
        nonzero = maxc > 0
        channel[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
        return channel
    raise ValueError(f"expected a 2-D grayscale or (H, W, 3/4) RGB image, got shape {img.shape}")


def detect_tissue(
    thumbnail_rgb: np.ndarray,
    *,
    sat_threshold: float | None = None,
) -> np.ndarray:
    """Segment tissue from a slide thumbnail.

    Parameters
    ----------
    thumbnail_rgb:
        ``(H, W, 3)`` / ``(H, W, 4)`` RGB(A) image or a 2-D grayscale image.
        Byte (0-255) and float (0-1) ranges are both accepted.
    sat_threshold:
        Threshold in ``[0, 1]`` applied to the saturation (or grayscale-darkness)
        channel. When ``None`` an Otsu threshold is computed automatically.

    Returns
    -------
    mask:
        ``(H, W)`` boolean array, ``True`` where tissue is present.
    """
    channel = _tissue_channel(thumbnail_rgb)
    if sat_threshold is None:
        # Threshold in integer byte space so the Otsu cut is exact (comparing the
        # float channel against ``t / 255`` is ambiguous at the boundary).
        as_bytes = (channel * 255.0).astype(np.uint8)
        return as_bytes > _otsu_threshold(as_bytes)
    return channel > float(sat_threshold)


def tissue_fraction(mask: np.ndarray) -> float:
    """Return the fraction of pixels flagged as tissue.

    Parameters
    ----------
    mask:
        ``(H, W)`` boolean (or 0/1) array.
    """
    mask = np.asarray(mask)
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask) / mask.size)


def tissue_regions(mask: np.ndarray, *, min_area: int = 1) -> list[tuple[int, int, int, int]]:
    """Return bounding boxes of connected tissue components.

    Parameters
    ----------
    mask:
        ``(H, W)`` boolean array (e.g. from :func:`detect_tissue`).
    min_area:
        Minimum number of tissue pixels a component must contain to be kept.

    Returns
    -------
    boxes:
        List of ``(row0, col0, row1, col1)`` bounding boxes with exclusive
        ``row1``/``col1``, ordered by connected-component label.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    labeled, n = ndimage.label(mask)
    if n == 0:
        return []
    slices = ndimage.find_objects(labeled)
    boxes: list[tuple[int, int, int, int]] = []
    for label, slc in enumerate(slices, start=1):
        if slc is None:  # pragma: no cover - find_objects has an entry per label
            continue
        area = int(np.count_nonzero(labeled[slc] == label))
        if area < min_area:
            continue
        row_slice, col_slice = slc
        boxes.append((row_slice.start, col_slice.start, row_slice.stop, col_slice.stop))
    return boxes
