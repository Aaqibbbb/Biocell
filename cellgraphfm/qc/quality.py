"""Automated image quality control for slide tiles (OptraScan-inspired analog).

A lightweight QC gate: score a tile's sharpness (variance of Laplacian),
contrast (grayscale standard deviation) and brightness (grayscale mean), then
flag blurry / low-quality tiles. Grayscale is computed on a ``0-255`` scale so
the default thresholds are interpretable regardless of the input's numeric
range. Pure ``numpy`` + ``scipy``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

__all__ = ["QCReport", "assess_quality"]


@dataclass
class QCReport:
    """Result of a single-image quality assessment.

    Attributes
    ----------
    sharpness:
        Variance of the grayscale Laplacian (higher is sharper).
    contrast:
        Standard deviation of grayscale intensities.
    brightness:
        Mean grayscale intensity (``0-255``).
    blurry:
        ``True`` when ``sharpness`` is below the sharpness threshold.
    passed:
        ``True`` when the image is not blurry and meets the contrast threshold.
    """

    sharpness: float
    contrast: float
    brightness: float
    blurry: bool
    passed: bool

    def as_dict(self) -> dict[str, float | bool]:
        """Return the report as a plain dictionary."""
        return asdict(self)


def _to_gray_0_255(image: np.ndarray) -> np.ndarray:
    """Return an ``(H, W)`` grayscale image on a ``0-255`` float scale."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        gray = img[..., :3] @ np.array([0.299, 0.587, 0.114])
    elif img.ndim == 2:
        gray = img
    else:
        raise ValueError(
            f"expected a 2-D grayscale or (H, W, 3/4) RGB image, got shape {img.shape}"
        )
    if gray.size and gray.max() <= 1.0:
        gray = gray * 255.0
    return gray


def assess_quality(
    image: np.ndarray,
    *,
    min_sharpness: float = 100.0,
    min_contrast: float = 10.0,
) -> QCReport:
    """Assess the imaging quality of a single tile.

    Parameters
    ----------
    image:
        ``(H, W)`` grayscale or ``(H, W, 3/4)`` RGB(A) image. Byte (0-255) and
        float (0-1) ranges are both accepted.
    min_sharpness:
        Minimum variance-of-Laplacian; below this the tile is flagged ``blurry``.
    min_contrast:
        Minimum grayscale standard deviation required to pass.

    Returns
    -------
    report:
        A :class:`QCReport` with the measured metrics and pass/fail flags.
    """
    gray = _to_gray_0_255(image)
    sharpness = float(ndimage.laplace(gray).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())
    blurry = sharpness < min_sharpness
    passed = (not blurry) and (contrast >= min_contrast)
    return QCReport(
        sharpness=sharpness,
        contrast=contrast,
        brightness=brightness,
        blurry=blurry,
        passed=passed,
    )
