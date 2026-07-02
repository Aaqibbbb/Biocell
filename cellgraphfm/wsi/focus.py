"""Focus measurement and extended-depth-of-field stacking (Composite Imaging).

An analog of OptraScan's "composite imaging" / extended-depth-of-field feature.
:func:`focus_measure` scores how sharp a single grayscale image is (variance of
its Laplacian), and :func:`focus_stack` fuses a z-stack into one all-in-focus
composite by, per pixel, selecting the z-slice with the highest local Laplacian
energy. Pure ``numpy`` + ``scipy``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

__all__ = ["focus_measure", "focus_stack"]


def focus_measure(gray: np.ndarray) -> float:
    """Return the variance-of-Laplacian focus score of a grayscale image.

    Higher values indicate a sharper (better focused) image; a blurred image has
    a low-variance Laplacian.

    Parameters
    ----------
    gray:
        ``(H, W)`` grayscale image.
    """
    gray = np.asarray(gray, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError(f"gray must be a 2-D image, got shape {gray.shape}")
    laplacian = ndimage.laplace(gray)
    return float(laplacian.var())


def focus_stack(zstack: np.ndarray, *, window: int = 9) -> np.ndarray:
    """Fuse a focal z-stack into an all-in-focus composite.

    For every pixel the z-slice with the greatest local Laplacian energy (a
    ``window x window`` mean of ``|laplacian|``) is selected, yielding a single
    sharp image assembled from the best-focused parts of each slice.

    Parameters
    ----------
    zstack:
        ``(Z, H, W)`` grayscale or ``(Z, H, W, 3)`` colour focal stack.
    window:
        Side length of the local averaging window used to pool Laplacian energy.

    Returns
    -------
    composite:
        ``(H, W)`` or ``(H, W, 3)`` all-in-focus image with the input dtype.
    """
    original = np.asarray(zstack)
    if original.ndim not in (3, 4):
        raise ValueError(f"zstack must be (Z, H, W) or (Z, H, W, 3), got shape {original.shape}")
    if original.shape[0] == 0:
        raise ValueError("zstack must contain at least one z-slice")
    if window < 1:
        raise ValueError("window must be >= 1")

    work = original.astype(np.float64)
    num_z = original.shape[0]
    is_color = original.ndim == 4

    energies = np.empty(original.shape[:3], dtype=np.float64)
    for z in range(num_z):
        gray = work[z].mean(axis=-1) if is_color else work[z]
        laplacian = ndimage.laplace(gray)
        energies[z] = ndimage.uniform_filter(np.abs(laplacian), size=window)

    best = np.argmax(energies, axis=0)  # (H, W)
    if is_color:
        index = best[None, :, :, None]
    else:
        index = best[None, :, :]
    composite = np.take_along_axis(original, index, axis=0)[0]
    return composite
