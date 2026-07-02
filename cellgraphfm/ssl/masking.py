"""Self-supervised objectives for cell graphs (Research Dimension 5).

The idea — "GPT for cell graphs" — is to learn universal representations without
labels by masking parts of the graph and reconstructing them. The masking
utilities here are pure ``numpy``; the reconstruction losses use ``torch`` and
are imported lazily so this module works without the deep-learning stack.
"""

from __future__ import annotations

import numpy as np

from cellgraphfm.utils.deps import require_dependency

_STRATEGIES = ("zero", "mean", "noise")


def random_node_mask(
    num_nodes: int,
    mask_rate: float = 0.5,
    *,
    seed: int | None = None,
) -> np.ndarray:
    """Return a boolean ``(N,)`` mask selecting a random subset of nodes.

    ``True`` marks a masked node. At least one node is masked whenever
    ``mask_rate > 0`` and ``num_nodes > 0``.
    """
    if not 0.0 <= mask_rate <= 1.0:
        raise ValueError("mask_rate must be in [0, 1]")
    mask = np.zeros(num_nodes, dtype=bool)
    if num_nodes == 0 or mask_rate == 0.0:
        return mask
    k = max(1, int(round(num_nodes * mask_rate)))
    rng = np.random.default_rng(seed)
    idx = rng.choice(num_nodes, size=k, replace=False)
    mask[idx] = True
    return mask


def mask_features(
    features: np.ndarray,
    mask: np.ndarray,
    *,
    strategy: str = "zero",
    seed: int | None = None,
) -> np.ndarray:
    """Return a copy of ``features`` with masked rows corrupted.

    Strategies:

    * ``"zero"``  — replace masked rows with zeros.
    * ``"mean"``  — replace with the mean of the *unmasked* rows.
    * ``"noise"`` — replace with Gaussian noise matching per-column statistics.
    """
    if strategy not in _STRATEGIES:
        raise ValueError(f"strategy must be one of {_STRATEGIES}")
    features = np.asarray(features, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] != features.shape[0]:
        raise ValueError("mask length must equal the number of rows in features")

    out = features.copy()
    if not mask.any():
        return out

    if strategy == "zero":
        out[mask] = 0.0
    elif strategy == "mean":
        keep = ~mask
        fill = features[keep].mean(axis=0) if keep.any() else np.zeros(features.shape[1])
        out[mask] = fill
    else:  # noise
        rng = np.random.default_rng(seed)
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        out[mask] = rng.normal(
            mean, np.clip(std, 1e-8, None), size=(int(mask.sum()), features.shape[1])
        )
    return out


def sce_loss(prediction, target, alpha: float = 2.0):
    """Scaled cosine error used by GraphMAE.

    ``(1 - cos(prediction, target)) ** alpha`` averaged over rows. ``prediction``
    and ``target`` are ``torch`` tensors of shape ``(N, F)``.
    """
    torch = require_dependency("torch")
    x = torch.nn.functional.normalize(prediction, p=2, dim=-1)
    y = torch.nn.functional.normalize(target, p=2, dim=-1)
    cos = (x * y).sum(dim=-1)
    return (1.0 - cos).pow(alpha).mean()


def masked_reconstruction_loss(reconstructed, target, mask, alpha: float = 2.0):
    """GraphMAE-style loss computed only over masked nodes.

    Parameters
    ----------
    reconstructed, target:
        ``torch`` tensors of shape ``(N, F)``.
    mask:
        Boolean tensor/array of shape ``(N,)`` selecting the masked nodes.
    """
    torch = require_dependency("torch")
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(np.asarray(mask, dtype=bool))
    mask = mask.bool()
    if mask.sum() == 0:
        return reconstructed.sum() * 0.0
    return sce_loss(reconstructed[mask], target[mask], alpha=alpha)
