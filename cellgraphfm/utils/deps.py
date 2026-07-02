"""Helpers for optional dependencies.

The CellGraphFM core depends only on ``numpy`` and ``scipy``. Heavier libraries
(``torch``, ``torch_geometric``, ``networkx``, ...) are optional and imported
lazily. These helpers give a consistent, actionable error when an optional
dependency is missing.
"""

from __future__ import annotations

import importlib
import importlib.util
from types import ModuleType

# Maps an importable module name to the pip extra that provides it.
_EXTRA_FOR_MODULE = {
    "torch": "torch",
    "torch_geometric": "torch",
    "networkx": "viz",
    "plotly": "viz",
    "matplotlib": "viz",
}


def has_dependency(module: str) -> bool:
    """Return ``True`` if ``module`` can be imported."""
    return importlib.util.find_spec(module) is not None


def require_dependency(module: str) -> ModuleType:
    """Import and return ``module`` or raise a helpful :class:`ImportError`.

    Parameters
    ----------
    module:
        The importable module name, e.g. ``"torch"``.
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised via error path test
        extra = _EXTRA_FOR_MODULE.get(module.split(".")[0])
        hint = f'pip install "cellgraphfm[{extra}]"' if extra else f"pip install {module}"
        raise ImportError(
            f"CellGraphFM needs the optional dependency '{module}' for this "
            f"feature, but it is not installed. Install it with: {hint}"
        ) from exc
