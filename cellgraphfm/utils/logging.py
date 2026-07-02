"""Minimal, dependency-free logging helpers."""

from __future__ import annotations

import logging

_DEFAULT_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


def get_logger(name: str = "cellgraphfm", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger.

    A stream handler with a concise format is attached once, so repeated calls
    do not add duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger
