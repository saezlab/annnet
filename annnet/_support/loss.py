"""The warning every reader and writer uses to say what a format could not hold."""

from __future__ import annotations

__all__ = ['AnnNetLossWarning']


class AnnNetLossWarning(UserWarning):
    """A format could not hold part of a graph."""
