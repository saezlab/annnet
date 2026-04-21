"""annnet.algorithms public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_lazy_symbols: dict[str, tuple[str, str]] = {
    'Traversal': ('annnet.algorithms.traversal', 'Traversal'),
}

__all__ = sorted(_lazy_symbols)


def __getattr__(name: str) -> Any:
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
