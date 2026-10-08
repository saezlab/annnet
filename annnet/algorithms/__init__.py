"""annnet.algorithms public API."""

from __future__ import annotations

from typing import Any
from importlib import import_module

_lazy_symbols: dict[str, tuple[str, str]] = {
    'directed_pairs': ('annnet.algorithms.structure', 'directed_pairs'),
    'sources': ('annnet.algorithms.structure', 'sources'),
    'targets': ('annnet.algorithms.structure', 'targets'),
    'directed_cycle': ('annnet.algorithms.structure', 'directed_cycle'),
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
