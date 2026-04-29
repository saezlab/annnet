"""annnet.utils public API."""

from __future__ import annotations

from typing import Any
from importlib import import_module

_lazy_symbols: dict[str, tuple[str, str]] = {
    'to_graphviz': ('annnet.utils.plotting', 'to_graphviz'),
    'to_matplotlib': ('annnet.utils.plotting', 'to_matplotlib'),
    'to_pydot': ('annnet.utils.plotting', 'to_pydot'),
    'plot': ('annnet.utils.plotting', 'plot'),
    'render': ('annnet.utils.plotting', 'render'),
}

__all__ = sorted(_lazy_symbols)


def __getattr__(name: str) -> Any:
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
