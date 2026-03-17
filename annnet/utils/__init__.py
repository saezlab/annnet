"""annnet.utils public API."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_lazy_submodules = {
    "plotting": "annnet.utils.plotting",
    "config": "annnet.utils.config",
    "typing": "annnet.utils.typing",
    "validation": "annnet.utils.validation",
}

_lazy_symbols: dict[str, tuple[str, str]] = {
    "Edge": ("annnet.utils.typing", "Edge"),
    "Attr": ("annnet.utils.typing", "Attr"),
    "Attributes": ("annnet.utils.typing", "Attributes"),
    "canonicalize": ("annnet.utils.validation", "canonicalize"),
    "obj_canonicalized_hash": ("annnet.utils.validation", "obj_canonicalized_hash"),
    "unique_iter": ("annnet.utils.validation", "unique_iter"),
    "build_vertex_labels": ("annnet.utils.plotting", "build_vertex_labels"),
    "build_edge_labels": ("annnet.utils.plotting", "build_edge_labels"),
    "edge_style_from_weights": ("annnet.utils.plotting", "edge_style_from_weights"),
    "to_graphviz": ("annnet.utils.plotting", "to_graphviz"),
    "to_pydot": ("annnet.utils.plotting", "to_pydot"),
    "plot": ("annnet.utils.plotting", "plot"),
    "render": ("annnet.utils.plotting", "render"),
}

__all__ = sorted(set(_lazy_submodules) | set(_lazy_symbols))


def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        return import_module(_lazy_submodules[name])
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
