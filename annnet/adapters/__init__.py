"""annnet.adapters: curated adapter API with call-time lazy symbol loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .._optional_components import GRAPH_BACKENDS, available_optional_components

_lazy_functions: dict[str, tuple[str, str]] = {
    'to_nx': ('annnet.adapters.networkx_adapter', 'to_nx'),
    'from_nx': ('annnet.adapters.networkx_adapter', 'from_nx'),
    'to_igraph': ('annnet.adapters.igraph_adapter', 'to_igraph'),
    'from_igraph': ('annnet.adapters.igraph_adapter', 'from_igraph'),
    'to_graphtool': ('annnet.adapters.graphtool_adapter', 'to_graphtool'),
    'from_graphtool': ('annnet.adapters.graphtool_adapter', 'from_graphtool'),
    'to_pyg': ('annnet.adapters.pyg_adapter', 'to_pyg'),
}

__all__ = sorted(set(_lazy_functions) | {'available_backends'})


def available_backends() -> dict[str, bool]:
    """Report which optional notebook-facing adapter backends are installed."""
    return available_optional_components(GRAPH_BACKENDS)


def _make_lazy_function(module_name: str, attr_name: str):
    def _lazy_function(*args, **kwargs):
        module = import_module(module_name)
        return getattr(module, attr_name)(*args, **kwargs)

    _lazy_function.__name__ = attr_name
    _lazy_function.__qualname__ = attr_name
    _lazy_function.__module__ = __name__
    return _lazy_function


def __getattr__(name: str) -> Any:
    if name in _lazy_functions:
        mod, attr = _lazy_functions[name]
        value = _make_lazy_function(mod, attr)
        globals()[name] = value
        return value
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
