"""annnet.adapters: curated adapter API with call-time lazy symbol loading."""

from __future__ import annotations

from typing import Any

from .._support.lazy_exports import export_dir, resolve_lazy_export
from .._support.optional_components import GRAPH_BACKENDS, available_optional_components

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


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(
        globals(),
        name,
        functions=_lazy_functions,
        package_name=__name__,
    )


def __dir__() -> list[str]:
    return export_dir(globals(), __all__)
