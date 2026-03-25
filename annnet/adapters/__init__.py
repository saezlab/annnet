"""annnet.adapters: curated adapter API with lazy symbol loading."""

from __future__ import annotations

from importlib import import_module, util
from typing import Any

_public_symbols: dict[str, tuple[str, str]] = {
    "available_backends": ("annnet.adapters", "available_backends"),
    "to_nx": ("annnet.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("annnet.adapters.networkx_adapter", "from_nx"),
    "to_igraph": ("annnet.adapters.igraph_adapter", "to_igraph"),
    "from_igraph": ("annnet.adapters.igraph_adapter", "from_igraph"),
    "to_graphtool": ("annnet.adapters.graphtool_adapter", "to_graphtool"),
    "from_graphtool": ("annnet.adapters.graphtool_adapter", "from_graphtool"),
    "to_pyg": ("annnet.adapters.pyg_adapter", "to_pyg"),
}

# Internal loader registry for legacy/class-based adapter access.
_adapter_classes: dict[str, tuple[str, str, str]] = {
    "networkx": ("networkx", ".networkx_adapter", "NetworkXAdapter"),
    "igraph": ("igraph", ".igraph_adapter", "IGraphAdapter"),
}

_backend_modules: dict[str, str] = {
    "networkx": "networkx",
    "igraph": "igraph",
    "graphtool": "graph_tool",
    "pyg": "torch_geometric",
}

__all__ = sorted(_public_symbols)


def _is_installed(modname: str) -> bool:
    return util.find_spec(modname) is not None


def available_backends() -> dict[str, bool]:
    """Report which optional notebook-facing adapter backends are installed."""
    return {name: _is_installed(modname) for name, modname in _backend_modules.items()}


def load_adapter(name: str, *args, **kwargs):
    """Internal legacy adapter loader kept for compatibility."""
    if name not in _adapter_classes:
        raise ValueError(f"Unknown adapter '{name}'")
    modname, submod, cls = _adapter_classes[name]
    if not _is_installed(modname):
        raise ModuleNotFoundError(
            f"Optional backend '{name}' is not installed. "
            f"Install with `pip install annnet[{name}]`."
        )
    mod = import_module(__name__ + submod)
    return getattr(mod, cls)(*args, **kwargs)


def __getattr__(name: str) -> Any:
    if name in _public_symbols:
        mod, attr = _public_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
