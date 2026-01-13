# annnet/__init__.py
"""annnet: single import, full API."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

# Lazily exposed submodules (imported on first attribute access)
_lazy_submodules = {
    # namespaces
    "adapters": "annnet.adapters",
    "io": "annnet.io",
    "core": "annnet.core",
    "algorithms": "annnet.algorithms",
    "utils": "annnet.utils",
    # adapter modules (direct convenience)
    "networkx": "annnet.adapters.networkx_adapter",
    "igraph": "annnet.adapters.igraph_adapter",
    "graphtool": "annnet.adapters.graphtool_adapter",
    "PyGeometric": "annnet.adapters.pyg_adapter",
    # io modules
    "csvio": "annnet.io.csv",
    "excel": "annnet.io.excel",
    "annnet": "annnet.io.io_annnet",
    "json": "annnet.io.json_io",
    "dataframe": "annnet.io.dataframe_io",
    "graphml": "annnet.io.GraphML_io",
    "sif": "annnet.io.SIF_io",
    "sbml": "annnet.io.sbml_io",
    "parquet": "annnet.io.GraphDir_Parquet_io",
}

# Curated top-level symbols (lazy). name -> (module, attribute)
_lazy_symbols: dict[str, tuple[str, str]] = {
    # Core
    "AnnNet": ("annnet.core.graph", "AnnNet"),
    # Stdlib JSON I/O
    "to_json": ("annnet.adapters.json_adapter", "to_json"),
    "from_json": ("annnet.adapters.json_adapter", "from_json"),
    # NetworkX adapter
    "to_nx": ("annnet.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("annnet.adapters.networkx_adapter", "from_nx"),
    "from_nx_only": ("annnet.adapters.networkx_adapter", "from_nx_only"),
    # iGraph adapter
    "to_igraph": ("annnet.adapters.igraph_adapter", "to_igraph"),
    "from_igraph": ("annnet.adapters.igraph_adapter", "from_igraph"),
    "from_igraph_only": ("annnet.adapters.igraph_adapter", "from_igraph_only"),
    # graph-tool adapter
    "to_graphtool": ("annnet.adapters.graphtool_adapter", "to_graphtool"),
    "from_graphtool": ("annnet.adapters.graphtool_adapter", "from_graphtool"),
    # Pytorch Geometric adapter
    "to_pyg": ("annnet.adapters.pyg_adapter", "to_pyg"),
    # GraphML
    "to_graphml": ("annnet.io.GraphML_io", "to_graphml"),
    "from_graphml": ("annnet.io.GraphML_io", "from_graphml"),
    # SIF
    "to_sif": ("annnet.io.SIF_io", "to_sif"),
    "from_sif": ("annnet.io.SIF_io", "from_sif"),
    # SBML (common direction)
    "from_sbml": ("annnet.io.sbml_io", "from_sbml"),
    # Parquet GraphDir
    "write_parquet_graphdir": (
        "annnet.io.GraphDir_Parquet_io",
        "write_parquet_graphdir",
    ),
    "read_parquet_graphdir": (
        "annnet.adaptioers.GraphDir_Parquet_io",
        "read_parquet_graphdir",
    ),
    # CX2
    "to_cx2" : ("annnet.io.cx2_io", "to_cx2"),
    "from_cx2": ("annnet.io.cx2_io", "from_cx2"),
    # JSON
    "to_json" : ("annnet.io.json_io", "to_json"),
    "from_json": ("annnet.io.json_io", "from_json"),
}

__all__ = sorted(set(list(_lazy_submodules) + list(_lazy_symbols)))


def __getattr__(name: str) -> Any:  # PEP 562: lazy attribute resolution
    if name in _lazy_submodules:
        return import_module(_lazy_submodules[name])
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))


try:
    from ._version import __version__  # type: ignore
except Exception:
    try:
        __version__ = _pkg_version("annnet")
    except PackageNotFoundError:
        __version__ = "0.0.0"
