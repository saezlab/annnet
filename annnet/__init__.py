"""annnet public package API."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from typing import Any

from ._metadata import (
    __author__,
    __authors__,
    __license__,
    __maintainers__,
    __title__,
    __version__,
    get_latest_version,
    get_metadata,
    info,
)
from ._metadata import metadata as __metadata__

_lazy_submodules = {
    "adapters": "annnet.adapters",
    "io": "annnet.io",
    "core": "annnet.core",
    "algorithms": "annnet.algorithms",
}

_lazy_objects: dict[str, tuple[str, str]] = {
    "AnnNet": ("annnet.core.graph", "AnnNet"),
    "Graph": ("annnet.core.graph", "AnnNet"),
    "EdgeType": ("annnet.core.graph", "EdgeType"),
    "Traversal": ("annnet.algorithms.traversal", "Traversal"),
}

_lazy_functions: dict[str, tuple[str, str]] = {
    "available_backends": ("annnet.adapters", "available_backends"),
    "available_dataframe_backends": ("annnet._dataframe_backend", "available_dataframe_backends"),
    "available_plot_backends": ("annnet._plotting_backend", "available_plot_backends"),
    "get_default_dataframe_backend": (
        "annnet._dataframe_backend",
        "get_default_dataframe_backend",
    ),
    "get_default_plot_backend": ("annnet._plotting_backend", "get_default_plot_backend"),
    "select_dataframe_backend": ("annnet._dataframe_backend", "select_dataframe_backend"),
    "select_plot_backend": ("annnet._plotting_backend", "select_plot_backend"),
    "set_default_dataframe_backend": (
        "annnet._dataframe_backend",
        "set_default_dataframe_backend",
    ),
    "set_default_plot_backend": ("annnet._plotting_backend", "set_default_plot_backend"),
    "to_nx": ("annnet.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("annnet.adapters.networkx_adapter", "from_nx"),
    "to_igraph": ("annnet.adapters.igraph_adapter", "to_igraph"),
    "from_igraph": ("annnet.adapters.igraph_adapter", "from_igraph"),
    "to_graphtool": ("annnet.adapters.graphtool_adapter", "to_graphtool"),
    "from_graphtool": ("annnet.adapters.graphtool_adapter", "from_graphtool"),
    "to_pyg": ("annnet.adapters.pyg_adapter", "to_pyg"),
    "write": ("annnet.io.io_annnet", "write"),
    "read": ("annnet.io.io_annnet", "read"),
    "to_json": ("annnet.io.json_io", "to_json"),
    "from_json": ("annnet.io.json_io", "from_json"),
    "write_ndjson": ("annnet.io.json_io", "write_ndjson"),
    "to_dataframes": ("annnet.io.dataframe_io", "to_dataframes"),
    "from_dataframes": ("annnet.io.dataframe_io", "from_dataframes"),
    "from_csv": ("annnet.io.csv_io", "load_csv_to_graph"),
    "from_dataframe": ("annnet.io.csv_io", "from_dataframe"),
    "edges_to_csv": ("annnet.io.csv_io", "export_edge_list_csv"),
    "hyperedge_to_csv": ("annnet.io.csv_io", "export_hyperedge_csv"),
    "from_excel": ("annnet.io.excel", "load_excel_to_graph"),
    "to_sif": ("annnet.io.SIF_io", "to_sif"),
    "from_sif": ("annnet.io.SIF_io", "from_sif"),
    "to_graphml": ("annnet.io.GraphML_io", "to_graphml"),
    "from_graphml": ("annnet.io.GraphML_io", "from_graphml"),
    "to_gexf": ("annnet.io.GraphML_io", "to_gexf"),
    "from_gexf": ("annnet.io.GraphML_io", "from_gexf"),
    "to_cx2": ("annnet.io.cx2_io", "to_cx2"),
    "from_cx2": ("annnet.io.cx2_io", "from_cx2"),
    "show_cx2": ("annnet.io.cx2_io", "show"),
    "to_parquet": ("annnet.io.Parquet_io", "to_parquet"),
    "from_parquet": ("annnet.io.Parquet_io", "from_parquet"),
    "from_sbml": ("annnet.io.SBML_io", "from_sbml"),
    "from_cobra_model": ("annnet.io.sbml_cobra_io", "from_cobra_model"),
    "from_sbml_cobra": ("annnet.io.sbml_cobra_io", "from_sbml"),
    "read_omnipath": ("annnet.io.read_omnipath", "read_omnipath"),
}

_metadata_exports = {
    "__title__",
    "__version__",
    "__author__",
    "__authors__",
    "__maintainers__",
    "__license__",
    "__metadata__",
    "get_metadata",
    "get_latest_version",
    "info",
}

__all__ = sorted(
    set(_lazy_submodules) | set(_lazy_objects) | set(_lazy_functions) | _metadata_exports
)


def _make_lazy_function(module_name: str, attr_name: str):
    def _lazy_function(*args, **kwargs):
        module = import_module(module_name)
        return getattr(module, attr_name)(*args, **kwargs)

    _lazy_function.__name__ = attr_name
    _lazy_function.__qualname__ = attr_name
    _lazy_function.__module__ = __name__
    return _lazy_function


def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        value = import_module(_lazy_submodules[name])
    elif name in _lazy_objects:
        mod, attr = _lazy_objects[name]
        value = getattr(import_module(mod), attr)
    elif name in _lazy_functions:
        mod, attr = _lazy_functions[name]
        value = _make_lazy_function(mod, attr)
    else:
        raise AttributeError(name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
