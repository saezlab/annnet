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
    "utils": "annnet.utils",
    "networkx": "annnet.adapters.networkx_adapter",
    "igraph": "annnet.adapters.igraph_adapter",
    "graphtool": "annnet.adapters.graphtool_adapter",
    "PyGeometric": "annnet.adapters.pyg_adapter",
    "csvio": "annnet.io.csv_io",
    "excel": "annnet.io.excel",
    "annnet": "annnet.io.io_annnet",
    "json": "annnet.io.json_io",
    "dataframe": "annnet.io.dataframe_io",
    "graphml": "annnet.io.GraphML_io",
    "sif": "annnet.io.SIF_io",
    "sbml": "annnet.io.SBML_io",
    "parquet": "annnet.io.Parquet_io",
}

_lazy_objects: dict[str, tuple[str, str]] = {
    "AnnNet": ("annnet.core.graph", "AnnNet"),
    "EdgeType": ("annnet.core.graph", "EdgeType"),
    "Traversal": ("annnet.algorithms.traversal", "Traversal"),
    "Edge": ("annnet.utils.typing", "Edge"),
    "Attr": ("annnet.utils.typing", "Attr"),
    "Attributes": ("annnet.utils.typing", "Attributes"),
}

_lazy_functions: dict[str, tuple[str, str]] = {
    "available_backends": ("annnet.adapters", "available_backends"),
    "load_adapter": ("annnet.adapters", "load_adapter"),
    "to_nx": ("annnet.adapters.networkx_adapter", "to_nx"),
    "from_nx": ("annnet.adapters.networkx_adapter", "from_nx"),
    "from_nx_only": ("annnet.adapters.networkx_adapter", "from_nx_only"),
    "to_igraph": ("annnet.adapters.igraph_adapter", "to_igraph"),
    "from_igraph": ("annnet.adapters.igraph_adapter", "from_igraph"),
    "from_ig_only": ("annnet.adapters.igraph_adapter", "from_ig_only"),
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
    "load_csv_to_graph": ("annnet.io.csv_io", "load_csv_to_graph"),
    "from_dataframe": ("annnet.io.csv_io", "from_dataframe"),
    "export_edge_list_csv": ("annnet.io.csv_io", "export_edge_list_csv"),
    "export_hyperedge_csv": ("annnet.io.csv_io", "export_hyperedge_csv"),
    "load_excel_to_graph": ("annnet.io.excel", "load_excel_to_graph"),
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
