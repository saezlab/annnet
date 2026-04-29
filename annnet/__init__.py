"""annnet public package API."""

from __future__ import annotations

from typing import Any
from importlib.metadata import (
    PackageNotFoundError,
    version as _pkg_version,
)

from ._support.metadata import (
    info,
    metadata as __metadata__,
    __title__,
    __author__,
    __authors__,
    __license__,
    __version__,
    get_metadata,
    __maintainers__,
    get_latest_version,
)
from ._support.lazy_exports import load_attr, export_dir, make_lazy_function

_lazy_submodules = {
    'adapters': 'annnet.adapters',
    'io': 'annnet.io',
    'core': 'annnet.core',
    'algorithms': 'annnet.algorithms',
}

_lazy_objects: dict[str, tuple[str, str]] = {
    'AnnNet': ('annnet.core.graph', 'AnnNet'),
    'Graph': ('annnet.core.graph', 'AnnNet'),
    'EdgeType': ('annnet.core.graph', 'EdgeType'),
    'Traversal': ('annnet.algorithms.traversal', 'Traversal'),
}

_lazy_functions: dict[str, tuple[str, str]] = {
    'available_backends': ('annnet.adapters', 'available_backends'),
    'available_dataframe_backends': (
        'annnet._support.dataframe_backend',
        'available_dataframe_backends',
    ),
    'available_plot_backends': ('annnet._support.plotting_backend', 'available_plot_backends'),
    'get_default_dataframe_backend': (
        'annnet._support.dataframe_backend',
        'get_default_dataframe_backend',
    ),
    'get_default_plot_backend': ('annnet._support.plotting_backend', 'get_default_plot_backend'),
    'select_dataframe_backend': ('annnet._support.dataframe_backend', 'select_dataframe_backend'),
    'select_plot_backend': ('annnet._support.plotting_backend', 'select_plot_backend'),
    'set_default_dataframe_backend': (
        'annnet._support.dataframe_backend',
        'set_default_dataframe_backend',
    ),
    'set_default_plot_backend': ('annnet._support.plotting_backend', 'set_default_plot_backend'),
    'to_nx': ('annnet.adapters.networkx_adapter', 'to_nx'),
    'from_nx': ('annnet.adapters.networkx_adapter', 'from_nx'),
    'to_igraph': ('annnet.adapters.igraph_adapter', 'to_igraph'),
    'from_igraph': ('annnet.adapters.igraph_adapter', 'from_igraph'),
    'to_graphtool': ('annnet.adapters.graphtool_adapter', 'to_graphtool'),
    'from_graphtool': ('annnet.adapters.graphtool_adapter', 'from_graphtool'),
    'to_pyg': ('annnet.adapters.pyg_adapter', 'to_pyg'),
    'write': ('annnet.io.annnet_format', 'write'),
    'read': ('annnet.io.annnet_format', 'read'),
    'to_json': ('annnet.io.json_format', 'to_json'),
    'from_json': ('annnet.io.json_format', 'from_json'),
    'write_ndjson': ('annnet.io.json_format', 'write_ndjson'),
    'to_dataframes': ('annnet.io.dataframes', 'to_dataframes'),
    'from_dataframes': ('annnet.io.dataframes', 'from_dataframes'),
    'from_csv': ('annnet.io.csv_format', 'from_csv'),
    'from_dataframe': ('annnet.io.csv_format', 'from_dataframe'),
    'edges_to_csv': ('annnet.io.csv_format', 'edges_to_csv'),
    'hyperedges_to_csv': ('annnet.io.csv_format', 'hyperedges_to_csv'),
    'from_excel': ('annnet.io.excel', 'from_excel'),
    'to_sif': ('annnet.io.sif', 'to_sif'),
    'from_sif': ('annnet.io.sif', 'from_sif'),
    'to_graphml': ('annnet.io.graphml', 'to_graphml'),
    'from_graphml': ('annnet.io.graphml', 'from_graphml'),
    'to_gexf': ('annnet.io.graphml', 'to_gexf'),
    'from_gexf': ('annnet.io.graphml', 'from_gexf'),
    'to_cx2': ('annnet.io.cx2', 'to_cx2'),
    'from_cx2': ('annnet.io.cx2', 'from_cx2'),
    'show_cx2': ('annnet.io.cx2', 'show_cx2'),
    'to_parquet': ('annnet.io.parquet', 'to_parquet'),
    'from_parquet': ('annnet.io.parquet', 'from_parquet'),
    'from_sbml': ('annnet.io.sbml', 'from_sbml'),
    'from_cobra_model': ('annnet.io.sbml_cobra', 'from_cobra_model'),
    'from_sbml_cobra': ('annnet.io.sbml_cobra', 'from_sbml'),
    'from_omnipath': ('annnet.io.omnipath', 'from_omnipath'),
}

_metadata_exports = {
    '__title__',
    '__version__',
    '__author__',
    '__authors__',
    '__maintainers__',
    '__license__',
    '__metadata__',
    'get_metadata',
    'get_latest_version',
    'info',
}

__all__ = sorted(
    set(_lazy_submodules) | set(_lazy_objects) | set(_lazy_functions) | _metadata_exports
)


def __getattr__(name: str) -> Any:
    if name in _lazy_submodules:
        from importlib import import_module

        value = import_module(_lazy_submodules[name])
    elif name in _lazy_objects:
        value = load_attr(_lazy_objects, name)
    elif name in _lazy_functions:
        mod, attr = _lazy_functions[name]
        value = make_lazy_function(mod, attr, __name__)
    else:
        raise AttributeError(name)

    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return export_dir(globals(), __all__)
