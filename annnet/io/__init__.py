"""annnet.io: consolidated I/O API with lazy symbol loading."""

from __future__ import annotations

from typing import Any

from .._support.lazy_exports import export_dir, resolve_lazy_export

_lazy_symbols: dict[str, tuple[str, str]] = {
    # annnet native format
    'write': ('annnet.io.annnet_format', 'write'),
    'read': ('annnet.io.annnet_format', 'read'),
    # JSON
    'to_json': ('annnet.io.json_format', 'to_json'),
    'from_json': ('annnet.io.json_format', 'from_json'),
    'write_ndjson': ('annnet.io.json_format', 'write_ndjson'),
    # DataFrame
    'to_dataframes': ('annnet.io.dataframes', 'to_dataframes'),
    'from_dataframes': ('annnet.io.dataframes', 'from_dataframes'),
    # CSV / Excel
    'from_csv': ('annnet.io.csv_format', 'from_csv'),
    'from_dataframe': ('annnet.io.csv_format', 'from_dataframe'),
    'edges_to_csv': ('annnet.io.csv_format', 'edges_to_csv'),
    'hyperedges_to_csv': ('annnet.io.csv_format', 'hyperedges_to_csv'),
    'from_excel': ('annnet.io.excel', 'from_excel'),
    # SIF / GraphML / GEXF / CX2
    'to_sif': ('annnet.io.sif', 'to_sif'),
    'from_sif': ('annnet.io.sif', 'from_sif'),
    'to_graphml': ('annnet.io.graphml', 'to_graphml'),
    'from_graphml': ('annnet.io.graphml', 'from_graphml'),
    'to_gexf': ('annnet.io.graphml', 'to_gexf'),
    'from_gexf': ('annnet.io.graphml', 'from_gexf'),
    'to_cx2': ('annnet.io.cx2', 'to_cx2'),
    'from_cx2': ('annnet.io.cx2', 'from_cx2'),
    'show_cx2': ('annnet.io.cx2', 'show_cx2'),
    # Parquet
    'to_parquet': ('annnet.io.parquet', 'to_parquet'),
    'from_parquet': ('annnet.io.parquet', 'from_parquet'),
    # SBML
    'from_sbml': ('annnet.io.sbml', 'from_sbml'),
    'from_cobra_model': ('annnet.io.sbml_cobra', 'from_cobra_model'),
    'from_sbml_cobra': ('annnet.io.sbml_cobra', 'from_sbml'),
    # OmniPath
    'from_omnipath': ('annnet.io.omnipath', 'from_omnipath'),
}

__all__ = sorted(_lazy_symbols)


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(globals(), name, attrs=_lazy_symbols)


def __dir__() -> list[str]:
    return export_dir(globals(), __all__)
