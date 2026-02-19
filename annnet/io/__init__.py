"""annnet.io: consolidated I/O API with lazy symbol loading."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_lazy_symbols: dict[str, tuple[str, str]] = {
    # annnet native format
    "write": ("annnet.io.io_annnet", "write"),
    "read": ("annnet.io.io_annnet", "read"),
    # JSON
    "to_json": ("annnet.io.json_io", "to_json"),
    "from_json": ("annnet.io.json_io", "from_json"),
    "write_ndjson": ("annnet.io.json_io", "write_ndjson"),
    # DataFrame
    "to_dataframes": ("annnet.io.dataframe_io", "to_dataframes"),
    "from_dataframes": ("annnet.io.dataframe_io", "from_dataframes"),
    # CSV / Excel
    "load_csv_to_graph": ("annnet.io.csv_io", "load_csv_to_graph"),
    "from_dataframe": ("annnet.io.csv_io", "from_dataframe"),
    "export_edge_list_csv": ("annnet.io.csv_io", "export_edge_list_csv"),
    "export_hyperedge_csv": ("annnet.io.csv_io", "export_hyperedge_csv"),
    "load_excel_to_graph": ("annnet.io.excel", "load_excel_to_graph"),
    # SIF / GraphML / GEXF / CX2
    "to_sif": ("annnet.io.SIF_io", "to_sif"),
    "from_sif": ("annnet.io.SIF_io", "from_sif"),
    "to_graphml": ("annnet.io.GraphML_io", "to_graphml"),
    "from_graphml": ("annnet.io.GraphML_io", "from_graphml"),
    "to_gexf": ("annnet.io.GraphML_io", "to_gexf"),
    "from_gexf": ("annnet.io.GraphML_io", "from_gexf"),
    "to_cx2": ("annnet.io.cx2_io", "to_cx2"),
    "from_cx2": ("annnet.io.cx2_io", "from_cx2"),
    "show_cx2": ("annnet.io.cx2_io", "show"),
    # Parquet
    "to_parquet": ("annnet.io.Parquet_io", "to_parquet"),
    "from_parquet": ("annnet.io.Parquet_io", "from_parquet"),
    # SBML
    "from_sbml": ("annnet.io.SBML_io", "from_sbml"),
    "from_cobra_model": ("annnet.io.sbml_cobra_io", "from_cobra_model"),
    "from_sbml_cobra": ("annnet.io.sbml_cobra_io", "from_sbml"),
    # OmniPath
    "read_omnipath": ("annnet.io.read_omnipath", "read_omnipath"),
}

__all__ = sorted(_lazy_symbols)


def __getattr__(name: str) -> Any:
    if name in _lazy_symbols:
        mod, attr = _lazy_symbols[name]
        return getattr(import_module(mod), attr)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(__all__))
