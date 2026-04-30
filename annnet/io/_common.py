"""
Shared helpers for IO implementations.

This module is an internal façade for IO modules. Format implementations should
import shared support utilities from here rather than reaching directly into
_support modules.
"""

from __future__ import annotations

from .._support.graph_records import (
    _rows_to_df,
    _is_directed_eid,
    _iter_vertex_ids,
    _iter_edge_records,
)
from .._support.serialization import (
    endpoint_coeff_map,
    serialize_endpoint,
    deserialize_endpoint,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)
from .._support.dataframe_backend import (
    dataframe_width,
    empty_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_read_tsv,
    dataframe_from_rows,
    dataframe_write_csv,
    dataframe_read_excel,
    dataframe_from_columns,
    dataframe_read_parquet,
    dataframe_column_values,
    dataframe_write_parquet,
    dataframe_read_delimited,
    rename_dataframe_columns,
    dataframe_select_to_numpy,
    dataframe_column_is_numeric,
)

__all__ = [
    'collect_slice_manifest',
    'dataframe_column_is_numeric',
    'dataframe_column_values',
    'dataframe_columns',
    'dataframe_from_columns',
    'dataframe_from_rows',
    'dataframe_height',
    'dataframe_read_delimited',
    'dataframe_read_excel',
    'dataframe_read_parquet',
    'dataframe_read_tsv',
    'dataframe_select_to_numpy',
    'dataframe_to_rows',
    'dataframe_width',
    'dataframe_write_csv',
    'dataframe_write_parquet',
    'deserialize_edge_layers',
    'deserialize_endpoint',
    'empty_dataframe',
    'endpoint_coeff_map',
    'rename_dataframe_columns',
    'restore_multilayer_manifest',
    'restore_slice_manifest',
    'serialize_edge_layers',
    'serialize_endpoint',
    'serialize_multilayer_manifest',
    '_is_directed_eid',
    '_iter_edge_records',
    '_iter_vertex_ids',
    '_rows_to_df',
]
