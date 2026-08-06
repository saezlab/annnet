"""
Shared helpers for adapter implementations.

This module is an internal façade for adapter modules. Adapter implementations
should import shared support utilities from here rather than reaching directly
into _support modules.
"""

from __future__ import annotations

from ..core._structure import (
    _iter_node_ids,
    _is_directed_eid,
    _iter_edge_records,
)
from .._support.graph_records import (
    _rows_like,
    _rows_to_df,
    _attrs_to_dict,
    _serialize_value,
)
from .._support.serialization import (
    stored as stored_key,
    endpoint_coeff_map,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)
from .._support.dataframe_backend import (
    empty_dataframe,
    dataframe_to_rows,
)

__all__ = [
    'dataframe_to_rows',
    'empty_dataframe',
    '_attrs_to_dict',
    '_is_directed_eid',
    '_iter_edge_records',
    '_iter_node_ids',
    '_rows_like',
    '_rows_to_df',
    '_serialize_value',
    'collect_slice_manifest',
    'stored_key',
    'deserialize_edge_layers',
    'endpoint_coeff_map',
    'restore_multilayer_manifest',
    'restore_slice_manifest',
    'serialize_edge_layers',
    'serialize_multilayer_manifest',
]
