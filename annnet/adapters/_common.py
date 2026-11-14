"""
Shared helpers for adapter implementations.

This module is an internal façade for adapter modules. Adapter implementations
should import shared support utilities from here rather than reaching directly
into _support modules.
"""

from __future__ import annotations

from ..core import _structure
from ..core._structure import (
    binary_ends,
    _iter_node_ids,
    iter_edge_sides,
    _is_directed_eid,
    edge_entity_record,
)
from .._support.entities import restore_entities, update_edge_attributes
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

# The word an adapter and a manifest use for each kind of edge. The facade names
# the kinds and this names them again in the vocabulary the files carry, the way
# ``annnet.io._shared.common`` does for the entity kinds.
STORED_EDGE_KIND = {
    _structure.BINARY: 'binary',
    _structure.HYPER: 'hyper',
    _structure.NODE_EDGE: 'node_edge',
    _structure.PLACEHOLDER: 'edge_placeholder',
}


def slice_members(graph, slice_ids) -> tuple[set | None, set | None]:
    """Return the node ids and edge ids the named slices hold together.

    ``(None, None)`` when no slice is named, which a caller reads as "no
    restriction". A name the graph does not hold contributes nothing. The node
    set also holds the endpoints of every slice edge, so an edge kept never
    names a node that was dropped.
    """
    if not slice_ids:
        return None, None
    nodes: set = set()
    edges: set = set()
    for sid in slice_ids:
        if not graph.slices.exists(sid):
            continue
        nodes.update(graph.slices.nodes(sid))
        edges.update(graph.slices.edges(sid))
    for eid, _ref, sides in iter_edge_sides(graph):
        if eid in edges:
            for endpoint in (*sides.source, *sides.target):
                nodes.add(endpoint[0] if _structure.is_entity_key(endpoint) else endpoint)
    return nodes, edges


def requested_slices(slice=None, slices=None) -> set:
    """Merge the ``slice`` and ``slices`` arguments every exporter takes."""
    out: set = set()
    if slice is not None:
        out.update([slice] if isinstance(slice, str) else list(slice))
    if slices is not None:
        out.update(slices)
    return out


__all__ = [
    'STORED_EDGE_KIND',
    'requested_slices',
    'slice_members',
    'binary_ends',
    'dataframe_to_rows',
    'edge_entity_record',
    'restore_entities',
    'update_edge_attributes',
    'empty_dataframe',
    '_attrs_to_dict',
    '_is_directed_eid',
    'iter_edge_sides',
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
