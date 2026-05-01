"""Shared graph serialization helpers.

Boundary:
- policy helper: how a graph concept is represented logically
- shared serialization helper: format-agnostic encode/decode and restore logic
- not responsible for bytes/files/archives or backend projection mechanics
"""

from __future__ import annotations

import ast as _ast
import json
from typing import Any

from .dataframe_backend import dataframe_to_rows


def coerce_coeff_mapping(val):
    """Normalize various serialized coefficient mappings."""
    if val is None:
        return {}
    if isinstance(val, str):
        try:
            return coerce_coeff_mapping(json.loads(val))
        except json.JSONDecodeError:
            return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, (list, tuple)):
        out = {}
        for item in val:
            if isinstance(item, dict):
                if 'vertex' in item and '__value' in item:
                    out[item['vertex']] = {'__value': item['__value']}
                else:
                    for key, value in item.items():
                        out[key] = value
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                key, value = item
                out[key] = value
        return out
    return {}


def endpoint_coeff_map(edge_attrs, private_key, endpoint_set):
    """Return ``{vertex: float_coeff}`` for one endpoint side."""
    raw_mapping = (edge_attrs or {}).get(private_key, {})
    mapping = coerce_coeff_mapping(raw_mapping)
    endpoints = list(endpoint_set or mapping.keys())
    out = {}
    for vertex in endpoints:
        value = mapping.get(vertex, 1.0)
        if isinstance(value, dict):
            value = value.get('__value', 1.0)
        try:
            out[vertex] = float(value)
        except (TypeError, ValueError):
            out[vertex] = 1.0
    return out


def serialize_endpoint(endpoint: Any) -> Any:
    """Convert an endpoint into a JSON-safe representation."""
    if isinstance(endpoint, tuple) and len(endpoint) == 2 and isinstance(endpoint[1], tuple):
        return {'kind': 'supra', 'vertex': endpoint[0], 'layer': list(endpoint[1])}
    return endpoint


def deserialize_endpoint(value: Any) -> Any:
    """Restore a structural endpoint from JSON-safe or legacy serialized forms."""
    if isinstance(value, dict) and value.get('kind') == 'supra':
        return (value.get('vertex'), tuple(value.get('layer') or []))
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):
        return (value[0], tuple(value[1]))
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None and parsed != value:
            return deserialize_endpoint(parsed)
        if value.startswith('(') and value.endswith(')'):
            try:
                parsed = _ast.literal_eval(value)
            except (SyntaxError, ValueError):
                parsed = None
            if (
                isinstance(parsed, tuple)
                and len(parsed) == 2
                and isinstance(parsed[1], (tuple, list))
            ):
                return (parsed[0], tuple(parsed[1]))
    return value


def serialize_edge_layers(edge_layers: dict[str, Any]) -> dict[str, Any]:
    """Convert ``edge_layers[eid]`` (aa or (aa, bb)) into a JSON-safe form."""
    out = {}
    for eid, layer_spec in edge_layers.items():
        if layer_spec is None:
            continue
        if isinstance(layer_spec, tuple) and (
            len(layer_spec) == 0 or isinstance(layer_spec[0], str)
        ):
            out[eid] = {'kind': 'single', 'layers': [list(layer_spec)]}
        elif (
            isinstance(layer_spec, tuple)
            and len(layer_spec) == 2
            and isinstance(layer_spec[0], tuple)
            and isinstance(layer_spec[1], tuple)
        ):
            out[eid] = {'kind': 'pair', 'layers': [list(layer_spec[0]), list(layer_spec[1])]}
        else:
            out[eid] = {'kind': 'raw', 'value': repr(layer_spec)}
    return out


def deserialize_edge_layers(data: dict[str, Any]) -> dict[str, Any]:
    """Inverse of :func:`serialize_edge_layers`."""
    out = {}
    for eid, rec in data.items():
        kind = rec.get('kind')
        if kind == 'single':
            out[eid] = tuple(rec['layers'][0])
        elif kind == 'pair':
            out[eid] = (tuple(rec['layers'][0]), tuple(rec['layers'][1]))
    return out


def serialize_multilayer_manifest(
    graph,
    *,
    table_to_rows,
    serialize_edge_layers,
):
    """Serialize multilayer bookkeeping through the public layer API when possible."""
    aspect_attrs = {}
    for aspect in graph.aspects:
        attrs = graph.layers.get_aspect_attrs(aspect)
        if attrs:
            aspect_attrs[aspect] = attrs

    vm_rows = []
    node_layer_attrs = []
    for vid in graph.vertices():
        for layer_tuple in graph.layers.iter_vertex_layers(vid):
            vm_rows.append({'node': vid, 'layer': list(layer_tuple)})
            attrs = graph.layers.get_vertex_layer_attrs(vid, layer_tuple)
            if attrs:
                node_layer_attrs.append({'node': vid, 'layer': list(layer_tuple), 'attrs': attrs})

    layer_tuple_attrs = []
    for layer_tuple in graph.layers.iter_layers():
        attrs = graph.layers.get_layer_attrs(layer_tuple)
        if attrs:
            layer_tuple_attrs.append({'layer': list(layer_tuple), 'attrs': attrs})

    layer_table = getattr(graph, 'layer_attributes', None)
    layer_attr_rows = table_to_rows(layer_table) if layer_table is not None else []

    return {
        'aspects': list(graph.aspects),
        'aspect_attrs': aspect_attrs,
        'elem_layers': dict(graph.elem_layers),
        'VM': vm_rows,
        'edge_kind': dict(graph.edge_kind),
        'edge_layers': serialize_edge_layers(dict(graph.edge_layers)),
        'node_layer_attrs': node_layer_attrs,
        'layer_tuple_attrs': layer_tuple_attrs,
        'layer_attributes': layer_attr_rows,
    }


def restore_multilayer_manifest(
    graph,
    manifest: dict,
    *,
    rows_to_table,
    deserialize_edge_layers,
):
    """Restore multilayer bookkeeping through public graph/layer APIs."""

    def _normalize_layer_tuple(raw_layer):
        if raw_layer is None:
            return ()
        if isinstance(raw_layer, str):
            layer_tuple = (raw_layer,)
        else:
            layer_tuple = tuple(raw_layer)
        if not layer_tuple:
            return ()
        if graph.aspects and len(layer_tuple) == 1 and layer_tuple == ('_',):
            return tuple('_' for _ in graph.aspects)
        return layer_tuple

    aspects = manifest.get('aspects', [])
    elem_layers = manifest.get('elem_layers', {})
    if aspects:
        graph.layers.set_aspects(aspects)
        if elem_layers:
            graph.layers.set_elementary_layers(elem_layers)

    if not graph.aspects:
        layer_attr_rows = manifest.get('layer_attributes', [])
        if layer_attr_rows:
            graph.layer_attributes = rows_to_table(layer_attr_rows)
        return

    for aspect, attrs in manifest.get('aspect_attrs', {}).items():
        if attrs:
            graph.layers.set_aspect_attrs(aspect, **attrs)

    for row in manifest.get('VM', []):
        vid = row.get('vertex_id') or row.get('node')
        layer_tuple = _normalize_layer_tuple(row.get('layer'))
        if vid is None or not layer_tuple:
            continue
        if not graph.layers.has_presence(vid, layer_tuple):
            graph.add_vertices_bulk([vid], layer=layer_tuple)

    for eid, kind in manifest.get('edge_kind', {}).items():
        if graph.has_edge(edge_id=eid):
            graph.edge_kind[eid] = kind

    edge_layers = deserialize_edge_layers(manifest.get('edge_layers', {}))
    if edge_layers:
        graph.edge_layers.update(edge_layers)

    for row in manifest.get('node_layer_attrs', []):
        vid = row.get('vertex_id') or row.get('node')
        layer_tuple = _normalize_layer_tuple(row.get('layer'))
        attrs = row.get('attributes') or row.get('attrs') or {}
        if vid is None or not layer_tuple or not attrs:
            continue
        if graph.layers.has_presence(vid, layer_tuple):
            graph.layers.set_vertex_layer_attrs(vid, layer_tuple, **attrs)

    for row in manifest.get('layer_tuple_attrs', []):
        layer_tuple = _normalize_layer_tuple(row.get('layer'))
        attrs = row.get('attributes') or row.get('attrs') or {}
        if layer_tuple and attrs:
            graph.layers.set_layer_attrs(layer_tuple, **attrs)

    layer_attr_rows = manifest.get('layer_attributes', [])
    if layer_attr_rows:
        graph.layer_attributes = rows_to_table(layer_attr_rows)


def collect_slice_manifest(graph, *, requested_lids=None):
    """Collect slice membership and per-slice weights via public APIs."""
    if requested_lids:
        lids = [lid for lid in requested_lids if graph.slices.has_slice(lid)]
    else:
        lids = list(graph.slices.list_slices(include_default=True))

    slice_attr_rows = dataframe_to_rows(getattr(graph, 'edge_slice_attributes', None))
    for row in slice_attr_rows:
        lid = row.get('slice_id', row.get('slice'))
        if lid is None:
            continue
        if requested_lids and lid not in requested_lids:
            continue
        if lid not in lids:
            lids.append(lid)

    slices_section = {}
    slice_weights = {}

    for lid in lids:
        eids = list(graph.slices.get_slice_edges(lid)) if graph.slices.has_slice(lid) else []
        seen = set(eids)
        for row in slice_attr_rows:
            row_lid = row.get('slice_id', row.get('slice'))
            eid = row.get('edge_id', row.get('edge'))
            if row_lid != lid or eid is None or eid in seen:
                continue
            eids.append(eid)
            seen.add(eid)
        if not eids:
            continue

        slices_section[lid] = eids
        for eid in eids:
            weight = graph.attrs.get_edge_slice_attr(lid, eid, 'weight', default=None)
            if weight is None:
                continue
            slice_weights.setdefault(lid, {})[eid] = float(weight)

    return slices_section, slice_weights


def restore_slice_manifest(graph, slices_section: dict, slice_weights: dict):
    """Restore slice membership and weights through public slice/attrs APIs."""
    known_slices = set(graph.slices.list_slices(include_default=True))

    for lid, eids in (slices_section or {}).items():
        if lid not in known_slices:
            graph.slices.add_slice(lid)
            known_slices.add(lid)
        if eids:
            graph.slices.add_edges(lid, eids)

    for lid, per_edge in (slice_weights or {}).items():
        if lid not in known_slices:
            graph.slices.add_slice(lid)
            known_slices.add(lid)
        if per_edge:
            graph.attrs.set_edge_slice_attrs_bulk(
                lid,
                {eid: {'weight': float(weight)} for eid, weight in per_edge.items()},
            )
