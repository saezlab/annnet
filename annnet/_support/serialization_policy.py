from __future__ import annotations

import json


def coerce_coeff_mapping(val):
    """Normalize various serialized coefficient mappings.

    Accepts dicts, list-of-pairs, list-of-dicts, tuples, and JSON strings.
    Returns a dict mapping vertex id -> float-like payload.
    """
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
                node_layer_attrs.append(
                    {
                        'node': vid,
                        'layer': list(layer_tuple),
                        'attrs': attrs,
                    }
                )

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
        layer_tuple = tuple(row.get('layer') or [])
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
        layer_tuple = tuple(row.get('layer') or [])
        attrs = row.get('attributes') or row.get('attrs') or {}
        if vid is None or not layer_tuple or not attrs:
            continue
        if graph.layers.has_presence(vid, layer_tuple):
            graph.layers.set_vertex_layer_attrs(vid, layer_tuple, **attrs)

    for row in manifest.get('layer_tuple_attrs', []):
        layer_tuple = tuple(row.get('layer') or [])
        attrs = row.get('attributes') or row.get('attrs') or {}
        if layer_tuple and attrs:
            graph.layers.set_layer_attrs(layer_tuple, **attrs)

    layer_attr_rows = manifest.get('layer_attributes', [])
    if layer_attr_rows:
        graph.layer_attributes = rows_to_table(layer_attr_rows)
