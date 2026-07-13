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
    include_edge_layers: bool = True,
):
    """Serialize multilayer bookkeeping through the public layer API when possible.

    Pass ``include_edge_layers=False`` to skip the (potentially expensive)
    serialize_edge_layers pass when the caller already reads ``graph.edge_layers``
    directly (e.g. the native parquet writer).
    """
    aspect_attrs = {}
    for aspect in graph.aspects:
        attrs = graph.layers.get_aspect_attrs(aspect)
        if attrs:
            aspect_attrs[aspect] = attrs

    # Single scan over ``_entities`` instead of calling
    # ``iter_vertex_layers`` per vertex — that helper does a full V-wide
    # scan internally, which would make this loop O(V²).
    vm_rows = []
    node_layer_attrs = []
    entities = getattr(graph, '_entities', {}) or {}
    for (uu, aa), rec in entities.items():
        if rec.kind != 'vertex':
            continue
        layer_tuple = aa
        vm_rows.append({'node': uu, 'layer': list(layer_tuple)})
        attrs = graph.layers.get_vertex_layer_attrs(uu, layer_tuple)
        if attrs:
            node_layer_attrs.append({'node': uu, 'layer': list(layer_tuple), 'attrs': attrs})

    layer_tuple_attrs = []
    for layer_tuple in graph.layers.iter_layers():
        attrs = graph.layers.get_layer_attrs(layer_tuple)
        if attrs:
            layer_tuple_attrs.append({'layer': list(layer_tuple), 'attrs': attrs})

    layer_table = getattr(graph, 'layer_attributes', None)
    layer_attr_rows = table_to_rows(layer_table) if layer_table is not None else []

    out = {
        'aspects': list(graph.aspects),
        'aspect_attrs': aspect_attrs,
        'elem_layers': dict(graph.elem_layers),
        'VM': vm_rows,
        'edge_kind': dict(graph.edge_kind),
        'node_layer_attrs': node_layer_attrs,
        'layer_tuple_attrs': layer_tuple_attrs,
        'layer_attributes': layer_attr_rows,
    }
    if include_edge_layers:
        out['edge_layers'] = serialize_edge_layers(dict(graph.edge_layers))
    return out


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
        # Skip set_aspects if the caller already declared the same aspects on
        # an entity-populated graph (e.g. native read() declares them up-front
        # to preserve stored layer coords). A second set_aspects would treat
        # the existing entities as flat and reassign them to the placeholder.
        same_aspects = tuple(graph.aspects) == tuple(aspects)
        if not same_aspects:
            graph.layers.set_aspects(aspects)
        if elem_layers and not same_aspects:
            graph.layers.set_elementary_layers(elem_layers)
        elif elem_layers:
            # Augment existing layer values without going through
            # set_elementary_layers — that helper calls
            # _drop_unused_placeholder_layers, which would strip the '_'
            # placeholder if no entity has the multi-aspect placeholder
            # coord, breaking subsequent _make_layer_coord validation.
            for aspect, values in elem_layers.items():
                if aspect in graph._layers:
                    graph._layers[aspect].update(values)
            graph.layers._rebuild_all_layers_cache()
        # Ensure '_' is available per aspect; legacy graphs may have stored
        # entities at single-aspect ('_',) coords on a multi-aspect graph.
        graph._ensure_placeholder_layers_declared()

    if not graph.aspects:
        layer_attr_rows = manifest.get('layer_attributes', [])
        if layer_attr_rows:
            graph.layer_attributes = rows_to_table(layer_attr_rows)
        return

    for aspect, attrs in manifest.get('aspect_attrs', {}).items():
        if attrs:
            graph.layers.set_aspect_attrs(aspect, **attrs)

    # VM rows: group by normalized layer_tuple, then one bulk insert per layer.
    # Cache _normalize_layer_tuple results — the same raw layer recurs many times.
    _norm_cache: dict = {}

    def _norm_cached(raw):
        # raw lists/tuples are unhashable as-is; hash by tuple form
        key = tuple(raw) if isinstance(raw, (list, tuple)) else raw
        cached = _norm_cache.get(key)
        if cached is not None:
            return cached
        result = _normalize_layer_tuple(raw)
        _norm_cache[key] = result
        return result

    vm_by_layer: dict = {}
    for row in manifest.get('VM', []):
        vid = row.get('vertex_id') or row.get('node')
        layer_tuple = _norm_cached(row.get('layer'))
        if vid is None or not layer_tuple:
            continue
        vm_by_layer.setdefault(layer_tuple, []).append(vid)

    entities = graph._entities
    placeholder = tuple('_' for _ in graph.aspects)
    single_placeholder = ('_',)
    for layer_tuple, vids in vm_by_layer.items():
        # Filter to missing vids without paying _validate_layer_tuple per vid.
        # Also treat single-aspect ('_',) and multi-aspect ('_','_',...) as
        # the same supra-node so legacy fixtures stored at ('_',) don't get
        # duplicated by a VM row asking for the multi-aspect placeholder.
        if layer_tuple == placeholder:
            missing = [
                v
                for v in vids
                if (v, layer_tuple) not in entities and (v, single_placeholder) not in entities
            ]
        else:
            missing = [v for v in vids if (v, layer_tuple) not in entities]
        if missing:
            graph._add_vertices_bulk(missing, layer=layer_tuple)

    # Drop spurious placeholder node-layers. Readers add vertices flat (before
    # aspects/VM are known), landing them at the ('_', ...) placeholder; the VM
    # pass above then ALSO places them at their real layer, doubling the
    # node-layer count. Remove the placeholder membership for any vertex that now
    # lives at a real layer and that the manifest never listed at the placeholder
    # — but only when it is a true orphan (no incident edges), so a genuine
    # placeholder-anchored vertex/edge is never disturbed.
    entities = graph._entities
    legit_placeholder = set(vm_by_layer.get(placeholder, ()))
    by_vid: dict = {}
    for (u, aa), rec in entities.items():
        if rec.kind == 'vertex':
            by_vid.setdefault(u, []).append(aa)
    candidates = [
        (u, placeholder)
        for u, layers in by_vid.items()
        if placeholder in layers
        and u not in legit_placeholder
        and any(a != placeholder for a in layers)
    ]
    if candidates:
        indptr = graph._get_csr().indptr
        drop = {
            ek
            for ek in candidates
            if indptr[entities[ek].row_idx + 1] == indptr[entities[ek].row_idx]
        }
        if drop:
            graph._remove_orphan_node_layers(drop)

    for eid, kind in manifest.get('edge_kind', {}).items():
        if graph.has_edge(edge_id=eid):
            graph.edge_kind[eid] = kind

    edge_layers = deserialize_edge_layers(manifest.get('edge_layers', {}))
    # edge_layers is a proxy over graph._edges; set only where the edge already
    # exists (mirrors the edge_kind guard above). Consumers that create edges
    # after the manifest restore (e.g. cx2's visual overlay) would otherwise
    # KeyError on a not-yet-created edge id.
    for eid, val in edge_layers.items():
        if graph.has_edge(edge_id=eid):
            graph.edge_layers[eid] = val

    # node_layer_attrs: write directly into _state_attrs (presence already enforced
    # by the VM pass above) — avoids 40k+ public-API calls each redoing validation.
    state_attrs = graph._state_attrs
    for row in manifest.get('node_layer_attrs', []):
        vid = row.get('vertex_id') or row.get('node')
        layer_tuple = _norm_cached(row.get('layer'))
        attrs = row.get('attributes') or row.get('attrs') or {}
        if vid is None or not layer_tuple or not attrs:
            continue
        if (vid, layer_tuple) not in entities:
            continue
        bucket = state_attrs.get((vid, layer_tuple))
        if bucket is None:
            state_attrs[(vid, layer_tuple)] = dict(attrs)
        else:
            bucket.update(attrs)

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
        lids = [lid for lid in requested_lids if graph.slices.exists(lid)]
    else:
        lids = list(graph.slices.list(include_default=True))

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

    # Snapshot the (slice, edge) → weight map once from slice_attr_rows;
    # the per-edge ``get_edge_slice_attr`` lookup is O(rows-in-table)
    # and would make this loop quadratic in edge count.
    weight_lookup: dict[tuple, float] = {}
    for row in slice_attr_rows:
        row_lid = row.get('slice_id', row.get('slice'))
        row_eid = row.get('edge_id', row.get('edge'))
        if row_lid is None or row_eid is None:
            continue
        w = row.get('weight')
        if w is None:
            continue
        try:
            weight_lookup[(row_lid, row_eid)] = float(w)
        except (TypeError, ValueError):
            continue

    for lid in lids:
        eids = list(graph.slices.edges(lid)) if graph.slices.exists(lid) else []
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
        per_slice: dict = {}
        for eid in eids:
            w = weight_lookup.get((lid, eid))
            if w is not None:
                per_slice[eid] = w
        if per_slice:
            slice_weights[lid] = per_slice

    return slices_section, slice_weights


def restore_slice_manifest(graph, slices_section: dict, slice_weights: dict):
    """Restore slice membership and weights through public slice/attrs APIs."""
    known_slices = set(graph.slices.list(include_default=True))

    for lid, eids in (slices_section or {}).items():
        if lid not in known_slices:
            graph.slices.add(lid)
            known_slices.add(lid)
        if eids:
            graph.slices.add_edges(lid, eids)

    for lid, per_edge in (slice_weights or {}).items():
        if lid not in known_slices:
            graph.slices.add(lid)
            known_slices.add(lid)
        if per_edge:
            graph.attrs.set_edge_slice_attrs_bulk(
                lid,
                {eid: {'weight': float(weight)} for eid, weight in per_edge.items()},
            )
