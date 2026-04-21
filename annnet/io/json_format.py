from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import AnnNet

from .._dataframe_backend import empty_dataframe
from ..adapters._utils import (
    _deserialize_edge_layers,
    _deserialize_endpoint,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_VM,
    _df_to_rows,
    _endpoint_coeff_map,
    _is_directed_eid,
    _rows_to_df,
    _serialize_edge_layers,
    _serialize_endpoint,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_VM,
)


def _coerce_coeff_mapping(val):
    """Normalize various serialized forms into {vertex: {__value: float}|float}
    Accepts:
      - dict({v: x} or {v: {"__value": x}})
      - list of pairs [(v,x), ...]
      - list of dicts [{"vertex": v, "__value": x} | {v: x}, ...]
      - JSON string of any of the above
    """
    if val is None:
        return {}
    # JSON string?
    if isinstance(val, str):
        try:
            return _coerce_coeff_mapping(json.loads(val))
        except Exception:
            return {}
    # Already dict?
    if isinstance(val, dict):
        return val
    # List-like
    if isinstance(val, (list, tuple)):
        out = {}
        for item in val:
            if isinstance(item, dict):
                if 'vertex' in item and '__value' in item:
                    out[item['vertex']] = {'__value': item['__value']}
                else:
                    # e.g., {"A": 2.0}
                    for k, v in item.items():
                        out[k] = v
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                out[k] = v
            else:
                # ignore unrecognized shapes
                pass
        return out
    # Fallback
    return {}


def _endpoint_coeff_map(edge_attrs, private_key, endpoint_set):
    """Return {vertex: float_coeff} for the given endpoint_set.
    Reads from edge_attrs[private_key] which may be serialized in multiple shapes.
    Missing endpoints default to 1.0.
    """
    raw_mapping = (edge_attrs or {}).get(private_key, {})
    mapping = _coerce_coeff_mapping(raw_mapping)
    endpoints = list(endpoint_set or mapping.keys())
    out = {}
    for u in endpoints:
        val = mapping.get(u, 1.0)
        if isinstance(val, dict):
            val = val.get('__value', 1.0)
        try:
            out[u] = float(val)
        except Exception:
            out[u] = 1.0
    return out


def _edge_endpoint_sets(rec):
    if rec.etype == 'hyper':
        return set(rec.src or []), set(rec.tgt or [])
    src = set() if rec.src is None else {rec.src}
    tgt = set() if rec.tgt is None else {rec.tgt}
    return src, tgt


def _attrs_by_id(table, id_col: str, *, public_only: bool = False) -> dict:
    out = {}
    for row in _df_to_rows(table):
        item_id = row.get(id_col)
        if item_id is None:
            continue
        attrs = dict(row)
        attrs.pop(id_col, None)
        if public_only:
            attrs = {k: val for k, val in attrs.items() if not str(k).startswith('__')}
        out[item_id] = attrs
    return out


def to_json(graph: AnnNet, path, *, public_only: bool = False, indent: int = 0):
    """Node-link JSON with x-extensions (slices, edge_slices, hyperedges).
    Lossless vs your core (IDs, attrs, parallel, hyperedges, slices).
    """
    vertex_attrs = _attrs_by_id(
        getattr(graph, 'vertex_attributes', None), 'vertex_id', public_only=public_only
    )
    edge_attrs = _attrs_by_id(
        getattr(graph, 'edge_attributes', None), 'edge_id', public_only=public_only
    )

    # nodes
    nodes = []
    for v in graph.vertices():
        row = {'id': v}
        row.update(vertex_attrs.get(v, {}))
        nodes.append(row)

    # edges + hyperedges
    edges = []
    hyperedges = []
    for eidx in range(graph.ne):
        eid = graph._col_to_edge[eidx]
        rec = graph._edges[eid]
        S, T = _edge_endpoint_sets(rec)
        is_hyper = rec.etype == 'hyper'

        # attrs
        d = dict(edge_attrs.get(eid, {}))

        # weight + directed
        try:
            w = float(1.0 if rec.weight is None else rec.weight)
        except Exception:
            w = 1.0
        try:
            directed = bool(_is_directed_eid(graph, eid))
        except Exception:
            directed = True

        if is_hyper:
            # endpoint coeffs from private maps if present; else 1.0
            head_map = _endpoint_coeff_map(d, '__source_attr', S) or dict.fromkeys(S or [], 1.0)
            tail_map = _endpoint_coeff_map(d, '__target_attr', T) or dict.fromkeys(T or [], 1.0)
            # directed hyperedge
            hyperedges.append(
                {
                    'id': eid,
                    'directed': bool(directed),
                    'head': [_serialize_endpoint(x) for x in head_map.keys()] if directed else None,
                    'tail': [_serialize_endpoint(x) for x in tail_map.keys()] if directed else None,
                    'members': (
                        None
                        if directed
                        else [_serialize_endpoint(x) for x in {*head_map.keys(), *tail_map.keys()}]
                    ),
                    'attrs': d,
                    'weight': w,
                }
            )
        else:
            # regular/binary
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                v = u
            else:
                u, v = sorted(members)
            edges.append(
                {
                    'id': eid,
                    'source': _serialize_endpoint(u),
                    'target': _serialize_endpoint(v),
                    'directed': bool(directed),
                    'weight': w,
                    'attrs': d,
                }
            )

    # slices + per-slice weights
    slices = []
    try:
        for lid in graph.list_slices(include_default=True):
            slices.append({'slice_id': lid})
    except Exception:
        pass

    edge_slices = []
    # Collect memberships + weights if available
    try:
        for lid in graph.list_slices(include_default=True):
            try:
                for eid in graph.get_slice_edges(lid):
                    rec = {'slice_id': lid, 'edge_id': eid}
                    try:
                        w = graph.get_edge_slice_attr(lid, eid, 'weight', default=None)
                    except Exception:
                        try:
                            w = graph.get_edge_slice_attr(lid, eid, 'weight')
                        except Exception:
                            w = None
                    if w is not None:
                        rec['weight'] = float(w)
                    edge_slices.append(rec)
            except Exception:
                continue
    except Exception:
        pass

    doc = {
        'directed': True,  # node-link convention; per-edge directedness is in edges[*].directed
        'multigraph': True,
        'nodes': nodes,
        'edges': [
            {
                'id': e['id'],
                'source': e['source'],
                'target': e['target'],
                'directed': e['directed'],
                'weight': e['weight'],
                **(e.get('attrs') or {}),
            }
            for e in edges
        ],
        'x-extensions': {
            'slices': slices,
            'edge_slices': edge_slices,
            'hyperedges': [
                (
                    {
                        'id': h['id'],
                        'directed': True,
                        'head': h['head'],
                        'tail': h['tail'],
                        'weight': h['weight'],
                        **(h.get('attrs') or {}),
                    }
                    if h['directed']
                    else {
                        'id': h['id'],
                        'directed': False,
                        'members': h['members'],
                        'weight': h['weight'],
                        **(h.get('attrs') or {}),
                    }
                )
                for h in hyperedges
            ],
            'multilayer': {
                'aspects': list(getattr(graph, 'aspects', [])),
                'aspect_attrs': dict(getattr(graph, '_aspect_attrs', {})),
                'elem_layers': dict(getattr(graph, 'elem_layers', {})),
                'VM': _serialize_VM(getattr(graph, '_VM', set())),
                'edge_kind': dict(getattr(graph, 'edge_kind', {})),
                'edge_layers': _serialize_edge_layers(getattr(graph, 'edge_layers', {})),
                'node_layer_attrs': _serialize_node_layer_attrs(getattr(graph, '_state_attrs', {})),
                'layer_tuple_attrs': _serialize_layer_tuple_attrs(
                    getattr(graph, '_layer_attrs', {})
                ),
                'layer_attributes': _df_to_rows(
                    getattr(graph, 'layer_attributes', empty_dataframe({}))
                ),
            },
        },
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=indent)


def from_json(path) -> AnnNet:
    """Load AnnNet from node-link JSON + x-extensions (lossless wrt schema above)."""
    from ..core.graph import AnnNet

    with open(path, encoding='utf-8') as f:
        doc = json.load(f)
    H = AnnNet()
    ext = doc.get('x-extensions') or {}
    mm = ext.get('multilayer', {})
    aspects = mm.get('aspects', [])
    elem_layers = mm.get('elem_layers', {})
    if aspects:
        H.set_aspects(aspects)
        if elem_layers:
            H.set_elementary_layers(elem_layers)

    # vertices
    # Multilayer graphs use _ensure_vertex_row (avoids layer contamination via bulk insert).
    if aspects:
        for nd in doc.get('nodes', []):
            vid = nd.get('id')
            if vid is None:
                continue
            vattrs = {k: v for k, v in nd.items() if k != 'id'}
            H._ensure_vertex_table()
            H._ensure_vertex_row(vid)
            if vattrs:
                H.set_vertex_attrs(vid, **vattrs)
    else:
        vertex_dicts = []
        for nd in doc.get('nodes', []):
            vid = nd.get('id')
            if vid is None:
                continue
            row = {'vertex_id': vid}
            row.update({k: v for k, v in nd.items() if k != 'id'})
            vertex_dicts.append(row)
        if vertex_dicts:
            H.add_vertices_bulk(vertex_dicts)

    # edges (binary)
    # Multilayer graphs use supra-node tuples as endpoints — add_edges_bulk is flat-only,
    # so fall back to scalar add_edge for multilayer.
    edge_dicts = []
    for e in doc.get('edges', []):
        eid = e.get('id')
        u = _deserialize_endpoint(e.get('source'))
        v = _deserialize_endpoint(e.get('target'))
        if eid is None or u is None or v is None:
            continue
        directed = bool(e.get('directed', True))
        w = e.get('weight', 1.0)
        attrs = {
            k: val
            for k, val in e.items()
            if k not in {'id', 'source', 'target', 'directed', 'weight'}
        }
        if aspects:
            # supra-node endpoints: must use scalar add_edge
            H.add_edge(u, v, edge_id=eid, directed=directed, parallel='parallel')
            rec = H._edges.get(eid)
            if rec is not None:
                rec.weight = float(w)
            if attrs:
                H.set_edge_attrs(eid, **attrs)
        else:
            entry = {'source': u, 'target': v, 'edge_id': eid, 'directed': directed, 'weight': w}
            if attrs:
                entry['attributes'] = attrs
            edge_dicts.append(entry)
    if edge_dicts:
        H.add_edges_bulk(edge_dicts)

    # hyperedges — bulk insert
    hyper_dicts = []
    hyper_weight_patch = {}
    hyper_attrs_pending = {}
    for h in ext.get('hyperedges', []):
        eid = h.get('id')
        directed = bool(h.get('directed', True))
        w = h.get('weight', 1.0)
        attrs = {
            k: v
            for k, v in h.items()
            if k not in {'id', 'directed', 'head', 'tail', 'members', 'weight'}
        }
        if directed:
            head = [_deserialize_endpoint(x) for x in list(h.get('head') or [])]
            tail = [_deserialize_endpoint(x) for x in list(h.get('tail') or [])]
            entry = {'head': head, 'tail': tail, 'edge_id': eid, 'edge_directed': True, 'weight': w}
        else:
            members = [_deserialize_endpoint(x) for x in list(h.get('members') or [])]
            entry = {'members': members, 'edge_id': eid, 'edge_directed': False, 'weight': w}
        hyper_dicts.append(entry)
        if attrs:
            hyper_attrs_pending[eid] = attrs
    if hyper_dicts:
        H.add_hyperedges_bulk(hyper_dicts)
    if hyper_attrs_pending:
        H.set_edge_attrs_bulk(hyper_attrs_pending)

    # slices + edge_slices — bulk
    known_slices = set(H.list_slices(include_default=True))
    for L in ext.get('slices', []):
        lid = L.get('slice_id')
        if lid is None:
            continue
        if lid not in known_slices:
            try:
                H.add_slice(lid)
                known_slices.add(lid)
            except Exception:
                pass

    slice_edges: dict = {}
    slice_weights: dict = {}
    for EL in ext.get('edge_slices', []):
        lid = EL.get('slice_id')
        eid = EL.get('edge_id')
        if lid is None or eid is None:
            continue
        slice_edges.setdefault(lid, set()).add(eid)
        if 'weight' in EL:
            slice_weights[(lid, eid)] = float(EL['weight'])
    for lid, eids in slice_edges.items():
        try:
            H.add_edges_to_slice_bulk(lid, eids)
        except Exception:
            pass
    for (lid, eid), w in slice_weights.items():
        try:
            H.set_edge_slice_attrs(lid, eid, weight=w)
        except Exception:
            pass

    # multilayer / Kivela
    mm = ext.get('multilayer', {})

    aspect_attrs = mm.get('aspect_attrs', {})
    if aspect_attrs:
        H._aspect_attrs.update(aspect_attrs)

    VM_data = mm.get('VM', [])
    if VM_data:
        H._VM = _deserialize_VM(VM_data)

    # edge_kind / edge_layers
    ek = mm.get('edge_kind', {})
    el_ser = mm.get('edge_layers', {})
    if ek:
        for eid, kind in ek.items():
            rec = H._edges.get(eid)
            if rec is None:
                continue
            if kind == 'hyper':
                rec.etype = 'hyper'
            else:
                rec.ml_kind = kind
    if el_ser:
        H.edge_layers.update(_deserialize_edge_layers(el_ser))

    nl_attrs_ser = mm.get('node_layer_attrs', [])
    if nl_attrs_ser:
        H._state_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

    layer_tuple_attrs_ser = mm.get('layer_tuple_attrs', [])
    if layer_tuple_attrs_ser:
        H._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

    layer_attr_rows = mm.get('layer_attributes', [])
    if layer_attr_rows:
        H.layer_attributes = _rows_to_df(layer_attr_rows)

    return H


def write_ndjson(graph: AnnNet, dir_path):
    """Write nodes.ndjson, edges.ndjson, hyperedges.ndjson, slices.ndjson, edge_slices.ndjson.
    Each line is one JSON object. Lossless wrt to_json schema.
    """
    import json
    import os

    os.makedirs(dir_path, exist_ok=True)
    vertex_attrs = _attrs_by_id(getattr(graph, 'vertex_attributes', None), 'vertex_id')
    edge_attrs = _attrs_by_id(getattr(graph, 'edge_attributes', None), 'edge_id')

    with open(f'{dir_path}/nodes.ndjson', 'w', encoding='utf-8') as f:
        for v in graph.vertices():
            obj = {'id': v}
            obj.update(vertex_attrs.get(v, {}))
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

    with (
        open(f'{dir_path}/edges.ndjson', 'w', encoding='utf-8') as fe,
        open(f'{dir_path}/hyperedges.ndjson', 'w', encoding='utf-8') as fh,
    ):
        for eidx in range(graph.ne):
            eid = graph._col_to_edge[eidx]
            rec = graph._edges[eid]
            S, T = _edge_endpoint_sets(rec)
            is_hyper = rec.etype == 'hyper'

            d = dict(edge_attrs.get(eid, {}))

            try:
                w = float(1.0 if rec.weight is None else rec.weight)
            except Exception:
                w = 1.0
            try:
                directed = bool(_is_directed_eid(graph, eid))
            except Exception:
                directed = True

            if is_hyper:
                head_map = _endpoint_coeff_map(d, '__source_attr', S) or dict.fromkeys(S or [], 1.0)
                tail_map = _endpoint_coeff_map(d, '__target_attr', T) or dict.fromkeys(T or [], 1.0)
                obj = {'id': eid, 'directed': directed, 'weight': w}
                if directed:
                    obj.update({'head': list(head_map), 'tail': list(tail_map)})
                else:
                    obj.update({'members': list({*head_map, *tail_map})})
                obj.update({k: v for k, v in d.items() if not str(k).startswith('__')})
                fh.write(json.dumps(obj, ensure_ascii=False) + '\n')
            else:
                members = S | T
                if len(members) == 1:
                    u = next(iter(members))
                    v = u
                else:
                    u, v = sorted(members)
                obj = {'id': eid, 'source': u, 'target': v, 'directed': directed, 'weight': w}
                obj.update({k: v for k, v in d.items() if not str(k).startswith('__')})
                fe.write(json.dumps(obj, ensure_ascii=False) + '\n')

    # slices
    with open(f'{dir_path}/slices.ndjson', 'w', encoding='utf-8') as fl:
        try:
            for lid in graph.list_slices(include_default=True):
                fl.write(json.dumps({'slice_id': lid}, ensure_ascii=False) + '\n')
        except Exception:
            pass

    with open(f'{dir_path}/edge_slices.ndjson', 'w', encoding='utf-8') as fel:
        try:
            for lid in graph.list_slices(include_default=True):
                for eid in graph.get_slice_edges(lid):
                    rec = {'slice_id': lid, 'edge_id': eid}
                    try:
                        w = graph.get_edge_slice_attr(lid, eid, 'weight', default=None)
                    except Exception:
                        try:
                            w = graph.get_edge_slice_attr(lid, eid, 'weight')
                        except Exception:
                            w = None
                    if w is not None:
                        rec['weight'] = float(w)
                    fel.write(json.dumps(rec, ensure_ascii=False) + '\n')
        except Exception:
            pass
