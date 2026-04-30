from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import AnnNet

from ..adapters._utils import (
    _df_to_rows,
    _rows_to_df,
    _is_directed_eid,
    _iter_edge_records,
)
from .._support.serialization import (
    endpoint_coeff_map,
    serialize_endpoint,
    deserialize_endpoint,
    serialize_edge_layers,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)


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
    for eid, rec in _iter_edge_records(graph):
        S, T = _edge_endpoint_sets(rec)
        is_hyper = rec.etype == 'hyper'

        # attrs
        d = dict(edge_attrs.get(eid, {}))

        # weight + directed
        try:
            w = float(1.0 if rec.weight is None else rec.weight)
        except (TypeError, ValueError):
            w = 1.0
        try:
            directed = bool(_is_directed_eid(graph, eid))
        except (AttributeError, KeyError, TypeError, ValueError):
            directed = True

        if is_hyper:
            # endpoint coeffs from private maps if present; else 1.0
            head_map = endpoint_coeff_map(d, '__source_attr', S) or dict.fromkeys(S or [], 1.0)
            tail_map = endpoint_coeff_map(d, '__target_attr', T) or dict.fromkeys(T or [], 1.0)
            # directed hyperedge
            hyperedges.append(
                {
                    'id': eid,
                    'directed': bool(directed),
                    'head': [serialize_endpoint(x) for x in head_map.keys()] if directed else None,
                    'tail': [serialize_endpoint(x) for x in tail_map.keys()] if directed else None,
                    'members': (
                        None
                        if directed
                        else [serialize_endpoint(x) for x in {*head_map.keys(), *tail_map.keys()}]
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
                    'source': serialize_endpoint(u),
                    'target': serialize_endpoint(v),
                    'directed': bool(directed),
                    'weight': w,
                    'attrs': d,
                }
            )

    # slices + per-slice weights
    slices = []
    for lid in graph.slices.list_slices(include_default=True):
        slices.append({'slice_id': lid})

    edge_slices = []
    for lid in graph.slices.list_slices(include_default=True):
        for eid in graph.slices.get_slice_edges(lid):
            rec = {'slice_id': lid, 'edge_id': eid}
            try:
                w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight', default=None)
            except TypeError:
                w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight')
            if w is not None:
                rec['weight'] = float(w)
            edge_slices.append(rec)

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
            'multilayer': serialize_multilayer_manifest(
                graph,
                table_to_rows=_df_to_rows,
                serialize_edge_layers=serialize_edge_layers,
            ),
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
        H.layers.set_aspects(aspects)
        if elem_layers:
            H.layers.set_elementary_layers(elem_layers)

    # vertices
    if aspects:
        vertex_attrs_pending = {}
        for nd in doc.get('nodes', []):
            vid = nd.get('id')
            if vid is None:
                continue
            vattrs = {k: v for k, v in nd.items() if k != 'id'}
            if vattrs:
                vertex_attrs_pending[vid] = vattrs
    else:
        vertex_attrs_pending = {}
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
        u = deserialize_endpoint(e.get('source'))
        v = deserialize_endpoint(e.get('target'))
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
            # supra-node endpoints: must use scalar add_edges
            H.add_edges(u, v, edge_id=eid, directed=directed, weight=float(w), parallel='parallel')
            if attrs:
                H.attrs.set_edge_attrs(eid, **attrs)
        else:
            entry = {'source': u, 'target': v, 'edge_id': eid, 'directed': directed, 'weight': w}
            if attrs:
                entry['attributes'] = attrs
            edge_dicts.append(entry)
    if edge_dicts:
        H.add_edges_bulk(edge_dicts)

    # hyperedges — bulk insert
    hyper_dicts = []
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
            head = [deserialize_endpoint(x) for x in list(h.get('head') or [])]
            tail = [deserialize_endpoint(x) for x in list(h.get('tail') or [])]
            entry = {'head': head, 'tail': tail, 'edge_id': eid, 'edge_directed': True, 'weight': w}
        else:
            members = [deserialize_endpoint(x) for x in list(h.get('members') or [])]
            entry = {'members': members, 'edge_id': eid, 'edge_directed': False, 'weight': w}
        hyper_dicts.append(entry)
        if attrs:
            hyper_attrs_pending[eid] = attrs
    if hyper_dicts:
        H.add_hyperedges_bulk(hyper_dicts)
    if hyper_attrs_pending:
        H.attrs.set_edge_attrs_bulk(hyper_attrs_pending)

    # slices + edge_slices — bulk
    known_slices = set(H.slices.list_slices(include_default=True))
    for L in ext.get('slices', []):
        lid = L.get('slice_id')
        if lid is None:
            continue
        if lid not in known_slices:
            H.slices.add_slice(lid)
            known_slices.add(lid)

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
    known_edges = set(H.edge_definitions) | set(H.hyperedge_definitions)
    for lid, eids in slice_edges.items():
        if lid not in known_slices:
            continue
        for eid in eids:
            if eid in known_edges:
                H.slices.add_edge_to_slice(lid, eid)
    for (lid, eid), w in slice_weights.items():
        if lid in known_slices and eid in known_edges:
            H.attrs.set_edge_slice_attrs(lid, eid, weight=w)

    restore_multilayer_manifest(
        H,
        mm,
        rows_to_table=_rows_to_df,
        deserialize_edge_layers=deserialize_edge_layers,
    )
    if vertex_attrs_pending:
        for vid, attrs in vertex_attrs_pending.items():
            H.attrs.set_vertex_attrs(vid, **attrs)

    return H


def write_ndjson(graph: AnnNet, dir_path):
    """Write nodes.ndjson, edges.ndjson, hyperedges.ndjson, slices.ndjson, edge_slices.ndjson.

    Each line is one JSON object. Lossless wrt to_json schema.
    """
    import os
    import json

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
        for eid, rec in _iter_edge_records(graph):
            S, T = _edge_endpoint_sets(rec)
            is_hyper = rec.etype == 'hyper'

            d = dict(edge_attrs.get(eid, {}))

            try:
                w = float(1.0 if rec.weight is None else rec.weight)
            except (TypeError, ValueError):
                w = 1.0
            try:
                directed = bool(_is_directed_eid(graph, eid))
            except (AttributeError, KeyError, TypeError, ValueError):
                directed = True

            if is_hyper:
                head_map = endpoint_coeff_map(d, '__source_attr', S) or dict.fromkeys(S or [], 1.0)
                tail_map = endpoint_coeff_map(d, '__target_attr', T) or dict.fromkeys(T or [], 1.0)
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
        for lid in graph.slices.list_slices(include_default=True):
            fl.write(json.dumps({'slice_id': lid}, ensure_ascii=False) + '\n')

    with open(f'{dir_path}/edge_slices.ndjson', 'w', encoding='utf-8') as fel:
        for lid in graph.slices.list_slices(include_default=True):
            for eid in graph.slices.get_slice_edges(lid):
                rec = {'slice_id': lid, 'edge_id': eid}
                try:
                    w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight', default=None)
                except TypeError:
                    w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight')
                if w is not None:
                    rec['weight'] = float(w)
                fel.write(json.dumps(rec, ensure_ascii=False) + '\n')
