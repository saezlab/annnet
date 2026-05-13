"""
AnnNet-tool adapter for AnnNet AnnNet.

Provides:
    to_graphtool(G)      -> (gt.Graph, manifest_dict)
    from_graphtool(gtG, manifest=None) -> AnnNet

graph-tool only gets what it can natively represent:
    - vertices (type 'vertex')
    - simple binary edges with a global directedness + a 'weight' edge property
Everything else (hyperedges, per-edge directedness, multilayer, slices,
all attribute tables, etc.) is preserved in `manifest`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    import graph_tool.all as gt
except ImportError:
    gt = None

from ._common import (
    _rows_to_df,
    empty_dataframe,
    _iter_vertex_ids,
    dataframe_to_rows,
    _iter_edge_records,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)

if TYPE_CHECKING:
    from ..core import AnnNet


# Core adapter: to_graphtool


def _serialize_slice_data(graph: AnnNet) -> dict[str, dict]:
    return {
        slice_id: {
            'vertices': list(graph.slices.vertices(slice_id)),
            'edges': list(graph.slices.edges(slice_id)),
            'attributes': graph.slices.info(slice_id).get('attributes', {}),
        }
        for slice_id in graph.slices.list(include_default=True)
    }


def to_graphtool(
    G: AnnNet,
    *,
    vertex_id_property: str = 'id',
    edge_id_property: str = 'id',
    weight_property: str = 'weight',
) -> tuple[gt.Graph, dict]:
    """
    Convert an AnnNet AnnNet -> (graph_tool.AnnNet, manifest).

    graph-tool graph:
      - vertices: only entities with entity_types[u] == "vertex"
      - edges: only binary edges whose endpoints are such vertices
      - vertex property vp[vertex_id_property] = AnnNet vertex id
      - edge property   ep[edge_id_property]   = AnnNet edge id
      - edge property   ep[weight_property]    = edge weight (float)

    manifest:
      - preserves everything graph-tool cannot: hyperedges, slices,
        multilayer, and ALL attribute tables.
    """
    if gt is None:
        raise RuntimeError('graph-tool is not installed; cannot call to_graphtool')

    def _project_vertex_id(node):
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
            return node[0]
        return node

    # 1) graph-tool AnnNet (directed flag from AnnNet)
    directed = bool(G.directed) if G.directed is not None else True
    gtG = gt.Graph(directed=directed)

    # 2) vertices (only type 'vertex')
    vmap = {}  # annnet_id -> gt.Vertex
    vp_id = gtG.new_vertex_property('string')

    vertex_ids = list(_iter_vertex_ids(G))

    for u in vertex_ids:
        v = gtG.add_vertex()
        vmap[u] = v
        vp_id[v] = str(u)

    gtG.vp[vertex_id_property] = vp_id

    # 3) edges (only binary edges between such vertices)
    ep_id = gtG.new_edge_property('string')
    ep_w = gtG.new_edge_property('double')

    # Prepare edge attribute properties if edge_attributes exists
    edge_props = {}
    if (
        hasattr(G, 'edge_attributes')
        and G.edge_attributes is not None
        and G.edge_attributes.height > 0
    ):
        for col in G.edge_attributes.columns:
            if col in ('edge_id', 'id', edge_id_property, weight_property):
                continue
            # Infer type from first non-null value
            sample = G.edge_attributes[col].drop_nulls()
            if len(sample) > 0:
                first_val = sample[0]
                if isinstance(first_val, (int, bool)):
                    edge_props[col] = gtG.new_edge_property('int')
                elif isinstance(first_val, float):
                    edge_props[col] = gtG.new_edge_property('double')
                else:
                    edge_props[col] = gtG.new_edge_property('string')

    for eid, rec in _iter_edge_records(G):
        if rec.col_idx < 0 or rec.etype == 'hyper':
            continue
        u, v = _project_vertex_id(rec.src), _project_vertex_id(rec.tgt)
        if u not in vmap or v not in vmap:
            # not a pure vertex-vertex edge; hyperedge/hybrid -> only in manifest
            continue

        e = gtG.add_edge(vmap[u], vmap[v])
        ep_id[e] = str(eid)
        ep_w[e] = float(1.0 if rec.weight is None else rec.weight)

        # Set additional edge properties from edge_attributes
        if edge_props and hasattr(G, 'edge_attributes'):
            id_col = 'edge_id' if 'edge_id' in G.edge_attributes.columns else 'id'
            if id_col in G.edge_attributes.columns:
                row = G.edge_attributes.filter(G.edge_attributes[id_col] == eid)
                if row.height > 0:
                    for col, prop in edge_props.items():
                        if col in row.columns:
                            val = row[col][0]
                            if val is not None:
                                prop[e] = val

    gtG.ep[edge_id_property] = ep_id
    gtG.ep[weight_property] = ep_w

    # Register additional edge properties
    for col, prop in edge_props.items():
        gtG.ep[col] = prop

    # 4) attribute tables as rows (DF [DataFrame] -> list[dict])

    vert_rows = dataframe_to_rows(getattr(G, 'vertex_attributes', empty_dataframe({})))
    edge_rows = dataframe_to_rows(getattr(G, 'edge_attributes', empty_dataframe({})))
    slice_rows = dataframe_to_rows(getattr(G, 'slice_attributes', empty_dataframe({})))
    edge_slice_rows = dataframe_to_rows(getattr(G, 'edge_slice_attributes', empty_dataframe({})))
    layer_attr_rows = dataframe_to_rows(getattr(G, 'layer_attributes', empty_dataframe({})))

    # 5) slices internal structure (vertex/edge sets + attributes)
    slices_data = _serialize_slice_data(G)
    slice_membership, slice_weights = collect_slice_manifest(G)

    # 6) hyperedges and direction info
    hyperedges = {
        eid: (
            {'directed': True, 'head': list(rec.src or []), 'tail': list(rec.tgt or [])}
            if rec.tgt is not None
            else {'directed': False, 'members': list(rec.src or [])}
        )
        for eid, rec in _iter_edge_records(G)
        if rec.col_idx >= 0 and rec.etype == 'hyper'
    }
    edge_directed = {
        eid: bool(rec.directed)
        for eid, rec in _iter_edge_records(G)
        if rec.col_idx >= 0 and rec.directed is not None
    }
    edge_direction_policy = dict(getattr(G, 'edge_direction_policy', {}))

    multilayer_manifest = serialize_multilayer_manifest(
        G,
        table_to_rows=dataframe_to_rows,
        serialize_edge_layers=serialize_edge_layers,
    )

    # 8) build manifest
    manifest = {
        'version': 1,
        'graph': {
            'directed': directed,
            'attributes': dict(getattr(G, 'graph_attributes', {})),
        },
        'vertices': {
            'types': dict.fromkeys(_iter_vertex_ids(G), 'vertex'),
            'attributes': vert_rows,
        },
        'edges': {
            'definitions': {
                eid: (rec.src, rec.tgt, rec.etype)
                for eid, rec in _iter_edge_records(G)
                if rec.col_idx >= 0 and rec.etype != 'hyper'
            },
            'weights': {
                eid: rec.weight
                for eid, rec in _iter_edge_records(G)
                if rec.col_idx >= 0 and rec.weight is not None
            },
            'directed': edge_directed,
            'direction_policy': edge_direction_policy,
            'hyperedges': hyperedges,
            'attributes': edge_rows,
            'kivela': {
                'edge_kind': multilayer_manifest.get('edge_kind', {}),
                'edge_layers': multilayer_manifest.get('edge_layers', {}),
            },
        },
        'slices': {
            'data': slices_data,
            'memberships': slice_membership,
            'weights': slice_weights,
            'slice_attributes': slice_rows,
            'edge_slice_attributes': edge_slice_rows,
        },
        'multilayer': multilayer_manifest,
        'tables': {
            'vertex_attributes': vert_rows,
            'edge_attributes': edge_rows,
            'slice_attributes': slice_rows,
            'edge_slice_attributes': edge_slice_rows,
            'layer_attributes': layer_attr_rows,
        },
    }

    return gtG, manifest


# Core adapter: from_graphtool


def from_graphtool(
    gtG: gt.Graph,
    manifest: dict | None = None,
    *,
    vertex_id_property: str = 'id',
    edge_id_property: str = 'id',
    weight_property: str = 'weight',
) -> AnnNet:
    """
    Convert graph_tool.AnnNet (+ optional manifest) back into AnnNet AnnNet.

    - Vertices: from vertex property `vertex_id_property` if present, else numeric index.
    - Edges:    from edges in gtG; edge_id from edge property `edge_id_property` if present,
                else auto; weight from edge property `weight_property` if present, else 1.0.

    If `manifest` is provided, rehydrates:
      - all attribute tables (vertex/edge/slice/edge_slice/layer),
      - _slices internal structure,
      - hyperedges,
      - edge_directed and edge_direction_policy,
      - multilayer (aspects, elem_layers, VM, aspect attrs, layer-tuple attrs,
        edge_kind, edge_layers, node-layer attrs),
      - graph_attributes.
    """
    if gt is None:
        raise RuntimeError('graph-tool is not installed; cannot call from_graphtool')

    from ..core import AnnNet

    directed = bool(gtG.is_directed())
    G = AnnNet(directed=directed)

    # 1) vertices
    vp = gtG.vp.get(vertex_id_property, None)
    v_to_id: dict[Any, str] = {}

    for v in gtG.vertices():
        if vp is not None:
            vid = str(vp[v])
        else:
            vid = str(int(v))  # fallback: numeric id
        G.add_vertices(vid)
        v_to_id[v] = vid

    # 2) edges
    ep_id = gtG.ep.get(edge_id_property, None)
    ep_w = gtG.ep.get(weight_property, None)

    for e in gtG.edges():
        u = v_to_id[e.source()]
        v = v_to_id[e.target()]
        eid = str(ep_id[e]) if ep_id is not None else None
        w = float(ep_w[e]) if ep_w is not None else 1.0
        G.add_edges(u, v, edge_id=eid, weight=w)

    # 3) no manifest -> projected graph only
    if manifest is None:
        return G

    # ----- graph-level attributes -----
    gmeta = manifest.get('graph', {})
    G.graph_attributes = dict(gmeta.get('attributes', {}))

    # ----- vertices -----
    vmeta = manifest.get('vertices', {})
    v_rows = vmeta.get('attributes', [])
    if v_rows:
        G.vertex_attributes = _rows_to_df(v_rows)
    v_types = vmeta.get('types', {})
    if v_types:
        G.entity_types.update(v_types)

    # ----- edges -----
    emeta = manifest.get('edges', {})
    e_rows = emeta.get('attributes', [])
    if e_rows:
        G.edge_attributes = _rows_to_df(e_rows)

    weights = emeta.get('weights', {})

    e_directed = emeta.get('directed', {})
    if e_directed:
        for eid, val in e_directed.items():
            rec = G._edges.get(eid)
            if rec is not None:
                rec.directed = bool(val)

    e_dir_policy = emeta.get('direction_policy', {})
    if e_dir_policy:
        G.edge_direction_policy.update(e_dir_policy)

    hyperedges = emeta.get('hyperedges', {})
    if hyperedges:
        hyperedge_bulk = []
        for eid, meta in hyperedges.items():
            rec = G._edges.get(eid)
            if rec is None:
                payload = {
                    'edge_id': eid,
                    'edge_directed': bool(meta.get('directed', False)),
                    'weight': float(weights.get(eid, 1.0)),
                }
                if meta.get('directed'):
                    payload['head'] = list(meta.get('head', []))
                    payload['tail'] = list(meta.get('tail', []))
                else:
                    payload['members'] = list(meta.get('members', []))
                hyperedge_bulk.append(payload)
                continue
            rec.etype = 'hyper'
            if meta.get('directed'):
                rec.src = list(meta.get('head', []))
                rec.tgt = list(meta.get('tail', []))
                rec.directed = True
            else:
                rec.src = list(meta.get('members', []))
                rec.tgt = None
                rec.directed = False
        if hyperedge_bulk:
            G.add_hyperedges_bulk(hyperedge_bulk)

    kivela_edge = emeta.get('kivela', {})
    if kivela_edge:
        ek = kivela_edge.get('edge_kind', {})
        el_ser = kivela_edge.get('edge_layers', {})
        if ek:
            for eid, kind in ek.items():
                rec = G._edges.get(eid)
                if rec is None:
                    continue
                if kind == 'hyper':
                    rec.etype = 'hyper'
                else:
                    rec.ml_kind = kind
        if el_ser:
            G.edge_layers.update(deserialize_edge_layers(el_ser))

    # ----- slices -----
    smeta = manifest.get('slices', {})

    slice_rows = smeta.get('slice_attributes', [])
    if slice_rows:
        G.slice_attributes = _rows_to_df(slice_rows)

    edge_slice_rows = smeta.get('edge_slice_attributes', [])
    if edge_slice_rows:
        G.edge_slice_attributes = _rows_to_df(edge_slice_rows)

    if smeta.get('data'):
        existing_slices = set(G.slices.list(include_default=True))
        for slice_id, info in smeta['data'].items():
            if slice_id not in existing_slices:
                G.slices.add(slice_id, **(info.get('attributes') or {}))
                existing_slices.add(slice_id)
            for vertex_id in info.get('vertices', []):
                G.slices.add_vertex_to_slice(slice_id, vertex_id)
    restore_slice_manifest(
        G,
        smeta.get('memberships')
        or {
            slice_id: info.get('edges', []) for slice_id, info in (smeta.get('data') or {}).items()
        },
        smeta.get('weights') or {},
    )

    # ----- multilayer / Kivela -----
    restore_multilayer_manifest(
        G,
        manifest.get('multilayer', {}),
        rows_to_table=_rows_to_df,
        deserialize_edge_layers=deserialize_edge_layers,
    )

    return G
