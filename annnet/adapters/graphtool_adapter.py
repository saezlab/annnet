"""
AnnNet-tool adapter for AnnNet AnnNet.

Provides:
    to_graphtool(G)      -> (gt.Graph, manifest_dict)
    from_graphtool(gtG, manifest=None) -> AnnNet

graph-tool only gets what it can natively represent:
    - nodes (type 'node')
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

from ..core import _structure
from ._common import (
    stored_key,
    _rows_to_df,
    _iter_node_ids,
    empty_dataframe,
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
            'nodes': list(graph.slices.nodes(slice_id)),
            'edges': list(graph.slices.edges(slice_id)),
            'attributes': graph.slices.info(slice_id).get('attributes', {}),
        }
        for slice_id in graph.slices.list(include_default=True)
    }


def to_graphtool(
    G: AnnNet,
    *,
    node_id_property: str = 'id',
    edge_id_property: str = 'id',
    weight_property: str = 'weight',
) -> tuple[gt.Graph, dict]:
    """
    Convert an AnnNet AnnNet -> (graph_tool.AnnNet, manifest).

    graph-tool graph:
      - nodes: only entities whose kind is a node
      - edges: only binary edges whose endpoints are such nodes
      - node property vp[node_id_property] = AnnNet node id
      - edge property   ep[edge_id_property]   = AnnNet edge id
      - edge property   ep[weight_property]    = edge weight (float)

    manifest:
      - preserves everything graph-tool cannot: hyperedges, slices,
        multilayer, and ALL attribute tables.
    """
    if gt is None:
        raise RuntimeError('graph-tool is not installed; cannot call to_graphtool')

    def _project_node_id(node):
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
            return node[0]
        return node

    # 1) graph-tool AnnNet (directed flag from AnnNet)
    directed = bool(G.directed) if G.directed is not None else True
    gtG = gt.Graph(directed=directed)

    # 2) nodes (only type 'node') — list materialised once and reused
    # for the manifest's 'nodes.types' section too.
    vmap = {}  # annnet_id -> gt.Node
    vp_id = gtG.new_vertex_property('string')

    node_ids = list(_iter_node_ids(G))

    for u in node_ids:
        v = gtG.add_vertex()
        vmap[u] = v
        vp_id[v] = str(u)

    gtG.vp[node_id_property] = vp_id

    # 3) edges (only binary edges between such nodes)
    ep_id = gtG.new_edge_property('string')
    ep_w = gtG.new_edge_property('double')

    # Pre-load _edge_table as a dict[eid -> dict] in one pass, instead of
    # a per-edge polars filter (the previous implementation called
    # `G._edge_table.filter(...)` once per edge — catastrophic at scale).
    edge_attr_rows: dict[str, dict] = {}
    edge_attr_cols: list[str] = []
    if hasattr(G, '_edge_table') and G._edge_table is not None and G._edge_table.height > 0:
        ea_id_col = 'edge_id' if 'edge_id' in G._edge_table.columns else 'id'
        skip_cols = {'edge_id', 'id', edge_id_property, weight_property}
        edge_attr_cols = [c for c in G._edge_table.columns if c not in skip_cols]
        for row in dataframe_to_rows(G._edge_table):
            rid = row.get(ea_id_col)
            if rid is not None:
                edge_attr_rows[str(rid)] = row

    # Prepare typed edge properties (typed by first non-null sample)
    edge_props = {}
    for col in edge_attr_cols:
        sample = G._edge_table[col].drop_nulls()
        if len(sample) > 0:
            first_val = sample[0]
            if isinstance(first_val, (int, bool)):
                edge_props[col] = gtG.new_edge_property('int')
            elif isinstance(first_val, float):
                edge_props[col] = gtG.new_edge_property('double')
            else:
                edge_props[col] = gtG.new_edge_property('string')

    # Single pass over edge records: build the graph_tool edges AND every
    # manifest section that depends on edge records (definitions, weights,
    # directed, hyperedges). Previously this was 5 separate passes.
    edges_definitions: dict = {}
    edges_weights: dict = {}
    edge_directed: dict = {}
    hyperedges: dict = {}

    for eid, rec in _iter_edge_records(G):
        if rec.etype == 'hyper':
            hyperedges[eid] = (
                {'directed': True, 'head': list(rec.src or []), 'tail': list(rec.tgt or [])}
                if rec.tgt is not None
                else {'directed': False, 'members': list(rec.src or [])}
            )
            if rec.directed is not None:
                edge_directed[eid] = bool(rec.directed)
            if rec.weight is not None:
                edges_weights[eid] = rec.weight
            continue

        # binary edge → also lands in gt graph and manifest 'definitions'
        edges_definitions[eid] = (rec.src, rec.tgt, rec.etype)
        if rec.directed is not None:
            edge_directed[eid] = bool(rec.directed)
        if rec.weight is not None:
            edges_weights[eid] = rec.weight

        u, v = _project_node_id(rec.src), _project_node_id(rec.tgt)
        if u not in vmap or v not in vmap:
            continue

        e = gtG.add_edge(vmap[u], vmap[v])
        ep_id[e] = str(eid)
        ep_w[e] = float(1.0 if rec.weight is None else rec.weight)

        eattr = edge_attr_rows.get(eid)
        if eattr and edge_props:
            for col, prop in edge_props.items():
                val = eattr.get(col)
                if val is not None:
                    prop[e] = val

    gtG.ep[edge_id_property] = ep_id
    gtG.ep[weight_property] = ep_w
    for col, prop in edge_props.items():
        gtG.ep[col] = prop

    # 4) attribute tables as rows (DF [DataFrame] -> list[dict])

    vert_rows = dataframe_to_rows(getattr(G, '_node_table', empty_dataframe({})))
    edge_rows = dataframe_to_rows(getattr(G, '_edge_table', empty_dataframe({})))
    slice_rows = dataframe_to_rows(getattr(G, 'slice_attributes', empty_dataframe({})))
    edge_slice_rows = dataframe_to_rows(getattr(G, 'edge_slice_attributes', empty_dataframe({})))
    layer_attr_rows = dataframe_to_rows(getattr(G, 'layer_attributes', empty_dataframe({})))

    # 5) slices internal structure (node/edge sets + attributes)
    slices_data = _serialize_slice_data(G)
    slice_membership, slice_weights = collect_slice_manifest(G)

    edge_direction_policy = dict(getattr(G, 'edge_direction_policy', {}))

    multilayer_manifest = serialize_multilayer_manifest(
        G,
        table_to_rows=dataframe_to_rows,
        serialize_edge_layers=serialize_edge_layers,
    )

    # 8) build manifest — all dicts already computed in the single pass above
    manifest = {
        'version': 1,
        'graph': {
            'directed': directed,
            'attributes': dict(getattr(G, 'graph_attributes', {})),
        },
        'nodes': {
            'types': dict.fromkeys(node_ids, 'node'),
            'attributes': vert_rows,
        },
        'edges': {
            'definitions': edges_definitions,
            'weights': edges_weights,
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
            '_node_table': vert_rows,
            '_edge_table': edge_rows,
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
    node_id_property: str = 'id',
    edge_id_property: str = 'id',
    weight_property: str = 'weight',
) -> AnnNet:
    """
    Convert graph_tool.AnnNet (+ optional manifest) back into AnnNet AnnNet.

    - Nodes: from node property `node_id_property` if present, else numeric index.
    - Edges:    from edges in gtG; edge_id from edge property `edge_id_property` if present,
                else auto; weight from edge property `weight_property` if present, else 1.0.

    If `manifest` is provided, rehydrates:
      - all attribute tables (node/edge/slice/edge_slice/layer),
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

    # 1) nodes — bulk collect, single insert
    vp = gtG.vp.get(node_id_property, None)
    v_to_id: dict[Any, str] = {}
    node_buf: list[str] = []
    for v in gtG.vertices():
        vid = str(vp[v]) if vp is not None else str(int(v))
        v_to_id[v] = vid
        node_buf.append(vid)
    if node_buf:
        G._add_nodes_bulk([{'node_id': v} for v in node_buf])

    # 2) edges — bulk collect, single insert
    ep_id = gtG.ep.get(edge_id_property, None)
    ep_w = gtG.ep.get(weight_property, None)

    edges_bulk: list[dict] = []
    for e in gtG.edges():
        u = v_to_id[e.source()]
        v = v_to_id[e.target()]
        eid = str(ep_id[e]) if ep_id is not None else None
        w = float(ep_w[e]) if ep_w is not None else 1.0
        payload = {'source': u, 'target': v, 'weight': w}
        if eid is not None:
            payload['edge_id'] = eid
        edges_bulk.append(payload)
    if edges_bulk:
        G._add_edges_bulk(edges_bulk, default_edge_directed=directed)

    # 3) no manifest -> projected graph only
    if manifest is None:
        return G

    # ----- graph-level attributes -----
    gmeta = manifest.get('graph', {})
    G.graph_attributes = dict(gmeta.get('attributes', {}))

    # ----- nodes -----
    vmeta = stored_key(manifest, 'nodes', {})
    v_rows = vmeta.get('attributes', [])
    if v_rows:
        G._node_table = _rows_to_df(v_rows)
    v_types = vmeta.get('types', {})
    if v_types:
        G._set_entity_kinds_by_id(v_types)

    # ----- edges -----
    emeta = manifest.get('edges', {})
    e_rows = emeta.get('attributes', [])
    if e_rows:
        G._edge_table = _rows_to_df(e_rows)

    weights = emeta.get('weights', {})

    e_directed = emeta.get('directed', {})
    if e_directed:
        for eid, val in e_directed.items():
            if _structure.has_edge(G, eid):
                G._set_edge_field(eid, 'directed', bool(val))

    e_dir_policy = emeta.get('direction_policy', {})
    if e_dir_policy:
        G.edge_direction_policy.update(e_dir_policy)

    hyperedges = emeta.get('hyperedges', {})
    if hyperedges:
        hyperedge_bulk = []
        for eid, meta in hyperedges.items():
            if not _structure.has_edge(G, eid):
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
            if meta.get('directed'):
                G._set_hyperedge_members(eid, head=meta.get('head', []), tail=meta.get('tail', []))
            else:
                G._set_hyperedge_members(eid, members=meta.get('members', []))
        if hyperedge_bulk:
            G.add_hyperedges_bulk(hyperedge_bulk)

    kivela_edge = emeta.get('kivela', {})
    if kivela_edge:
        ek = kivela_edge.get('edge_kind', {})
        el_ser = kivela_edge.get('edge_layers', {})
        if ek:
            for eid, kind in ek.items():
                if _structure.has_edge(G, eid):
                    G.edge_kind[eid] = kind
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
            for node_id in stored_key(info, 'nodes', []):
                G.slices.add_node_to_slice(slice_id, node_id)
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
