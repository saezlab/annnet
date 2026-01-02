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

from typing import Any, Optional

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

try:
    import graph_tool.all as gt
except ImportError:
    gt = None
from ..core.graph import AnnNet
from ._utils import (
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_slices,
    _deserialize_VM,
    _df_to_rows,
    _rows_to_df,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_slices,
    _serialize_VM,
)

# Core adapter: to_graphtool


def to_graphtool(
    G: AnnNet,
    *,
    vertex_id_property: str = "id",
    edge_id_property: str = "id",
    weight_property: str = "weight",
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
        raise RuntimeError("graph-tool is not installed; cannot call to_graphtool")

    # 1) graph-tool AnnNet (directed flag from AnnNet)
    directed = bool(G.directed) if G.directed is not None else True
    gtG = gt.Graph(directed=directed)

    # 2) vertices (only type 'vertex')
    vmap = {}  # annnet_id -> gt.Vertex
    vp_id = gtG.new_vertex_property("string")

    for u, t in G.entity_types.items():
        if t != "vertex":
            continue
        v = gtG.add_vertex()
        vmap[u] = v
        vp_id[v] = str(u)

    gtG.vp[vertex_id_property] = vp_id

    # 3) edges (only binary edges between such vertices)
    ep_id = gtG.new_edge_property("string")
    ep_w = gtG.new_edge_property("double")

    # Prepare edge attribute properties if edge_attributes exists
    edge_props = {}
    if (
        hasattr(G, "edge_attributes")
        and G.edge_attributes is not None
        and G.edge_attributes.height > 0
    ):
        for col in G.edge_attributes.columns:
            if col in ("edge_id", "id", edge_id_property, weight_property):
                continue
            # Infer type from first non-null value
            sample = G.edge_attributes[col].drop_nulls()
            if len(sample) > 0:
                first_val = sample[0]
                if isinstance(first_val, (int, bool)):
                    edge_props[col] = gtG.new_edge_property("int")
                elif isinstance(first_val, float):
                    edge_props[col] = gtG.new_edge_property("double")
                else:
                    edge_props[col] = gtG.new_edge_property("string")

    for eid, defn in G.edge_definitions.items():
        try:
            u, v, etype = defn
        except ValueError:
            # weird or malformed definition; skip
            continue

        if u not in vmap or v not in vmap:
            # not a pure vertex-vertex edge; hyperedge/hybrid -> only in manifest
            continue

        e = gtG.add_edge(vmap[u], vmap[v])
        ep_id[e] = str(eid)
        ep_w[e] = float(G.edge_weights.get(eid, 1.0))

        # Set additional edge properties from edge_attributes
        if edge_props and hasattr(G, "edge_attributes"):
            id_col = "edge_id" if "edge_id" in G.edge_attributes.columns else "id"
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

    vert_rows = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    edge_rows = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    slice_rows = _df_to_rows(getattr(G, "slice_attributes", pl.DataFrame()))
    edge_slice_rows = _df_to_rows(getattr(G, "edge_slice_attributes", pl.DataFrame()))
    layer_attr_rows = _df_to_rows(getattr(G, "layer_attributes", pl.DataFrame()))

    # 5) slices internal structure (vertex/edge sets + attributes)
    slices_data = _serialize_slices(getattr(G, "_slices", {}))

    # 6) hyperedges and direction info
    hyperedges = dict(getattr(G, "hyperedge_definitions", {}))
    edge_directed = dict(getattr(G, "edge_directed", {}))
    edge_direction_policy = dict(getattr(G, "edge_direction_policy", {}))

    # 7) multilayer / Kivela metadata
    aspects = list(getattr(G, "aspects", []))
    elem_layers = dict(getattr(G, "elem_layers", {}))
    VM_serialized = _serialize_VM(getattr(G, "_VM", set()))
    edge_kind = dict(getattr(G, "edge_kind", {}))
    edge_layers_ser = _serialize_edge_layers(getattr(G, "edge_layers", {}))
    node_layer_attrs_ser = _serialize_node_layer_attrs(getattr(G, "_vertex_layer_attrs", {}))

    # aspect and layer-tuple level attributes (dicts)
    aspect_attrs = dict(getattr(G, "_aspect_attrs", {}))
    layer_tuple_attrs_ser = _serialize_layer_tuple_attrs(getattr(G, "_layer_attrs", {}))

    # 8) build manifest
    manifest = {
        "version": 1,
        "graph": {
            "directed": directed,
            "attributes": dict(getattr(G, "graph_attributes", {})),
        },
        "vertices": {
            "types": dict(G.entity_types),
            "attributes": vert_rows,
        },
        "edges": {
            "definitions": dict(G.edge_definitions),
            "weights": dict(G.edge_weights),
            "directed": edge_directed,
            "direction_policy": edge_direction_policy,
            "hyperedges": hyperedges,
            "attributes": edge_rows,
            "kivela": {
                "edge_kind": edge_kind,
                "edge_layers": edge_layers_ser,
            },
        },
        "slices": {
            "data": slices_data,
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
        },
        "multilayer": {
            "aspects": aspects,
            "aspect_attrs": aspect_attrs,
            "elem_layers": elem_layers,
            "VM": VM_serialized,
            "edge_kind": edge_kind,  # redundant but convenient
            "edge_layers": edge_layers_ser,
            "node_layer_attrs": node_layer_attrs_ser,
            "layer_tuple_attrs": layer_tuple_attrs_ser,
            "layer_attributes": layer_attr_rows,  # elementary 'aspect_layer' DF
        },
        "tables": {
            "vertex_attributes": vert_rows,
            "edge_attributes": edge_rows,
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
            "layer_attributes": layer_attr_rows,
        },
    }

    return gtG, manifest


# Core adapter: from_graphtool


def from_graphtool(
    gtG: gt.Graph,
    manifest: dict | None = None,
    *,
    vertex_id_property: str = "id",
    edge_id_property: str = "id",
    weight_property: str = "weight",
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
        raise RuntimeError("graph-tool is not installed; cannot call from_graphtool")

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
        G.add_vertex(vid)
        v_to_id[v] = vid

    # 2) edges
    ep_id = gtG.ep.get(edge_id_property, None)
    ep_w = gtG.ep.get(weight_property, None)

    for e in gtG.edges():
        u = v_to_id[e.source()]
        v = v_to_id[e.target()]
        eid = str(ep_id[e]) if ep_id is not None else None
        w = float(ep_w[e]) if ep_w is not None else 1.0
        G.add_edge(u, v, edge_id=eid, weight=w)

    # 3) no manifest -> projected graph only
    if manifest is None:
        return G

    # ----- graph-level attributes -----
    gmeta = manifest.get("graph", {})
    G.graph_attributes = dict(gmeta.get("attributes", {}))

    # ----- vertices -----
    vmeta = manifest.get("vertices", {})
    v_rows = vmeta.get("attributes", [])
    if v_rows:
        G.vertex_attributes = _rows_to_df(v_rows)
    v_types = vmeta.get("types", {})
    if v_types:
        G.entity_types.update(v_types)

    # ----- edges -----
    emeta = manifest.get("edges", {})
    e_rows = emeta.get("attributes", [])
    if e_rows:
        G.edge_attributes = _rows_to_df(e_rows)

    weights = emeta.get("weights", {})
    if weights:
        G.edge_weights.update(weights)

    e_directed = emeta.get("directed", {})
    if e_directed:
        G.edge_directed.update(e_directed)

    e_dir_policy = emeta.get("direction_policy", {})
    if e_dir_policy:
        G.edge_direction_policy.update(e_dir_policy)

    hyperedges = emeta.get("hyperedges", {})
    if hyperedges:
        G.hyperedge_definitions = dict(hyperedges)

    kivela_edge = emeta.get("kivela", {})
    if kivela_edge:
        ek = kivela_edge.get("edge_kind", {})
        el_ser = kivela_edge.get("edge_layers", {})
        if ek:
            G.edge_kind.update(ek)
        if el_ser:
            G.edge_layers.update(_deserialize_edge_layers(el_ser))

    # ----- slices -----
    smeta = manifest.get("slices", {})
    slices_data = smeta.get("data", {})
    if slices_data:
        G._slices.update(_deserialize_slices(slices_data))

    slice_rows = smeta.get("slice_attributes", [])
    if slice_rows:
        G.slice_attributes = _rows_to_df(slice_rows)

    edge_slice_rows = smeta.get("edge_slice_attributes", [])
    if edge_slice_rows:
        G.edge_slice_attributes = _rows_to_df(edge_slice_rows)

    # ----- multilayer / Kivela -----
    mm = manifest.get("multilayer", {})
    aspects = mm.get("aspects", [])
    elem_layers = mm.get("elem_layers", {})

    if aspects:
        G.aspects = list(aspects)
        G.elem_layers = dict(elem_layers or {})
        G._rebuild_all_layers_cache()

    aspect_attrs = mm.get("aspect_attrs", {})
    if aspect_attrs:
        G._aspect_attrs.update(aspect_attrs)

    VM_data = mm.get("VM", [])
    if VM_data:
        G._VM = _deserialize_VM(VM_data)

    # edge_kind / edge_layers again (if present under multilayer)
    ek2 = mm.get("edge_kind", {})
    el2_ser = mm.get("edge_layers", {})
    if ek2:
        G.edge_kind.update(ek2)
    if el2_ser:
        G.edge_layers.update(_deserialize_edge_layers(el2_ser))

    nl_attrs_ser = mm.get("node_layer_attrs", [])
    if nl_attrs_ser:
        G._vertex_layer_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

    layer_tuple_attrs_ser = mm.get("layer_tuple_attrs", [])
    if layer_tuple_attrs_ser:
        G._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

    layer_attr_rows = mm.get("layer_attributes", [])
    if layer_attr_rows:
        G.layer_attributes = _rows_to_df(layer_attr_rows)

    return G
