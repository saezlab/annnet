"""
CX2 Adapter for AnnNet Graph.

Provides:
    to_cx2(G)        -> List[Dict[str, Any]] (CX2 JSON object)
    from_cx2(cx2_data) -> Graph

This adapter maps AnnNet Graphs to the Cytoscape Exchange (CX2) format.
It creates a lossless representation by serializing complex features
(hyperedges, slices, multilayer) into a 'manifest' stored within the
CX2 'networkAttributes'.
"""

from __future__ import annotations
import json
from typing import Dict, Any, List
import polars as pl

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import Graph
from ._utils import (
    _serialize_edge_layers,
    _deserialize_edge_layers,
    _serialize_VM,
    _deserialize_VM,
    _serialize_node_layer_attrs,
    _deserialize_node_layer_attrs,
    _serialize_slices,
    _deserialize_slices,
    _df_to_rows,
    _serialize_layer_tuple_attrs,
    _deserialize_layer_tuple_attrs,
)

# --- Helpers ---

def _rows_to_df(rows):
    import polars as pl

    if not rows:
        return pl.DataFrame()

    # --- 1) Normalize all rows to full schema ---
    keys = set().union(*(r.keys() for r in rows))
    norm = [{k: r.get(k, None) for k in keys} for r in rows]

    # --- 2) Let Polars infer schema on the first few rows ---
    # but allow nulls / mixed types
    try:
        return pl.DataFrame(norm, infer_schema_length=1000)
    except Exception:
        # --- 3) Fallback: cast everything to string but KEEP columns ---
        # This preserves attributes instead of dropping them
        return pl.DataFrame(
            [{k: (str(v) if v is not None else None) for k,v in r.items()} for r in norm]
        )

def _map_pl_to_cx2_type(dtype: pl.DataType) -> str:
    """Map Polars DataType to CX2 attribute data type string."""
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.UInt8, pl.UInt16, pl.UInt32):
        return "integer"
    elif dtype in (pl.Int64, pl.UInt64):
        return "long"
    elif dtype in (pl.Float32, pl.Float64):
        return "double"
    elif dtype == pl.Boolean:
        return "boolean"
    elif dtype in (pl.Utf8, pl.String, pl.Object):
        return "string"
    elif isinstance(dtype, pl.List):
        inner = _map_pl_to_cx2_type(dtype.inner)
        return f"list_of_{inner}"
    else:
        # Fallback for unknown types
        return "string"

def _cx2_collect_reified(aspects):
    """
    Detect reified hyperedges from CX2 nodes + edges.

    Returns:
      hyperdefs: list of (eid, directed, head_map, tail_map, attrs, he_node_id)
      membership_edges: set of edge-ids in CX2 that belong to hyperedge membership structure.
    """
    nodes = aspects.get("nodes", [])
    edges = aspects.get("edges", [])

    he_nodes = {}
    for n in nodes:
        v = n.get("v", {})
        if v.get("is_hyperedge", False):
            eid = v.get("eid")
            if eid is None:
                continue
            he_nodes[n["id"]] = (eid, v)

    if not he_nodes:
        return [], set()

    # Build adjacency around hyperedge nodes
    hyperdefs = []
    membership_edges = set()

    for he_id, (eid, attrs) in he_nodes.items():
        head_map = {}
        tail_map = {}
        members_map = {}

        for e in edges:
            u = e["s"]
            v = e["t"]
            vid = e["id"]
            ev = e.get("v", {})

            if u != he_id and v != he_id:
                continue

            membership_edges.add(vid)
            other = v if u == he_id else u
            role = ev.get("role", None)
            coeff = float(ev.get("weight", ev.get("coeff", 1.0)))

            if role == "head":
                head_map[other] = coeff
            elif role == "tail":
                tail_map[other] = coeff
            else:
                # undirected membership
                head_map[other] = coeff
                tail_map[other] = coeff

        # Determine directedness
        if any(k for k in (head_map or {})) or any(k for k in (tail_map or {})):
            directed = True if any(ev.get("role") in ("head", "tail") for ev in attrs.values()) else False
        else:
            directed = False

        hyperdefs.append((eid, directed, head_map, tail_map, attrs, he_id))

    return hyperdefs, membership_edges

def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    else:
        return obj

# --- Core Adapter: to_cx2 ---

def to_cx2(G: Graph, *, hyperedges="skip") -> List[Dict[str, Any]]:
    """
    Convert an AnnNet Graph -> CX2 compliant JSON list.

    The output is a list of aspect dictionaries (CX2 format).
    Complex AnnNet features (hyperedges, slices, etc.) are serialized
    into a JSON string and stored in 'networkAttributes' under '__AnnNet_Manifest__'.
    """
    
    # 1. Prepare Manifest (Lossless storage of complex features, same logci as from_graphtool)
    vert_rows = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    edge_rows = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    slice_rows = _df_to_rows(getattr(G, "slice_attributes", pl.DataFrame()))
    edge_slice_rows = _df_to_rows(getattr(G, "edge_slice_attributes", pl.DataFrame()))
    layer_attr_rows = _df_to_rows(getattr(G, "layer_attributes", pl.DataFrame()))

    manifest = {
        "version": 1,
        "graph": {
            "directed": bool(G.directed) if G.directed is not None else True,
            "attributes": dict(getattr(G, "graph_attributes", {})),
        },
        "vertices": {
            "types": dict(G.entity_types),
            "attributes": vert_rows,
        },
        "edges": {
            "definitions": dict(G.edge_definitions),
            "weights": dict(G.edge_weights),
            "directed": dict(getattr(G, "edge_directed", {})),
            "direction_policy": dict(getattr(G, "edge_direction_policy", {})),
            "hyperedges": dict(getattr(G, "hyperedge_definitions", {})),
            "attributes": edge_rows,
            "kivela": {
                "edge_kind": dict(getattr(G, "edge_kind", {})),
                "edge_layers": _serialize_edge_layers(getattr(G, "edge_layers", {})),
            },
        },
        "slices": {
            "data": _serialize_slices(getattr(G, "_slices", {})),
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
        },
        "multilayer": {
            "aspects": list(getattr(G, "aspects", [])),
            "aspect_attrs": dict(getattr(G, "_aspect_attrs", {})),
            "elem_layers": dict(getattr(G, "elem_layers", {})),
            "VM": _serialize_VM(getattr(G, "_VM", set())),
            "edge_kind": dict(getattr(G, "edge_kind", {})),
            "edge_layers": _serialize_edge_layers(getattr(G, "edge_layers", {})),
            "node_layer_attrs": _serialize_node_layer_attrs(getattr(G, "_vertex_layer_attrs", {})),
            "layer_tuple_attrs": _serialize_layer_tuple_attrs(getattr(G, "_layer_attrs", {})),
            "layer_attributes": layer_attr_rows,
        },
        "tables": {
            "vertex_attributes": vert_rows,
            "edge_attributes": edge_rows,
            "slice_attributes": slice_rows,
            "edge_slice_attributes": edge_slice_rows,
            "layer_attributes": layer_attr_rows,
        },
    }
    manifest["edges"]["expanded"] = {}

    # 2. Build Core CX2 Aspects
    
    # ID Mapping: AnnNet ID (str/int) -> CX2 ID (int)
    # CX2 IDs must be integers.
    node_map: Dict[Any, int] = {}
    cx_nodes: List[Dict[str, Any]] = []
    cx_edges: List[Dict[str, Any]] = []
    
    # -- Nodes --
    # We only map entities of type 'vertex' to visual nodes.
    current_node_id = 0
    
    # Pre-fetch attribute data for fast lookup
    v_attrs_df = getattr(G, "vertex_attributes", pl.DataFrame())
    v_attrs_map = {str(r.get("id")): r for r in vert_rows} if not v_attrs_df.is_empty() else {}

    for uid, utype in G.entity_types.items():
        if utype != "vertex":
            continue
            
        cx_id = current_node_id
        current_node_id += 1
        node_map[uid] = cx_id
        
        # Build node object
        n_obj = {
            "id": cx_id,
            "v": {"name": str(uid)} # 'name' is standard in Cytoscape
        }
        
        # Attach attributes if present
        if str(uid) in v_attrs_map:
            # Copy attributes, excluding 'id' to avoid redundancy
            attrs = v_attrs_map[str(uid)].copy()
            attrs.pop("id", None)
            n_obj["v"].update(attrs)
            
        cx_nodes.append(n_obj)

    # -- Edges --
    # Only binary edges between mapped vertices
    current_edge_id = 0
    e_attrs_df = getattr(G, "edge_attributes", pl.DataFrame())
    
    # Create lookup for edge attributes (handle both 'edge_id' and 'id')
    e_attrs_map = {}
    if not e_attrs_df.is_empty():
        id_col = "edge_id" if "edge_id" in e_attrs_df.columns else "id"
        for r in edge_rows:
            if id_col in r:
                e_attrs_map[str(r[id_col])] = r

    for eid, defn in G.edge_definitions.items():
        is_hyper = eid in G.hyperedge_definitions

        # --- Hyperedge handling ---
        if is_hyper:
            if hyperedges == "skip":
                continue

            hdef = G.hyperedge_definitions[eid]
            directed = hdef.get("directed", True)
            if directed:
                S = set(hdef["head"])
                T = set(hdef["tail"])
            else:
                members = set(hdef["members"])
                S = members
                T = members

            if hyperedges == "expand":
                exp_entry = {
                    "mode": "directed" if directed else "undirected",
                    "tail": list(T) if directed else None,
                    "head": list(S) if directed else None,
                    "members": list(S | T) if not directed else None,
                    "expanded_edges": []
                }                
                members = S | T
                if directed:
                    # tail → head cartesian
                    for u in T:
                        for v in S:
                            exp_entry["expanded_edges"].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            cx_edges.append({
                                "id": cx_eid,
                                "s": node_map[u],
                                "t": node_map[v],
                                "v": {
                                    "interaction": str(eid),
                                    "weight": float(G.edge_weights.get(eid, 1.0)),
                                    **(e_attrs_map.get(str(eid), {}))
                                }
                            })
                else:
                    # undirected clique
                    mem = list(members)
                    for i in range(len(mem)):
                        for j in range(i+1, len(mem)):
                            u, v = mem[i], mem[j]
                            exp_entry["expanded_edges"].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            cx_edges.append({
                                "id": cx_eid,
                                "s": node_map[u],
                                "t": node_map[v],
                                "v": {
                                    "interaction": str(eid),
                                    "weight": float(G.edge_weights.get(eid, 1.0)),
                                    **(e_attrs_map.get(str(eid), {}))
                                }
                            })
                manifest["edges"]["expanded"][str(eid)] = exp_entry
                continue

            if hyperedges == "reify":
                he_cx_id = current_node_id
                current_node_id += 1

                # filter edge attrs so we don't leak edge_id/id onto the node
                he_attrs = e_attrs_map.get(str(eid), {}).copy()
                he_attrs.pop("edge_id", None)
                he_attrs.pop("id", None)

                he_node = {
                    "id": he_cx_id,
                    "v": {
                        "name": f"hyperedge::{eid}",
                        "is_hyperedge": True,
                        "eid": str(eid),
                        **he_attrs,
                    },
                }
                cx_nodes.append(he_node)

                weight = float(G.edge_weights.get(eid,1.0))

                if directed:
                    # tail → HE
                    for u in T:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append({
                            "id": cx_eid,
                            "s": node_map[u],
                            "t": he_cx_id,
                            "v": {
                                "interaction": f"{eid}::tail",
                                "role": "tail",
                                "weight": weight,
                            }
                        })
                    # HE → head
                    for v in S:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append({
                            "id": cx_eid,
                            "s": he_cx_id,
                            "t": node_map[v],
                            "v": {
                                "interaction": f"{eid}::head",
                                "role": "head",
                                "weight": weight,
                            }
                        })

                else:
                    # undirected: HE ↔ members
                    members = S | T
                    for u in members:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append({
                            "id": cx_eid,
                            "s": node_map[u],
                            "t": he_cx_id,
                            "v": {
                                "interaction": f"{eid}::member",
                                "role": "member",
                                "weight": weight,
                            }
                        })
                continue

            # unknown hyperedge mode → skip
            continue

        try:
            u, v, _ = defn
        except Exception:
            continue

        if u not in node_map or v not in node_map:
            continue         

        cx_u = node_map[u]
        cx_v = node_map[v]
        cx_eid = current_edge_id
        current_edge_id += 1

        e_obj = {
            "id": cx_eid,
            "s": cx_u,
            "t": cx_v,
            "v": {
                "interaction": str(eid),
                "weight": float(G.edge_weights.get(eid, 1.0))
            }
        }

        # Attach attributes
        if str(eid) in e_attrs_map:
            attrs = e_attrs_map[str(eid)].copy()
            # Remove redundant keys
            attrs.pop("edge_id", None)
            attrs.pop("id", None)
            e_obj["v"].update(attrs)

        cx_edges.append(e_obj)

    # 3. Attribute Declarations
    # CX2 requires explicit typing for attributes.
    attr_decls = {"nodes": {}, "edges": {}, "networkAttributes": {}}

    # Define Node Attributes
    if not v_attrs_df.is_empty():
        for col, dtype in v_attrs_df.schema.items():
            if col == "id": continue
            attr_decls["nodes"][col] = {"d": _map_pl_to_cx2_type(dtype)}

    attr_decls["nodes"]["name"] = {"d": "string"}  # Always added
    attr_decls["nodes"]["is_hyperedge"] = {"d": "boolean"}
    attr_decls["nodes"]["eid"] = {"d": "string"}
    attr_decls["nodes"]["tag"] = {"d": "string"}
    attr_decls["nodes"]["reaction"] = {"d": "string"}

    # Define Edge Attributes
    if not e_attrs_df.is_empty():
        id_col = "edge_id" if "edge_id" in e_attrs_df.columns else "id"
        for col, dtype in e_attrs_df.schema.items():
            if col == id_col: continue
            attr_decls["edges"][col] = {"d": _map_pl_to_cx2_type(dtype)}
    attr_decls["edges"]["interaction"] = {"d": "string"}
    attr_decls["edges"]["weight"] = {"d": "double"}
    attr_decls["edges"]["edge_id"] = {"d": "string"}
    attr_decls["edges"]["role"] = {"d": "string"}

    # Define Network Attributes
    # We store the manifest as a JSON string to ensure compatibility
    attr_decls["networkAttributes"]["__AnnNet_Manifest__"] = {"d": "string"}
    attr_decls["networkAttributes"]["name"] = {"d": "string"}
    attr_decls["networkAttributes"]["directed"] = {"d": "boolean"}

    # 4. Construct Final CX2 List
    cx2 = [
            {"CXVersion": "2.0", "hasFragments": False},
            {"metaData": [
                {"name": "attributeDeclarations", "elementCount": 1},
                {"name": "networkAttributes", "elementCount": 1},
                {"name": "nodes", "elementCount": len(cx_nodes)},
                {"name": "edges", "elementCount": len(cx_edges)}
            ]},
            {"attributeDeclarations": [attr_decls]},
            {"networkAttributes": [{
                "name": "AnnNet Export",
                "directed": bool(G.directed) if G.directed is not None else True,
                "__AnnNet_Manifest__": json.dumps(_jsonify(manifest))
            }]},
            {"nodes": cx_nodes},
            {"edges": cx_edges}, 
            {"status": [{"success": True}]} 
        ]
        
    return cx2
    
# --- Core Adapter: from_cx2 ---

def from_cx2(cx2_data, *, hyperedges="manifest"):
    """
    Fully robust CX2 → AnnNet importer.
    Supports:
      - manifest reconstruction (full fidelity)
      - reified hyperedges
      - expanded hyperedges
      - Cytoscape-edited files
      - sparse attribute tables
      - arbitrary attribute modifications
    """

    # -------------------------------------------------------------
    # Small helper: normalize rows so Polars won't crash
    # -------------------------------------------------------------
    def _normalize_rows(rows):
        if not rows:
            return []
        keys = set().union(*(r.keys() for r in rows))
        return [{k: r.get(k, None) for k in keys} for r in rows]

    # -------------------------------------------------------------
    # Load file or JSON string
    # -------------------------------------------------------------
    import os, json
    if isinstance(cx2_data, str):
        if os.path.exists(cx2_data):
            with open(cx2_data, "r") as f:
                cx2_data = json.load(f)
        else:
            try:
                cx2_data = json.loads(cx2_data)
            except Exception:
                raise ValueError("Invalid CX2 string or file")

    # -------------------------------------------------------------
    # Parse aspects into a dict
    # -------------------------------------------------------------
    aspects = {}
    for item in cx2_data:
        if not item:
            continue
        key = list(item.keys())[0]

        if key in ("CXVersion", "metaData", "status"):
            aspects[key] = item[key]
        else:
            aspects.setdefault(key, []).extend(item[key])

    # -------------------------------------------------------------
    # Extract networkAttributes + manifest JSON
    # -------------------------------------------------------------
    net_attrs = {}
    for na in aspects.get("networkAttributes", []):
        net_attrs.update(na)

    manifest_str = net_attrs.get("__AnnNet_Manifest__")
    manifest = None
    if manifest_str:
        try:
            manifest = json.loads(manifest_str)
        except:
            manifest = None

    # -------------------------------------------------------------
    # Construct Graph
    # -------------------------------------------------------------
    from annnet.core.graph import Graph
    G = Graph()

    # =====================================================================
    #                        PATH A: MANIFEST RECONSTRUCTION
    # =====================================================================
    if manifest and hyperedges in ("manifest", "reified"):

        # --- Base graph attrs ---
        gmeta = manifest.get("graph", {})
        G.directed = gmeta.get("directed", True)
        G.graph_attributes = dict(gmeta.get("attributes", {}))

        # --- Vertices ---
        vmeta = manifest.get("vertices", {})
        v_rows = _normalize_rows(vmeta.get("attributes", []))
        if v_rows:
            G.vertex_attributes = _rows_to_df(v_rows)
        if vmeta.get("types"):
            G.entity_types.update(vmeta["types"])

        # --- Edges + hyperedges ---
        emeta = manifest.get("edges", {})

        # edge_attributes
        e_rows = _normalize_rows(emeta.get("attributes", []))
        if e_rows:
            G.edge_attributes = _rows_to_df(e_rows)

        # weights, directed flags, definitions
        if emeta.get("weights"): G.edge_weights.update(emeta["weights"])
        if emeta.get("directed"): G.edge_directed.update(emeta["directed"])
        if emeta.get("definitions"): G.edge_definitions = dict(emeta["definitions"])
        if emeta.get("direction_policy"): G.edge_direction_policy.update(emeta["direction_policy"])

        # hyperedge definitions
        if emeta.get("hyperedges"):
            # these are raw dicts: convert lists back to sets
            fixed = {}
            for eid, info in emeta["hyperedges"].items():
                if info.get("directed"):
                    fixed[eid] = {
                        "directed": True,
                        "head": set(info["head"]),
                        "tail": set(info["tail"]),
                    }
                else:
                    fixed[eid] = {
                        "directed": False,
                        "members": set(info["members"]),
                    }
            G.hyperedge_definitions = fixed

        # --- Expanded hyperedges (if present) ---
        exp = emeta.get("expanded", {})
        for eid, info in exp.items():
            directed = info.get("mode") == "directed"
            if directed:
                G.add_hyperedge(
                    head=info.get("head", []),
                    tail=info.get("tail", []),
                    edge_id=eid,
                    edge_directed=True
                )
            else:
                G.add_hyperedge(
                    members=info.get("members", []),
                    edge_id=eid,
                    edge_directed=False
                )

        # --- Kivela ---
        kiv = emeta.get("kivela", {})
        if kiv.get("edge_kind"): G.edge_kind.update(kiv["edge_kind"])
        if kiv.get("edge_layers"): 
            G.edge_layers.update(kiv["edge_layers"])

        # --- Slices ---
        smeta = manifest.get("slices", {})
        if smeta.get("data"):
            for sname, sdata in smeta["data"].items():
                verts = set(sdata.get("vertices", []))
                edgs = set(sdata.get("edges", []))
                attrs = dict(sdata.get("attributes", {}))

                G._slices[sname] = {
                    "vertices": verts,
                    "edges": edgs,
                    "attributes": attrs,
                }
        if smeta.get("slice_attributes"):
            G.slice_attributes = _rows_to_df(_normalize_rows(smeta["slice_attributes"]))
        if smeta.get("edge_slice_attributes"):
            G.edge_slice_attributes = _rows_to_df(_normalize_rows(smeta["edge_slice_attributes"]))

        # --- Multilayer ---
        mm = manifest.get("multilayer", {})
        if mm.get("aspects"): G.aspects = mm["aspects"]
        if mm.get("elem_layers"): G.elem_layers = dict(mm["elem_layers"])
        if mm.get("aspect_attrs"):
            G._aspect_attrs = mm["aspect_attrs"]
        if mm.get("node_layer_attrs"):
            G._vertex_layer_attrs = mm["node_layer_attrs"]
        if mm.get("layer_tuple_attrs"):
            G._layer_attrs = mm["layer_tuple_attrs"]
        if mm.get("layer_attributes"):
            G.layer_attributes = _rows_to_df(_normalize_rows(mm["layer_attributes"]))

        # --- OPTIONAL: overlay reified hyperedges ---
        if hyperedges == "reified":
            _cx2_overlay_reified(aspects, G)

    # =====================================================================
    #                        PATH B: NO MANIFEST
    # =====================================================================
    else:
        directed = net_attrs.get("directed", True)
        G = Graph(directed=directed)

    # =====================================================================
    #            Overlay Cytoscape edits: nodes + edges from CX2
    # =====================================================================

    # Map CX numeric ids → AnnNet string ids
    cx2node = {}
    node_aspects = aspects.get("nodes", [])

    # --- build a row map of existing vertex attributes ---
    vmap = {}
    existing = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    for r in existing:
        vid = str(r.get("vertex_id", r.get("id")))
        vmap[vid] = dict(r)

    # --- update vertex attrs ---
    for n in node_aspects:
        cx_id = n["id"]
        attrs = dict(n.get("v", {}))

        ann_id = str(attrs.get("name", cx_id))
        cx2node[cx_id] = ann_id

        if ann_id not in G.entity_types:
            G.add_vertex(ann_id)

        row = vmap.get(ann_id, {"vertex_id": ann_id})
        for k, v in attrs.items():
            if k != "name":
                row[k] = v
        vmap[ann_id] = row

    # rebuild vertex table
    vnorm = _normalize_rows(list(vmap.values()))
    G.vertex_attributes = _rows_to_df(vnorm)

    # --- edges ---
    emap = {}
    existing = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    for r in existing:
        eid = str(r.get("edge_id", r.get("id")))
        emap[eid] = dict(r)

    for e in aspects.get("edges", []):
        s = cx2node.get(e["s"])
        t = cx2node.get(e["t"])
        if not s or not t:
            continue

        attrs = e.get("v", {})
        eid = str(attrs.get("edge_id", attrs.get("interaction", e["id"])))
        w = float(attrs.get("weight", 1.0))

        G.add_edge(s, t, edge_id=eid, weight=w)

        row = emap.get(eid, {"edge_id": eid})
        for k, v in attrs.items():
            if k not in ("interaction", "weight"):
                row[k] = v
        emap[eid] = row

    enorm = _normalize_rows(list(emap.values()))
    G.edge_attributes = _rows_to_df(enorm)

    return G



