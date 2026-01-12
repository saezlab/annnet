"""
CX2 Adapter for AnnNet AnnNet.

Provides:
    to_cx2(G)        -> List[Dict[str, Any]] (CX2 JSON object)
    from_cx2(cx2_data) -> AnnNet

This adapter maps AnnNet Graphs to the Cytoscape Exchange (CX2) format.
It creates a lossless representation by serializing complex features
(hyperedges, slices, multilayer) into a 'manifest' stored within the
CX2 'networkAttributes'.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None
try:
    import pandas as pd  # optional fallback
except Exception:
    pd = None

if TYPE_CHECKING:
    from ..core.graph import AnnNet
from ..adapters._utils import (
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_slices,
    _deserialize_VM,
    _df_to_rows,
    _safe_df_to_rows,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_slices,
    _serialize_VM,
)

# --- Helpers ---
CX_STYLE_KEY = "__cx_style__"


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
            directed = (
                True if any(ev.get("role") in ("head", "tail") for ev in attrs.values()) else False
            )
        else:
            directed = False

        hyperdefs.append((eid, directed, head_map, tail_map, attrs, he_id))

    return hyperdefs, membership_edges


def _rows_to_df(rows):
    if not rows:
        if pl is not None:
            return pl.DataFrame()
        if pd is not None:
            return pd.DataFrame()
        return []  # keep as rows

    # --- 1) Normalize all rows to full schema ---
    keys = set().union(*(r.keys() for r in rows))
    norm = [{k: r.get(k, None) for k in keys} for r in rows]

    # --- 2) Let Polars infer schema on the first few rows ---
    # allow nulls / mixed types
    if pl is not None:
        try:
            return pl.DataFrame(norm, infer_schema_length=1000)
        except Exception:
            return pl.DataFrame(
                [{k: (str(v) if v is not None else None) for k, v in r.items()} for r in norm]
            )
    # pandas fallback
    if pd is not None:
        try:
            return pd.DataFrame(norm)
        except Exception:
            return pd.DataFrame(
                [{k: (str(v) if v is not None else None) for k, v in r.items()} for r in norm]
            )
    return norm


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


def to_cx2(G: AnnNet, *, export_name="annnet export", hyperedges="skip") -> list[dict[str, Any]]:
    """
    Convert an AnnNet AnnNet -> CX2 compliant JSON list.

    The output is a list of aspect dictionaries (CX2 format).
    Complex AnnNet features (hyperedges, slices, etc.) are serialized
    into a JSON string and stored in 'networkAttributes' under '__AnnNet_Manifest__'.
    """

    # 1. Prepare Manifest (Lossless storage of complex features)
    vert_rows = _safe_df_to_rows(getattr(G, "vertex_attributes", None))
    edge_rows = _safe_df_to_rows(getattr(G, "edge_attributes", None))
    slice_rows = _safe_df_to_rows(getattr(G, "slice_attributes", None))
    edge_slice_rows = _safe_df_to_rows(getattr(G, "edge_slice_attributes", None))
    layer_attr_rows = _safe_df_to_rows(getattr(G, "layer_attributes", None))

    # strip CX-specific style from what we embed into the manifest
    g_attrs = dict(getattr(G, "graph_attributes", {}))
    g_attrs.pop(CX_STYLE_KEY, None)

    manifest = {
        "version": 1,
        "graph": {
            "directed": bool(G.directed) if G.directed is not None else True,
            "attributes": g_attrs,
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
    node_map: dict[Any, int] = {}
    cx_nodes: list[dict[str, Any]] = []
    cx_edges: list[dict[str, Any]] = []

    def _clean_cx2_attrs(attrs, string_cols, list_cols=None):
        """Clean attributes: remove nulls, convert non-JSON types."""
        if not attrs:
            return {}
        if list_cols is None:
            list_cols = set()
        out = {}
        for k, v in attrs.items():
            if v is None:
                if k in string_cols:
                    out[k] = ""
                elif k in list_cols:
                    out[k] = []
                # else: drop null entirely (CX2 doesn't accept null)
            elif isinstance(v, (set, frozenset)):
                out[k] = list(v)
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
                    for x in v
                ]
            else:
                out[k] = str(v)  # fallback: stringify unknown types
        return out

    # -- Nodes --
    # We only map entities of type 'vertex' to visual nodes.
    current_node_id = 0

    # Pre-fetch attribute data for fast lookup
    v_attrs_df = getattr(G, "vertex_attributes", None)

    # Identify which vertex columns are string vs numeric (for cleaning None)
    v_string_cols: set[str] = set()
    v_numeric_cols: set[str] = set()
    # Only do dtype-based classification if this is actually a Polars DF
    if pl is not None and isinstance(v_attrs_df, pl.DataFrame) and v_attrs_df.height:
        for col, dtype in v_attrs_df.schema.items():
            # strings
            if dtype in (pl.Utf8, pl.String, pl.Object):
                v_string_cols.add(col)
            # numeric: all ints + floats
            elif dtype in (
                pl.Int8,
                pl.Int16,
                pl.Int32,
                pl.Int64,
                pl.UInt8,
                pl.UInt16,
                pl.UInt32,
                pl.UInt64,
                pl.Float32,
                pl.Float64,
            ):
                v_numeric_cols.add(col)

    # Build map: vertex_id -> attribute row dict
    v_attrs_map: dict[str, dict[str, Any]] = {}
    if v_attrs_df is not None and vert_rows:
        cols = set(v_attrs_df.columns)
        id_col = "vertex_id" if "vertex_id" in cols else "id"
        for r in vert_rows:  # vert_rows is _df_to_rows(vertex_attributes)
            key = r.get(id_col)
            if key is not None:
                v_attrs_map[str(key)] = r

    def _clean_vertex_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
        """Replace None in strings with '', drop None for numeric/others."""
        out: dict[str, Any] = {}
        for k, v in attrs.items():
            if v is None:
                if k in v_string_cols:
                    out[k] = ""  # string: None -> ""
                elif k in v_numeric_cols:
                    # numeric: drop the key entirely (no null numbers in CX2)
                    continue
                else:
                    # unknown type: safest is to drop null
                    continue
            else:
                out[k] = v
        return out

    for uid, utype in G.entity_types.items():
        if utype != "vertex":
            continue

        cx_id = current_node_id
        current_node_id += 1
        node_map[uid] = cx_id

        # base node object
        n_obj: dict[str, Any] = {
            "id": cx_id,
            "v": {"name": str(uid)},  # 'name' is standard in Cytoscape
        }

        coords: dict[str, float] = {}

        # Attach attributes if present
        row = v_attrs_map.get(str(uid))
        if row is not None:
            attrs = dict(row)  # copy
            # get rid of id column from attributes
            attrs.pop("id", None)
            attrs.pop("vertex_id", None)

            # pull layout_* into x/y/z for Cytoscape layout
            for src, dst in (("layout_x", "x"), ("layout_y", "y"), ("layout_z", "z")):
                val = attrs.get(src)
                if val is not None:
                    try:
                        coords[dst] = float(val)
                    except (TypeError, ValueError):
                        pass
                # never expose layout_* as regular attributes
                attrs.pop(src, None)

            # clean None values: "" for strings, drop for numeric/others
            attrs = _clean_vertex_attrs(attrs)
            n_obj["v"].update(attrs)

        # put node coordinates at top-level (where Cytoscape expects them)
        if coords:
            n_obj.update(coords)

        cx_nodes.append(n_obj)

    # -- Edges --
    # Only binary edges between mapped vertices
    current_edge_id = 0
    e_attrs_df = getattr(G, "edge_attributes", None)

    # string columns in edge_attributes (for None -> "")
    e_string_cols = set()
    if pl is not None and isinstance(e_attrs_df, pl.DataFrame) and e_attrs_df.height:
        for col, dtype in e_attrs_df.schema.items():
            if dtype in (pl.Utf8, pl.String, pl.Object):
                e_string_cols.add(col)

    # Create lookup for edge attributes (handle both 'edge_id' and 'id')
    e_attrs_map = {}
    if e_attrs_df is not None and edge_rows:
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
                    "expanded_edges": [],
                }
                members = S | T
                if directed:
                    # tail -> head cartesian
                    for u in T:
                        for v in S:
                            exp_entry["expanded_edges"].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            raw_attrs = e_attrs_map.get(str(eid), {})
                            clean_attrs = _clean_cx2_attrs(raw_attrs, e_string_cols)
                            cx_edges.append(
                                {
                                    "id": cx_eid,
                                    "s": node_map[u],
                                    "t": node_map[v],
                                    "v": {
                                        "interaction": str(eid),
                                        "weight": float(G.edge_weights.get(eid, 1.0)),
                                        **clean_attrs,
                                    },
                                }
                            )

                else:
                    # undirected clique
                    mem = list(members)
                    for i in range(len(mem)):
                        for j in range(i + 1, len(mem)):
                            u, v = mem[i], mem[j]
                            exp_entry["expanded_edges"].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            raw_attrs = e_attrs_map.get(str(eid), {})
                            clean_attrs = _clean_cx2_attrs(raw_attrs, e_string_cols)
                            cx_edges.append(
                                {
                                    "id": cx_eid,
                                    "s": node_map[u],
                                    "t": node_map[v],
                                    "v": {
                                        "interaction": str(eid),
                                        "weight": float(G.edge_weights.get(eid, 1.0)),
                                        **clean_attrs,
                                    },
                                }
                            )
                manifest["edges"]["expanded"][str(eid)] = exp_entry
                continue

            if hyperedges == "reify":
                he_cx_id = current_node_id
                current_node_id += 1

                # filter edge attrs so we don't leak edge_id/id onto the node
                he_attrs = e_attrs_map.get(str(eid), {}).copy()
                he_attrs.pop("edge_id", None)
                he_attrs.pop("id", None)
                he_attrs = _clean_cx2_attrs(he_attrs, e_string_cols)

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

                weight = float(G.edge_weights.get(eid, 1.0))

                if directed:
                    # tail -> HE
                    for u in T:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                "id": cx_eid,
                                "s": node_map[u],
                                "t": he_cx_id,
                                "v": {
                                    "interaction": f"{eid}::tail",
                                    "role": "tail",
                                    "weight": weight,
                                },
                            }
                        )
                    # HE -> head
                    for v in S:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                "id": cx_eid,
                                "s": he_cx_id,
                                "t": node_map[v],
                                "v": {
                                    "interaction": f"{eid}::head",
                                    "role": "head",
                                    "weight": weight,
                                },
                            }
                        )

                else:
                    # undirected: HE - members
                    members = S | T
                    for u in members:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                "id": cx_eid,
                                "s": node_map[u],
                                "t": he_cx_id,
                                "v": {
                                    "interaction": f"{eid}::member",
                                    "role": "member",
                                    "weight": weight,
                                },
                            }
                        )
                continue

            # unknown hyperedge mode - skip
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
            "v": {"interaction": str(eid), "weight": float(G.edge_weights.get(eid, 1.0))},
        }

        # Attach attributes
        if str(eid) in e_attrs_map:
            attrs = e_attrs_map[str(eid)].copy()
            # Remove redundant keys
            attrs.pop("edge_id", None)
            attrs.pop("id", None)
            attrs = _clean_cx2_attrs(attrs, e_string_cols)
            e_obj["v"].update(attrs)

        cx_edges.append(e_obj)

    # 3. Attribute Declarations
    attr_decls = {"nodes": {}, "edges": {}, "networkAttributes": {}}

    # Define Node Attributes
    if pl is not None and isinstance(v_attrs_df, pl.DataFrame) and v_attrs_df.height:
        for col, dtype in v_attrs_df.schema.items():
            if col == "id":
                continue
            attr_decls["nodes"][col] = {"d": _map_pl_to_cx2_type(dtype)}
    elif v_attrs_df is not None and hasattr(v_attrs_df, "columns"):
        for col in list(v_attrs_df.columns):
            if col == "id":
                continue
            attr_decls["nodes"][col] = {"d": "string"}

    attr_decls["nodes"]["name"] = {"d": "string"}  # Always added
    attr_decls["nodes"]["is_hyperedge"] = {"d": "boolean"}
    attr_decls["nodes"]["eid"] = {"d": "string"}
    attr_decls["nodes"]["tag"] = {"d": "string"}
    attr_decls["nodes"]["reaction"] = {"d": "string"}

    # Define Edge Attributes
    if pl is not None and isinstance(e_attrs_df, pl.DataFrame) and e_attrs_df.height:
        id_col = "edge_id" if "edge_id" in e_attrs_df.columns else "id"
        for col, dtype in e_attrs_df.schema.items():
            if col == id_col:
                continue
            attr_decls["edges"][col] = {"d": _map_pl_to_cx2_type(dtype)}
    elif e_attrs_df is not None and hasattr(e_attrs_df, "columns"):
        id_col = "edge_id" if "edge_id" in e_attrs_df.columns else "id"
        for col in list(e_attrs_df.columns):
            if col == id_col:
                continue
            attr_decls["edges"][col] = {"d": "string"}
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

    # Start with basic metadata
    meta = [
        {"name": "attributeDeclarations", "elementCount": 1},
        {"name": "networkAttributes", "elementCount": 1},
        {"name": "nodes", "elementCount": len(cx_nodes)},
        {"name": "edges", "elementCount": len(cx_edges)},
    ]

    cx2: list[dict[str, Any]] = [
        {"CXVersion": "2.0", "hasFragments": False},
        {"metaData": meta},
        {"attributeDeclarations": [attr_decls]},
        {
            "networkAttributes": [
                {
                    "name": export_name,
                    "directed": bool(G.directed) if G.directed is not None else True,
                    "__AnnNet_Manifest__": json.dumps(_jsonify(manifest)),
                }
            ]
        },
        {"nodes": cx_nodes},
        {"edges": cx_edges},
    ]

    # Re-emit Cytoscape visual style if we have it
    style = dict(getattr(G, "graph_attributes", {})).get(CX_STYLE_KEY, {}) or {}

    vp = style.get("visualProperties")
    if vp:
        meta.append({"name": "visualProperties", "elementCount": 1})
        cx2.append({"visualProperties": [vp]})

    nb = style.get("nodeBypasses")
    if nb:
        meta.append({"name": "nodeBypasses", "elementCount": len(nb)})
        cx2.append({"nodeBypasses": nb})

    eb = style.get("edgeBypasses")
    if eb:
        meta.append({"name": "edgeBypasses", "elementCount": len(eb)})
        cx2.append({"edgeBypasses": eb})

    vep = style.get("visualEditorProperties")
    if vep:
        meta.append({"name": "visualEditorProperties", "elementCount": 1})
        cx2.append({"visualEditorProperties": [vep]})

    # Status goes last
    cx2.append({"status": [{"success": True}]})

    return cx2

    return cx2


# --- Core Adapter: from_cx2 ---


def from_cx2(cx2_data, *, hyperedges="manifest"):
    """
    Fully robust CX2 - AnnNet importer.
    Supports:
      - manifest reconstruction (full fidelity)
      - reified hyperedges
      - expanded hyperedges
      - Cytoscape-edited files
      - sparse attribute tables
      - arbitrary attribute modifications
    """
    # Small helper: normalize rows so Polars won't crash

    def _normalize_rows(rows):
        if not rows:
            return []
        keys = set().union(*(r.keys() for r in rows))
        return [{k: r.get(k, None) for k in keys} for r in rows]

    # Load file or JSON string

    import json
    import os

    if isinstance(cx2_data, str):
        if os.path.exists(cx2_data):
            with open(cx2_data) as f:
                cx2_data = json.load(f)
        else:
            try:
                cx2_data = json.loads(cx2_data)
            except Exception:
                raise ValueError("Invalid CX2 string or file")

    # Parse aspects into a dict

    aspects = {}
    for item in cx2_data:
        if not item:
            continue
        key = list(item.keys())[0]

        if key in ("CXVersion", "metaData", "status"):
            aspects[key] = item[key]
        else:
            aspects.setdefault(key, []).extend(item[key])

    # Extract networkAttributes + manifest JSON

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
    visual_props = aspects.get("visualProperties", [])

    # Extract Cytoscape visual style aspects (kept opaque but preserved)
    style_aspects: dict[str, Any] = {}

    vp = aspects.get("visualProperties")
    if vp:
        style_aspects["visualProperties"] = vp[0] if isinstance(vp, list) else vp

    nb = aspects.get("nodeBypasses")
    if nb:
        style_aspects["nodeBypasses"] = nb

    eb = aspects.get("edgeBypasses")
    if eb:
        style_aspects["edgeBypasses"] = eb

    vep = aspects.get("visualEditorProperties")
    if vep:
        style_aspects["visualEditorProperties"] = vep[0] if isinstance(vep, list) else vep

    # Construct AnnNet

    from annnet.core.graph import AnnNet

    G = AnnNet()

    # PATH A: MANIFEST RECONSTRUCTION

    if manifest and hyperedges in ("manifest", "reified"):
        # --- Base graph attrs ---
        gmeta = manifest.get("graph", {})
        G.directed = gmeta.get("directed", True)
        G.graph_attributes = dict(gmeta.get("attributes", {}))

        if visual_props:
            G.graph_attributes["__cx_visualProperties__"] = visual_props

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
        if emeta.get("weights"):
            G.edge_weights.update(emeta["weights"])
        if emeta.get("directed"):
            G.edge_directed.update(emeta["directed"])
        if emeta.get("definitions"):
            G.edge_definitions = dict(emeta["definitions"])
        if emeta.get("direction_policy"):
            G.edge_direction_policy.update(emeta["direction_policy"])

        # hyperedge definitions
        if emeta.get("hyperedges"):
            fixed = {}
            for eid, info in emeta["hyperedges"].items():
                # Older / simple manifests store hyperedges as a plain list of members
                # e.g. "he1": ["n1", "n2", "n3"]
                if isinstance(info, list):
                    fixed[eid] = {
                        "directed": False,
                        "members": set(info),
                    }
                    continue

                # Newer manifests: dict form with keys like "directed", "members" or "head"/"tail"
                directed = bool(info.get("directed", False))
                if directed:
                    fixed[eid] = {
                        "directed": True,
                        "head": set(info.get("head", [])),
                        "tail": set(info.get("tail", [])),
                    }
                else:
                    fixed[eid] = {
                        "directed": False,
                        "members": set(info.get("members", [])),
                    }

            G.hyperedge_definitions = fixed

        # --- Expanded hyperedges (if present) ---
        exp = emeta.get("expanded", {})
        if exp:
            hyperedge_bulk_data = []
            for eid, info in exp.items():
                directed = info.get("mode") == "directed"
                if directed:
                    hyperedge_bulk_data.append({
                        "head": info.get("head", []),
                        "tail": info.get("tail", []),
                        "edge_id": eid,
                        "edge_directed": True,
                    })
                else:
                    hyperedge_bulk_data.append({
                        "members": info.get("members", []),
                        "edge_id": eid,
                        "edge_directed": False,
                    })
            
            if hyperedge_bulk_data:
                G.add_hyperedges_bulk(hyperedge_bulk_data)

        # --- Layers (Kivela)---
        kiv = emeta.get("kivela", {})
        if kiv.get("edge_kind"):
            G.edge_kind.update(kiv["edge_kind"])
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
        if mm.get("aspects"):
            G.aspects = mm["aspects"]
        if mm.get("elem_layers"):
            G.elem_layers = dict(mm["elem_layers"])
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
            _cx2_collect_reified(aspects, G)

    # PATH B: NO MANIFEST

    else:
        directed = net_attrs.get("directed", True)
        G = AnnNet(directed=directed)

        G.entity_types = {}
        if visual_props:
            # make sure we have a dict
            if not hasattr(G, "graph_attributes") or G.graph_attributes is None:
                G.graph_attributes = {}
            G.graph_attributes["__cx_visualProperties__"] = visual_props

    # Overlay Cytoscape edits: nodes + edges from CX2

    # Map CX numeric ids - AnnNet string ids
    cx2node = {}
    node_aspects = aspects.get("nodes", [])

    # --- build a row map of existing vertex attributes ---
    vmap = {}
    existing = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    for r in existing:
        vid = str(r.get("vertex_id", r.get("id")))
        vmap[vid] = dict(r)

    # --- update vertex attrs ---
    vertex_bulk_data = []
    for n in node_aspects:
        cx_id = n["id"]
        attrs = dict(n.get("v", {}))
        ann_id = str(attrs.get("name", cx_id))
        cx2node[cx_id] = ann_id
        
        row = vmap.get(ann_id, {"vertex_id": ann_id})
        
        # Merge attributes from Cytoscape (except display name)
        for k, v in attrs.items():
            if k != "name":
                row[k] = v
        
        # Layout coordinates live on the node, not in v
        if "x" in n and n["x"] is not None:
            row["layout_x"] = float(n["x"])
        if "y" in n and n["y"] is not None:
            row["layout_y"] = float(n["y"])
        if "z" in n and n["z"] is not None:
            row["layout_z"] = float(n["z"])
        
        vertex_bulk_data.append(row)

    # Single bulk vertex insert
    if vertex_bulk_data:
        G.add_vertices_bulk(vertex_bulk_data)

    # rebuild vertex table
    if vertex_bulk_data:
        G.vertex_attributes = _rows_to_df(_normalize_rows(vertex_bulk_data))
    else:
        G.vertex_attributes = _rows_to_df([])

    # Normalise ID column name: prefer 'vertex_id' consistently
    cols = set(G.vertex_attributes.columns)
    if "vertex_id" not in cols and "id" in cols:
        G.vertex_attributes = G.vertex_attributes.rename({"id": "vertex_id"})

    # --- edges ---
    emap = {}
    existing = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    for r in existing:
        eid = str(r.get("edge_id", r.get("id")))
        emap[eid] = dict(r)

    edge_bulk_data = []
    for e in aspects.get("edges", []):
        s = cx2node.get(e["s"])
        t = cx2node.get(e["t"])
        if not s or not t:
            continue
        
        attrs = e.get("v", {})
        eid = str(attrs.get("edge_id", attrs.get("interaction", e["id"])))
        w = float(attrs.get("weight", 1.0))
        
        # Collect edge data for bulk insert
        edge_dict = {
            "source": s,
            "target": t,
            "edge_id": eid,
            "weight": w,
        }
        
        # Collect additional attributes (excluding interaction and weight)
        extra_attrs = {k: v for k, v in attrs.items() if k not in ("interaction", "weight")}
        if extra_attrs:
            edge_dict["attributes"] = extra_attrs
        
        edge_bulk_data.append(edge_dict)

    # Single bulk edge insert
    if edge_bulk_data:
        G.add_edges_bulk(edge_bulk_data)

    enorm = _normalize_rows(list(emap.values()))
    G.edge_attributes = _rows_to_df(enorm)

    # Attach Cytoscape style blob if we captured any
    if style_aspects:
        # make sure graph_attributes exists and is a dict
        G.graph_attributes = dict(getattr(G, "graph_attributes", {}))
        G.graph_attributes[CX_STYLE_KEY] = style_aspects

    return G
