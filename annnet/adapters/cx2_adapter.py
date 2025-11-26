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
    _rows_to_df,
    _serialize_layer_tuple_attrs,
    _deserialize_layer_tuple_attrs,
)

# --- Type Mapping Helper ---

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


# --- Core Adapter: to_cx2 ---

def to_cx2(G: Graph) -> List[Dict[str, Any]]:
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
        if eid in G.hyperedge_definitions:
            continue # Skip hyperedges
        try:
            u, v, _ = defn
        except ValueError:
            continue # Skip hyperedges

        if u not in node_map or v not in node_map:
            continue # Skip edges involving non-vertex entities

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
    attr_decls["nodes"]["name"] = {"d": "string"} # Always added

    # Define Edge Attributes
    if not e_attrs_df.is_empty():
        id_col = "edge_id" if "edge_id" in e_attrs_df.columns else "id"
        for col, dtype in e_attrs_df.schema.items():
            if col == id_col: continue
            attr_decls["edges"][col] = {"d": _map_pl_to_cx2_type(dtype)}
    attr_decls["edges"]["interaction"] = {"d": "string"}
    attr_decls["edges"]["weight"] = {"d": "double"}
    
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
                "__AnnNet_Manifest__": json.dumps(manifest)
            }]},
            {"nodes": cx_nodes},
            {"edges": cx_edges}, 
            {"status": [{"success": True}]} 
        ]
        
    return cx2
    


# --- Core Adapter: from_cx2 ---

def from_cx2(cx2_data: List[Dict[str, Any]]) -> Graph:
    """
    Convert a CX2 JSON list back into an AnnNet Graph.

    Strategy:
    1. Look for '__AnnNet_Manifest__' in networkAttributes.
    2. If found, fully rehydrate the graph from the manifest (restoring
       hyperedges, slices, etc.).
    3. Then, overlay any changes made to nodes/edges in the CX2 data
       (e.g., if the user edited attributes in Cytoscape).
    4. If no manifest is found, build a basic Graph from nodes/edges.
    """
    
    # 1. Parse Aspects
    aspects = {}
    for item in cx2_data:
        # CX2 aspects are dicts with a single key (aspect name)
        # except for CXVersion/metaData which are distinct
        keys = list(item.keys())
        if not keys: continue
        
        key = keys[0]
        if key in ("CXVersion", "metaData", "status"):
            aspects[key] = item[key]
        else:
            # Other aspects are lists of objects
            if key not in aspects:
                aspects[key] = []
            aspects[key].extend(item[key])

    # 2. Extract Network Attributes & Manifest
    net_attrs_list = aspects.get("networkAttributes", [])
    net_attrs = {}
    for na in net_attrs_list:
        net_attrs.update(na)
        
    manifest_str = net_attrs.get("__AnnNet_Manifest__")
    manifest = None
    if manifest_str:
        try:
            manifest = json.loads(manifest_str)
        except json.JSONDecodeError:
            pass # Manifest invalid

    # 3. Initial Graph Construction
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from annnet.core.graph import Graph
    G = Graph()
    
    if manifest:
        # --- PATH A: Full Rehydration from Manifest ---
        # Logic like from_graphtool with manifest
        
        # Graph attrs
        gmeta = manifest.get("graph", {})
        G.directed = gmeta.get("directed", True)
        G.graph_attributes = dict(gmeta.get("attributes", {}))

        # Vertices
        vmeta = manifest.get("vertices", {})
        v_rows = vmeta.get("attributes", [])
        if v_rows: G.vertex_attributes = _rows_to_df(v_rows)
        if vmeta.get("types"): G.entity_types.update(vmeta["types"])

        # Edges
        emeta = manifest.get("edges", {})
        if emeta.get("attributes"): G.edge_attributes = _rows_to_df(emeta["attributes"])
        if emeta.get("weights"): G.edge_weights.update(emeta["weights"])
        if emeta.get("definitions"): 
             # JSON keys are always strings, but definitions can rely on specific types
             G.edge_definitions = emeta["definitions"]
        
        if emeta.get("directed"): G.edge_directed.update(emeta["directed"])
        if emeta.get("direction_policy"): G.edge_direction_policy.update(emeta["direction_policy"])
        if emeta.get("hyperedges"): G.hyperedge_definitions = dict(emeta["hyperedges"])
        
        kivela_edge = emeta.get("kivela", {})
        if kivela_edge.get("edge_kind"): G.edge_kind.update(kivela_edge["edge_kind"])
        if kivela_edge.get("edge_layers"): 
            G.edge_layers.update(_deserialize_edge_layers(kivela_edge["edge_layers"]))

        # Slices
        smeta = manifest.get("slices", {})
        if smeta.get("data"): G._slices.update(_deserialize_slices(smeta["data"]))
        if smeta.get("slice_attributes"): G.slice_attributes = _rows_to_df(smeta["slice_attributes"])
        if smeta.get("edge_slice_attributes"): G.edge_slice_attributes = _rows_to_df(smeta["edge_slice_attributes"])

        # Layers
        mm = manifest.get("multilayer", {})
        if mm.get("aspects"):
            G.aspects = list(mm["aspects"])
            G.elem_layers = dict(mm.get("elem_layers") or {})
            G._rebuild_all_layers_cache() # Assuming this method exists on Graph
        
        if mm.get("aspect_attrs"): G._aspect_attrs.update(mm["aspect_attrs"])
        if mm.get("VM"): G._VM = _deserialize_VM(mm["VM"])
        
        # Redundant multilayer data for safety
        if mm.get("edge_kind"): G.edge_kind.update(mm["edge_kind"])
        if mm.get("edge_layers"): G.edge_layers.update(_deserialize_edge_layers(mm["edge_layers"]))
        
        if mm.get("node_layer_attrs"): 
            G._vertex_layer_attrs = _deserialize_node_layer_attrs(mm["node_layer_attrs"])
        if mm.get("layer_tuple_attrs"): 
            G._layer_attrs = _deserialize_layer_tuple_attrs(mm["layer_tuple_attrs"])
        if mm.get("layer_attributes"): 
            G.layer_attributes = _rows_to_df(mm["layer_attributes"])

    else:
        # --- PATH B: Basic Construction (No Manifest) ---
        directed = net_attrs.get("directed", True)
        G = Graph(directed=directed)
        
    # 4. Apply Updates from CX2 Data (Nodes/Edges)
    # This ensures that if the user moved nodes or changed attributes in Cytoscape,
    # those changes are reflected in G.
    
    # Parse Attribute Declarations to know types
    attr_decls = {}
    for decl_block in aspects.get("attributeDeclarations", []):
        attr_decls.update(decl_block)
        
    node_decls = attr_decls.get("nodes", {})
    edge_decls = attr_decls.get("edges", {})

    # -- Vertices --
    # In CX2, node IDs are integers. We need to find the "AnnNet ID".
    # We look for a 'name' attribute or fallback to the CX2 integer ID.
    
    # We collect rows to rebuild/update the DataFrame
    current_v_rows = _df_to_rows(getattr(G, "vertex_attributes", pl.DataFrame()))
    v_rows_map = {str(r["id"]): r for r in current_v_rows} if "id" in (current_v_rows[0] if current_v_rows else {}) else {}
    
    cx_id_to_annnet_id = {}
    
    for n in aspects.get("nodes", []):
        cx_id = n["id"]
        attrs = n.get("v", {})
        
        # Determine AnnNet ID
        annnet_id = str(attrs.get("name", str(cx_id)))
        cx_id_to_annnet_id[cx_id] = annnet_id
        
        # Add to graph structure if missing
        if annnet_id not in G.entity_types:
            G.add_vertex(annnet_id)
            
        # Update attributes
        if annnet_id not in v_rows_map:
            v_rows_map[annnet_id] = {"id": annnet_id}

        v_rows_map[annnet_id]["vertex_id"] = annnet_id

        for k, v in attrs.items():
            if k == "name": continue # 'name' maps to ID, no duplicate
            v_rows_map[annnet_id][k] = v

    # Rebuild vertex_attributes DataFrame
    if v_rows_map:
        G.vertex_attributes = _rows_to_df(list(v_rows_map.values()))

    # -- Edges --
    # Similar logic for edges
    current_e_rows = _df_to_rows(getattr(G, "edge_attributes", pl.DataFrame()))
    # Try to map by edge_id if exists
    e_rows_map = {}
    if current_e_rows:
        id_col = "edge_id" if "edge_id" in current_e_rows[0] else "id"
        e_rows_map = {str(r[id_col]): r for r in current_e_rows}

    for e in aspects.get("edges", []):
        sid = e["s"]
        tid = e["t"]
        
        # Skip if endpoints unknown
        if sid not in cx_id_to_annnet_id or tid not in cx_id_to_annnet_id:
            continue
            
        u = cx_id_to_annnet_id[sid]
        v = cx_id_to_annnet_id[tid]
        
        attrs = e.get("v", {})
        eid = str(attrs.get("interaction", e["id"]))
        weight = float(attrs.get("weight", 1.0))
        
        # Update/Add edge structure
        G.add_edge(u, v, edge_id=eid, weight=weight)
        
        # Update attributes
        if eid not in e_rows_map:
            e_rows_map[eid] = {"edge_id": eid} # favor 'edge_id'
            
        for k, val in attrs.items():
            if k in ("interaction", "weight"): continue
            e_rows_map[eid][k] = val

    # Rebuild edge_attributes DataFrame
    if e_rows_map:
        G.edge_attributes = _rows_to_df(list(e_rows_map.values()))

    return G