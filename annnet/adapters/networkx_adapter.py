from __future__ import annotations

try:
    import networkx as nx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency 'networkx' is not installed. "
        "Install with: pip install annnet[networkx]"
    ) from e

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import AnnNet
import json
from enum import Enum
from typing import Any

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

# ---------------
import time
from contextlib import contextmanager

from ._utils import (
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_VM,
    _endpoint_coeff_map,
    _is_directed_eid,
    _rows_to_df,
    _safe_df_to_rows,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_VM,
)


@contextmanager
def _time(label, timings):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timings[label] = timings.get(label, 0.0) + (t1 - t0)

# ---------------

def _serialize_value(v: Any) -> Any:
    if isinstance(v, Enum):
        return v.name
    if hasattr(v, "items"):
        return dict(v)
    return v


def _attrs_to_dict(attrs_dict: dict) -> dict:
    out = {}
    for k, v in attrs_dict.items():
        if isinstance(v, Enum):
            out[k] = v.name
        elif hasattr(v, "items"):
            out[k] = {kk: (vv.name if isinstance(vv, Enum) else vv) for kk, vv in dict(v).items()}
        else:
            out[k] = v
    return out


def _export_legacy(
    graph: AnnNet,
    *,
    directed: bool = True,
    skip_hyperedges: bool = True,
    public_only: bool = False,
):
    """Export AnnNet to NetworkX Multi(Di)Graph without manifest.

    Parameters
    ----------
    graph : AnnNet
        Source graph instance.
    directed : bool
        If True, export as MultiDiGraph; else MultiGraph.
        Undirected edges in directed export are emitted bidirectionally.
    skip_hyperedges : bool
        If True, drop hyperedges. If False:
          - directed hyperedges expand head×tail (cartesian product)
          - undirected hyperedges expand to clique
    public_only : bool
        If True, strip private attrs starting with "__".

    Returns
    -------
    networkx.MultiGraph | networkx.MultiDiGraph

    """
    G = nx.MultiDiGraph() if directed else nx.MultiGraph()

    def _rows(t):
        if t is None:
            return []
        if hasattr(t, "to_dicts"):
            try:
                return list(t.to_dicts())
            except Exception:
                pass
        if hasattr(t, "to_dict"):
            try:
                recs = t.to_dict(orient="records")
                if isinstance(recs, list):
                    return recs
            except Exception:
                pass
        if hasattr(t, "to_pylist"):
            try:
                return list(t.to_pylist())
            except Exception:
                pass
        if hasattr(t, "fetchall") and hasattr(t, "columns"):
            try:
                cols = list(t.columns)
                return [dict(zip(cols, row)) for row in t.fetchall()]
            except Exception:
                pass
        if isinstance(t, dict):
            keys = list(t.keys())
            if keys and isinstance(t[keys[0]], list):
                n = len(t[keys[0]])
                return [{k: t[k][i] for k in keys} for i in range(n)]
        if isinstance(t, list) and t and isinstance(t[0], dict):
            return list(t)
        return []

    # BATCH READ VERTEX ATTRIBUTES
    v_rows = _rows(graph.vertex_attributes)
    v_attrs_map = {}
    for row in v_rows:
        vid = row.get("vertex_id")
        if vid:
            attrs = dict(row)
            attrs.pop("vertex_id", None)
            if public_only:
                attrs = {k: _serialize_value(v) for k, v in attrs.items() if not str(k).startswith("__")}
            else:
                attrs = {k: _serialize_value(v) for k, v in attrs.items()}
            v_attrs_map[vid] = attrs

    # BATCH READ EDGE ATTRIBUTES
    e_rows = _rows(graph.edge_attributes)
    e_attrs_map = {}
    for row in e_rows:
        eid = row.get("edge_id")
        if eid:
            attrs = dict(row)
            attrs.pop("edge_id", None)
            if public_only:
                attrs = {k: _serialize_value(v) for k, v in attrs.items() if not str(k).startswith("__")}
            else:
                attrs = {k: _serialize_value(v) for k, v in attrs.items()}
            e_attrs_map[eid] = attrs

    # ADD VERTICES WITH CACHED ATTRIBUTES
    for v in graph.vertices():
        v_attr = v_attrs_map.get(v, {})
        G.add_node(v, **v_attr)

    # ADD EDGES WITH CACHED ATTRIBUTES
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)

        e_attr = dict(e_attrs_map.get(eid, {}))

        weight = graph.edge_weights.get(eid, 1.0)
        if public_only:
            e_attr["weight"] = weight
        else:
            e_attr["__weight"] = weight

        is_hyper = graph.edge_kind.get(eid) == "hyper"
        is_dir = _is_directed_eid(graph, eid)
        members = S | T

        if not is_hyper and len(members) <= 2:
            if len(members) == 1:
                u = next(iter(members))
                G.add_edge(u, u, key=eid, **e_attr)
            else:
                if is_dir:
                    uu = next(iter(S))
                    vv = next(iter(T))
                    G.add_edge(uu, vv, key=eid, **e_attr)
                else:
                    u, v = tuple(members)
                    if directed:
                        G.add_edge(u, v, key=eid, **e_attr)
                        G.add_edge(v, u, key=eid, **e_attr)
                    else:
                        G.add_edge(u, v, key=eid, **e_attr)
            continue

        if skip_hyperedges:
            continue

        if is_dir:
            for u in S:
                for v in T:
                    if directed:
                        G.add_edge(u, v, key=eid, **e_attr)
                    else:
                        G.add_edge(u, v, key=eid, directed=True, **e_attr)
        else:
            mem = list(members)
            n = len(mem)
            if directed:
                for a in range(n):
                    for b in range(n):
                        if a == b:
                            continue
                        G.add_edge(mem[a], mem[b], key=eid, **e_attr)
            else:
                for a in range(n):
                    for b in range(a + 1, n):
                        G.add_edge(mem[a], mem[b], key=eid, **e_attr)

    return G


def _coeff_from_obj(obj) -> float:
    if isinstance(obj, (int, float)):
        return float(obj)
    if hasattr(obj, "items"):
        v = obj.get("__value", 1)
        if hasattr(v, "items"):
            v = v.get("__value", 1)
        try:
            return float(v)
        except Exception:
            return 1.0
    return 1.0


def to_nx(
    graph: AnnNet,
    directed=True,
    hyperedge_mode="skip",
    slice=None,
    slices=None,
    public_only=False,
    reify_prefix="he::",
):
    """Export AnnNet → (networkx.AnnNet, manifest).
    Manifest preserves hyperedges with per-endpoint coefficients, slices,
    vertex/edge attrs, and stable edge IDs.

    Parameters
    ----------
    graph : AnnNet
    directed : bool
    hyperedge_mode : {"skip", "expand", "reify"}
    slice : str, optional
        Export single slice only (affects which hyperedges are reified).
    slices : list[str], optional
        Export union of specified slices (affects which hyperedges are reified).
    public_only : bool

    Returns
    -------
    tuple[networkx.AnnNet, dict]
        (nxG, manifest)

    """

    def _public(d):
        if not d:
            return {}
        return {k: v for k, v in d.items() if not str(k).startswith("__")}

    def _rows_from_table(t):
        if t is None:
            return []
        if hasattr(t, "to_dicts"):
            try:
                return list(t.to_dicts())
            except Exception:
                pass
        if hasattr(t, "to_dict"):
            try:
                recs = t.to_dict(orient="records")
                if isinstance(recs, list):
                    return recs
            except Exception:
                pass
        if hasattr(t, "to_pylist"):
            try:
                return list(t.to_pylist())
            except Exception:
                pass
        if hasattr(t, "fetchall") and hasattr(t, "columns"):
            try:
                cols = list(t.columns)
                return [dict(zip(cols, row)) for row in t.fetchall()]
            except Exception:
                pass
        if isinstance(t, dict):
            keys = list(t.keys())
            if keys and isinstance(t[keys[0]], list):
                n = len(t[keys[0]])
                return [{k: t[k][i] for k in keys} for i in range(n)]
        if isinstance(t, list) and t and isinstance(t[0], dict):
            return list(t)
        return []

    # Figure out which hyperedges should be included if user filters by slice(s)
    requested_lids = set()
    if slice is not None:
        requested_lids.update([slice] if isinstance(slice, str) else list(slice))
    if slices is not None:
        requested_lids.update(list(slices))

    selected_eids = None
    if requested_lids:
        selected_eids = set()
        for lid in requested_lids:
            try:
                for eid in graph.get_slice_edges(lid):
                    selected_eids.add(eid)
            except Exception:
                pass

    # Base NX graph (binary edges only)
    nxG = _export_legacy(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ("skip", "reify")),
        public_only=public_only,
    )

    # HOIST LOOKUPS
    vertex_attributes_df = graph.vertex_attributes
    edge_attributes_df = graph.edge_attributes
    idx_to_edge = graph.idx_to_edge
    edge_kind_dict = graph.edge_kind
    num_edges = graph.number_of_edges()

    # Vertex attributes - BATCH READ
    vertex_attrs = {}
    v_rows = _rows_from_table(vertex_attributes_df)
    for row in v_rows:
        v = row.get("vertex_id")
        if v is None:
            continue
        attrs = dict(row)
        attrs.pop("vertex_id", None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith("__")}
        vertex_attrs[v] = _attrs_to_dict(attrs)

    # Edge attributes - BATCH READ
    edge_attrs = {}
    e_rows = _rows_from_table(edge_attributes_df)
    for row in e_rows:
        eid = row.get("edge_id")
        if eid is None:
            continue
        attrs = dict(row)
        attrs.pop("edge_id", None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith("__")}
        edge_attrs[eid] = _attrs_to_dict(attrs)

    # Edge topology snapshot - BATCH BUILD
    manifest_edges = {}
    for eidx in range(num_edges):
        S, T = graph.get_edge(eidx)
        eid = idx_to_edge[eidx]
        is_hyper = edge_kind_dict.get(eid) == "hyper"

        if not is_hyper:
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                manifest_edges[eid] = (u, u, "regular")
            elif len(members) == 2:
                u, v = sorted(members)
                manifest_edges[eid] = (u, v, "regular")
            else:
                eattr = edge_attrs.get(eid, {})
                head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
                tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
                manifest_edges[eid] = (head_map, tail_map, "hyper")
        else:
            eattr = edge_attrs.get(eid, {})
            head_map = _endpoint_coeff_map(eattr, "__source_attr", S)
            tail_map = _endpoint_coeff_map(eattr, "__target_attr", T)
            manifest_edges[eid] = (head_map, tail_map, "hyper")

    # Baseline edge weights
    try:
        weights_map = {eid: float(w) for eid, w in getattr(graph, "edge_weights", {}).items()}
    except Exception:
        weights_map = {}

    all_eids = list(manifest_edges.keys())

    # Slice discovery
    lids = set()
    try:
        lids.update(list(graph.list_slices(include_default=True)))
    except Exception:
        try:
            lids.update(list(graph.list_slices()))
        except Exception:
            pass

    t = getattr(graph, "edge_slice_attributes", None)
    if isinstance(t, dict):
        lids.update(t.keys())
    for r in _rows_from_table(t):
        lid = r.get("slice") or r.get("slice_id") or r.get("lid")
        if lid is not None:
            lids.add(lid)

    le = getattr(graph, "slice_edges", None)
    if isinstance(le, dict):
        lids.update(le.keys())

    etl = getattr(graph, "edge_to_slices", None)
    if isinstance(etl, dict):
        for arr in etl.values():
            for lid in arr or []:
                lids.add(lid)

    slices_section = {lid: [] for lid in lids}

    # Build slices membership
    for lid in list(lids):
        try:
            eids = list(graph.get_slice_edges(lid))
        except Exception:
            eids = []
        if eids:
            seen = set(slices_section[lid])
            for e in eids:
                if e not in seen:
                    slices_section[lid].append(e)
                    seen.add(e)

    if isinstance(t, dict):
        for lid, mapping in t.items():
            if isinstance(mapping, dict):
                arr = slices_section.setdefault(lid, [])
                seen = set(arr)
                for eid in list(mapping.keys()):
                    if eid not in seen:
                        arr.append(eid)
                        seen.add(eid)
    
    for r in _rows_from_table(t):
        lid = r.get("slice") or r.get("slice_id") or r.get("lid")
        if lid is None:
            continue
        eid = r.get("edge_id", r.get("edge"))
        if eid is not None:
            arr = slices_section.setdefault(lid, [])
            if eid not in arr:
                arr.append(eid)

    if isinstance(le, dict):
        for lid, eids in le.items():
            arr = slices_section.setdefault(lid, [])
            seen = set(arr)
            for eid in list(eids):
                if eid not in seen:
                    arr.append(eid)
                    seen.add(eid)
    
    if isinstance(etl, dict):
        for eid, arr_lids in etl.items():
            for lid in arr_lids or []:
                arr = slices_section.setdefault(lid, [])
                if eid not in arr:
                    arr.append(eid)

    # Per-edge slice weights - BATCH READ
    slice_weights = {}
    candidate_lids = set(slices_section.keys()) or lids
    
    # Read entire slice attributes table once
    slice_attr_table = getattr(graph, "edge_slice_attributes", None)
    slice_attr_rows = _rows_from_table(slice_attr_table)
    
    # Build lookup: {(slice, edge_id): weight}
    slice_weight_lookup = {}
    for row in slice_attr_rows:
        lid = row.get("slice") or row.get("slice_id") or row.get("lid")
        eid = row.get("edge_id") or row.get("edge")
        w = row.get("weight")
        if lid is not None and eid is not None and w is not None:
            slice_weight_lookup[(lid, eid)] = float(w)
    
    # Populate slice_weights from lookup
    for lid in candidate_lids:
        arr = slices_section.setdefault(lid, [])
        seen = set(arr)
        for eid in all_eids:
            w = slice_weight_lookup.get((lid, eid))
            if w is not None:
                if eid not in seen:
                    arr.append(eid)
                    seen.add(eid)
                slice_weights.setdefault(lid, {})[eid] = w

    # Drop empties
    slices_section = {lid: eids for lid, eids in slices_section.items() if eids}

    # Respect slice/slices filters
    if requested_lids:
        req_norm = {str(x) for x in requested_lids}
        slices_section = {lid: eids for lid, eids in slices_section.items() if str(lid) in req_norm}
        slice_weights = {lid: m for lid, m in slice_weights.items() if str(lid) in req_norm}

    # REIFY: add HE nodes + membership edges
    if hyperedge_mode == "reify":
        import networkx as nx

        allowed = None
        if requested_lids:
            allowed = set()
            for lid, eids in slices_section.items():
                for eid in eids:
                    allowed.add(eid)

        is_multi_di = isinstance(nxG, nx.MultiDiGraph)

        # BATCH REIFY OPERATIONS
        he_nodes_to_add = []
        membership_edges_to_add = []

        for eid, spec in manifest_edges.items():
            if spec[-1] != "hyper":
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_id = f"{reify_prefix}{eid}"
            
            he_attrs = _public(edge_attrs.get(eid, {}))
            he_attrs.update({
                "is_hyperedge": True,
                "eid": eid,
                "directed": bool(_is_directed_eid(graph, eid)),
                "hyper_weight": float(weights_map.get(eid, 1.0)),
            })
            
            he_nodes_to_add.append((he_id, he_attrs))

            if _is_directed_eid(graph, eid):
                for u, coeff in (tail_map or {}).items():
                    membership_edges_to_add.append((
                        u, he_id,
                        f"m::{eid}::{u}::tail",
                        {"role": "tail", "coeff": float(coeff), "membership_of": eid}
                    ))
                for v, coeff in (head_map or {}).items():
                    membership_edges_to_add.append((
                        he_id, v,
                        f"m::{eid}::{v}::head",
                        {"role": "head", "coeff": float(coeff), "membership_of": eid}
                    ))
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if is_multi_di:
                    for u, coeff in members.items():
                        base = f"m::{eid}::{u}::m"
                        membership_edges_to_add.append((
                            u, he_id,
                            base + "::fwd",
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        ))
                        membership_edges_to_add.append((
                            he_id, u,
                            base + "::rev",
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        ))
                else:
                    for u, coeff in members.items():
                        membership_edges_to_add.append((
                            u, he_id,
                            f"m::{eid}::{u}::m",
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        ))

        # BULK ADD TO NETWORKX
        for he_id, attrs in he_nodes_to_add:
            if he_id not in nxG:
                nxG.add_node(he_id, **attrs)
        
        for u, v, key, attrs in membership_edges_to_add:
            nxG.add_edge(u, v, key=key, **attrs)

    # Multilayer metadata
    aspects = list(getattr(graph, "aspects", []))
    elem_layers = dict(getattr(graph, "elem_layers", {}))
    VM_serialized = _serialize_VM(getattr(graph, "_VM", set()))
    edge_kind = dict(edge_kind_dict)
    edge_layers_ser = _serialize_edge_layers(getattr(graph, "edge_layers", {}))
    node_layer_attrs_ser = _serialize_node_layer_attrs(getattr(graph, "_vertex_layer_attrs", {}))
    aspect_attrs = dict(getattr(graph, "_aspect_attrs", {}))
    layer_tuple_attrs_ser = _serialize_layer_tuple_attrs(getattr(graph, "_layer_attrs", {}))
    layer_attr_rows = _safe_df_to_rows(getattr(graph, "layer_attributes", None))

    # BUILD EDGE_DIRECTED DICT ONCE
    edge_directed_dict = {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids}

    manifest = {
        "edges": manifest_edges,
        "weights": weights_map,
        "slices": slices_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "slice_weights": slice_weights,
        "edge_directed": edge_directed_dict,
        "manifest_version": 1,
        "multilayer": {
            "aspects": aspects,
            "aspect_attrs": aspect_attrs,
            "elem_layers": elem_layers,
            "VM": VM_serialized,
            "edge_kind": edge_kind,
            "edge_layers": edge_layers_ser,
            "node_layer_attrs": node_layer_attrs_ser,
            "layer_tuple_attrs": layer_tuple_attrs_ser,
            "layer_attributes": layer_attr_rows,
        },
    }

    return nxG, manifest

def save_manifest(manifest: dict, path: str):
    """Write manifest to JSON file.

    Parameters
    ----------
    manifest : dict
        Manifest dictionary from to_nx().
    path : str
        Output file path (typically .json extension).

    Returns
    -------
    None

    Raises
    ------
    OSError
        If file cannot be written.

    """
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> dict:
    """Load manifest from JSON file.

    Parameters
    ----------
    path : str
        Path to manifest JSON file created by save_manifest().

    Returns
    -------
    dict
        Manifest dictionary suitable for from_nx().

    Raises
    ------
    OSError
        If file cannot be read.
    json.JSONDecodeError
        If file contains invalid JSON.

    """
    with open(path) as f:
        return json.load(f)


def _nx_collect_reified(
    nxG,
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
):
    """Scan nxG for reified hyperedges and return:
    - hyperdefs: list of (eid, directed:bool, head_map:dict, tail_map:dict, he_node_attrs:dict, he_node_id)
    - membership_edges: set of (u,v,key) tuples to skip when also importing binary edges
    """

    if nxG.is_multigraph():
        edge_iter = nxG.edges(keys=True, data=True)

        def EK(u, v, k):
            return (u, v, k)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in nxG.edges(data=True))

        def EK(u, v, k):
            return (u, v, None)

    # find HE nodes
    he_nodes = {n for n, d in nxG.nodes(data=True) if (d or {}).get(he_node_flag, False)}
    if not he_nodes:
        return [], set()

    hyperdefs = []
    membership_edges = set()

    for he in he_nodes:
        nd = dict(nxG.nodes[he])
        eid = nd.get(he_id_attr, f"he::{he}")

        head_map, tail_map = {}, {}
        saw_head = False
        saw_tail = False
        saw_member = False

        # collect all incident edges around the HE node
        for u, v, key, d in edge_iter:
            if u != he and v != he:
                continue
            membership_edges.add(EK(u, v, key))
            other = v if u == he else u
            role = (d or {}).get(role_attr, None)
            coeff = (d or {}).get(coeff_attr, d.get("__value", 1.0))
            try:
                coeff = float(coeff)
            except Exception:
                coeff = 1.0

            if role == "head":
                head_map[other] = coeff
                saw_head = True
            elif role == "tail":
                tail_map[other] = coeff
                saw_tail = True
            else:
                # treat as undirected membership
                head_map[other] = coeff
                tail_map[other] = coeff
                saw_member = True

        # decide directedness
        if saw_head or saw_tail:
            directed = True
        else:
            directed = False  # only 'member' edges

        hyperdefs.append((eid, directed, head_map, tail_map, nd, he))

    return hyperdefs, membership_edges


def from_nx(
    nxG,
    manifest,
    *,
    hyperedge="none",
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
    reify_prefix="he::",
) -> AnnNet:
    """Reconstruct a AnnNet from NetworkX graph + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in nxG and rebuild true hyperedges
      (in addition to those specified in the manifest).
    """
    from ..core.graph import AnnNet

    H = AnnNet()
    timings = {}

    known_vertices = set()
    existing_eids = set()
    existing_slices = set(H.list_slices(include_default=True))
    
    edge_directed_cache = manifest.get("edge_directed", {}) or {}
    weights_cache = manifest.get("weights", {}) or {}
    vertex_attrs_cache = manifest.get("vertex_attrs", {}) or {}
    edge_attrs_cache = manifest.get("edge_attrs", {}) or {}
    slices_cache = manifest.get("slices", {}) or {}
    slice_weights_cache = manifest.get("slice_weights", {}) or {}
    edges_def = manifest.get("edges", {}) or {}
    mm = manifest.get("multilayer", {})

    def add_vertex_once(v):
        if v not in known_vertices:
            known_vertices.add(v)

    with _time("nx_vertices", timings):
        try:
            for v, d in nxG.nodes(data=True):
                if hyperedge == "reified":
                    d = d or {}
                    if bool(d.get(he_node_flag, False)):
                        continue
                    if isinstance(v, str) and v.startswith(reify_prefix):
                        continue
                add_vertex_once(v)
        except Exception:
            pass

    # BULK INSERT VERTICES ONCE
    if known_vertices:
        H.add_vertices_bulk([{"vertex_id": v} for v in known_vertices])

    regular_edges_bulk = []
    hyperedges_bulk = []

    with _time("manifest_edges", timings):
        for eid, defn in edges_def.items():
            kind = defn[-1]
            is_dir = bool(edge_directed_cache.get(eid, True))

            if kind == "hyper":
                head_map, tail_map = defn[0], defn[1]
                if isinstance(head_map, dict) and isinstance(tail_map, dict):
                    head = list(head_map)
                    tail = list(tail_map)
                    all_vertices = set(head) | set(tail)
                    
                    for x in all_vertices:
                        add_vertex_once(x)

                    # BUILD ATTRIBUTES INLINE
                    attrs = {
                        "__source_attr": {u: {"__value": float(c)} for u, c in head_map.items()},
                        "__target_attr": {v: {"__value": float(c)} for v, c in tail_map.items()},
                    }

                    if is_dir:
                        hyperedges_bulk.append({
                            "head": head,
                            "tail": tail,
                            "edge_id": eid,
                            "edge_directed": True,
                            "attributes": attrs,
                        })
                    else:
                        hyperedges_bulk.append({
                            "members": list(all_vertices),
                            "edge_id": eid,
                            "edge_directed": False,
                            "attributes": attrs,
                        })
                    existing_eids.add(eid)
            else:
                u, v = defn[0], defn[1]
                regular_edges_bulk.append({
                    "source": u,
                    "target": v,
                    "edge_id": eid,
                    "edge_directed": is_dir,
                })
                existing_eids.add(eid)

        # SINGLE BULK CALL FOR REGULAR EDGES
        if regular_edges_bulk:
            H.add_edges_bulk(regular_edges_bulk, default_edge_directed=True)

        # SINGLE BULK CALL FOR HYPEREDGES
        if hyperedges_bulk:
            H.add_hyperedges_bulk(hyperedges_bulk)

    # BATCH WEIGHTS
    with _time("weights", timings):
        if weights_cache:
            for eid, w in weights_cache.items():
                try:
                    H.edge_weights[eid] = float(w)
                except Exception:
                    pass

    # BATCH SLICES
    with _time("slices", timings):
        for lid, eids in slices_cache.items():
            if lid not in existing_slices:
                try:
                    H.add_slice(lid)
                    existing_slices.add(lid)
                except Exception:
                    pass
            if eids:
                H.add_edges_to_slice_bulk(lid, eids)

        for lid, per_edge in slice_weights_cache.items():
            if lid not in existing_slices:
                try:
                    H.add_slice(lid)
                    existing_slices.add(lid)
                except Exception:
                    pass
            for eid, w in per_edge.items():
                try:
                    H.set_edge_slice_attrs(lid, eid, weight=float(w))
                except Exception:
                    pass

    with _time("multilayer", timings):                    
        aspects = mm.get("aspects", [])
        elem_layers = mm.get("elem_layers", {})

        if aspects:
            H.aspects = list(aspects)
            H.elem_layers = dict(elem_layers or {})
            H._rebuild_all_layers_cache()

        aspect_attrs = mm.get("aspect_attrs", {})
        if aspect_attrs:
            H._aspect_attrs.update(aspect_attrs)

        VM_data = mm.get("VM", [])
        if VM_data:
            H._VM = _deserialize_VM(VM_data)

        ek = mm.get("edge_kind", {})
        el_ser = mm.get("edge_layers", {})
        if ek:
            H.edge_kind.update(ek)
        if el_ser:
            H.edge_layers.update(_deserialize_edge_layers(el_ser))

        nl_attrs_ser = mm.get("node_layer_attrs", [])
        if nl_attrs_ser:
            H._vertex_layer_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

        layer_tuple_attrs_ser = mm.get("layer_tuple_attrs", [])
        if layer_tuple_attrs_ser:
            H._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

        layer_attr_rows = mm.get("layer_attributes", [])
        if layer_attr_rows:
            H.layer_attributes = _rows_to_df(layer_attr_rows)

    # BATCH ATTRIBUTES
    with _time("attrs", timings):
        if vertex_attrs_cache:
            for vid, attrs in vertex_attrs_cache.items():
                if attrs:
                    try:
                        H.set_vertex_attrs(vid, **attrs)
                    except Exception:
                        pass
        
        if edge_attrs_cache:
            for eid, attrs in edge_attrs_cache.items():
                if attrs:
                    try:
                        H.set_edge_attrs(eid, **attrs)
                    except Exception:
                        pass

    if hyperedge == "reified":
        with _time("reified", timings):
            hyperdefs, membership_edges = _nx_collect_reified_optimized(
                nxG,
                he_node_flag=he_node_flag,
                he_id_attr=he_id_attr,
                role_attr=role_attr,
                coeff_attr=coeff_attr,
                membership_attr=membership_attr,
            )
            
            reified_hyperedges_bulk = []
            
            for eid, directed, head_map, tail_map, he_attrs, he_node in hyperdefs:
                if eid in existing_eids:
                    continue
                
                all_vertices = set(head_map) | set(tail_map)
                for x in all_vertices:
                    add_vertex_once(x)
                
                # BUILD ATTRIBUTES INLINE
                attrs = {}
                if directed:
                    attrs["__source_attr"] = {u: {"__value": c} for u, c in head_map.items()}
                    attrs["__target_attr"] = {v: {"__value": c} for v, c in tail_map.items()}
                else:
                    attrs["__source_attr"] = {u: {"__value": head_map[u]} for u in all_vertices if u in head_map}
                    attrs["__target_attr"] = {v: {"__value": tail_map[v]} for v in all_vertices if v in tail_map}

                clean_attrs = {k: v for k, v in (he_attrs or {}).items() if k not in {he_node_flag, he_id_attr}}
                if clean_attrs:
                    attrs.update(clean_attrs)

                if directed:
                    reified_hyperedges_bulk.append({
                        "head": list(head_map),
                        "tail": list(tail_map),
                        "edge_id": eid,
                        "edge_directed": True,
                        "attributes": attrs,
                    })
                else:
                    reified_hyperedges_bulk.append({
                        "members": list(all_vertices),
                        "edge_id": eid,
                        "edge_directed": False,
                        "attributes": attrs,
                    })
                existing_eids.add(eid)

            # SINGLE BULK CALL FOR REIFIED HYPEREDGES
            if reified_hyperedges_bulk:
                H.add_hyperedges_bulk(reified_hyperedges_bulk)

    return H

def _nx_collect_reified_optimized(
    nxG, *, he_node_flag, he_id_attr, role_attr, coeff_attr, membership_attr
):
    """Optimized: pre-index edges by incident node to avoid O(H×E) full scans."""
    
    node_to_edges = {}
    for u, v, d in nxG.edges(data=True):
        if u not in node_to_edges:
            node_to_edges[u] = []
        if v not in node_to_edges:
            node_to_edges[v] = []
        node_to_edges[u].append((u, v, d))
        node_to_edges[v].append((u, v, d))
    
    hyperdefs = []
    membership_edges = []
    
    for he_node, he_attrs in nxG.nodes(data=True):
        he_attrs = he_attrs or {}
        if not bool(he_attrs.get(he_node_flag, False)):
            continue
        
        eid = he_attrs.get(he_id_attr, he_node)
        
        head_map = {}
        tail_map = {}
        directed = False
        
        incident_edges = node_to_edges.get(he_node, [])
        
        for u, v, d in incident_edges:
            d = d or {}
            if membership_attr in d:
                membership_edges.append((u, v, d))
                
                other = v if u == he_node else u
                role = d.get(role_attr, "member")
                coeff = d.get(coeff_attr, 1.0)
                
                if role == "head":
                    head_map[other] = coeff
                    directed = True
                elif role == "tail":
                    tail_map[other] = coeff
                    directed = True
                else:
                    head_map[other] = coeff
                    tail_map[other] = coeff
        
        if head_map or tail_map:
            hyperdefs.append((eid, directed, head_map, tail_map, he_attrs, he_node))
    
    return hyperdefs, membership_edges

def to_backend(graph, **kwargs):
    """Export AnnNet to NetworkX without manifest (legacy compatibility).

    Parameters
    ----------
    graph : AnnNet
        Source AnnNet instance to export.
    **kwargs
        Forwarded to _export_legacy(). Supported:
        - directed : bool, default True
            Export as MultiDiGraph (True) or MultiGraph (False).
        - skip_hyperedges : bool, default True
            If True, drop hyperedges. If False, expand them
            (cartesian product for directed, clique for undirected).
        - public_only : bool, default False
            Strip attributes starting with "__" if True.

    Returns
    -------
    networkx.MultiGraph | networkx.MultiDiGraph
        NetworkX graph containing binary edges only. Hyperedges are
        either dropped or expanded. No manifest is returned, so
        round-tripping will lose hyperedge structure, slices, and
        precise edge IDs.

    Notes
    -----
    This is a lossy export. Use to_nx() with manifest for full fidelity.

    """
    return _export_legacy(graph, **kwargs)


def from_nx_only(
    nxG,
    *,
    hyperedge="none",
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
    reify_prefix="he::",
):
    """Best-effort import from a bare NetworkX graph (no manifest).
    hyperedge: "none" (default) | "reified"
      When "reified", detect hyperedge nodes + membership edges and rebuild true hyperedges.
    """

    from ..core.graph import AnnNet

    H = AnnNet()

    # 1) Nodes + node attributes (verbatim, but skip HE nodes if reified)
    for v, d in nxG.nodes(data=True):
        if hyperedge == "reified":
            if bool((d or {}).get(he_node_flag, False)):
                continue
            if isinstance(v, str) and str(v).startswith(reify_prefix):
                continue
        try:
            H.add_vertex(v)
        except Exception:
            pass
        if d:
            try:
                H.set_vertex_attrs(v, **dict(d))
            except Exception:
                pass

    # 2) Optionally collect reified hyperedges
    membership_edges = set()
    if hyperedge == "reified":
        hyperdefs, membership_edges = _nx_collect_reified(
            nxG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, he_attrs, he_node in hyperdefs:
            for x in set(head_map) | set(tail_map):
                try:
                    H.add_vertex(x)
                except Exception:
                    pass
            if directed:
                try:
                    H.add_hyperedge(
                        head=list(head_map), tail=list(tail_map), edge_id=eid, edge_directed=True
                    )
                except Exception:
                    pass
                H.set_edge_attrs(
                    eid,
                    __source_attr={u: {"__value": c} for u, c in head_map.items()},
                    __target_attr={v: {"__value": c} for v, c in tail_map.items()},
                )
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_hyperedge(members=members, edge_id=eid, edge_directed=False)
                except Exception:
                    pass
                H.set_edge_attrs(
                    eid,
                    __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                    __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members},
                )
            # copy HE node attrs (minus markers)
            clean_attrs = {
                k: v for k, v in (he_attrs or {}).items() if k not in {he_node_flag, he_id_attr}
            }
            if clean_attrs:
                try:
                    H.set_edge_attrs(eid, **clean_attrs)
                except Exception:
                    pass

    # 3) Binary edges (skip membership edges if we consumed them above)
    is_multi = nxG.is_multigraph()
    is_dir = nxG.is_directed()
    if is_multi:
        iterator = nxG.edges(keys=True, data=True)

        def EK(u, v, k):
            return (u, v, k)
    else:
        iterator = ((u, v, None, d) for u, v, d in nxG.edges(data=True))

        def EK(u, v, k):
            return (u, v, None)

    seen_auto = 0
    for u, v, key, d in iterator:
        if hyperedge == "reified" and EK(u, v, key) in membership_edges:
            continue  # this edge was a membership edge; skip importing as binary

        eid = (d.get("eid") if isinstance(d, dict) else None) or (key if key is not None else None)
        if eid is None:
            seen_auto += 1
            eid = f"nx::e#{seen_auto}"

        e_directed = bool(d.get("directed", is_dir))
        w = d.get("weight", d.get("__weight", 1.0))

        try:
            H.add_vertex(u)
            H.add_vertex(v)
        except Exception:
            pass
        try:
            H.add_edge(u, v, edge_id=eid, edge_directed=e_directed)
        except Exception:
            H.add_edge(u, v, edge_id=eid, edge_directed=True)

        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

        if isinstance(d, dict) and d:
            try:
                H.set_edge_attrs(eid, **dict(d))
            except Exception:
                pass

    try:
        H.edge_weights[eid] = float(w)
    except Exception:
        pass

    # Copy edge attributes but drop structural keys that shouldn't live in user attrs
    if isinstance(d, dict) and d:
        try:
            clean = dict(d)
            # structural keys used for reconstruction only
            for k in ("eid", "weight", "__weight", "directed"):
                clean.pop(k, None)
            if clean:
                clean = dict(d)
                for k in ("eid", "id", "key", "weight", "__weight", "directed"):
                    clean.pop(k, None)
                if clean:
                    H.set_edge_attrs(eid, **clean)
        except Exception:
            pass
    return H


class NetworkXAdapter:
    def export(self, graph, **kwargs):
        return _export_legacy(graph, **kwargs)
