from __future__ import annotations

from typing import TYPE_CHECKING

from .._dataframe_backend import empty_dataframe
from ._utils import (
    _attrs_to_dict,
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_VM,
    _df_to_rows,
    _endpoint_coeff_map,
    _is_directed_eid,
    _rows_like,
    _rows_to_df,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_value,
    _serialize_VM,
    load_manifest,
    save_manifest,
)

if TYPE_CHECKING:
    from ..core.graph import AnnNet


def _collect_slices_and_weights(graph) -> tuple[dict, dict]:
    """Returns:
      slices_section: {slice_id: [edge_id, ...]}
      slice_weights:  {slice_id: {edge_id: weight}}
    Backends supported: Polars-like, .to_dicts()-like, dict.

    """
    slices_section: dict = {}
    slice_weights: dict = {}

    # --- Source A: edge_slice_attributes table (Polars-like)
    t = getattr(graph, "edge_slice_attributes", None)
    if t is not None and hasattr(t, "filter"):
        try:
            # Attempt Polars path without hard dependency

            # Get all rows then group by slice in Python (keeps us backend-agnostic)
            if hasattr(t, "to_dicts"):
                rows = t.to_dicts()
            else:
                # last-ditch: try turning the entire table into a list of dicts
                # many DataFrame-likes support .rows(named=True)
                rows = getattr(t, "rows", lambda named=False: [])(named=True)  # type: ignore
            for r in rows:
                lid = r.get("slice")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                slices_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    slice_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass  # fall through to other sources

    # --- Source B: edge_slice_attributes with .to_dicts() but no Polars
    if not slices_section and t is not None and hasattr(t, "to_dicts"):
        try:
            for r in t.to_dicts():
                lid = r.get("slice")
                if lid is None:
                    continue
                eid = r.get("edge_id", r.get("edge"))
                if eid is None:
                    continue
                slices_section.setdefault(lid, []).append(eid)
                w = r.get("weight")
                if w is not None:
                    slice_weights.setdefault(lid, {})[eid] = float(w)
        except Exception:
            pass

    # --- Source C: dict mapping slice -> {edge_id: attrs}
    if not slices_section and isinstance(t, dict):
        for lid, ed in t.items():
            if isinstance(ed, dict):
                eids = list(ed.keys())
                slices_section.setdefault(lid, []).extend(eids)
                # pick weights if present
                for eid, attrs in ed.items():
                    if (
                        isinstance(attrs, dict)
                        and "weight" in attrs
                        and attrs["weight"] is not None
                    ):
                        slice_weights.setdefault(lid, {})[eid] = float(attrs["weight"])

    # --- Fallback D: per-edge scan (if graph exposes edge iteration + get_edge_slices)
    if not slices_section:
        edges_iter = None
        for attr in ("edges", "iter_edges", "edge_ids"):
            if hasattr(graph, attr):
                try:
                    edges_iter = list(getattr(graph, attr)())
                    break
                except Exception:
                    pass
        if edges_iter:
            for eid in edges_iter:
                lids = []
                for probe in ("get_edge_slices", "edge_slices"):
                    if hasattr(graph, probe):
                        try:
                            lids = list(getattr(graph, probe)(eid))
                            break
                        except Exception:
                            pass
                for lid in lids or []:
                    slices_section.setdefault(lid, []).append(eid)

    # --- Collect per-slice weight overrides using canonical accessor
    if hasattr(graph, "get_edge_slice_attr"):
        for lid, eids in list(slices_section.items()):
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_slice_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        # some implementations don't support default=
                        w = graph.get_edge_slice_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    slice_weights.setdefault(lid, {})[eid] = float(w)

    # Ensure deterministic and non-empty lists in manifest
    for lid, eids in slices_section.items():
        # unique, stable order
        seen = set()
        uniq = []
        for e in eids:
            if e not in seen:
                seen.add(e)
                uniq.append(e)
        slices_section[lid] = uniq

    return slices_section, slice_weights


def _safe_df_to_rows(df):
    """Never crash if df is None or backend is missing."""
    if df is None:
        return []
    try:
        return _df_to_rows(df)
    except Exception:
        # fallback to generic rows() conversion
        return _rows_like(df)


def _export_binary_graph(
    graph: AnnNet,
    *,
    directed: bool = True,
    skip_hyperedges: bool = True,
    public_only: bool = False,
):
    """Export AnnNet to igraph.AnnNet without manifest.

    igraph requires integer vertex indices; external vertex IDs are preserved
    in vertex attribute 'name'. Edge IDs stored in edge attribute 'eid'.

    Parameters
    ----------
    graph : AnnNet
        Source graph instance.
    directed : bool
        If True, export as directed igraph.AnnNet; else undirected.
        Undirected edges in directed export are emitted bidirectionally.
    skip_hyperedges : bool
        If True, drop hyperedges. If False:
          - directed hyperedges expand head×tail (cartesian product)
          - undirected hyperedges expand to clique
    public_only : bool
        If True, strip private attrs starting with "__".

    Returns
    -------
    igraph.AnnNet

    """
    import igraph as ig

    # Build the vertex universe robustly
    # Start with declared vertices
    base_vertices = set(graph.vertices())

    # Ensure endpoints that appear in edges are also included
    endpoints = set()
    for eidx in range(graph.ne):
        eid = graph._col_to_edge[eidx]
        rec = graph._edges[eid]
        if rec.etype == "hyper":
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}
        endpoints.update(S)
        endpoints.update(T)

    # If we are going to expand hyperedges, include their members/head/tail too
    if not skip_hyperedges:
        for rec in graph._edges.values():
            if rec.col_idx < 0 or rec.etype != "hyper":
                continue
            endpoints.update(rec.src or [])
            endpoints.update(rec.tgt or [])

    vertices = list(dict.fromkeys(list(base_vertices) + [v for v in endpoints]))  # stable order
    vidx = {v: i for i, v in enumerate(vertices)}

    # Create igraph graph and set vertex 'name'
    G = ig.Graph(directed=bool(directed))
    G.add_vertices(len(vertices))
    G.vs["name"] = vertices

    # Attach vertex attributes (works for both vertices and edge-entities)
    # Polars-safe extraction; ignore if table missing/empty.
    vtab = getattr(graph, "vertex_attributes", None)
    # Pre-scan to a dict for O(1) lookup
    vattr_map = {}
    try:
        if (
            vtab is not None
            and hasattr(vtab, "to_dicts")
            and vtab.height > 0
            and "vertex_id" in vtab.columns
        ):
            for row in vtab.to_dicts():
                d = dict(row)
                vid = d.pop("vertex_id", None)
                if vid is not None:
                    vattr_map[vid] = d
    except Exception:
        vattr_map = {}

    processed_vattrs = {}
    for v in vertices:
        v_attr = dict(vattr_map.get(v, {}))
        if public_only:
            v_attr = {
                k: _serialize_value(val) for k, val in v_attr.items() if not str(k).startswith("__")
            }
        else:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()}
        processed_vattrs[v] = v_attr

    all_vattr_keys = set().union(*processed_vattrs.values()) if processed_vattrs else set()
    for k in all_vattr_keys:
        G.vs[k] = [processed_vattrs[v].get(k) for v in vertices]

    # Helper: directedness per edge-id (fallback if helper missing)
    def _is_dir_eid(g, eid):
        try:
            return _is_directed_eid(g, eid)  # use existing helper if present
        except NameError:
            rec = getattr(g, "_edges", {}).get(eid)
            if rec is not None and rec.directed is not None:
                return bool(rec.directed)
            return bool(getattr(g, "directed", True))

    # Add edges (binary & vertex-edge). Hyperedges: skip or expand
    eattr_map = {}
    try:
        for row in _rows_like(getattr(graph, "edge_attributes", None)):
            eid = row.get("edge_id")
            if eid is not None:
                d = dict(row)
                d.pop("edge_id", None)
                eattr_map[eid] = d
    except Exception:
        eattr_map = {}
    # collect all edges as tuples first, write them in one bulk call
    edge_tuples = []
    edge_payloads = []  # list of dicts, parallel to edge_tuples

    for eidx in range(graph.ne):
        eid = graph._col_to_edge[eidx]
        rec = graph._edges[eid]
        if rec.etype == "hyper":
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}

        e_attr = dict(eattr_map.get(eid, {}))
        if public_only:
            e_attr = {
                k: _serialize_value(val) for k, val in e_attr.items() if not str(k).startswith("__")
            }
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = 1.0 if rec.weight is None else rec.weight
        e_attr["weight" if public_only else "__weight"] = weight
        e_attr["eid"] = eid

        is_hyper = rec.etype == "hyper"
        is_dir = _is_dir_eid(graph, eid)
        members = S | T

        if not is_hyper and len(members) <= 2:
            if len(members) == 1:
                u = next(iter(members))
                if u not in vidx:
                    continue
                edge_tuples.append((vidx[u], vidx[u]))
                edge_payloads.append(e_attr)
            else:
                if is_dir:
                    uu, vv = next(iter(S)), next(iter(T))
                    if uu in vidx and vv in vidx:
                        edge_tuples.append((vidx[uu], vidx[vv]))
                        edge_payloads.append(e_attr)
                else:
                    u, v = tuple(members)
                    if u in vidx and v in vidx:
                        if directed:
                            edge_tuples.append((vidx[u], vidx[v]))
                            edge_payloads.append(e_attr)
                            edge_tuples.append((vidx[v], vidx[u]))
                            edge_payloads.append(e_attr)
                        else:
                            edge_tuples.append((vidx[u], vidx[v]))
                            edge_payloads.append(e_attr)
            continue

        if skip_hyperedges:
            continue

        if is_dir:
            for u in S:
                for v in T:
                    if u not in vidx or v not in vidx:
                        continue
                    p = dict(e_attr)
                    if not directed:
                        p["directed"] = True
                    edge_tuples.append((vidx[u], vidx[v]))
                    edge_payloads.append(p)
        else:
            mem = [m for m in members if m in vidx]
            n = len(mem)
            pairs = (
                [(a, b) for a in range(n) for b in range(n) if a != b]
                if directed
                else [(a, b) for a in range(n) for b in range(a + 1, n)]
            )
            for a, b in pairs:
                edge_tuples.append((vidx[mem[a]], vidx[mem[b]]))
                edge_payloads.append(e_attr)

    # ONE bulk C-level call instead of individual add_edge() calls
    G.add_edges(edge_tuples)

    # Write edge attrs column-wise — one C call per attribute key
    if edge_payloads:
        all_eattr_keys = set().union(*edge_payloads)
        for k in all_eattr_keys:
            G.es[k] = [p.get(k) for p in edge_payloads]

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


def to_igraph(
    graph: AnnNet,
    directed=True,
    hyperedge_mode="skip",
    slice=None,
    slices=None,
    public_only=False,
    reify_prefix="he::",
):
    """Export AnnNet → (igraph.AnnNet, manifest).

    hyperedge_mode: {"skip","expand","reify"}
      - "skip": drop HE edges from igG (manifest keeps them)
      - "expand": cartesian product (directed) / clique (undirected)
      - "reify": add a node per HE and membership edges V↔HE carrying roles/coeffs
    """
    # -------------- base igraph build (binary edges only) --------------
    # For "reify" we start with hyperedges skipped, then add them as nodes+membership edges.
    igG = _export_binary_graph(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ("skip", "reify")),
        public_only=public_only,
    )

    # -------------- collect vertex/edge attrs for manifest --------------
    _raw_vertex_attrs = {
        row["vertex_id"]: {k: v for k, v in row.items() if k != "vertex_id"}
        for row in _rows_like(getattr(graph, "vertex_attributes", None))
        if row.get("vertex_id") is not None
    }
    vertex_attrs = {
        v: _attrs_to_dict(
            {
                k: val
                for k, val in _raw_vertex_attrs.get(v, {}).items()
                if not public_only or not str(k).startswith("__")
            }
        )
        for v in graph.vertices()
    }

    _raw_edge_attrs = {
        row["edge_id"]: {k: v for k, v in row.items() if k != "edge_id"}
        for row in _rows_like(getattr(graph, "edge_attributes", None))
        if row.get("edge_id") is not None
    }
    edge_attrs = {
        graph._col_to_edge[eidx]: _attrs_to_dict(
            {
                k: val
                for k, val in _raw_edge_attrs.get(graph._col_to_edge[eidx], {}).items()
                if not public_only or not str(k).startswith("__")
            }
        )
        for eidx in range(graph.ne)
    }

    # -------------- topology snapshot (regular vs hyper) --------------
    manifest_edges = {}
    for eidx in range(graph.ne):
        eid = graph._col_to_edge[eidx]
        rec = graph._edges[eid]
        is_hyper = rec.etype == "hyper"
        if is_hyper:
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}
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

    # ---------- slices + per-slice weights for manifest ----------
    all_eids = list(manifest_edges.keys())

    # discover slices
    lids = set()
    try:
        lids.update(list(graph.list_slices(include_default=True)))
    except Exception:
        try:
            lids.update(list(graph.list_slices()))
        except Exception:
            pass

    slices_section = {lid: [] for lid in lids}
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

    t = getattr(graph, "edge_slice_attributes", None)
    if isinstance(t, dict):
        for lid, mapping in t.items():
            if isinstance(mapping, dict):
                arr = slices_section.setdefault(lid, [])
                seen = set(arr)
                for eid in list(mapping.keys()):
                    if eid not in seen:
                        arr.append(eid)
                        seen.add(eid)
    for r in _rows_like(t):
        lid = r.get("slice") or r.get("slice_id") or r.get("lid")
        if lid is None:
            continue
        eid = r.get("edge_id", r.get("edge"))
        if eid is not None:
            arr = slices_section.setdefault(lid, [])
            if eid not in arr:
                arr.append(eid)

    etl = getattr(graph, "edge_to_slices", None)
    if isinstance(etl, dict):
        for eid, arr_lids in etl.items():
            for lid in arr_lids or []:
                arr = slices_section.setdefault(lid, [])
                if eid not in arr:
                    arr.append(eid)

    le = getattr(graph, "slice_edges", None)
    if isinstance(le, dict):
        for lid, eids in le.items():
            arr = slices_section.setdefault(lid, [])
            seen = set(arr)
            for eid in list(eids):
                if eid not in seen:
                    arr.append(eid)
                    seen.add(eid)

    # remove empties
    slices_section = {lid: eids for lid, eids in slices_section.items() if eids}

    # filter slices if requested
    requested_lids = set()
    if slice is not None:
        requested_lids.update([slice] if isinstance(slice, str) else list(slice))
    if slices is not None:
        requested_lids.update(list(slices))
    if requested_lids:
        req_norm = {str(x) for x in requested_lids}
        slices_section = {lid: eids for lid, eids in slices_section.items() if str(lid) in req_norm}

    # per-slice weights
    slice_weights = {}
    if hasattr(graph, "get_edge_slice_attr"):
        for lid, eids in slices_section.items():
            for eid in eids:
                w = None
                try:
                    w = graph.get_edge_slice_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_slice_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    slice_weights.setdefault(lid, {})[eid] = float(w)

    # baseline weights
    try:
        base_weights = {
            eid: float(rec.weight) for eid, rec in graph._edges.items() if rec.weight is not None
        }
    except Exception:
        base_weights = {}

    # -------------- REIFY: add HE nodes + membership edges to igG --------------
    if hyperedge_mode == "reify":
        # allowed HE set under slice filter (None => all)
        allowed = None
        if requested_lids:
            allowed = set()
            for lid, eids in slices_section.items():
                for eid in eids:
                    allowed.add(eid)

        # ensure we have a 'name' attribute for vertices
        if "name" not in igG.vs.attributes():
            igG.vs["name"] = list(range(igG.vcount()))
        names = list(igG.vs["name"])
        name_to_idx = {str(n): i for i, n in enumerate(names)}

        def ensure_vertex(name: str) -> int:
            sname = str(name)
            if sname in name_to_idx:
                return name_to_idx[sname]
            # add with name set
            igG.add_vertices([sname])
            idx = igG.vcount() - 1
            # igraph sets 'name' from the list given to add_vertices
            name_to_idx[sname] = idx
            return idx

        new_edges = []
        payloads = []  # dict per new edge

        for eid, spec in manifest_edges.items():
            if spec[-1] != "hyper":
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_name = f"{reify_prefix}{eid}"
            he_idx = ensure_vertex(he_name)

            # copy selected edge attrs to HE node (public only)
            he_attrs = edge_attrs.get(eid, {}) or {}
            if public_only:
                he_attrs = {k: v for k, v in he_attrs.items() if not str(k).startswith("__")}
            igG.vs[he_idx]["is_hyperedge"] = True
            igG.vs[he_idx]["eid"] = eid
            igG.vs[he_idx]["directed"] = bool(_is_directed_eid(graph, eid))
            igG.vs[he_idx]["hyper_weight"] = float(base_weights.get(eid, 1.0))
            # also copy public user attrs
            for k, v in he_attrs.items():
                igG.vs[he_idx][k] = v

            if _is_directed_eid(graph, eid):
                # tail -> HE
                for u, coeff in (tail_map or {}).items():
                    ui = ensure_vertex(u)
                    new_edges.append((ui, he_idx))
                    payloads.append({"role": "tail", "coeff": float(coeff), "membership_of": eid})
                # HE -> head
                for v, coeff in (head_map or {}).items():
                    vi = ensure_vertex(v)
                    new_edges.append((he_idx, vi))
                    payloads.append({"role": "head", "coeff": float(coeff), "membership_of": eid})
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if directed:  # directed container: add both directions
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )
                        new_edges.append((he_idx, ui))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )
                else:  # undirected container: one edge is enough
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {"role": "member", "coeff": float(coeff), "membership_of": eid}
                        )

        if new_edges:
            start = igG.ecount()
            igG.add_edges(new_edges)
            # set attributes for the newly added membership edges
            keys = set().union(*(d.keys() for d in payloads))
            for k in keys:
                igG.es[start:][k] = [d.get(k) for d in payloads]

    # ----------------- multilayer / Kivela metadata -----------------
    aspects = list(getattr(graph, "aspects", []))
    elem_layers = dict(getattr(graph, "elem_layers", {}))
    VM_serialized = _serialize_VM(getattr(graph, "_VM", set()))
    edge_kind = {
        eid: ("hyper" if rec.etype == "hyper" else rec.ml_kind)
        for eid, rec in graph._edges.items()
        if rec.col_idx >= 0 and (rec.etype == "hyper" or rec.ml_kind is not None)
    }
    edge_layers_ser = _serialize_edge_layers(getattr(graph, "edge_layers", {}))
    node_layer_attrs_ser = _serialize_node_layer_attrs(getattr(graph, "_state_attrs", {}))

    # aspect and layer-tuple level attributes (dicts)
    aspect_attrs = dict(getattr(graph, "_aspect_attrs", {}))
    layer_tuple_attrs_ser = _serialize_layer_tuple_attrs(getattr(graph, "_layer_attrs", {}))
    layer_df = getattr(graph, "layer_attributes", None)
    if layer_df is None:
        layer_df = empty_dataframe({})
    layer_attr_rows = _safe_df_to_rows(layer_df)

    # -------------- manifest (unchanged semantics) --------------
    manifest = {
        "edges": manifest_edges,
        "weights": base_weights,
        "slices": slices_section,
        "vertex_attrs": vertex_attrs,
        "edge_attrs": edge_attrs,
        "slice_weights": slice_weights,
        "edge_directed": {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids},
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

    return igG, manifest


def _ig_collect_reified(
    igG,
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
):
    """Scan igG for reified hyperedges.

    Returns:
      - hyperdefs: list of (eid, directed:bool, head_map:dict, tail_map:dict, he_node_attrs:dict, he_vertex_index)
      - membership_edge_idx: set of edge indices that are membership edges (to skip for binary import)

    """
    import math

    vattrs = set(igG.vs.attributes())
    if he_node_flag not in vattrs:
        return [], set()

    he_idxs = [i for i, flag in enumerate(igG.vs[he_node_flag]) if bool(flag)]
    if not he_idxs:
        return [], set()

    names = igG.vs["name"] if "name" in vattrs else list(range(igG.vcount()))
    membership_edge_idx = set()
    hyperdefs = []

    for hi in he_idxs:
        nd = {k: igG.vs[hi][k] for k in vattrs}  # HE node attrs
        eid = nd.get(he_id_attr, f"he::{names[hi]}")
        head_map, tail_map = {}, {}
        saw_head = saw_tail = saw_member = False

        for eidx in igG.incident(hi, mode="ALL"):
            membership_edge_idx.add(eidx)
            e = igG.es[eidx]
            u, v = e.tuple
            other_i = v if u == hi else u
            other = names[other_i]

            role = e[role_attr] if role_attr in igG.es.attributes() else None
            coeff = e[coeff_attr] if coeff_attr in igG.es.attributes() else (e.get("__value", 1.0))
            try:
                coeff = float(coeff)
                if math.isnan(coeff):
                    coeff = 1.0
            except Exception:
                coeff = 1.0

            if role == "head":
                head_map[other] = coeff
                saw_head = True
            elif role == "tail":
                tail_map[other] = coeff
                saw_tail = True
            else:
                head_map[other] = coeff
                tail_map[other] = coeff
                saw_member = True

        directed = bool(saw_head or saw_tail) and not (saw_member and not (saw_head or saw_tail))
        hyperdefs.append((eid, directed, head_map, tail_map, nd, hi))

    return hyperdefs, membership_edge_idx


def from_igraph(
    igG,
    manifest,
    *,
    hyperedge: str = "none",
    he_node_flag: str = "is_hyperedge",
    he_id_attr: str = "eid",
    reify_prefix: str = "he::",
) -> AnnNet:
    """Reconstruct a AnnNet from igraph.AnnNet + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in igG and rebuild true hyperedges
      that are NOT present in the manifest.
    """
    from ..core.graph import AnnNet

    H = AnnNet()

    # -------- helper: scan reified HE nodes in igG (used only if hyperedge == "reified") --------
    def _ig_collect_reified(ig):
        """Return list of tuples:
          (eid, directed, head_map, tail_map, he_attrs, he_index)
        where head_map/tail_map are {vertex_id: coeff}.
        """
        out = []
        # names for external vertex IDs
        names = ig.vs["name"] if "name" in ig.vs.attributes() else list(range(ig.vcount()))
        name_of = lambda idx: names[idx]

        # identify HE nodes
        he_indices = []
        for i in range(ig.vcount()):
            is_he = False
            try:
                is_he = bool(ig.vs[i][he_node_flag])
            except Exception:
                is_he = False
            if not is_he:
                nm = name_of(i)
                if isinstance(nm, str) and nm.startswith(reify_prefix):
                    is_he = True
            if is_he:
                he_indices.append(i)

        # membership edge attrs
        edge_attr_names = set(ig.es.attributes())
        role_attr = "role"
        coeff_attr = "coeff"
        membership_attr = "membership_of"

        for hi in he_indices:
            # hyperedge id from node attr or fallback from name sans prefix
            he_name = name_of(hi)
            eid = None
            try:
                eid = ig.vs[hi][he_id_attr]
            except Exception:
                eid = None
            if not eid and isinstance(he_name, str) and he_name.startswith(reify_prefix):
                eid = he_name[len(reify_prefix) :]
            if not eid:
                eid = f"he::{hi}"

            head_map, tail_map = {}, {}
            saw_head = saw_tail = False

            # all incident edges to this HE node
            try:
                inc = ig.incident(hi, mode="ALL")
            except Exception:
                inc = []
            for eidx in inc:
                e = ig.es[eidx]
                s, t = e.tuple
                other = t if s == hi else s
                v = name_of(other)

                # role / coeff
                role = None
                if "role" in edge_attr_names:
                    try:
                        role = e["role"]
                    except Exception:
                        role = None
                coeff = 1.0
                if "coeff" in edge_attr_names:
                    try:
                        coeff = float(e["coeff"])
                    except Exception:
                        coeff = 1.0

                if role == "head":
                    head_map[v] = coeff
                    saw_head = True
                elif role == "tail":
                    tail_map[v] = coeff
                    saw_tail = True
                else:
                    # undirected membership → symmetric
                    head_map[v] = head_map.get(v, coeff)
                    tail_map[v] = tail_map.get(v, coeff)

            directed = bool(saw_head or saw_tail)

            # copy HE-node attrs (minus markers)
            he_attrs = {}
            for k in ig.vs.attributes():
                if k in {he_node_flag, he_id_attr}:
                    continue
                try:
                    he_attrs[k] = ig.vs[hi][k]
                except Exception:
                    pass

            out.append((eid, directed, head_map, tail_map, he_attrs, hi))

        return out

    # -------- vertices (from manifest = SSOT) --------
    # Collect all vertex IDs referenced by manifest (attrs + edges)
    vertex_ids = set()

    for vid in (manifest.get("vertex_attrs", {}) or {}).keys():
        vertex_ids.add(vid)

    edges_def = manifest.get("edges", {}) or {}
    for eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == "regular":
            u, v = defn[0], defn[1]
            vertex_ids.add(u)
            vertex_ids.add(v)
        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict):
                for u in head_map.keys():
                    vertex_ids.add(u)
            if isinstance(tail_map, dict):
                for v in tail_map.keys():
                    vertex_ids.add(v)

    # Add vertices now (no he:: nodes will be included since they aren't in the manifest)
    if vertex_ids:
        H.add_vertices_bulk([{"vertex_id": v} for v in vertex_ids])

    # -------- edges/hyperedges (from manifest = SSOT) --------
    edge_directed_cache = manifest.get("edge_directed", {}) or {}
    regular_edges_bulk = []
    hyperedges_bulk = []

    for eid, defn in edges_def.items():
        kind = defn[-1]
        is_dir = bool(edge_directed_cache.get(eid, True))
        if kind == "regular":
            u, v = defn[0], defn[1]
            regular_edges_bulk.append(
                {"source": u, "target": v, "edge_id": eid, "edge_directed": is_dir}
            )
        elif kind == "hyper":
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head, tail = list(head_map), list(tail_map)
                attrs = {
                    "__source_attr": {u: {"__value": float(c)} for u, c in head_map.items()},
                    "__target_attr": {v: {"__value": float(c)} for v, c in tail_map.items()},
                }
                if is_dir:
                    hyperedges_bulk.append(
                        {
                            "head": head,
                            "tail": tail,
                            "edge_id": eid,
                            "edge_directed": True,
                            "attributes": attrs,
                        }
                    )
                else:
                    hyperedges_bulk.append(
                        {
                            "members": list(set(head) | set(tail)),
                            "edge_id": eid,
                            "edge_directed": False,
                            "attributes": attrs,
                        }
                    )

    if regular_edges_bulk:
        H.add_edges_bulk(regular_edges_bulk, default_edge_directed=True)
    if hyperedges_bulk:
        H.add_hyperedges_bulk(hyperedges_bulk)

    # -------- baseline weights --------
    for eid, w in (manifest.get("weights", {}) or {}).items():
        try:
            rec = H._edges.get(eid)
            if rec is not None:
                rec.weight = float(w)
        except Exception:
            pass

    # -------- slices + per-slice overrides --------
    existing_slices = set(H.list_slices(include_default=True))
    for lid, eids in (manifest.get("slices", {}) or {}).items():
        if lid not in existing_slices:
            try:
                H.add_slice(lid)
                existing_slices.add(lid)
            except Exception:
                pass
        if eids:
            H.add_edges_to_slice_bulk(lid, eids)

    for lid, per_edge in (manifest.get("slice_weights", {}) or {}).items():
        if lid not in existing_slices:
            try:
                H.add_slice(lid)
                existing_slices.add(lid)
            except Exception:
                pass
        for eid, w in (per_edge or {}).items():
            try:
                H.set_edge_slice_attrs(lid, eid, weight=float(w))
            except Exception:
                try:
                    H.set_edge_slice_attr(lid, eid, "weight", float(w))
                except Exception:
                    pass

    # ----- multilayer / Kivela -----
    mm = manifest.get("multilayer", {})
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

    # edge_kind / edge_layers
    ek = mm.get("edge_kind", {})
    el_ser = mm.get("edge_layers", {})
    if ek:
        for eid, kind in ek.items():
            rec = H._edges.get(eid)
            if rec is None:
                continue
            if kind == "hyper":
                rec.etype = "hyper"
            else:
                rec.ml_kind = kind
    if el_ser:
        H.edge_layers.update(_deserialize_edge_layers(el_ser))

    nl_attrs_ser = mm.get("node_layer_attrs", [])
    if nl_attrs_ser:
        H._state_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

    layer_tuple_attrs_ser = mm.get("layer_tuple_attrs", [])
    if layer_tuple_attrs_ser:
        H._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

    layer_attr_rows = mm.get("layer_attributes", [])
    if layer_attr_rows:
        H.layer_attributes = _rows_to_df(layer_attr_rows)

    # -------- restore vertex/edge attrs --------
    vertex_attrs_cache = manifest.get("vertex_attrs", {}) or {}
    if vertex_attrs_cache:
        for vid, attrs in vertex_attrs_cache.items():
            if attrs:
                try:
                    H.set_vertex_attrs(vid, **attrs)
                except Exception:
                    pass

    edge_attrs_cache = manifest.get("edge_attrs", {}) or {}
    if edge_attrs_cache:
        for eid, attrs in edge_attrs_cache.items():
            if attrs:
                try:
                    H.set_edge_attrs(eid, **attrs)
                except Exception:
                    pass

    # -------- OPTIONAL: pull in reified HEs from igG not present in manifest --------
    if hyperedge == "reified":
        try:
            hyperdefs = _ig_collect_reified(igG)
        except Exception:
            hyperdefs = []
        existing_eids = set(edges_def.keys())

        for eid, directed, head_map, tail_map, he_attrs, hi in hyperdefs:
            if eid in existing_eids:
                continue
            # ensure vertices
            for x in set(head_map) | set(tail_map):
                try:
                    H.add_vertex(x)
                except Exception:
                    pass

            if directed:
                try:
                    H.add_edge(src=list(head_map), tgt=list(tail_map), edge_id=eid, directed=True)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": c} for u, c in head_map.items()},
                        __target_attr={v: {"__value": c} for v, c in tail_map.items()},
                    )
                except Exception:
                    pass
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_edge(src=members, edge_id=eid, directed=False)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                        __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members},
                    )
                except Exception:
                    pass

            # copy HE-node attrs minus markers
            if he_attrs:
                try:
                    H.set_edge_attrs(eid, **he_attrs)
                except Exception:
                    pass

    return H


def _from_ig_without_manifest(
    igG,
    *,
    hyperedge="none",
    he_node_flag="is_hyperedge",
    he_id_attr="eid",
    role_attr="role",
    coeff_attr="coeff",
    membership_attr="membership_of",
):
    """Best-effort import from a *plain* igraph.AnnNet (no manifest).
    Preserves all vertex/edge attributes.
    hyperedge: "none" | "reified"
      When "reified", rebuild true hyperedges and skip membership edges from binary import.
    """
    from ..core.graph import AnnNet

    H = AnnNet()

    # vertices
    names = igG.vs["name"] if "name" in igG.vs.attributes() else list(range(igG.vcount()))
    for i, vid in enumerate(names):
        try:
            H.add_vertex(vid)
        except Exception:
            pass
        vattrs = {k: igG.vs[i][k] for k in igG.vs.attributes()}
        if vattrs:
            try:
                H.set_vertex_attrs(vid, **vattrs)
            except Exception:
                pass

    membership_idx = set()
    if hyperedge == "reified":
        hyperdefs, membership_idx = _ig_collect_reified(
            igG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, he_attrs, hi in hyperdefs:
            for x in set(head_map) | set(tail_map):
                try:
                    H.add_vertex(x)
                except Exception:
                    pass
            if directed:
                try:
                    H.add_edge(src=list(head_map), tgt=list(tail_map), edge_id=eid, directed=True)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": c} for u, c in head_map.items()},
                        __target_attr={v: {"__value": c} for v, c in tail_map.items()},
                    )
                except Exception:
                    pass
            else:
                members = list(set(head_map) | set(tail_map))
                try:
                    H.add_edge(src=members, edge_id=eid, directed=False)
                except Exception:
                    pass
                try:
                    H.set_edge_attrs(
                        eid,
                        __source_attr={u: {"__value": head_map.get(u, 1.0)} for u in members},
                        __target_attr={v: {"__value": tail_map.get(v, 1.0)} for v in members},
                    )
                except Exception:
                    pass
            # copy HE-node attrs (minus markers)
            he_node_attrs = {
                k: igG.vs[hi][k] for k in igG.vs.attributes() if k not in {he_node_flag, he_id_attr}
            }
            if he_node_attrs:
                try:
                    H.set_edge_attrs(eid, **he_node_attrs)
                except Exception:
                    pass

    # binary edges (skip membership edges if reified)
    is_dir = igG.is_directed()
    seen_auto = 0
    for e in igG.es:
        if e.index in membership_idx:
            continue
        src = names[e.source]
        dst = names[e.target]
        d = {k: e[k] for k in igG.es.attributes()}

        eid = d.get("eid")
        if eid is None:
            seen_auto += 1
            eid = f"ig::e#{seen_auto}"

        e_directed = bool(d.get("directed", is_dir))
        w = d.get("weight", d.get("__weight", 1.0))

        try:
            H.add_vertex(src)
            H.add_vertex(dst)
        except Exception:
            pass
        try:
            H.add_edge(src, dst, edge_id=eid, directed=e_directed)
        except Exception:
            H.add_edge(src, dst, edge_id=eid, directed=True)

        try:
            H.edge_weights[eid] = float(w)
        except Exception:
            pass

        if d:
            try:
                H.set_edge_attrs(eid, **d)
            except Exception:
                pass

    return H

