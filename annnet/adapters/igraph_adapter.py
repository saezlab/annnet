"""
AnnNet-igraph adapter for AnnNet.

Provides:
    to_igraph(G)      -> igraph.Graph
    from_igraph(igG)  -> AnnNet

igraph natively represents:
    - vertices
    - binary edges
    - graph, vertex, and edge attributes

AnnNet-specific structures such as hyperedges, slices, multilayer metadata,
per-edge directedness, and richer attribute tables are preserved through
manifest-style graph attributes where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._common import (
    _attrs_to_dict,
    _is_directed_eid,
    _iter_edge_records,
    _rows_like,
    _rows_to_df,
    _serialize_value,
    collect_slice_manifest,
    dataframe_to_rows,
    deserialize_edge_layers,
    endpoint_coeff_map,
    restore_multilayer_manifest,
    restore_slice_manifest,
    serialize_edge_layers,
    serialize_multilayer_manifest,
)

if TYPE_CHECKING:
    from ..core import AnnNet


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
    for _eid, rec in _iter_edge_records(graph):
        if rec.etype == 'hyper':
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}
        endpoints.update(S)
        endpoints.update(T)

    vertices = list(dict.fromkeys(list(base_vertices) + list(endpoints)))  # stable order
    vidx = {v: i for i, v in enumerate(vertices)}

    # Create igraph graph and set vertex 'name'
    G = ig.Graph(directed=bool(directed))
    G.add_vertices(len(vertices))
    G.vs['name'] = vertices

    # Attach vertex attributes (works for both vertices and edge-entities)
    vtab = getattr(graph, 'vertex_attributes', None)
    # Pre-scan to a dict for O(1) lookup
    vattr_map = {}
    if vtab is not None:
        for row in dataframe_to_rows(vtab):
            d = dict(row)
            vid = d.pop('vertex_id', None)
            if vid is not None:
                vattr_map[vid] = d

    processed_vattrs = {}
    for v in vertices:
        v_attr = dict(vattr_map.get(v, {}))
        if public_only:
            v_attr = {
                k: _serialize_value(val) for k, val in v_attr.items() if not str(k).startswith('__')
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
            rec = getattr(g, '_edges', {}).get(eid)
            if rec is not None and rec.directed is not None:
                return bool(rec.directed)
            return bool(getattr(g, 'directed', True))

    # Add edges (binary & vertex-edge). Hyperedges: skip or expand
    eattr_map = {}
    for row in _rows_like(getattr(graph, 'edge_attributes', None)):
        eid = row.get('edge_id')
        if eid is not None:
            d = dict(row)
            d.pop('edge_id', None)
            eattr_map[eid] = d
    # collect all edges as tuples first, write them in one bulk call
    edge_tuples = []
    edge_payloads = []  # list of dicts, parallel to edge_tuples

    for eid, rec in _iter_edge_records(graph):
        if rec.etype == 'hyper':
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}

        e_attr = dict(eattr_map.get(eid, {}))
        if public_only:
            e_attr = {
                k: _serialize_value(val) for k, val in e_attr.items() if not str(k).startswith('__')
            }
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = 1.0 if rec.weight is None else rec.weight
        e_attr['weight' if public_only else '__weight'] = weight
        e_attr['eid'] = eid

        is_hyper = rec.etype == 'hyper'
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
                        p['directed'] = True
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
    if hasattr(obj, 'items'):
        v = obj.get('__value', 1)
        if hasattr(v, 'items'):
            v = v.get('__value', 1)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def to_igraph(
    graph: AnnNet,
    directed=True,
    hyperedge_mode='skip',
    slice=None,
    slices=None,
    public_only=False,
    reify_prefix='he::',
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
        skip_hyperedges=(hyperedge_mode in ('skip', 'reify')),
        public_only=public_only,
    )

    # -------------- collect vertex/edge attrs for manifest --------------
    _raw_vertex_attrs = {
        row['vertex_id']: {k: v for k, v in row.items() if k != 'vertex_id'}
        for row in _rows_like(getattr(graph, 'vertex_attributes', None))
        if row.get('vertex_id') is not None
    }
    vertex_attrs = {
        v: _attrs_to_dict(
            {
                k: val
                for k, val in _raw_vertex_attrs.get(v, {}).items()
                if not public_only or not str(k).startswith('__')
            }
        )
        for v in graph.vertices()
    }

    _raw_edge_attrs = {
        row['edge_id']: {k: v for k, v in row.items() if k != 'edge_id'}
        for row in _rows_like(getattr(graph, 'edge_attributes', None))
        if row.get('edge_id') is not None
    }
    edge_attrs = {
        eid: _attrs_to_dict(
            {
                k: val
                for k, val in _raw_edge_attrs.get(eid, {}).items()
                if not public_only or not str(k).startswith('__')
            }
        )
        for eid, _rec in _iter_edge_records(graph)
    }

    # -------------- topology snapshot (regular vs hyper) --------------
    manifest_edges = {}
    for eid, rec in _iter_edge_records(graph):
        is_hyper = rec.etype == 'hyper'
        if is_hyper:
            S, T = set(rec.src or []), set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}
        if not is_hyper:
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                manifest_edges[eid] = (u, u, 'regular')
            elif len(members) == 2:
                u, v = sorted(members)
                manifest_edges[eid] = (u, v, 'regular')
            else:
                eattr = edge_attrs.get(eid, {})
                head_map = endpoint_coeff_map(eattr, '__source_attr', S)
                tail_map = endpoint_coeff_map(eattr, '__target_attr', T)
                manifest_edges[eid] = (head_map, tail_map, 'hyper')
        else:
            eattr = edge_attrs.get(eid, {})
            head_map = endpoint_coeff_map(eattr, '__source_attr', S)
            tail_map = endpoint_coeff_map(eattr, '__target_attr', T)
            manifest_edges[eid] = (head_map, tail_map, 'hyper')

    # ---------- slices + per-slice weights for manifest ----------
    all_eids = list(manifest_edges.keys())

    requested_lids = set()
    if slice is not None:
        requested_lids.update([slice] if isinstance(slice, str) else list(slice))
    if slices is not None:
        requested_lids.update(list(slices))
    slices_section, slice_weights = collect_slice_manifest(
        graph,
        requested_lids=(list(requested_lids) if requested_lids else None),
    )

    base_weights = {
        eid: float(rec.weight) for eid, rec in _iter_edge_records(graph) if rec.weight is not None
    }

    # -------------- REIFY: add HE nodes + membership edges to igG --------------
    if hyperedge_mode == 'reify':
        # allowed HE set under slice filter (None => all)
        allowed = None
        if requested_lids:
            allowed = set()
            for _lid, eids in slices_section.items():
                for eid in eids:
                    allowed.add(eid)

        # ensure we have a 'name' attribute for vertices
        if 'name' not in igG.vs.attributes():
            igG.vs['name'] = list(range(igG.vcount()))
        names = list(igG.vs['name'])
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
            if spec[-1] != 'hyper':
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_name = f'{reify_prefix}{eid}'
            he_idx = ensure_vertex(he_name)

            # copy selected edge attrs to HE node (public only)
            he_attrs = edge_attrs.get(eid, {}) or {}
            if public_only:
                he_attrs = {k: v for k, v in he_attrs.items() if not str(k).startswith('__')}
            igG.vs[he_idx]['is_hyperedge'] = True
            igG.vs[he_idx]['eid'] = eid
            igG.vs[he_idx]['directed'] = bool(_is_directed_eid(graph, eid))
            igG.vs[he_idx]['hyper_weight'] = float(base_weights.get(eid, 1.0))
            # also copy public user attrs
            for k, v in he_attrs.items():
                igG.vs[he_idx][k] = v

            if _is_directed_eid(graph, eid):
                # tail -> HE
                for u, coeff in (tail_map or {}).items():
                    ui = ensure_vertex(u)
                    new_edges.append((ui, he_idx))
                    payloads.append({'role': 'tail', 'coeff': float(coeff), 'membership_of': eid})
                # HE -> head
                for v, coeff in (head_map or {}).items():
                    vi = ensure_vertex(v)
                    new_edges.append((he_idx, vi))
                    payloads.append({'role': 'head', 'coeff': float(coeff), 'membership_of': eid})
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if directed:  # directed container: add both directions
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {'role': 'member', 'coeff': float(coeff), 'membership_of': eid}
                        )
                        new_edges.append((he_idx, ui))
                        payloads.append(
                            {'role': 'member', 'coeff': float(coeff), 'membership_of': eid}
                        )
                else:  # undirected container: one edge is enough
                    for u, coeff in members.items():
                        ui = ensure_vertex(u)
                        new_edges.append((ui, he_idx))
                        payloads.append(
                            {'role': 'member', 'coeff': float(coeff), 'membership_of': eid}
                        )

        if new_edges:
            start = igG.ecount()
            igG.add_edges(new_edges)
            # set attributes for the newly added membership edges
            keys = set().union(*(d.keys() for d in payloads))
            for k in keys:
                igG.es[start:][k] = [d.get(k) for d in payloads]

    # -------------- manifest (unchanged semantics) --------------
    manifest = {
        'edges': manifest_edges,
        'weights': base_weights,
        'slices': slices_section,
        'vertex_attrs': vertex_attrs,
        'edge_attrs': edge_attrs,
        'slice_weights': slice_weights,
        'edge_directed': {eid: bool(_is_directed_eid(graph, eid)) for eid in all_eids},
        'manifest_version': 1,
        'multilayer': serialize_multilayer_manifest(
            graph,
            table_to_rows=dataframe_to_rows,
            serialize_edge_layers=serialize_edge_layers,
        ),
    }

    return igG, manifest


def _ig_collect_reified(
    igG,
    he_node_flag='is_hyperedge',
    he_id_attr='eid',
    role_attr='role',
    coeff_attr='coeff',
    membership_attr='membership_of',
):
    """Scan igG for reified hyperedges.

    Returns
    -------
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

    names = igG.vs['name'] if 'name' in vattrs else list(range(igG.vcount()))
    membership_edge_idx = set()
    hyperdefs = []

    for hi in he_idxs:
        nd = {k: igG.vs[hi][k] for k in vattrs}  # HE node attrs
        eid = nd.get(he_id_attr, f'he::{names[hi]}')
        head_map, tail_map = {}, {}
        saw_head = saw_tail = saw_member = False

        for eidx in igG.incident(hi, mode='ALL'):
            membership_edge_idx.add(eidx)
            e = igG.es[eidx]
            u, v = e.tuple
            other_i = v if u == hi else u
            other = names[other_i]

            role = e[role_attr] if role_attr in igG.es.attributes() else None
            coeff = e[coeff_attr] if coeff_attr in igG.es.attributes() else (e.get('__value', 1.0))
            try:
                coeff = float(coeff)
                if math.isnan(coeff):
                    coeff = 1.0
            except (TypeError, ValueError):
                coeff = 1.0

            if role == 'head':
                head_map[other] = coeff
                saw_head = True
            elif role == 'tail':
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
    hyperedge: str = 'none',
    he_node_flag: str = 'is_hyperedge',
    he_id_attr: str = 'eid',
    reify_prefix: str = 'he::',
) -> AnnNet:
    """Reconstruct a AnnNet from igraph.AnnNet + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in igG and rebuild true hyperedges
      that are NOT present in the manifest.
    """
    from ..core import AnnNet

    H = AnnNet()
    known_vertices = set()

    def ensure_vertex(vertex_id):
        if vertex_id in known_vertices:
            return
        H.add_vertices(vertex_id)
        known_vertices.add(vertex_id)

    # -------- helper: scan reified HE nodes in igG (used only if hyperedge == "reified") --------
    def _ig_collect_reified(ig):
        """Return list of tuples:

          (eid, directed, head_map, tail_map, he_attrs, he_index)
        where head_map/tail_map are {vertex_id: coeff}.
        """
        out = []
        # names for external vertex IDs
        names = ig.vs['name'] if 'name' in ig.vs.attributes() else list(range(ig.vcount()))
        name_of = lambda idx: names[idx]

        # identify HE nodes
        he_indices = []
        for i in range(ig.vcount()):
            is_he = False
            try:
                is_he = bool(ig.vs[i][he_node_flag])
            except (KeyError, IndexError, TypeError, ValueError):
                is_he = False
            if not is_he:
                nm = name_of(i)
                if isinstance(nm, str) and nm.startswith(reify_prefix):
                    is_he = True
            if is_he:
                he_indices.append(i)

        # membership edge attrs
        edge_attr_names = set(ig.es.attributes())

        for hi in he_indices:
            # hyperedge id from node attr or fallback from name sans prefix
            he_name = name_of(hi)
            eid = None
            try:
                eid = ig.vs[hi][he_id_attr]
            except (KeyError, IndexError, TypeError, ValueError):
                eid = None
            if not eid and isinstance(he_name, str) and he_name.startswith(reify_prefix):
                eid = he_name[len(reify_prefix) :]
            if not eid:
                eid = f'he::{hi}'

            head_map, tail_map = {}, {}
            saw_head = saw_tail = False

            # all incident edges to this HE node
            try:
                inc = ig.incident(hi, mode='ALL')
            except (ValueError, TypeError):
                inc = []
            for eidx in inc:
                e = ig.es[eidx]
                s, t = e.tuple
                other = t if s == hi else s
                v = name_of(other)

                # role / coeff
                role = None
                if 'role' in edge_attr_names:
                    try:
                        role = e['role']
                    except (KeyError, IndexError, TypeError, ValueError):
                        role = None
                coeff = 1.0
                if 'coeff' in edge_attr_names:
                    try:
                        coeff = float(e['coeff'])
                    except (KeyError, IndexError, TypeError, ValueError):
                        coeff = 1.0

                if role == 'head':
                    head_map[v] = coeff
                    saw_head = True
                elif role == 'tail':
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
                except (KeyError, IndexError, TypeError, ValueError):
                    pass

            out.append((eid, directed, head_map, tail_map, he_attrs, hi))

        return out

    # -------- vertices (from manifest = SSOT) --------
    # Collect all vertex IDs referenced by manifest (attrs + edges)
    vertex_ids = set()

    for vid in (manifest.get('vertex_attrs', {}) or {}).keys():
        vertex_ids.add(vid)

    edges_def = manifest.get('edges', {}) or {}
    for _eid, defn in edges_def.items():
        kind = defn[-1]
        if kind == 'regular':
            u, v = defn[0], defn[1]
            vertex_ids.add(u)
            vertex_ids.add(v)
        elif kind == 'hyper':
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict):
                for u in head_map.keys():
                    vertex_ids.add(u)
            if isinstance(tail_map, dict):
                for v in tail_map.keys():
                    vertex_ids.add(v)

    # Add vertices now (no he:: nodes will be included since they aren't in the manifest)
    if vertex_ids:
        H.add_vertices_bulk([{'vertex_id': v} for v in vertex_ids])

    # -------- edges/hyperedges (from manifest = SSOT) --------
    edge_directed_cache = manifest.get('edge_directed', {}) or {}
    regular_edges_bulk = []
    hyperedges_bulk = []

    for eid, defn in edges_def.items():
        kind = defn[-1]
        is_dir = bool(edge_directed_cache.get(eid, True))
        if kind == 'regular':
            u, v = defn[0], defn[1]
            regular_edges_bulk.append(
                {
                    'source': u,
                    'target': v,
                    'edge_id': eid,
                    'edge_directed': is_dir,
                    'weight': (manifest.get('weights', {}) or {}).get(eid, 1.0),
                }
            )
        elif kind == 'hyper':
            head_map, tail_map = defn[0], defn[1]
            if isinstance(head_map, dict) and isinstance(tail_map, dict):
                head, tail = list(head_map), list(tail_map)
                attrs = {
                    '__source_attr': {u: {'__value': float(c)} for u, c in head_map.items()},
                    '__target_attr': {v: {'__value': float(c)} for v, c in tail_map.items()},
                }
                if is_dir:
                    hyperedges_bulk.append(
                        {
                            'head': head,
                            'tail': tail,
                            'edge_id': eid,
                            'edge_directed': True,
                            'weight': (manifest.get('weights', {}) or {}).get(eid, 1.0),
                            'attributes': attrs,
                        }
                    )
                else:
                    hyperedges_bulk.append(
                        {
                            'members': list(set(head) | set(tail)),
                            'edge_id': eid,
                            'edge_directed': False,
                            'weight': (manifest.get('weights', {}) or {}).get(eid, 1.0),
                            'attributes': attrs,
                        }
                    )

    if regular_edges_bulk:
        H.add_edges_bulk(regular_edges_bulk, default_edge_directed=True)
    if hyperedges_bulk:
        H.add_hyperedges_bulk(hyperedges_bulk)

    # -------- slices + per-slice overrides --------
    restore_slice_manifest(
        H,
        manifest.get('slices', {}) or {},
        manifest.get('slice_weights', {}) or {},
    )

    # ----- multilayer / Kivela -----
    restore_multilayer_manifest(
        H,
        manifest.get('multilayer', {}),
        rows_to_table=_rows_to_df,
        deserialize_edge_layers=deserialize_edge_layers,
    )

    # -------- restore vertex/edge attrs --------
    vertex_attrs_cache = manifest.get('vertex_attrs', {}) or {}
    if vertex_attrs_cache:
        for vid, attrs in vertex_attrs_cache.items():
            if attrs:
                H.attrs.set_vertex_attrs(vid, **attrs)

    edge_attrs_cache = manifest.get('edge_attrs', {}) or {}
    if edge_attrs_cache:
        for eid, attrs in edge_attrs_cache.items():
            if attrs:
                H.attrs.set_edge_attrs(eid, **attrs)

    # -------- OPTIONAL: pull in reified HEs from igG not present in manifest --------
    if hyperedge == 'reified':
        hyperdefs = _ig_collect_reified(igG)
        existing_eids = set(edges_def.keys())

        for eid, directed, head_map, tail_map, he_attrs, _hi in hyperdefs:
            if eid in existing_eids:
                continue
            # ensure vertices
            for x in set(head_map) | set(tail_map):
                ensure_vertex(x)

            if directed:
                H.add_edges(src=list(head_map), tgt=list(tail_map), edge_id=eid, directed=True)
                H.attrs.set_edge_attrs(
                    eid,
                    __source_attr={u: {'__value': c} for u, c in head_map.items()},
                    __target_attr={v: {'__value': c} for v, c in tail_map.items()},
                )
            else:
                members = list(set(head_map) | set(tail_map))
                H.add_edges(src=members, edge_id=eid, directed=False)
                H.attrs.set_edge_attrs(
                    eid,
                    __source_attr={u: {'__value': head_map.get(u, 1.0)} for u in members},
                    __target_attr={v: {'__value': tail_map.get(v, 1.0)} for v in members},
                )

            # copy HE-node attrs minus markers
            if he_attrs:
                H.attrs.set_edge_attrs(eid, **he_attrs)

    return H


def _from_ig_without_manifest(
    igG,
    *,
    hyperedge='none',
    he_node_flag='is_hyperedge',
    he_id_attr='eid',
    role_attr='role',
    coeff_attr='coeff',
    membership_attr='membership_of',
):
    """Best-effort import from a *plain* igraph.AnnNet (no manifest).

    Preserves all vertex/edge attributes.
    hyperedge: "none" | "reified"
      When "reified", rebuild true hyperedges and skip membership edges from binary import.
    """
    from ..core import AnnNet

    H = AnnNet()
    known_vertices = set()

    def ensure_vertex(vertex_id):
        if vertex_id in known_vertices:
            return
        H.add_vertices(vertex_id)
        known_vertices.add(vertex_id)

    # vertices
    names = igG.vs['name'] if 'name' in igG.vs.attributes() else list(range(igG.vcount()))
    for i, vid in enumerate(names):
        ensure_vertex(vid)
        vattrs = {k: igG.vs[i][k] for k in igG.vs.attributes()}
        if vattrs:
            H.attrs.set_vertex_attrs(vid, **vattrs)

    membership_idx = set()
    if hyperedge == 'reified':
        hyperdefs, membership_idx = _ig_collect_reified(
            igG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, _he_attrs, hi in hyperdefs:
            for x in set(head_map) | set(tail_map):
                ensure_vertex(x)
            if directed:
                H.add_edges(src=list(head_map), tgt=list(tail_map), edge_id=eid, directed=True)
                H.attrs.set_edge_attrs(
                    eid,
                    __source_attr={u: {'__value': c} for u, c in head_map.items()},
                    __target_attr={v: {'__value': c} for v, c in tail_map.items()},
                )
            else:
                members = list(set(head_map) | set(tail_map))
                H.add_edges(src=members, edge_id=eid, directed=False)
                H.attrs.set_edge_attrs(
                    eid,
                    __source_attr={u: {'__value': head_map.get(u, 1.0)} for u in members},
                    __target_attr={v: {'__value': tail_map.get(v, 1.0)} for v in members},
                )
            # copy HE-node attrs (minus markers)
            he_node_attrs = {
                k: igG.vs[hi][k] for k in igG.vs.attributes() if k not in {he_node_flag, he_id_attr}
            }
            if he_node_attrs:
                H.attrs.set_edge_attrs(eid, **he_node_attrs)

    # binary edges (skip membership edges if reified)
    is_dir = igG.is_directed()
    seen_auto = 0
    for e in igG.es:
        if e.index in membership_idx:
            continue
        src = names[e.source]
        dst = names[e.target]
        d = {k: e[k] for k in igG.es.attributes()}

        eid = d.get('eid')
        if eid is None:
            seen_auto += 1
            eid = f'ig::e#{seen_auto}'

        e_directed = bool(d.get('directed', is_dir))
        w = d.get('weight', d.get('__weight', 1.0))

        ensure_vertex(src)
        ensure_vertex(dst)
        H.add_edges(src, dst, edge_id=eid, directed=e_directed, weight=float(w))

        if d:
            clean = dict(d)
            for k in ('eid', 'weight', '__weight', 'directed'):
                clean.pop(k, None)
            if clean:
                H.attrs.set_edge_attrs(eid, **clean)

    return H
