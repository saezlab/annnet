"""
AnnNet-igraph adapter for AnnNet.

Provides:
    to_igraph(anet)      -> igraph.Graph
    from_igraph(igG)  -> AnnNet

igraph natively represents:
    - nodes
    - binary edges
    - graph, node, and edge attributes

AnnNet-specific structures such as hyperedges, slices, multilayer metadata,
per-edge directedness, and richer attribute tables are preserved through
manifest-style graph attributes where possible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._common import (
    _rows_like,
    iter_edge_sides,
    _is_directed_eid,
    _serialize_value,
    requested_slices,
    dataframe_to_rows,
)
from ._manifest import build_manifest, graph_from_manifest

if TYPE_CHECKING:
    from ..core import AnnNet
from ..core import _structure
from .._support import projection as _projection


def _export_binary_graph(
    graph: AnnNet,
    *,
    directed: bool = True,
    skip_hyperedges: bool = True,
    public_only: bool = False,
    slice_filter: set[str] | None = None,
):
    """Export AnnNet to igraph.AnnNet without manifest.

    igraph requires integer node indices; external node IDs are preserved
    in node attribute 'name'. Edge IDs stored in edge attribute 'eid'.

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

    # When a slice filter is active we restrict to nodes that belong to
    # the slice AND to edges in the slice. Without this, callers that ask
    # for `slice='signaling'` get the full node set with only the slice's
    # edges, which yields a degenerate "signaling subgraph" full of
    # singleton nodes from every other compartment/layer.
    slice_node_set: set[str] | None = None
    slice_edge_set: set[str] | None = None
    if slice_filter:
        slice_node_set = set()
        slice_edge_set = set()
        for lid in slice_filter:
            try:
                slice_node_set.update(graph.slices.nodes(lid))
            except Exception:  # noqa: BLE001
                pass
            try:
                slice_edge_set.update(graph.slices.edges(lid))
            except Exception:  # noqa: BLE001
                pass

    # Build the node universe robustly
    # Start with declared nodes
    base_nodes = set(graph.N)
    if slice_node_set is not None:
        base_nodes &= slice_node_set

    # Ensure endpoints that appear in edges are also included
    endpoints = set()
    for eid, _ref, sides in iter_edge_sides(graph):
        if slice_edge_set is not None and eid not in slice_edge_set:
            continue
        S, T = set(sides.source), set(sides.target)
        endpoints.update(S)
        endpoints.update(T)
    if slice_node_set is not None:
        endpoints &= slice_node_set

    nodes = list(dict.fromkeys(list(base_nodes) + list(endpoints)))  # stable order
    vidx = {v: i for i, v in enumerate(nodes)}

    # Create igraph graph and set node 'name'
    G = ig.Graph(directed=bool(directed))
    G.add_vertices(len(nodes))
    G.vs['name'] = nodes

    # Attach node attributes (works for both nodes and edge-entities)
    vtab = getattr(graph, '_node_table', None)
    # Pre-scan to a dict for O(1) lookup
    vattr_map = {}
    if vtab is not None:
        for row in dataframe_to_rows(vtab):
            d = dict(row)
            vid = d.pop('node_id', None)
            if vid is not None:
                vattr_map[vid] = d

    processed_vattrs = {}
    for v in nodes:
        v_attr = dict(vattr_map.get(v, {}))
        if public_only:
            v_attr = {
                k: _serialize_value(val) for k, val in v_attr.items() if not str(k).startswith('__')
            }
        else:
            v_attr = {k: _serialize_value(val) for k, val in v_attr.items()}
        # igraph reserves the 'name' node attribute for node IDs;
        # rename any user attr called 'name' so it doesn't clobber them.
        if 'name' in v_attr:
            v_attr['__attr_name'] = v_attr.pop('name')
        processed_vattrs[v] = v_attr

    all_vattr_keys = set().union(*processed_vattrs.values()) if processed_vattrs else set()
    for k in all_vattr_keys:
        G.vs[k] = [processed_vattrs[v].get(k) for v in nodes]

    # Helper: directedness per edge-id (fallback if helper missing)
    def _is_dir_eid(g, eid):
        try:
            return _is_directed_eid(g, eid)  # use existing helper if present
        except NameError:
            rec = getattr(g, '_edges', {}).get(eid)
            if rec is not None and rec.directed is not None:
                return bool(rec.directed)
            return bool(getattr(g, 'directed', True))

    # Add edges (binary & node-edge). Hyperedges: skip or expand
    eattr_map = {}
    for row in _rows_like(getattr(graph, '_edge_table', None)):
        eid = row.get('edge_id')
        if eid is not None:
            d = dict(row)
            d.pop('edge_id', None)
            eattr_map[eid] = d
    # collect all edges as tuples first, write them in one bulk call
    edge_tuples = []
    edge_payloads = []  # list of dicts, parallel to edge_tuples

    for eid, ref, sides in iter_edge_sides(graph):
        if slice_edge_set is not None and eid not in slice_edge_set:
            continue
        S, T = set(sides.source), set(sides.target)

        e_attr = dict(eattr_map.get(eid, {}))
        if public_only:
            e_attr = {
                k: _serialize_value(val) for k, val in e_attr.items() if not str(k).startswith('__')
            }
        else:
            e_attr = {k: _serialize_value(val) for k, val in e_attr.items()}

        weight = ref.weight
        e_attr['weight' if public_only else '__weight'] = weight
        e_attr['eid'] = eid

        is_hyper = ref.kind == 'hyper'
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
    directed: bool = True,
    hyperedge_mode: str = 'skip',
    hyperedges: str | None = None,
    coefficients: str = 'error',
    slice: str | None = None,
    slices: list[str] | None = None,
    public_only: bool = False,
    reify_prefix: str = 'he::',
) -> tuple[Any, dict]:
    """Export AnnNet → ``(igraph graph, manifest)``.

    ``from_igraph`` takes the manifest as its second argument, so
    ``from_igraph(*to_igraph(anet))`` is the round trip.

    hyperedges : str, optional
        The same choice as ``hyperedge_mode``, under the name every exporter
        shares, and accepting ``"star"`` and ``"clique"`` as aliases for
        ``"reify"`` and ``"expand"``. Takes precedence when both are given.
    coefficients : {"error", "drop"}, default "error"
        What to do when the projection would drop per-member coefficients. Only
        ``"expand"`` loses them, and losing them silently is what this refuses.
    hyperedge_mode: {"skip","expand","reify"}
      - "skip": drop HE edges from igG (manifest keeps them)
      - "expand": cartesian product (directed) / clique (undirected)
      - "reify": add a node per HE and membership edges V↔HE carrying roles/coeffs
    """
    hyperedge_mode = _projection.normalise(hyperedges if hyperedges is not None else hyperedge_mode)
    _projection.check_coefficients(
        _structure.hyperedges_with_coefficients(graph), hyperedge_mode, coefficients
    )
    # -------------- base igraph build (binary edges only) --------------
    # For "reify" we start with hyperedges skipped, then add them as nodes+membership edges.
    _slice_filter_for_export: set[str] | None = None
    if slice is not None:
        _slice_filter_for_export = set([slice] if isinstance(slice, str) else list(slice))
    if slices is not None:
        if _slice_filter_for_export is None:
            _slice_filter_for_export = set()
        _slice_filter_for_export.update(slices)

    igG = _export_binary_graph(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ('skip', 'reify')),
        public_only=public_only,
        slice_filter=_slice_filter_for_export,
    )

    manifest = build_manifest(graph, public_only=public_only, slice=slice, slices=slices)
    manifest_edges = manifest['edges']
    edge_attrs = manifest['edge_attrs']
    base_weights = manifest['weights']
    edge_directed_dict = manifest['edge_directed']
    slices_section = manifest['slices']
    requested_lids = requested_slices(slice, slices)

    # -------------- REIFY: add HE nodes + membership edges to igG --------------
    if hyperedge_mode == 'reify':
        # allowed HE set under slice filter (None => all)
        allowed = None
        if requested_lids:
            allowed = set()
            for _lid, eids in slices_section.items():
                for eid in eids:
                    allowed.add(eid)

        # ensure we have a 'name' attribute for nodes
        if 'name' not in igG.vs.attributes():
            igG.vs['name'] = list(range(igG.vcount()))
        names = list(igG.vs['name'])
        name_to_idx = {str(n): i for i, n in enumerate(names)}

        def ensure_node(name: str) -> int:
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
            he_idx = ensure_node(he_name)

            # copy selected edge attrs to HE node (public only)
            he_attrs = edge_attrs.get(eid, {}) or {}
            if public_only:
                he_attrs = {k: v for k, v in he_attrs.items() if not str(k).startswith('__')}
            he_directed_flag = edge_directed_dict.get(eid, True)
            igG.vs[he_idx]['is_hyperedge'] = True
            igG.vs[he_idx]['eid'] = eid
            igG.vs[he_idx]['directed'] = bool(he_directed_flag)
            igG.vs[he_idx]['hyper_weight'] = float(base_weights.get(eid, 1.0))
            for k, v in he_attrs.items():
                igG.vs[he_idx][k] = v

            if he_directed_flag:
                # tail -> HE
                for u, coeff in (tail_map or {}).items():
                    ui = ensure_node(u)
                    new_edges.append((ui, he_idx))
                    payloads.append({'role': 'tail', 'coeff': float(coeff), 'membership_of': eid})
                # HE -> head
                for v, coeff in (head_map or {}).items():
                    vi = ensure_node(v)
                    new_edges.append((he_idx, vi))
                    payloads.append({'role': 'head', 'coeff': float(coeff), 'membership_of': eid})
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if directed:  # directed container: add both directions
                    for u, coeff in members.items():
                        ui = ensure_node(u)
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
                        ui = ensure_node(u)
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
      - hyperdefs: list of (eid, directed:bool, head_map:dict, tail_map:dict, he_node_attrs:dict, he_node_index)
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
            coeff = (
                e[coeff_attr]
                if coeff_attr in igG.es.attributes()
                else (e['__value'] if '__value' in igG.es.attributes() else 1.0)
            )
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
    igG: Any,
    manifest: dict[str, Any],
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
    H = graph_from_manifest(manifest)

    def ensure_node(node_id):
        if not H.has_node(node_id):
            H.add_nodes(node_id)

    # -------- helper: scan reified HE nodes in igG (used only if hyperedge == "reified") --------
    def _ig_collect_reified(ig):
        """Return list of tuples:

          (eid, directed, head_map, tail_map, he_attrs, he_index)
        where head_map/tail_map are {node_id: coeff}.
        """
        out = []
        # names for external node IDs
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

    edges_def = manifest.get('edges', {}) or {}

    # -------- OPTIONAL: pull in reified HEs from igG not present in manifest --------
    if hyperedge == 'reified':
        hyperdefs = _ig_collect_reified(igG)
        existing_eids = set(edges_def.keys())

        for eid, directed, head_map, tail_map, he_attrs, _hi in hyperdefs:
            if eid in existing_eids:
                continue
            # ensure nodes
            for x in set(head_map) | set(tail_map):
                ensure_node(x)

            if directed:
                H.add_edges(src=list(head_map), tgt=list(tail_map), edge_id=eid, directed=True)
                H.attrs.update(
                    'edges',
                    {
                        eid: {
                            '__source_attr': {u: {'__value': c} for u, c in head_map.items()},
                            '__target_attr': {v: {'__value': c} for v, c in tail_map.items()},
                        }
                    },
                )
            else:
                members = list(set(head_map) | set(tail_map))
                H.add_edges(src=members, edge_id=eid, directed=False)
                H.attrs.update(
                    'edges',
                    {
                        eid: {
                            '__source_attr': {
                                u: {'__value': head_map.get(u, 1.0)} for u in members
                            },
                            '__target_attr': {
                                v: {'__value': tail_map.get(v, 1.0)} for v in members
                            },
                        }
                    },
                )

            # copy HE-node attrs minus markers
            if he_attrs:
                H.attrs.update('edges', {eid: dict(he_attrs)})

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

    Bulk-batched: collect nodes, edges, hyperedges, and attribute maps into
    Python lists/dicts, then issue exactly one bulk call per kind. Previous
    per-element inserts were O(N²) in dataframe-concat work.

    Preserves all node/edge attributes.
    hyperedge: "none" | "reified"
      When "reified", rebuild true hyperedges and skip membership edges.
    """
    from ..core import AnnNet

    H = AnnNet()

    # 1) Nodes + their attrs (column-wise read from igraph)
    igvs_attrs = igG.vs.attributes()
    names = igG.vs['name'] if 'name' in igvs_attrs else list(range(igG.vcount()))
    user_vattr_keys = [k for k in igvs_attrs if k != 'name']
    user_vattr_cols = {k: igG.vs[k] for k in user_vattr_keys}

    node_buf: list = []
    node_seen: set = set()
    node_attrs_buf: dict = {}

    def _add_node(v):
        if v not in node_seen:
            node_seen.add(v)
            node_buf.append(v)

    for i, vid in enumerate(names):
        _add_node(vid)
        vattrs = {
            k: user_vattr_cols[k][i] for k in user_vattr_keys if user_vattr_cols[k][i] is not None
        }
        if '__attr_name' in vattrs:
            vattrs['name'] = vattrs.pop('__attr_name')
        if vattrs:
            node_attrs_buf[vid] = vattrs

    # 2) Optionally collect reified hyperedges
    membership_idx: set = set()
    hyperedges_bulk: list = []
    hyperedge_attrs_buf: dict = {}

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
                _add_node(x)

            attrs = {}
            if directed:
                attrs['__source_attr'] = {u: {'__value': c} for u, c in head_map.items()}
                attrs['__target_attr'] = {v: {'__value': c} for v, c in tail_map.items()}
                hyperedges_bulk.append(
                    {
                        'head': list(head_map),
                        'tail': list(tail_map),
                        'edge_id': eid,
                        'edge_directed': True,
                        'attributes': attrs,
                    }
                )
            else:
                members = list(set(head_map) | set(tail_map))
                attrs['__source_attr'] = {u: {'__value': head_map.get(u, 1.0)} for u in members}
                attrs['__target_attr'] = {v: {'__value': tail_map.get(v, 1.0)} for v in members}
                hyperedges_bulk.append(
                    {
                        'members': members,
                        'edge_id': eid,
                        'edge_directed': False,
                        'attributes': attrs,
                    }
                )
            he_node_attrs = {
                k: user_vattr_cols.get(k, [None] * igG.vcount())[hi]
                for k in user_vattr_keys
                if k not in {he_node_flag, he_id_attr}
            }
            he_node_attrs = {k: v for k, v in he_node_attrs.items() if v is not None}
            if he_node_attrs:
                hyperedge_attrs_buf[eid] = he_node_attrs

    # 3) Binary edges — bulk collect
    is_dir = igG.is_directed()
    iges_attrs = igG.es.attributes()
    user_eattr_cols = {k: igG.es[k] for k in iges_attrs}

    binary_edges_bulk: list = []
    edge_attrs_buf: dict = {}
    seen_auto = 0

    for e in igG.es:
        if e.index in membership_idx:
            continue
        idx = e.index
        src = names[e.source]
        dst = names[e.target]
        d = {k: user_eattr_cols[k][idx] for k in iges_attrs}

        eid = d.get('eid')
        if eid is None:
            seen_auto += 1
            eid = f'ig::e#{seen_auto}'

        e_directed = bool(d.get('directed', is_dir))
        w = d.get('weight', d.get('__weight', 1.0))

        _add_node(src)
        _add_node(dst)
        binary_edges_bulk.append(
            {
                'source': src,
                'target': dst,
                'edge_id': eid,
                'edge_directed': e_directed,
                'weight': float(w if w is not None else 1.0),
            }
        )

        if d:
            clean = {k: v for k, v in d.items() if v is not None}
            for k in ('eid', 'weight', '__weight', 'directed'):
                clean.pop(k, None)
            if clean:
                edge_attrs_buf[eid] = clean

    # 4) BULK INSERTS
    if node_buf:
        H._add_nodes_bulk([{'node_id': v} for v in node_buf])
    if binary_edges_bulk:
        H._add_edges_bulk(binary_edges_bulk, default_edge_directed=is_dir)
    if hyperedges_bulk:
        H.add_hyperedges_bulk(hyperedges_bulk)
    if node_attrs_buf:
        H.attrs.update('nodes', node_attrs_buf)
    if edge_attrs_buf or hyperedge_attrs_buf:
        H.attrs.update('edges', {**edge_attrs_buf, **hyperedge_attrs_buf})

    return H
