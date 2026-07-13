"""
AnnNet-NetworkX adapter for AnnNet.

Provides:
    to_nx(G)      -> networkx.Graph, networkx.DiGraph, networkx.MultiGraph, or networkx.MultiDiGraph
    from_nx(nxG)  -> AnnNet

NetworkX natively represents:
    - vertices/nodes
    - binary edges
    - graph, node, and edge attributes
    - directed, undirected, simple, and multigraph structures

AnnNet-specific structures such as hyperedges, slices, multilayer metadata,
per-edge directedness, and richer attribute tables are preserved through
manifest-style graph attributes where possible.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from contextlib import contextmanager

try:
    import networkx as nx
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Optional dependency 'networkx' is not installed. "
        'Install with: pip install annnet[networkx]'
    ) from e

from ._common import (
    _rows_like,
    _rows_to_df,
    _attrs_to_dict,
    _is_directed_eid,
    _serialize_value,
    dataframe_to_rows,
    _iter_edge_records,
    endpoint_coeff_map,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)

if TYPE_CHECKING:
    from ..core import AnnNet


# Structural keys that must not be re-applied as hyperedge attributes on import
# (they are part of the edge dispatch contract).
_HE_RESERVED_ATTRS = frozenset(
    {
        'directed',
        'edge_directed',
        'weight',
        'edge_id',
        'id',
        'source',
        'target',
        'head',
        'tail',
        'members',
        'kind',
    }
)


@contextmanager
def _time(label, timings):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    timings[label] = timings.get(label, 0.0) + (t1 - t0)


# ---------------


def _export_binary_graph(
    graph: AnnNet,
    *,
    directed: bool = True,
    skip_hyperedges: bool = True,
    public_only: bool = False,
    edge_directed_map: dict | None = None,
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

    # BATCH READ VERTEX ATTRIBUTES
    v_rows = _rows_like(graph.vertex_attributes)
    v_attrs_map = {}
    for row in v_rows:
        vid = row.get('vertex_id')
        if vid:
            attrs = dict(row)
            attrs.pop('vertex_id', None)
            if public_only:
                attrs = {
                    k: _serialize_value(v) for k, v in attrs.items() if not str(k).startswith('__')
                }
            else:
                attrs = {k: _serialize_value(v) for k, v in attrs.items()}
            v_attrs_map[vid] = attrs

    # BATCH READ EDGE ATTRIBUTES
    e_rows = _rows_like(graph.edge_attributes)
    e_attrs_map = {}
    for row in e_rows:
        eid = row.get('edge_id')
        if eid:
            attrs = dict(row)
            attrs.pop('edge_id', None)
            if public_only:
                attrs = {
                    k: _serialize_value(v) for k, v in attrs.items() if not str(k).startswith('__')
                }
            else:
                attrs = {k: _serialize_value(v) for k, v in attrs.items()}
            e_attrs_map[eid] = attrs

    # ADD VERTICES WITH CACHED ATTRIBUTES
    for v in graph.vertices():
        v_attr = v_attrs_map.get(v, {})
        G.add_node(v, **v_attr)

    # ADD EDGES WITH CACHED ATTRIBUTES
    for eid, rec in _iter_edge_records(graph):
        is_hyper = rec.etype == 'hyper'
        if is_hyper:
            S = set(rec.src or [])
            T = set(rec.tgt or [])
        else:
            S = set() if rec.src is None else {rec.src}
            T = set() if rec.tgt is None else {rec.tgt}

        e_attr = dict(e_attrs_map.get(eid, {}))

        weight = 1.0 if rec.weight is None else rec.weight
        if public_only:
            e_attr['weight'] = weight
        else:
            e_attr['__weight'] = weight

        is_dir = (
            edge_directed_map[eid]
            if edge_directed_map is not None and eid in edge_directed_map
            else _is_directed_eid(graph, eid)
        )
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
    if hasattr(obj, 'items'):
        v = obj.get('__value', 1)
        if hasattr(v, 'items'):
            v = v.get('__value', 1)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 1.0
    return 1.0


def to_nx(
    graph: AnnNet,
    directed: bool = True,
    hyperedge_mode: str = 'skip',
    slice: str | None = None,
    slices: list[str] | None = None,
    public_only: bool = False,
    reify_prefix: str = 'he::',
) -> nx.Graph:
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
        return {k: v for k, v in d.items() if not str(k).startswith('__')}

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
                for eid in graph.slices.edges(lid):
                    selected_eids.add(eid)
            except Exception:  # noqa: BLE001
                pass

    # HOIST LOOKUPS
    vertex_attributes_df = graph.vertex_attributes
    edge_attributes_df = graph.edge_attributes

    # Vertex attributes - BATCH READ
    vertex_attrs = {}
    v_rows = _rows_like(vertex_attributes_df)
    for row in v_rows:
        v = row.get('vertex_id')
        if v is None:
            continue
        attrs = dict(row)
        attrs.pop('vertex_id', None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith('__')}
        vertex_attrs[v] = _attrs_to_dict(attrs)

    # Edge attributes - BATCH READ
    edge_attrs = {}
    e_rows = _rows_like(edge_attributes_df)
    for row in e_rows:
        eid = row.get('edge_id')
        if eid is None:
            continue
        attrs = dict(row)
        attrs.pop('edge_id', None)
        if public_only:
            attrs = {k: v for k, v in attrs.items() if not str(k).startswith('__')}
        edge_attrs[eid] = _attrs_to_dict(attrs)

    # Single pass over edge records: build manifest_edges, weights_map, and
    # edge_directed_dict together. Previously this was 3 separate iterations.
    manifest_edges: dict = {}
    weights_map: dict = {}
    edge_directed_dict: dict = {}
    default_dir = True if graph.directed is None else bool(graph.directed)

    for eid, rec in _iter_edge_records(graph):
        is_hyper = rec.etype == 'hyper'
        if is_hyper:
            S = set(rec.src or [])
            T = set(rec.tgt or [])
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

        if rec.weight is not None:
            weights_map[eid] = float(rec.weight)
        edge_directed_dict[eid] = bool(rec.directed) if rec.directed is not None else default_dir

    # Base NX graph (binary edges only) — reuse the precomputed directedness map
    # so `_export_binary_graph` skips the polars-backed `_is_directed_eid` probe.
    nxG = _export_binary_graph(
        graph,
        directed=directed,
        skip_hyperedges=(hyperedge_mode in ('skip', 'reify')),
        public_only=public_only,
        edge_directed_map=edge_directed_dict,
    )

    slices_section, slice_weights = collect_slice_manifest(
        graph,
        requested_lids=(list(requested_lids) if requested_lids else None),
    )

    # REIFY: add HE nodes + membership edges
    if hyperedge_mode == 'reify':
        import networkx as nx

        allowed = None
        if requested_lids:
            allowed = set()
            for _lid, eids in slices_section.items():
                for eid in eids:
                    allowed.add(eid)

        is_multi_di = isinstance(nxG, nx.MultiDiGraph)

        # BATCH REIFY OPERATIONS
        he_nodes_to_add = []
        membership_edges_to_add = []

        for eid, spec in manifest_edges.items():
            if spec[-1] != 'hyper':
                continue
            if allowed is not None and eid not in allowed:
                continue

            head_map, tail_map = spec[0], spec[1]
            he_id = f'{reify_prefix}{eid}'

            he_attrs = _public(edge_attrs.get(eid, {}))
            he_directed_flag = edge_directed_dict.get(eid, True)
            he_attrs.update(
                {
                    'is_hyperedge': True,
                    'eid': eid,
                    'directed': bool(he_directed_flag),
                    'hyper_weight': float(weights_map.get(eid, 1.0)),
                }
            )

            he_nodes_to_add.append((he_id, he_attrs))

            if he_directed_flag:
                for u, coeff in (tail_map or {}).items():
                    membership_edges_to_add.append(
                        (
                            u,
                            he_id,
                            f'm::{eid}::{u}::tail',
                            {'role': 'tail', 'coeff': float(coeff), 'membership_of': eid},
                        )
                    )
                for v, coeff in (head_map or {}).items():
                    membership_edges_to_add.append(
                        (
                            he_id,
                            v,
                            f'm::{eid}::{v}::head',
                            {'role': 'head', 'coeff': float(coeff), 'membership_of': eid},
                        )
                    )
            else:
                members = {}
                members.update(tail_map or {})
                members.update(head_map or {})
                if is_multi_di:
                    for u, coeff in members.items():
                        base = f'm::{eid}::{u}::m'
                        membership_edges_to_add.append(
                            (
                                u,
                                he_id,
                                base + '::fwd',
                                {'role': 'member', 'coeff': float(coeff), 'membership_of': eid},
                            )
                        )
                        membership_edges_to_add.append(
                            (
                                he_id,
                                u,
                                base + '::rev',
                                {'role': 'member', 'coeff': float(coeff), 'membership_of': eid},
                            )
                        )
                else:
                    for u, coeff in members.items():
                        membership_edges_to_add.append(
                            (
                                u,
                                he_id,
                                f'm::{eid}::{u}::m',
                                {'role': 'member', 'coeff': float(coeff), 'membership_of': eid},
                            )
                        )

        # BULK ADD TO NETWORKX
        for he_id, attrs in he_nodes_to_add:
            if he_id not in nxG:
                nxG.add_node(he_id, **attrs)

        for u, v, key, attrs in membership_edges_to_add:
            nxG.add_edge(u, v, key=key, **attrs)

    manifest = {
        'edges': manifest_edges,
        'weights': weights_map,
        'slices': slices_section,
        'vertex_attrs': vertex_attrs,
        'edge_attrs': edge_attrs,
        'slice_weights': slice_weights,
        'edge_directed': edge_directed_dict,
        'manifest_version': 1,
        'multilayer': serialize_multilayer_manifest(
            graph,
            table_to_rows=dataframe_to_rows,
            serialize_edge_layers=serialize_edge_layers,
        ),
    }

    return nxG, manifest


def from_nx(
    nxG: nx.Graph,
    manifest: dict[str, Any],
    *,
    hyperedge: str = 'none',
    he_node_flag: str = 'is_hyperedge',
    he_id_attr: str = 'eid',
    role_attr: str = 'role',
    coeff_attr: str = 'coeff',
    membership_attr: str = 'membership_of',
    reify_prefix: str = 'he::',
) -> AnnNet:
    """Reconstruct a AnnNet from NetworkX graph + manifest.

    hyperedge: "none" (default) | "reified"
      When "reified", also detect hyperedge nodes in nxG and rebuild true hyperedges
      (in addition to those specified in the manifest).
    """
    from ..core import AnnNet

    H = AnnNet()
    timings = {}

    known_vertices = set()
    existing_eids = set()
    edge_directed_cache = manifest.get('edge_directed', {}) or {}
    weights_cache = manifest.get('weights', {}) or {}
    vertex_attrs_cache = manifest.get('vertex_attrs', {}) or {}
    edge_attrs_cache = manifest.get('edge_attrs', {}) or {}
    slices_cache = manifest.get('slices', {}) or {}
    slice_weights_cache = manifest.get('slice_weights', {}) or {}
    edges_def = manifest.get('edges', {}) or {}
    mm = manifest.get('multilayer', {})

    # Declare aspects up front so multilayer supra-node coordinates resolve while
    # edges and hyperedges are added below. The full multilayer state (presence,
    # attrs, edge layers) is restored from the same manifest at the end.
    _asp = mm.get('aspects')
    if _asp and tuple(H.aspects) != tuple(_asp):
        H.layers.set_aspects(list(_asp), mm.get('elem_layers') or None)

    def add_vertex_once(v):
        # Multilayer endpoints/members are (vid, layer_coord) supra-node keys, but
        # the vertex ENTITY is keyed by the bare vid; its layer placement is
        # restored from the manifest below. Collapsing here avoids registering a
        # duplicate vertex per (vid, layer).
        if isinstance(v, tuple):
            v = v[0]
        if v not in known_vertices:
            known_vertices.add(v)

    with _time('nx_vertices', timings):
        for v, d in nxG.nodes(data=True):
            if hyperedge == 'reified':
                d = d or {}
                if bool(d.get(he_node_flag, False)):
                    continue
                if isinstance(v, str) and v.startswith(reify_prefix):
                    continue
            add_vertex_once(v)

    # BULK INSERT VERTICES ONCE
    if known_vertices:
        H._add_vertices_bulk([{'vertex_id': v} for v in known_vertices])

    regular_edges_bulk = []
    hyperedges_bulk = []

    with _time('manifest_edges', timings):
        for eid, defn in edges_def.items():
            kind = defn[-1]
            is_dir = bool(edge_directed_cache.get(eid, True))

            if kind == 'hyper':
                head_map, tail_map = defn[0], defn[1]
                if isinstance(head_map, dict) and isinstance(tail_map, dict):
                    head = list(head_map)
                    tail = list(tail_map)
                    all_vertices = set(head) | set(tail)

                    for x in all_vertices:
                        add_vertex_once(x)

                    # BUILD ATTRIBUTES INLINE. Only emit the stoichiometry coeff
                    # maps when they carry real (non-unit) coefficients: their keys
                    # are member ids, which for multilayer graphs are supra-node
                    # tuples that the edge-attribute table cannot store.
                    if any(c != 1.0 for c in head_map.values()) or any(
                        c != 1.0 for c in tail_map.values()
                    ):
                        attrs = {
                            '__source_attr': {
                                u: {'__value': float(c)} for u, c in head_map.items()
                            },
                            '__target_attr': {
                                v: {'__value': float(c)} for v, c in tail_map.items()
                            },
                        }
                    else:
                        attrs = {}

                    if is_dir:
                        hyperedges_bulk.append(
                            {
                                'head': head,
                                'tail': tail,
                                'edge_id': eid,
                                'edge_directed': True,
                                'weight': weights_cache.get(eid, 1.0),
                                'attributes': attrs,
                            }
                        )
                    else:
                        hyperedges_bulk.append(
                            {
                                'members': list(all_vertices),
                                'edge_id': eid,
                                'edge_directed': False,
                                'weight': weights_cache.get(eid, 1.0),
                                'attributes': attrs,
                            }
                        )
                    existing_eids.add(eid)
            else:
                u, v = defn[0], defn[1]
                regular_edges_bulk.append(
                    {
                        'source': u,
                        'target': v,
                        'edge_id': eid,
                        'edge_directed': is_dir,
                        'weight': weights_cache.get(eid, 1.0),
                    }
                )
                existing_eids.add(eid)

        # SINGLE BULK CALL FOR REGULAR EDGES
        if regular_edges_bulk:
            H._add_edges_bulk(regular_edges_bulk, default_edge_directed=True)

        # SINGLE BULK CALL FOR HYPEREDGES
        if hyperedges_bulk:
            H.add_hyperedges_bulk(hyperedges_bulk)

    with _time('weights', timings):
        pass

    # BATCH SLICES
    with _time('slices', timings):
        restore_slice_manifest(H, slices_cache, slice_weights_cache)

    with _time('multilayer', timings):
        restore_multilayer_manifest(
            H,
            mm,
            rows_to_table=_rows_to_df,
            deserialize_edge_layers=deserialize_edge_layers,
        )

    # BATCH ATTRIBUTES — bulk reattach. Per-call set_*_attrs is O(N) and
    # also leads to incremental schema drift (a column inferred as String
    # from the first incoming row may then reject Int64 values from a
    # later row); bulk inference avoids both.
    with _time('attrs', timings):
        if vertex_attrs_cache:
            v_updates = {vid: a for vid, a in vertex_attrs_cache.items() if a}
            if v_updates:
                H.attrs.set_vertex_attrs_bulk(v_updates)

        if edge_attrs_cache:
            e_updates = {eid: a for eid, a in edge_attrs_cache.items() if a}
            if e_updates:
                H.attrs.set_edge_attrs_bulk(e_updates)

    if hyperedge == 'reified':
        with _time('reified', timings):
            hyperdefs, membership_edges = _nx_collect_reified_optimized(
                nxG,
                he_node_flag=he_node_flag,
                he_id_attr=he_id_attr,
                role_attr=role_attr,
                coeff_attr=coeff_attr,
                membership_attr=membership_attr,
            )

            reified_hyperedges_bulk = []

            for eid, directed, head_map, tail_map, he_attrs, _he_node in hyperdefs:
                if eid in existing_eids:
                    continue

                all_vertices = set(head_map) | set(tail_map)
                for x in all_vertices:
                    add_vertex_once(x)

                # BUILD ATTRIBUTES INLINE. Only emit stoichiometry coeff maps when
                # they carry real (non-unit) coefficients — their keys are member
                # ids, which for multilayer graphs are unserialisable supra-node
                # tuples (and unit coefficients carry no information).
                attrs = {}
                if any(c != 1.0 for c in head_map.values()) or any(
                    c != 1.0 for c in tail_map.values()
                ):
                    attrs['__source_attr'] = {u: {'__value': c} for u, c in head_map.items()}
                    attrs['__target_attr'] = {v: {'__value': c} for v, c in tail_map.items()}

                clean_attrs = {
                    k: v
                    for k, v in (he_attrs or {}).items()
                    if k not in _HE_RESERVED_ATTRS and k not in {he_node_flag, he_id_attr}
                }
                if clean_attrs:
                    attrs.update(clean_attrs)

                if directed:
                    reified_hyperedges_bulk.append(
                        {
                            'head': list(head_map),
                            'tail': list(tail_map),
                            'edge_id': eid,
                            'edge_directed': True,
                            'attributes': attrs,
                        }
                    )
                else:
                    reified_hyperedges_bulk.append(
                        {
                            'members': list(all_vertices),
                            'edge_id': eid,
                            'edge_directed': False,
                            'attributes': attrs,
                        }
                    )
                existing_eids.add(eid)

            # SINGLE BULK CALL FOR REIFIED HYPEREDGES
            if reified_hyperedges_bulk:
                H.add_hyperedges_bulk(reified_hyperedges_bulk)

    return H


def _nx_collect_reified_optimized(
    nxG, *, he_node_flag, he_id_attr, role_attr, coeff_attr, membership_attr
):
    """Optimized: pre-index edges by incident node to avoid O(H×E) full scans."""

    if nxG.is_multigraph():
        edge_iter = nxG.edges(keys=True, data=True)

        def edge_key(u, v, k):
            return (u, v, k)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in nxG.edges(data=True))

        def edge_key(u, v, k):
            return (u, v, None)

    node_to_edges = {}
    for u, v, k, d in edge_iter:
        node_to_edges.setdefault(u, []).append((u, v, k, d))
        node_to_edges.setdefault(v, []).append((u, v, k, d))

    hyperdefs = []
    membership_edges = set()

    for he_node, he_attrs in nxG.nodes(data=True):
        he_attrs = he_attrs or {}
        if not bool(he_attrs.get(he_node_flag, False)):
            continue

        eid = he_attrs.get(he_id_attr, he_node)

        head_map = {}
        tail_map = {}
        directed = False

        incident_edges = node_to_edges.get(he_node, [])

        for u, v, k, d in incident_edges:
            d = d or {}
            membership_edges.add(edge_key(u, v, k))

            other = v if u == he_node else u
            role = d.get(role_attr, 'member')
            coeff = d.get(coeff_attr, d.get('__value', 1.0))
            try:
                coeff = float(coeff)
            except (TypeError, ValueError):
                coeff = 1.0

            if role == 'head':
                head_map[other] = coeff
                directed = True
            elif role == 'tail':
                tail_map[other] = coeff
                directed = True
            else:
                head_map[other] = coeff
                tail_map[other] = coeff

        if head_map or tail_map:
            hyperdefs.append((eid, directed, head_map, tail_map, he_attrs, he_node))

    return hyperdefs, membership_edges


def _from_nx_without_manifest(
    nxG,
    *,
    hyperedge='none',
    he_node_flag='is_hyperedge',
    he_id_attr='eid',
    role_attr='role',
    coeff_attr='coeff',
    membership_attr='membership_of',
    reify_prefix='he::',
):
    """Best-effort import from a bare NetworkX graph (no manifest).

    Bulk-batched: collect vertices, edges, and attributes into lists, then
    issue exactly one bulk insert call per kind. Previous per-vertex /
    per-edge inserts were O(N²) in dataframe-concat work and timed out at
    medium scale (85K edges → 600s+).

    hyperedge: "none" (default) | "reified"
      When "reified", detect hyperedge nodes + membership edges and rebuild
      true hyperedges.
    """

    from ..core import AnnNet

    H = AnnNet()

    # 1) Collect vertices + their attrs (skip HE nodes if reified)
    vertex_buf: list[str] = []
    vertex_seen: set = set()
    vertex_attrs_buf: dict[str, dict] = {}

    def _add_vertex(v):
        if v not in vertex_seen:
            vertex_seen.add(v)
            vertex_buf.append(v)

    for v, d in nxG.nodes(data=True):
        if hyperedge == 'reified':
            if bool((d or {}).get(he_node_flag, False)):
                continue
            if isinstance(v, str) and str(v).startswith(reify_prefix):
                continue
        _add_vertex(v)
        if d:
            vertex_attrs_buf[v] = dict(d)

    # 2) Optionally collect reified hyperedges
    membership_edges: set = set()
    hyperedges_bulk: list[dict] = []
    hyperedge_attrs_buf: dict[str, dict] = {}

    if hyperedge == 'reified':
        hyperdefs, membership_edges = _nx_collect_reified_optimized(
            nxG,
            he_node_flag=he_node_flag,
            he_id_attr=he_id_attr,
            role_attr=role_attr,
            coeff_attr=coeff_attr,
            membership_attr=membership_attr,
        )
        for eid, directed, head_map, tail_map, he_attrs, _he_node in hyperdefs:
            edge_weight = 1.0
            if isinstance(he_attrs, dict):
                try:
                    edge_weight = float(he_attrs.get('hyper_weight', 1.0))
                except (TypeError, ValueError):
                    edge_weight = 1.0
            for x in set(head_map) | set(tail_map):
                _add_vertex(x)

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
                        'weight': edge_weight,
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
                        'weight': edge_weight,
                        'attributes': attrs,
                    }
                )

            clean_attrs = {
                k: v
                for k, v in (he_attrs or {}).items()
                if k not in {he_node_flag, he_id_attr, 'directed', 'hyper_weight'}
            }
            if clean_attrs:
                hyperedge_attrs_buf[eid] = clean_attrs

    # 3) Collect binary edges (skip membership edges consumed by reify)
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

    binary_edges_bulk: list[dict] = []
    edge_attrs_buf: dict[str, dict] = {}
    seen_auto = 0
    for u, v, key, d in iterator:
        if hyperedge == 'reified' and EK(u, v, key) in membership_edges:
            continue

        d = d or {}
        eid = (d.get('eid') if isinstance(d, dict) else None) or (key if key is not None else None)
        if eid is None:
            seen_auto += 1
            eid = f'nx::e#{seen_auto}'

        e_directed = bool(d.get('directed', is_dir))
        w = d.get('weight', d.get('__weight', 1.0))
        _add_vertex(u)
        _add_vertex(v)

        binary_edges_bulk.append(
            {
                'source': u,
                'target': v,
                'edge_id': eid,
                'edge_directed': e_directed,
                'weight': float(w),
            }
        )

        if isinstance(d, dict) and d:
            clean = dict(d)
            for k in ('eid', 'id', 'key', 'weight', '__weight', 'directed'):
                clean.pop(k, None)
            if clean:
                edge_attrs_buf[eid] = clean

    # 4) BULK INSERTS — one call per kind. This is what the manifest path
    # already does; the bare path was previously calling add_vertices /
    # add_edges / set_*_attrs once per element.
    if vertex_buf:
        H._add_vertices_bulk([{'vertex_id': v} for v in vertex_buf])
    if binary_edges_bulk:
        H._add_edges_bulk(binary_edges_bulk, default_edge_directed=is_dir)
    if hyperedges_bulk:
        H.add_hyperedges_bulk(hyperedges_bulk)
    if vertex_attrs_buf:
        H.attrs.set_vertex_attrs_bulk(vertex_attrs_buf)
    if edge_attrs_buf or hyperedge_attrs_buf:
        merged_edge_attrs = {**edge_attrs_buf, **hyperedge_attrs_buf}
        H.attrs.set_edge_attrs_bulk(merged_edge_attrs)

    return H


def from_nx_without_manifest(
    nxG,
    *,
    hyperedge='none',
    he_node_flag='is_hyperedge',
    he_id_attr='eid',
    role_attr='role',
    coeff_attr='coeff',
    membership_attr='membership_of',
    reify_prefix='he::',
):
    """Public façade for best-effort import from bare NetworkX graphs."""
    return _from_nx_without_manifest(
        nxG,
        hyperedge=hyperedge,
        he_node_flag=he_node_flag,
        he_id_attr=he_id_attr,
        role_attr=role_attr,
        coeff_attr=coeff_attr,
        membership_attr=membership_attr,
        reify_prefix=reify_prefix,
    )
