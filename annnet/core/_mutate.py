"""The mutation gateway — the only writer of canonical state."""

from __future__ import annotations

from sys import intern as _intern
import json

from . import _store as ST, _derive as D, _identity as I, _structure as S
from ._records import (
    EdgeRecord,
    SliceRecord,
    EntityRecord,
    _df_filter_not_equal,
    _internal_entity_kind,
)
from .._support.dataframe_backend import (
    dataframe_columns,
    dataframe_drop_rows,
    is_polars_dataframe,
    dataframe_upsert_rows,
    polars_upsert_vertices,
)

# ---------------------------------------------------------------------------
# The slot store, written beside the records
# ---------------------------------------------------------------------------
# A graph built with ``store='slots'`` carries a slot store as well as its
# records, and the gateway writes both. The query facade answers from the slot
# store as soon as one is there, so every read of such a graph exercises the new
# core while the records remain the thing the gateway knows how to build.
#
# These functions read the records directly, which the facade would not let them
# do, because the facade would answer from the slot store they are here to fill.
# The gateway owns both stores, so it is the one place that may.
#
# When the records go, the sync points become the writes themselves.

_SLOT_ENTITY_KIND = {'vertex': ST.NODE, 'edge_entity': ST.EDGE_ENTITY}
_SLOT_EDGE_KIND = {
    'binary': ST.BINARY,
    'hyper': ST.HYPER,
    'vertex_edge': ST.NODE_EDGE,
    'edge_placeholder': ST.PLACEHOLDER,
}


def slot_store(g):
    """Return the slot store of a graph, or None when it keeps records alone."""
    return getattr(g, '_store', None)


def sync_entity(g, ekey) -> None:
    """Make the slot store agree with the records about one entity."""
    store = slot_store(g)
    if store is None:
        return
    record = g._entities.get(ekey)
    slot = store.entity_slot(ekey)
    if record is None:
        if slot is not None:
            store.remove_entity(ekey)
        return
    kind = _SLOT_ENTITY_KIND.get(record.kind, ST.NODE)
    if slot is None:
        store.add_entity(ekey, kind)
    else:
        store.entity_kind[slot] = kind


def _members_of_record(g, store, edge_id, record) -> list:
    """Return the member entries the slot store holds for one edge record."""
    sides = S.Endpoints(S._raw_side(record.src), S._raw_side(record.tgt))
    reference = S._edge_ref_of_record(g, edge_id, record)
    coefficients = dict(record.coeffs) if record.coeffs is not None else None
    return ST.members_from_sides(store, g, sides, coefficients, reference.weight, reference)


def sync_edge(g, edge_id) -> None:
    """Make the slot store agree with the records about one edge.

    An edge is replaced rather than patched, because a change of definition can
    change how many entries the member list holds.
    """
    store = slot_store(g)
    if store is None:
        return
    if store.edge_slot(edge_id) is not None:
        store.remove_edge(edge_id)
    record = g._edges.get(edge_id)
    if record is None:
        return
    store.add_edge(
        edge_id,
        _members_of_record(g, store, edge_id, record),
        kind=_SLOT_EDGE_KIND.get(record.etype, ST.BINARY),
        directed=record.directed,
        weight=record.weight,
        explicit_coefficients=record.coeffs is not None,
        ml_kind=record.ml_kind,
        ml_layers=record.ml_layers,
        direction_policy=record.direction_policy,
    )


def rewrite_edge(g, edge_id) -> None:
    """Give the slot store the definition an edge record now carries.

    The edge keeps the slot it holds and everything the write did not change,
    where :func:`sync_edge` would remove it and add it again. A write that
    changes which entities an edge names uses this; a write that changes one
    field of an edge reaches the store through the narrow write for that field.
    """
    store = slot_store(g)
    if store is None:
        return
    record = g._edges.get(edge_id)
    if record is None or store.edge_slot(edge_id) is None:
        sync_edge(g, edge_id)
        return
    store.set_edge_kind(edge_id, _SLOT_EDGE_KIND.get(record.etype, ST.BINARY))
    store.set_edge_directed(edge_id, record.directed)
    store.replace_members(edge_id, _members_of_record(g, store, edge_id, record))


def sync_edges(g, edge_ids) -> None:
    """Make the slot store agree with the records about several edges."""
    if slot_store(g) is None:
        return
    for edge_id in edge_ids:
        sync_edge(g, edge_id)


def sync_aspects(g) -> None:
    """Make the slot store agree with the graph about the declared aspects.

    The store answers in the identity form its aspects imply: a bare id when it
    holds one layer, and an ``(id, layer_coord)`` pair otherwise. So a graph that
    declares aspects has to tell the store, or the store keeps answering in the
    flat form for a graph that is no longer flat.
    """
    store = slot_store(g)
    if store is not None:
        store.aspects = tuple(g._aspects)


def resync(g) -> None:
    """Rebuild the whole slot store from the records.

    A few mutations replace the entity registry outright rather than changing one
    entry, and every edge that named a replaced entity has to be written again.
    Those are rare, so they pay for a rebuild instead of the gateway tracking what
    each of them touched.
    """
    store = slot_store(g)
    if store is None:
        return
    rebuilt = ST.CoreState(directed=g.directed, aspects=g._aspects)
    for ekey, record in sorted(g._entities.items(), key=lambda item: item[1].row_idx):
        rebuilt.add_entity(ekey, _SLOT_ENTITY_KIND.get(record.kind, ST.NODE))
    g._store = rebuilt
    for edge_id in sorted(g._edges, key=lambda eid: g._edges[eid].col_idx):
        sync_edge(g, edge_id)


# ---------------------------------------------------------------------------
# Which writes reach the slot store, and how
# ---------------------------------------------------------------------------
# A write that changes canonical structure has to reach the slot store too, and
# nearly every write here does it itself, from what the call already holds. What
# is left below is for the writes that do not.
#
# ``add_vertex`` and ``register_edge_as_entity`` need nothing. They reach the
# entity registry through ``register_entity_record``, which writes the slot store
# itself. ``batch_add_vertices`` registers its entities inline rather than through
# that setter, so it writes the store inline as well, at the same place.


def _keep_identity(routed, write):
    routed.__name__ = write.__name__
    routed.__doc__ = write.__doc__
    routed.__wrapped__ = write
    return routed


def rebuilds_the_store(write):
    """Rebuild the slot store whole.

    One legacy setter is left that replaces the entity registry outright rather
    than going through the two doors. Every edge names an entity, so no slot and
    no member list survives it. It runs once per call and never per element, and
    it goes with the records.
    """

    def routed(g, *args, **kwargs):
        result = write(g, *args, **kwargs)
        resync(g)
        return result

    return _keep_identity(routed, write)


# ---------------------------------------------------------------------------
# Entity record bookkeeping
# ---------------------------------------------------------------------------


def register_entity_record(g, ekey, rec: EntityRecord) -> None:
    """Register an entity record and update derived entity indexes."""
    old = g._entities.get(ekey)
    if old is not None and old.row_idx in g._row_to_entity:
        g._row_to_entity.pop(old.row_idx, None)
    g._entities[ekey] = rec
    g._row_to_entity[rec.row_idx] = ekey
    D.index_entity_key(g, ekey)
    sync_entity(g, ekey)


def remove_entity_record(g, ekey):
    """Remove an entity record and update derived entity indexes."""
    rec = g._entities.pop(ekey)
    g._row_to_entity.pop(rec.row_idx, None)
    D.unindex_entity_key(g, ekey)
    sync_entity(g, ekey)
    return rec


# ---------------------------------------------------------------------------
# Vertices
# ---------------------------------------------------------------------------


def add_vertex(g, vertex_id, slice=None, layer=None, **attributes):
    """Add or update a vertex; returns its id."""
    if slice is None:
        slice = g._current_slice

    coord = I.resolve_vertex_insert_coord(g, layer, vertex_ids=vertex_id, context='add_vertex')
    key = (vertex_id, coord)

    if key not in g._entities:
        idx = len(g._entities)
        register_entity_record(g, key, EntityRecord(row_idx=idx, kind='vertex'))
        D.mark_matrix_stale(g)

    if slice not in g._slices:
        g._slices[slice] = SliceRecord()
    g._slices[slice]['vertices'].add(vertex_id)

    g._ensure_vertex_table()
    g._ensure_vertex_row(vertex_id)

    if attributes:
        g.vertex_attributes = g._upsert_row(g.vertex_attributes, vertex_id, attributes)

    return vertex_id


# ---------------------------------------------------------------------------
# Edge-entities
# ---------------------------------------------------------------------------


def register_edge_as_entity(g, edge_id):
    """Ensure an edge ID has a matching edge-entity record."""
    ekey = I.resolve_ekey(g, edge_id)
    if ekey in g._entities:
        return
    idx = len(g._entities)
    register_entity_record(g, ekey, EntityRecord(row_idx=idx, kind='edge_entity'))
    D.mark_matrix_stale(g)


def ensure_edge_entity_placeholder(g, edge_id, slice=None, **attributes):
    """Ensure a placeholder edge-entity exists and is attached to a slice."""
    register_edge_as_entity(g, edge_id)
    if edge_id not in g._edges:
        g._edges[edge_id] = EdgeRecord(
            src=None,
            tgt=None,
            weight=1.0,
            directed=False,
            etype='edge_placeholder',
            col_idx=-1,
            ml_kind=None,
            ml_layers=None,
            direction_policy=None,
        )
        # A placeholder is an edge the graph knows the name of and nothing else,
        # so it holds no members and occupies no column.
        store = slot_store(g)
        if store is not None and store.edge_slot(edge_id) is None:
            store.add_edge(edge_id, (), kind=ST.PLACEHOLDER, directed=False, weight=1.0)
    slice = slice or g._current_slice
    if slice is not None:
        g.slices._ensure_slice(slice)['edges'].add(edge_id)
    if attributes:
        g.attrs.set_edge_attrs(edge_id, **attributes)
    g._ensure_edge_row(edge_id)
    return edge_id


# ---------------------------------------------------------------------------
# Edge input parsing + multilayer-role inference
# ---------------------------------------------------------------------------


def parse_edge_inputs(g, src, tgt, weight):
    """Normalize src/tgt to (src_nodes, tgt_nodes, col_entries_or_None, etype)."""
    if isinstance(src, dict):
        if tgt is None:
            src_nodes = frozenset(k for k, v in src.items() if v <= 0)
            tgt_nodes = frozenset(k for k, v in src.items() if v > 0)
            return src_nodes, tgt_nodes, dict(src), 'stoich'
        if isinstance(tgt, dict):
            return frozenset(src), frozenset(tgt), {**src, **tgt}, 'stoich'
        raise TypeError(f'If src is dict, tgt must be dict or None, got {type(tgt).__name__!r}')

    if (
        isinstance(src, tuple)
        and len(src) == 2
        and isinstance(src[1], tuple)
        and tgt is not None
        and isinstance(tgt, tuple)
        and len(tgt) == 2
        and isinstance(tgt[1], tuple)
    ):
        return frozenset({src}), frozenset({tgt}), None, 'binary'

    if isinstance(src, str):
        if tgt is None:
            raise ValueError('Binary edge requires tgt when src is a string.')
        if not isinstance(tgt, str):
            raise TypeError(f'tgt must be str for binary edge, got {type(tgt).__name__!r}')
        return frozenset({src}), frozenset({tgt}), None, 'binary'

    if isinstance(src, (list, set, frozenset)):
        src_seq = list(src)
        if tgt is None:
            return frozenset(src_seq), frozenset(), None, 'hyper'
        if isinstance(tgt, (list, set, frozenset)):
            return frozenset(src_seq), frozenset(tgt), None, 'hyper'
        raise TypeError(
            f'If src is list/set, tgt must be list/set or None, got {type(tgt).__name__!r}'
        )

    raise TypeError(
        f'src must be str, tuple (supra-node), list, set, or dict; got {type(src).__name__!r}'
    )


def infer_ml_kind(src_key, tgt_key):
    """Classify a binary supra-edge as intra, inter, or coupling."""
    vid_s, lay_s = src_key
    vid_t, lay_t = tgt_key
    if lay_s == lay_t:
        return 'intra'
    if vid_s == vid_t:
        return 'coupling'
    return 'inter'


def infer_hyper_ml(head_keys, tail_keys):
    """Infer the multilayer role and layer pairing for a hyperedge."""
    head_layers = {k[1] for k in head_keys} if head_keys else set()
    tail_layers = {k[1] for k in tail_keys} if tail_keys else set()
    all_layers = head_layers | tail_layers
    if len(all_layers) <= 1:
        return 'intra', (next(iter(all_layers)) if all_layers else None)
    head_vids = {k[0] for k in head_keys} if head_keys else set()
    tail_vids = {k[0] for k in tail_keys} if tail_keys else set()
    kind = 'coupling' if len(head_vids | tail_vids) == 1 else 'inter'
    if len(head_layers) == 1 and len(tail_layers) == 1:
        return kind, (next(iter(head_layers)), next(iter(tail_layers)))
    return kind, None


def find_parallel_edges(g, endpoint_set, etype):
    """Return the ids of the edges that hold exactly this endpoint set.

    A parallel edge is one that names the same entities as the edge about to be
    added, in either direction. The answer comes from the edges that touch one of
    those entities, so it costs the degree of one of them rather than a pass over
    every edge. The endpoints of a new edge need not exist yet, and an endpoint
    the graph does not hold can be in no edge.
    """
    wanted = frozenset(endpoint_set)
    if not wanted:
        return []
    probe = next(iter(wanted))
    if not S.has_entity(g, probe):
        return []
    is_hyper = etype != 'binary'
    result = []
    for edge_id in S.entity_edges(g, probe, 'both'):
        if (S.edge_ref(g, edge_id).kind == S.HYPER) != is_hyper:
            continue
        sides = S.edge_sides(g, edge_id)
        if sides.source | sides.target == wanted:
            result.append(edge_id)
    return result


def zero_edge_column(g, rec, col_idx):
    """Zero the realized matrix entries for an edge column in place."""
    M = g._matrix
    _fast = getattr(M, '_set_intXint', None)

    def _z(node):
        try:
            r = S.entity_row(g, node)
            if _fast:
                _fast(r, col_idx, 0)
            else:
                M[r, col_idx] = 0
        except (KeyError, ValueError, TypeError):
            pass

    if rec.etype == 'hyper':
        for side in (rec.src, rec.tgt):
            if isinstance(side, frozenset):
                for n in side:
                    _z(n)
            elif side is not None:
                _z(side)
    else:
        if rec.src is not None:
            _z(rec.src)
        if rec.tgt is not None and rec.tgt != rec.src:
            _z(rec.tgt)


# ---------------------------------------------------------------------------
# The canonical edge constructor (15 stages)
# ---------------------------------------------------------------------------


def add_edge(
    g,
    src=None,
    tgt=None,
    *,
    weight=1.0,
    edge_id=None,
    directed=None,
    parallel='update',
    slice=None,
    as_entity=False,
    propagate='none',
    flexible=None,
    **attrs,
):
    """Add or update an edge of any type; returns its id. The only edge writer."""
    if parallel not in {'update', 'error', 'parallel'}:
        raise ValueError(f"parallel must be 'update'|'error'|'parallel', got {parallel!r}")
    if propagate not in {'none', 'shared', 'all'}:
        raise ValueError(f"propagate must be 'none'|'shared'|'all', got {propagate!r}")
    if not isinstance(weight, (int, float)):
        raise TypeError(f'weight must be numeric, got {type(weight).__name__!r}')
    if flexible is not None and (
        not isinstance(flexible, dict) or 'var' not in flexible or 'threshold' not in flexible
    ):
        raise ValueError(
            "flexible must be a dict with keys {'var','threshold'[,'scope','above','tie']}"
        )

    slice = slice if slice is not None else g._current_slice

    if src is None and tgt is None:
        if as_entity:
            if edge_id is None:
                raise ValueError(
                    'edge_id is required when creating an edge-entity without endpoints.'
                )
            return ensure_edge_entity_placeholder(g, edge_id, slice=slice, **attrs)
        raise ValueError('add_edge requires structural endpoints unless as_entity=True.')

    # 1. Parse inputs
    src_nodes, tgt_nodes, col_entries_literal, etype = parse_edge_inputs(g, src, tgt, weight)

    is_multilayer = g._aspects != ('_',)
    if is_multilayer and col_entries_literal is None:

        def _promote(node_set):
            promoted = set()
            bare = []
            for node in node_set:
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                    promoted.add(node)
                    continue
                ekey = I.resolve_ekey(g, node)
                promoted.add(ekey)
                if ekey not in g._entities:
                    bare.append(node)
            return promoted, bare

        src_nodes, bare_src = _promote(src_nodes)
        tgt_nodes, bare_tgt = _promote(tgt_nodes)
        bare_total = bare_src + bare_tgt
        if bare_total:
            I.ensure_placeholder_layers_declared(g)
            I.warn_placeholder_vertex_assignment(g, bare_total, context='add_edges')

    # 2. Resolve direction
    if directed is not None:
        is_dir = bool(directed)
    elif etype == 'hyper':
        is_dir = bool(tgt_nodes)
    elif g.directed is not None:
        is_dir = bool(g.directed)
    else:
        is_dir = True

    # 3. Build column entries
    if col_entries_literal is not None:
        col_entries = col_entries_literal
    else:
        col_entries = {}
        for n in src_nodes:
            col_entries[n] = float(weight)
        for n in tgt_nodes:
            col_entries[n] = -float(weight) if is_dir else float(weight)

    endpoint_set = frozenset(col_entries)

    # 4. Resolve parallel
    explicit_id = edge_id is not None
    if explicit_id and edge_id in g._edges:
        pass
    elif explicit_id:
        if parallel == 'error':
            if find_parallel_edges(g, endpoint_set, etype):
                raise ValueError(
                    f'Edge already exists between {endpoint_set}. '
                    "Use parallel='parallel' to allow parallel edges."
                )
    else:
        existing = find_parallel_edges(g, endpoint_set, etype)
        if existing:
            if parallel == 'error':
                raise ValueError(
                    f'Edge already exists between {endpoint_set}. '
                    "Use parallel='parallel' to allow parallel edges."
                )
            if parallel == 'update':
                edge_id = existing[-1]
        if edge_id is None:
            edge_id = g._get_next_edge_id()

    # 5. Ensure endpoints exist
    _ent = g._entities
    for node in endpoint_set:
        ekey = I.resolve_ekey(g, node)
        if ekey not in _ent:
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                add_vertex(g, node[0], layer=node[1], slice=slice)
            else:
                add_vertex(g, node, slice=slice)
    D.mark_matrix_stale(g)

    # 6. Column allocation
    _edg = g._edges
    is_new = edge_id not in _edg or _edg[edge_id].col_idx < 0
    if is_new:
        col_idx = len(g._col_to_edge)
        g._col_to_edge[col_idx] = edge_id
        D.mark_matrix_stale(g)
    else:
        col_idx = _edg[edge_id].col_idx

    # 7. Incidence is derived lazily from the record written below; marking the
    #    cache dirty lets the next reader rebuild this column (old contents, if
    #    any, are recomputed from the overwritten record — no explicit zeroing).
    g._mark_matrix_dirty()

    # 8. Compute src_store / tgt_store
    if etype == 'binary':
        src_store = next(iter(src_nodes))
        tgt_store = next(iter(tgt_nodes)) if tgt_nodes else None
        rec_etype = 'vertex_edge' if as_entity else 'binary'
    else:
        src_store = frozenset(src_nodes) if src_nodes else None
        tgt_store = frozenset(tgt_nodes) if tgt_nodes else None
        rec_etype = 'hyper'

    # 9. Infer ml_kind / ml_layers
    ml_kind = None
    ml_layers = None
    if not is_multilayer:
        ml_kind = 'intra'
    elif (
        etype == 'binary'
        and isinstance(src, tuple)
        and len(src) == 2
        and isinstance(src[1], tuple)
        and isinstance(tgt, tuple)
        and len(tgt) == 2
        and isinstance(tgt[1], tuple)
    ):
        ml_kind = infer_ml_kind(src, tgt)
        ml_layers = (src[1], tgt[1])
    elif etype == 'hyper':
        ml_kind, ml_layers = infer_hyper_ml(src_nodes, tgt_nodes)

    # 10. Store / update EdgeRecord (+ adjacency indexes)
    # Literal coefficients (stoich) make the record complete; plain +/-weight columns
    # leave coeffs=None (derivable from weight + directed).
    rec_coeffs = dict(col_entries) if col_entries_literal is not None else None
    if is_new:
        _edg[edge_id] = EdgeRecord(
            src=src_store,
            tgt=tgt_store,
            weight=float(weight),
            directed=is_dir,
            etype=rec_etype,
            col_idx=col_idx,
            ml_kind=ml_kind,
            ml_layers=ml_layers,
            direction_policy=flexible,
            coeffs=rec_coeffs,
        )
    else:
        rec = _edg[edge_id]
        rec.src = src_store
        rec.tgt = tgt_store
        rec.weight = float(weight)
        rec.directed = is_dir
        rec.etype = rec_etype
        rec.ml_kind = ml_kind
        rec.ml_layers = ml_layers
        rec.coeffs = rec_coeffs
        if flexible is not None:
            rec.direction_policy = flexible

    # 10b. The slot store. The two sides and the coefficient each endpoint takes
    #      are what this call has just worked out, so the store takes them from
    #      here rather than reading the record back to work them out again.
    store = slot_store(g)
    if store is not None:
        if store.edge_slot(edge_id) is not None:
            store.remove_edge(edge_id)
        store.add_edge(
            edge_id,
            ST.members_from_endpoints(
                store, g, src_nodes, tgt_nodes, rec_coeffs, float(weight), is_dir
            ),
            kind=_SLOT_EDGE_KIND.get(rec_etype, ST.BINARY),
            directed=is_dir,
            weight=float(weight),
            explicit_coefficients=rec_coeffs is not None,
            ml_kind=ml_kind,
            ml_layers=ml_layers,
            direction_policy=_edg[edge_id].direction_policy,
        )

    # 11. as_entity
    if as_entity:
        register_edge_as_entity(g, edge_id)

    # 12. Slice (tracks bare vids)
    if slice is not None:
        slices = g._slices
        if slice not in slices:
            slices[slice] = SliceRecord()
        slices[slice]['edges'].add(edge_id)
        for n in endpoint_set:
            slices[slice]['vertices'].add(n[0] if isinstance(n, tuple) else n)

    # 13. Propagate
    if propagate == 'shared':
        propagate_to_shared_slices(g, edge_id, src_store, tgt_store)
    elif propagate == 'all':
        propagate_to_all_slices(g, edge_id, src_store, tgt_store)

    # 14. Flexible direction. The policy reads the edge back, so the store has to
    # hold the change before the hook runs.
    if flexible is not None:
        _edg[edge_id].directed = True
        if store is not None:
            store.set_edge_directed(edge_id, True)
        g._apply_flexible_direction(edge_id)

    # 15. Attributes + ensure var row
    if attrs:
        g.attrs.set_edge_attrs(edge_id, **attrs)
    g._ensure_edge_row(edge_id)

    return edge_id


# ---------------------------------------------------------------------------
# Slice propagation
# ---------------------------------------------------------------------------


def propagate_to_shared_slices(g, edge_id, source, target):
    """Add an edge to slices that already contain both endpoints."""
    for slice_data in g._slices.values():
        sv = slice_data['vertices']
        if I.slice_contains_endpoint(g, sv, source) and I.slice_contains_endpoint(g, sv, target):
            slice_data['edges'].add(edge_id)


def propagate_to_all_slices(g, edge_id, source, target):
    """Propagate an edge to slices containing either endpoint, adding the other endpoint as needed."""
    for slice_data in g._slices.values():
        sv = slice_data['vertices']
        source_present = I.slice_contains_endpoint(g, sv, source)
        target_present = I.slice_contains_endpoint(g, sv, target)
        if source_present or target_present:
            slice_data['edges'].add(edge_id)
            if source_present:
                I.add_endpoint_to_slice_vertices(g, sv, target)
            if target_present:
                I.add_endpoint_to_slice_vertices(g, sv, source)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def remove_edge(g, edge_id):
    """Remove a single edge, its column, attributes, and slice memberships."""
    if edge_id not in g._edges:
        raise KeyError(f'Edge {edge_id} not found')

    rec = g._edges[edge_id]
    col_idx = rec.col_idx

    D.mark_matrix_stale(g)
    D.invalidate_sparse_caches(g)

    del g._col_to_edge[col_idx]
    num_edges = len(g._col_to_edge) + 1
    for old_c in range(col_idx + 1, num_edges):
        eid = g._col_to_edge.pop(old_c)
        g._col_to_edge[old_c - 1] = eid
        g._edges[eid].col_idx = old_c - 1

    del g._edges[edge_id]
    drop_edges(g, (edge_id,))

    ea = g.edge_attributes
    if ea is not None and hasattr(ea, 'columns'):
        is_empty = (getattr(ea, 'height', None) == 0) or (hasattr(ea, '__len__') and len(ea) == 0)
        if (not is_empty) and ('edge_id' in list(ea.columns)):
            g.edge_attributes = _df_filter_not_equal(ea, 'edge_id', edge_id)

    for slice_data in g._slices.values():
        slice_data['edges'].discard(edge_id)

    esa = g.edge_slice_attributes
    if esa is not None and hasattr(esa, 'columns'):
        is_empty = (getattr(esa, 'height', None) == 0) or (
            hasattr(esa, '__len__') and len(esa) == 0
        )
        if (not is_empty) and ('edge_id' in list(esa.columns)):
            g.edge_slice_attributes = _df_filter_not_equal(esa, 'edge_id', edge_id)

    drop_orphan_edge_entities(g, (edge_id,))
    g._rebuild_slice_edge_weights_cache()


def remove_edges_bulk(g, edge_ids):
    """Remove many edges and compact the remaining edge columns."""
    drop = set(edge_ids)
    if not drop:
        return

    keep_pairs = [(col, eid) for col, eid in g._col_to_edge.items() if eid not in drop]

    D.mark_matrix_stale(g)
    D.invalidate_sparse_caches(g)

    g._col_to_edge.clear()
    for new_idx, (_old_idx, eid) in enumerate(keep_pairs):
        g._col_to_edge[new_idx] = eid
        g._edges[eid].col_idx = new_idx

    for eid in drop:
        g._edges.pop(eid, None)
    drop_edges(g, drop)

    for slice_data in g._slices.values():
        slice_data['edges'].difference_update(drop)
    for d in g.slice_edge_weights.values():
        for eid in drop:
            d.pop(eid, None)

    ea = g.edge_attributes
    if ea is not None and 'edge_id' in dataframe_columns(ea):
        g.edge_attributes = dataframe_drop_rows(ea, 'edge_id', drop)
    ela = g.edge_slice_attributes
    if ela is not None and 'edge_id' in dataframe_columns(ela):
        g.edge_slice_attributes = dataframe_drop_rows(ela, 'edge_id', drop)

    drop_orphan_edge_entities(g, drop)


# ---------------------------------------------------------------------------
# Legacy structural setters (routed through the gateway, not poked directly)
# ---------------------------------------------------------------------------


def set_edge_definition(g, eid, src, tgt, etype):
    """Rewrite a binary edge's endpoints/type (legacy edge_definitions setter)."""
    rec = g._edges.get(eid)
    if rec is None:
        return
    rec.src = src
    rec.tgt = tgt
    rec.etype = etype if etype != 'hyper' else 'binary'
    g._mark_matrix_dirty()
    rewrite_edge(g, eid)


def set_hyperedge_definition(g, eid, defn):
    """Rewrite a hyperedge's membership (legacy hyperedge_definitions setter)."""
    rec = g._edges.get(eid)
    if rec is None:
        return
    rec.etype = 'hyper'
    if isinstance(defn, list):
        rec.src = frozenset(defn)
        rec.tgt = None
        rec.directed = False
    elif bool(defn.get('directed', False)):
        rec.src = frozenset(defn.get('head', []))
        rec.tgt = frozenset(defn.get('tail', []))
        rec.directed = True
    else:
        rec.src = frozenset(defn.get('members', []))
        rec.tgt = None
        rec.directed = False
    g._mark_matrix_dirty()
    rewrite_edge(g, eid)


def set_edge_direction_policy(g, eid, policy):
    """Attach a flexible-direction policy to an edge (legacy setter)."""
    rec = g._edges.get(eid)
    if rec is None:
        return
    rec.direction_policy = policy
    store = slot_store(g)
    if store is not None and store.edge_slot(eid) is not None:
        store.set_edge_policy(eid, policy)


@rebuilds_the_store
def set_entity_to_idx(g, mapping):
    """Rebuild the entity registry from a ``vid -> row_idx`` map (legacy setter)."""
    g._entities.clear()
    g._row_to_entity.clear()
    g._vid_to_ekeys.clear()
    coord = I.placeholder_layer_coord(g)
    for vid, row_idx in dict(mapping).items():
        register_entity_record(g, (vid, coord), EntityRecord(row_idx=int(row_idx), kind='vertex'))
    g._mark_matrix_dirty()


def set_entity_types(g, mapping):
    """Set entity kinds from a ``vid -> kind`` map (legacy setter)."""
    for vid, kind in dict(mapping).items():
        ekey = I.resolve_ekey(g, vid)
        rec = g._entities.get(ekey)
        row_idx = rec.row_idx if rec is not None else len(g._entities)
        register_entity_record(
            g, ekey, EntityRecord(row_idx=row_idx, kind=_internal_entity_kind(kind))
        )


def set_entity_kinds(g, mapping):
    """Set the kind of each entity an ``ekey -> kind`` map names.

    An entity the map names but the graph does not hold is registered at the next
    free row. A reader needs that, because a file may record the kind of an
    entity that the structure it read never mentions.
    """
    for ekey, kind in mapping.items():
        rec = g._entities.get(ekey)
        if rec is not None:
            rec.kind = kind
            sync_entity(g, ekey)
            continue
        row_idx = max(g._row_to_entity, default=-1) + 1
        register_entity_record(g, ekey, EntityRecord(row_idx=row_idx, kind=kind))


def remap_entity_keys(g, remap):
    """Move each entity an ``ekey -> ekey`` map names, keeping the row it holds.

    A reader needs this when the coordinate a file stored for an entity only
    makes sense once the aspects the same file declares are known, and declaring
    aspects over a flat graph needs it for every vertex it already holds.

    An identity changes and an address does not, so this reaches the slot store
    directly rather than rebuilding it. Every member list and every position
    survives, which is what makes the whole call cost the entities alone.
    """
    if not remap:
        return
    g._entities = {remap.get(ekey, ekey): rec for ekey, rec in g._entities.items()}
    D.rebuild_entity_indexes(g)
    store = slot_store(g)
    if store is not None:
        store.rekey(remap)


def _edge_member_nodes(rec):
    """Return the incident-node set of an edge record (handles str/frozenset endpoints)."""
    out = set()
    for side in (rec.src, rec.tgt):
        if isinstance(side, frozenset):
            out |= side
        elif side is not None:
            out.add(side)
    return out


def set_edge_coeffs(g, edge_id, coeffs):
    """Overwrite an edge column's incidence coefficients (stoichiometry)."""
    rec = g._edges[edge_id]
    # Records are the complete source of truth: fold the new coefficients over the
    # edge's existing column (member endpoints default to their derived +/- weight)
    # and store the resulting nonzero column so the matrix can be rebuilt from it.
    base = {}
    if rec.coeffs is not None:
        base.update(rec.coeffs)
    else:
        w = float(rec.weight) if rec.weight is not None else 1.0
        tv = -w if rec.directed else w
        for side, val in ((rec.src, w), (rec.tgt, tv)):
            if isinstance(side, frozenset):
                for n in side:
                    base[n] = val
            elif side is not None:
                base[side] = val
    # On a multilayer graph the endpoints above are supra-node keys ``(vid, coord)``,
    # while callers key coefficients by bare vertex id (from_sbml passes raw SBML
    # species ids). The two spellings never collide, so without resolving them the
    # update *adds* a second entry for the same vertex and the rebuilt column sums
    # both — inflating every coefficient by the derived +/- weight.
    by_vid = {}
    for key in base:
        if I.is_explicit_entity_key(key):
            by_vid.setdefault(key[0], []).append(key)
    resolved = {}
    for vid, coeff in coeffs.items():
        hits = by_vid.get(vid) if isinstance(vid, str) else None
        # Only rewrite when the bare id names exactly one endpoint; a vid present at
        # several coords on one edge is ambiguous, so leave it for the caller to key.
        resolved[hits[0] if hits and len(hits) == 1 else vid] = float(coeff)
    base.update(resolved)
    rec.coeffs = {n: v for n, v in base.items() if v != 0.0}
    g._mark_matrix_dirty()
    D.invalidate_sparse_caches(g)
    store = slot_store(g)
    if store is not None and store.edge_slot(edge_id) is not None:
        store.set_edge_coefficients(edge_id, rec.coeffs)


def replace_edge_coeffs(g, edge_id, coeffs):
    """Set the coefficients of an edge column outright, dropping what it held.

    Unlike :func:`set_edge_coeffs`, this folds nothing over the column the edge
    already has. A reader that recovers a whole column from a file needs the
    column it read and nothing else.
    """
    rec = g._edges.get(edge_id)
    if rec is None:
        return
    rec.coeffs = {member: value for member, value in coeffs.items() if value != 0.0}
    D.invalidate_sparse_caches(g)
    store = slot_store(g)
    if store is not None and store.edge_slot(edge_id) is not None:
        store.set_edge_coefficients(edge_id, rec.coeffs)


def set_hyperedge_members(g, eid, *, members=None, head=None, tail=None):
    """Turn an edge the graph already holds into a hyperedge over these members.

    Pass ``head`` and ``tail`` for a directed hyperedge, or ``members`` for an
    undirected one. A reader that meets the members of an edge after the edge
    itself needs this.
    """
    rec = g._edges.get(eid)
    if rec is None:
        return
    rec.etype = 'hyper'
    if members is None:
        rec.src = frozenset(head or ())
        rec.tgt = frozenset(tail or ())
        rec.directed = True
    else:
        rec.src = frozenset(members)
        rec.tgt = None
        rec.directed = False
    g._mark_matrix_dirty()
    D.invalidate_sparse_caches(g)
    rewrite_edge(g, eid)


# The three fields the dict views write. Each is one field of one edge in the
# slot store too, so each reaches it through the narrow write for that field
# rather than through a member list the write never touched.
_EDGE_FIELD_WRITES = {
    'ml_layers': 'set_edge_ml_layers',
    'ml_kind': 'set_edge_ml_kind',
    'directed': 'set_edge_directed',
}


def set_edge_field(g, eid, field, value):
    """Set one field on an edge record (backs the legacy dict-view writers)."""
    rec = g._edges.get(eid)
    if rec is None:
        return
    setattr(rec, field, value)
    store = slot_store(g)
    if store is None or store.edge_slot(eid) is None:
        return
    write = _EDGE_FIELD_WRITES.get(field)
    if write is None:
        sync_edge(g, eid)
    else:
        getattr(store, write)(eid, value)


def set_edge_kind(g, eid, kind):
    """Set an edge's kind: ``'hyper'`` flips etype, anything else sets ml_kind."""
    rec = g._edges.get(eid)
    if rec is None:
        return
    if kind == 'hyper':
        rec.etype = 'hyper'
    else:
        rec.ml_kind = kind
    store = slot_store(g)
    if store is None or store.edge_slot(eid) is None:
        return
    if kind == 'hyper':
        store.set_edge_kind(eid, ST.HYPER)
    else:
        store.set_edge_ml_kind(eid, kind)


def reverse_directions(g):
    """Flip src<->tgt on directed edges/hyperedges in place; rebuild adjacency indexes."""
    reversed_ids = []
    for eid, rec in g._edges.items():
        if rec.col_idx < 0:
            continue
        if rec.etype == 'hyper':
            if rec.tgt is not None:
                rec.src, rec.tgt = rec.tgt, rec.src
                reversed_ids.append(eid)
            continue
        edge_is_directed = (
            rec.directed
            if rec.directed is not None
            else (True if g.directed is None else g.directed)
        )
        if edge_is_directed:
            rec.src, rec.tgt = rec.tgt, rec.src
            reversed_ids.append(eid)
    g._mark_matrix_dirty()
    D.invalidate_sparse_caches(g)
    store = slot_store(g)
    if store is not None:
        for eid in reversed_ids:
            if store.edge_slot(eid) is not None:
                store.reverse_edge(eid)


def make_undirected(g, *, drop_flexible=True, update_default=True):
    """Convert all existing edges to undirected form in place; returns the graph.

    Records are the source of truth for the (lazily rebuilt) incidence matrix, so
    this only rewrites records: binary edges flip to ``directed=False`` (both
    endpoints then rebuild to ``+w``), and hyperedges collapse head/tail into a
    single undirected member set. Explicit signed ``coeffs`` are dropped so the
    symmetric ``+w`` column is derived — matching the legacy overwrite behavior.
    """
    store = slot_store(g)

    def reaches_the_store(edge_id):
        return store is not None and store.edge_slot(edge_id) is not None

    # 1) Binary / vertex-edge edges
    for eid, rec in list(g._edges.items()):
        if rec.etype == 'hyper':
            continue
        if rec.src is None or rec.tgt is None:
            continue
        if rec.col_idx < 0:
            continue
        rec.coeffs = None
        rec.directed = False
        if reaches_the_store(eid):
            # An undirected binary edge keeps its two sides and changes only
            # what its members carry, so the direction comes first and the
            # coefficients are then derived once.
            store.set_edge_directed(eid, False)
            store.set_edge_explicit(eid, False)

    # 2) Hyperedges
    for eid, rec in list(g._edges.items()):
        if rec.etype != 'hyper':
            continue
        if rec.col_idx < 0:
            continue
        members = rec.src | rec.tgt if rec.tgt is not None else rec.src
        rec.src = frozenset(members)
        rec.tgt = None
        rec.directed = False
        rec.coeffs = None
        if reaches_the_store(eid):
            store.set_edge_directed(eid, False)
            store.set_edge_explicit(eid, False)
            store.merge_sides(eid)

    if drop_flexible:
        for eid, rec in g._edges.items():
            rec.direction_policy = None
            if reaches_the_store(eid):
                store.set_edge_policy(eid, None)
    if update_default:
        g.directed = False
    g._mark_matrix_dirty()
    D.invalidate_sparse_caches(g)
    return g


def drop_edges(g, edge_ids) -> None:
    """Remove edges from the slot store.

    The record store renumbers every column after the one it drops. The slot
    store frees a slot and moves no other edge, so this is the whole of the work
    there.
    """
    store = slot_store(g)
    if store is None:
        return
    for edge_id in edge_ids:
        if store.edge_slot(edge_id) is not None:
            store.remove_edge(edge_id)


def drop_orphan_edge_entities(g, edge_ids) -> None:
    """Remove the entity of an edge-entity whose edge has just been removed.

    An edge-entity is one identity on both axes: the entity is the edge. Once the
    edge is gone the entity names nothing, so it goes the way a removed vertex
    goes, and an edge that held it as an endpoint goes with it. That is the same
    cascade a vertex removal runs, and it is why this calls it.
    """
    orphans = [
        edge_id
        for edge_id in edge_ids
        if edge_id not in g._edges and S.has_entity_id(g, edge_id, kind=S.EDGE_ENTITY)
    ]
    if orphans:
        remove_vertices_bulk(g, orphans)


def drop_entities(g, keys) -> None:
    """Remove entities from the slot store, once every edge that named one is gone.

    The record store renumbers its rows on a delete, so the callers below rebuild
    their registry. The slot store frees a slot and moves no other entity, so this
    is the whole of the work there.
    """
    store = slot_store(g)
    if store is None:
        return
    for key in keys:
        if store.entity_slot(key) is not None:
            store.remove_entity(key)


def remove_vertices_bulk(g, vertex_ids):
    """Remove many vertices, their incident edges, and compact entity rows."""
    drop_keys = set()
    drop_vertex_ids = set()
    for vid in vertex_ids:
        try:
            ekey = I.resolve_ekey(g, vid)
        except (KeyError, ValueError, TypeError):
            continue
        if ekey not in g._entities:
            continue
        drop_keys.add(ekey)
        drop_vertex_ids.add(ekey[0] if isinstance(ekey, tuple) and len(ekey) == 2 else ekey)

    if not drop_keys:
        return

    drop_es: set = set()
    for eid, rec in list(g._edges.items()):
        if rec.etype == 'hyper':
            if drop_vertex_ids & I.endpoint_slice_vertex_ids(g, rec.src):
                drop_es.add(eid)
            elif rec.tgt is not None and (
                drop_vertex_ids & I.endpoint_slice_vertex_ids(g, rec.tgt)
            ):
                drop_es.add(eid)
        else:
            if drop_vertex_ids & I.endpoint_slice_vertex_ids(g, rec.src):
                drop_es.add(eid)
            elif drop_vertex_ids & I.endpoint_slice_vertex_ids(g, rec.tgt):
                drop_es.add(eid)

    if drop_es:
        remove_edges_bulk(g, drop_es)

    keep_idx = sorted(rec.row_idx for eid, rec in g._entities.items() if eid not in drop_keys)

    D.mark_matrix_stale(g)
    D.invalidate_sparse_caches(g)

    new_entities: dict = {}
    new_row_to_entity: dict = {}
    for new_i, old_i in enumerate(keep_idx):
        ent = g._row_to_entity[old_i]
        old_rec = g._entities[ent]
        new_entities[ent] = EntityRecord(row_idx=new_i, kind=old_rec.kind)
        new_row_to_entity[new_i] = ent
    g._entities = new_entities
    g._row_to_entity = new_row_to_entity
    D.rebuild_entity_indexes(g)
    drop_entities(g, drop_keys)

    va = g.vertex_attributes
    if va is not None and 'vertex_id' in dataframe_columns(va):
        g.vertex_attributes = dataframe_drop_rows(va, 'vertex_id', drop_vertex_ids)

    for slice_data in g._slices.values():
        slice_data['vertices'].difference_update(drop_vertex_ids)


def remove_orphan_node_layers(g, drop_keys):
    """Drop specific ``(vid, layer)`` vertex entities that carry no incident edges.

    Unlike :func:`remove_vertices_bulk` (which drops every edge touching the bare
    vertex id, and the vertex-attribute row for that id), this removes only the
    given node-layer entity rows and compacts the incidence matrix. The vertex id
    itself survives through its other node-layers, so ``vertex_attributes``,
    slice membership, and incident edges are all left untouched.

    Callers MUST guarantee every key in ``drop_keys`` is an orphan node-layer
    (no edge references it); the incidence matrix is rebuilt lazily from edge
    records keyed by node name, so nothing here inspects or rewrites edges.
    """
    drop_keys = {k for k in drop_keys if k in g._entities}
    if not drop_keys:
        return

    keep_idx = sorted(rec.row_idx for ekey, rec in g._entities.items() if ekey not in drop_keys)
    D.mark_matrix_stale(g)
    D.invalidate_sparse_caches(g)

    new_entities: dict = {}
    new_row_to_entity: dict = {}
    for new_i, old_i in enumerate(keep_idx):
        ent = g._row_to_entity[old_i]
        old_rec = g._entities[ent]
        new_entities[ent] = EntityRecord(row_idx=new_i, kind=old_rec.kind)
        new_row_to_entity[new_i] = ent
    g._entities = new_entities
    g._row_to_entity = new_row_to_entity
    D.rebuild_entity_indexes(g)
    drop_entities(g, drop_keys)

    state_attrs = getattr(g, '_state_attrs', None)
    if state_attrs:
        for k in drop_keys:
            state_attrs.pop(k, None)


# -------------------------------------------------------------------------
# Relocated module-level helpers + vectorized batch write paths (the bulk
# gateway: the only place batch structural writes happen).
# -------------------------------------------------------------------------


def _sanitize(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


_BINARY_BATCH_RESERVED_KEYS = frozenset(
    {
        'source',
        'target',
        'src',
        'tgt',
        'edge_id',
        'slice',
        'weight',
        'edge_directed',
        'directed',
        'edge_type',
        'propagate',
        'flexible',
        'attributes',
        'attrs',
        'slice_weight',
    }
)

_HYPER_BATCH_RESERVED_KEYS = frozenset(
    {
        'members',
        'head',
        'tail',
        'edge_id',
        'slice',
        'weight',
        'edge_directed',
        'directed',
        'attributes',
        'attrs',
        'layer',
        '_resolved_members',
        '_resolved_head',
        '_resolved_tail',
    }
)


def batch_add_vertices(g, vertices, layer=None, slice=None, default_attrs=None):
    """Add many vertices through the bulk mutation path."""
    slice = slice or g._current_slice
    default_attrs = default_attrs or {}

    # --- normalize input ---
    norm = []
    for it in vertices:
        if isinstance(it, dict):
            if it.get('vertex_id'):
                vid = it['vertex_id']
                _id_keys = {'vertex_id'}
            elif it.get('id'):
                vid = it['id']
                _id_keys = {'vertex_id', 'id'}
            elif it.get('name'):
                vid = it['name']
                _id_keys = {'vertex_id', 'id', 'name'}
            else:
                vid = None
            if vid is None:
                continue
            attrs = {k: v for k, v in it.items() if k not in _id_keys}
        elif isinstance(it, (tuple, list)) and it:
            vid = it[0]
            attrs = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
        else:
            vid = it
            attrs = {}
        if default_attrs:
            merged = dict(default_attrs)
            merged.update(attrs)
            attrs = merged
        norm.append((vid, attrs))

    if not norm:
        return

    try:
        norm = [(_intern(vid) if isinstance(vid, str) else vid, attrs) for vid, attrs in norm]
        if isinstance(slice, str):
            slice = _intern(slice)
    except TypeError:
        pass

    # --- entity registration ---
    # New vertices are registered inline (record + row-index map + vid index)
    # instead of via the general setter, which re-checks for an existing record
    # on every call — unnecessary here since we only register unseen keys.
    coord = g._resolve_vertex_insert_coord(
        layer, vertex_ids=[vid for vid, _ in norm], context='_add_vertices_batch'
    )
    _entities = g._entities
    _row_to_entity = g._row_to_entity
    store = slot_store(g)
    new_rows = 0
    for vid, _ in norm:
        ekey = (vid, coord)
        if ekey not in _entities:
            idx = len(_entities)
            _entities[ekey] = EntityRecord(row_idx=idx, kind='vertex')
            _row_to_entity[idx] = ekey
            D.index_entity_key(g, ekey)
            if store is not None:
                store.add_entity(ekey, ST.NODE)
            new_rows += 1
    if new_rows:
        D.mark_matrix_stale(g)

    # --- slice ---
    g.slices._ensure_slice(slice)['vertices'].update(vid for vid, _ in norm)

    # --- attribute table (Polars fast path) ---
    g._ensure_vertex_table()
    if is_polars_dataframe(g.vertex_attributes):
        keys = {k for _, attrs in norm for k in attrs}
        df = g.vertex_attributes
        if keys:
            df = g._ensure_attr_columns(df, dict.fromkeys(keys))
        if is_polars_dataframe(df):
            result = polars_upsert_vertices(df, norm)
            if result is not None:
                g.vertex_attributes = result
                return

    # --- generic fallback (pandas / pyarrow, or polars with non-string vids) ---
    # Non-string vids (e.g. multilayer supra-node tuples) are tracked in
    # ``_entities`` only and stay out of the obs table. The polars backend
    # cannot represent tuples in a String-typed vertex_id column anyway.
    df2 = g.vertex_attributes

    rows = [
        {'vertex_id': vid, **{k: _sanitize(v) for k, v in attrs.items()}}
        for vid, attrs in norm
        if isinstance(vid, str)
    ]
    if rows:
        df2 = dataframe_upsert_rows(df2, rows, ('vertex_id',))

    g.vertex_attributes = df2


def batch_add_edges(
    g,
    edges,
    *,
    slice=None,
    as_entity=False,
    default_weight=1.0,
    default_edge_type='regular',
    default_propagate='none',
    default_slice_weight=None,
    default_edge_directed=None,
):
    """Add many binary edges through the bulk mutation path."""
    slice = g._current_slice if slice is None else slice
    pending_attrs = {}

    # Single-pass bulk builder. Each input item is normalized, its endpoint
    # entities ensured, and its EdgeRecord written in one iteration — no
    # intermediate ``norm`` dict copy and no separate scans. Row/col indices and
    # the incidence matrix are derived lazily from the records, so the loop only
    # needs the ordering guarantee that entities are registered before/while the
    # record referencing them is created (satisfied here inline).
    _entities = g._entities
    _edges = g._edges
    _col_to_edge = g._col_to_edge
    _row_to_entity = g._row_to_entity
    # An endpoint the graph has not met yet becomes a node here, inline rather
    # than through the setter, so the slot store is written inline as well. The
    # edge itself is written inline too: the loop already holds both endpoint
    # keys and the coefficient each of them takes, so reading the record back to
    # work them out again is the whole of what a bulk load used to pay twice for.
    _slot = slot_store(g)
    # The shapes the inline write does not cover. Each is rare, and each is
    # written from its record once the loop is done.
    _deferred: list = []
    _flat = g._aspects == ('_',)
    _flat_coord = ('_',)
    _is_multilayer = not _flat
    _g_directed = g.directed
    _RESERVED = _BINARY_BATCH_RESERVED_KEYS

    _col = len(_col_to_edge)
    _col0 = _col
    _next_id = g._next_edge_id

    out_ids: list = []
    entity_out: list = []
    _slice_eids: dict = {}
    _slice_vids: dict = {}
    _slice_weights: list = []

    def _ensure_endpoint(vid, et):
        """Register the entity one endpoint names, and return its key."""
        if isinstance(vid, tuple) and len(vid) == 2 and isinstance(vid[1], tuple):
            ekey = vid
        elif _flat:
            ekey = (vid, _flat_coord)
        else:
            ekey = (
                vid,
                g._resolve_vertex_insert_coord(None, vertex_ids=vid, context='_add_edges_batch'),
            )
        if ekey not in _entities:
            if (
                (et == 'vertex_edge' or et == 'edge_placeholder')
                and isinstance(vid, str)
                and vid.startswith('edge_')
            ):
                g._ensure_edge_entity_placeholder(vid)
                # The placeholder registers the edge entity under the key its own
                # resolution gives, which need not be the one worked out above.
                return I.resolve_ekey(g, vid)
            idx = len(_entities)
            _entities[ekey] = EntityRecord(row_idx=idx, kind='vertex')
            _row_to_entity[idx] = ekey
            D.index_entity_key(g, ekey)
            if _slot is not None:
                _slot.add_entity(ekey, ST.NODE)
        return ekey

    for idx, it in enumerate(edges):
        # ── extract endpoints + fields without copying the input dict ──────────
        if isinstance(it, dict):
            if 'source' in it:
                s = it['source']
                has_src = True
            elif 'src' in it:
                s = it['src']
                has_src = True
            else:
                s = None
                has_src = False
            if 'target' in it:
                t = it['target']
                has_tgt = True
            elif 'tgt' in it:
                t = it['tgt']
                has_tgt = True
            else:
                t = None
                has_tgt = False

            if has_src ^ has_tgt:
                missing = 'target' if has_src else 'source'
                raise ValueError(
                    f'add_edges batch item at index {idx} is missing '
                    f"'{missing}' (or its alias '{'tgt' if missing == 'target' else 'src'}'): "
                    f'{it!r}'
                )

            if not has_src:
                # Null-endpoint edge-entity placeholder (requires as_entity).
                if not as_entity:
                    raise ValueError(
                        'Batch items without source/target require as_entity=True to be '
                        'treated as edge-entity placeholders.'
                    )
                e_id = it.get('edge_id')
                if not e_id:
                    e_id = f'edge_{_next_id}'
                    _next_id += 1
                elif isinstance(e_id, str):
                    e_id = _intern(e_id)
                sl = it.get('slice', slice)
                if type(sl) is str:
                    sl = _intern(sl)
                extra = {k: v for k, v in it.items() if k not in _RESERVED}
                g._ensure_edge_entity_placeholder(e_id, slice=sl, **extra)
                entity_out.append(e_id)
                continue

            w = it.get('weight', default_weight)
            edge_type = it.get('edge_type', default_edge_type)
            prop = it.get('propagate', default_propagate)
            slice_local = it.get('slice', slice)
            slice_w = it.get('slice_weight', default_slice_weight)
            if 'edge_directed' in it:
                e_dir = it['edge_directed']
            elif 'directed' in it:
                e_dir = it['directed']
            else:
                e_dir = default_edge_directed
            edge_id = it.get('edge_id')
            _item = it
        elif isinstance(it, (tuple, list)):
            s = it[0]
            t = it[1]
            w = it[2] if len(it) > 2 else default_weight
            edge_type = default_edge_type
            prop = default_propagate
            slice_local = slice
            slice_w = default_slice_weight
            e_dir = default_edge_directed
            edge_id = None
            _item = None
        else:
            continue

        # ── intern id-like strings + coerce weight ─────────────────────────────
        if type(s) is str:
            s = _intern(s)
        if type(t) is str:
            t = _intern(t)
        if type(slice_local) is str:
            slice_local = _intern(slice_local)
        try:
            w = float(w)
        except (TypeError, ValueError):
            pass

        # ── ensure endpoint entities exist ─────────────────────────────────────
        source_key = _ensure_endpoint(s, edge_type)
        target_key = _ensure_endpoint(t, edge_type)

        # ── direction ──────────────────────────────────────────────────────────
        if e_dir is not None:
            is_dir = bool(e_dir)
        elif _g_directed is not None:
            is_dir = _g_directed
        else:
            is_dir = True

        # ── multilayer role (flat graphs are always intra) ─────────────────────
        ml_kind = None
        ml_layers = None
        if not _is_multilayer:
            ml_kind = 'intra'
        elif (
            isinstance(s, tuple)
            and len(s) == 2
            and isinstance(s[1], tuple)
            and isinstance(t, tuple)
            and len(t) == 2
            and isinstance(t[1], tuple)
        ):
            ml_kind = g._infer_ml_kind(s, t)
            ml_layers = (s[1], t[1])

        # ── edge id (auto-assign in input order) ───────────────────────────────
        if edge_id is None:
            edge_id = f'edge_{_next_id}'
            _next_id += 1
        elif type(edge_id) is str:
            edge_id = _intern(edge_id)

        # ── record + column (adjacency indexes rebuild lazily from records) ────
        existing = _edges.get(edge_id)
        if _slot is not None:
            # A new edge with two endpoints is two member entries, and the loop
            # already holds both. Anything else — an open side, or an edge the
            # graph already holds, which keeps a kind, a policy or coefficients
            # this write does not name — is written from its finished record.
            if existing is not None or s is None or t is None:
                _deferred.append(edge_id)
            else:
                weight = float(w) if w is not None else 1.0
                _slot.add_edge(
                    edge_id,
                    (
                        (source_key, weight, ST.SOURCE),
                        (target_key, -weight if is_dir else weight, ST.TARGET),
                    ),
                    kind=ST.BINARY,
                    directed=is_dir,
                    weight=w,
                    ml_kind=ml_kind,
                    ml_layers=ml_layers,
                )
        if existing is not None and existing.col_idx >= 0:
            rec = existing
            rec.src = s
            rec.tgt = t
            rec.weight = w
            rec.directed = is_dir
            rec.ml_kind = ml_kind
            rec.ml_layers = ml_layers
        else:
            col = _col
            _col += 1
            _col_to_edge[col] = edge_id
            if existing is not None:
                rec = existing
                rec.src = s
                rec.tgt = t
                rec.weight = w
                rec.directed = is_dir
                rec.etype = 'binary'
                rec.col_idx = col
                rec.ml_kind = ml_kind
                rec.ml_layers = ml_layers
            else:
                _edges[edge_id] = EdgeRecord(
                    src=s,
                    tgt=t,
                    weight=w,
                    directed=is_dir,
                    etype='binary',
                    col_idx=col,
                    ml_kind=ml_kind,
                    ml_layers=ml_layers,
                    direction_policy=None,
                )

        # ── slice membership (tracks bare vertex ids) ──────────────────────────
        if slice_local is not None:
            s_bare = s[0] if isinstance(s, tuple) else s
            t_bare = t[0] if isinstance(t, tuple) else t
            _lst = _slice_eids.get(slice_local)
            if _lst is None:
                _slice_eids[slice_local] = [edge_id]
                _slice_vids[slice_local] = [s_bare, t_bare]
            else:
                _lst.append(edge_id)
                _slice_vids[slice_local].extend((s_bare, t_bare))
            if slice_w is not None:
                _slice_weights.append((slice_local, edge_id, float(slice_w)))

        if prop == 'shared':
            g._propagate_to_shared_slices(edge_id, s, t)
        elif prop == 'all':
            g._propagate_to_all_slices(edge_id, s, t)

        # ── attributes (only for dict inputs) ──────────────────────────────────
        # Cheap subset test avoids allocating an (almost always empty) attribute
        # dict for every plain {source, target, weight} edge.
        if _item is not None:
            sub_attrs = _item.get('attributes') or _item.get('attrs')
            has_flat = not (_item.keys() <= _RESERVED)
            if sub_attrs or has_flat:
                merged_attrs = dict(sub_attrs) if sub_attrs else {}
                if has_flat:
                    for k, v in _item.items():
                        if k not in _RESERVED:
                            merged_attrs[k] = v
                pending_attrs.setdefault(edge_id, {}).update(merged_attrs)

        out_ids.append(edge_id)

    if not out_ids and not entity_out:
        return []

    g._next_edge_id = _next_id
    D.mark_matrix_stale(g)
    if _col > _col0:
        D.mark_matrix_stale(g)

    # Incidence is a lazy cache derived from the edge records mutated above, so
    # the batch never patches matrix cells — it just marks the cache dirty and
    # lets the next reader rebuild a compact CSR from records in one pass.
    if out_ids:
        g._mark_matrix_dirty()
        g._invalidate_sparse_caches()

    for sid, eids in _slice_eids.items():
        g.slices._ensure_slice(sid)['edges'].update(eids)
    for sid, vids in _slice_vids.items():
        g._slices[sid]['vertices'].update(vids)
    for sid, eid, sw in _slice_weights:
        g.attrs.set_edge_slice_attrs(sid, eid, weight=sw)

    if pending_attrs:
        g.attrs.set_edge_attrs_bulk(pending_attrs)

    if as_entity:
        entities = g._entities
        flat = g._aspects == ('_',)
        flat_coord = ('_',)
        for eid in out_ids:
            ekey = (
                (eid, flat_coord) if flat and isinstance(eid, str) else g._resolve_entity_key(eid)
            )
            if ekey not in entities:
                g._register_entity_record(
                    ekey,
                    EntityRecord(row_idx=len(entities), kind='edge_entity'),
                )
            rec = g._edges[eid]
            if rec.etype == 'binary':
                rec.etype = 'vertex_edge'
                if _slot is not None and _slot.edge_slot(eid) is not None:
                    _slot.set_edge_kind(eid, ST.NODE_EDGE)
        D.mark_matrix_stale(g)

    sync_edges(g, _deferred)

    g._ensure_edge_rows_bulk(entity_out + out_ids)

    return entity_out + out_ids


def batch_add_hyperedges(
    g,
    hyperedges,
    *,
    slice=None,
    default_weight=1.0,
    default_edge_directed=None,
    layer=None,
):
    """Add many hyperedges through the bulk mutation path."""
    slice = g._current_slice if slice is None else slice
    _slot = slot_store(g)

    items = []
    for it in hyperedges:
        if not isinstance(it, dict):
            continue
        d = dict(it)
        if 'directed' in d and 'edge_directed' not in d:
            d['edge_directed'] = d.pop('directed')

        # ── Normalize user-facing src/tgt to internal members/head/tail ──
        # annnet stores rec.src = head (the +w side in the incidence
        # matrix) and rec.tgt = tail (the -w side). To keep the batch
        # path consistent with the single-edge path — where the user's
        # ``src`` ends up in rec.src and gets +w — map user.src → head
        # and user.tgt → tail here.
        if 'src' in d and 'source' not in d:
            d['source'] = d.pop('src')
        if 'tgt' in d and 'target' not in d:
            d['target'] = d.pop('tgt')
        has_legacy = any(k in d for k in ('members', 'head', 'tail'))
        has_new = 'source' in d or 'target' in d
        if has_new and not has_legacy:
            src_val = d.pop('source', None)
            tgt_val = d.pop('target', None)

            def _as_list(v):
                if v is None:
                    return None
                if isinstance(v, str):
                    return [v]
                return list(v)

            src_list = _as_list(src_val)
            tgt_list = _as_list(tgt_val)
            if tgt_list is None:
                d['members'] = src_list or []
            else:
                d['head'] = src_list or []
                d['tail'] = tgt_list

        d.setdefault('weight', default_weight)
        if 'slice' not in d:
            d['slice'] = slice
        if 'edge_directed' not in d:
            d['edge_directed'] = default_edge_directed
        items.append(d)

    if not items:
        return []

    try:
        import sys as _sys

        for d in items:
            if 'members' in d and d['members'] is not None:
                d['members'] = [_sys.intern(x) if isinstance(x, str) else x for x in d['members']]
            else:
                d['head'] = [_sys.intern(x) if isinstance(x, str) else x for x in d.get('head', [])]
                d['tail'] = [_sys.intern(x) if isinstance(x, str) else x for x in d.get('tail', [])]
            if isinstance(d.get('slice'), str):
                d['slice'] = _sys.intern(d['slice'])
            if isinstance(d.get('edge_id'), str):
                d['edge_id'] = _sys.intern(d['edge_id'])
            try:
                d['weight'] = float(d['weight'])
            except (TypeError, ValueError):
                pass
    except Exception:  # noqa: BLE001
        pass

    # Per-hyperedge layer override; falls back to batch-level `layer`.
    # A member endpoint may also be a (vid, layer_coord) tuple, in which
    # case it carries its own coord and the layer parameter is ignored.
    def _member_layer(d):
        return d.get('layer', layer)

    def _resolve_member_key(u, layer_for_d):
        if isinstance(u, tuple) and len(u) == 2 and isinstance(u[1], tuple):
            # Pre-keyed (vid, layer_coord) endpoint.
            vid = u[0]
            coord = g._make_layer_coord(u[1])
            return vid, coord
        # Bare string vid: place into layer_for_d if given, else fall
        # through to the existing placeholder/single-match resolution.
        if layer_for_d is not None and g._aspects != ('_',):
            coord = g._make_layer_coord(layer_for_d)
            return u, coord
        # No layer hint. If the vid already lives in exactly one real
        # layer (i.e. it was previously inserted), reuse that coord
        # instead of forking a placeholder copy — this is what previously
        # produced the "Ambiguous bare vertex_id" failures.
        if g._aspects != ('_',):
            ekeys = g._vid_to_ekeys.get(u, ())
            real_keys = [k for k in ekeys if k[1] != g._placeholder_layer_coord()]
            if len(real_keys) == 1:
                return u, real_keys[0][1]
        coord = g._resolve_vertex_insert_coord(None, vertex_ids=u, context='_add_hyperedges_batch')
        return u, coord

    # Resolve every member endpoint up-front and stash the resolved keys
    # back onto each item so the matrix-write loop below doesn't need to
    # re-resolve.
    for d in items:
        layer_for_d = _member_layer(d)
        if 'members' in d and d['members'] is not None:
            d['_resolved_members'] = [_resolve_member_key(u, layer_for_d) for u in d['members']]
        else:
            d['_resolved_head'] = [_resolve_member_key(u, layer_for_d) for u in d.get('head', [])]
            d['_resolved_tail'] = [_resolve_member_key(u, layer_for_d) for u in d.get('tail', [])]

    for d in items:
        for ekey in (
            (d.get('_resolved_members') or [])
            + (d.get('_resolved_head') or [])
            + (d.get('_resolved_tail') or [])
        ):
            if ekey not in g._entities:
                idx = len(g._entities)
                g._register_entity_record(ekey, EntityRecord(row_idx=idx, kind='vertex'))

    D.mark_matrix_stale(g)

    new_count = sum(1 for d in items if d.get('edge_id') not in g._edges)
    if new_count:
        D.mark_matrix_stale(g)

    slices = g._slices

    out_ids = []
    attrs_batch = {}

    for d in items:
        members = d.get('members')
        slice_local = d.get('slice', slice)
        w = float(d.get('weight', default_weight))
        e_id = d.get('edge_id')
        directed = d.get('edge_directed')
        if directed is None:
            directed = members is None

        if e_id is None:
            e_id = g._get_next_edge_id()

        # Classify the hyperedge by its multilayer role from the layers of
        # its (already resolved) endpoints. A flat graph is a single layer,
        # so the role is always intra (no scan needed).
        if g._aspects == ('_',):
            ml_kind_for_e, ml_layers_for_e = 'intra', None
        else:
            ml_kind_for_e, ml_layers_for_e = g._infer_hyper_ml(
                d.get('_resolved_head') or d.get('_resolved_members'),
                d.get('_resolved_tail'),
            )

        if e_id in g._edges:
            rec = g._edges[e_id]
            col = rec.col_idx
            # In-place update: the record is overwritten below and the incidence
            # column is rebuilt lazily from it, so no explicit old-column zeroing.
        else:
            col = len(g._col_to_edge)
            g._col_to_edge[col] = e_id
            rec = EdgeRecord(
                src=None,
                tgt=None,
                weight=1.0,
                directed=False,
                etype='hyper',
                col_idx=col,
                ml_kind=ml_kind_for_e,
                ml_layers=ml_layers_for_e,
                direction_policy=None,
            )
            g._edges[e_id] = rec

        resolved_members = d.get('_resolved_members')
        resolved_head = d.get('_resolved_head')
        resolved_tail = d.get('_resolved_tail')

        # Incidence is rebuilt from the record (coeffs=None -> members +w,
        # head +w / tail -w), so only the endpoint sets are recorded here.
        if members is not None:
            if g._aspects == ('_',):
                rec.src = frozenset(ekey[0] for ekey in resolved_members)
            else:
                rec.src = frozenset(resolved_members)
            rec.tgt = None
            rec.directed = False
        else:
            if g._aspects == ('_',):
                rec.src = frozenset(ekey[0] for ekey in resolved_head)
                rec.tgt = frozenset(ekey[0] for ekey in resolved_tail)
            else:
                rec.src = frozenset(resolved_head)
                rec.tgt = frozenset(resolved_tail)
            rec.directed = True

        rec.weight = w
        rec.etype = 'hyper'
        rec.coeffs = None
        # Refresh on both create and in-place update paths.
        rec.ml_kind = ml_kind_for_e
        rec.ml_layers = ml_layers_for_e

        # The slot store. Every member is already resolved to an entity key, so
        # the member list is the resolved sides and the coefficient each side
        # takes. An entity named twice on one side is one member of it, exactly
        # as the record set is.
        if _slot is not None:
            if resolved_members is not None:
                source_keys, target_keys = dict.fromkeys(resolved_members), ()
            else:
                source_keys = dict.fromkeys(resolved_head)
                target_keys = dict.fromkeys(resolved_tail)
            role = ST.SOURCE if target_keys else ST.MEMBER
            member_entries = [(key, w, role) for key in source_keys]
            target_coefficient = -w if rec.directed else w
            member_entries += [(key, target_coefficient, ST.TARGET) for key in target_keys]
            if _slot.edge_slot(e_id) is not None:
                _slot.remove_edge(e_id)
            _slot.add_edge(
                e_id,
                member_entries,
                kind=ST.HYPER,
                directed=rec.directed,
                weight=w,
                ml_kind=ml_kind_for_e,
                ml_layers=ml_layers_for_e,
                direction_policy=rec.direction_policy,
            )

        if slice_local is not None:
            if slice_local not in slices:
                slices[slice_local] = SliceRecord()
            slices[slice_local]['edges'].add(e_id)
            # Slice membership tracks bare vids, not (vid, layer) keys.
            if resolved_members is not None:
                slices[slice_local]['vertices'].update(k[0] for k in resolved_members)
            else:
                slices[slice_local]['vertices'].update(k[0] for k in resolved_head)
                slices[slice_local]['vertices'].update(k[0] for k in resolved_tail)

        sub_attrs = d.get('attributes') or d.get('attrs') or {}
        flat_attrs = {k: v for k, v in d.items() if k not in _HYPER_BATCH_RESERVED_KEYS}
        if sub_attrs or flat_attrs:
            merged = dict(sub_attrs)
            merged.update(flat_attrs)
            attrs_batch[e_id] = merged

        out_ids.append(e_id)

    g._mark_matrix_dirty()
    g._invalidate_sparse_caches()
    g._ensure_edge_rows_bulk(out_ids)
    if attrs_batch:
        g.attrs.set_edge_attrs_bulk(attrs_batch)

    return out_ids


# ---------------------------------------------------------------------------
# Routing the gateway to the slot store
# ---------------------------------------------------------------------------
# Every write below changes canonical structure, so a graph that carries a slot
# store has to see the change too. The entity writes route themselves, because
# ``register_entity_record`` and ``remove_entity_record`` are the only two doors.
# The edge writes do not share a door, and several of them replace the entity
# registry outright, so each is followed by a rebuild of the store from the
# records.
#
# A rebuild is the plain answer, not the fast one. It is what makes the new core
# reachable through the public API today. The hot paths are narrowed to the edges
# they touch below, and the whole mechanism goes when the records do.
