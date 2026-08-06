"""The mutation gateway — the only writer of canonical state."""

from __future__ import annotations

from sys import intern as _intern
import json

from . import _store as ST, _derive as D, _identity as I, _structure as S
from ._records import (
    SliceRecord,
    _internal_entity_kind,
)

# ---------------------------------------------------------------------------
# Writing the canonical store
# ---------------------------------------------------------------------------
# The store is the only store, and every write here reaches it from what the
# call already holds. Nothing derives one description of a change from another:
# the two sides of an edge and the coefficient each endpoint takes are worked
# out once, by the write that knows them.
#
# The functions here read the store directly rather than through the query
# facade, because the facade answers about a finished graph and these are what
# finish it. The gateway owns the store, so it is the one place that may.

_SLOT_ENTITY_KIND = {'node': ST.NODE, 'edge_entity': ST.EDGE_ENTITY}
_SLOT_EDGE_KIND = {
    'binary': ST.BINARY,
    'hyper': ST.HYPER,
    'node_edge': ST.NODE_EDGE,
    'edge_placeholder': ST.PLACEHOLDER,
}


def slot_store(g):
    """Return the canonical store of a graph."""
    return g._store


def sync_aspects(g) -> None:
    """Make the store agree with the graph about the declared aspects.

    The store answers in the identity form its aspects imply: a bare id when it
    holds one layer, and an ``(id, layer_coord)`` pair otherwise. So a graph that
    declares aspects has to tell the store, or the store keeps answering in the
    flat form for a graph that is no longer flat.
    """
    g._store.aspects = tuple(g._aspects)


# ---------------------------------------------------------------------------
# The two doors onto the entity registry
# ---------------------------------------------------------------------------


def register_entity(g, ekey, kind: str = 'node') -> None:
    """Give the graph an entity, or change the kind of one it already holds."""
    store = g._store
    slot = store.entity_slot(ekey)
    if slot is None:
        store.add_entity(ekey, _SLOT_ENTITY_KIND.get(kind, ST.NODE))
    else:
        store.entity_kind[slot] = _SLOT_ENTITY_KIND.get(kind, ST.NODE)


def remove_entity(g, ekey) -> None:
    """Take one entity out of the graph.

    The store frees the slot and moves no other entity, so a row after it keeps
    the address it had. Every edge that named it has to be gone already.
    """
    if g._store.entity_slot(ekey) is not None:
        g._store.remove_entity(ekey)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def add_node(g, node_id, slice=None, layer=None, **attributes):
    """Add or update a node; returns its id."""
    if slice is None:
        slice = g._current_slice

    coord = I.resolve_node_insert_coord(g, layer, node_ids=node_id, context='add_node')
    key = (node_id, coord)

    if g._store.entity_slot(key) is None:
        register_entity(g, key)
        D.bump_structure(g)

    if slice not in g._slices:
        g._slices[slice] = SliceRecord()
    g._slices[slice]['nodes'].add(node_id)

    if attributes:
        g._attr_store.set_node_attrs(node_id, attributes)

    return node_id


# ---------------------------------------------------------------------------
# Edge-entities
# ---------------------------------------------------------------------------


def register_edge_as_entity(g, edge_id):
    """Ensure an edge id names an edge-entity as well as an edge."""
    ekey = I.resolve_ekey(g, edge_id)
    if g._store.entity_slot(ekey) is not None:
        return
    register_entity(g, ekey, 'edge_entity')
    D.bump_structure(g)


def register_entity_as_edge(g, entity_id) -> None:
    """Ensure an entity marked as an edge-entity has an edge to be.

    The mirror of :func:`register_edge_as_entity`, and the other half of
    data-model rule 6. An edge-entity is one identity on both axes, so marking
    an entity as one says an edge of that name exists. Until the definition of
    that edge arrives it is a placeholder, which a later add of the same id
    replaces.
    """
    if S.has_edge(g, entity_id):
        return
    g._store.add_edge(entity_id, (), kind=ST.PLACEHOLDER, directed=False, weight=1.0)


def ensure_edge_entity_placeholder(g, edge_id, slice=None, **attributes):
    """Ensure a placeholder edge-entity exists and is attached to a slice."""
    register_edge_as_entity(g, edge_id)
    # A placeholder is an edge the graph knows the name of and nothing else, so
    # it holds no members and occupies no column.
    register_entity_as_edge(g, edge_id)
    slice = slice or g._current_slice
    if slice is not None:
        g.slices._ensure_slice(slice)['edges'].add(edge_id)
    if attributes:
        g.attrs.set_edge_attrs(edge_id, **attributes)
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


# ---------------------------------------------------------------------------
# The canonical edge constructor
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
                if g._store.entity_slot(ekey) is None:
                    bare.append(node)
            return promoted, bare

        src_nodes, bare_src = _promote(src_nodes)
        tgt_nodes, bare_tgt = _promote(tgt_nodes)
        bare_total = bare_src + bare_tgt
        if bare_total:
            I.ensure_placeholder_layers_declared(g)
            I.warn_placeholder_node_assignment(g, bare_total, context='add_edges')

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
    if explicit_id and S.has_edge(g, edge_id):
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
    store = g._store
    for node in endpoint_set:
        ekey = I.resolve_ekey(g, node)
        if store.entity_slot(ekey) is None:
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                add_node(g, node[0], layer=node[1], slice=slice)
            else:
                add_node(g, node, slice=slice)
    # 6. The incidence matrix is derived from the store and keyed to its clock,
    #    so nothing here holds a column and nothing here has to invalidate one.
    g._mark_structure_changed()

    # 7. Compute src_store / tgt_store
    if etype == 'binary':
        src_store = next(iter(src_nodes))
        tgt_store = next(iter(tgt_nodes)) if tgt_nodes else None
        edge_kind = 'node_edge' if as_entity else 'binary'
    else:
        src_store = frozenset(src_nodes) if src_nodes else None
        tgt_store = frozenset(tgt_nodes) if tgt_nodes else None
        edge_kind = 'hyper'

    # 8. Infer ml_kind / ml_layers
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

    # 9. Write the edge. The two sides and the coefficient each endpoint takes
    #    are what this call has just worked out, so the store takes them from
    #    here. Literal coefficients (stoich) are the column outright; a plain
    #    +/- weight column is derived from the weight and the directedness.
    #    An edge the graph already holds keeps the policy it carries, because
    #    nothing about it is stated here.
    coefficients = dict(col_entries) if col_entries_literal is not None else None
    old_slot = store.edge_slot(edge_id)
    policy = flexible if flexible is not None else store.edge_policy.get(old_slot)
    if old_slot is not None:
        store.remove_edge(edge_id)
    store.add_edge(
        edge_id,
        ST.members_from_endpoints(
            store, g, src_nodes, tgt_nodes, coefficients, float(weight), is_dir
        ),
        kind=_SLOT_EDGE_KIND.get(edge_kind, ST.BINARY),
        directed=is_dir,
        weight=float(weight),
        explicit_coefficients=coefficients is not None,
        ml_kind=ml_kind,
        ml_layers=ml_layers,
        direction_policy=policy,
    )

    # 10. as_entity
    if as_entity:
        register_edge_as_entity(g, edge_id)

    # 11. Slice (tracks bare vids)
    if slice is not None:
        slices = g._slices
        if slice not in slices:
            slices[slice] = SliceRecord()
        slices[slice]['edges'].add(edge_id)
        for n in endpoint_set:
            slices[slice]['nodes'].add(n[0] if isinstance(n, tuple) else n)

    # 12. Propagate
    if propagate == 'shared':
        propagate_to_shared_slices(g, edge_id, src_store, tgt_store)
    elif propagate == 'all':
        propagate_to_all_slices(g, edge_id, src_store, tgt_store)

    # 13. Flexible direction. The policy reads the edge back, so the store has to
    # hold the change before the hook runs.
    if flexible is not None:
        store.set_edge_directed(edge_id, True)
        g._apply_flexible_direction(edge_id)

    # 14. Attributes
    if attrs:
        g.attrs.set_edge_attrs(edge_id, **attrs)

    return edge_id


# ---------------------------------------------------------------------------
# Slice propagation
# ---------------------------------------------------------------------------


def propagate_to_shared_slices(g, edge_id, source, target):
    """Add an edge to slices that already contain both endpoints."""
    for slice_data in g._slices.values():
        sv = slice_data['nodes']
        if I.slice_contains_endpoint(g, sv, source) and I.slice_contains_endpoint(g, sv, target):
            slice_data['edges'].add(edge_id)


def propagate_to_all_slices(g, edge_id, source, target):
    """Propagate an edge to slices containing either endpoint, adding the other endpoint as needed."""
    for slice_data in g._slices.values():
        sv = slice_data['nodes']
        source_present = I.slice_contains_endpoint(g, sv, source)
        target_present = I.slice_contains_endpoint(g, sv, target)
        if source_present or target_present:
            slice_data['edges'].add(edge_id)
            if source_present:
                I.add_endpoint_to_slice_nodes(g, sv, target)
            if target_present:
                I.add_endpoint_to_slice_nodes(g, sv, source)


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


def remove_edge(g, edge_id):
    """Remove a single edge, its column, attributes, and slice memberships.

    One edge is a set of one, and the two removes agree about every step of it.
    """
    if not S.has_edge(g, edge_id):
        raise KeyError(f'Edge {edge_id} not found')

    remove_edges_bulk(g, (edge_id,))


def remove_edges_bulk(g, edge_ids):
    """Remove many edges and compact the remaining edge columns."""
    drop = set(edge_ids)
    if not drop:
        return

    D.bump_structure(g)
    D.invalidate_sparse_caches(g)

    # The column of an edge is its position among the structural ones, so
    # dropping one moves every column after it and nothing has to be renumbered.
    drop_edges(g, drop)

    for slice_data in g._slices.values():
        slice_data['edges'].difference_update(drop)
    for d in g.slice_edge_weights.values():
        for eid in drop:
            d.pop(eid, None)

    # The generic attributes of a removed edge go with the slot the store frees.
    # The edge-by-slice table is a frame still, and takes the removal buffered:
    # a filter costs the call rather than the rows, so removing edges one at a
    # time paid for one filter each where the set they name needs one between
    # them.
    g._drop_edge_slice_rows(drop)

    drop_orphan_edge_entities(g, drop)


# ---------------------------------------------------------------------------
# Legacy structural setters (routed through the gateway, not poked directly)
# ---------------------------------------------------------------------------


def _rewrite_sides(g, eid, source, target, *, directed=None, kind=None) -> None:
    """Give one edge the two sides it now names, keeping everything else.

    The edge keeps its slot, its weight, its multilayer role and its policy. A
    member list is written whole, because a change of definition can change how
    many entries it holds.
    """
    store = g._store
    slot = store.edge_slot(eid)
    if slot is None:
        return
    if kind is not None:
        store.set_edge_kind(eid, kind)
    if directed is not None:
        store.set_edge_directed(eid, directed)
    reference = S.edge_ref(g, eid)
    store.replace_members(
        eid,
        ST.members_from_endpoints(
            store,
            g,
            S._raw_side(source),
            S._raw_side(target),
            None,
            reference.weight,
            reference.directed,
        ),
    )
    g._mark_structure_changed()


def set_edge_definition(g, eid, src, tgt, etype):
    """Rewrite a binary edge's endpoints/type (legacy edge_definitions setter)."""
    _rewrite_sides(
        g,
        eid,
        src,
        tgt,
        kind=_SLOT_EDGE_KIND.get('binary' if etype == 'hyper' else etype, ST.BINARY),
    )


def set_hyperedge_definition(g, eid, defn):
    """Rewrite a hyperedge's membership (legacy hyperedge_definitions setter)."""
    if isinstance(defn, list):
        source, target, directed = frozenset(defn), None, False
    elif bool(defn.get('directed', False)):
        source = frozenset(defn.get('head', []))
        target = frozenset(defn.get('tail', []))
        directed = True
    else:
        source, target, directed = frozenset(defn.get('members', [])), None, False
    _rewrite_sides(g, eid, source, target, directed=directed, kind=ST.HYPER)


def set_edge_direction_policy(g, eid, policy):
    """Attach a flexible-direction policy to an edge (legacy setter)."""
    if g._store.edge_slot(eid) is not None:
        g._store.set_edge_policy(eid, policy)


def set_entity_types(g, mapping):
    """Set entity kinds from a ``vid -> kind`` map.

    A reader that rebuilds a graph from a file knows which of the entities it
    read are edges in their own right, and it names them by their id.
    """
    for vid, kind in dict(mapping).items():
        ekey = I.resolve_ekey(g, vid)
        internal = _internal_entity_kind(kind)
        register_entity(g, ekey, internal)
        if internal == 'edge_entity':
            register_entity_as_edge(g, ekey[0])


def set_entity_kinds(g, mapping):
    """Set the kind of each entity an ``ekey -> kind`` map names.

    An entity the map names but the graph does not hold is added at the next free
    row. A reader needs that, because a file may record the kind of an entity
    that the structure it read never mentions.
    """
    for ekey, kind in mapping.items():
        register_entity(g, ekey, kind)
        if kind == 'edge_entity':
            register_entity_as_edge(g, ekey[0])


def remap_entity_keys(g, remap):
    """Move each entity an ``ekey -> ekey`` map names, keeping the row it holds.

    A reader needs this when the coordinate a file stored for an entity only
    makes sense once the aspects the same file declares are known, and declaring
    aspects over a flat graph needs it for every node it already holds.

    An identity changes and an address does not, so this changes the store rather
    than rebuilding it. Every member list and every position survives, which is
    what makes the whole call cost the entities alone.
    """
    if remap:
        g._store.rekey(remap)


def set_edge_coeffs(g, edge_id, coeffs):
    """Overwrite an edge column's incidence coefficients (stoichiometry)."""
    store = g._store
    slot = store.edge_slot(edge_id)
    if slot is None:
        raise KeyError(f'Edge {edge_id} not found')
    # Fold the new coefficients over the column the edge already has, whether it
    # carries explicit ones or derives them from its weight and its directedness.
    # The keys are the identities the edge names, which is the form the caller
    # holds too.
    base = {S.endpoint_form(g, key): value for key, value in S.edge_members(g, edge_id).items()}
    # On a multilayer graph the endpoints above are supra-node keys ``(vid, coord)``,
    # while callers key coefficients by bare node id (from_sbml passes raw SBML
    # species ids). The two spellings never collide, so without resolving them the
    # update *adds* a second entry for the same node and the rebuilt column sums
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
    g._mark_structure_changed()
    D.invalidate_sparse_caches(g)
    store.set_edge_coefficients(edge_id, {n: v for n, v in base.items() if v != 0.0})


def replace_edge_coeffs(g, edge_id, coeffs):
    """Set the coefficients of an edge column outright, dropping what it held.

    Unlike :func:`set_edge_coeffs`, this folds nothing over the column the edge
    already has. A reader that recovers a whole column from a file needs the
    column it read and nothing else.
    """
    if g._store.edge_slot(edge_id) is None:
        return
    D.invalidate_sparse_caches(g)
    g._store.set_edge_coefficients(
        edge_id, {member: value for member, value in coeffs.items() if value != 0.0}
    )


def set_hyperedge_members(g, eid, *, members=None, head=None, tail=None):
    """Turn an edge the graph already holds into a hyperedge over these members.

    Pass ``head`` and ``tail`` for a directed hyperedge, or ``members`` for an
    undirected one. A reader that meets the members of an edge after the edge
    itself needs this.
    """
    if members is None:
        source, target, directed = frozenset(head or ()), frozenset(tail or ()), True
    else:
        source, target, directed = frozenset(members), None, False
    D.invalidate_sparse_caches(g)
    _rewrite_sides(g, eid, source, target, directed=directed, kind=ST.HYPER)


# The three fields the dict views write. Each is one field of one edge, so each
# reaches the store through the narrow write for that field rather than through
# a member list the write never touched.
_EDGE_FIELD_WRITES = {
    'ml_layers': 'set_edge_ml_layers',
    'ml_kind': 'set_edge_ml_kind',
    'directed': 'set_edge_directed',
    'weight': 'set_edge_weight',
}


def set_edge_field(g, eid, field, value):
    """Set one field of one edge (backs the legacy dict-view writers)."""
    store = g._store
    if store.edge_slot(eid) is None:
        return
    write = _EDGE_FIELD_WRITES.get(field)
    if write is not None:
        getattr(store, write)(eid, value)


def set_edge_kind(g, eid, kind):
    """Set an edge's kind: ``'hyper'`` makes it a hyperedge, anything else its role."""
    store = g._store
    if store.edge_slot(eid) is None:
        return
    if kind == 'hyper':
        store.set_edge_kind(eid, ST.HYPER)
    else:
        store.set_edge_ml_kind(eid, kind)


def reverse_directions(g):
    """Swap the two sides of every directed edge and hyperedge.

    An undirected edge names the same entities whichever way round its sides are
    stored, so nothing about it changes. A hyperedge is directed exactly when it
    names a target side.
    """
    store = g._store
    for reference in S.iter_edges(g):
        if reference.directed:
            store.reverse_edge(reference.id)
    g._mark_structure_changed()
    D.invalidate_sparse_caches(g)


def make_undirected(g, *, drop_flexible=True, update_default=True):
    """Convert all existing edges to undirected form in place; returns the graph.

    A binary edge keeps its two sides and both endpoints then carry ``+w``. A
    hyperedge collapses head and tail into one member set. An explicit signed
    column is dropped either way, so the symmetric one is derived — which is what
    the legacy overwrite did.
    """
    store = g._store
    for reference in S.iter_edges(g):
        edge_id = reference.id
        # The direction comes first and the coefficients are derived once from it.
        store.set_edge_directed(edge_id, False)
        store.set_edge_explicit(edge_id, False)
        if reference.kind == S.HYPER:
            store.merge_sides(edge_id)

    if drop_flexible:
        for edge_id in list(S.edge_policies(g)):
            store.set_edge_policy(edge_id, None)
    if update_default:
        g.directed = False
    g._mark_structure_changed()
    D.invalidate_sparse_caches(g)
    return g


def drop_edges(g, edge_ids) -> None:
    """Take edges out of the store.

    The store frees a slot and moves no other edge, so the column of an edge
    after the dropped one is one lower and nothing has to be written to say so.
    """
    store = g._store
    for edge_id in edge_ids:
        if store.edge_slot(edge_id) is not None:
            store.remove_edge(edge_id)


def drop_orphan_edge_entities(g, edge_ids) -> None:
    """Remove the entity of an edge-entity whose edge has just been removed.

    An edge-entity is one identity on both axes: the entity is the edge. Once the
    edge is gone the entity names nothing, so it goes the way a removed node
    goes, and an edge that held it as an endpoint goes with it. That is the same
    cascade a node removal runs, and it is why this calls it.
    """
    orphans = [
        edge_id
        for edge_id in edge_ids
        if not S.has_edge(g, edge_id) and S.has_entity_id(g, edge_id, kind=S.EDGE_ENTITY)
    ]
    if orphans:
        remove_nodes_bulk(g, orphans)


def drop_entities(g, keys) -> None:
    """Take entities out of the store, once every edge that named one is gone.

    The store frees a slot and moves no other entity, so a row after the dropped
    one is one lower and nothing has to be written to say so.
    """
    store = g._store
    for key in keys:
        if store.entity_slot(key) is not None:
            store.remove_entity(key)


def remove_nodes_bulk(g, node_ids):
    """Remove many nodes, their incident edges, and compact entity rows."""
    drop_keys = set()
    drop_node_ids = set()
    for vid in node_ids:
        try:
            ekey = I.resolve_ekey(g, vid)
        except (KeyError, ValueError, TypeError):
            continue
        if g._store.entity_slot(ekey) is None:
            continue
        drop_keys.add(ekey)
        drop_node_ids.add(ekey[0] if isinstance(ekey, tuple) and len(ekey) == 2 else ekey)

    if not drop_keys:
        return

    # A node id names one entity per layer, and removing the id removes the
    # edges of every one of them. The store keeps the edges each entity takes
    # part in, so this costs their degree rather than a pass over every edge.
    drop_es: set = set()
    for vid in drop_node_ids:
        drop_es |= S.edges_of_id(g, vid)

    if drop_es:
        remove_edges_bulk(g, drop_es)

    D.bump_structure(g)
    D.invalidate_sparse_caches(g)
    drop_entities(g, drop_keys)

    for slice_data in g._slices.values():
        slice_data['nodes'].difference_update(drop_node_ids)


def remove_orphan_node_layers(g, drop_keys):
    """Drop specific ``(vid, layer)`` node entities that carry no incident edges.

    Unlike :func:`remove_nodes_bulk` (which drops every edge touching the bare
    node id, and the node-attribute row for that id), this removes only the
    given node-layer entity rows and compacts the incidence matrix. The node id
    itself survives through its other node-layers, so ``_node_table``,
    slice membership, and incident edges are all left untouched.

    Callers MUST guarantee every key in ``drop_keys`` is an orphan node-layer
    (no edge references it); nothing here inspects or rewrites edges.
    """
    drop_keys = {k for k in drop_keys if g._store.entity_slot(k) is not None}
    if not drop_keys:
        return

    D.bump_structure(g)
    D.invalidate_sparse_caches(g)
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


def batch_add_nodes(g, nodes, layer=None, slice=None, default_attrs=None):
    """Add many nodes through the bulk mutation path."""
    slice = slice or g._current_slice
    default_attrs = default_attrs or {}

    # --- normalize input ---
    norm = []
    for it in nodes:
        if isinstance(it, dict):
            if it.get('node_id'):
                vid = it['node_id']
                _id_keys = {'node_id'}
            elif it.get('id'):
                vid = it['id']
                _id_keys = {'node_id', 'id'}
            elif it.get('name'):
                vid = it['name']
                _id_keys = {'node_id', 'id', 'name'}
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
    # Written straight to the store rather than through the general door, which
    # re-checks the kind of an entity the graph already holds. Nothing here
    # changes a kind, so only the unseen keys are added.
    coord = g._resolve_node_insert_coord(
        layer, node_ids=[vid for vid, _ in norm], context='_add_nodes_batch'
    )
    store = g._store
    entity_slot = store.entity_slot
    new_rows = 0
    for vid, _ in norm:
        ekey = (vid, coord)
        if entity_slot(ekey) is None:
            store.add_entity(ekey, ST.NODE)
            new_rows += 1
    if new_rows:
        D.bump_structure(g)

    # --- slice ---
    g.slices._ensure_slice(slice)['nodes'].update(vid for vid, _ in norm)

    # --- attributes ---
    # A cell each, straight into the columns. There is no table to rebuild and no
    # backend to take a fast path for: a non-string id names a supra-node, which
    # the node table does not hold a row for either way.
    attr_store = g._attr_store
    for vid, attrs in norm:
        if attrs and isinstance(vid, str):
            attr_store.set_node_attrs(vid, {k: _sanitize(v) for k, v in attrs.items()})


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
    # entities ensured, and its edge spec built in one iteration — no
    # intermediate ``norm`` dict copy and no separate scans. Rows, columns and
    # the incidence matrix are all derived from the store, so the loop only needs
    # the ordering guarantee that an entity is added before the edge that names
    # it.
    #
    # An endpoint the graph has not met yet becomes a node here, written to the
    # store inline rather than through the general door. The edges go in
    # together once the loop ends: the store grows each of its arrays once for
    # the whole batch, where a write per edge grew them once per edge.
    _slot = g._store
    # Both accessors are one dict probe, and the loop makes three of them an
    # edge, so it takes the probe rather than the method that wraps it. Neither
    # dict is rebound while a batch is being written — only a copy of the store
    # gives itself new ones.
    _entity_slot = _slot._entity_slot.get
    _edge_slot = _slot._edge_slot.get
    _flat = g._aspects == ('_',)
    _flat_coord = ('_',)
    _is_multilayer = not _flat
    _g_directed = g.directed
    _RESERVED = _BINARY_BATCH_RESERVED_KEYS

    _next_id = g._next_edge_id
    _added = 0

    out_ids: list = []
    entity_out: list = []
    _slice_eids: dict = {}
    _slice_vids: dict = {}
    _slice_weights: list = []
    _specs: list = []
    _spec_at: dict = {}

    def _ensure_endpoint(vid, et):
        """Register the entity one endpoint names, and return its key."""
        if isinstance(vid, tuple) and len(vid) == 2 and isinstance(vid[1], tuple):
            ekey = vid
        elif _flat:
            ekey = (vid, _flat_coord)
        else:
            ekey = (
                vid,
                g._resolve_node_insert_coord(None, node_ids=vid, context='_add_edges_batch'),
            )
        if _entity_slot(ekey) is None:
            if (
                (et == 'node_edge' or et == 'edge_placeholder')
                and isinstance(vid, str)
                and vid.startswith('edge_')
            ):
                g._ensure_edge_entity_placeholder(vid)
                # The placeholder registers the edge entity under the key its own
                # resolution gives, which need not be the one worked out above.
                return I.resolve_ekey(g, vid)
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

            _fields = len(it)
            if _fields == 2 or (_fields == 3 and 'weight' in it):
                # A plain edge names its two endpoints and at most a weight
                # beside them, which is what a bulk load is made of. Its length
                # says so, because every key it could hold is a reserved one: a
                # length of two or three leaves every field below on its default
                # and leaves nothing to collect as an attribute, so one length
                # read stands for six lookups and the subset test at the end of
                # the loop.
                w = it['weight'] if _fields == 3 else default_weight
                edge_type = default_edge_type
                prop = default_propagate
                slice_local = slice
                slice_w = default_slice_weight
                e_dir = default_edge_directed
                edge_id = None
                _item = None
            else:
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
        # A flat graph keys an entity by its id and the placeholder coordinate,
        # so an endpoint the store already holds is answered by its key alone.
        # A load names each of its nodes many times and the general path is
        # only needed for the first, which is where it registers the entity.
        if _flat and type(s) is str and type(t) is str:
            source_key = (s, _flat_coord)
            target_key = (t, _flat_coord)
            if _entity_slot(source_key) is None:
                source_key = _ensure_endpoint(s, edge_type)
            if _entity_slot(target_key) is None:
                target_key = _ensure_endpoint(t, edge_type)
        else:
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

        # ── the edge ───────────────────────────────────────────────────────────
        # Two endpoints are two member entries, and the loop already holds both.
        # An edge the graph already holds keeps the policy it carries, because
        # this write says nothing about one; everything else it stated is
        # replaced. An id the batch names twice is replaced where it stands, so
        # it keeps the column its first mention took, which is the column a
        # remove and an add would have given it back.
        weight = float(w) if w is not None else 1.0
        at = _spec_at.get(edge_id)
        if at is None:
            old_slot = _edge_slot(edge_id)
            policy = None
            if old_slot is not None:
                policy = _slot.edge_policy.get(old_slot)
                _slot.remove_edge(edge_id)
        else:
            policy = _specs[at][8]
        spec = (
            edge_id,
            (
                (source_key, weight, ST.SOURCE),
                (target_key, -weight if is_dir else weight, ST.TARGET),
            ),
            ST.BINARY,
            is_dir,
            w,
            False,
            ml_kind,
            ml_layers,
            policy,
        )
        if at is None:
            _spec_at[edge_id] = len(_specs)
            _specs.append(spec)
        else:
            _specs[at] = spec
        if len(_specs) >= ST.BULK_CHUNK:
            _slot.add_edges(_specs)
            _specs = []
            _spec_at = {}
        _added += 1

        # ── slice membership (tracks bare node ids) ──────────────────────────
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

    if _specs:
        _slot.add_edges(_specs)

    g._next_edge_id = _next_id
    D.bump_structure(g)
    if _added:
        D.bump_structure(g)

    # Incidence is derived from the store written above, so the batch never
    # patches matrix cells. A read after it extends the cached matrix by the
    # columns these edges appended.
    if out_ids:
        g._mark_structure_changed()
        g._invalidate_sparse_caches()

    for sid, eids in _slice_eids.items():
        g.slices._ensure_slice(sid)['edges'].update(eids)
    for sid, vids in _slice_vids.items():
        g._slices[sid]['nodes'].update(vids)
    for sid, eid, sw in _slice_weights:
        g.attrs.set_edge_slice_attrs(sid, eid, weight=sw)

    if pending_attrs:
        g.attrs.set_edge_attrs_bulk(pending_attrs)

    if as_entity:
        flat = g._aspects == ('_',)
        flat_coord = ('_',)
        for eid in out_ids:
            ekey = (
                (eid, flat_coord) if flat and isinstance(eid, str) else g._resolve_entity_key(eid)
            )
            if _entity_slot(ekey) is None:
                _slot.add_entity(ekey, ST.EDGE_ENTITY)
            slot = _edge_slot(eid)
            if slot is not None and int(_slot.edge_kind[slot]) == ST.BINARY:
                _slot.set_edge_kind(eid, ST.NODE_EDGE)
        D.bump_structure(g)

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
    _slot = g._store

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
        # produced the "Ambiguous bare node_id" failures.
        if g._aspects != ('_',):
            placeholder = g._placeholder_layer_coord()
            real_keys = [k for k in _slot.entity_keys_of_id(u) if k[1] != placeholder]
            if len(real_keys) == 1:
                return u, real_keys[0][1]
        coord = g._resolve_node_insert_coord(None, node_ids=u, context='_add_hyperedges_batch')
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
            if _slot.entity_slot(ekey) is None:
                _slot.add_entity(ekey, ST.NODE)

    D.bump_structure(g)

    slices = g._slices

    out_ids = []
    attrs_batch = {}
    specs: list = []
    spec_at: dict = {}

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

        resolved_members = d.get('_resolved_members')
        resolved_head = d.get('_resolved_head')
        resolved_tail = d.get('_resolved_tail')
        is_dir = members is None

        # Every member is already resolved to an entity key, so the member list
        # is the resolved sides and the coefficient each side takes. An entity
        # named twice on one side is one member of it.
        if resolved_members is not None:
            source_keys, target_keys = dict.fromkeys(resolved_members), ()
        else:
            source_keys = dict.fromkeys(resolved_head)
            target_keys = dict.fromkeys(resolved_tail)
        role = ST.SOURCE if target_keys else ST.MEMBER
        member_entries = [(key, w, role) for key in source_keys]
        target_coefficient = -w if is_dir else w
        member_entries += [(key, target_coefficient, ST.TARGET) for key in target_keys]
        # The edges go in together once the loop ends, in chunks, for the reason
        # ``ST.BULK_CHUNK`` gives. An id the batch names twice is replaced where it
        # stands, so it keeps the column its first mention took.
        at = spec_at.get(e_id)
        if at is None:
            old_slot = _slot.edge_slot(e_id)
            policy = None
            if old_slot is not None:
                policy = _slot.edge_policy.get(old_slot)
                _slot.remove_edge(e_id)
        else:
            policy = specs[at][8]
        spec = (
            e_id,
            member_entries,
            ST.HYPER,
            is_dir,
            w,
            False,
            ml_kind_for_e,
            ml_layers_for_e,
            policy,
        )
        if at is None:
            spec_at[e_id] = len(specs)
            specs.append(spec)
        else:
            specs[at] = spec
        if len(specs) >= ST.BULK_CHUNK:
            _slot.add_edges(specs)
            specs = []
            spec_at = {}

        if slice_local is not None:
            if slice_local not in slices:
                slices[slice_local] = SliceRecord()
            slices[slice_local]['edges'].add(e_id)
            # Slice membership tracks bare vids, not (vid, layer) keys.
            if resolved_members is not None:
                slices[slice_local]['nodes'].update(k[0] for k in resolved_members)
            else:
                slices[slice_local]['nodes'].update(k[0] for k in resolved_head)
                slices[slice_local]['nodes'].update(k[0] for k in resolved_tail)

        sub_attrs = d.get('attributes') or d.get('attrs') or {}
        flat_attrs = {k: v for k, v in d.items() if k not in _HYPER_BATCH_RESERVED_KEYS}
        if sub_attrs or flat_attrs:
            merged = dict(sub_attrs)
            merged.update(flat_attrs)
            attrs_batch[e_id] = merged

        out_ids.append(e_id)

    if specs:
        _slot.add_edges(specs)

    g._mark_structure_changed()
    g._invalidate_sparse_caches()
    if attrs_batch:
        g.attrs.set_edge_attrs_bulk(attrs_batch)

    return out_ids
