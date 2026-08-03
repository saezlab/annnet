"""The read-only structural query facade.

This module is the one boundary between the canonical store and the rest of the
package. Input-output code, adapters, and bridges read topology through the
functions here. They never read a private store attribute of the graph.

The facade answers questions about structure only. It reports which entities and
edges exist, which entities an edge holds, and which edges touch an entity. It
does not report attributes, and it never writes.

Every address in the facade is an identity. An entity is addressed by an entity
key, which is a ``(node_id, layer_coord)`` pair. An edge is addressed by its
edge id. A row number and a column number belong to one materialized matrix, so
neither appears in an answer.

The facade hides which store backs the graph. The current store keeps entity and
edge records. A later store keeps slot-addressed member lists. The signatures
here stay the same across that change, so the callers stay the same too.
"""

from __future__ import annotations

from typing import NamedTuple
from collections.abc import Iterator

import numpy as np

from . import _store as ST, _identity as I

# Entity kinds.
NODE = 'node'
EDGE_ENTITY = 'edge_entity'

# Edge kinds.
BINARY = 'binary'
HYPER = 'hyper'
NODE_EDGE = 'node_edge'
PLACEHOLDER = 'placeholder'

DIRECTIONS = ('in', 'out', 'both')

_ENTITY_KIND_OF_RECORD = {
    'vertex': NODE,
    'edge_entity': EDGE_ENTITY,
}
_EDGE_KIND_OF_RECORD = {
    'binary': BINARY,
    'hyper': HYPER,
    'vertex_edge': NODE_EDGE,
    'edge_placeholder': PLACEHOLDER,
}


class EntityRef(NamedTuple):
    """One entity of a graph, addressed by identity."""

    id: str
    kind: str
    layer: tuple

    @property
    def key(self) -> tuple:
        """The entity key, which is the ``(id, layer)`` pair."""
        return (self.id, self.layer)


class EdgeRef(NamedTuple):
    """One edge of a graph, addressed by identity.

    ``directed`` and ``weight`` are the answers the graph gives for this edge.
    The two ``declared`` fields are what the edge itself states, where ``None``
    means the edge inherits the default of the graph. Persistence keeps the
    declared values, so that a file records what the user set and not what the
    graph resolved.
    """

    id: str
    kind: str
    directed: bool
    weight: float
    ml_kind: object = None
    ml_layers: object = None
    declared_directed: object = None
    declared_weight: object = None


class Endpoints(NamedTuple):
    """The two sides of an edge, as sets of entity keys.

    The source side carries the positive coefficient and the target side carries
    the negative one. An undirected edge keeps the sides it was stored with.
    Read ``EdgeRef.directed`` to learn whether the sides mean a direction.
    """

    source: frozenset
    target: frozenset


# ---------------------------------------------------------------------------
# Which store backs the graph
# ---------------------------------------------------------------------------
# The facade answers the same questions whichever store holds the graph. A caller
# passes a graph or a store, and every function below picks the reader that fits.
# This is what lets the package move to the slot store one caller at a time.


def is_slot_backed(graph) -> bool:
    """Return True when the slot-addressed store answers for this graph."""
    return hasattr(graph, 'member_start') or getattr(graph, '_store', None) is not None


def store_of(graph):
    """Return the slot store behind a graph, or the graph when it is one."""
    return graph if hasattr(graph, 'member_start') else graph._store


_SLOT_ENTITY_KIND = {0: NODE, 1: EDGE_ENTITY}
_SLOT_EDGE_KIND = {0: BINARY, 1: HYPER, 2: NODE_EDGE, 3: PLACEHOLDER}

# The slot store holds no matrix library and imports nothing from the core, so
# naming it here costs no cycle. These are read once per edge of an enumeration,
# which is why they are bound to locals rather than reached through the module.
_INHERIT = ST.INHERIT
_TARGET = ST.TARGET
_ON_SOURCE = ST.ON_SOURCE
_ON_TARGET = ST.ON_TARGET
_SLOT_NODE = ST.NODE
_SLOT_PLACEHOLDER = ST.PLACEHOLDER
_SLOT_HYPER = ST.HYPER
_MEMBER = ST.MEMBER


def _slot_key(store, ref):
    """Return the entity key of a reference against a slot store."""
    if is_entity_key(ref):
        return ref
    if store.aspects == ('_',):
        return (ref, ('_',))
    matches = [key for _slot, key in store.live_entities() if key[0] == ref]
    if len(matches) == 1:
        return matches[0]
    return (ref, ('_',))


def _slot_entity_ref(store, key) -> EntityRef:
    slot = store.entity_slot(key)
    return EntityRef(key[0], _SLOT_ENTITY_KIND.get(int(store.entity_kind[slot]), NODE), key[1])


def _slot_hyper_directed(store, slot) -> bool:
    """Return whether a hyperedge names a target side at all.

    A hyperedge that names no target side is undirected, whichever flag it was
    declared with. That is what a boundary reaction looks like. Its members carry
    the plain member role rather than a source role, and one entry says so.
    """
    if int(store.member_len[slot]) == 0:
        return False
    return int(store.member_role[int(store.member_start[slot])]) != _MEMBER


def _slot_edge_ref(store, edge_id: str) -> EdgeRef:
    # Every value comes out of an array, and reading one element of an array is
    # dear enough that each is read once and reused. The construction is
    # positional for the same reason. This runs once per edge of an enumeration.
    slot = store.edge_slot(edge_id)
    kind = int(store.edge_kind[slot])
    declared = int(store.edge_directed[slot])
    weight = float(store.edge_weight[slot])
    if kind == _SLOT_HYPER:
        directed = _slot_hyper_directed(store, slot)
        declared_directed = None if declared == _INHERIT else bool(declared)
    elif declared != _INHERIT:
        directed = bool(declared)
        declared_directed = directed
    else:
        directed = True if store.directed is None else bool(store.directed)
        declared_directed = None
    return EdgeRef(
        edge_id,
        _SLOT_EDGE_KIND.get(kind, BINARY),
        directed,
        weight,
        store.edge_ml_kind.get(slot),
        store.edge_ml_layers.get(slot),
        declared_directed,
        weight,
    )


def _slot_carries_structure(store, slot) -> bool:
    """Return True when an edge slot holds structure rather than a name alone.

    A placeholder edge is an id the graph knows before the edge exists. It holds
    no members and occupies no column, so every count and every enumeration of
    the structural edges leaves it out, on either store.
    """
    return int(store.edge_kind[slot]) != _SLOT_PLACEHOLDER


def _slot_structural_edges(store) -> list:
    """Return the ``(slot, edge_id)`` pairs that carry structure, in slot order.

    The store keeps this against its clock, because the column of an edge is its
    position among these and a caller asks for one column at a time.
    """
    return store.structural_edges()


def _slot_require_edge(store, edge_id: str) -> int:
    slot = store.edge_slot(edge_id)
    if slot is None:
        raise KeyError(f'Unknown edge id: {edge_id!r}')
    return slot


# ---------------------------------------------------------------------------
# Identity resolution
# ---------------------------------------------------------------------------


def entity_key(graph, ref) -> tuple:
    """Return the entity key for a bare id or an entity key.

    The function resolves the address. It does not check that the entity exists.
    """
    return I.resolve_ekey(graph, ref)


def entity_key_of_row(graph, row: int) -> tuple:
    """Return the entity key that a materialized row belongs to.

    This is the inverse of :func:`entity_row`, and it carries the same warning: a
    row belongs to one materialized matrix and to nothing else.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        slots = store.live_entity_slots()
        if not 0 <= row < slots.size:
            raise KeyError(f'No entity at row {row}')
        return store.entity_key(int(slots[row]))
    try:
        return graph._row_to_entity[row]
    except KeyError:
        raise KeyError(f'No entity at row {row}') from None


def _require_entity(graph, ref) -> tuple:
    # An entity key that the store already holds needs no resolution. Resolving it
    # again would re-check its layer, and a stored layer is not always one the
    # graph still declares.
    if is_entity_key(ref) and ref in graph._entities:
        return ref
    key = entity_key(graph, ref)
    if key not in graph._entities:
        raise KeyError(f'Unknown entity: {ref!r}')
    return key


def _require_edge(graph, edge_id: str):
    try:
        return graph._edges[edge_id]
    except KeyError:
        raise KeyError(f'Unknown edge id: {edge_id!r}') from None


# ---------------------------------------------------------------------------
# Existence
# ---------------------------------------------------------------------------


def has_entity(graph, ref) -> bool:
    """Return True when the graph holds this entity."""
    if is_slot_backed(graph):
        store = store_of(graph)
        return store.entity_slot(_slot_key(store, ref)) is not None
    try:
        _require_entity(graph, ref)
    except (KeyError, ValueError, TypeError):
        return False
    return True


def has_entity_id(graph, entity_id: str, kind: str | None = None) -> bool:
    """Return True when any entity of the graph carries this id, in any layer.

    A bare id names one entity in a flat graph and any number of them in a
    multilayer graph. Ask this when the id alone is the question. Ask
    :func:`has_entity` when the answer has to be one entity. Pass ``kind`` to
    ask about nodes alone, or about edge entities alone.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        return any(
            key[0] == entity_id
            and (kind is None or _SLOT_ENTITY_KIND[int(store.entity_kind[slot])] == kind)
            for slot, key in store.live_entities()
        )
    if graph._aspects == ('_',):
        keys = ((entity_id, ('_',)),)
    else:
        keys = tuple(graph._vid_to_ekeys.get(entity_id, ()))
    for key in keys:
        record = graph._entities.get(key)
        if record is None:
            continue
        if kind is None or _ENTITY_KIND_OF_RECORD.get(record.kind, record.kind) == kind:
            return True
    return False


def has_edge(graph, edge_id: str) -> bool:
    """Return True when the graph holds this edge."""
    if is_slot_backed(graph):
        return store_of(graph).edge_slot(edge_id) is not None
    return edge_id in graph._edges


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
# A count is the size of the matching enumeration. Each store answers it from
# what it already holds, so a caller that needs a size never walks the graph.


def entity_count(graph) -> int:
    """Return how many entities the graph holds, nodes and edge entities."""
    if is_slot_backed(graph):
        return store_of(graph).entity_count
    return len(graph._entities)


def node_count(graph) -> int:
    """Return how many nodes the graph holds, leaving out the edge entities."""
    if is_slot_backed(graph):
        store = store_of(graph)
        return sum(
            1
            for slot, _key in store.live_entities()
            if _SLOT_ENTITY_KIND[int(store.entity_kind[slot])] == NODE
        )
    return sum(1 for record in graph._entities.values() if record.kind == 'vertex')


def edge_count(graph) -> int:
    """Return how many edges carry structure.

    An edge that occupies no column holds no members, so it counts for nothing
    here. :func:`iter_edges` leaves the same edges out.
    """
    if is_slot_backed(graph):
        return len(_slot_structural_edges(store_of(graph)))
    return len(graph._col_to_edge)


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def entity_ref(graph, ref) -> EntityRef:
    """Return the entity reference for one entity."""
    if is_slot_backed(graph):
        store = store_of(graph)
        key = _slot_key(store, ref)
        if store.entity_slot(key) is None:
            raise KeyError(f'Unknown entity: {ref!r}')
        return _slot_entity_ref(store, key)
    key = _require_entity(graph, ref)
    record = graph._entities[key]
    return EntityRef(key[0], _ENTITY_KIND_OF_RECORD.get(record.kind, record.kind), key[1])


def iter_entities(graph) -> Iterator[EntityRef]:
    """Yield every entity of the graph, in materialized row order.

    A multilayer graph holds one entity per node and layer, so one node id can
    appear more than once. The entity key tells the two apart.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        for _slot, key in store.live_entities():
            yield _slot_entity_ref(store, key)
        return
    for key, record in sorted(graph._entities.items(), key=lambda item: item[1].row_idx):
        yield EntityRef(key[0], _ENTITY_KIND_OF_RECORD.get(record.kind, record.kind), key[1])


# ---------------------------------------------------------------------------
# Cheap enumeration
# ---------------------------------------------------------------------------
# A caller that wants identities alone should not pay for a reference object per
# element. These three answer with the identities in the same order the full
# enumerations use, and they build nothing else.


def entity_keys(graph) -> list:
    """Return the key of every entity, in materialized row order."""
    if is_slot_backed(graph):
        # A free slot holds a null, so the live keys are the rest, in slot order.
        return [key for key in store_of(graph)._entity_key if key is not None]
    records = graph._entities
    # The row index is a maintained map from position to identity, so reading it
    # in order costs nothing. It is derived state, so a sort is the fallback when
    # it does not account for every entity.
    index = graph._row_to_entity
    if len(index) == len(records):
        return [index[row] for row in range(len(index))]
    return sorted(records, key=lambda key: records[key].row_idx)


def node_keys(graph) -> list:
    """Return the key of every node, in row order, leaving out the edge entities."""
    if is_slot_backed(graph):
        store = store_of(graph)
        # The kind array is converted once. Reading one element of it per entity
        # would cost more than the walk itself.
        kinds = store.entity_kind.tolist()
        return [
            key
            for slot, key in enumerate(store._entity_key)
            if key is not None and kinds[slot] == _SLOT_NODE
        ]
    records = graph._entities
    index = graph._row_to_entity
    if len(index) == len(records):
        # One pass. Going through ``entity_keys`` would build the whole key list
        # first and then walk it again to drop the edge entities.
        return [key for row in range(len(index)) if records[key := index[row]].kind == 'vertex']
    return [key for key in entity_keys(graph) if records[key].kind == 'vertex']


def node_ids(graph) -> list:
    """Return the bare id of every node, once each, in row order.

    A flat graph holds one entity per node, so the ids are already unique and
    nothing has to be set aside to notice a repeat. A multilayer graph holds one
    per layer, and the same id comes back from each of them.
    """
    keys = node_keys(graph)
    aspects = store_of(graph).aspects if is_slot_backed(graph) else graph._aspects
    if aspects == ('_',):
        return [key[0] for key in keys]
    seen: set = set()
    out: list = []
    for node_id, _layer in keys:
        if node_id not in seen:
            seen.add(node_id)
            out.append(node_id)
    return out


def edge_ids(graph) -> list:
    """Return the id of every structural edge, in materialized column order.

    An edge that occupies no column carries no structure and is left out, so
    this lists exactly what :func:`iter_edges` yields.
    """
    if is_slot_backed(graph):
        return [edge_id for _slot, edge_id in _slot_structural_edges(store_of(graph))]
    # The column index holds exactly the edges that carry structure, and
    # :func:`edge_at_column` and :func:`edge_count` already read it as the
    # authority on them, so reading it in order is the answer.
    index = graph._col_to_edge
    return [index[column] for column in range(len(index))]


def entities_by_id(graph) -> dict:
    """Group every entity of the graph under its bare id.

    One id covers one entity in a flat graph and one per layer in a multilayer
    graph. The result answers "which entities does this id stand for" in one
    pass, which is what a caller needs when it holds bare ids and has to place
    them in layers.
    """
    grouped: dict = {}
    for ref in iter_entities(graph):
        grouped.setdefault(ref.id, []).append(ref)
    return grouped


def _edge_directed(graph, record) -> bool:
    """Return the directedness of an edge, as the public edge view reports it."""
    if record.etype == 'hyper':
        return record.tgt is not None
    if record.etype in ('vertex_edge', 'edge_placeholder'):
        return bool(record.directed) if record.directed is not None else False
    if record.directed is not None:
        return bool(record.directed)
    return True if graph.directed is None else bool(graph.directed)


def _edge_ref_of_record(graph, edge_id: str, record) -> EdgeRef:
    # Positional construction, because a reference is built once per edge on
    # every enumeration and the keyword form costs nearly twice as much.
    weight = record.weight
    return EdgeRef(
        edge_id,
        _EDGE_KIND_OF_RECORD.get(record.etype, record.etype),
        _edge_directed(graph, record),
        float(weight) if weight is not None else 1.0,
        record.ml_kind,
        record.ml_layers,
        record.directed,
        weight,
    )


def edge_ref(graph, edge_id: str) -> EdgeRef:
    """Return the edge reference for one edge."""
    if is_slot_backed(graph):
        store = store_of(graph)
        _slot_require_edge(store, edge_id)
        return _slot_edge_ref(store, edge_id)
    return _edge_ref_of_record(graph, edge_id, _require_edge(graph, edge_id))


def iter_edges(graph, *, include_placeholders: bool = False) -> Iterator[EdgeRef]:
    """Yield every structural edge of the graph, in materialized column order.

    An edge with no column carries no structure. Set ``include_placeholders`` to
    see those too.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        pairs = store.live_edges() if include_placeholders else _slot_structural_edges(store)
        for _slot, edge_id in pairs:
            yield _slot_edge_ref(store, edge_id)
        return
    items = [
        (record.col_idx, edge_id, record)
        for edge_id, record in graph._edges.items()
        if include_placeholders or record.col_idx >= 0
    ]
    for _col, edge_id, record in sorted(items, key=lambda item: item[0]):
        yield _edge_ref_of_record(graph, edge_id, record)


# ---------------------------------------------------------------------------
# Member lists
# ---------------------------------------------------------------------------


def member_entries(record) -> dict:
    """Return the member list of one edge record, keyed by the stored endpoint.

    The result is the incidence column of the edge. An explicit coefficient wins.
    Otherwise the source side takes the weight and the target side takes the
    negated weight when the edge is directed.

    The keys are the endpoint forms the store holds, so this function allocates
    nothing. Use :func:`edge_members` for keys resolved to entity keys.
    """
    if record.coeffs is not None:
        return record.coeffs

    weight = record.weight if record.weight is not None else 1.0
    target_value = -weight if record.directed else weight
    entries: dict = {}
    source, target = record.src, record.tgt
    if isinstance(source, frozenset):
        for member in source:
            entries[member] = weight
    elif source is not None:
        entries[source] = weight
    if isinstance(target, frozenset):
        for member in target:
            entries[member] = target_value
    elif target is not None:
        entries[target] = target_value
    return entries


def _resolved_key(graph, member):
    """Return the entity key of a stored endpoint, or None when it has none.

    A multilayer graph may hold an endpoint as a bare id that covers more than
    one layer. Such an endpoint names no single entity, so it resolves to
    nothing and the materialized matrix leaves it out.
    """
    try:
        return entity_key(graph, member)
    except (KeyError, ValueError, TypeError):
        return None


def edge_members(graph, edge_id: str) -> dict:
    """Return the member list of one edge, keyed by entity key.

    The member list is the incidence column of the edge. The value of a member
    is its coefficient.

    An endpoint that names no single entity is left out, exactly as the
    materialized matrix leaves it out. Use :func:`edge_sides` to see every
    stored endpoint, and the invariant checker to find the unresolved ones.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        entries = store.members(_slot_require_edge(store, edge_id))
        summed: dict = {}
        for entity_slot, coefficient in zip(entries.entities, entries.coefficients, strict=False):
            member_key = store.entity_key(int(entity_slot))
            summed[member_key] = summed.get(member_key, 0.0) + float(coefficient)
        return summed
    record = _require_edge(graph, edge_id)
    members = {}
    for member, coefficient in member_entries(record).items():
        key = _resolved_key(graph, member)
        if key is not None:
            members[key] = float(coefficient)
    return members


def edge_sides(graph, edge_id: str) -> Endpoints:
    """Return the two sides of one edge, as the identities the store holds.

    An identity here is what the public API of the graph shows: a bare id in a
    flat graph, and normally an ``(id, layer_coord)`` pair in a multilayer
    graph. Nothing is resolved, so this works on an edge whose endpoint names no
    single entity. A writer that persists a graph needs exactly this.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        # A flat graph names an entity by its bare id, and the answer has to be in
        # the form the caller holds its own ids in.
        return store.endpoints(_slot_require_edge(store, edge_id), bare=store.aspects == ('_',))
    record = _require_edge(graph, edge_id)
    return Endpoints(_raw_side(record.src), _raw_side(record.tgt))


def _raw_side(side) -> frozenset:
    if side is None:
        return frozenset()
    if isinstance(side, (frozenset, set, list, tuple)) and not _is_entity_key(side):
        return frozenset(side)
    return frozenset({side})


def is_entity_key(value) -> bool:
    """Return True when a value is an explicit ``(id, layer_coord)`` entity key."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], tuple)
    )


_is_entity_key = is_entity_key


def edge_coefficients(graph, edge_id: str):
    """Return the explicit coefficients of an edge, or None when it has none.

    An explicit coefficient is one the user set, for example a stoichiometric
    value. An edge without them takes its coefficients from its weight and its
    directedness. The keys are the identities the store holds, so nothing is
    resolved and nothing is lost.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        slot = _slot_require_edge(store, edge_id)
        if not bool(store.edge_explicit[slot]):
            return None
        members = edge_members(graph, edge_id)
        if store.aspects != ('_',):
            return members
        # A flat graph names an entity by its bare id, and the answer has to be in
        # the form the caller holds its own ids in. See :func:`edge_sides`.
        return {key[0]: value for key, value in members.items()}
    record = _require_edge(graph, edge_id)
    if record.coeffs is None:
        return None
    return dict(record.coeffs)


def entity_row(graph, ref) -> int:
    """Return the row an entity occupies in the materialized matrix.

    A row is a position, so it belongs to one materialized matrix and to nothing
    else. Use this only to persist or to rebuild that matrix.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        slot = store.entity_slot(_slot_key(store, ref))
        if slot is None:
            raise KeyError(f'Unknown entity: {ref!r}')
        return int(np.count_nonzero(store.live_entity_slots() < slot))
    return int(graph._entities[_require_entity(graph, ref)].row_idx)


def edge_column(graph, edge_id: str) -> int:
    """Return the column an edge occupies in the materialized matrix.

    The result is ``-1`` when the edge carries no structure. A column is a
    position, so it belongs to one materialized matrix and to nothing else. Use
    this only to persist or to rebuild that matrix.
    """
    if is_slot_backed(graph):
        store = store_of(graph)
        return store.structural_column(_slot_require_edge(store, edge_id))
    return int(_require_edge(graph, edge_id).col_idx)


def edge_at_column(graph, column: int) -> str:
    """Return the edge that occupies one column of the materialized matrix.

    This is the inverse of :func:`edge_column`, and it carries the same warning: a
    column belongs to one materialized matrix and to nothing else. It exists for
    the public methods that still accept a position, and it goes away with them.
    """
    if is_slot_backed(graph):
        structural = _slot_structural_edges(store_of(graph))
        if not 0 <= column < len(structural):
            raise KeyError(f'No edge at column {column}')
        return structural[column][1]
    try:
        return graph._col_to_edge[column]
    except KeyError:
        raise KeyError(f'No edge at column {column}') from None


def _side_keys(graph, side) -> frozenset:
    if side is None:
        return frozenset()
    if isinstance(side, frozenset):
        return frozenset(entity_key(graph, member) for member in side)
    return frozenset({entity_key(graph, side)})


def edge_endpoints(graph, edge_id: str) -> Endpoints:
    """Return the source side and the target side of one edge, as entity keys."""
    if is_slot_backed(graph):
        store = store_of(graph)
        return store.endpoints(_slot_require_edge(store, edge_id))
    record = _require_edge(graph, edge_id)
    return Endpoints(
        source=_side_keys(graph, record.src),
        target=_side_keys(graph, record.tgt),
    )


# ---------------------------------------------------------------------------
# Incidence
# ---------------------------------------------------------------------------


def iter_hyperedges(graph):
    """Return ``(edge_id, record)`` for the live hyperedges of the graph.

    The result is cached against the structural clock, so a graph with no
    hyperedge never pays a full edge scan.
    """
    version = getattr(graph, '_structure_version', None)
    cache = getattr(graph, '_hyper_items_cache', None)
    if cache is None or cache[0] != version:
        items = [
            (edge_id, record)
            for edge_id, record in graph._edges.items()
            if record.etype == 'hyper' and record.col_idx >= 0
        ]
        graph._hyper_items_cache = (version, items)
        return items
    return cache[1]


def endpoint_form(graph, key):
    """Return the identity of an entity in the form the public API uses.

    A flat graph names an entity by its bare id. A multilayer graph names it by
    the ``(id, layer_coord)`` pair, because one id covers more than one layer.
    The adjacency indexes are keyed the same way.
    """
    aspects = store_of(graph).aspects if is_slot_backed(graph) else graph._aspects
    return key if aspects != ('_',) else key[0]


_probe_form = endpoint_form


def _stored_directed(graph, record) -> bool:
    """Return directedness the way the adjacency traversal reads it."""
    return bool(record.directed if record.directed is not None else graph.directed)


def _slot_incident(store, key, *, ordered: bool = True):
    """Yield ``(edge_slot, on_source, on_target, directed)`` for the edges of one entity.

    The store keeps an index from an entity to the edges that name it, so this
    costs the degree of the entity rather than a scan over every edge. The roles
    come from the member list of each edge, which is where a self-loop shows up as
    both sides at once.

    Set ``ordered`` to False when the caller collects the result into a set, so
    that the slots do not have to be sorted first.

    The index carries the sides the entity takes as well as the edges, so this
    reads no member list at all. It is what makes a traversal cost the degree of
    the entity rather than the members of every edge that touches it.
    """
    slot = store.entity_slot(key)
    edge_directed = store.edge_directed
    graph_directed = True if store.directed is None else bool(store.directed)

    incident = store._entity_edges.get(slot, {})
    for edge_slot, (sides, peer) in sorted(incident.items()) if ordered else incident.items():
        declared = int(edge_directed[edge_slot])
        directed = bool(declared) if declared != _INHERIT else graph_directed
        yield edge_slot, bool(sides & _ON_SOURCE), bool(sides & _ON_TARGET), directed, peer


def _slot_entity_edges(store, key, direction: str) -> tuple:
    wants_out = direction in ('out', 'both')
    wants_in = direction in ('in', 'both')
    # ``_slot_incident`` already walks the edges in slot order, which is column
    # order, so the result needs no sort of its own.
    edge_id = store.edge_id
    return tuple(
        edge_id(edge_slot)
        for edge_slot, on_source, on_target, directed, _peer in _slot_incident(store, key)
        if not directed or (wants_out and on_source) or (wants_in and on_target)
    )


def _slot_neighbors(store, key, direction: str) -> list:
    # Which entities an edge leads to can only come from its member list, so this
    # reads one per incident edge. It reads it once and adds the keys straight to
    # the result, because the two sides as sets would be built and thrown away.
    wants_out = direction in ('out', 'both')
    wants_in = direction in ('in', 'both')
    entity_is_edge = _SLOT_ENTITY_KIND.get(int(store.entity_kind[store.entity_slot(key)])) == (
        EDGE_ENTITY
    )
    member_start = store.member_start
    member_len = store.member_len
    member_ent = store.member_ent
    member_role = store.member_role
    entity_key_of_slot = store._entity_key
    found = set()

    for edge_slot, on_source, on_target, directed, peer in _slot_incident(
        store, key, ordered=False
    ):
        if peer is not None:
            # The index already names the entity on the other entry, so an edge
            # of two entries needs no member list. This is the common shape.
            if directed and not (
                (wants_out and on_source)
                or (
                    wants_in
                    and on_target
                    and (
                        direction == 'in'
                        or entity_is_edge
                        or _SLOT_EDGE_KIND.get(int(store.edge_kind[edge_slot])) == HYPER
                    )
                )
            ):
                continue
            member = entity_key_of_slot[peer]
            if directed or member != key:
                found.add(member)
            continue
        if directed:
            if _SLOT_EDGE_KIND.get(int(store.edge_kind[edge_slot])) == HYPER:
                take_target = wants_out and on_source
                take_source = wants_in and on_target and not take_target
            else:
                take_target = wants_out and on_source
                take_source = wants_in and on_target and (direction == 'in' or entity_is_edge)
            if not (take_target or take_source):
                continue
            drop_self = False
        else:
            # An undirected edge leads to everyone else on it, and not to the
            # entity the query started from.
            take_target = take_source = True
            drop_self = True

        start = int(member_start[edge_slot])
        stop = start + int(member_len[edge_slot])
        for entity_slot, role in zip(
            member_ent[start:stop].tolist(), member_role[start:stop].tolist(), strict=False
        ):
            if take_target if role == _TARGET else take_source:
                member = entity_key_of_slot[entity_slot]
                if not (drop_self and member == key):
                    found.add(member)

    flat = store.aspects == ('_',)
    return [member[0] if flat else member for member in found]


def neighbors(graph, ref, direction: str = 'both') -> list:
    """Return the entities adjacent to one entity, in the public identity form.

    This is a structural query, so it lives here rather than in a caller. It
    reads the store directly and builds no intermediate object per edge, because
    a traversal is a hot path. Building an edge reference and two endpoint sets
    for every incident edge costs about ten times as much.

    ``direction`` is ``"out"``, ``"in"``, or ``"both"``. An undirected edge
    answers in both directions. A directed edge answers on the side that holds
    the entity, with one exception: an unqualified query does not walk backwards
    along a directed edge unless the entity is an edge-entity, because both sides
    of that edge describe it.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}, got {direction!r}')
    if is_slot_backed(graph):
        store = store_of(graph)
        key = _slot_key(store, ref)
        if store.entity_slot(key) is None:
            return []
        return _slot_neighbors(store, key, direction)
    try:
        key = _require_entity(graph, ref)
    except (KeyError, ValueError, TypeError):
        return []

    probe = endpoint_form(graph, key)
    graph._ensure_edge_indexes()
    edges = graph._edges
    default_directed = graph.directed if graph.directed is not None else True
    entity_is_edge = graph._entities[key].kind == 'edge_entity'
    wants_out = direction in ('out', 'both')
    wants_in = direction in ('in', 'both')

    found = set()

    # The entity is on the source side of these edges.
    for edge_id in graph._src_to_edges.get(probe, ()):
        record = edges[edge_id]
        if record.col_idx < 0:
            continue
        directed = record.directed if record.directed is not None else default_directed
        if wants_out:
            found.add(record.tgt)
        elif not directed:
            found.add(record.tgt)

    # The entity is on the target side of these edges.
    for edge_id in graph._tgt_to_edges.get(probe, ()):
        record = edges[edge_id]
        if record.col_idx < 0:
            continue
        directed = record.directed if record.directed is not None else default_directed
        if not directed:
            found.add(record.src)
        elif wants_in and (direction == 'in' or entity_is_edge):
            found.add(record.src)

    # A hyperedge is not in the adjacency indexes, so it is scanned from the
    # cached list of live hyperedges.
    for _edge_id, record in iter_hyperedges(graph):
        if record.tgt is not None:
            if wants_out and probe in record.src:
                found |= set(record.tgt)
            elif wants_in and probe in record.tgt:
                found |= set(record.src)
        else:
            members = record.src
            if probe in members:
                found |= set(members) - {probe}

    return list(found)


def entity_edges(graph, ref, direction: str = 'both') -> tuple:
    """Return the ids of the edges that touch one entity, in column order.

    An undirected edge counts in both directions. A hyperedge counts on the side
    that holds the entity.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}, got {direction!r}')
    if is_slot_backed(graph):
        store = store_of(graph)
        key = _slot_key(store, ref)
        if store.entity_slot(key) is None:
            raise KeyError(f'Unknown entity: {ref!r}')
        return _slot_entity_edges(store, key, direction)
    key = _require_entity(graph, ref)
    probe = _probe_form(graph, key)
    graph._ensure_edge_indexes()

    found: dict[str, int] = {}
    wants_out = direction in ('out', 'both')
    wants_in = direction in ('in', 'both')

    for edge_id in graph._src_to_edges.get(probe, ()):
        record = graph._edges.get(edge_id)
        if record is None or record.col_idx < 0:
            continue
        if wants_out or not _stored_directed(graph, record):
            found[edge_id] = record.col_idx

    for edge_id in graph._tgt_to_edges.get(probe, ()):
        record = graph._edges.get(edge_id)
        if record is None or record.col_idx < 0:
            continue
        if wants_in or not _stored_directed(graph, record):
            found[edge_id] = record.col_idx

    for edge_id, record in iter_hyperedges(graph):
        if edge_id in found:
            continue
        on_source = probe in record.src if record.src is not None else False
        on_target = probe in record.tgt if record.tgt is not None else False
        if not (on_source or on_target):
            continue
        directed = record.tgt is not None
        if not directed or (wants_out and on_source) or (wants_in and on_target):
            found[edge_id] = record.col_idx

    return tuple(sorted(found, key=found.__getitem__))
