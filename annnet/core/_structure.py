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

from . import _identity as I

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
# Identity resolution
# ---------------------------------------------------------------------------


def entity_key(graph, ref) -> tuple:
    """Return the entity key for a bare id or an entity key.

    The function resolves the address. It does not check that the entity exists.
    """
    return I.resolve_ekey(graph, ref)


def entity_key_of_row(graph, row: int) -> tuple:
    """Return the entity key that a materialized row belongs to."""
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
    try:
        _require_entity(graph, ref)
    except (KeyError, ValueError, TypeError):
        return False
    return True


def has_edge(graph, edge_id: str) -> bool:
    """Return True when the graph holds this edge."""
    return edge_id in graph._edges


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def entity_ref(graph, ref) -> EntityRef:
    """Return the entity reference for one entity."""
    key = _require_entity(graph, ref)
    record = graph._entities[key]
    return EntityRef(
        id=key[0],
        kind=_ENTITY_KIND_OF_RECORD.get(record.kind, record.kind),
        layer=key[1],
    )


def iter_entities(graph) -> Iterator[EntityRef]:
    """Yield every entity of the graph, in materialized row order.

    A multilayer graph holds one entity per node and layer, so one node id can
    appear more than once. The entity key tells the two apart.
    """
    for key, record in sorted(graph._entities.items(), key=lambda item: item[1].row_idx):
        yield EntityRef(
            id=key[0],
            kind=_ENTITY_KIND_OF_RECORD.get(record.kind, record.kind),
            layer=key[1],
        )


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
    return EdgeRef(
        id=edge_id,
        kind=_EDGE_KIND_OF_RECORD.get(record.etype, record.etype),
        directed=_edge_directed(graph, record),
        weight=float(record.weight) if record.weight is not None else 1.0,
        ml_kind=record.ml_kind,
        ml_layers=record.ml_layers,
        declared_directed=record.directed,
        declared_weight=record.weight,
    )


def edge_ref(graph, edge_id: str) -> EdgeRef:
    """Return the edge reference for one edge."""
    return _edge_ref_of_record(graph, edge_id, _require_edge(graph, edge_id))


def iter_edges(graph, *, include_placeholders: bool = False) -> Iterator[EdgeRef]:
    """Yield every structural edge of the graph, in materialized column order.

    An edge with no column carries no structure. Set ``include_placeholders`` to
    see those too.
    """
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
    record = _require_edge(graph, edge_id)
    return Endpoints(source=_raw_side(record.src), target=_raw_side(record.tgt))


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
    record = _require_edge(graph, edge_id)
    if record.coeffs is None:
        return None
    return dict(record.coeffs)


def entity_row(graph, ref) -> int:
    """Return the row an entity occupies in the materialized matrix.

    A row is a position, so it belongs to one materialized matrix and to nothing
    else. Use this only to persist or to rebuild that matrix.
    """
    return int(graph._entities[_require_entity(graph, ref)].row_idx)


def edge_column(graph, edge_id: str) -> int:
    """Return the column an edge occupies in the materialized matrix.

    The result is ``-1`` when the edge carries no structure. A column is a
    position, so it belongs to one materialized matrix and to nothing else. Use
    this only to persist or to rebuild that matrix.
    """
    return int(_require_edge(graph, edge_id).col_idx)


def _side_keys(graph, side) -> frozenset:
    if side is None:
        return frozenset()
    if isinstance(side, frozenset):
        return frozenset(entity_key(graph, member) for member in side)
    return frozenset({entity_key(graph, side)})


def edge_endpoints(graph, edge_id: str) -> Endpoints:
    """Return the source side and the target side of one edge, as entity keys."""
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
    return key if graph._aspects != ('_',) else key[0]


_probe_form = endpoint_form


def _stored_directed(graph, record) -> bool:
    """Return directedness the way the adjacency traversal reads it."""
    return bool(record.directed if record.directed is not None else graph.directed)


def entity_edges(graph, ref, direction: str = 'both') -> tuple:
    """Return the ids of the edges that touch one entity, in column order.

    An undirected edge counts in both directions. A hyperedge counts on the side
    that holds the entity.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f'direction must be one of {DIRECTIONS}, got {direction!r}')
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
