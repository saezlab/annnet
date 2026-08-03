"""The slot-addressed canonical store.

The store holds the canonical state of a graph. An element is addressed by a
stable identity, which is an entity key or an edge id, or by a stable slot, which
is an integer that the store assigns on insert and frees on delete. The store
never renumbers a slot, so a delete touches only the deleted element.

Topology lives in the member lists. A member list is the incidence column of one
edge, so the member lists together are an incidence matrix in compressed sparse
column form, addressed by slot. One member list holds every edge kind.

**One entry per role.** A member entry records one role of one entity in one edge,
not one entity. An entity that takes two roles in one edge therefore appears
twice in that edge. This is what keeps a self-loop distinct from a boundary edge:
a self-loop holds two entries on one entity slot, and a boundary edge holds one.
Without it a directed self-loop would collapse to a single entry and look exactly
like a one-sided edge.

The store holds no matrix object and imports no matrix library. A matrix is
derived state, and ``_matrices`` builds it.

Only the mutation gateway writes the store. The derive layer and the query facade
read it. One structural clock rises on every write, and every derived structure
rebuilds when its recorded clock value differs.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .._session import logger

log = logger(__name__)

# Roles of a member inside one edge. The role says which side of the edge the
# member is on. The coefficient says how much, and the two are independent,
# because an explicit coefficient may carry any value.
SOURCE = 1
TARGET = -1
MEMBER = 0

# Entity kinds.
NODE = 0
EDGE_ENTITY = 1

# Edge kinds.
BINARY = 0
HYPER = 1
NODE_EDGE = 2
PLACEHOLDER = 3

# ``edge_directed`` holds a declared value, where this code means "inherit the
# default of the graph".
INHERIT = -1

# Sides an entity takes in one edge, as bits of the entity-to-edge index. An
# entity on both sides of the same edge is a self-loop, so the two bits are set
# together. The index carries them so that a traversal reads the side it wants
# instead of walking the member list of every incident edge.
ON_SOURCE = 0b01
ON_TARGET = 0b10
_SIDE_OF_ROLE = {SOURCE: ON_SOURCE, TARGET: ON_TARGET, MEMBER: ON_SOURCE}

_INITIAL_CAPACITY = 8


class MemberList(NamedTuple):
    """The member entries of one edge, as views into the store arrays.

    The three arrays line up. Entry ``i`` says that entity ``entities[i]`` takes
    role ``roles[i]`` in the edge with coefficient ``coefficients[i]``.
    """

    entities: np.ndarray
    coefficients: np.ndarray
    roles: np.ndarray


class Endpoints(NamedTuple):
    """The two sides of an edge, as sets of entity keys."""

    source: frozenset
    target: frozenset


def _grown(array: np.ndarray, target: int, fill=0) -> np.ndarray:
    """Return an array at least ``target`` long, doubling to amortize the cost."""
    if target <= array.size:
        return array
    size = max(_INITIAL_CAPACITY, array.size)
    while size < target:
        size *= 2
    out = np.full(size, fill, dtype=array.dtype)
    out[: array.size] = array
    return out


class CoreState:
    """The canonical store of one graph.

    Every public method here either reads the store or performs one canonical
    write. A write advances :attr:`structure_version`.
    """

    def __init__(self, *, directed=None, aspects=('_',)):
        self.directed = directed
        self.aspects = tuple(aspects)

        # Identity, both ways.
        self._entity_slot: dict[tuple, int] = {}
        self._entity_key: list = []
        self._edge_slot: dict[str, int] = {}
        self._edge_id: list = []

        # Entity arrays, indexed by entity slot.
        self.entity_kind = np.zeros(_INITIAL_CAPACITY, dtype=np.uint8)

        # Edge arrays, indexed by edge slot.
        self.edge_kind = np.zeros(_INITIAL_CAPACITY, dtype=np.uint8)
        self.edge_directed = np.full(_INITIAL_CAPACITY, INHERIT, dtype=np.int8)
        self.edge_weight = np.ones(_INITIAL_CAPACITY, dtype=np.float64)
        self.edge_explicit = np.zeros(_INITIAL_CAPACITY, dtype=bool)

        # Member lists. Each edge owns a segment of the three pools. A delete
        # empties the segment and leaves a hole, which a compaction pass reclaims.
        self.member_start = np.zeros(_INITIAL_CAPACITY, dtype=np.int64)
        self.member_len = np.zeros(_INITIAL_CAPACITY, dtype=np.int32)
        self.member_ent = np.zeros(_INITIAL_CAPACITY, dtype=np.int64)
        self.member_coef = np.zeros(_INITIAL_CAPACITY, dtype=np.float64)
        self.member_role = np.zeros(_INITIAL_CAPACITY, dtype=np.int8)
        self._member_used = 0

        # Rare per-edge state, kept out of the hot arrays.
        self.edge_ml_kind: dict[int, object] = {}
        self.edge_ml_layers: dict[int, object] = {}
        self.edge_policy: dict[int, dict] = {}

        # Freelists. A freed slot is reused before the arrays grow.
        self.entity_free: list[int] = []
        self.edge_free: list[int] = []

        # A derived index, maintained rather than rebuilt, so that removing an
        # entity stays local. It is rebuildable from the member lists and it is
        # never authoritative.
        # entity slot -> {edge slot: (the sides that entity takes, the peer)}
        # The peer is the entity on the other entry of an edge that has exactly
        # two, and None when the edge has any other number.
        self._entity_edges: dict[int, dict[int, tuple[int, int | None]]] = {}

        # Slot lifecycle hooks. A freed slot must hold a null in every structure
        # that is indexed by slot, and the store does not know what those are. So it
        # announces the free and the attribute layer clears its own cells.
        self.entity_freed_hooks: list = []
        self.edge_freed_hooks: list = []

        # Tier 3, the one clock.
        self.structure_version = 0

        # The append log lets a cached matrix survive an append. It holds the edge
        # slots appended at the frontier since the last write that was not such an
        # append, and the clock value that log starts from.
        self.append_log: list[int] = []
        self.append_log_from_version = 0

        # The structural edges and their columns, against the clock they were
        # read at. A placeholder edge occupies no column, so the column of an
        # edge is its position among the others, and working that out per call
        # would cost a walk over every edge.
        self._structural_cache = None

    # -- clock ------------------------------------------------------------

    def _bump(self) -> int:
        self.structure_version += 1
        return self.structure_version

    def _note_append(self, edge_slot: int) -> None:
        self.append_log.append(edge_slot)
        self._bump()

    def _note_change(self) -> None:
        self.append_log.clear()
        self.append_log_from_version = self._bump()

    # -- entities ---------------------------------------------------------

    @property
    def entity_count(self) -> int:
        """How many entities the store holds."""
        return len(self._entity_slot)

    @property
    def entity_capacity(self) -> int:
        """How many entity slots the arrays can address."""
        return len(self._entity_key)

    def add_entity(self, key: tuple, kind: int = NODE) -> int:
        """Add one entity and return its slot. An existing identity keeps its slot."""
        existing = self._entity_slot.get(key)
        if existing is not None:
            return existing

        if self.entity_free:
            slot = self.entity_free.pop()
            self._entity_key[slot] = key
        else:
            slot = len(self._entity_key)
            self._entity_key.append(key)
            self.entity_kind = _grown(self.entity_kind, slot + 1)

        self._entity_slot[key] = slot
        self.entity_kind[slot] = kind
        self._entity_edges[slot] = {}
        self._note_change()
        return slot

    def remove_entity(self, key: tuple) -> list[str]:
        """Remove one entity and return the ids of the edges it leaves dangling.

        The store never keeps a member that names no entity, so the caller has to
        deal with the returned edges. Removing the entity itself touches no other
        entity and no other address.
        """
        slot = self._entity_slot.get(key)
        if slot is None:
            raise KeyError(f'Unknown entity: {key!r}')

        dangling = [self._edge_id[edge_slot] for edge_slot in sorted(self._entity_edges[slot])]
        del self._entity_slot[key]
        self._entity_key[slot] = None
        self.entity_kind[slot] = NODE
        del self._entity_edges[slot]
        self.entity_free.append(slot)
        for hook in self.entity_freed_hooks:
            hook(slot, key)
        self._note_change()
        return dangling

    def entity_slot(self, key: tuple):
        """Return the slot of an entity key, or None when the store has no such entity."""
        return self._entity_slot.get(key)

    def entity_key(self, slot: int):
        """Return the key of an entity slot, or None when the slot is free."""
        if 0 <= slot < len(self._entity_key):
            return self._entity_key[slot]
        return None

    def live_entities(self):
        """Yield every live entity as a ``(slot, key)`` pair, in slot order."""
        for slot, key in enumerate(self._entity_key):
            if key is not None:
                yield slot, key

    def live_entity_slots(self) -> np.ndarray:
        """Return the live entity slots in slot order."""
        return np.fromiter(
            (slot for slot, key in enumerate(self._entity_key) if key is not None),
            dtype=np.int64,
            count=self.entity_count,
        )

    # -- edges ------------------------------------------------------------

    @property
    def edge_count(self) -> int:
        """How many edges the store holds."""
        return len(self._edge_slot)

    def add_edge(
        self,
        edge_id: str,
        members,
        *,
        kind: int = BINARY,
        directed=None,
        weight: float = 1.0,
        explicit_coefficients: bool = False,
        ml_kind=None,
        ml_layers=None,
    ) -> int:
        """Add one edge with its member list and return its slot.

        ``members`` is a sequence of ``(entity_key, coefficient, role)``. One entry
        per role, so an entity that takes two roles appears twice.
        """
        if edge_id in self._edge_slot:
            raise KeyError(f'Duplicate edge id: {edge_id!r}')

        entity_slots = []
        for entity_key, _coefficient, _role in members:
            slot = self._entity_slot.get(entity_key)
            if slot is None:
                raise KeyError(
                    f'Edge {edge_id!r} names an entity the store does not hold: {entity_key!r}'
                )
            entity_slots.append(slot)

        appended_at_frontier = not self.edge_free
        if self.edge_free:
            slot = self.edge_free.pop()
            self._edge_id[slot] = edge_id
        else:
            slot = len(self._edge_id)
            self._edge_id.append(edge_id)
            for name in ('edge_kind', 'edge_directed', 'edge_weight', 'edge_explicit'):
                setattr(self, name, _grown(getattr(self, name), slot + 1))
            self.member_start = _grown(self.member_start, slot + 1)
            self.member_len = _grown(self.member_len, slot + 1)

        self._edge_slot[edge_id] = slot
        self.edge_kind[slot] = kind
        self.edge_directed[slot] = INHERIT if directed is None else int(bool(directed))
        self.edge_weight[slot] = 1.0 if weight is None else weight
        self.edge_explicit[slot] = explicit_coefficients
        if ml_kind is not None:
            self.edge_ml_kind[slot] = ml_kind
        if ml_layers is not None:
            self.edge_ml_layers[slot] = ml_layers

        count = len(entity_slots)
        start = self._member_used
        needed = start + count
        self.member_ent = _grown(self.member_ent, needed)
        self.member_coef = _grown(self.member_coef, needed)
        self.member_role = _grown(self.member_role, needed)
        for offset, ((_key, coefficient, role), entity_slot) in enumerate(
            zip(members, entity_slots, strict=False)
        ):
            self.member_ent[start + offset] = entity_slot
            self.member_coef[start + offset] = coefficient
            self.member_role[start + offset] = role
            sides = self._entity_edges[entity_slot]
            held = sides.get(slot)
            side = _SIDE_OF_ROLE.get(role, ON_SOURCE)
            sides[slot] = (side if held is None else held[0] | side, None)
        # An edge that names two entries has one entry on the other side of each
        # of them, so each member can record its peer and a neighbour query needs
        # no member list at all. An edge with any other number cannot, and the
        # query falls back to reading its members.
        if count == 2:
            first, second = entity_slots
            self._entity_edges[first][slot] = (self._entity_edges[first][slot][0], second)
            self._entity_edges[second][slot] = (self._entity_edges[second][slot][0], first)
        self._member_used = needed
        self.member_start[slot] = start
        self.member_len[slot] = count

        if appended_at_frontier:
            self._note_append(slot)
        else:
            self._note_change()
        return slot

    def remove_edge(self, edge_id: str) -> None:
        """Remove one edge. No other edge changes its address or its member list."""
        slot = self._edge_slot.get(edge_id)
        if slot is None:
            raise KeyError(f'Unknown edge id: {edge_id!r}')

        for entity_slot in self.members(slot).entities:
            edges = self._entity_edges.get(int(entity_slot))
            if edges is not None:
                edges.pop(slot, None)

        del self._edge_slot[edge_id]
        self._edge_id[slot] = None
        self.member_len[slot] = 0
        self.edge_ml_kind.pop(slot, None)
        self.edge_ml_layers.pop(slot, None)
        self.edge_policy.pop(slot, None)
        self.edge_free.append(slot)
        for hook in self.edge_freed_hooks:
            hook(slot, edge_id)
        self._note_change()

    def structural_edges(self) -> list:
        """Return the ``(slot, edge_id)`` pairs that carry structure, in slot order.

        A placeholder edge is an id the graph knows before the edge exists. It
        holds no members and occupies no column, so it is not one of these.
        """
        return self._structural()[0]

    def structural_column(self, slot: int) -> int:
        """Return the column an edge slot occupies, or -1 when it carries none."""
        return self._structural()[1].get(slot, -1)

    def _structural(self):
        cached = self._structural_cache
        if cached is not None and cached[0] == self.structure_version:
            return cached[1], cached[2]
        kinds = self.edge_kind.tolist()
        pairs = [
            (slot, edge_id) for slot, edge_id in self.live_edges() if kinds[slot] != PLACEHOLDER
        ]
        columns = {slot: column for column, (slot, _id) in enumerate(pairs)}
        self._structural_cache = (self.structure_version, pairs, columns)
        return pairs, columns

    def edge_slot(self, edge_id: str):
        """Return the slot of an edge id, or None when the store has no such edge."""
        return self._edge_slot.get(edge_id)

    def edge_id(self, slot: int):
        """Return the id of an edge slot, or None when the slot is free."""
        if 0 <= slot < len(self._edge_id):
            return self._edge_id[slot]
        return None

    def live_edges(self):
        """Yield every live edge as a ``(slot, edge_id)`` pair, in slot order."""
        for slot, edge_id in enumerate(self._edge_id):
            if edge_id is not None:
                yield slot, edge_id

    def live_edge_ids(self) -> list[str]:
        """Return the live edge ids in slot order."""
        return [edge_id for _slot, edge_id in self.live_edges()]

    def live_edge_slots(self) -> np.ndarray:
        """Return the live edge slots in slot order."""
        return np.fromiter(
            (slot for slot, edge_id in enumerate(self._edge_id) if edge_id is not None),
            dtype=np.int64,
            count=self.edge_count,
        )

    # -- member lists -----------------------------------------------------

    def members(self, edge_slot: int) -> MemberList:
        """Return the member list of one edge, as views into the store arrays."""
        start = int(self.member_start[edge_slot])
        stop = start + int(self.member_len[edge_slot])
        return MemberList(
            entities=self.member_ent[start:stop],
            coefficients=self.member_coef[start:stop],
            roles=self.member_role[start:stop],
        )

    def member_count(self, edge_slot: int) -> int:
        """How many member entries one edge holds."""
        return int(self.member_len[edge_slot])

    def endpoints(self, edge_slot: int) -> Endpoints:
        """Return the two sides of one edge, as sets of entity keys.

        A member with no direction sits on the source side, which is the side that
        carries the positive coefficient of an edge without explicit ones.
        """
        # The two member slices are converted to Python lists in one step each.
        # Walking a numpy array element by element yields an array scalar per
        # entry, and this runs once per edge of every enumeration.
        start = int(self.member_start[edge_slot])
        stop = start + int(self.member_len[edge_slot])
        keys = self._entity_key
        source, target = [], []
        for entity_slot, role in zip(
            self.member_ent[start:stop].tolist(),
            self.member_role[start:stop].tolist(),
            strict=False,
        ):
            if role == TARGET:
                target.append(keys[entity_slot])
            else:
                source.append(keys[entity_slot])
        return Endpoints(frozenset(source), frozenset(target))

    def is_self_loop(self, edge_slot: int) -> bool:
        """Return True when one entity takes more than one role in this edge."""
        entities = self.members(edge_slot).entities
        return entities.size > np.unique(entities).size

    def is_boundary(self, edge_slot: int) -> bool:
        """Return True when the edge holds a single member entry.

        A boundary edge has one open side, so it names one entity once. The entry
        count separates it from a self-loop, which names one entity twice.
        """
        return self.member_count(edge_slot) == 1

    def is_directed(self, edge_slot: int) -> bool:
        """Return the directedness of one edge, falling back to the graph default."""
        declared = int(self.edge_directed[edge_slot])
        if declared != INHERIT:
            return bool(declared)
        return True if self.directed is None else bool(self.directed)

    def degree(self, key: tuple) -> int:
        """Return how many member entries name one entity.

        A member entry is one role, so a self-loop counts twice and a boundary
        edge counts once.
        """
        slot = self._entity_slot.get(key)
        if slot is None:
            raise KeyError(f'Unknown entity: {key!r}')
        total = 0
        for edge_slot in self._entity_edges[slot]:
            entities = self.members(edge_slot).entities
            total += int(np.count_nonzero(entities == slot))
        return total

    def entity_edge_slots(self, key: tuple) -> list[int]:
        """Return the slots of the edges that name one entity, in slot order."""
        slot = self._entity_slot.get(key)
        if slot is None:
            raise KeyError(f'Unknown entity: {key!r}')
        return sorted(self._entity_edges[slot])

    # -- maintenance ------------------------------------------------------

    @property
    def member_fragmentation(self) -> float:
        """The share of the member pools that freed segments still hold."""
        if self._member_used == 0:
            return 0.0
        live = int(self.member_len[self.live_edge_slots()].sum()) if self.edge_count else 0
        return 1.0 - live / self._member_used

    def compact_members(self) -> None:
        """Reclaim the holes that freed member segments leave behind.

        The pass rewrites every live segment into one contiguous block. It changes
        no slot and no identity, so it is safe at any time. It is never on a hot
        path.
        """
        slots = self.live_edge_slots()
        before = self._member_used
        total = int(self.member_len[slots].sum()) if slots.size else 0
        ent = np.zeros(max(_INITIAL_CAPACITY, total), dtype=np.int64)
        coef = np.zeros(max(_INITIAL_CAPACITY, total), dtype=np.float64)
        role = np.zeros(max(_INITIAL_CAPACITY, total), dtype=np.int8)
        cursor = 0
        for slot in slots:
            members = self.members(int(slot))
            width = members.entities.size
            ent[cursor : cursor + width] = members.entities
            coef[cursor : cursor + width] = members.coefficients
            role[cursor : cursor + width] = members.roles
            self.member_start[slot] = cursor
            cursor += width
        reclaimed = before - cursor
        self.member_ent, self.member_coef, self.member_role = ent, coef, role
        self._member_used = cursor
        self._note_change()
        log.info(
            'Compacted the member pools of a graph with %d edges and reclaimed %d entries.',
            self.edge_count,
            reclaimed,
        )

    def __repr__(self) -> str:
        return (
            f'CoreState(entities={self.entity_count}, edges={self.edge_count}, '
            f'version={self.structure_version})'
        )


# ---------------------------------------------------------------------------
# The bridge from the record store
# ---------------------------------------------------------------------------


def from_graph(graph) -> CoreState:
    """Build a slot store that holds the same graph as a record-backed one.

    The bridge reads the record store only through the structural query facade, so
    it depends on the boundary rather than on the layout behind it. It is the
    migration path for the new core, and it is what lets one graph be compared
    across the two store models.

    One shape changes on purpose. A directed self-loop keeps both of its roles
    here, where the record store collapsed them into one entry. That is the
    intended difference, and it is why a signed incidence column for a self-loop
    now sums to zero instead of holding one negative value.
    """
    from . import _structure as S

    state = CoreState(directed=graph.directed, aspects=graph._aspects)
    kind_of_entity = {S.NODE: NODE, S.EDGE_ENTITY: EDGE_ENTITY}
    kind_of_edge = {
        S.BINARY: BINARY,
        S.HYPER: HYPER,
        S.NODE_EDGE: NODE_EDGE,
        S.PLACEHOLDER: PLACEHOLDER,
    }

    for ref in S.iter_entities(graph):
        state.add_entity(ref.key, kind_of_entity.get(ref.kind, NODE))

    for edge in S.iter_edges(graph, include_placeholders=True):
        sides = S.edge_sides(graph, edge.id)
        coefficients = S.edge_coefficients(graph, edge.id)
        explicit = coefficients is not None
        weight = edge.weight

        members = members_from_sides(state, graph, sides, coefficients, weight, edge)

        state.add_edge(
            edge.id,
            members,
            kind=kind_of_edge.get(edge.kind, BINARY),
            directed=edge.declared_directed,
            weight=weight,
            explicit_coefficients=explicit,
            ml_kind=edge.ml_kind,
            ml_layers=edge.ml_layers,
        )
    return state


def members_from_sides(state, graph, sides, coefficients, weight, edge) -> list:
    """Return the member entries of one edge, given its two sides.

    One entry per role, so an entity on both sides appears twice. An endpoint that
    names no entity the store holds is left out, exactly as the incidence matrix
    leaves it out.

    Both the bridge from a record graph and the mutation gateway build member
    entries, and the rules for the role and the coefficient are subtle enough that
    they live here once.
    """
    members = []
    role = MEMBER if not sides.target else SOURCE
    for endpoint in sorted(sides.source, key=repr):
        key = _bridged_key(graph, endpoint)
        if key is None or state.entity_slot(key) is None:
            continue
        members.append(
            (key, _bridged_coefficient(coefficients, endpoint, weight, role, edge), role)
        )
    for endpoint in sorted(sides.target, key=repr):
        key = _bridged_key(graph, endpoint)
        if key is None or state.entity_slot(key) is None:
            continue
        members.append(
            (key, _bridged_coefficient(coefficients, endpoint, weight, TARGET, edge), TARGET)
        )
    return members


def _bridged_key(graph, endpoint):
    """Resolve a stored endpoint to an entity key, or None when it names none."""
    from . import _structure as S

    if S.is_entity_key(endpoint):
        return endpoint
    try:
        return S.entity_key(graph, endpoint)
    except (KeyError, ValueError, TypeError):
        return None


def _bridged_coefficient(coefficients, endpoint, weight, role, edge):
    """Return the coefficient one member entry carries.

    An edge with explicit coefficients keeps them. Otherwise the source side takes
    the weight and the target side takes the negated weight when the edge is
    directed, which is the rule the record store used.
    """
    if coefficients is not None:
        return float(coefficients.get(endpoint, 0.0))
    if role == TARGET and edge.directed:
        return -float(weight)
    return float(weight)
