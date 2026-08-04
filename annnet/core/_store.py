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

# Every array a store owns, named once so a copy cannot leave one behind.
_ARRAYS = (
    'entity_kind',
    'edge_kind',
    'edge_directed',
    'edge_weight',
    'edge_explicit',
    'member_start',
    'member_len',
    'member_ent',
    'member_coef',
    'member_role',
)


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

        # Identity, both ways.
        self._entity_slot: dict[tuple, int] = {}
        self._entity_key: list = []
        # Which slots one bare id stands for. A flat graph needs none, because an
        # id names exactly one entity there. A multilayer graph asks this on every
        # resolution of a bare id, and a scan over the entities would make that
        # cost the size of the graph.
        self._id_slots: dict[str, list] = {}
        self._aspects = tuple(aspects)
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
        self._rows_cache = None

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

    @property
    def aspects(self) -> tuple:
        """The aspects the graph declares, which set the form of an identity."""
        return self._aspects

    @aspects.setter
    def aspects(self, value) -> None:
        """Declare the aspects, and build the bare-id index when one is now needed.

        A flat graph keeps no such index, because an id names one entity there. A
        graph that declares aspects after it holds entities therefore has to index
        the entities it already holds.
        """
        value = tuple(value)
        was_flat = self._aspects == ('_',)
        self._aspects = value
        if value == ('_',):
            self._id_slots = {}
        elif was_flat:
            self._id_slots = {}
            for slot, key in self.live_entities():
                self._id_slots.setdefault(key[0], []).append(slot)

    def entity_slots_of_id(self, entity_id: str) -> list:
        """Return the slots a bare id stands for, in slot order.

        One in a flat graph, and one per layer the id lives in otherwise.
        """
        if self._aspects == ('_',):
            slot = self._entity_slot.get((entity_id, ('_',)))
            return [] if slot is None else [slot]
        return self._id_slots.get(entity_id, [])

    def entity_keys_of_id(self, entity_id: str) -> list:
        """Return the entity keys a bare id stands for, in slot order."""
        keys = self._entity_key
        return [keys[slot] for slot in self.entity_slots_of_id(entity_id)]

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
        if self._aspects != ('_',):
            self._id_slots.setdefault(key[0], []).append(slot)
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
        held = self._id_slots.get(key[0])
        if held is not None:
            held.remove(slot)
            if not held:
                del self._id_slots[key[0]]
        self.entity_free.append(slot)
        for hook in self.entity_freed_hooks:
            hook(slot, key)
        self._note_change()
        return dangling

    def rekey(self, mapping) -> None:
        """Move each entity a key-to-key map names, keeping the slot it holds.

        An identity changes and an address does not, so every member list, the
        incidence index and every matrix position survive untouched. The map is
        applied in two passes, so a set of keys may be permuted among themselves.

        A key the store does not hold is ignored, and so is a move onto a key the
        store still holds after the first pass, which would be two entities at one
        identity.
        """
        if not mapping:
            return
        moved = {}
        for old, new in mapping.items():
            if old == new:
                continue
            slot = self._entity_slot.pop(old, None)
            if slot is not None:
                moved[new] = slot
        for new, slot in moved.items():
            if new in self._entity_slot:
                raise KeyError(f'Rekeying an entity onto one the store holds: {new!r}')
            self._entity_slot[new] = slot
            self._entity_key[slot] = new
        if self._aspects != ('_',):
            self._id_slots = {}
            for slot, key in self.live_entities():
                self._id_slots.setdefault(key[0], []).append(slot)
        self._note_change()

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

    def _rows(self):
        """Return the row of each live slot and the slot of each row, against the clock.

        A row is the position of an entity among the live ones, which is the
        address a materialized matrix uses. Working one out per call would walk
        every entity, and building a matrix asks once per member of every edge.
        """
        cached = self._rows_cache
        if cached is not None and cached[0] == self.structure_version:
            return cached[1], cached[2]
        slots = [slot for slot, _key in self.live_entities()]
        rows = {slot: row for row, slot in enumerate(slots)}
        self._rows_cache = (self.structure_version, rows, slots)
        return rows, slots

    def entity_row(self, slot: int) -> int:
        """Return the row an entity slot occupies in a materialized matrix."""
        return self._rows()[0][slot]

    def entity_at_row(self, row: int):
        """Return the entity key at one row, or None when the row holds none."""
        slots = self._rows()[1]
        if not 0 <= row < len(slots):
            return None
        return self._entity_key[slots[row]]

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
        direction_policy=None,
    ) -> int:
        """Add one edge with its member list and return its slot.

        ``members`` is a sequence of ``(entity_key, coefficient, role)``. One entry
        per role, so an entity that takes two roles appears twice.
        """
        if edge_id in self._edge_slot:
            raise KeyError(f'Duplicate edge id: {edge_id!r}')

        entity_slots = self._member_slots(edge_id, members)

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
        if direction_policy is not None:
            self.edge_policy[slot] = direction_policy

        self._write_members(slot, members, entity_slots)

        if appended_at_frontier:
            self._note_append(slot)
        else:
            self._note_change()
        return slot

    def _member_slots(self, edge_id: str, members) -> list:
        """Return the entity slot of every member entry, or raise naming the first gap."""
        slots = []
        for entity_key, _coefficient, _role in members:
            slot = self._entity_slot.get(entity_key)
            if slot is None:
                raise KeyError(
                    f'Edge {edge_id!r} names an entity the store does not hold: {entity_key!r}'
                )
            slots.append(slot)
        return slots

    def _write_members(self, slot: int, members, entity_slots) -> None:
        """Give one edge slot a member segment, and index every entry it holds.

        The segment is always appended at the frontier of the pools, because a
        list written again may hold a different number of entries. What the edge
        held before is the caller's to unlink.
        """
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

    def _link_members(self, slot: int) -> None:
        """Record the side one edge gives every entity it names.

        This reads the member segment the edge already holds, so it is what a
        write that changes a role in place uses. A write that appends a segment
        indexes it as it goes, because that costs no second pass.
        """
        start = int(self.member_start[slot])
        stop = start + int(self.member_len[slot])
        entity_slots = self.member_ent[start:stop].tolist()
        roles = self.member_role[start:stop].tolist()
        for entity_slot, role in zip(entity_slots, roles, strict=False):
            sides = self._entity_edges[entity_slot]
            held = sides.get(slot)
            side = _SIDE_OF_ROLE.get(role, ON_SOURCE)
            sides[slot] = (side if held is None else held[0] | side, None)
        if len(entity_slots) == 2:
            first, second = entity_slots
            self._entity_edges[first][slot] = (self._entity_edges[first][slot][0], second)
            self._entity_edges[second][slot] = (self._entity_edges[second][slot][0], first)

    def _unlink_members(self, slot: int) -> None:
        """Drop one edge out of the incidence index of every entity it names."""
        start = int(self.member_start[slot])
        stop = start + int(self.member_len[slot])
        for entity_slot in self.member_ent[start:stop].tolist():
            edges = self._entity_edges.get(entity_slot)
            if edges is not None:
                edges.pop(slot, None)

    def _require_edge_slot(self, edge_id: str) -> int:
        slot = self._edge_slot.get(edge_id)
        if slot is None:
            raise KeyError(f'Unknown edge id: {edge_id!r}')
        return slot

    def remove_edge(self, edge_id: str) -> None:
        """Remove one edge. No other edge changes its address or its member list."""
        slot = self._require_edge_slot(edge_id)
        self._unlink_members(slot)

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

    # -- changing one edge ------------------------------------------------
    # A write below changes one field of one edge and leaves its slot, its
    # identity and every other edge alone. Removing the edge and adding it again
    # would give the same answer, and it would pay for a member list the edge
    # already holds.

    def set_edge_kind(self, edge_id: str, kind: int) -> None:
        """Set the kind of one edge."""
        self.edge_kind[self._require_edge_slot(edge_id)] = kind
        self._note_change()

    def set_edge_ml_kind(self, edge_id: str, ml_kind) -> None:
        """Set the multilayer role of one edge, or clear it with ``None``."""
        slot = self._require_edge_slot(edge_id)
        if ml_kind is None:
            self.edge_ml_kind.pop(slot, None)
        else:
            self.edge_ml_kind[slot] = ml_kind
        self._note_change()

    def set_edge_ml_layers(self, edge_id: str, ml_layers) -> None:
        """Set the layers one edge runs between, or clear them with ``None``."""
        slot = self._require_edge_slot(edge_id)
        if ml_layers is None:
            self.edge_ml_layers.pop(slot, None)
        else:
            self.edge_ml_layers[slot] = ml_layers
        self._note_change()

    def set_edge_policy(self, edge_id: str, policy) -> None:
        """Attach a flexible-direction policy to one edge, or clear it with ``None``."""
        slot = self._require_edge_slot(edge_id)
        if policy is None:
            self.edge_policy.pop(slot, None)
        else:
            self.edge_policy[slot] = policy
        self._note_change()

    def set_edge_directed(self, edge_id: str, directed) -> None:
        """Declare the directedness of one edge, and derive its coefficients again.

        An edge that carries no explicit coefficients takes its member
        coefficients from its weight and its directedness, so changing one
        changes the other. ``None`` means the edge inherits the graph default.
        """
        slot = self._require_edge_slot(edge_id)
        self.edge_directed[slot] = INHERIT if directed is None else int(bool(directed))
        self._derive_coefficients(slot)
        self._note_change()

    def set_edge_weight(self, edge_id: str, weight) -> None:
        """Set the weight of one edge, and derive its coefficients again."""
        slot = self._require_edge_slot(edge_id)
        self.edge_weight[slot] = 1.0 if weight is None else weight
        self._derive_coefficients(slot)
        self._note_change()

    def set_edge_coefficients(self, edge_id: str, coefficients) -> None:
        """Give the members of one edge the coefficients a map names.

        The map is the whole column, so a member it leaves out takes zero. It
        may key an entity by its key or by its bare id. An entity that takes two
        roles in the edge takes the one value in both, which is what a column
        keyed by entity can say.

        The edge then carries explicit coefficients, so nothing derives them
        from its weight again.
        """
        slot = self._require_edge_slot(edge_id)
        start = int(self.member_start[slot])
        stop = start + int(self.member_len[slot])
        keys = self._entity_key
        for position in range(start, stop):
            key = keys[int(self.member_ent[position])]
            value = coefficients.get(key)
            if value is None:
                value = coefficients.get(key[0], 0.0)
            self.member_coef[position] = float(value)
        self.edge_explicit[slot] = True
        self._note_change()

    def set_edge_explicit(self, edge_id: str, explicit: bool) -> None:
        """Say whether one edge states its own coefficients.

        An edge that stops stating them derives them from its weight and its
        directedness again, which is what the coefficients of an edge mean when
        it states none.
        """
        slot = self._require_edge_slot(edge_id)
        self.edge_explicit[slot] = explicit
        self._derive_coefficients(slot)
        self._note_change()

    def reverse_edge(self, edge_id: str) -> None:
        """Swap the two sides of one edge.

        Every entry on the target side moves to the source side, and every other
        entry moves to the target side. An edge left with nothing on its target
        side carries the plain role instead, which is the role a member takes in
        an edge with one side.

        An edge that states its own coefficients keeps them, because a
        coefficient belongs to an entity and not to a side. Otherwise they are
        derived again, so the new source side carries the weight.
        """
        slot = self._require_edge_slot(edge_id)
        start = int(self.member_start[slot])
        stop = start + int(self.member_len[slot])
        # One edge holds a handful of entries, so this walks them in Python. A
        # numpy pass over two of them costs more than the walk.
        roles = self.member_role[start:stop].tolist()
        source_role = MEMBER if all(role == TARGET for role in roles) else SOURCE
        self._unlink_members(slot)
        self.member_role[start:stop] = [source_role if role == TARGET else TARGET for role in roles]
        self._link_members(slot)
        self._derive_coefficients(slot)
        self._note_change()

    def merge_sides(self, edge_id: str) -> None:
        """Give every member of one edge the plain role, dropping the two sides.

        An entity on both sides becomes one member of the result, exactly as the
        union of the two sides holds it once. The coefficients are derived
        again, because an edge with one side has no negated side.
        """
        slot = self._require_edge_slot(edge_id)
        members = self.members(slot)
        keys = self._entity_key
        weight = float(self.edge_weight[slot])
        merged = {}
        for entity_slot in members.entities.tolist():
            key = keys[entity_slot]
            if key not in merged:
                merged[key] = (key, weight, MEMBER)
        self.replace_members(edge_id, list(merged.values()))

    def replace_members(self, edge_id: str, members) -> None:
        """Give one edge a new member list, and leave everything else it holds.

        The edge keeps its slot, its identity, its kind, its weight and its
        policy. The old segment is dropped and a new one is appended, because a
        list written again may hold a different number of entries.
        """
        slot = self._require_edge_slot(edge_id)
        entity_slots = self._member_slots(edge_id, members)
        self._unlink_members(slot)
        self._write_members(slot, members, entity_slots)
        self._note_change()

    def _derive_coefficients(self, slot: int) -> None:
        """Take the member coefficients of one edge from its weight and direction.

        An edge that states its own coefficients keeps them. Otherwise the source
        side carries the weight and the target side carries it negated when the
        edge is directed.
        """
        if bool(self.edge_explicit[slot]):
            return
        start = int(self.member_start[slot])
        stop = start + int(self.member_len[slot])
        weight = float(self.edge_weight[slot])
        if not self.is_directed(slot):
            self.member_coef[start:stop] = weight
            return
        negated = -weight
        self.member_coef[start:stop] = [
            negated if role == TARGET else weight for role in self.member_role[start:stop].tolist()
        ]

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

    def endpoints(self, edge_slot: int, *, bare: bool = False) -> Endpoints:
        """Return the two sides of one edge, as sets of entity keys.

        A member with no direction sits on the source side, which is the side that
        carries the positive coefficient of an edge without explicit ones.

        Set ``bare`` for the id of each entity rather than its key, which is the
        form a flat graph shows. The projection happens inside the one walk. Doing
        it over the result would build two more sets on a path that runs once per
        edge of every enumeration.
        """
        # The two member slices are converted to Python lists in one step each.
        # Walking a numpy array element by element yields an array scalar per
        # entry, and this runs once per edge of every enumeration.
        start = int(self.member_start[edge_slot])
        stop = start + int(self.member_len[edge_slot])
        keys = self._entity_key
        entity_slots = self.member_ent[start:stop].tolist()
        roles = self.member_role[start:stop].tolist()
        source, target = [], []
        if bare:
            for entity_slot, role in zip(entity_slots, roles, strict=False):
                if role == TARGET:
                    target.append(keys[entity_slot][0])
                else:
                    source.append(keys[entity_slot][0])
        else:
            for entity_slot, role in zip(entity_slots, roles, strict=False):
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

    def select(self, entity_keys, edge_ids, *, weights=None) -> CoreState:
        """Return a store holding only these entities and these edges, in this order.

        The result is a new store, so it numbers its slots from zero in the order
        given. A member entry whose entity the selection leaves out is left out
        with it, exactly as the incidence matrix leaves it out.

        ``weights`` names the edges the selection gives a new weight. An edge that
        carries no explicit coefficients derives its member coefficients from its
        weight, so those are derived again from the new one.
        """
        other = CoreState(directed=self.directed, aspects=self._aspects)
        for key in entity_keys:
            slot = self._entity_slot.get(key)
            other.add_entity(key, NODE if slot is None else int(self.entity_kind[slot]))
        weights = weights or {}
        for edge_id in edge_ids:
            slot = self._edge_slot.get(edge_id)
            if slot is None:
                continue
            weight = float(weights.get(edge_id, self.edge_weight[slot]))
            declared = int(self.edge_directed[slot])
            other.add_edge(
                edge_id,
                self._selected_members(slot, other, weight),
                kind=int(self.edge_kind[slot]),
                directed=None if declared == INHERIT else bool(declared),
                weight=weight,
                explicit_coefficients=bool(self.edge_explicit[slot]),
                ml_kind=self.edge_ml_kind.get(slot),
                ml_layers=self.edge_ml_layers.get(slot),
                direction_policy=self.edge_policy.get(slot),
            )
        return other

    def _selected_members(self, slot: int, other: CoreState, weight: float) -> list:
        """Return the member entries of one edge that a selection keeps."""
        members = self.members(slot)
        explicit = bool(self.edge_explicit[slot])
        directed = self.is_directed(slot)
        kept = []
        for entity_slot, coefficient, role in zip(
            members.entities, members.coefficients, members.roles, strict=True
        ):
            key = self._entity_key[int(entity_slot)]
            if key is None or other.entity_slot(key) is None:
                continue
            if not explicit:
                coefficient = -weight if role == TARGET and directed else weight
            kept.append((key, float(coefficient), int(role)))
        return kept

    def copy(self) -> CoreState:
        """Return a store that holds the same graph, sharing nothing with this one.

        Every slot keeps its address, so a position taken from one store still
        addresses the same element in the other. The arrays are copied, and so is
        every map and every list, down to the per-edge dictionaries.

        The slot-freed hooks are not carried over. A hook belongs to the
        attribute layer of one graph, and the copy has its own.
        """
        other = CoreState.__new__(CoreState)
        other.directed = self.directed
        other._aspects = self._aspects

        other._entity_slot = dict(self._entity_slot)
        other._entity_key = list(self._entity_key)
        other._id_slots = {key: list(slots) for key, slots in self._id_slots.items()}
        other._edge_slot = dict(self._edge_slot)
        other._edge_id = list(self._edge_id)

        for name in _ARRAYS:
            setattr(other, name, getattr(self, name).copy())
        other._member_used = self._member_used

        other.edge_ml_kind = dict(self.edge_ml_kind)
        other.edge_ml_layers = dict(self.edge_ml_layers)
        other.edge_policy = {slot: dict(policy) for slot, policy in self.edge_policy.items()}

        other.entity_free = list(self.entity_free)
        other.edge_free = list(self.edge_free)
        other._entity_edges = {slot: dict(edges) for slot, edges in self._entity_edges.items()}

        other.entity_freed_hooks = []
        other.edge_freed_hooks = []

        other.structure_version = self.structure_version
        other.append_log = list(self.append_log)
        other.append_log_from_version = self.append_log_from_version
        other._structural_cache = None
        other._rows_cache = None
        return other

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
    S = _facade()
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
    """Return the member entries of one edge, given its two sides as a record holds them."""
    return members_from_endpoints(
        state, graph, sides.source, sides.target, coefficients, weight, edge.directed
    )


def members_from_endpoints(state, graph, source, target, coefficients, weight, directed) -> list:
    """Return the member entries of one edge, given the endpoints on each side.

    One entry per role, so an entity on both sides appears twice. An endpoint that
    names no entity the store holds is left out, exactly as the incidence matrix
    leaves it out.

    An edge with explicit coefficients keeps them, and the map may name an
    endpoint it does not carry, which takes zero. Otherwise the source side takes
    the weight and the target side takes the negated weight when the edge is
    directed.

    Both the bridge from a record graph and the mutation gateway build member
    entries, and the rules for the role and the coefficient are subtle enough that
    they live here once.
    """
    weight = float(weight)
    members = []
    for side, role in ((source, SOURCE if target else MEMBER), (target, TARGET)):
        derived = -weight if role == TARGET and directed else weight
        for endpoint in _ordered(side):
            key = _bridged_key(graph, endpoint)
            if key is None or state.entity_slot(key) is None:
                continue
            coefficient = (
                derived if coefficients is None else float(coefficients.get(endpoint, 0.0))
            )
            members.append((key, coefficient, role))
    return members


def _ordered(side):
    """Return one side of an edge in a stable order.

    A member list is compared entry by entry, so a set has to be walked in the
    same order every time. A side of one needs no sort, and that is the shape of
    every binary edge, which is what a bulk load is made of.
    """
    return side if len(side) < 2 else sorted(side, key=repr)


_FACADE = None


def _facade():
    """Return the query facade, bound on first use.

    The facade imports this module, so this module cannot import it at import
    time. Binding it once keeps an import statement off a path that runs once per
    member of every edge written.
    """
    global _FACADE
    if _FACADE is None:
        from . import _structure

        _FACADE = _structure
    return _FACADE


def _bridged_key(graph, endpoint):
    """Resolve a stored endpoint to an entity key, or None when it names none."""
    S = _facade()
    if S.is_entity_key(endpoint):
        return endpoint
    try:
        return S.entity_key(graph, endpoint)
    except (KeyError, ValueError, TypeError):
        return None
