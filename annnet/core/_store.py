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
from itertools import chain, repeat

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

# Below this many edges a bulk write costs more than the single writes it
# replaces. Its fixed part — one transposition of the batch and thirteen array
# assignments — is about three single writes, and what it saves is about half of
# one per edge.
_BULK_MINIMUM = 8

# How many edges a caller should hold back before it calls the bulk write.
#
# What a bulk write saves — the growth of ten arrays and the interpreted write of
# every cell — is amortized within a few dozen edges. What a larger batch costs
# is the garbage collector. A spec and its member entries are tracked
# containers, and a batch that survives one collection of the youngest
# generation is promoted, so every collection after it scans the batch again.
# The default threshold of that generation is 700 allocations, so a batch of this
# size is collected where it stands. A load of 25 600 edges written in one batch
# instead spent 50 ms in the collector, which is the whole of what the bulk write
# saved.
BULK_CHUNK = 128

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


class EdgeSpec(NamedTuple):
    """One edge, in the form a bulk write takes it.

    The fields are those of :meth:`CoreState.add_edge` and so are the defaults,
    so a caller that carries none of the rare ones names two of the nine.

    A bulk write reads its batch by transposing it, so what it needs of a spec is
    that it is a tuple of these nine fields in this order. This class is where
    they are named and where a caller reads what they mean. A caller on a load
    path builds the plain tuple instead, because naming the fields costs four
    times what the tuple does and a load builds one per edge.
    """

    id: str
    members: tuple
    kind: int = BINARY
    directed: object = None
    weight: float = 1.0
    explicit_coefficients: bool = False
    ml_kind: object = None
    ml_layers: object = None
    direction_policy: object = None


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


def _first_repeated(ids, held) -> str | None:
    """Return the first id a batch repeats, or that a store already holds.

    A bulk write finds out that a batch carries one with two set operations. This
    walk names which one, and it runs only to raise.
    """
    seen = set()
    for edge_id in ids:
        if edge_id in held or edge_id in seen:
            return edge_id
        seen.add(edge_id)
    return None


def _edge_of_member(lengths, position: int) -> int:
    """Return the index of the edge whose member segment holds one flat position.

    A bulk write flattens every member list into one sequence, so a bad entry is
    found by its position in that sequence. This names the edge it came from, and
    it runs only to raise.
    """
    for index, width in enumerate(lengths):
        position -= width
        if position < 0:
            return index
    return len(lengths) - 1


def _as_indexes(slots) -> list:
    """Return ``slots`` as a list of Python integers, whatever it arrives as.

    A numpy array indexes a Python list only through its scalars, and converting
    the whole array once costs far less than converting one element at a time.
    """
    return slots.tolist() if isinstance(slots, np.ndarray) else [int(slot) for slot in slots]


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

        # How many live entities are edge-entities. The node axis filters those
        # out, so a fast path over a node column has to know there are none, and
        # counting them is a pass over the kind array rather than one read. It is
        # maintained where an entity is allocated, freed, or changes kind.
        self._edge_entity_count = 0

        # How many live edges are placeholders. A placeholder holds no members
        # and occupies no column, so every enumeration of the structural edges
        # leaves it out, and a column read of an intrinsic field addresses those
        # rather than the slots. The same reasoning as the counter above, on the
        # other axis.
        self._placeholder_edge_count = 0

        # The resolved direction and the kind of every edge slot, against the
        # clock. Neither is the array the store holds: a direction inherits the
        # default of the graph, and a hyperedge takes its own from the roles of
        # its members; a kind is a name where the array holds a code. Both are
        # one vectorized pass, so a read after a write pays microseconds where
        # walking the edges paid milliseconds.
        self._intrinsic_cache: dict = {}

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

        # Capacity hooks, the other half of the same idea. A structure indexed by
        # slot has to reach the new frontier before a reader can address it with
        # a range, so the store announces that the frontier moved and the
        # attribute layer grows its own columns. It fires once per write and only
        # when the frontier actually moved, so reusing a freed slot fires nothing.
        self.entity_capacity_hooks: list = []
        self.edge_capacity_hooks: list = []

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

        # Buffers over the three per-slot arrays a traversal reads one element
        # of per incident edge. See :meth:`slot_buffers`.
        self._slot_buffers = None

    def __getstate__(self):
        # A buffer cannot be pickled, and it is a cache of what the arrays
        # beside it already hold.
        state = self.__dict__.copy()
        state['_slot_buffers'] = None
        return state

    def slot_buffers(self):
        """Return buffers over ``entity_kind``, ``edge_kind`` and ``edge_directed``.

        A traversal reads one element of each per incident edge, and reading a
        numpy scalar costs about twice what reading the same byte through a
        buffer does. Building a buffer costs about as much again as the reads it
        saves over one walk, so the three are kept and rebuilt only when an array
        behind one is replaced — which growing it is the only way to do.
        """
        held = self._slot_buffers
        entity_kind = self.entity_kind
        edge_kind = self.edge_kind
        edge_directed = self.edge_directed
        if (
            held is None
            or held[0] is not entity_kind
            or held[1] is not edge_kind
            or held[2] is not edge_directed
        ):
            held = (
                entity_kind,
                edge_kind,
                edge_directed,
                memoryview(entity_kind),
                memoryview(edge_kind),
                memoryview(edge_directed),
            )
            self._slot_buffers = held
        return held[3], held[4], held[5]

    # -- clock ------------------------------------------------------------

    def _bump(self) -> int:
        self.structure_version += 1
        return self.structure_version

    def _note_append(self, edge_slot: int) -> None:
        self.append_log.append(edge_slot)
        self._bump()

    def _note_frontier_remove(self, edge_slot: int) -> None:
        """Log a removal of the highest edge slot as the mirror of an append.

        Such a removal takes the last column off a matrix and moves no row and
        no other column, so a cached matrix can drop its last column instead of
        being rebuilt. It is written as the bitwise complement of the slot, so
        one log carries both kinds of event and an append stays a plain slot.
        """
        self.append_log.append(~edge_slot)
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
    def edge_entity_count(self) -> int:
        """How many live entities are edge-entities.

        Maintained rather than counted, so that :attr:`node_axis_contiguous` is
        one read instead of a pass over :attr:`entity_kind`. The internal
        validator checks it against the kinds it counts.
        """
        return self._edge_entity_count

    def set_entity_kind(self, slot: int, kind: int) -> None:
        """Change the kind of an entity that already has a slot.

        Promoting an edge to an endpoint is the one thing that does this, and it
        moves the edge-entity count, so the change goes through here rather than
        into :attr:`entity_kind` directly.
        """
        held = int(self.entity_kind[slot])
        if held == kind:
            return
        self.entity_kind[slot] = kind
        if kind == EDGE_ENTITY:
            self._edge_entity_count += 1
        elif held == EDGE_ENTITY:
            self._edge_entity_count -= 1

    # -- contiguity, which is what a borrowing read is guarded by -----------

    @property
    def entity_slots_contiguous(self) -> bool:
        """Whether the live entity slots are exactly ``0 .. entity_count-1``.

        Two O(1) reads, and neither of them walks. The freelist is empty when no
        slot has been freed and not yet reused, and the live count equals the
        capacity when the key list holds no hole past the end. Together they say
        that indexing a slot-addressed array with a range gives the live
        elements, in order.

        :meth:`live_entity_slots` is deliberately not part of this. It is a
        generator over the key list and costs what the walk a caller of this
        predicate is trying to avoid costs.
        """
        return not self.entity_free and self.entity_count == len(self._entity_key)

    @property
    def edge_slots_contiguous(self) -> bool:
        """Whether the live edge slots are exactly ``0 .. edge_count-1``."""
        return not self.edge_free and self.edge_count == len(self._edge_id)

    @property
    def placeholder_edge_count(self) -> int:
        """How many live edges hold a name and no structure."""
        return self._placeholder_edge_count

    @property
    def edge_axis_contiguous(self) -> bool:
        """Whether one structural edge sits in each edge slot, in order.

        The edge axis of the query facade is the structural edges, which is not
        every live edge: a placeholder holds an id the graph knows before the
        edge exists, occupies no column, and is left out of every enumeration.
        So a slice over the slots fits only when the store holds none, which is
        one read because they are counted as they arrive.
        """
        return self._placeholder_edge_count == 0 and self.edge_slots_contiguous

    @property
    def node_axis_contiguous(self) -> bool:
        """Whether one node of the node axis sits in each entity slot, in order.

        The node axis is not the entity axis, so it asks two questions beyond
        :attr:`entity_slots_contiguous`. A multilayer graph holds one entity per
        layer a node lives in and shows the bare id once, so its rows are fewer
        than its slots. An edge-entity is an entity the node axis leaves out.
        Both are O(1) here: the aspects are a tuple and the edge-entities are
        counted as they arrive.
        """
        return (
            self._aspects == ('_',)
            and self._edge_entity_count == 0
            and self.entity_slots_contiguous
        )

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
            for hook in self.entity_capacity_hooks:
                hook(slot + 1)

        self._entity_slot[key] = slot
        self.entity_kind[slot] = kind
        if kind == EDGE_ENTITY:
            self._edge_entity_count += 1
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
        if int(self.entity_kind[slot]) == EDGE_ENTITY:
            self._edge_entity_count -= 1
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

    def entity_keys_at(self, slots) -> list:
        """Return the key of each of ``slots``, in the order given.

        Building a matrix asks for every row at once, so this answers them
        together rather than one call per entity.
        """
        return list(map(self._entity_key.__getitem__, _as_indexes(slots)))

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
        """Return the live entity slots in slot order.

        A contiguous store answers with a range, which numpy builds in one call.
        Otherwise the slots have to be found, and that is a walk over the key
        list in Python — 2.2 milliseconds at 100 000 entities, which is why the
        contiguity predicate exists and why nothing that has to stay O(1) may
        call this.
        """
        if self.entity_slots_contiguous:
            return np.arange(self.entity_count, dtype=np.int64)
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
            for hook in self.edge_capacity_hooks:
                hook(slot + 1)

        self._edge_slot[edge_id] = slot
        self.edge_kind[slot] = kind
        if kind == PLACEHOLDER:
            self._placeholder_edge_count += 1
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

    def add_edges(self, specs) -> list[int]:
        """Add many edges in one pass and return the slot of each, in order.

        Each of ``specs`` is an :class:`EdgeSpec`, or any tuple of its nine
        fields. The result is the store the same edges added one at a time would
        give. What differs is the cost: every array grows once rather than once
        per edge, every member pool takes one assignment rather than one per
        entry, and what is left per edge is the two dictionaries an identity
        writes and the incidence index its members touch.

        A spec that names an entity the store does not hold, or an id it already
        holds, raises before anything is written, so a bad batch leaves the store
        as it was — however many edges it names, and whichever chunk the bad one
        falls in.

        **A batch larger than :data:`BULK_CHUNK` is written in chunks of it**,
        for the reason recorded there. The limit is kept here rather than by the
        callers, so a caller that submits work in any shape gets what a caller
        that respects the limit gets: the same slots, the same arrays, the same
        clock and the same append log.
        """
        specs = list(specs) if not isinstance(specs, list) else specs
        if len(specs) <= BULK_CHUNK:
            return self._add_edges(specs)
        # Every check the chunks would make, made once over the whole batch, so
        # that a bad spec in the last chunk cannot leave the first ones written.
        self._check_batch(specs)
        slots: list[int] = []
        for start in range(0, len(specs), BULK_CHUNK):
            slots.extend(self._add_edges(specs[start : start + BULK_CHUNK]))
        return slots

    def _check_batch(self, specs) -> None:
        """Raise if any spec of a batch names a duplicate id or a missing entity."""
        ids = [spec[0] for spec in specs]
        seen = set(ids)
        if len(seen) != len(ids) or not self._edge_slot.keys().isdisjoint(seen):
            raise KeyError(f'Duplicate edge id: {_first_repeated(ids, self._edge_slot)!r}')
        entity_slot = self._entity_slot
        for edge_id, members, *_rest in specs:
            for entity_key, _coefficient, _role in members:
                if entity_key not in entity_slot:
                    raise KeyError(
                        f'Edge {edge_id!r} names an entity the store does not hold: {entity_key!r}'
                    )

    def _add_edges(self, specs: list) -> list[int]:
        """Write one batch, of at most :data:`BULK_CHUNK` edges."""
        if not specs:
            return []
        count = len(specs)
        if count < _BULK_MINIMUM:
            return self._add_edges_singly(specs)

        entity_slot = self._entity_slot
        entity_edges = self._entity_edges
        edge_slot = self._edge_slot
        edge_ids = self._edge_id
        side_of_role = _SIDE_OF_ROLE.get

        # The first pass reads the specs and resolves every member. It writes
        # nothing, so the raise a bad spec earns costs the caller nothing else.
        #
        # A spec is a tuple, so one transposition gives every field of every edge
        # as a column, and a member list is a tuple of tuples, so one chain and
        # one more transposition give the three member pools. Both happen inside
        # the interpreter, where a loop over the specs would spend a bytecode per
        # field of every edge.
        ids, member_lists, kinds, directions, weights, explicit, ml_kinds, ml_layers, policies = (
            zip(*specs, strict=True)
        )
        # ``edge_slot.keys().isdisjoint(seen)`` and not ``seen.isdisjoint(...)``.
        # A view asked this way walks the shorter of the two; a set asked it
        # walks the whole argument, which is the store.
        seen = set(ids)
        if len(seen) != count or not edge_slot.keys().isdisjoint(seen):
            raise KeyError(f'Duplicate edge id: {_first_repeated(ids, edge_slot)!r}')

        declared_directions = (
            [INHERIT if value is None else int(bool(value)) for value in directions]
            if (None in directions)
            else list(directions)
        )
        declared_weights = (
            [1.0 if value is None else value for value in weights]
            if (None in weights)
            else list(weights)
        )

        lengths = list(map(len, member_lists))
        flat = list(chain.from_iterable(member_lists))
        if flat:
            keys, coefficients, roles = zip(*flat, strict=True)
            resolved = list(map(entity_slot.get, keys))
            if None in resolved:
                gap = resolved.index(None)
                raise KeyError(
                    f'Edge {ids[_edge_of_member(lengths, gap)]!r} names an entity the '
                    f'store does not hold: {keys[gap]!r}'
                )
            # Every one of them resolved, which the raise above is what makes
            # true. Saying so here is what keeps the walks below reading a slot
            # rather than a slot that might be missing.
            entities: list[int] = [slot for slot in resolved if slot is not None]
        else:
            coefficients, roles, entities = (), (), []
        cursor = self._member_used + len(flat)

        # The second pass gives every edge its slot. A freed slot is reused
        # before the frontier grows, as a single add does, so a batch after a
        # removal leaves no hole behind. A store that has never freed one takes
        # the run after the frontier, and then both maps take one update.
        free = self.edge_free
        frontier = frontier_before = len(edge_ids)
        if free:
            slots = []
            reused = 0
            for edge_id in ids:
                if free:
                    slot = free.pop()
                    edge_ids[slot] = edge_id
                    reused += 1
                else:
                    slot = frontier
                    frontier += 1
                    edge_ids.append(edge_id)
                edge_slot[edge_id] = slot
                slots.append(slot)
        else:
            slots = list(range(frontier, frontier + count))
            frontier += count
            edge_ids.extend(ids)
            edge_slot.update(zip(ids, slots, strict=True))
            reused = 0

        # The third pass puts every member in the incidence index. The side each
        # of them takes is read for the whole batch at once, because the walk
        # below is where the remaining cost of a bulk write lives.
        sides_of = list(map(side_of_role, roles, repeat(ON_SOURCE)))
        if lengths.count(2) == count:
            # Two entries are one on each side of each other, so each member
            # records its peer and a neighbour query needs no member list. That
            # is the shape of a binary edge, which is what a bulk load is made
            # of, so the whole batch is walked as pairs.
            for slot, first, second, side, peer_side in zip(
                slots, entities[0::2], entities[1::2], sides_of[0::2], sides_of[1::2], strict=True
            ):
                if first == second:
                    entity_edges[first][slot] = (side | peer_side, first)
                else:
                    entity_edges[first][slot] = (side, second)
                    entity_edges[second][slot] = (peer_side, first)
        else:
            position = 0
            for slot, width in zip(slots, lengths, strict=True):
                stop = position + width
                if width == 2:
                    first, second = entities[position], entities[position + 1]
                    side, peer_side = sides_of[position], sides_of[position + 1]
                    if first == second:
                        entity_edges[first][slot] = (side | peer_side, first)
                    else:
                        entity_edges[first][slot] = (side, second)
                        entity_edges[second][slot] = (peer_side, first)
                else:
                    for offset in range(position, stop):
                        sides = entity_edges[entities[offset]]
                        held = sides.get(slot)
                        side = sides_of[offset]
                        sides[slot] = (side if held is None else held[0] | side, None)
                position = stop

        # The arrays, each grown once and written once. A member segment starts
        # where the one before it ends, so the starts are a running sum of the
        # widths and no loop has to carry one.
        # One announcement for the whole batch, which is what makes eager growth
        # of the attribute columns cost a bulk load one resize per column rather
        # than one per edge.
        if frontier > frontier_before:
            for hook in self.edge_capacity_hooks:
                hook(frontier)

        base = self._member_used
        index = np.fromiter(slots, dtype=np.int64, count=count)
        widths = np.fromiter(lengths, dtype=np.int64, count=count)
        for name in ('edge_kind', 'edge_directed', 'edge_weight', 'edge_explicit'):
            setattr(self, name, _grown(getattr(self, name), frontier))
        self.member_start = _grown(self.member_start, frontier)
        self.member_len = _grown(self.member_len, frontier)
        self.edge_kind[index] = kinds
        self._placeholder_edge_count += kinds.count(PLACEHOLDER)
        self.edge_directed[index] = declared_directions
        self.edge_weight[index] = declared_weights
        self.edge_explicit[index] = explicit
        self.member_start[index] = np.cumsum(widths) - widths + base
        self.member_len[index] = widths

        self.member_ent = _grown(self.member_ent, cursor)
        self.member_coef = _grown(self.member_coef, cursor)
        self.member_role = _grown(self.member_role, cursor)
        self.member_ent[base:cursor] = entities
        self.member_coef[base:cursor] = coefficients
        self.member_role[base:cursor] = roles
        self._member_used = cursor

        # The rare per-edge state, only when a spec in the batch carries any.
        if ml_kinds.count(None) < count:
            self.edge_ml_kind.update(
                (slot, value)
                for slot, value in zip(slots, ml_kinds, strict=True)
                if value is not None
            )
        if ml_layers.count(None) < count:
            self.edge_ml_layers.update(
                (slot, value)
                for slot, value in zip(slots, ml_layers, strict=True)
                if value is not None
            )
        if policies.count(None) < count:
            self.edge_policy.update(
                (slot, value)
                for slot, value in zip(slots, policies, strict=True)
                if value is not None
            )

        # One clock tick per edge, which is what the append log is counted
        # against. A freed slot is reused before the frontier grows, so the
        # reuses are the head of the batch and the log holds the tail after them
        # — the same log the same adds one at a time would leave.
        self.structure_version += count
        if reused:
            self.append_log.clear()
            self.append_log_from_version = self.structure_version - (count - reused)
        self.append_log.extend(slots[reused:])
        return slots

    def _add_edges_singly(self, specs) -> list[int]:
        """Add a batch too small to pay for the vectorized write, one edge at a time.

        The transposition, the ten array assignments and the three pool
        assignments of a bulk write cost about as much as three single writes
        whatever the batch holds, so a handful of edges are cheaper written the
        ordinary way. The checks come first, because a batch that raises leaves
        the store as it was however few edges it names.
        """
        held = self._edge_slot
        entity_slot = self._entity_slot
        seen = set()
        for edge_id, members, *_rest in specs:
            if edge_id in held or edge_id in seen:
                raise KeyError(f'Duplicate edge id: {edge_id!r}')
            seen.add(edge_id)
            for entity_key, _coefficient, _role in members:
                if entity_key not in entity_slot:
                    raise KeyError(
                        f'Edge {edge_id!r} names an entity the store does not hold: {entity_key!r}'
                    )
        return [
            self.add_edge(
                edge_id,
                members,
                kind=kind,
                directed=directed,
                weight=weight,
                explicit_coefficients=explicit,
                ml_kind=ml_kind,
                ml_layers=ml_layers,
                direction_policy=policy,
            )
            for edge_id, members, kind, directed, weight, explicit, ml_kind, ml_layers, policy in (
                specs
            )
        ]

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
        """Remove one edge. No other edge changes its address or its member list.

        **A removal of the highest slot gives the slot up rather than freeing
        it.** The slot list shrinks by one, so the frontier moves back to where
        it was before the edge was added, exactly as the freelist would have put
        the next edge there. What it buys is that the removal is then the mirror
        of an append: a cached matrix drops its last column instead of being
        rebuilt, and a store that has never freed a slot in the middle keeps the
        contiguity a borrowing read is guarded by.
        """
        slot = self._require_edge_slot(edge_id)
        self._unlink_members(slot)

        del self._edge_slot[edge_id]
        self._edge_id[slot] = None
        if int(self.edge_kind[slot]) == PLACEHOLDER:
            self._placeholder_edge_count -= 1
            self.edge_kind[slot] = BINARY
        self.member_len[slot] = 0
        self.edge_ml_kind.pop(slot, None)
        self.edge_ml_layers.pop(slot, None)
        self.edge_policy.pop(slot, None)
        removed_at_frontier = slot == len(self._edge_id) - 1
        if removed_at_frontier:
            self._edge_id.pop()
        else:
            self.edge_free.append(slot)
        for hook in self.edge_freed_hooks:
            hook(slot, edge_id)
        if removed_at_frontier:
            self._note_frontier_remove(slot)
        else:
            self._note_change()

    # -- changing one edge ------------------------------------------------
    # A write below changes one field of one edge and leaves its slot, its
    # identity and every other edge alone. Removing the edge and adding it again
    # would give the same answer, and it would pay for a member list the edge
    # already holds.

    def set_edge_kind(self, edge_id: str, kind: int) -> None:
        """Set the kind of one edge."""
        slot = self._require_edge_slot(edge_id)
        held = int(self.edge_kind[slot])
        self.edge_kind[slot] = kind
        if kind == PLACEHOLDER and held != PLACEHOLDER:
            self._placeholder_edge_count += 1
        elif held == PLACEHOLDER and kind != PLACEHOLDER:
            self._placeholder_edge_count -= 1
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

    def edge_ids_at(self, slots) -> list:
        """Return the id of each of ``slots``, in the order given.

        Building a matrix asks for every column at once, so this answers them
        together rather than one call per edge.
        """
        return list(map(self._edge_id.__getitem__, _as_indexes(slots)))

    def live_edge_slots(self) -> np.ndarray:
        """Return the live edge slots in slot order.

        A range when no slot has been freed, and a walk otherwise. See
        :meth:`live_entity_slots` for what the walk costs.
        """
        if self.edge_slots_contiguous:
            return np.arange(self.edge_count, dtype=np.int64)
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

    # -- the intrinsic columns --------------------------------------------
    # The weight of an edge is the array the store holds and needs nothing here.
    # The other two are derived from an array rather than held in one, so each is
    # one vectorized pass, kept against the clock. The pass is what makes the
    # read after a write cost microseconds instead of the milliseconds that
    # asking the graph for a record per edge cost.

    def _intrinsic_column(self, name: str, build):
        held = self._intrinsic_cache.get(name)
        if held is not None and held[0] == self.structure_version:
            return held[1]
        column = build()
        column.flags.writeable = False
        self._intrinsic_cache[name] = (self.structure_version, column)
        return column

    def edge_directed_column(self) -> np.ndarray:
        """Return the resolved direction of every edge slot, in slot order.

        ``edge_directed`` holds what an edge *declares*, which is one of three
        values, so the answer is not that array. An edge that declares nothing
        takes the default of the graph. A hyperedge takes neither: its direction
        follows from whether its members hold roles, exactly as one read one at a
        time does.
        """
        return self._intrinsic_column('directed', self._build_directed_column)

    def _build_directed_column(self) -> np.ndarray:
        count = len(self._edge_id)
        declared = self.edge_directed[:count]
        default = True if self.directed is None else bool(self.directed)
        # Declared is INHERIT, 0 or 1. With a directed default every value but 0
        # resolves to true, and with an undirected one only 1 does, so one
        # comparison answers the whole column.
        column = (declared != 0) if default else (declared == 1)
        hyper = np.flatnonzero(self.edge_kind[:count] == HYPER)
        for slot in hyper.tolist():
            column[slot] = self.hyper_directed(slot)
        return column

    def hyper_directed(self, slot: int) -> bool:
        """Return whether a hyperedge names a target side at all.

        A hyperedge that names no target side is undirected, whichever flag it
        was declared with. Its members carry the plain member role rather than a
        source role, and one entry says so.
        """
        if int(self.member_len[slot]) == 0:
            return False
        return int(self.member_role[int(self.member_start[slot])]) != MEMBER

    def edge_kind_column(self, names) -> np.ndarray:
        """Return the name of the kind of every edge slot, in slot order.

        ``names`` is the code-to-name table of the caller, because the store
        holds a code and the vocabulary of names belongs to the layer above it.
        A cached column is reused only for the table it was built from.
        """
        held = self._intrinsic_cache.get('kind')
        if held is not None and held[0] == self.structure_version and held[2] == names:
            return held[1]
        column = np.asarray(names)[self.edge_kind[: len(self._edge_id)]]
        column.flags.writeable = False
        self._intrinsic_cache['kind'] = (self.structure_version, column, names)
        return column

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
        specs = []
        for edge_id in edge_ids:
            slot = self._edge_slot.get(edge_id)
            if slot is None:
                continue
            weight = float(weights.get(edge_id, self.edge_weight[slot]))
            declared = int(self.edge_directed[slot])
            specs.append(
                (
                    edge_id,
                    self._selected_members(slot, other, weight),
                    int(self.edge_kind[slot]),
                    None if declared == INHERIT else bool(declared),
                    weight,
                    bool(self.edge_explicit[slot]),
                    self.edge_ml_kind.get(slot),
                    self.edge_ml_layers.get(slot),
                    self.edge_policy.get(slot),
                )
            )
            if len(specs) >= BULK_CHUNK:
                other.add_edges(specs)
                specs = []
        if specs:
            other.add_edges(specs)
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
        other._edge_entity_count = self._edge_entity_count
        other._placeholder_edge_count = self._placeholder_edge_count
        other._intrinsic_cache = {}
        other._entity_edges = {slot: dict(edges) for slot, edges in self._entity_edges.items()}

        other.entity_freed_hooks = []
        other.edge_freed_hooks = []
        other.entity_capacity_hooks = []
        other.edge_capacity_hooks = []

        other.structure_version = self.structure_version
        other.append_log = list(self.append_log)
        other.append_log_from_version = self.append_log_from_version
        other._structural_cache = None
        other._rows_cache = None
        other._slot_buffers = None
        return other

    def __repr__(self) -> str:
        return (
            f'CoreState(entities={self.entity_count}, edges={self.edge_count}, '
            f'version={self.structure_version})'
        )


# ---------------------------------------------------------------------------
# Building member entries
# ---------------------------------------------------------------------------


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

    ``graph`` is what a bare endpoint is resolved against, and a loader passes
    None so the store resolves it itself.

    The mutation gateway and every loader build member entries, and the rules
    for the role and the coefficient are subtle enough that they live here once.
    """
    weight = float(weight)
    members = []
    for side, role in ((source, SOURCE if target else MEMBER), (target, TARGET)):
        derived = -weight if role == TARGET and directed else weight
        for endpoint in _ordered(side):
            key = _bridged_key(state, graph, endpoint)
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


def _bridged_key(state, graph, endpoint):
    """Resolve a stored endpoint to an entity key, or None when it names none.

    A loader passes no graph, because the graph it is filling holds nothing to
    resolve against yet. The store already holds every entity by then, so it
    answers instead.
    """
    S = _facade()
    if S.is_entity_key(endpoint):
        return endpoint
    if graph is None:
        return S._slot_key(state, endpoint)
    try:
        return S.entity_key(graph, endpoint)
    except (KeyError, ValueError, TypeError):
        return None
