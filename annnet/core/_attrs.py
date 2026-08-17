"""The slot-indexed attribute columns and the derived tables.

One generic attribute is one typed array. The array is indexed by slot, so a value
keeps its place when another element goes away, and one write lands in one cell. A
free slot holds a null.

**Reading a whole column borrows that array.** On a graph whose live slots are
contiguous, the array cut to the live count *is* the answer, so the read copies
nothing and walks nothing, and it costs what slicing an array costs whether or
not a write came before it. A graph with a freed slot falls back to gathering the
live slots, which gives the same values in the same order and costs more.

Two things make the borrowing read possible. The store answers in constant time
whether its slots are contiguous, and the columns grow when the frontier moves
rather than when a value is written, so a column is never shorter than the live
count and the slice always applies.

**A read gives back a read-only array**, on every path, because it is a window
onto the canonical state and a write through it would reach the graph with no
validation, no clock bump and no history entry. A column is good until the next
write to the graph; a caller who wants to hold values across a change copies.

A contextual attribute belongs to a pair rather than to one element, for example
one edge in one slice or one node in one layer. Almost no pair carries a value, so
a dense column per pair would waste nearly every cell, and those stores stay keyed
by the pair instead.

**This store does not hold them.** It owns the two generic axes and nothing else.
It once carried a second, parallel set of the contextual stores; nothing wrote to
them on any public path, so a value written through the API never appeared in
them and a reader who found them here looked in the wrong place. They are gone.

The node table and the edge table are derived. They gather the live slots of every
column and hand the result to narwhals, so one materialization serves every
dataframe backend. This storage does not depend on the backend of the user. The
backend matters only when a table materializes.

**A generic node attribute belongs to a node and not to a node-layer.** A
multilayer graph holds one entity per layer the node lives in, so one bare id
covers several slots. A write by bare id lands in the cell of each of them, and
the derived table shows the id once. What one node-layer carries alone is a
contextual attribute, written through ``G.layers.set_node_attrs``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._store import NODE as _NODE_ENTITY
from .._session import logger
from .._support.dataframe_backend import dataframe_from_columns

log = logger(__name__)

NODE_AXIS = 'node'
EDGE_AXIS = 'edge'

# A column of a fixed-width type uses numpy. Anything else, such as a string or a
# list, uses an object array, which is what a dataframe backend takes anyway.
_NUMERIC_KINDS = 'biufc'

# A capacity no store reaches, so an axis that holds no column never grows one.
_NO_COLUMNS = 1 << 62


def _is_null(value) -> bool:
    """Return True for a cell that carries nothing.

    A free cell of an object column holds None and a free cell of a float column
    holds a NaN, so the two spellings of "nothing" both have to be read.
    """
    return value is None or (isinstance(value, float) and value != value)


def _empty_column(size: int, template) -> np.ndarray:
    if isinstance(template, np.ndarray) and template.dtype.kind in _NUMERIC_KINDS:
        return np.full(size, np.nan if template.dtype.kind == 'f' else 0, dtype=template.dtype)
    return np.full(size, None, dtype=object)


def _column_for(value, size: int) -> np.ndarray:
    """Return a fresh column wide enough for ``size`` slots, typed for ``value``.

    A float column marks a free slot with a NaN, which is the value a float
    already has for "no number". An integer takes an object column instead: a
    float column would give it back as a float, and a count that reads as 3.0 is
    not the count that was written.
    """
    if isinstance(value, (bool, np.bool_)):
        return np.full(size, None, dtype=object)
    if isinstance(value, (float, np.floating)):
        return np.full(size, np.nan, dtype=np.float64)
    return np.full(size, None, dtype=object)


def read_only(values: np.ndarray) -> np.ndarray:
    """Return ``values`` with writing refused.

    A column read hands back a window onto the canonical store. A write through
    that window would reach the graph with no validation, no clock bump and no
    history entry, which is worse than either copying or refusing. So it is
    refused, and it is refused on every read path — the borrowing one and the
    gathering one alike — so that a caller never has to ask which answered.

    A caller who means to change values copies, which is one call and is visible
    in their code. The entry points that write the graph are unaffected.
    """
    values.flags.writeable = False
    return values


def _borrowed(column: np.ndarray, count: int) -> np.ndarray:
    """Return the first ``count`` cells of ``column`` without copying them.

    The caller has established that the live slots are ``0 .. count-1``, so the
    range and the elements are the same thing. Eager growth is what makes the
    column long enough for this to hold; a column that is somehow short still
    answers, by growing here rather than by giving back fewer values than there
    are elements.
    """
    if column.size < count:  # pragma: no cover - eager growth keeps this unreachable
        column = _grown(column, count)
    return read_only(column[:count])


def _grown(column: np.ndarray, size: int) -> np.ndarray:
    if size <= column.size:
        return column
    width = max(8, column.size)
    while width < size:
        width *= 2
    out = _empty_column(width, column)
    out[: column.size] = column
    return out


class AttributeStore:
    """The attributes of one graph, generic and contextual.

    The store holds no dataframe. It holds columns and mappings, and it builds a
    table when a reader asks for one. :attr:`table_builds` counts those builds, so
    a test can show that a write does not trigger one.
    """

    def __init__(self, store, *, node_id_column: str = 'node_id', edge_id_column: str = 'edge_id'):
        self._store = store
        self.node_id_column = node_id_column
        self.edge_id_column = edge_id_column
        self.node_columns: dict[str, np.ndarray] = {}
        self.edge_columns: dict[str, np.ndarray] = {}

        # The capacity every column of an axis is already long enough for. It is
        # what makes eager growth one comparison per allocation rather than one
        # per column per allocation. See :meth:`_forget_floors`.
        self._node_floor = 0
        self._edge_floor = 0

        self._tables: dict[str, tuple] = {}
        self._row_cache: dict[str, tuple] = {}
        self.table_builds = 0

        # A freed slot must hold a null, so the store announces a free and these
        # hooks clear the cells that belonged to the element that went away. A
        # new slot has to be addressable in every column before a read indexes
        # one with a range, so the store announces that too and these grow.
        self._follow(store)

    def rebind(self, store) -> None:
        """Follow the graph onto a canonical store it was handed whole.

        A copy, a selection and a loader all install a store the graph did not
        build one element at a time. The columns are indexed by slot, so they
        belong to the store they were written against and none of them survives
        the swap.
        """
        if store is self._store:
            return
        self._store = store
        self._follow(store)
        self.node_columns = {}
        self.edge_columns = {}
        self._forget_floors()
        self._tables.clear()
        self._row_cache.clear()

    def _follow(self, store) -> None:
        """Subscribe to the slot lifecycle of one store."""
        store.entity_freed_hooks.append(self._on_entity_freed)
        store.edge_freed_hooks.append(self._on_edge_freed)
        store.entity_capacity_hooks.append(self._on_entity_capacity)
        store.edge_capacity_hooks.append(self._on_edge_capacity)

    # -- eager growth -----------------------------------------------------
    # A column shorter than the live count is the one thing that stops a read
    # from being a slice: the missing values are the elements added since the
    # column last grew, and each of them carries a null. Padding on read would
    # copy, which is what the borrowing read exists to stop, so the columns
    # grow when the frontier moves instead.
    #
    # The cost lands on the write path, where a growth block amortizes it: a
    # column doubles, so it grows a logarithmic number of times over the life
    # of a graph, and the announcement of a bulk write comes once per batch.

    @staticmethod
    def _grow_all(columns: dict, capacity: int) -> int:
        """Grow every column to ``capacity`` and return the shortest one after.

        The answer is the capacity up to which nothing has to be done again. A
        column doubles when it grows, so it comes back well clear of the
        capacity that made it grow, and the next few thousand allocations are
        one integer comparison each.
        """
        floor = _NO_COLUMNS
        for name, column in columns.items():
            if column.size < capacity:
                column = _grown(column, capacity)
                columns[name] = column
            if column.size < floor:
                floor = column.size
        return floor

    def _on_entity_capacity(self, capacity: int) -> None:
        if capacity > self._node_floor:
            self._node_floor = self._grow_all(self.node_columns, capacity)

    def _on_edge_capacity(self, capacity: int) -> None:
        if capacity > self._edge_floor:
            self._edge_floor = self._grow_all(self.edge_columns, capacity)

    def _forget_floors(self) -> None:
        """Say that nothing is known about how long the columns are.

        Every path that replaces a column, or a whole set of them, calls this.
        A column that is grown is never shortened, so growth alone never needs
        it.
        """
        self._node_floor = 0
        self._edge_floor = 0

    # -- generic columns --------------------------------------------------

    def _set(self, columns: dict, capacity: int, slot: int, name: str, value) -> None:
        column = columns.get(name)
        if column is None:
            column = _column_for(value, max(8, capacity))
            columns[name] = column
            self._forget_floors()
        column = _grown(column, capacity)
        if column.dtype.kind in _NUMERIC_KINDS and not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            # The column has to widen to hold a value its type cannot.
            log.debug(
                'Widening attribute column %r from %s to hold a value of type %s.',
                name,
                column.dtype,
                type(value).__name__,
            )
            widened = np.full(column.size, None, dtype=object)
            widened[:] = column
            column = widened
        columns[name] = column
        column[slot] = value

    def set_node(self, key: tuple, name: str, value) -> None:
        """Set one attribute of one node. The cost is one cell."""
        slot = self._require_entity(key)
        self._set(self.node_columns, self._store.entity_capacity, slot, name, value)
        self._tables.pop(NODE_AXIS, None)

    def set_edge(self, edge_id: str, name: str, value) -> None:
        """Set one attribute of one edge. The cost is one cell."""
        slot = self._require_edge(edge_id)
        self._set(self.edge_columns, len(self._store._edge_id), slot, name, value)
        self._tables.pop(EDGE_AXIS, None)

    def set_node_column(self, name: str, values) -> None:
        """Set a whole node column, in the slot order of the live nodes."""
        self._set_column(self.node_columns, self._store.live_entity_slots(), name, values)
        self._tables.pop(NODE_AXIS, None)

    def set_edge_column(self, name: str, values) -> None:
        """Set a whole edge column, in the slot order of the live edges."""
        self._set_column(self.edge_columns, self._store.live_edge_slots(), name, values)
        self._tables.pop(EDGE_AXIS, None)

    def _set_column(self, columns: dict, slots: np.ndarray, name: str, values) -> None:
        values = np.asarray(values, dtype=object) if not isinstance(values, np.ndarray) else values
        if values.size != slots.size:
            raise ValueError(f'column {name!r} needs {slots.size} values, got {values.size}')
        capacity = int(slots.max()) + 1 if slots.size else 8
        column = _empty_column(max(8, capacity), values)
        column[slots] = values
        columns[name] = column
        self._forget_floors()

    def node_column(self, name: str) -> np.ndarray:
        """Return the raw node column, indexed by slot."""
        return self.node_columns[name]

    def edge_column(self, name: str) -> np.ndarray:
        """Return the raw edge column, indexed by slot."""
        return self.edge_columns[name]

    # -- generic columns, addressed by bare id ----------------------------
    # A graph names a node by its bare id and an edge by its id. An edge id is
    # one slot, so the two differ only in that a node id may cover several.

    def set_node_attrs(self, node_id: str, attrs: dict) -> None:
        """Set attributes of one node, in every layer it lives in."""
        if not attrs:
            return
        slots = self._store.entity_slots_of_id(node_id)
        if not slots:
            return
        capacity = self._store.entity_capacity
        columns = self.node_columns
        for name, value in attrs.items():
            for slot in slots:
                self._set(columns, capacity, slot, name, value)
        self._tables.pop(NODE_AXIS, None)

    def set_edge_attrs(self, edge_id: str, attrs: dict) -> None:
        """Set attributes of one edge."""
        if not attrs:
            return
        slot = self._store.edge_slot(edge_id)
        if slot is None:
            return
        capacity = len(self._store._edge_id)
        columns = self.edge_columns
        for name, value in attrs.items():
            self._set(columns, capacity, slot, name, value)
        self._tables.pop(EDGE_AXIS, None)

    def node_attr(self, node_id: str, name: str, default=None):
        """Return one attribute of one node, or ``default`` when it carries none."""
        column = self.node_columns.get(name)
        if column is None:
            return default
        for slot in self._store.entity_slots_of_id(node_id):
            if slot < column.size:
                value = column[slot]
                if not _is_null(value):
                    return value
        return default

    def edge_attr(self, edge_id: str, name: str, default=None):
        """Return one attribute of one edge, or ``default`` when it carries none."""
        column = self.edge_columns.get(name)
        slot = self._store.edge_slot(edge_id)
        if column is None or slot is None or slot >= column.size:
            return default
        value = column[slot]
        return default if _is_null(value) else value

    def node_attrs(self, node_id: str) -> dict:
        """Return every attribute one node carries."""
        slots = self._store.entity_slots_of_id(node_id)
        if not slots:
            return {}
        return self._cells(self.node_columns, slots)

    def edge_attrs(self, edge_id: str) -> dict:
        """Return every attribute one edge carries."""
        slot = self._store.edge_slot(edge_id)
        if slot is None:
            return {}
        return self._cells(self.edge_columns, (slot,))

    @staticmethod
    def _cells(columns: dict, slots) -> dict:
        out = {}
        for name, column in columns.items():
            for slot in slots:
                if slot < column.size:
                    value = column[slot]
                    if not _is_null(value):
                        out[name] = value
                        break
        return out

    def node_attr_map(self, name: str) -> dict | None:
        """Return one node column as a mapping from bare id, or None when unset."""
        column = self.node_columns.get(name)
        if column is None:
            return None
        return {
            node_id: (None if slot >= column.size or _is_null(column[slot]) else column[slot])
            for node_id, slot in self._node_rows()
        }

    def edge_attr_map(self, name: str) -> dict | None:
        """Return one edge column as a mapping from edge id, or None when unset."""
        column = self.edge_columns.get(name)
        if column is None:
            return None
        return {
            edge_id: (None if slot >= column.size or _is_null(column[slot]) else column[slot])
            for edge_id, slot in self._edge_rows()
        }

    def drop_node_columns(self) -> None:
        """Forget every generic node attribute, keeping one row per node."""
        self.node_columns = {}
        self._forget_floors()
        self._tables.pop(NODE_AXIS, None)

    def drop_edge_columns(self) -> None:
        """Forget every generic edge attribute, keeping one row per edge."""
        self.edge_columns = {}
        self._forget_floors()
        self._tables.pop(EDGE_AXIS, None)

    def load_node_rows(self, rows) -> None:
        """Replace every node column with what these rows say.

        A row names its node under the id column of this store. A row for a node
        the store does not hold is left out, exactly as the derived table leaves
        it out.
        """
        self.drop_node_columns()
        id_column = self.node_id_column
        for row in rows:
            node_id = row.get(id_column)
            if node_id is None:
                continue
            self.set_node_attrs(
                node_id,
                {name: value for name, value in row.items() if name != id_column},
            )

    def load_edge_rows(self, rows) -> None:
        """Replace every edge column with what these rows say."""
        self.drop_edge_columns()
        id_column = self.edge_id_column
        for row in rows:
            edge_id = row.get(id_column)
            if edge_id is None:
                continue
            self.set_edge_attrs(
                edge_id,
                {name: value for name, value in row.items() if name != id_column},
            )

    def _require_entity(self, key: tuple) -> int:
        slot = self._store.entity_slot(key)
        if slot is None:
            raise KeyError(f'Unknown entity: {key!r}')
        return slot

    def _require_edge(self, edge_id: str) -> int:
        slot = self._store.edge_slot(edge_id)
        if slot is None:
            raise KeyError(f'Unknown edge id: {edge_id!r}')
        return slot

    # -- the derived tables -----------------------------------------------

    def _node_rows(self) -> list[tuple]:
        """Return one ``(bare id, slot)`` pair per node, in slot order.

        An edge entity is not a node and takes no row. A multilayer graph holds
        one entity per layer a node lives in, and the node takes one row, whose
        cells the first of its slots carries.

        The walk is over every entity, and a column read asks for it, so the
        answer is kept against the clock of the store, together with the slots
        as one array, which is what a column read indexes with.
        """
        return self._rows_of(NODE_AXIS, self._built_node_rows)[0]

    def _built_node_rows(self) -> list[tuple]:
        store = self._store
        kinds = store.entity_kind
        pairs = []
        if store.aspects == ('_',):
            for slot, key in enumerate(store._entity_key):
                if key is not None and kinds[slot] == _NODE_ENTITY:
                    pairs.append((key[0], slot))
            return pairs
        seen: set = set()
        for slot, key in enumerate(store._entity_key):
            if key is None or kinds[slot] != _NODE_ENTITY or key[0] in seen:
                continue
            seen.add(key[0])
            pairs.append((key[0], slot))
        return pairs

    def _built_edge_rows(self) -> list[tuple]:
        return [(edge_id, slot) for slot, edge_id in self._store.live_edges()]

    def _edge_rows(self) -> list[tuple]:
        """Return one ``(edge id, slot)`` pair per live edge, in slot order."""
        return self._rows_of(EDGE_AXIS, self._built_edge_rows)[0]

    def _rows_of(self, axis: str, build):
        held = self._row_cache.get(axis)
        version = self._store.structure_version
        if held is not None and held[0] == version:
            return held[1], held[2]
        pairs = build()
        slots = np.fromiter((slot for _id, slot in pairs), dtype=np.int64, count=len(pairs))
        self._row_cache[axis] = (version, pairs, slots)
        return pairs, slots

    def node_vector(self, name: str):
        """Return one node column as a vector, aligned with :meth:`node_ids`.

        This is the whole point of a column store: the answer is a slice of the
        array the store already holds, not a walk over rows. ``None`` says the
        store holds no such attribute, which is not the same as one every node
        leaves empty.

        The slice applies when one node of the node axis sits in each entity
        slot, in order — which is what :attr:`CoreState.node_axis_contiguous`
        answers, in constant time, from the freelist, the aspects and the count
        of edge-entities. Anything else falls back to the gather, which is
        slower and gives the same values in the same order.
        """
        column = self.node_columns.get(name)
        if column is None:
            return None
        store = self._store
        if store.node_axis_contiguous:
            return _borrowed(column, store.entity_count)
        return self._vector(column, self._rows_of(NODE_AXIS, self._built_node_rows)[1])

    def edge_vector(self, name: str):
        """Return one edge column as a vector, aligned with :meth:`edge_ids`.

        The edge axis holds one live edge per slot, so it asks the one question
        the node axis asks first: whether any slot has been freed and not yet
        reused.
        """
        column = self.edge_columns.get(name)
        if column is None:
            return None
        store = self._store
        if store.edge_slots_contiguous:
            return _borrowed(column, store.edge_count)
        return self._vector(column, self._rows_of(EDGE_AXIS, self._built_edge_rows)[1])

    @staticmethod
    def _vector(column, slots: np.ndarray):
        if column is None:
            return None
        if slots.size and int(slots[-1]) >= column.size:
            # A slot past the end of the column is an element written after the
            # column was last grown, and it carries nothing.
            grown = _empty_column(int(slots[-1]) + 1, column)
            grown[: column.size] = column
            column = grown
        return read_only(column[slots])

    def _rows(self, columns: dict, pairs: list[tuple], id_column: str) -> list[dict]:
        rows = []
        for element_id, slot in pairs:
            row = {id_column: element_id}
            for name, column in columns.items():
                if slot < column.size:
                    value = column[slot]
                    if not _is_null(value):
                        row[name] = value
            rows.append(row)
        return rows

    def obs_rows(self) -> list[dict]:
        """Return the node table as row dictionaries, one per live node."""
        return self._rows(self.node_columns, self._node_rows(), self.node_id_column)

    def var_rows(self) -> list[dict]:
        """Return the edge table as row dictionaries, one per live edge."""
        return self._rows(self.edge_columns, self._edge_rows(), self.edge_id_column)

    def _table(self, axis: str, backend, id_column: str, columns: dict, pairs: list[tuple]):
        cached = self._tables.get(axis)
        if cached is not None and cached[0] == backend:
            return cached[1]
        ids = [element_id for element_id, _slot in pairs]
        slots = [slot for _element_id, slot in pairs]
        data = {id_column: ids}
        for name, column in columns.items():
            data[name] = [
                None if slot >= column.size or _is_null(column[slot]) else column[slot]
                for slot in slots
            ]
        table = dataframe_from_columns(data, backend=backend)
        self.table_builds += 1
        self._tables[axis] = (backend, table)
        return table

    def obs(self, *, backend='auto'):
        """Materialize the node table. One build serves every read until a write."""
        return self._table(
            NODE_AXIS, backend, self.node_id_column, self.node_columns, self._node_rows()
        )

    def var(self, *, backend='auto'):
        """Materialize the edge table. One build serves every read until a write."""
        return self._table(
            EDGE_AXIS, backend, self.edge_id_column, self.edge_columns, self._edge_rows()
        )

    def copy_columns_from(self, other: AttributeStore) -> None:
        """Take the generic columns of another store, cell for cell.

        The caller has to have copied the canonical store, so that every slot
        addresses the same element in both. A selection renumbers its slots and
        loads its rows instead.
        """
        self.node_columns = {name: column.copy() for name, column in other.node_columns.items()}
        self.edge_columns = {name: column.copy() for name, column in other.edge_columns.items()}
        self._forget_floors()
        self._tables.clear()

    def node_attr_rows(self, node_ids=None) -> dict:
        """Return the attributes of every node, keyed by bare id.

        Pass ``node_ids`` to ask about those alone. A node that carries nothing
        answers with an empty mapping, so every id asked for is a key of the
        result.
        """
        return self._attr_rows(self.node_columns, self._node_rows(), node_ids)

    def edge_attr_rows(self, edge_ids=None) -> dict:
        """Return the attributes of every edge, keyed by edge id."""
        return self._attr_rows(self.edge_columns, self._edge_rows(), edge_ids)

    @staticmethod
    def _attr_rows(columns: dict, pairs: list[tuple], wanted) -> dict:
        if wanted is not None:
            wanted = set(wanted)
            pairs = [pair for pair in pairs if pair[0] in wanted]
        out: Any = {element_id: {} for element_id, _slot in pairs}
        for name, column in columns.items():
            size = column.size
            for element_id, slot in pairs:
                if slot < size:
                    value = column[slot]
                    if not _is_null(value):
                        out[element_id][name] = value
        return out

    def node_ids(self) -> list:
        """The bare id of every node the table holds a row for, in row order."""
        return [node_id for node_id, _slot in self._node_rows()]

    def edge_ids(self) -> list:
        """The id of every edge the table holds a row for, in row order."""
        return [edge_id for edge_id, _slot in self._edge_rows()]

    def node_column_names(self) -> list[str]:
        """The generic node attributes the store holds, in insertion order."""
        return list(self.node_columns)

    def edge_column_names(self) -> list[str]:
        """The generic edge attributes the store holds, in insertion order."""
        return list(self.edge_columns)

    def drop_tables(self) -> None:
        """Forget the materialized tables. A table is always safe to drop."""
        self._tables.clear()

    # -- forgetting -------------------------------------------------------

    @staticmethod
    def _clear_cell(columns: dict, slot: int) -> None:
        for column in columns.values():
            if slot < column.size:
                column[slot] = np.nan if column.dtype.kind == 'f' else None

    def _on_entity_freed(self, slot: int, key: tuple) -> None:
        self._clear_cell(self.node_columns, slot)
        self.forget_node(key)

    def _on_edge_freed(self, slot: int, edge_id: str) -> None:
        self._clear_cell(self.edge_columns, slot)
        self.forget_edge(edge_id)

    def forget_edge(self, edge_id: str) -> None:
        """Drop what this store held for one edge.

        The slot-indexed cells are cleared by the caller; what is left here is
        the materialized edge table, which named the edge and is now stale.
        Contextual state keyed by a pair the edge belongs to lives outside this
        store, and its owner drops its own.
        """
        self._tables.pop(EDGE_AXIS, None)

    def forget_node(self, key: tuple) -> None:
        """Drop what this store held for one node. See :meth:`forget_edge`."""
        self._tables.pop(NODE_AXIS, None)
