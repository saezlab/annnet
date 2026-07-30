"""The slot-indexed attribute columns and the derived tables.

One generic attribute is one typed array. The array is indexed by slot, so a value
keeps its place when another element goes away, and one write lands in one cell. A
free slot holds a null.

A contextual attribute belongs to a pair rather than to one element, for example
one edge in one slice or one node in one layer. Almost no pair carries a value, so
a dense column per pair would waste nearly every cell. Those stores therefore stay
keyed by the pair.

The node table and the edge table are derived. They gather the live slots of every
column and hand the result to narwhals, so one materialization serves every
dataframe backend. This storage does not depend on the backend of the user. The
backend matters only when a table materializes.
"""

from __future__ import annotations

import numpy as np

from .._session import logger
from .._support.dataframe_backend import dataframe_from_columns

log = logger(__name__)

NODE_AXIS = 'node'
EDGE_AXIS = 'edge'

# A column of a fixed-width type uses numpy. Anything else, such as a string or a
# list, uses an object array, which is what a dataframe backend takes anyway.
_NUMERIC_KINDS = 'biufc'


def _empty_column(size: int, template) -> np.ndarray:
    if isinstance(template, np.ndarray) and template.dtype.kind in _NUMERIC_KINDS:
        return np.full(size, np.nan if template.dtype.kind == 'f' else 0, dtype=template.dtype)
    return np.full(size, None, dtype=object)


def _column_for(value, size: int) -> np.ndarray:
    """Return a fresh column wide enough for ``size`` slots, typed for ``value``."""
    if isinstance(value, (bool, np.bool_)):
        return np.full(size, None, dtype=object)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.full(size, np.nan, dtype=np.float64)
    return np.full(size, None, dtype=object)


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

    def __init__(self, store):
        self._store = store
        self.node_columns: dict[str, np.ndarray] = {}
        self.edge_columns: dict[str, np.ndarray] = {}

        # Contextual stores, each keyed by its own pair.
        self.slice_attributes: dict[str, dict] = {}
        self.edge_slice_attributes: dict[tuple[str, str], dict] = {}
        self.node_layer_attributes: dict[tuple[tuple, tuple], dict] = {}
        self.aspect_attributes: dict[str, dict] = {}
        self.layer_attributes: dict[tuple, dict] = {}
        self.graph_attrs: dict = {}

        # The per-slice weight override, which decides the effective weight of an
        # edge. It is separate from the other contextual attributes because the
        # materialization of a matrix reads it.
        self.slice_weights: dict[tuple[str, str], float] = {}

        self._tables: dict[str, tuple] = {}
        self.table_builds = 0

        # A freed slot must hold a null, so the store announces a free and these
        # hooks clear the cells that belonged to the element that went away.
        store.entity_freed_hooks.append(self._on_entity_freed)
        store.edge_freed_hooks.append(self._on_edge_freed)

    # -- generic columns --------------------------------------------------

    def _set(self, columns: dict, capacity: int, slot: int, name: str, value) -> None:
        column = columns.get(name)
        if column is None:
            column = _column_for(value, max(8, capacity))
            columns[name] = column
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

    def node_column(self, name: str) -> np.ndarray:
        """Return the raw node column, indexed by slot."""
        return self.node_columns[name]

    def edge_column(self, name: str) -> np.ndarray:
        """Return the raw edge column, indexed by slot."""
        return self.edge_columns[name]

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

    def _rows(self, columns: dict, slots: np.ndarray, id_column: str, ids) -> list[dict]:
        rows = []
        for position, slot in enumerate(slots):
            row = {id_column: ids[position]}
            for name, column in columns.items():
                if slot < column.size:
                    value = column[slot]
                    if value is not None and not (isinstance(value, float) and np.isnan(value)):
                        row[name] = value
            rows.append(row)
        return rows

    def obs_rows(self) -> list[dict]:
        """Return the node table as row dictionaries, one per live node."""
        slots = self._store.live_entity_slots()
        ids = [self._store.entity_key(int(slot))[0] for slot in slots]
        return self._rows(self.node_columns, slots, 'node_id', ids)

    def var_rows(self) -> list[dict]:
        """Return the edge table as row dictionaries, one per live edge."""
        slots = self._store.live_edge_slots()
        ids = [self._store.edge_id(int(slot)) for slot in slots]
        return self._rows(self.edge_columns, slots, 'edge_id', ids)

    def _table(self, axis: str, backend, id_column: str, columns: dict, slots, ids):
        cached = self._tables.get(axis)
        if cached is not None and cached[0] == backend:
            return cached[1]
        data = {id_column: ids}
        for name, column in columns.items():
            data[name] = [column[slot] if slot < column.size else None for slot in slots]
        table = dataframe_from_columns(data, backend=backend)
        self.table_builds += 1
        self._tables[axis] = (backend, table)
        return table

    def obs(self, *, backend='auto'):
        """Materialize the node table. One build serves every read until a write."""
        slots = self._store.live_entity_slots()
        ids = [self._store.entity_key(int(slot))[0] for slot in slots]
        return self._table(NODE_AXIS, backend, 'node_id', self.node_columns, slots, ids)

    def var(self, *, backend='auto'):
        """Materialize the edge table. One build serves every read until a write."""
        slots = self._store.live_edge_slots()
        ids = [self._store.edge_id(int(slot)) for slot in slots]
        return self._table(EDGE_AXIS, backend, 'edge_id', self.edge_columns, slots, ids)

    def drop_tables(self) -> None:
        """Forget the materialized tables. A table is always safe to drop."""
        self._tables.clear()

    # -- contextual attributes --------------------------------------------

    def set_slice(self, slice_id: str, **attrs) -> None:
        """Set attributes of one slice."""
        self.slice_attributes.setdefault(slice_id, {}).update(attrs)

    def slice_attrs(self, slice_id: str) -> dict:
        """Return the attributes of one slice."""
        return dict(self.slice_attributes.get(slice_id, {}))

    def set_edge_slice(self, slice_id: str, edge_id: str, **attrs) -> None:
        """Set attributes of one edge inside one slice."""
        self.edge_slice_attributes.setdefault((slice_id, edge_id), {}).update(attrs)

    def edge_slice_attrs(self, slice_id: str, edge_id: str) -> dict:
        """Return the attributes of one edge inside one slice."""
        return dict(self.edge_slice_attributes.get((slice_id, edge_id), {}))

    def set_node_layer(self, key: tuple, layer: tuple, **attrs) -> None:
        """Set attributes of one node inside one layer."""
        self.node_layer_attributes.setdefault((key, layer), {}).update(attrs)

    def node_layer_attrs(self, key: tuple, layer: tuple) -> dict:
        """Return the attributes of one node inside one layer."""
        return dict(self.node_layer_attributes.get((key, layer), {}))

    def set_aspect(self, aspect: str, **attrs) -> None:
        """Set attributes of one aspect."""
        self.aspect_attributes.setdefault(aspect, {}).update(attrs)

    def aspect_attrs(self, aspect: str) -> dict:
        """Return the attributes of one aspect."""
        return dict(self.aspect_attributes.get(aspect, {}))

    def set_layer(self, layer: tuple, **attrs) -> None:
        """Set attributes of one layer."""
        self.layer_attributes.setdefault(layer, {}).update(attrs)

    def layer_attrs(self, layer: tuple) -> dict:
        """Return the attributes of one layer."""
        return dict(self.layer_attributes.get(layer, {}))

    # -- the per-slice weight override ------------------------------------

    def set_slice_weight(self, slice_id: str, edge_id: str, weight: float) -> None:
        """Override the weight of one edge inside one slice."""
        self.slice_weights[(slice_id, edge_id)] = float(weight)

    def effective_weight(self, edge_id: str, *, slice_id: str | None = None) -> float:
        """Return the weight of one edge, honouring a slice override.

        A matrix materialization reads this, so the override is a structural fact
        and not only an annotation.
        """
        if slice_id is not None:
            override = self.slice_weights.get((slice_id, edge_id))
            if override is not None:
                return override
        slot = self._require_edge(edge_id)
        return float(self._store.edge_weight[slot])

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
        """Drop the contextual attributes of one edge."""
        for mapping in (self.edge_slice_attributes, self.slice_weights):
            for pair in [pair for pair in mapping if pair[1] == edge_id]:
                del mapping[pair]
        self._tables.pop(EDGE_AXIS, None)

    def forget_node(self, key: tuple) -> None:
        """Drop the contextual attributes of one node."""
        for pair in [pair for pair in self.node_layer_attributes if pair[0] == key]:
            del self.node_layer_attributes[pair]
        self._tables.pop(NODE_AXIS, None)
