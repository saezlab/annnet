"""Atomic attribute writes and rollback of their structural side effects.

Attribute writes can trigger flexible-direction policies and update the
composite node-key index. AttributeTransaction snapshots affected rows, column
schema, the key index and incidence columns that those policies may rewrite.
An ordinary scalar write without policies snapshots one row and schema metadata,
without copying the graph.

On failure, rows and schema are restored through the attribute stores and
incidence columns through the mutation gateway. Clock changes invalidate derived
caches. The original exception propagates after rollback.

``_attribute_api`` opens the transaction around each commit.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np

from . import _mutate

RowReader = Callable[[str, Any], dict]
RowRestorer = Callable[[str, dict], None]
SchemaReader = Callable[[str], Any]
SchemaRestorer = Callable[[str, Any], None]


class AttributeTransaction:
    """Snapshot what one attribute batch may change; restore it if the batch fails.

    Parameters
    ----------
    graph : AnnNet
    address : str
        The attribute address being written.
    keys : Iterable
        The normalized keys the batch names.
    read_row : callable
        ``read_row(address, key) -> dict`` — the stored row, detached.
    restore_rows : callable
        ``restore_rows(address, {key: row})`` — put rows back exactly as read
        (fields not in ``row`` are removed).
    read_schema, restore_schema : callable, optional
        ``read_schema(address)`` returns what identifies the fields of the
        address now (or ``None`` when they are derived from the rows);
        ``restore_schema(address, schema)`` makes the address carry exactly
        that again: no field the batch introduced, none it removed, the same
        order and types.
    whole_axis : bool, default False
        The batch replaces the whole address (a table assignment); every edge
        with a policy is then a candidate for re-orientation.

    Use as a context manager around the commit::

        with AttributeTransaction(graph, address, keys, read_row=..., restore_rows=...):
            commit()
    """

    __slots__ = (
        '_graph',
        '_address',
        '_keys',
        '_read_row',
        '_restore_rows',
        '_read_schema',
        '_restore_schema',
        '_schema',
        '_rows',
        '_key_index',
        '_columns',
    )

    def __init__(
        self,
        graph,
        address: str,
        keys,
        *,
        read_row: RowReader,
        restore_rows: RowRestorer,
        read_schema: SchemaReader | None = None,
        restore_schema: SchemaRestorer | None = None,
        whole_axis: bool = False,
    ):
        self._graph = graph
        self._address = address
        self._keys = list(keys)
        self._read_row = read_row
        self._restore_rows = restore_rows
        self._read_schema = read_schema
        self._restore_schema = restore_schema
        self._schema = read_schema(address) if read_schema is not None else None
        self._rows = {key: read_row(address, key) for key in self._keys}
        self._key_index = (
            dict(graph._node_key_index)
            if address == 'nodes' and graph._node_key_enabled()
            else None
        )
        self._columns = self._snapshot_columns(whole_axis)

    # -- what a policy may rewrite ------------------------------------------

    def _policy_edges(self, whole_axis: bool) -> list:
        """The edges whose incidence column a flexible-direction policy may rewrite."""
        graph = self._graph
        policies = graph.edge_direction_policy
        if not policies:
            return []
        if self._address == 'edges':
            if whole_axis:
                return [
                    eid for eid, policy in policies.items() if policy.get('scope', 'edge') == 'edge'
                ]
            return [
                eid
                for eid in self._keys
                if eid in policies and policies[eid].get('scope', 'edge') == 'edge'
            ]
        if self._address == 'nodes':
            if not graph._variables_watched_by_nodes():
                return []
            found: dict = {}
            for node_id in self._keys:
                for eid in graph._incident_flexible_edges(node_id):
                    found.setdefault(eid, None)
            return list(found)
        return []

    def _snapshot_columns(self, whole_axis: bool) -> dict:
        store = self._graph._store
        snapshot = {}
        for eid in self._policy_edges(whole_axis):
            slot = store.edge_slot(eid)
            if slot is None:
                continue
            members = store.members(slot)
            snapshot[eid] = (members.coefficients.copy(), bool(store.edge_explicit[slot]))
        return snapshot

    # -- the protocol --------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.rollback()
        return False

    def rollback(self) -> None:
        """Put back the rows, the schema, the key index and every rewritten incidence column.

        A column that was not rewritten is left alone, so a batch that failed
        before any policy fired moves no structural clock.
        """
        self._restore_rows(self._address, self._rows)
        if self._restore_schema is not None and self._schema is not None:
            self._restore_schema(self._address, self._schema)
        if self._key_index is not None:
            self._graph._node_key_index = self._key_index
        store = self._graph._store
        for eid, (coefficients, explicit) in self._columns.items():
            slot = store.edge_slot(eid)
            if slot is None:
                continue
            members = store.members(slot)
            if (
                bool(store.edge_explicit[slot]) == explicit
                and members.coefficients.shape == coefficients.shape
                and np.array_equal(members.coefficients, coefficients)
            ):
                continue
            _mutate.restore_edge_column(self._graph, eid, coefficients, explicit)


__all__ = ['AttributeTransaction']
