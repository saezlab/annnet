"""The derived matrices.

Every matrix here is built from the member lists of the canonical store. A matrix
is derived state. It is safe to drop, and it is never authoritative.

The package builds several purpose-built matrices instead of one matrix that mixes
every edge kind. A rule that one matrix needs lives in that matrix and does not
reach the canonical store or another matrix. The rules for the two awkward shapes
are:

============  =========================================  ==============================
Matrix        Self-loop                                  Boundary edge
============  =========================================  ==============================
incidence     the two entries sum, which gives zero      the single entry stays
adjacency     one diagonal entry                         left out, so no false loop
Laplacian     follows the adjacency                      follows the adjacency
============  =========================================  ==============================

A matrix is not a second copy of the topology. The member lists already are an
incidence matrix in compressed sparse column form, so building one is a gather
over those arrays and a slot-to-row remap. Nothing has to stay in step with
anything.

:class:`MatrixCache` keeps a built matrix against the clock of the store. A write
that only appends edges at the frontier extends the cached matrix instead of
dropping it, so a read after an append costs the new columns and not a rebuild.
Without that, a loop of N appends with a read after each one would be quadratic.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from . import _store as ST
from .._session import logger

log = logger(__name__)


class MatrixView(NamedTuple):
    """One materialized matrix and the maps from identity to position.

    A position belongs to this matrix and to nothing else. Read a position only to
    index this matrix.
    """

    matrix: object
    row_of_entity: dict
    column_of_edge: dict
    entity_of_row: tuple
    edge_of_column: list


def _row_lookup(store) -> tuple[np.ndarray, np.ndarray]:
    """Return the live entity slots and a slot-to-row map wide enough to index."""
    slots = store.live_entity_slots()
    width = max(1, store.entity_capacity)
    lookup = np.full(width, -1, dtype=np.int64)
    lookup[slots] = np.arange(slots.size, dtype=np.int64)
    return slots, lookup


def _selected_edge_slots(store, kinds) -> np.ndarray:
    slots = store.live_edge_slots()
    if kinds is None or slots.size == 0:
        return slots
    wanted = np.asarray(tuple(kinds), dtype=np.uint8)
    return slots[np.isin(store.edge_kind[slots], wanted)]


def _gather_members(store, edge_slots: np.ndarray):
    """Gather the member entries of many edges into flat arrays.

    Returns the entity slot, the coefficient, the role, and the position of the
    edge within ``edge_slots`` for every member entry.
    """
    if edge_slots.size == 0:
        empty_int = np.zeros(0, dtype=np.int64)
        return empty_int, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int8), empty_int
    lengths = store.member_len[edge_slots].astype(np.int64)
    total = int(lengths.sum())
    if total == 0:
        empty_int = np.zeros(0, dtype=np.int64)
        return empty_int, np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.int8), empty_int
    columns = np.repeat(np.arange(edge_slots.size, dtype=np.int64), lengths)
    ends = np.cumsum(lengths)
    within = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths)
    index = np.repeat(store.member_start[edge_slots], lengths) + within
    return store.member_ent[index], store.member_coef[index], store.member_role[index], columns


def _view(store, matrix, entity_slots, edge_slots, row_lookup) -> MatrixView:
    """Build a view. The column maps stay mutable so an append can extend them.

    Rebuilding the identity-to-column map on every append would cost the number of
    columns, which would make a loop of appends quadratic however cheap the matrix
    itself is. So the map is grown in place instead.
    """
    edge_ids = [store.edge_id(int(slot)) for slot in edge_slots]
    return MatrixView(
        matrix=matrix,
        row_of_entity={int(slot): int(row_lookup[slot]) for slot in entity_slots},
        column_of_edge={edge_id: column for column, edge_id in enumerate(edge_ids)},
        entity_of_row=tuple(store.entity_key(int(slot)) for slot in entity_slots),
        edge_of_column=edge_ids,
    )


def incidence(store, *, kinds=None, signed: bool = True) -> MatrixView:
    """Materialize an incidence matrix over the selected edge kinds.

    A signed matrix carries the coefficient of every member entry, so the two
    entries of a self-loop cancel. An unsigned matrix carries one for every
    member entry, so it reports membership rather than direction.
    """
    entity_slots, row_lookup = _row_lookup(store)
    edge_slots = _selected_edge_slots(store, kinds)
    entities, coefficients, _roles, columns = _gather_members(store, edge_slots)
    rows = row_lookup[entities]
    data = coefficients.astype(np.float32) if signed else np.ones(rows.size, dtype=np.float32)
    shape = (int(entity_slots.size), int(edge_slots.size))
    matrix = sp.coo_array((data, (rows, columns)), shape=shape).tocsc()
    matrix.eliminate_zeros()
    return _view(store, matrix, entity_slots, edge_slots, row_lookup)


def _adjacency_pairs(store, edge_slots: np.ndarray, row_lookup: np.ndarray):
    """Return the adjacency entries the selected edges imply.

    An edge joins two entities only when it has a member on each side. A one-sided
    edge, which is a boundary edge, therefore adds nothing, and no false self-loop
    appears on the node it drains.
    """
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for slot in edge_slots:
        slot = int(slot)
        members = store.members(slot)
        sources = members.entities[members.roles != ST.TARGET]
        targets = members.entities[members.roles == ST.TARGET]
        if sources.size == 0 or targets.size == 0:
            continue
        weight = float(store.edge_weight[slot])
        directed = store.is_directed(slot)
        for source in sources:
            for target in targets:
                rows.append(int(row_lookup[source]))
                columns.append(int(row_lookup[target]))
                data.append(weight)
                if not directed:
                    rows.append(int(row_lookup[target]))
                    columns.append(int(row_lookup[source]))
                    data.append(weight)
    return rows, columns, data


def adjacency(store, *, kinds=(ST.BINARY, ST.NODE_EDGE)) -> MatrixView:
    """Materialize the adjacency matrix over the edges that join two entities.

    A self-loop lands on the diagonal. A boundary edge is left out, because it
    joins nothing. A hyperedge is left out by default, because projecting one onto
    pairs is a choice that belongs to the caller rather than to this matrix.
    """
    entity_slots, row_lookup = _row_lookup(store)
    edge_slots = _selected_edge_slots(store, kinds)
    rows, columns, data = _adjacency_pairs(store, edge_slots, row_lookup)
    size = int(entity_slots.size)
    matrix = sp.coo_array(
        (
            np.asarray(data, dtype=np.float32),
            (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
        ),
        shape=(size, size),
    ).tocsr()
    matrix.eliminate_zeros()
    view = _view(store, matrix, entity_slots, edge_slots, row_lookup)
    return view._replace(column_of_edge={}, edge_of_column=[])


def laplacian(store, **kwargs) -> MatrixView:
    """Materialize the graph Laplacian, which is the degree matrix minus the adjacency.

    Every row of the result sums to zero, which is the property the Laplacian is
    used for. A self-loop is on the diagonal of the adjacency, so it raises the
    degree and cancels itself.
    """
    view = adjacency(store, **kwargs)
    degrees = np.asarray(view.matrix.sum(axis=1)).ravel()
    size = view.matrix.shape[0]
    degree_matrix = sp.diags_array(degrees, shape=(size, size), format='csr', dtype=np.float32)
    return view._replace(matrix=(degree_matrix - view.matrix).tocsr())


# ---------------------------------------------------------------------------
# The cache
# ---------------------------------------------------------------------------


class _CscBuffer:
    """A compressed sparse column matrix in buffers that a column append can grow.

    ``scipy`` offers no way to add a column to an existing matrix in place, and
    stacking two matrices copies both. So the cache keeps the three arrays of the
    format itself and appends into them, which costs the new column and nothing
    more. :meth:`matrix` then wraps the used prefix of those arrays without
    copying, so a read is free.
    """

    def __init__(self, n_rows: int):
        self.n_rows = int(n_rows)
        self.data = np.zeros(16, dtype=np.float32)
        self.indices = np.zeros(16, dtype=np.int32)
        self.indptr = np.zeros(17, dtype=np.int32)
        self.n_cols = 0
        self.nnz = 0

    def append_column(self, rows: np.ndarray, values: np.ndarray) -> None:
        """Add one column, given its row indices and their values."""
        width = int(rows.size)
        needed = self.nnz + width
        if needed > self.data.size:
            self.data = _grown_buffer(self.data, needed)
            self.indices = _grown_buffer(self.indices, needed)
        if self.n_cols + 2 > self.indptr.size:
            self.indptr = _grown_buffer(self.indptr, self.n_cols + 2)
        if width:
            self.indices[self.nnz : needed] = rows
            self.data[self.nnz : needed] = values
        self.nnz = needed
        self.n_cols += 1
        self.indptr[self.n_cols] = self.nnz

    def matrix(self):
        """Wrap the used prefix of the buffers, without copying them."""
        return sp.csc_array(
            (
                self.data[: self.nnz],
                self.indices[: self.nnz],
                self.indptr[: self.n_cols + 1],
            ),
            shape=(self.n_rows, self.n_cols),
            copy=False,
        )


def _grown_buffer(array: np.ndarray, target: int) -> np.ndarray:
    size = max(16, array.size)
    while size < target:
        size *= 2
    out = np.zeros(size, dtype=array.dtype)
    out[: array.size] = array
    return out


def _column_entries(store, edge_slot: int, row_lookup: np.ndarray, signed: bool):
    """Return the row indices and values of one edge column, duplicates summed.

    A self-loop names one entity twice, so its column holds one row index twice.
    Summing here keeps the buffer free of duplicates, which is what the format
    expects, and it is where the two entries of a self-loop cancel.
    """
    members = store.members(edge_slot)
    if members.entities.size == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    rows = row_lookup[members.entities]
    values = (
        members.coefficients.astype(np.float32) if signed else np.ones(rows.size, dtype=np.float32)
    )
    unique, inverse = np.unique(rows, return_inverse=True)
    summed = np.zeros(unique.size, dtype=np.float32)
    np.add.at(summed, inverse, values)
    keep = summed != 0.0
    return unique[keep].astype(np.int32), summed[keep]


class _Entry(NamedTuple):
    view: MatrixView
    version: int
    buffer: object = None


class MatrixCache:
    """Materialized matrices, kept against the clock of one store.

    A read returns the cached matrix while the clock has not moved. A read after a
    run of frontier appends extends the cached matrix with the new columns. Any
    other write drops the cache, because a delete or an entity insert can move a
    row or reuse a column.

    :attr:`rebuilds` and :attr:`extends` count which path each read took, so a
    benchmark and a test can tell them apart.
    """

    def __init__(self, store):
        self._store = store
        self._entries: dict[tuple, _Entry] = {}
        self.rebuilds = 0
        self.extends = 0

    def _appended_since(self, version: int):
        """Return the edge slots appended since ``version``, or None if not all were.

        The store keeps a log of the edge slots it appended at the frontier, and
        the clock value the log starts from. Any other write clears the log, so a
        gap the log cannot account for means the cache has to rebuild.
        """
        store = self._store
        if version < store.append_log_from_version:
            return None
        offset = version - store.append_log_from_version
        if offset > len(store.append_log):
            return None
        tail = store.append_log[offset:]
        if len(tail) != store.structure_version - version:
            return None
        return tail

    def _cached(self, name: tuple, build, extend):
        version = self._store.structure_version
        entry = self._entries.get(name)
        if entry is not None:
            if entry.version == version:
                return entry.view
            appended = self._appended_since(entry.version)
            if appended is not None:
                extended = extend(entry, np.asarray(appended, dtype=np.int64))
                if extended is not None:
                    self.extends += 1
                    self._entries[name] = extended._replace(version=version)
                    return extended.view
        view, buffer = build()
        self.rebuilds += 1
        self._entries[name] = _Entry(view, version, buffer)
        return view

    def incidence(self, *, kinds=None, signed: bool = True) -> MatrixView:
        """Return the incidence matrix, extending the cache after an append."""
        name = ('incidence', None if kinds is None else tuple(sorted(kinds)), signed)

        def build():
            store = self._store
            entity_slots, row_lookup = _row_lookup(store)
            edge_slots = _selected_edge_slots(store, kinds)
            buffer = _CscBuffer(int(entity_slots.size))
            for slot in edge_slots:
                buffer.append_column(*_column_entries(store, int(slot), row_lookup, signed))
            view = _view(store, buffer.matrix(), entity_slots, edge_slots, row_lookup)
            return view, (buffer, row_lookup)

        def extend(entry, appended):
            store = self._store
            buffer, row_lookup = entry.buffer
            selected = appended
            if kinds is not None and selected.size:
                wanted = np.asarray(tuple(kinds), dtype=np.uint8)
                selected = selected[np.isin(store.edge_kind[selected], wanted)]
            if selected.size == 0:
                return entry
            view = entry.view
            for slot in selected:
                buffer.append_column(*_column_entries(store, int(slot), row_lookup, signed))
                edge_id = store.edge_id(int(slot))
                view.column_of_edge[edge_id] = len(view.edge_of_column)
                view.edge_of_column.append(edge_id)
            return entry._replace(view=view._replace(matrix=buffer.matrix()))

        return self._cached(name, build, extend)

    def adjacency(self, **kwargs) -> MatrixView:
        """Return the adjacency matrix. An append extends it in place."""
        name = ('adjacency', tuple(sorted(kwargs.items())))

        def build():
            return adjacency(self._store, **kwargs), None

        # An appended edge adds entries to rows that already exist, so an
        # adjacency matrix has no append-only structure to exploit and it
        # rebuilds. The fast path belongs to a matrix whose columns are edges.
        return self._cached(name, build, lambda entry, appended: None)

    def laplacian(self, **kwargs) -> MatrixView:
        """Return the Laplacian. It follows the adjacency, so it is built from it."""
        name = ('laplacian', tuple(sorted(kwargs.items())))

        def build():
            return laplacian(self._store, **kwargs), None

        return self._cached(name, build, lambda entry, appended: None)

    def drop(self) -> None:
        """Forget every cached matrix. A cache is always safe to drop."""
        self._entries.clear()


# ---------------------------------------------------------------------------
# The named matrices
# ---------------------------------------------------------------------------
# One member list feeds several purpose-built matrices. Each name below selects the
# edges that belong in it, so no matrix has to carry a convention that another one
# needs. The public API of the graph exposes these as G.B, G.H, G.S, G.A and G.L.


def binary_incidence(store) -> MatrixView:
    """The incidence matrix of the binary edges, which the public API calls ``B``."""
    return incidence(store, kinds=(ST.BINARY, ST.NODE_EDGE), signed=True)


def hypergraph_incidence(store) -> MatrixView:
    """The incidence matrix of the hyperedges, which the public API calls ``H``."""
    return incidence(store, kinds=(ST.HYPER,), signed=False)


def signed_incidence(store) -> MatrixView:
    """The coefficient incidence matrix of every edge, which the public API calls ``S``."""
    return incidence(store, kinds=None, signed=True)
