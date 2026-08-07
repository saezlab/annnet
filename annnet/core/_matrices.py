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

from typing import Any, NamedTuple

import numpy as np
import scipy.sparse as sp

from . import _store as ST
from .._session import logger

log = logger(__name__)


class MatrixView(NamedTuple):
    """One materialized matrix and the maps from identity to position.

    An entity is addressed by its entity key and an edge by its edge id, because
    those are what a caller holds. A position belongs to this matrix and to
    nothing else. Read a position only to index this matrix.
    """

    # A scipy sparse array or a numpy array, per matrix. Nothing here reads it.
    matrix: Any
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


def _selected(store, slots: np.ndarray, kinds) -> np.ndarray:
    """Return the subset of ``slots`` that belongs in a matrix over ``kinds``.

    A placeholder edge is an id the graph knows before the edge exists. It holds
    no members and occupies no column, so it is never one of these, and asking
    for every kind still leaves it out.
    """
    if slots.size == 0:
        return slots
    if kinds is None:
        return slots[store.edge_kind[slots] != ST.PLACEHOLDER]
    wanted = np.asarray(tuple(kinds), dtype=np.uint8)
    return slots[np.isin(store.edge_kind[slots], wanted)]


def _selected_edge_slots(store, kinds) -> np.ndarray:
    """Return the edge slots a matrix is built over, in slot order."""
    return _selected(store, store.live_edge_slots(), kinds)


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


def _view(store, matrix, entity_slots, edge_slots=None) -> MatrixView:
    """Build a view. The column maps stay mutable so an append can extend them.

    Rebuilding the identity-to-column map on every append would cost the number of
    columns, which would make a loop of appends quadratic however cheap the matrix
    itself is. So the map is grown in place instead.

    A matrix whose columns are not edges passes no ``edge_slots`` and gets no
    column maps, which is what the adjacency and the Laplacian do.
    """
    edge_ids = [] if edge_slots is None else store.edge_ids_at(edge_slots)
    entity_keys = tuple(store.entity_keys_at(entity_slots))
    return MatrixView(
        matrix=matrix,
        row_of_entity={key: row for row, key in enumerate(entity_keys)},
        column_of_edge={edge_id: column for column, edge_id in enumerate(edge_ids)},
        entity_of_row=entity_keys,
        edge_of_column=edge_ids,
    )


def _incidence_matrix(store, n_rows: int, edge_slots, row_lookup, signed: bool):
    """Build the incidence matrix of the selected edges, in one gather, as CSC.

    Every member entry of every selected edge is placed in one pass. The two
    entries a self-loop puts in one cell are summed by the conversion, which is
    where they cancel, and the zero they leave is then dropped.
    """
    entities, coefficients, _roles, columns = _gather_members(store, edge_slots)
    rows = row_lookup[entities]
    data = coefficients.astype(np.float32) if signed else np.ones(rows.size, dtype=np.float32)
    matrix = sp.coo_array(
        (data, (rows, columns)), shape=(int(n_rows), int(edge_slots.size))
    ).tocsc()
    matrix.eliminate_zeros()
    return matrix


def incidence(store, *, kinds=None, signed: bool = True) -> MatrixView:
    """Materialize an incidence matrix over the selected edge kinds.

    A signed matrix carries the coefficient of every member entry, so the two
    entries of a self-loop cancel. An unsigned matrix carries one for every
    member entry, so it reports membership rather than direction.
    """
    entity_slots, row_lookup = _row_lookup(store)
    edge_slots = _selected_edge_slots(store, kinds)
    matrix = _incidence_matrix(store, entity_slots.size, edge_slots, row_lookup, signed)
    return _view(store, matrix, entity_slots, edge_slots)


def _adjacency_pairs(store, edge_slots: np.ndarray, row_lookup: np.ndarray):
    """Return the adjacency entries the selected edges imply.

    An edge joins two entities only when it has a member on each side. A one-sided
    edge, which is a boundary edge, therefore adds nothing, and no false self-loop
    appears on the node it drains.

    Almost every edge that reaches here has one member on each side, so that case
    is paired for every edge at once rather than one edge at a time. An edge with
    a wider side, which is a directed hyperedge, needs the outer product of its two
    sides and is handled on its own. Pairing one edge at a time cost about eight
    times an incidence build.
    """
    empty = (
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
    )
    if edge_slots.size == 0:
        return empty

    entities, _coefficients, roles, columns = _gather_members(store, edge_slots)
    if entities.size == 0:
        return empty

    is_target = roles == ST.TARGET
    width = int(edge_slots.size)
    source_count = np.bincount(columns[~is_target], minlength=width)
    target_count = np.bincount(columns[is_target], minlength=width)

    weights = store.edge_weight[edge_slots].astype(np.float32)
    declared = store.edge_directed[edge_slots]
    default_directed = True if store.directed is None else bool(store.directed)
    directed = np.where(declared == ST.INHERIT, default_directed, declared.astype(bool))

    rows: list = []
    cols: list = []
    data: list = []

    # The simple shape: one member on each side.
    simple = (source_count == 1) & (target_count == 1)
    if simple.any():
        simple_columns = np.flatnonzero(simple)
        keep = np.isin(columns, simple_columns)
        order = np.argsort(columns[keep], kind='stable')
        kept_columns = columns[keep][order]
        kept_entities = entities[keep][order]
        kept_is_target = is_target[keep][order]
        source_rows = row_lookup[kept_entities[~kept_is_target]]
        target_rows = row_lookup[kept_entities[kept_is_target]]
        # Both sides are now in edge order, so they line up one to one.
        edge_of_pair = kept_columns[~kept_is_target]
        pair_weights = weights[edge_of_pair]
        rows.append(source_rows)
        cols.append(target_rows)
        data.append(pair_weights)
        undirected = ~directed[edge_of_pair]
        if undirected.any():
            rows.append(target_rows[undirected])
            cols.append(source_rows[undirected])
            data.append(pair_weights[undirected])

    # The wide shape: a side with more than one member needs an outer product.
    wide = (~simple) & (source_count > 0) & (target_count > 0)
    for column in np.flatnonzero(wide):
        members = store.members(int(edge_slots[column]))
        member_is_target = members.roles == ST.TARGET
        sources = members.entities[~member_is_target]
        targets = members.entities[member_is_target]
        source_rows = row_lookup[np.repeat(sources, targets.size)]
        target_rows = row_lookup[np.tile(targets, sources.size)]
        pair_weights = np.full(source_rows.size, weights[column], dtype=np.float32)
        rows.append(source_rows)
        cols.append(target_rows)
        data.append(pair_weights)
        if not directed[column]:
            rows.append(target_rows)
            cols.append(source_rows)
            data.append(pair_weights)

    if not rows:
        return empty
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(data)


def _adjacency_matrix(store, *, kinds=(ST.BINARY, ST.NODE_EDGE)):
    """Return the adjacency matrix and the entity slots its rows stand for."""
    entity_slots, row_lookup = _row_lookup(store)
    edge_slots = _selected_edge_slots(store, kinds)
    rows, columns, data = _adjacency_pairs(store, edge_slots, row_lookup)
    size = int(entity_slots.size)
    matrix = sp.coo_array(
        (data, (rows.astype(np.int64), columns.astype(np.int64))), shape=(size, size)
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix, entity_slots


def adjacency(store, **kwargs) -> MatrixView:
    """Materialize the adjacency matrix over the edges that join two entities.

    A self-loop lands on the diagonal. A boundary edge is left out, because it
    joins nothing. A hyperedge is left out by default, because projecting one onto
    pairs is a choice that belongs to the caller rather than to this matrix.

    Both axes are entities, so the view carries no column map.
    """
    matrix, entity_slots = _adjacency_matrix(store, **kwargs)
    return _view(store, matrix, entity_slots)


def _laplacian_matrix(store, **kwargs):
    """Return the Laplacian and the entity slots its rows stand for."""
    matrix, entity_slots = _adjacency_matrix(store, **kwargs)
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    size = matrix.shape[0]
    degree_matrix = sp.diags_array(degrees, shape=(size, size), format='csr', dtype=np.float32)
    return (degree_matrix - matrix).tocsr(), entity_slots


def laplacian(store, **kwargs) -> MatrixView:
    """Materialize the graph Laplacian, which is the degree matrix minus the adjacency.

    Every row of the result sums to zero, which is the property the Laplacian is
    used for. A self-loop is on the diagonal of the adjacency, so it raises the
    degree and cancels itself.
    """
    matrix, entity_slots = _laplacian_matrix(store, **kwargs)
    return _view(store, matrix, entity_slots)


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

    @classmethod
    def of(cls, matrix) -> _CscBuffer:
        """Take over a matrix that is already built, so an append can extend it.

        Filling a buffer a column at a time costs a Python call and a sort per
        edge, which is two orders of magnitude more than placing every member
        entry of every edge in one pass. So a build goes through
        :func:`_incidence_matrix` and hands the result here.

        The arrays are copied rather than aliased, because the buffer writes
        past the end of what it holds and a matrix handed out earlier must not
        see that.
        """
        buffer = cls(matrix.shape[0])
        buffer.data = np.array(matrix.data, dtype=np.float32)
        buffer.indices = np.array(matrix.indices, dtype=np.int32)
        buffer.indptr = np.array(matrix.indptr, dtype=np.int32)
        buffer.n_cols = int(matrix.shape[1])
        buffer.nnz = int(matrix.nnz)
        return buffer

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


class _Entry:
    """One cached matrix, and what it takes to name the positions it holds.

    The identity maps of a view cost a Python entry per row and per column,
    which is more than the matrix itself costs to build. Most callers want the
    matrix alone — that is what the graph asks for on every read — so the maps
    are built by the first call that asks for them and not before.
    """

    def __init__(self, store, matrix, version, *, entity_slots, edge_slots=None, buffer=None):
        self.store = store
        self.matrix = matrix
        self.version = version
        self.entity_slots = entity_slots
        self.edge_slots = edge_slots
        self.buffer = buffer
        self._view = None

    def view(self) -> MatrixView:
        """Return the view of this matrix, building its maps at most once."""
        if self._view is None:
            self._view = _view(self.store, self.matrix, self.entity_slots, self.edge_slots)
        elif self._view.matrix is not self.matrix:
            self._view = self._view._replace(matrix=self.matrix)
        return self._view

    def note_appended(self, edge_slots) -> None:
        """Record the columns an extend has just added to the matrix.

        Growing the map in place is what keeps a run of appends linear. A map
        that was never built stays unbuilt, because the slots it would be built
        from are recorded here too.
        """
        if self._view is not None:
            column_of_edge = self._view.column_of_edge
            edge_of_column = self._view.edge_of_column
            for edge_id in self.store.edge_ids_at(edge_slots):
                column_of_edge[edge_id] = len(edge_of_column)
                edge_of_column.append(edge_id)
        self.edge_slots = np.concatenate((self.edge_slots, edge_slots))


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

    def _cached(self, name: tuple, build, extend) -> _Entry:
        version = self._store.structure_version
        entry = self._entries.get(name)
        if entry is not None:
            if entry.version == version:
                return entry
            appended = self._appended_since(entry.version)
            if appended is not None and extend(entry, np.asarray(appended, dtype=np.int64)):
                self.extends += 1
                entry.version = version
                return entry
        entry = build()
        self.rebuilds += 1
        self._entries[name] = entry
        return entry

    def _incidence(self, kinds, signed: bool) -> _Entry:
        name = ('incidence', None if kinds is None else tuple(sorted(kinds)), signed)

        def build():
            store = self._store
            entity_slots, row_lookup = _row_lookup(store)
            edge_slots = _selected_edge_slots(store, kinds)
            buffer = _CscBuffer.of(
                _incidence_matrix(store, entity_slots.size, edge_slots, row_lookup, signed)
            )
            return _Entry(
                store,
                buffer.matrix(),
                store.structure_version,
                entity_slots=entity_slots,
                edge_slots=edge_slots,
                buffer=(buffer, row_lookup),
            )

        def extend(entry, appended):
            store = self._store
            selected = _selected(store, appended, kinds)
            if selected.size:
                buffer, row_lookup = entry.buffer
                for slot in selected:
                    buffer.append_column(*_column_entries(store, int(slot), row_lookup, signed))
                entry.matrix = buffer.matrix()
                entry.note_appended(selected)
            return True

        return self._cached(name, build, extend)

    def incidence(self, *, kinds=None, signed: bool = True) -> MatrixView:
        """Return the incidence matrix and the maps that name its positions."""
        return self._incidence(kinds, signed).view()

    def incidence_matrix(self, *, kinds=None, signed: bool = True):
        """Return the incidence matrix alone, extending the cache after an append."""
        return self._incidence(kinds, signed).matrix

    # An appended edge adds entries to rows that already exist, so neither of the
    # two matrices below has an append-only structure to exploit and both
    # rebuild. The fast path belongs to a matrix whose columns are edges.

    def _square(self, name: str, build_matrix, kwargs) -> _Entry:
        def build():
            store = self._store
            matrix, entity_slots = build_matrix(store, **kwargs)
            return _Entry(store, matrix, store.structure_version, entity_slots=entity_slots)

        return self._cached((name, tuple(sorted(kwargs.items()))), build, lambda entry, _: False)

    def adjacency(self, **kwargs) -> MatrixView:
        """Return the adjacency matrix and the map from an entity to its row."""
        return self._square('adjacency', _adjacency_matrix, kwargs).view()

    def adjacency_matrix(self, **kwargs):
        """Return the adjacency matrix alone."""
        return self._square('adjacency', _adjacency_matrix, kwargs).matrix

    def laplacian(self, **kwargs) -> MatrixView:
        """Return the Laplacian. It follows the adjacency, so it is built from it."""
        return self._square('laplacian', _laplacian_matrix, kwargs).view()

    def laplacian_matrix(self, **kwargs):
        """Return the Laplacian alone."""
        return self._square('laplacian', _laplacian_matrix, kwargs).matrix

    def drop(self) -> None:
        """Forget every cached matrix. A cache is always safe to drop."""
        self._entries.clear()
