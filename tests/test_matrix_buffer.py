"""The buffer of a cached matrix borrows until it grows.

`FR-012` and `FR-013`. A cached matrix is kept in the three arrays of the
compressed-sparse-column format, so that appending an edge costs the new column
and nothing more. The buffer was seeded by copying those three arrays out of the
matrix a build had just produced, because the buffer writes past the end of what
it holds and a matrix handed out earlier must not see that.

But a buffer that never grows never writes, so the copy is paid by every rebuild
and earned by only the ones an append follows. The buffer therefore borrows on
seed and takes ownership at its first growth — once, in one direction, and at the
one place a write can happen.
"""

from __future__ import annotations

import numpy as np

from annnet import AnnNet
from annnet.core import _matrices as M


def _graph(edges: int = 32) -> AnnNet:
    nodes = max(2, edges // 2)
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(nodes)])
    graph.add_edges(
        [
            {'source': f'v{i % nodes}', 'target': f'v{(i + 1) % nodes}', 'edge_id': f'e{i}'}
            for i in range(edges)
        ]
    )
    return graph


def _built(graph):
    store = graph._store
    entity_slots, row_lookup = M._row_lookup(store)
    edge_slots = M._selected_edge_slots(store, None)
    return M._incidence_matrix(store, entity_slots.size, edge_slots, row_lookup, True)


class TestSeedingBorrows:
    def test_the_three_arrays_are_the_ones_the_matrix_holds(self):
        matrix = _built(_graph())
        buffer = M._CscBuffer.of(matrix)
        assert buffer.owns is False
        assert buffer.data is matrix.data
        assert buffer.indices is matrix.indices
        assert buffer.indptr is matrix.indptr

    def test_the_matrix_it_hands_back_is_a_window_on_them(self):
        matrix = _built(_graph())
        buffer = M._CscBuffer.of(matrix)
        wrapped = buffer.matrix()
        assert wrapped.shape == matrix.shape
        assert (wrapped != matrix).nnz == 0

    def test_a_fresh_buffer_owns_what_it_allocated(self):
        assert M._CscBuffer(4).owns is True


class TestTheFirstGrowthTakesOwnership:
    def test_ownership_moves_once_and_in_one_direction(self):
        matrix = _built(_graph())
        buffer = M._CscBuffer.of(matrix)
        assert buffer.owns is False

        buffer.append_column(np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))
        assert buffer.owns is True
        held = buffer.data

        buffer.append_column(np.array([1], dtype=np.int32), np.array([1.0], dtype=np.float32))
        assert buffer.owns is True
        # A second append does not copy again; the arrays only grow when full.
        assert buffer.data is held or buffer.data.size > held.size

    def test_the_matrix_it_was_seeded_from_is_left_alone(self):
        matrix = _built(_graph())
        before = matrix.data.copy()
        buffer = M._CscBuffer.of(matrix)
        for i in range(8):
            buffer.append_column(
                np.array([i % matrix.shape[0]], dtype=np.int32),
                np.array([9.0], dtype=np.float32),
            )
        assert np.array_equal(matrix.data, before)

    def test_a_matrix_handed_out_before_the_growth_is_left_alone(self):
        matrix = _built(_graph())
        buffer = M._CscBuffer.of(matrix)
        handed = buffer.matrix()
        snapshot = handed.toarray()
        for i in range(8):
            buffer.append_column(
                np.array([i % matrix.shape[0]], dtype=np.int32),
                np.array([9.0], dtype=np.float32),
            )
        assert np.array_equal(handed.toarray(), snapshot)

    def test_dropping_a_column_writes_nothing_and_keeps_the_borrow(self):
        matrix = _built(_graph())
        buffer = M._CscBuffer.of(matrix)
        columns = buffer.n_cols
        buffer.drop_last_column()
        assert buffer.owns is False
        assert buffer.n_cols == columns - 1
        assert buffer.matrix().shape[1] == columns - 1


class TestThroughTheCache:
    def test_a_rebuild_that_is_never_appended_to_copies_nothing(self):
        graph = _graph()
        _ = graph.S
        entry = next(iter(graph.matrices.cache._entries.values()))
        buffer, _lookup = entry.buffer
        assert buffer.owns is False

    def test_an_append_after_a_read_takes_ownership(self):
        graph = _graph()
        _ = graph.S
        graph.add_edges('v0', 'v3', edge_id='later')
        _ = graph.S
        entry = next(iter(graph.matrices.cache._entries.values()))
        buffer, _lookup = entry.buffer
        assert buffer.owns is True

    def test_the_matrix_is_the_same_either_way(self):
        graph = _graph()
        first = graph.S.toarray()
        graph.matrices.cache.drop()
        assert np.array_equal(graph.S.toarray(), first)


class TestTheInvariant:
    """A borrowing buffer has written to none of its arrays."""

    def test_a_borrowing_buffer_passes(self):
        from annnet.core._validate import validate_internal_consistency

        graph = _graph()
        _ = graph.S
        assert validate_internal_consistency(graph, strict=False) == []

    def test_a_buffer_that_wrote_while_borrowing_is_reported(self):
        from annnet.core._validate import validate_internal_consistency

        graph = _graph()
        _ = graph.S
        entry = next(iter(graph.matrices.cache._entries.values()))
        buffer, _lookup = entry.buffer
        # Force the state the invariant exists to catch.
        buffer.append_column(np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32))
        buffer.owns = False
        problems = validate_internal_consistency(graph, strict=False)
        assert any('borrow' in problem for problem in problems)
