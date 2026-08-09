"""A self-loop and a boundary edge must stay distinct in the slot store.

This is the test that gates the rest of the slot-addressed core. The record core
loses the difference: a directed self-loop writes the source coefficient and then
the target coefficient into the same cell, so the second overwrites the first and
the column keeps one entry. A sink boundary edge on the same node keeps one entry
too, with the same value, so nothing tells the two apart.

The slot store keeps one member entry per role. An entity that takes two roles in
one edge therefore appears twice in that edge. The entry count and the roles carry
the difference, and each matrix then applies its own rule.
"""

from __future__ import annotations

import numpy as np
import pytest

import unittest

from annnet.core import _matrices as M, _store as ST, _structure as S
from annnet.core.graph import AnnNet

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


@pytest.fixture
def store():
    state = ST.CoreState(directed=True)
    for node_id in ('A', 'B'):
        state.add_entity(key(node_id))
    return state


def add_self_loop(state, *, weight=0.5, edge_id='e_loop'):
    """One directed edge from a node to itself."""
    return state.add_edge(
        edge_id,
        [(key('A'), weight, ST.SOURCE), (key('A'), -weight, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=weight,
    )


def add_sink_boundary(state, *, weight=1.0, edge_id='b_out'):
    """One edge that drains a node and has no other side."""
    return state.add_edge(
        edge_id,
        [(key('A'), -weight, ST.SOURCE)],
        kind=ST.HYPER,
        directed=False,
        weight=weight,
        explicit_coefficients=True,
    )


# ---------------------------------------------------------------------------
# The store keeps them apart
# ---------------------------------------------------------------------------


def test_a_self_loop_holds_two_member_entries_on_one_slot(store):
    slot = add_self_loop(store)
    members = store.members(slot)
    assert len(members.entities) == 2
    assert set(members.entities) == {store.entity_slot(key('A'))}
    assert sorted(members.roles) == [ST.TARGET, ST.SOURCE]


def test_a_boundary_edge_holds_one_member_entry(store):
    slot = add_sink_boundary(store)
    members = store.members(slot)
    assert len(members.entities) == 1
    assert list(members.roles) == [ST.SOURCE]


def test_the_two_shapes_are_distinguishable_from_the_store_alone(store):
    loop = add_self_loop(store)
    boundary = add_sink_boundary(store)
    assert store.members(loop).entities.size != store.members(boundary).entities.size
    assert store.is_self_loop(loop) is True
    assert store.is_self_loop(boundary) is False
    assert store.is_boundary(boundary) is True
    assert store.is_boundary(loop) is False


def test_an_undirected_self_loop_also_holds_two_entries(store):
    slot = store.add_edge(
        'u_loop',
        [(key('A'), 2.0, ST.SOURCE), (key('A'), 2.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=False,
        weight=2.0,
    )
    assert len(store.members(slot).entities) == 2
    assert store.is_self_loop(slot) is True


def test_the_two_sides_of_a_self_loop_both_report_the_node(store):
    slot = add_self_loop(store)
    sides = store.endpoints(slot)
    assert sides.source == frozenset({key('A')})
    assert sides.target == frozenset({key('A')})


def test_a_boundary_edge_leaves_its_open_side_empty(store):
    slot = add_sink_boundary(store)
    sides = store.endpoints(slot)
    assert sides.source == frozenset({key('A')})
    assert sides.target == frozenset()


def test_degree_counts_a_self_loop_twice(store):
    add_self_loop(store)
    store.add_edge(
        'e_ab',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    assert store.degree(key('A')) == 3
    assert store.degree(key('B')) == 1


def test_degree_counts_a_boundary_edge_once(store):
    add_sink_boundary(store)
    assert store.degree(key('A')) == 1


# ---------------------------------------------------------------------------
# Each matrix applies its own rule
# ---------------------------------------------------------------------------


def _column(matrix, col):
    block = matrix[:, [col]].tocoo()
    return {
        int(r): float(v) for r, v in zip(block.row, block.data, strict=False) if float(v) != 0.0
    }


def test_signed_incidence_sums_the_two_entries_of_a_self_loop(store):
    add_self_loop(store)
    view = M.incidence(store, signed=True)
    column = _column(view.matrix, view.column_of_edge['e_loop'])
    assert column == {}, 'the two entries of a self-loop cancel in a signed incidence'


def test_signed_incidence_keeps_the_single_entry_of_a_boundary_edge(store):
    add_sink_boundary(store)
    view = M.incidence(store, signed=True)
    column = _column(view.matrix, view.column_of_edge['b_out'])
    row = view.row_of_entity[key('A')]
    assert column == {row: pytest.approx(-1.0)}


def test_the_two_shapes_differ_in_the_signed_incidence(store):
    add_self_loop(store)
    add_sink_boundary(store)
    view = M.incidence(store, signed=True)
    loop = _column(view.matrix, view.column_of_edge['e_loop'])
    boundary = _column(view.matrix, view.column_of_edge['b_out'])
    assert loop != boundary


def test_hypergraph_incidence_keeps_one_entry_per_member(store):
    add_sink_boundary(store)
    store.add_edge(
        'h_ab',
        [(key('A'), 1.0, ST.MEMBER), (key('B'), 1.0, ST.MEMBER)],
        kind=ST.HYPER,
        directed=False,
        weight=1.0,
    )
    view = M.incidence(store, kinds=(ST.HYPER,), signed=False)
    assert len(_column(view.matrix, view.column_of_edge['b_out'])) == 1
    assert len(_column(view.matrix, view.column_of_edge['h_ab'])) == 2


def test_adjacency_gives_a_self_loop_a_diagonal_entry(store):
    add_self_loop(store)
    view = M.adjacency(store)
    row = view.row_of_entity[key('A')]
    dense = np.asarray(view.matrix.todense())
    assert dense[row, row] != 0.0


def test_adjacency_leaves_a_boundary_edge_out(store):
    add_sink_boundary(store)
    view = M.adjacency(store)
    dense = np.asarray(view.matrix.todense())
    assert not dense.any(), 'a boundary edge must not create an adjacency entry'


def test_adjacency_holds_both_a_self_loop_and_a_plain_edge(store):
    add_self_loop(store)
    add_sink_boundary(store)
    store.add_edge(
        'e_ab',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    view = M.adjacency(store)
    dense = np.asarray(view.matrix.todense())
    row_a = view.row_of_entity[key('A')]
    row_b = view.row_of_entity[key('B')]
    assert dense[row_a, row_a] != 0.0, 'the self-loop is on the diagonal'
    assert dense[row_a, row_b] != 0.0, 'the plain edge is off the diagonal'


def test_the_laplacian_follows_the_adjacency(store):
    add_self_loop(store)
    store.add_edge(
        'e_ab',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    laplacian = M.laplacian(store).matrix
    assert np.allclose(np.asarray(laplacian.todense()).sum(axis=1), 0.0)


# ---------------------------------------------------------------------------
# The matrix the graph exposes applies the same rules
# ---------------------------------------------------------------------------
# The graph builds ``X()`` from the member lists, so the rules above are the ones
# a user sees. The record core answered differently on a self-loop, which is the
# defect this file names, so these graphs are built on the slot store by name.


def graph():
    """A directed graph with a self-loop next to a plain edge."""
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'A', edge_id='e_loop', weight=0.5)
    G.add_edges('A', 'B', edge_id='e_ab', weight=1.0)
    return G


def test_the_graph_matrix_cancels_the_two_entries_of_a_self_loop():
    G = graph()
    column = _column(G.S.tocsc(), G.idx.edge_to_col('e_loop'))
    assert column == {}, 'the two entries of a self-loop cancel in the incidence matrix'


def test_the_graph_matrix_keeps_the_plain_edge_beside_it():
    G = graph()
    column = _column(G.S.tocsc(), G.idx.edge_to_col('e_ab'))
    assert column == {
        G.idx.entity_to_row('A'): pytest.approx(1.0),
        G.idx.entity_to_row('B'): pytest.approx(-1.0),
    }


def test_a_placeholder_edge_occupies_no_column_of_the_graph_matrix():
    G = graph()
    rows, columns = G.S.shape
    G._ensure_edge_entity_placeholder('e_later')
    # The placeholder is an entity the graph now knows, so it takes a row. It
    # holds no members, so it takes no column.
    assert G.S.shape == (rows + 1, columns)


# ---------------------------------------------------------------------------
# What a caller can say about a self-loop, and what it cannot
# ---------------------------------------------------------------------------
# `FR-022`, and decision `D6` of cycle 003. The store holds a self-loop as two
# member entries on one entity slot, and the two entries may carry different
# coefficients. Nothing above the store can say so: a coefficient is addressed by
# endpoint, at every layer between the public call and the file, and a self-loop
# names one endpoint twice. `D6` records that the addressing stays as it is in
# this cycle and names what a caller states instead, and these fix both halves of
# it so the limit is a tested fact rather than something rediscovered.


def _loop_graph() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B'])
    graph.add_edges('A', 'A', edge_id='loop')
    return graph


def _boundary_pair() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A'])
    graph.add_edges([{'members': ['A'], 'edge_id': 'out'}])
    graph.add_edges([{'members': ['A'], 'edge_id': 'into'}])
    graph.set_edge_coeffs('out', {'A': 2.0})
    graph.set_edge_coeffs('into', {'A': -3.0})
    return graph


class TestTheStoreHoldsWhatTheCallerCannotState(unittest.TestCase):
    def test_a_self_loop_holds_one_entry_per_role(self):
        store = _loop_graph()._store
        members = store.members(store.edge_slot('loop'))
        self.assertEqual(members.entities.tolist(), [0, 0])
        self.assertEqual(sorted(members.roles.tolist()), [-1, 1])

    def test_the_two_entries_are_independent_in_the_store(self):
        store = _loop_graph()._store
        slot = store.edge_slot('loop')
        members = store.members(slot)
        members.coefficients[0] = 2.0
        members.coefficients[1] = -3.0
        self.assertEqual(store.members(slot).coefficients.tolist(), [2.0, -3.0])

    def test_setting_a_coefficient_reaches_both_entries(self):
        graph = _loop_graph()
        graph.set_edge_coeffs('loop', {'A': 2.0})
        members = graph._store.members(graph._store.edge_slot('loop'))
        self.assertEqual(members.coefficients.tolist(), [2.0, 2.0])

    def test_the_read_back_is_one_value_and_it_is_not_the_one_that_was_set(self):
        """The sharpest form of the limit, and the reason `D6` names it.

        A coefficient is addressed by endpoint on the way in and on the way out.
        Going in, one value reaches both entries of the self-loop. Coming out,
        the two entries are summed under the one endpoint that names them. So a
        set followed by a get doubles the value, and no spelling of the call
        avoids it.
        """
        graph = _loop_graph()
        graph.set_edge_coeffs('loop', {'A': 2.0})
        self.assertEqual(S.edge_coefficients(graph, 'loop'), {'A': 4.0})

    def test_a_definition_states_one_coefficient_per_endpoint(self):
        graph = _loop_graph()
        graph.set_edge_coeffs('loop', {'A': 2.0})
        self.assertEqual(set(S.edge_definition(graph, 'loop').coefficients), {'A'})

    def test_the_record_says_what_it_cannot_describe(self):
        self.assertIn('two different coefficients', S.EdgeDefinition.__doc__)
        self.assertIn('self-loop', S.EdgeDefinition.__doc__)


class TestWhatACallerStatesInstead(unittest.TestCase):
    """`D6`: two boundary edges, which is not the same object and says so."""

    def test_the_pair_carries_the_two_coefficients(self):
        graph = _boundary_pair()
        self.assertEqual(S.edge_coefficients(graph, 'out'), {'A': 2.0})
        self.assertEqual(S.edge_coefficients(graph, 'into'), {'A': -3.0})

    def test_the_pair_is_two_columns_and_not_one(self):
        self.assertEqual(_boundary_pair().S.shape[1], 2)

    def test_a_self_loop_column_cancels_where_the_pair_does_not(self):
        """Why the two are not interchangeable, in one number.

        The two entries of a self-loop land in one cell and cancel, so its column
        is empty. Two boundary edges hold two cells that do not.
        """
        self.assertEqual(_loop_graph().S.nnz, 0)
        self.assertEqual(_boundary_pair().S.nnz, 2)

    def test_the_pair_round_trips_through_the_native_format(self):
        import tempfile
        from pathlib import Path

        from annnet.io.annnet_format import read as annnet_read
        from annnet.io.annnet_format import write as annnet_write

        graph = _boundary_pair()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pair.annnet'
            annnet_write(graph, path)
            restored = annnet_read(path)
        self.assertEqual(S.edge_coefficients(restored, 'out'), {'A': 2.0})
        self.assertEqual(S.edge_coefficients(restored, 'into'), {'A': -3.0})
