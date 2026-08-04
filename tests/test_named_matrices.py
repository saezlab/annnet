"""The named matrices, and the namespace behind them.

One member list feeds several matrices. Each name selects the edges that belong
in it, so no matrix carries a convention another one needs. These tests pin what
each name reports and, for the shapes the specification calls out, where a
self-loop and a boundary edge land.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet.core import _structure as S
from annnet.core.graph import AnnNet

from ._fixtures import CASE_NAMES, build_case

NAMES = ('B', 'H', 'S', 'A', 'L')


def _mixed() -> AnnNet:
    """Two binary edges, one hyperedge, and a node they all name."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e_ab')
    G.add_edges('B', 'C', edge_id='e_bc')
    G.add_edges([{'members': ['A', 'C', 'D'], 'edge_id': 'h1'}])
    return G


# ---------------------------------------------------------------------------
# What each name selects
# ---------------------------------------------------------------------------


def test_the_binary_incidence_holds_the_binary_edges_alone():
    G = _mixed()
    assert G.B.shape == (4, 2)
    assert list(G.matrices.binary().edge_of_column) == ['e_ab', 'e_bc']


def test_the_hypergraph_incidence_holds_the_hyperedges_alone():
    G = _mixed()
    assert G.H.shape == (4, 1)
    assert list(G.matrices.hypergraph().edge_of_column) == ['h1']


def test_the_hypergraph_incidence_reports_membership_rather_than_direction():
    """It is unsigned, so every member of a hyperedge carries the same entry."""
    G = _mixed()
    assert set(G.H.tocoo().data.tolist()) == {1.0}


def test_the_signed_incidence_holds_every_structural_edge():
    G = _mixed()
    assert G.S.shape == (4, 3)
    assert list(G.matrices.signed().edge_of_column) == S.edge_ids(G)


def test_the_signed_incidence_carries_the_coefficients_the_user_set():
    G = _mixed()
    G.set_edge_coeffs('e_ab', {'A': -2.0, 'B': 3.0})
    view = G.matrices.signed()
    column = view.matrix.tocsc()[:, [view.column_of_edge['e_ab']]].toarray().ravel()
    assert column[view.row_of_entity[('A', ('_',))]] == pytest.approx(-2.0)
    assert column[view.row_of_entity[('B', ('_',))]] == pytest.approx(3.0)


def test_the_adjacency_leaves_out_the_hyperedges():
    """Projecting a hyperedge onto pairs is a choice that belongs to the caller."""
    G = _mixed()
    view = G.matrices.adjacency()
    found = {
        (view.entity_of_row[row][0], view.entity_of_row[col][0])
        for row, col in zip(*G.A.nonzero(), strict=False)
    }
    assert found == {('A', 'B'), ('B', 'C')}


def test_every_row_of_the_laplacian_sums_to_zero():
    G = _mixed()
    assert np.allclose(np.asarray(G.L.sum(axis=1)).ravel(), 0.0)


# ---------------------------------------------------------------------------
# The shapes the specification calls out
# ---------------------------------------------------------------------------


def test_a_self_loop_cancels_in_the_signed_incidence_and_sits_on_the_diagonal():
    G = build_case('self_loop')
    view = G.matrices.signed()
    edge_id = S.edge_ids(G)[0]
    column = view.matrix.tocsc()[:, [view.column_of_edge[edge_id]]]
    assert column.nnz == 0, 'the two entries of a self-loop cancel'
    view = G.matrices.adjacency()
    row = view.row_of_entity[next(iter(S.edge_endpoints(G, edge_id).source))]
    assert G.A[row, row] != 0.0


def test_a_boundary_edge_joins_nothing_so_it_is_not_in_the_adjacency():
    """It holds one member, so there is no pair for the adjacency to record."""
    G = build_case('boundary_edge')
    view = G.matrices.signed()
    for edge_id in ('b_out', 'b_in'):
        assert view.matrix.tocsc()[:, [view.column_of_edge[edge_id]]].nnz == 1
    assert G.A.nnz == 1, 'the one binary edge of the case, and neither boundary'


# ---------------------------------------------------------------------------
# Rows agree with the graph, whatever the name selects
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_every_named_matrix_has_one_row_per_entity(case):
    G = build_case(case)
    for name in NAMES:
        assert getattr(G, name).shape[0] == S.entity_count(G), f'{case}/{name}'


@pytest.mark.parametrize('case', CASE_NAMES)
def test_the_rows_are_the_entities_in_the_order_the_graph_holds_them(case):
    G = build_case(case)
    assert list(G.matrices.signed().entity_of_row) == S.entity_keys(G), case


# ---------------------------------------------------------------------------
# The adjacency action
# ---------------------------------------------------------------------------


def test_matmul_applies_the_adjacency():
    G = _mixed()
    x = np.arange(4, dtype=float)
    assert np.allclose(G @ x, G.A @ x)


def test_rmatmul_applies_the_adjacency_from_the_left():
    G = _mixed()
    x = np.arange(4, dtype=float)
    assert np.allclose(x @ G, x @ G.A)


# ---------------------------------------------------------------------------
# The cache behind the namespace
# ---------------------------------------------------------------------------


def test_reading_the_same_matrix_twice_builds_it_once():
    G = _mixed()
    assert G.A is not None
    before = G.matrices.cache.rebuilds
    assert G.A is not None
    assert G.matrices.cache.rebuilds == before


def test_a_write_makes_the_next_read_report_it():
    G = _mixed()
    assert G.A.nnz == 2
    G.add_edges('C', 'D', edge_id='e_cd')
    assert G.A.nnz == 3
    G.remove_edges('e_ab')
    assert G.A.nnz == 2


def test_an_append_extends_the_incidence_instead_of_rebuilding_it():
    G = _mixed()
    assert G.S is not None
    before = G.matrices.cache.extends
    G.add_edges('C', 'D', edge_id='e_cd')
    assert G.S is not None
    assert G.matrices.cache.extends == before + 1


def test_a_copy_holds_a_cache_of_its_own():
    """A copy installs its own store, and a matrix cached against another names
    nothing in it."""
    G = _mixed()
    assert G.A is not None
    H = G.ops.copy()
    assert H.matrices.cache is not G.matrices.cache
    H.remove_edges('e_ab')
    assert H.A.nnz == 1
    assert G.A.nnz == 2
