"""Behaviour the guide notebooks show, which no other test pins down.

The guide notebooks print these results instead of asserting them, so a
change in behaviour would only show as a different number on a docs page.
"""

from __future__ import annotations

import warnings

import numpy as np

from annnet import AnnNet


def _two_layers() -> AnnNet:
    """Nodes A and B on two layers of one aspect, one edge on the first."""
    G = AnnNet(directed=False)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.set_aspects(['t'], {'t': ['t1', 't2']})

        for node in ('A', 'B'):
            for label in ('t1', 't2'):
                G.add_nodes([node], layer={'t': label})

        G.add_edges(('A', ('t1',)), ('B', ('t1',)), edge_id='e_t1')

    return G


def test_updating_an_attribute_to_none_removes_it() -> None:
    G = AnnNet(directed=True)
    G.add_edges('A', 'B', edge_id='e1')
    G.attrs.update('nodes', {'A': {'score': 1.0}})

    G.attrs.update('nodes', {'A': {'score': None}})

    assert 'score' not in G.attrs.row('nodes', 'A')


def test_history_diff_counts_structure_not_attributes() -> None:
    G = AnnNet(directed=True)
    G.add_edges('A', 'B', edge_id='e1')
    before = G.history.snapshot('before')

    G.attrs.update('nodes', {'A': {'note': 'reviewed'}})

    assert G.history.diff(before).is_empty()

    G.remove_edges('e1')

    assert G.history.diff(before).edges_removed == {'e1'}


def test_setting_coefficients_keeps_source_and_target() -> None:
    G = AnnNet(directed=True)
    G.add_edges(['A', 'B'], ['C'], edge_id='reaction')

    G.set_edge_coeffs('reaction', {'A': -1.0, 'B': -2.0, 'C': 3.0})

    S = G.matrices.signed()
    column = S.matrix[:, [S.column_of_edge['reaction']]].toarray().ravel()
    coeffs = {
        entity[0]: float(value) for entity, value in zip(S.entity_of_row, column, strict=True)
    }

    assert coeffs == {'A': -1.0, 'B': -2.0, 'C': 3.0}
    assert G.E.at('reaction').source == frozenset({'A', 'B'})
    assert G.E.at('reaction').target == frozenset({'C'})


def test_self_loop_has_a_zero_incidence_column_and_its_weight_on_the_diagonal() -> None:
    G = AnnNet(directed=True)
    G.add_edges('A', 'B', edge_id='e1')
    G.add_edges('A', 'A', edge_id='loop', weight=2.0, directed=True)

    S = G.matrices.signed()
    A = G.matrices.adjacency()
    row = A.row_of_entity[('A', ('_',))]

    assert np.allclose(S.matrix[:, [S.column_of_edge['loop']]].toarray(), 0)
    assert A.matrix[row, row] == 2.0


def test_categorical_coupling_puts_its_weight_on_the_cross_layer_diagonal() -> None:
    G = _two_layers()
    G.slices.add('coupling')
    G.slices.active = 'coupling'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        count = G.layers.couple('t', kind='categorical', weight=0.2)

    A = G.layers.supra_adjacency().toarray()
    first = [G.layers.nl_to_row(node, ('t1',)) for node in ('A', 'B')]
    second = [G.layers.nl_to_row(node, ('t2',)) for node in ('A', 'B')]

    assert count == 2
    assert np.allclose(A[np.ix_(first, second)], 0.2 * np.eye(2))


def test_row_to_nl_inverts_nl_to_row() -> None:
    G = _two_layers()

    for node in ('A', 'B'):
        for label in ('t1', 't2'):
            row = G.layers.nl_to_row(node, (label,))

            assert G.layers.row_to_nl(row) == (node, (label,))


def test_diffusion_step_is_one_explicit_euler_step_of_the_laplacian() -> None:
    G = _two_layers()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.add_layer_coupling_pairs([(('t1',), ('t2',))])

    L = G.layers.supra_laplacian(kind='comb')
    x = np.zeros(L.shape[0])
    x[G.layers.nl_to_row('A', ('t1',))] = 1.0

    assert np.allclose(G.layers.diffusion_step(x, tau=0.1), x - 0.1 * (L @ x))
