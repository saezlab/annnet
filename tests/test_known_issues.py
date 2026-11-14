"""Regression tests for the defects once listed in dev/known-issues.md."""

from __future__ import annotations

import pytest

import annnet
from annnet import AnnNet


def _multilayer():
    anet = AnnNet(directed=True)
    anet.layers.set_aspects(['condition'], {'condition': ['control', 'treated']})
    anet.add_nodes(['A', 'B'], layer={'condition': 'control'})
    return anet


class TestBatchBareIdsOnLayers:
    def test_batch_puts_bare_ids_on_their_one_layer(self):
        anet = _multilayer()
        anet.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1'}])
        assert anet.E.at('e1').members == frozenset({('A', ('control',)), ('B', ('control',))})

    def test_batch_and_single_forms_agree(self):
        one, many = _multilayer(), _multilayer()
        one.add_edges('A', 'B', edge_id='e1')
        many.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1'}])
        assert one.E.at('e1').members == many.E.at('e1').members

    def test_batch_bare_id_on_two_layers_raises(self):
        anet = _multilayer()
        anet.add_nodes(['A'], layer={'condition': 'treated'})
        with pytest.raises(ValueError, match='Ambiguous'):
            anet.add_edges([{'source': 'A', 'target': 'B'}])

    def test_batch_edge_is_intra_layer(self):
        anet = _multilayer()
        anet.add_edges([('A', 'B')])
        (edge_id,) = anet.E
        assert {coord for _, coord in anet.E.at(edge_id).members} == {('control',)}

    def test_hyperedge_batch_counts_the_placeholder_as_a_layer(self):
        anet = AnnNet(directed=True)
        anet.layers.set_aspects(['condition'], {'condition': ['control', 'treated']})
        anet.add_nodes(['A', 'B'])
        anet.add_nodes(['A', 'B'], layer=('control',))
        for call in (
            lambda: anet.add_edges([{'members': ['A', 'B'], 'edge_id': 'h'}]),
            lambda: anet.add_edges([('A', 'B')]),
            lambda: anet.add_edges('A', 'B', edge_id='e'),
        ):
            with pytest.raises(ValueError, match='Ambiguous'):
                call()


def _flexible_graph():
    anet = AnnNet(directed=True)
    anet.add_nodes(['a', 'b'])
    anet.attrs.update('nodes', {'a': {'x': 5.0}, 'b': {'x': 1.0}})
    return anet


_NODE_POLICY = {'var': 'x', 'threshold': 0, 'scope': 'node', 'above': 't->s'}


class TestFlexibleDirection:
    def test_reversal_moves_source_and_neighbours(self):
        anet = _flexible_graph()
        anet.add_edges('a', 'b', edge_id='f', flexible=dict(_NODE_POLICY))
        assert anet.E.at('f').source == frozenset({'b'})
        assert anet.E.at('f').target == frozenset({'a'})
        assert anet.successors('b') == ['a']
        assert anet.successors('a') == []

    def test_attribute_update_reverses_back(self):
        anet = _flexible_graph()
        anet.add_edges('a', 'b', edge_id='f', flexible=dict(_NODE_POLICY))
        anet.attrs.update('nodes', {'a': {'x': 0.0}})
        assert anet.E.at('f').source == frozenset({'a'})
        assert anet.successors('a') == ['b']

    @pytest.mark.parametrize(
        ('value', 'source'), [(-1.0, 'b'), (3.0, 'a'), (-2.0, 'b'), (-2.0, 'b'), (4.0, 'a')]
    )
    def test_edge_scope_tracks_value(self, value, source):
        anet = AnnNet(directed=True)
        anet.add_nodes(['a', 'b'])
        anet.add_edges('a', 'b', edge_id='e', flexible={'var': 'w', 'threshold': 0}, w=1.0)
        for v in (-1.0, 3.0, value):
            anet.attrs.update('edges', {'e': {'w': v}})
        assert anet.E.at('e').source == frozenset({source})

    def test_coefficients_agree_with_roles(self):
        anet = _flexible_graph()
        anet.add_edges('a', 'b', edge_id='f', flexible=dict(_NODE_POLICY))
        column = anet.S.todense()
        rows = list(anet.N)
        assert column[rows.index('b'), 0] > 0 > column[rows.index('a'), 0]


class TestFlexibleInBatch:
    def test_batch_item_policy_is_attached_and_applied(self):
        anet = _flexible_graph()
        anet.add_edges([{'source': 'a', 'target': 'b', 'edge_id': 'g', 'flexible': _NODE_POLICY}])
        assert anet.edge_direction_policy['g']['var'] == 'x'
        assert anet.E.at('g').source == frozenset({'b'})

    def test_batch_item_malformed_policy_raises(self):
        anet = _flexible_graph()
        with pytest.raises(ValueError, match='flexible'):
            anet.add_edges([{'source': 'a', 'target': 'b', 'flexible': {'x': 1}}])

    def test_batch_keyword_still_refused(self):
        anet = _flexible_graph()
        with pytest.raises(TypeError):
            anet.add_edges([('a', 'b')], flexible=dict(_NODE_POLICY))


def _sliced_chain():
    anet = AnnNet(directed=True)
    anet.add_edges('A', 'B', edge_id='e1')
    anet.add_edges('B', 'C', edge_id='e2')
    anet.add_edges('C', 'D', edge_id='e3')
    anet.slices.add('s', edges=['e1'])
    return anet


class TestSliceLimitsBackend:
    def test_nx_backend(self):
        pytest.importorskip('networkx')
        anet = _sliced_chain()
        assert anet.nx.backend(slice='s').number_of_edges() == 1
        assert sorted(anet.nx.pagerank(anet, _nx_slice='s')) == ['A', 'B']
        assert anet.nx.backend().number_of_edges() == 3

    def test_nx_round_trip_of_a_slice(self):
        pytest.importorskip('networkx')
        anet = _sliced_chain()
        back = annnet.from_nx(*annnet.to_nx(anet, slice='s'))
        assert list(back.E) == ['e1']
        assert sorted(back.N) == ['A', 'B']

    def test_ig_backend(self):
        pytest.importorskip('igraph')
        anet = _sliced_chain()
        g = anet.ig.backend(slice='s')
        assert (g.vcount(), g.ecount()) == (2, 1)
