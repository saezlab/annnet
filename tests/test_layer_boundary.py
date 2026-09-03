"""What a layer selection does at its own edge.

The layer algebra used to have one behaviour here and no word for it. Asked for
the union of layers ``a`` and ``b`` *with coupling edges*, it returned every
coupling edge **touching** either — including one running from ``b`` out to
``c``, which is not in the window at all. A selection that reaches outside the
window it names is a leak, and an unnamed leak is one nobody can ask for or
refuse.

``boundary="closed"`` is now the default and does not leak. ``boundary="open"``
is the old behaviour, and it has a name.

With the default ``include_inter=False`` / ``include_coupling=False`` the two
agree — an intra-layer edge never leaves its layer — so this only changes an
answer for a caller who asked for a crossing edge.
"""

from __future__ import annotations

import warnings

import pytest

import annnet as an
from annnet import BOUNDARIES, Aspect

VALUES = ('a', 'b', 'c')


@pytest.fixture
def G():
    """Three layers, one intra edge in each, and a coupling edge between each pair."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.Graph(directed=True)
        graph.layers.set_aspects(['t'], {'t': Aspect(VALUES, ordered=True)})
        for value in VALUES:
            graph.add_nodes([{'node_id': 'X'}, {'node_id': 'Y'}], layer=(value,))
        for value in VALUES:
            graph.add_edges(('X', (value,)), ('Y', (value,)), edge_id=f'intra_{value}')
        graph.add_edges(('X', ('a',)), ('X', ('b',)), edge_id='couple_ab')
        graph.add_edges(('X', ('b',)), ('X', ('c',)), edge_id='couple_bc')
    return graph


AB = [('a',), ('b',)]


class TestTheDefaultDoesNotLeak:
    """The behaviour the word was invented for."""

    def test_closed_drops_the_edge_that_leaves_the_window(self, G):
        found = G.layers.layer_union(AB, include_coupling=True)['edges']
        assert 'couple_bc' not in found

    def test_closed_keeps_the_edge_that_stays_inside(self, G):
        found = G.layers.layer_union(AB, include_coupling=True)['edges']
        assert found == {'intra_a', 'intra_b', 'couple_ab'}

    def test_open_is_the_old_behaviour_and_leaks(self, G):
        found = G.layers.layer_union(AB, include_coupling=True, boundary='open')['edges']
        assert found == {'intra_a', 'intra_b', 'couple_ab', 'couple_bc'}

    def test_closed_is_the_default(self, G):
        assert (
            G.layers.layer_union(AB, include_coupling=True)['edges']
            == G.layers.layer_union(AB, include_coupling=True, boundary='closed')['edges']
        )

    def test_the_boundaries_agree_when_no_crossing_edge_was_asked_for(self, G):
        """An intra-layer edge never leaves its layer, so there is nothing to leak."""
        for boundary in BOUNDARIES:
            assert G.layers.layer_union(AB, boundary=boundary)['edges'] == {
                'intra_a',
                'intra_b',
            }

    def test_a_window_of_every_layer_leaks_nowhere(self, G):
        every = [(value,) for value in VALUES]
        assert (
            G.layers.layer_union(every, include_coupling=True)['edges']
            == G.layers.layer_union(every, include_coupling=True, boundary='open')['edges']
        )

    def test_nodes_are_the_same_either_way(self, G):
        """Boundary is a question about edges. A node is in a layer or it is not."""
        closed = G.layers.layer_union(AB, include_coupling=True)['nodes']
        opened = G.layers.layer_union(AB, include_coupling=True, boundary='open')['nodes']
        assert closed == opened


class TestEveryOperationTakesIt:
    """The parameter is on the whole algebra, not on one entry point."""

    def test_intersection(self, G):
        found = G.layers.layer_intersection(AB, include_coupling=True)['edges']
        assert found == {'couple_ab'}

    def test_intersection_of_intra_edges_is_empty(self, G):
        """An intra edge belongs to one layer, so it is in no intersection of two."""
        assert G.layers.layer_intersection(AB)['edges'] == set()

    def test_difference(self, G):
        found = G.layers.layer_difference(('a',), ('b',), include_coupling=True)['edges']
        assert found == {'intra_a'}

    def test_create_slice_from_layer(self, G):
        G.layers.create_slice_from_layer('only_a', ('a',), include_coupling=True)
        assert G.slices.edges('only_a') == {'intra_a'}

    def test_create_slice_from_layer_open_keeps_the_coupling(self, G):
        G.layers.create_slice_from_layer(
            'touching_a', ('a',), include_coupling=True, boundary='open'
        )
        assert G.slices.edges('touching_a') == {'intra_a', 'couple_ab'}

    def test_subgraph_from_layer_tuple(self, G):
        sub = G.layers.subgraph_from_layer_tuple(('a',), include_coupling=True)
        assert set(sub.edges()) == {'intra_a'}

    def test_subgraph_from_layer_union(self, G):
        sub = G.layers.subgraph_from_layer_union(AB, include_coupling=True)
        assert set(sub.edges()) == {'intra_a', 'intra_b', 'couple_ab'}

    def test_subgraph_from_layer_union_open(self, G):
        sub = G.layers.subgraph_from_layer_union(AB, include_coupling=True, boundary='open')
        assert 'couple_bc' in set(sub.edges())

    @pytest.mark.parametrize(
        'call',
        [
            lambda g: g.layers.layer_union(AB, boundary='half'),
            lambda g: g.layers.layer_intersection(AB, boundary='half'),
            lambda g: g.layers.layer_difference(('a',), ('b',), boundary='half'),
            lambda g: g.layers.subgraph_from_layer_tuple(('a',), boundary='half'),
        ],
    )
    def test_an_unknown_boundary_raises(self, G, call):
        with pytest.raises(ValueError, match='boundary must be one of'):
            call(G)


class TestAgreementWithTheWindow:
    """`layers.where` and the algebra answer the same question the same way."""

    def test_closed_union_agrees_with_a_windows_edges(self, G):
        window = G.layers.where(t__lte='b')
        union = G.layers.layer_union(AB, include_coupling=True)['edges']
        assert union == window.edges

    def test_open_union_is_the_closed_one_plus_what_crosses(self, G):
        window = G.layers.where(t__lte='b')
        opened = G.layers.layer_union(AB, include_coupling=True, boundary='open')['edges']
        assert opened == window.edges | window.crossing


class TestOnePass:
    """`layer_edge_set` stays the primitive, and the algebra stops re-walking."""

    def test_layer_edge_set_still_means_touching(self, G):
        found = G.layers.layer_edge_set(('a',), include_coupling=True)
        assert found == {'intra_a', 'couple_ab'}

    def test_it_takes_no_boundary(self, G):
        """Touching is what one layer answers; closure is a question about a selection."""
        with pytest.raises(TypeError):
            G.layers.layer_edge_set(('a',), boundary='closed')

    def test_the_union_is_built_from_one_walk_not_one_per_layer(self, G, monkeypatch):
        from annnet.core import _structure

        walks = {'n': 0}
        real = _structure.iter_edges

        def counted(graph):
            walks['n'] += 1
            return real(graph)

        monkeypatch.setattr(_structure, 'iter_edges', counted)
        G.layers.layer_union([(v,) for v in VALUES], include_coupling=True)
        assert walks['n'] == 1
