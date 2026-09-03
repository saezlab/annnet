"""A window over layers, named on the aspects.

The comprehension this replaces held three facts about the graph at the call
site — which position an aspect is in, what its values are, and what order they
come in::

    window = [aa for aa in G.layers._all_layers if aa[0] in TIMES[:3]]

All three are the graph's, and all three go wrong silently the first time an
aspect is added. ``where`` takes them back.
"""

from __future__ import annotations

import warnings

import pytest

import annnet as an
from annnet import Aspect

TIMES = ('0h', '1h', '12h', '24h')
MECHANISMS = ('mapk', 'pi3k')


@pytest.fixture
def G():
    """Two aspects, one ordered; nodes everywhere; one edge inside and one across."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.Graph(directed=True)
        graph.layers.set_aspects(
            ['time', 'mechanism'],
            {'time': Aspect(TIMES, ordered=True), 'mechanism': list(MECHANISMS)},
        )
        for time in TIMES:
            for mechanism in MECHANISMS:
                graph.add_nodes(
                    [{'node_id': 'A'}, {'node_id': 'B'}, {'node_id': 'C'}],
                    layer=(time, mechanism),
                )
        graph.add_edges(('A', ('0h', 'mapk')), ('B', ('0h', 'mapk')), edge_id='inside')
        graph.add_edges(('B', ('12h', 'mapk')), ('C', ('24h', 'mapk')), edge_id='across_time')
        graph.add_edges(('A', ('0h', 'mapk')), ('A', ('0h', 'pi3k')), edge_id='across_mechanism')
    return graph


def _coords(selection):
    return sorted(selection.layers)


class TestWindow:
    """Which layers a predicate names."""

    def test_no_predicate_selects_every_layer(self, G):
        assert len(G.layers.where()) == len(TIMES) * len(MECHANISMS)

    def test_a_bare_value_is_equality(self, G):
        found = G.layers.where(time='12h')
        assert _coords(found) == [('12h', 'mapk'), ('12h', 'pi3k')]

    def test_predicates_are_combined_with_and(self, G):
        found = G.layers.where(time='12h', mechanism='mapk')
        assert _coords(found) == [('12h', 'mapk')]

    def test_lte_reads_the_declared_order(self, G):
        found = {layer[0] for layer in G.layers.where(time__lte='12h')}
        assert found == {'0h', '1h', '12h'}

    def test_lt_excludes_the_bound(self, G):
        found = {layer[0] for layer in G.layers.where(time__lt='12h')}
        assert found == {'0h', '1h'}

    def test_gte_and_gt_are_the_mirror(self, G):
        assert {layer[0] for layer in G.layers.where(time__gte='12h')} == {'12h', '24h'}
        assert {layer[0] for layer in G.layers.where(time__gt='12h')} == {'24h'}

    def test_ne_drops_one_value(self, G):
        assert {layer[1] for layer in G.layers.where(mechanism__ne='mapk')} == {'pi3k'}

    def test_in_and_not_in_take_a_collection(self, G):
        assert {layer[0] for layer in G.layers.where(time__in=['0h', '24h'])} == {'0h', '24h'}
        assert {layer[1] for layer in G.layers.where(mechanism__not_in=['mapk'])} == {'pi3k'}

    def test_a_window_is_iterable_and_sized(self, G):
        found = G.layers.where(time__lte='1h')
        assert len(found) == 4
        assert ('0h', 'mapk') in found
        assert list(found) == [tuple(layer) for layer in found]

    def test_the_order_of_the_window_is_the_graphs(self, G):
        found = [layer[0] for layer in G.layers.where(mechanism='mapk')]
        assert found == list(TIMES)


class TestRefusals:
    """A malformed question raises rather than answering something else."""

    def test_an_unknown_aspect_raises(self, G):
        with pytest.raises(KeyError, match='unknown aspect'):
            G.layers.where(condition='x')

    def test_an_unknown_operator_raises(self, G):
        with pytest.raises(ValueError, match='unknown operator'):
            G.layers.where(time__around='12h')

    def test_a_comparison_on_a_categorical_aspect_raises(self, G):
        """The order would be the declaration order pretending to be a meaning."""
        with pytest.raises(ValueError, match='categorical'):
            G.layers.where(mechanism__lte='mapk')

    def test_equality_on_a_categorical_aspect_is_fine(self, G):
        assert len(G.layers.where(mechanism='mapk')) == len(TIMES)

    def test_a_value_the_aspect_does_not_hold_raises(self, G):
        with pytest.raises(KeyError, match='not a value of this aspect'):
            G.layers.where(time__lte='99h')

    def test_a_flat_graph_has_nothing_to_select_on(self):
        flat = an.Graph()
        flat.add_nodes(['A'])
        with pytest.raises(ValueError, match='no aspects'):
            flat.layers.where()


class TestWhatSitsOnTheWindow:
    """The four questions a window is asked, each in one pass."""

    def test_nodes_are_the_ids_on_those_layers(self, G):
        assert G.layers.where(time='0h').nodes == {'A', 'B', 'C'}

    def test_node_layers_keep_the_coordinate(self, G):
        found = G.layers.where(time='0h', mechanism='mapk').node_layers
        assert found == {
            ('A', ('0h', 'mapk')),
            ('B', ('0h', 'mapk')),
            ('C', ('0h', 'mapk')),
        }

    def test_one_node_on_two_layers_is_two_node_layers_and_one_node(self, G):
        window = G.layers.where(time__lte='1h', mechanism='mapk')
        assert len(window.nodes) == 3
        assert len(window.node_layers) == 6

    def test_edges_are_closed(self, G):
        """An edge with an endpoint outside the window is not in it."""
        window = G.layers.where(time='0h', mechanism='mapk')
        assert window.edges == {'inside'}

    def test_crossing_is_what_closed_leaves_out(self, G):
        window = G.layers.where(time='0h', mechanism='mapk')
        assert window.crossing == {'across_mechanism'}

    def test_an_edge_wholly_outside_is_in_neither(self, G):
        window = G.layers.where(time='0h', mechanism='mapk')
        assert 'across_time' not in window.edges | window.crossing

    def test_widening_the_window_turns_crossing_into_inside(self, G):
        narrow = G.layers.where(time='0h', mechanism='mapk')
        wide = G.layers.where(time='0h')
        assert 'across_mechanism' in narrow.crossing
        assert 'across_mechanism' in wide.edges
        assert wide.crossing == set()

    def test_boundary_is_where_the_window_is_cut(self, G):
        window = G.layers.where(time='0h', mechanism='mapk')
        assert window.boundary == {'A'}

    def test_boundary_is_empty_when_nothing_crosses(self, G):
        assert G.layers.where().boundary == set()

    def test_boundary_is_a_subset_of_nodes(self, G):
        window = G.layers.where(time__lte='12h', mechanism='mapk')
        assert window.boundary <= window.nodes

    def test_inside_and_crossing_do_not_overlap(self, G):
        window = G.layers.where(time__lte='12h')
        assert window.edges & window.crossing == set()
