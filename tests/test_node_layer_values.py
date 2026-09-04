"""Node-layer values, wherever they live.

A value keyed by ``(node, layer)`` lives in the contextual store when a person
typed it and in an attached array when it arrived as a table. The reader is the
same either way, and the point of nearly every test here is that the two paths
agree — because the moment they do not, an analysis silently changes answer
depending on how its inputs were loaded.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annnet as an
from annnet import Aspect
from annnet.core._values import (
    MISSING,
    ContextualValues,
)

CONDITIONS = ('c0', 'c1', 'c2')
NODES = ('n0', 'n1', 'n2', 'n3')


@pytest.fixture
def G():
    """Three conditions, four nodes, and an attached array of known values."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.Graph(directed=True)
        graph.layers.set_aspects(['cond'], {'cond': Aspect(CONDITIONS, ordered=True)})
        graph.layers.place(NODES, [(c,) for c in CONDITIONS])
    return graph


@pytest.fixture
def attached(G):
    """``expr[row, column] == row * 10 + column``, so every cell is identifiable."""
    values = np.arange(len(CONDITIONS) * len(NODES), dtype='float32')
    values = (values // len(NODES)) * 10 + (values % len(NODES))
    array = values.reshape(len(CONDITIONS), len(NODES))
    G.layers.attach_values({'expr': array}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES))
    return G


def _slow(graph, nodes, layers, name, missing=np.nan):
    """The per-cell path, spelled out, for parity checks."""
    resolver = graph.layers.values()
    return np.array(
        [[resolver.get(n, layer, name, missing) for n in nodes] for layer in layers],
        dtype=float,
    ).reshape(len(layers), len(nodes))


# ---------------------------------------------------------------------------
# The backings
# ---------------------------------------------------------------------------


class TestMatrixValues:
    def test_it_reads_a_cell_off_the_array(self, attached):
        assert attached.layers.values().get('n2', ('c1',), 'expr') == 12.0

    def test_an_unknown_node_or_layer_gives_the_default(self, attached):
        resolver = attached.layers.values()
        assert resolver.get('ghost', ('c1',), 'expr', -1) == -1
        assert resolver.get('n0', ('c9',), 'expr', -1) == -1

    def test_an_unknown_name_gives_the_default(self, attached):
        assert attached.layers.values().get('n0', ('c0',), 'absent', -1) == -1

    def test_the_array_is_not_copied(self, G):
        array = np.zeros((len(CONDITIONS), len(NODES)))
        G.layers.attach_values({'v': array}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES))
        array[1, 1] = 7.0
        assert G.layers.values().get('n1', ('c1',), 'v') == 7.0

    def test_an_empty_arrays_dict_raises_rather_than_stopping_iteration(self, G):
        with pytest.raises(ValueError, match='at least one array'):
            G.layers.attach_values({}, layers=[('c0',)], nodes=['n0'])

    def test_a_one_dimensional_array_raises(self, G):
        with pytest.raises(ValueError, match='two-dimensional'):
            G.layers.attach_values({'v': np.zeros(3)}, layers=[('c0',)], nodes=['n0'])

    def test_an_array_too_small_for_the_maps_raises(self, G):
        with pytest.raises(ValueError, match='the maps address'):
            G.layers.attach_values(
                {'v': np.zeros((1, 1))},
                layers=[(c,) for c in CONDITIONS],
                nodes=list(NODES),
            )

    def test_a_mask_of_the_wrong_shape_raises(self, G):
        with pytest.raises(ValueError, match='same shape'):
            G.layers.attach_values(
                {'v': np.zeros((3, 4))},
                layers=[(c,) for c in CONDITIONS],
                nodes=list(NODES),
                mask=np.zeros((2, 2), dtype=bool),
            )

    def test_a_gated_cell_holds_nothing_whatever_the_array_carries(self, G):
        array = np.ones((len(CONDITIONS), len(NODES)))
        mask = np.ones_like(array, dtype=bool)
        mask[0, 0] = False
        G.layers.attach_values(
            {'v': array}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES), mask=mask
        )
        resolver = G.layers.values()
        assert resolver.get('n0', ('c0',), 'v', -1) == -1
        assert resolver.get('n1', ('c0',), 'v', -1) == 1.0

    def test_two_nodes_may_share_one_column(self, G):
        """What a join of many nodes onto one measured entity needs."""
        array = np.array([[5.0], [6.0], [7.0]])
        G.layers.attach_values(
            {'v': array},
            layers=[(c,) for c in CONDITIONS],
            nodes=['n0', 'n1'],
            columns={'n0': 0, 'n1': 0},
        )
        resolver = G.layers.values()
        assert resolver.get('n0', ('c1',), 'v') == 6.0
        assert resolver.get('n1', ('c1',), 'v') == 6.0


class TestResolverOrder:
    def test_a_later_backing_wins(self, G):
        G.layers.set_node_attrs('n0', ('c0',), v=1.0)
        G.layers.attach_values(
            {'v': np.full((3, 4), 9.0)}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES)
        )
        assert G.layers.values().get('n0', ('c0',), 'v') == 9.0

    def test_the_dict_store_answers_where_no_array_does(self, attached):
        attached.layers.set_node_attrs('n0', ('c0',), typed=3.0)
        assert attached.layers.values().get('n0', ('c0',), 'typed') == 3.0

    def test_detaching_gives_the_earlier_answer_back(self, G):
        G.layers.set_node_attrs('n0', ('c0',), v=1.0)
        backing = G.layers.attach_values(
            {'v': np.full((3, 4), 9.0)}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES)
        )
        G.layers.detach_values(backing)
        assert G.layers.values().get('n0', ('c0',), 'v') == 1.0

    def test_missing_tells_a_stored_none_from_an_absent_one(self):
        backing = ContextualValues({('n0', ('c0',)): {'v': None}})
        assert backing.get('n0', ('c0',), 'v', MISSING) is None
        assert backing.get('n0', ('c1',), 'v', MISSING) is MISSING

    def test_names_layers_and_nodes_are_the_union(self, attached):
        attached.layers.set_node_attrs('n0', ('c0',), typed=1.0)
        resolver = attached.layers.values()
        assert {'expr', 'typed'} <= resolver.names()
        assert set(NODES) <= resolver.nodes()
        assert {(c,) for c in CONDITIONS} <= resolver.layers()


# ---------------------------------------------------------------------------
# block(): the whole point, and the parity that makes it safe
# ---------------------------------------------------------------------------


class TestBlock:
    """The rectangle read, and its equality with the per-cell path."""

    def test_the_dict_store_offers_no_rectangle(self):
        assert ContextualValues({}).block(['n0'], [('c0',)], 'v') is None

    def test_a_resolver_over_the_dict_store_alone_returns_none(self, G):
        G.layers.set_node_attrs('n0', ('c0',), v=1.0)
        assert G.layers.values().block(list(NODES), [('c0',)], 'v') is None

    def test_a_name_no_backing_holds_returns_none(self, attached):
        assert attached.layers.values().block(list(NODES), [('c0',)], 'absent') is None

    def test_the_rectangle_matches_the_per_cell_path(self, attached):
        layers = [(c,) for c in CONDITIONS]
        block = attached.layers.values().block(list(NODES), layers, 'expr')
        assert np.array_equal(block, _slow(attached, NODES, layers, 'expr'))

    def test_it_matches_under_a_gate(self, G):
        array = np.arange(12, dtype=float).reshape(3, 4)
        mask = np.ones_like(array, dtype=bool)
        mask[1, 2] = False
        G.layers.attach_values(
            {'v': array}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES), mask=mask
        )
        layers = [(c,) for c in CONDITIONS]
        block = G.layers.values().block(list(NODES), layers, 'v')
        assert np.array_equal(
            np.isnan(block), np.isnan(_slow(G, NODES, layers, 'v')), equal_nan=True
        )
        assert np.isnan(block[1, 2])

    def test_it_matches_when_a_node_is_unknown(self, attached):
        layers = [(c,) for c in CONDITIONS]
        nodes = ['n0', 'ghost', 'n2']
        block = attached.layers.values().block(nodes, layers, 'expr')
        slow = _slow(attached, nodes, layers, 'expr')
        assert np.array_equal(np.isnan(block), np.isnan(slow))
        assert np.array_equal(block[~np.isnan(block)], slow[~np.isnan(slow)])

    def test_two_arrays_layer_with_the_later_one_winning(self, G):
        layers = [(c,) for c in CONDITIONS]
        G.layers.attach_values({'v': np.zeros((3, 4))}, layers=layers, nodes=list(NODES))
        G.layers.attach_values({'v': np.full((3, 1), 9.0)}, layers=layers, nodes=['n1'])
        block = G.layers.values().block(list(NODES), layers, 'v')
        assert np.array_equal(block, _slow(G, NODES, layers, 'v'))
        assert (block[:, 1] == 9.0).all()
        assert (block[:, 0] == 0.0).all()

    def test_a_name_in_both_stores_falls_back_to_the_slow_path(self, attached):
        """The dict store has no rectangle, so the whole read goes cell by cell.

        Conservative rather than clever: here the array would have won every cell
        anyway, so the fast answer would have been right. But a dict entry on a
        cell no array covers is a cell only the slow path can reach, and telling
        the two cases apart costs more than taking the slow path does.
        """
        attached.layers.set_node_attrs('n0', ('c0',), expr=99.0)
        layers = [(c,) for c in CONDITIONS]
        assert attached.layers.values().block(list(NODES), layers, 'expr') is None
        found = attached.layers.matrix('expr', nodes=list(NODES), layers=layers)
        assert np.array_equal(found.values, _slow(attached, NODES, layers, 'expr'))

    def test_the_dict_store_answers_a_cell_no_array_covers(self, attached):
        """Which is why falling back is necessary rather than merely safe."""
        attached.layers.set_node_attrs('n0', ('c0',), expr=99.0)
        outside = attached.layers.matrix('expr', nodes=['n0'], layers=[('c0',)])
        # The array covers this cell too and was attached later, so it wins.
        assert outside.values[0, 0] == 0.0
        attached.layers.set_node_attrs('n0', ('c0',), only_typed=5.0)
        typed = attached.layers.matrix('only_typed', nodes=['n0'], layers=[('c0',)])
        assert typed.values[0, 0] == 5.0


# ---------------------------------------------------------------------------
# matrix(): what a method is handed
# ---------------------------------------------------------------------------


class TestValueMatrix:
    def test_it_carries_the_values_and_both_labels(self, attached):
        block = attached.layers.matrix('expr')
        assert block.name == 'expr'
        assert block.nodes == list(NODES)
        assert block.layers == [(c,) for c in CONDITIONS]
        assert block.shape == (len(CONDITIONS), len(NODES))

    def test_the_rows_are_layers_and_the_columns_are_nodes(self, attached):
        block = attached.layers.matrix('expr')
        assert block.values[1, 2] == 12.0

    def test_it_reads_as_an_array(self, attached):
        assert np.asarray(attached.layers.matrix('expr')).shape == (3, 4)

    def test_nodes_and_layers_select_and_order(self, attached):
        block = attached.layers.matrix('expr', nodes=['n2', 'n0'], layers=[('c2',)])
        assert block.values.tolist() == [[22.0, 20.0]]

    def test_a_cell_nobody_answers_for_holds_missing(self, attached):
        block = attached.layers.matrix('expr', nodes=['ghost'], missing=-1.0)
        assert (block.values == -1.0).all()

    def test_the_two_paths_give_the_same_matrix(self, G):
        """The acceptance line: an array-backed read and a dict-backed one agree."""
        layers = [(c,) for c in CONDITIONS]
        for node_id in NODES:
            for row, layer in enumerate(layers):
                G.layers.set_node_attrs(node_id, layer, v=float(row * 10 + NODES.index(node_id)))
        typed = G.layers.matrix('v', nodes=list(NODES), layers=layers)

        other = an.Graph(directed=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            other.layers.set_aspects(['cond'], {'cond': Aspect(CONDITIONS, ordered=True)})
            other.layers.place(NODES, layers)
        array = np.array([[row * 10 + col for col in range(4)] for row in range(3)], dtype=float)
        other.layers.attach_values({'v': array}, layers=layers, nodes=list(NODES))
        attached_matrix = other.layers.matrix('v', nodes=list(NODES), layers=layers)

        assert np.array_equal(typed.values, attached_matrix.values)


# ---------------------------------------------------------------------------
# node_frame
# ---------------------------------------------------------------------------


class TestNodeFrame:
    def test_wide_is_one_row_per_layer(self, attached):
        rows = _rows(attached.layers.node_frame(attrs=['expr']))
        assert [row['layer_id'] for row in rows] == list(CONDITIONS)

    def test_one_attribute_names_the_columns_for_the_nodes(self, attached):
        frame = attached.layers.node_frame(attrs=['expr'])
        assert set(_columns(frame)) == {'layer', 'layer_id', *NODES}

    def test_one_node_names_the_columns_for_the_attributes(self, attached):
        frame = attached.layers.node_frame(nodes=['n0'], attrs=['expr'])
        assert set(_columns(frame)) == {'layer', 'layer_id', 'expr'}

    def test_the_values_land_on_the_right_cells(self, attached):
        rows = {r['layer_id']: r for r in _rows(attached.layers.node_frame(attrs=['expr']))}
        assert rows['c1']['n2'] == 12.0

    def test_long_is_one_row_per_cell(self, attached):
        rows = _rows(attached.layers.node_frame(attrs=['expr'], format='long'))
        assert len(rows) == len(CONDITIONS) * len(NODES)
        assert set(rows[0]) == {'node_id', 'layer', 'layer_id', 'attr', 'value'}

    def test_pairs_cost_the_pairs_asked_for(self, attached):
        frame = attached.layers.node_frame(pairs={'first': ('n0', 'expr')})
        assert set(_columns(frame)) == {'layer', 'layer_id', 'first'}

    def test_an_unknown_format_raises(self, attached):
        with pytest.raises(ValueError, match="format must be 'wide' or 'long'"):
            attached.layers.node_frame(format='tall')

    def test_it_agrees_with_reading_cell_by_cell(self, attached):
        rows = {r['layer_id']: r for r in _rows(attached.layers.node_frame(attrs=['expr']))}
        resolver = attached.layers.values()
        for condition in CONDITIONS:
            for node_id in NODES:
                assert rows[condition][node_id] == resolver.get(node_id, (condition,), 'expr')

    def test_a_graph_with_no_values_gives_an_empty_frame(self, G):
        frame = G.layers.node_frame()
        assert _rows(frame) == []


# ---------------------------------------------------------------------------
# place, and set_node_attrs_bulk
# ---------------------------------------------------------------------------


class TestPlace:
    def test_it_places_the_whole_rectangle(self, G):
        other = an.Graph(directed=True)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            other.layers.set_aspects(['cond'], {'cond': list(CONDITIONS)})
        assert other.layers.place(NODES, [(c,) for c in CONDITIONS]) == 12

    def test_every_pair_is_present_afterwards(self, G):
        for node_id in NODES:
            for condition in CONDITIONS:
                assert G.exists(node_id, cond=condition)

    def test_placing_twice_creates_nothing_new(self, G):
        assert G.layers.place(NODES, [(c,) for c in CONDITIONS]) == 0

    def test_a_mask_places_only_what_it_holds(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = an.Graph(directed=True)
            graph.layers.set_aspects(['cond'], {'cond': list(CONDITIONS)})
        mask = np.zeros((len(CONDITIONS), len(NODES)), dtype=bool)
        mask[0, 0] = True
        mask[2, 3] = True
        assert graph.layers.place(NODES, [(c,) for c in CONDITIONS], mask=mask) == 2
        assert graph.exists('n0', cond='c0')
        assert not graph.exists('n1', cond='c0')

    def test_an_empty_rectangle_places_nothing(self, G):
        assert G.layers.place([], [('c0',)]) == 0
        assert G.layers.place(['n0'], []) == 0

    def test_an_undeclared_layer_raises(self, G):
        with pytest.raises((KeyError, ValueError)):
            G.layers.place(['n0'], [('c9',)])

    def test_it_agrees_with_adding_nodes_one_layer_at_a_time(self):
        """The bulk path and the general one leave the same graph."""
        made = []
        for bulk in (True, False):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                graph = an.Graph(directed=True)
                graph.layers.set_aspects(['cond'], {'cond': list(CONDITIONS)})
                if bulk:
                    graph.layers.place(NODES, [(c,) for c in CONDITIONS])
                else:
                    for condition in CONDITIONS:
                        graph.add_nodes([{'node_id': n} for n in NODES], layer=(condition,))
            made.append(graph)
        left, right = made
        assert sorted(left.nodes()) == sorted(right.nodes())
        assert left.nv_supra == right.nv_supra
        for node_id in NODES:
            for condition in CONDITIONS:
                assert left.exists(node_id, cond=condition) == right.exists(node_id, cond=condition)


class TestSetNodeAttrsBulk:
    def test_explicit_pairs(self, G):
        assert G.layers.set_node_attrs_bulk({('n0', ('c0',)): {'v': 1.0}}) == 1
        assert G.layers.node_attrs('n0', ('c0',))['v'] == 1.0

    def test_bare_ids_with_a_layer(self, G):
        assert (
            G.layers.set_node_attrs_bulk({'n0': {'v': 2.0}, 'n1': {'v': 3.0}}, layer=('c1',)) == 2
        )
        assert G.layers.node_attrs('n1', ('c1',))['v'] == 3.0

    def test_scalars_with_a_key(self, G):
        assert G.layers.set_node_attrs_bulk({'n0': 4.0}, layer=('c2',), key='v') == 1
        assert G.layers.node_attrs('n0', ('c2',))['v'] == 4.0

    def test_a_bare_id_without_a_layer_raises(self, G):
        with pytest.raises(ValueError, match='needs layer='):
            G.layers.set_node_attrs_bulk({'n0': {'v': 1.0}})

    def test_a_scalar_without_a_key_raises(self, G):
        with pytest.raises(ValueError, match='needs '):
            G.layers.set_node_attrs_bulk({'n0': 1.0}, layer=('c0',))


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)


def _columns(frame):
    from annnet._support.dataframe_backend import dataframe_columns

    return list(dataframe_columns(frame))
