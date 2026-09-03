"""The questions a caller asks, and whether each one is a single call.

Every assertion here stands for a helper somebody wrote by hand: reading an
endpoint out of an edge view, keeping the rows of one layer, keeping the rows of
one slice, or building a node-layer coordinate in the graph's aspect order.

The endpoint case is the one worth naming. ``views.edges()`` used to render an
endpoint as ``"('A', ('ctrl',))"`` — the repr of an internal tuple, as a string.
Nothing downstream consumes that without ``ast.literal_eval``, so the column was
unusable for a join and said nothing about it.
"""

from __future__ import annotations

import warnings

import pytest

import annnet as an
from annnet import Endpoint, as_endpoint, as_endpoints

ASPECT = 'cond'
CONDITIONS = ('ctrl', 'stim')


@pytest.fixture
def flat():
    """A graph with no aspects, where an endpoint is a bare id."""
    G = an.Graph(directed=True)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges('B', 'C', edge_id='e2', weight=1.0)
    return G


@pytest.fixture
def layered():
    """Two conditions, an intra edge in each, a coupling edge, and a hyperedge."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = an.Graph(directed=True)
        G.layers.set_aspects([ASPECT], {ASPECT: list(CONDITIONS)})
        for condition in CONDITIONS:
            G.add_nodes([{'node_id': 'A'}, {'node_id': 'B'}], layer=(condition,))
        G.add_edges(('A', ('ctrl',)), ('B', ('ctrl',)), edge_id='intra_ctrl')
        G.add_edges(('A', ('stim',)), ('B', ('stim',)), edge_id='intra_stim')
        G.add_edges(('A', ('ctrl',)), ('A', ('stim',)), edge_id='coupling')
        G.add_edges([{'members': [('A', ('stim',)), ('B', ('stim',))], 'edge_id': 'hyper_stim'}])
        G.slices.add('prior')
        G.slices.add_edges('prior', ['intra_ctrl', 'hyper_stim'])
    return G


# ---------------------------------------------------------------------------
# A1 — an endpoint is a node id
# ---------------------------------------------------------------------------


class TestEndpoint:
    """One shape for an endpoint, whether or not the graph is layered."""

    def test_a_bare_id_becomes_an_endpoint_with_no_layer(self):
        assert as_endpoint('akt') == Endpoint('akt', None)

    def test_a_stored_pair_becomes_an_endpoint_with_its_layer(self):
        assert as_endpoint(('akt', ('stim',))) == Endpoint('akt', ('stim',))

    def test_an_endpoint_is_idempotent(self):
        one = Endpoint('akt', ('stim',))
        assert as_endpoint(one) is one

    def test_str_is_the_id_because_that_is_what_a_label_wants(self):
        assert str(as_endpoint(('akt', ('stim',)))) == 'akt'

    def test_key_is_what_the_store_spells(self):
        assert as_endpoint(('akt', ('stim',))).key == ('akt', ('stim',))
        assert as_endpoint('akt').key == 'akt'

    def test_positional_shape_is_the_stores(self):
        endpoint = as_endpoint(('akt', ('stim',)))
        assert endpoint[0] == 'akt'
        assert endpoint[1] == ('stim',)

    def test_a_side_becomes_a_frozenset_of_endpoints(self):
        side = frozenset({('A', ('ctrl',)), ('B', ('ctrl',))})
        assert as_endpoints(side) == frozenset({Endpoint('A', ('ctrl',)), Endpoint('B', ('ctrl',))})


class TestEdgeViewIdentity:
    """``source_id``, ``target_id`` and ``layer`` answer without a type check."""

    def test_binary_edge_reports_its_two_ids(self, layered):
        edge = layered.get_edge('intra_ctrl')
        assert edge.source_id == 'A'
        assert edge.target_id == 'B'

    def test_a_flat_edge_reports_the_same_way(self, flat):
        edge = flat.get_edge('e1')
        assert (edge.source_id, edge.target_id) == ('A', 'B')

    def test_layer_is_the_one_both_endpoints_sit_in(self, layered):
        assert layered.get_edge('intra_ctrl').layer == ('ctrl',)

    def test_layer_is_none_when_the_edge_crosses_two(self, layered):
        assert layered.get_edge('coupling').layer is None

    def test_layer_is_none_on_a_flat_graph(self, flat):
        assert flat.get_edge('e1').layer is None

    def test_a_many_member_side_reports_no_single_id(self, layered):
        edge = layered.get_edge('hyper_stim')
        assert edge.source_id is None


class TestEdgeFrameIdentity:
    """The columns a join reads."""

    def test_source_and_target_are_bare_node_ids(self, layered):
        frame = layered.views.edges()
        rows = {row['edge_id']: row for row in _rows(frame)}
        assert rows['intra_ctrl']['source'] == 'A'
        assert rows['intra_ctrl']['target'] == 'B'

    def test_the_endpoint_columns_join_against_the_node_table(self, layered):
        frame = layered.views.edges(include_hyper=False)
        node_ids = set(layered.nodes())
        for row in _rows(frame):
            assert row['source'] in node_ids
            assert row['target'] in node_ids

    def test_layer_columns_carry_the_coordinate(self, layered):
        rows = {row['edge_id']: row for row in _rows(layered.views.edges())}
        assert rows['intra_ctrl']['src_layer'] == 'ctrl'
        assert rows['intra_ctrl']['dst_layer'] == 'ctrl'

    def test_a_crossing_edge_reports_two_different_layers(self, layered):
        rows = {row['edge_id']: row for row in _rows(layered.views.edges())}
        assert rows['coupling']['src_layer'] == 'ctrl'
        assert rows['coupling']['dst_layer'] == 'stim'

    def test_a_flat_graph_has_null_layer_columns(self, flat):
        for row in _rows(flat.views.edges()):
            assert row['src_layer'] is None
            assert row['dst_layer'] is None

    def test_hyperedge_members_are_node_ids(self, layered):
        rows = {row['edge_id']: row for row in _rows(layered.views.hyperedges())}
        assert sorted(rows['hyper_stim']['members']) == ['A', 'B']


# ---------------------------------------------------------------------------
# A2 — the filters, and the one that is easy to confuse
# ---------------------------------------------------------------------------


class TestViewFilters:
    """``layer=``, ``in_slice=``, ``include_hyper=``, and ``slice=`` beside them."""

    def test_layer_keeps_the_edges_of_one_layer(self, layered):
        found = set(_ids(layered.views.edges(layer=('ctrl',))))
        assert found == {'intra_ctrl'}

    def test_layer_keeps_a_hyperedge_of_that_layer(self, layered):
        found = set(_ids(layered.views.edges(layer=('stim',))))
        assert found == {'intra_stim', 'hyper_stim'}

    def test_layer_agrees_with_layer_edge_set(self, layered):
        for condition in CONDITIONS:
            assert set(_ids(layered.views.edges(layer=(condition,)))) == set(
                layered.layers.layer_edge_set((condition,))
            )

    def test_include_hyper_false_leaves_only_binary_rows(self, layered):
        frame = layered.views.edges(include_hyper=False)
        assert set(_column(frame, 'kind')) == {'binary'}
        assert 'hyper_stim' not in set(_ids(frame))

    def test_hyperedges_leaves_only_hyper_rows(self, layered):
        frame = layered.views.hyperedges()
        assert set(_ids(frame)) == {'hyper_stim'}

    def test_in_slice_keeps_only_that_slices_rows(self, layered):
        found = set(_ids(layered.views.edges(in_slice='prior')))
        assert found == {'intra_ctrl', 'hyper_stim'}

    def test_slice_joins_without_filtering(self, layered):
        """The distinction both adapters got wrong once: join, not filter."""
        joined = layered.views.edges(slice='prior')
        assert set(_ids(joined)) == {'intra_ctrl', 'intra_stim', 'coupling', 'hyper_stim'}

    def test_slice_and_in_slice_are_different_calls(self, layered):
        assert len(_ids(layered.views.edges(slice='prior'))) > len(
            _ids(layered.views.edges(in_slice='prior'))
        )

    def test_filters_compose(self, layered):
        found = set(_ids(layered.views.edges(in_slice='prior', include_hyper=False)))
        assert found == {'intra_ctrl'}

    def test_an_empty_result_still_has_the_identity_columns(self, layered):
        frame = layered.views.edges(layer=('ctrl',), include_hyper=False, in_slice='prior')
        empty = layered.views.edges(layer=('stim',), in_slice='prior', include_hyper=False)
        assert set(_ids(frame)) == {'intra_ctrl'}
        assert list(_ids(empty)) == []
        for name in ('edge_id', 'source', 'target', 'src_layer', 'dst_layer'):
            assert name in _columns(empty)


# ---------------------------------------------------------------------------
# A3 — the node-layer, named rather than spelled
# ---------------------------------------------------------------------------


class TestNodeLayerLookup:
    """``G.at`` builds the coordinate; ``G.exists`` asks without raising."""

    def test_at_returns_the_key_every_layered_call_takes(self, layered):
        assert layered.at('A', cond='stim') == ('A', ('stim',))

    def test_the_key_is_usable_where_a_key_is_wanted(self, layered):
        key = layered.at('A', cond='stim')
        layered.layers.set_node_attrs(key[0], key[1], observed=1.0)
        assert layered.layers.node_attrs('A', ('stim',))['observed'] == 1.0

    def test_at_raises_for_a_node_layer_that_is_not_there(self, layered):
        with pytest.raises(KeyError, match='not on layer'):
            layered.at('Z', cond='stim')

    def test_at_names_the_non_raising_question_in_the_message(self, layered):
        with pytest.raises(KeyError, match='G.exists'):
            layered.at('Z', cond='stim')

    def test_exists_answers_true_and_false(self, layered):
        assert layered.exists('A', cond='stim') is True
        assert layered.exists('Z', cond='stim') is False

    def test_an_unknown_aspect_raises_rather_than_answering_false(self, layered):
        with pytest.raises(KeyError, match='unknown aspect'):
            layered.exists('A', mechanism='stim')

    def test_an_unnamed_aspect_raises(self, layered):
        with pytest.raises(KeyError, match='needs a value'):
            layered.exists('A')

    def test_a_flat_graph_needs_no_aspect(self, flat):
        assert flat.exists('A') is True
        assert flat.at('A') == ('A', ('_',))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)


def _columns(frame):
    from annnet._support.dataframe_backend import dataframe_columns

    return list(dataframe_columns(frame))


def _column(frame, name):
    return [row[name] for row in _rows(frame)]


def _ids(frame):
    return _column(frame, 'edge_id')
