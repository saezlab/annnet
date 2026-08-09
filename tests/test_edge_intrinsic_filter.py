"""A filter over an intrinsic column gives what it gave before.

``select`` and ``find`` read a column and compare it element by element, so a
change to how the column is read reaches every filter. The values and their
order are what those filters are built on.
"""

from __future__ import annotations

import pytest

from annnet import AnnNet


def _graph() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(8)])
    graph.add_edges(
        [
            {'source': 'v0', 'target': 'v1', 'edge_id': 'e0', 'weight': 1.0, 'tag': 'a'},
            {'source': 'v1', 'target': 'v2', 'edge_id': 'e1', 'weight': 2.0, 'tag': 'b'},
            {'source': 'v2', 'target': 'v3', 'edge_id': 'e2', 'weight': 2.0, 'tag': 'b'},
        ]
    )
    graph.add_edges(
        [{'source': 'v3', 'target': 'v4', 'edge_id': 'e3', 'weight': 3.0, 'directed': False}]
    )
    return graph


class TestSelect:
    def test_a_filter_on_weight(self):
        assert list(_graph().E.select(weight=2.0).ids) == ['e1', 'e2']

    def test_a_filter_on_direction(self):
        graph = _graph()
        assert list(graph.E.select(directed=False).ids) == ['e3']
        assert list(graph.E.select(directed=True).ids) == ['e0', 'e1', 'e2']

    def test_a_filter_on_kind(self):
        graph = _graph()
        assert list(graph.E.select(kind='binary').ids) == ['e0', 'e1', 'e2', 'e3']

    def test_a_filter_over_an_intrinsic_and_an_attribute_together(self):
        assert list(_graph().E.select(weight=2.0, tag='b').ids) == ['e1', 'e2']

    def test_a_filter_that_matches_nothing(self):
        assert list(_graph().E.select(weight=99.0).ids) == []

    def test_a_filter_over_a_subsequence(self):
        graph = _graph()
        assert list(graph.E[0:3].select(weight=2.0).ids) == ['e1', 'e2']


class TestFind:
    def test_it_finds_the_one_match(self):
        assert _graph().E.find(weight=1.0) == 'e0'

    def test_more_than_one_match_is_an_error(self):
        with pytest.raises(ValueError, match='2 elements match'):
            _graph().E.find(weight=2.0)

    def test_no_match_is_an_error(self):
        with pytest.raises(KeyError):
            _graph().E.find(weight=99.0)


class TestAFilterAfterAWrite:
    def test_a_weight_write_moves_the_filter(self):
        graph = _graph()
        graph.E['weight'] = [2.0, 2.0, 2.0, 2.0]
        assert list(graph.E.select(weight=2.0).ids) == ['e0', 'e1', 'e2', 'e3']

    def test_a_removal_moves_the_filter(self):
        graph = _graph()
        graph.remove_edge('e1')
        assert list(graph.E.select(weight=2.0).ids) == ['e2']
