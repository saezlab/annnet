"""A removal at the frontier leaves the cached matrix usable.

`FR-014`. A cached matrix survives a run of appends at the frontier, because the
store logs them and the buffer takes the new columns one at a time. It did not
survive a removal of any kind, so removing the edge that was appended last threw
away every column that removal did not touch.

A removal that frees the highest live edge slot is the mirror of an append at the
frontier: it takes the last column off the matrix and moves no row and no other
column. So the log records it as such, and the buffer drops its last column.

Anything else — a removal in the middle, a removal that also takes an entity —
falls back to a rebuild, which is what it did before.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet


def _graph(edges: int = 12) -> AnnNet:
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


def _counts(graph) -> tuple[int, int]:
    cache = graph.matrices.cache
    return cache.rebuilds, cache.extends


class TestTheCacheSurvivesAFrontierRemoval:
    def test_the_next_read_does_not_rebuild(self):
        graph = _graph()
        _ = graph.S
        before = _counts(graph)
        graph.remove_edge('e11')
        _ = graph.S
        rebuilds, extends = _counts(graph)
        assert rebuilds == before[0], 'a frontier removal must not force a rebuild'
        assert extends == before[1] + 1

    def test_the_matrix_loses_exactly_that_column(self):
        graph = _graph()
        before = graph.S.toarray()
        graph.remove_edge('e11')
        assert np.array_equal(graph.S.toarray(), before[:, :-1])

    def test_a_run_of_frontier_removals_survives(self):
        graph = _graph()
        before = graph.S.toarray()
        counts = _counts(graph)
        for edge_id in ('e11', 'e10', 'e9'):
            graph.remove_edge(edge_id)
            _ = graph.S
        assert _counts(graph)[0] == counts[0]
        assert np.array_equal(graph.S.toarray(), before[:, :-3])

    def test_an_append_after_a_frontier_removal_survives_too(self):
        graph = _graph()
        _ = graph.S
        counts = _counts(graph)
        graph.remove_edge('e11')
        graph.add_edges('v0', 'v2', edge_id='later')
        _ = graph.S
        assert _counts(graph)[0] == counts[0]
        assert graph.S.shape[1] == 12

    def test_the_edge_ids_of_the_view_follow(self):
        graph = _graph()
        view = graph.matrices.signed()
        assert 'e11' in view.column_of_edge
        graph.remove_edge('e11')
        view = graph.matrices.signed()
        assert 'e11' not in view.column_of_edge
        assert view.matrix.shape[1] == len(view.edge_of_column)


class TestWhatItCannotSurvive:
    @pytest.mark.parametrize('victim', ['e0', 'e5'])
    def test_a_removal_in_the_middle_rebuilds(self, victim):
        graph = _graph()
        _ = graph.S
        before = _counts(graph)
        graph.remove_edge(victim)
        _ = graph.S
        assert _counts(graph)[0] == before[0] + 1

    def test_a_removal_that_also_takes_an_entity_rebuilds(self):
        graph = _graph()
        _ = graph.S
        before = _counts(graph)
        graph.remove_node('v3')
        _ = graph.S
        assert _counts(graph)[0] > before[0]

    def test_a_frontier_removal_of_an_edge_entity_rebuilds(self):
        graph = AnnNet(directed=True)
        graph.add_nodes(['A', 'B', 'C'])
        graph.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
        graph.add_edges('ee_ab', 'C', edge_id='e_meta')
        _ = graph.S
        before = _counts(graph)
        graph.remove_edge('e_meta')
        _ = graph.S
        assert _counts(graph) != before


class TestTheAnswerIsTheSameEitherWay:
    @pytest.mark.parametrize('victim', ['e11', 'e5', 'e0'])
    def test_it_matches_a_full_rebuild(self, victim):
        graph = _graph()
        _ = graph.S
        graph.remove_edge(victim)
        kept = graph.S.toarray()
        graph.matrices.cache.drop()
        assert np.array_equal(kept, graph.S.toarray())
