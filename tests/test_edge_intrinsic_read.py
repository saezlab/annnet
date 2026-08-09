"""An intrinsic edge field reads as one pass over the edge arrays.

`FR-016`. Three fields of an edge are not attributes — its weight, its direction
and its kind — and they read like a column, because a filter over them is as
common as a filter over an attribute. The read used to ask the graph for a record
per edge, and a record carries two frozensets of endpoints, so reading the weight
column of 40 000 edges cost 90 milliseconds on this host where the attribute
column beside it cost 19 microseconds.

The store holds ``edge_weight``, ``edge_directed`` and ``edge_kind`` as arrays
already.
"""

from __future__ import annotations

import time

import pytest

from annnet import AnnNet

INTRINSIC = ('weight', 'directed', 'kind')


def _graph(edges: int = 64) -> AnnNet:
    nodes = max(2, edges // 4)
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(nodes)])
    graph.add_edges(
        [
            {
                'source': f'v{i % nodes}',
                'target': f'v{(i + 1) % nodes}',
                'edge_id': f'e{i}',
                'weight': float(i),
                'w2': float(i),
            }
            for i in range(edges)
        ]
    )
    return graph


class TestTheReadDoesNotBuildARecordPerEdge:
    @pytest.mark.parametrize('name', INTRINSIC)
    def test_get_edge_is_never_called(self, name, monkeypatch):
        graph = _graph()

        def refuse(*_args, **_kwargs):
            raise AssertionError(f'reading {name!r} built a record per edge')

        monkeypatch.setattr(type(graph), 'get_edge', refuse)
        column = graph.E[name]
        assert len(column) == len(graph.E.ids)

    def test_the_weight_column_borrows_the_stored_array(self):
        graph = _graph()
        column = graph.E['weight']
        assert column.base is graph._store.edge_weight

    @pytest.mark.parametrize('name', INTRINSIC)
    def test_a_subsequence_still_reads_element_by_element(self, name):
        """A subsequence names its own ids, so the slice does not address it."""
        graph = _graph()
        column = graph.E[0:4][name]
        assert len(column) == 4


class TestTheCostDoesNotGrowWithTheGraph:
    """`SC-006`: within reach of an attribute column of the same graph."""

    @staticmethod
    def _best(fn, calls: int) -> float:
        out = None
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(calls):
                fn()
            out = min(out or 1e9, (time.perf_counter() - start) / calls)
        return out

    @pytest.mark.parametrize('name', INTRINSIC)
    def test_it_is_within_reach_of_an_attribute_column(self, name):
        graph = _graph(5_000)
        attribute = self._best(lambda: graph.E['w2'], calls=50)
        intrinsic = self._best(lambda: graph.E[name], calls=50)

        # The gap was about five thousand times. A hundred leaves room for the
        # one vectorized pass that resolving a direction or a kind costs.
        assert intrinsic < attribute * 100

    def test_the_weight_column_matches_the_attribute_column_in_cost(self):
        graph = _graph(5_000)
        attribute = self._best(lambda: graph.E['w2'], calls=50)
        weight = self._best(lambda: graph.E['weight'], calls=50)
        assert weight < attribute * 2
