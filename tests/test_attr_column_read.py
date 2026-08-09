"""Reading a whole attribute column costs what slicing an array costs.

The store holds one typed array per attribute, indexed by slot. On a graph whose
live slots are contiguous, that array — cut to the live count — *is* the answer,
so the read has nothing to build.

Two reads, and the second is the expensive one. A read that follows another read
gathers the live slots. A read that follows any structural write first rebuilds
the map from element id to slot, by walking every element in Python, because that
map is cached against the clock of the store and every write invalidates it. At
100 000 nodes the first is 48 microseconds on this host and the second is 9 700.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from annnet import AnnNet
from annnet.core._attrs import NODE_AXIS


def _graph(nodes: int = 64, *, edges: int = 0) -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(nodes)])
    if edges:
        graph.add_edges(
            [
                {
                    'source': f'v{i % nodes}',
                    'target': f'v{(i + 1) % nodes}',
                    'edge_id': f'e{i}',
                    'w2': float(i),
                }
                for i in range(edges)
            ]
        )
    return graph


class TestTheReadBorrows:
    """`FR-007`: the read does not copy the array the store holds."""

    def test_a_node_column_is_a_view_of_the_stored_array(self):
        graph = _graph()
        held = graph._attr_store.node_columns['score']
        column = graph.N['score']
        assert column.base is held
        assert column.size == graph._store.entity_count

    def test_an_edge_column_is_a_view_of_the_stored_array(self):
        graph = _graph(edges=32)
        held = graph._attr_store.edge_columns['w2']
        column = graph.E['w2']
        assert column.base is held
        assert column.size == graph._store.edge_count

    def test_two_reads_share_one_buffer(self):
        graph = _graph()
        first, second = graph.N['score'], graph.N['score']
        assert first.base is second.base

    def test_a_graph_with_a_freed_slot_falls_back_and_still_answers(self):
        graph = _graph()
        graph.remove_node('v3')
        column = graph.N['score']
        assert column.size == graph._store.entity_count
        assert 3.0 not in set(column.tolist())

    def test_a_column_shorter_than_the_live_count_never_happens(self):
        """`D1`: a column grows when a slot is allocated, so the slice applies."""
        graph = _graph()
        graph.add_nodes([{'node_id': f'w{i}'} for i in range(40)])
        held = graph._attr_store.node_columns['score']
        assert held.size >= graph._store.entity_capacity
        column = graph.N['score']
        assert column.base is held
        assert column.size == graph._store.entity_count


class TestTheReadAfterAWriteDoesNotWalk:
    """`FR-008`: a structural write does not make the next read walk the elements."""

    def test_the_row_map_is_not_rebuilt(self, monkeypatch):
        graph = _graph()
        store = graph._attr_store
        walked = []

        original = store._built_node_rows
        monkeypatch.setattr(store, '_built_node_rows', lambda: (walked.append(1), original())[1])

        graph.N['score']
        walked.clear()
        graph.add_nodes([{'node_id': 'w0', 'score': 1.0}])
        graph.N['score']
        assert walked == []

    def test_the_same_holds_on_the_edge_axis(self, monkeypatch):
        graph = _graph(edges=32)
        store = graph._attr_store
        walked = []

        original = store._built_edge_rows
        monkeypatch.setattr(store, '_built_edge_rows', lambda: (walked.append(1), original())[1])

        graph.E['w2']
        walked.clear()
        graph.add_edges([{'source': 'v0', 'target': 'v1', 'edge_id': 'z0', 'w2': 1.0}])
        graph.E['w2']
        assert walked == []

    def test_the_read_after_a_write_gives_what_the_walk_gives(self):
        graph = _graph()
        graph.add_nodes([{'node_id': 'w0', 'score': 1.0}])
        borrowed = np.asarray(graph.N['score'])
        walked = graph._attr_store._vector(
            graph._attr_store.node_columns['score'],
            graph._attr_store._rows_of(NODE_AXIS, graph._attr_store._built_node_rows)[1],
        )
        assert np.array_equal(borrowed, walked, equal_nan=True)


class TestTheCostDoesNotGrowWithTheGraph:
    """`SC-002`: neither read grows with the size of the graph."""

    @staticmethod
    def _best(fn, calls: int) -> float:
        out = None
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(calls):
                fn()
            out = min(out or 1e9, (time.perf_counter() - start) / calls)
        return out

    @pytest.mark.parametrize('after_a_write', [False, True])
    def test_the_read_is_flat_across_two_scales(self, after_a_write):
        small, large = _graph(1_000), _graph(50_000)
        fresh = iter(range(1_000_000))

        def read(graph):
            if after_a_write:
                graph.add_nodes([{'node_id': f'z{next(fresh)}', 'score': 0.0}])
            return graph.N['score']

        cheap = self._best(lambda: read(small), calls=20)
        dear = self._best(lambda: read(large), calls=20)

        # Fifty times the nodes. A walk would cost fifty times as much; a slice
        # costs the same. Five times leaves room for the noise of a shared host.
        assert dear < cheap * 5
