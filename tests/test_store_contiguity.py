"""The store answers, in constant time, whether its slots are contiguous.

A fast path over a column indexes the array with ``0 .. count-1`` instead of
with the slot of every live element. It may do so only when the live slots
*are* ``0 .. count-1``, and it has to find that out without walking anything —
a predicate that costs what the walk costs would defeat what it guards.

The predicate is therefore built from the freelist and the slot count alone.
``live_entity_slots`` is not part of it: it is a generator over the key list and
costs 2.2 milliseconds at 100 000 entities on the reference host.
"""

from __future__ import annotations

import time

import pytest

from annnet import AnnNet
from annnet.core import _store as ST


def _flat_graph(count: int = 8) -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(count)])
    return graph


class TestEntityContiguity:
    """``entity_slots_contiguous`` is true exactly when the entity slots are."""

    def test_a_fresh_graph_is_contiguous(self):
        store = _flat_graph()._store
        assert store.entity_slots_contiguous is True

    def test_an_empty_store_is_contiguous(self):
        assert ST.CoreState().entity_slots_contiguous is True

    def test_a_freed_slot_breaks_it(self):
        graph = _flat_graph()
        graph.remove_node('v3')
        assert graph._store.entity_slots_contiguous is False

    def test_reusing_the_freed_slot_restores_it(self):
        graph = _flat_graph()
        graph.remove_node('v3')
        graph.add_nodes([{'node_id': 'v99'}])
        assert graph._store.entity_slots_contiguous is True

    def test_it_matches_the_slots_the_store_actually_holds(self):
        graph = _flat_graph(12)
        for victim in ('v2', 'v7'):
            graph.remove_node(victim)
        store = graph._store
        live = store.live_entity_slots()
        expected = live.size == store.entity_capacity and (
            not live.size or int(live[-1]) == live.size - 1
        )
        assert store.entity_slots_contiguous is expected


class TestEdgeContiguity:
    """``edge_slots_contiguous`` is the same predicate on the edge axis."""

    def test_a_fresh_graph_is_contiguous(self):
        graph = _flat_graph()
        graph.add_edges([{'source': 'v0', 'target': 'v1'}])
        assert graph._store.edge_slots_contiguous is True

    def test_a_freed_edge_slot_breaks_it(self):
        graph = _flat_graph()
        graph.add_edges(
            [
                {'source': 'v0', 'target': 'v1', 'edge_id': 'e0'},
                {'source': 'v1', 'target': 'v2', 'edge_id': 'e1'},
            ]
        )
        graph.remove_edge('e0')
        assert graph._store.edge_slots_contiguous is False

    def test_it_matches_the_slots_the_store_actually_holds(self):
        graph = _flat_graph()
        graph.add_edges(
            [{'source': 'v0', 'target': f'v{i}', 'edge_id': f'e{i}'} for i in (1, 2, 3)]
        )
        graph.remove_edge('e2')
        store = graph._store
        live = store.live_edge_slots()
        expected = live.size == len(store._edge_id) and (
            not live.size or int(live[-1]) == live.size - 1
        )
        assert store.edge_slots_contiguous is expected


class TestItIsConstantTime:
    """The cost of the answer does not grow with the size of the graph."""

    @staticmethod
    def _seconds(fn, calls: int = 2000) -> float:
        best = None
        for _ in range(3):
            start = time.perf_counter()
            for _ in range(calls):
                fn()
            elapsed = (time.perf_counter() - start) / calls
            best = elapsed if best is None else min(best, elapsed)
        return best

    @pytest.mark.parametrize('axis', ['entity_slots_contiguous', 'edge_slots_contiguous'])
    def test_the_answer_does_not_grow_with_the_graph(self, axis):
        small = AnnNet(directed=True)
        small.add_edges([{'source': f'v{i}', 'target': f'v{i + 1}'} for i in range(50)])
        large = AnnNet(directed=True)
        large.add_edges([{'source': f'v{i}', 'target': f'v{i + 1}'} for i in range(20_000)])

        cheap = self._seconds(lambda: getattr(small._store, axis))
        dear = self._seconds(lambda: getattr(large._store, axis))

        # A walk over the large graph would be four hundred times the small one.
        # Three times leaves room for the noise of a shared host.
        assert dear < cheap * 3

    def test_it_never_reaches_the_walk(self, monkeypatch):
        graph = _flat_graph(64)
        store = graph._store

        def refuse():
            raise AssertionError('the predicate walked the entities')

        monkeypatch.setattr(type(store), 'live_entity_slots', lambda self: refuse())
        monkeypatch.setattr(type(store), 'live_edge_slots', lambda self: refuse())
        assert store.entity_slots_contiguous is True
        assert store.edge_slots_contiguous is True


class TestTheNodeAxisCarriesTwoMoreConditions:
    """The node axis is not the entity axis, so it asks two further questions.

    A multilayer graph holds one entity per layer a node lives in and the node
    table shows the bare id once. An edge-entity is an entity the node axis
    filters out. Neither is a node-axis row per entity slot, so neither may take
    the fast path.
    """

    def test_a_flat_graph_of_plain_nodes_qualifies(self):
        assert _flat_graph()._store.node_axis_contiguous is True

    def test_a_multilayer_graph_does_not(self):
        graph = AnnNet(directed=True, aspects={'condition': ['healthy', 'treated']})
        graph.add_nodes(['A', 'B'])
        assert graph._store.entity_slots_contiguous is True
        assert graph._store.node_axis_contiguous is False

    def test_an_edge_entity_among_the_entities_does_not(self):
        graph = AnnNet(directed=True)
        graph.add_nodes(['A', 'B', 'C'])
        graph.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
        graph.add_edges('ee_ab', 'C', edge_id='e_meta')
        assert graph._store.entity_slots_contiguous is True
        assert graph._store.node_axis_contiguous is False
