"""The store keeps a live count of edge-entities, and keeps it right.

The node axis filters edge-entities out, so a fast path over a node column may
apply only when the store holds none. Counting them is a pass over the kind
array — fast in numpy, and neither free nor constant. So the store maintains
one integer where an entity is allocated, freed, or has its kind changed, and
the internal validator checks it against the kinds it claims to count.
"""

from __future__ import annotations

import time

from annnet import AnnNet
from annnet.core import _store as ST
from annnet.core._validate import validate_internal_consistency


def _true_count(store) -> int:
    return sum(
        1 for slot, _key in store.live_entities() if int(store.entity_kind[slot]) == ST.EDGE_ENTITY
    )


def _graph_with_an_edge_entity() -> AnnNet:
    """A, B and C, an edge-entity over A and B, and an edge from it to C."""
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C'])
    graph.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
    graph.add_edges('ee_ab', 'C', edge_id='e_meta')
    return graph


class TestTheCount:
    def test_a_plain_graph_holds_none(self):
        graph = AnnNet(directed=True)
        graph.add_edges([{'source': 'v0', 'target': 'v1'}])
        assert graph._store.edge_entity_count == 0

    def test_an_edge_entity_is_counted(self):
        store = _graph_with_an_edge_entity()._store
        assert store.edge_entity_count == _true_count(store)
        assert store.edge_entity_count == 1

    def test_removing_the_edge_entity_uncounts_it(self):
        graph = _graph_with_an_edge_entity()
        graph.remove_edge('e_meta')
        graph.remove_edge('ee_ab')
        store = graph._store
        assert store.edge_entity_count == _true_count(store)
        assert store.edge_entity_count == 0

    def test_a_copy_carries_the_count(self):
        graph = _graph_with_an_edge_entity()
        other = graph.ops.copy()
        assert other._store.edge_entity_count == _true_count(other._store)

    def test_a_selection_carries_the_count(self):
        graph = _graph_with_an_edge_entity()
        store = graph._store
        picked = store.select(
            [key for _slot, key in store.live_entities()], list(store.live_edge_ids())
        )
        assert picked.edge_entity_count == _true_count(picked)

    def test_changing_a_kind_in_place_moves_the_count(self):
        store = ST.CoreState()
        slot = store.add_entity(('a', ('_',)), ST.NODE)
        assert store.edge_entity_count == 0
        store.set_entity_kind(slot, ST.EDGE_ENTITY)
        assert store.edge_entity_count == 1
        store.set_entity_kind(slot, ST.NODE)
        assert store.edge_entity_count == 0

    def test_it_is_constant_time(self):
        large = AnnNet(directed=True)
        large.add_edges([{'source': f'v{i}', 'target': f'v{i + 1}'} for i in range(20_000)])
        store = large._store

        def best(fn, calls=2000):
            out = None
            for _ in range(3):
                start = time.perf_counter()
                for _ in range(calls):
                    fn()
                out = min(out or 1e9, (time.perf_counter() - start) / calls)
            return out

        # One attribute read against the numpy pass it replaces.
        counted = best(lambda: store.edge_entity_count)
        scanned = best(lambda: int((store.entity_kind[: store.entity_capacity]).sum()), calls=200)
        assert counted < scanned


class TestTheInvariant:
    """The validator checks the count against the kinds it claims to count."""

    def test_a_consistent_store_passes(self):
        graph = _graph_with_an_edge_entity()
        assert validate_internal_consistency(graph, strict=False) == []

    def test_a_wrong_count_is_reported(self):
        graph = _graph_with_an_edge_entity()
        graph._store._edge_entity_count += 3
        problems = validate_internal_consistency(graph, strict=False)
        assert any('edge-entit' in problem for problem in problems)
