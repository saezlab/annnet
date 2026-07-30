"""A single delete touches only the deleted element.

The record core renumbers every later element on a delete, so the cost of one
delete grows with the size of the graph. The slot store frees the address and
touches nothing else, so the cost stays flat. These tests state both halves:
locality, and the growth curve that locality implies.
"""

from __future__ import annotations

import gc
import time

import pytest

from annnet.core import _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


def chain(n_edges: int) -> ST.CoreState:
    """A directed chain with one node more than it has edges."""
    store = ST.CoreState(directed=True)
    for i in range(n_edges + 1):
        store.add_entity(key(f'v{i}'))
    for i in range(n_edges):
        store.add_edge(
            f'e{i}',
            [(key(f'v{i}'), 1.0, ST.SOURCE), (key(f'v{i + 1}'), -1.0, ST.TARGET)],
            kind=ST.BINARY,
            directed=True,
            weight=1.0,
        )
    return store


# ---------------------------------------------------------------------------
# Locality
# ---------------------------------------------------------------------------


def test_deleting_an_edge_leaves_every_other_address_untouched():
    store = chain(50)
    before = {eid: store.edge_slot(eid) for eid in store.live_edge_ids()}
    store.remove_edge('e10')
    after = {eid: store.edge_slot(eid) for eid in store.live_edge_ids()}
    assert set(before) - set(after) == {'e10'}
    for eid, slot in after.items():
        assert before[eid] == slot


def test_deleting_an_edge_leaves_every_other_member_list_untouched():
    store = chain(50)
    before = {
        eid: tuple(store.members(store.edge_slot(eid)).entities) for eid in store.live_edge_ids()
    }
    store.remove_edge('e10')
    for eid in store.live_edge_ids():
        assert tuple(store.members(store.edge_slot(eid)).entities) == before[eid]


def test_deleting_a_node_leaves_every_other_address_untouched():
    store = chain(50)
    before = {k: slot for slot, k in store.live_entities()}
    store.remove_entity(key('v25'))
    for slot, entity_key in store.live_entities():
        assert before[entity_key] == slot


def test_deleting_a_node_reports_the_edges_it_leaves_dangling():
    """The store never silently keeps a member that names no entity."""
    store = chain(5)
    dangling = store.remove_entity(key('v2'))
    assert set(dangling) == {'e1', 'e2'}


# ---------------------------------------------------------------------------
# The cost does not grow with the graph
# ---------------------------------------------------------------------------


def _median_delete_seconds(n_edges: int, *, samples: int = 5) -> float:
    times = []
    for _ in range(samples):
        store = chain(n_edges)
        target = f'e{n_edges // 2}'
        gc.collect()
        gc.disable()
        start = time.perf_counter_ns()
        store.remove_edge(target)
        elapsed = time.perf_counter_ns() - start
        gc.enable()
        times.append(elapsed / 1e9)
    return sorted(times)[len(times) // 2]


@pytest.mark.slow
def test_the_cost_of_one_delete_does_not_grow_with_the_edge_count():
    small = _median_delete_seconds(200)
    large = _median_delete_seconds(3_200)
    # A sixteen-fold graph must not cost far more for one delete. The bound is
    # loose on purpose, because a timing test has to survive a busy machine.
    assert large <= max(small * 4, 5e-5), (
        f'one delete cost {small:.3e}s at 200 edges and {large:.3e}s at 3200 edges'
    )
