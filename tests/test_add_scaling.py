"""Adding elements one at a time stays proportional to how many are added.

A single insert must not depend on how much the graph already holds. The record
core fails this: one add costs 13 microseconds on a 400-edge graph and 144 on a
40 000-edge one. The slot store appends to its arrays, so the cost stays flat and
a loop of N adds costs N times one add.
"""

from __future__ import annotations

import gc
import time

import pytest

from annnet.core import _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


def _build_seconds(n_edges: int) -> float:
    """Time a loop of N single adds, with the nodes created outside the clock."""
    store = ST.CoreState(directed=True)
    for i in range(n_edges + 1):
        store.add_entity(key(f'v{i}'))
    gc.collect()
    gc.disable()
    start = time.perf_counter_ns()
    for i in range(n_edges):
        store.add_edge(
            f'e{i}',
            [(key(f'v{i}'), 1.0, ST.SOURCE), (key(f'v{i + 1}'), -1.0, ST.TARGET)],
            kind=ST.BINARY,
            directed=True,
            weight=1.0,
        )
    elapsed = time.perf_counter_ns() - start
    gc.enable()
    return elapsed / 1e9


def test_one_add_does_not_read_the_rest_of_the_graph():
    """The structural property behind the timing: an append touches its own edge."""
    store = ST.CoreState(directed=True)
    for i in range(200):
        store.add_entity(key(f'v{i}'))
    for i in range(100):
        store.add_edge(
            f'e{i}',
            [(key(f'v{i}'), 1.0, ST.SOURCE), (key(f'v{i + 1}'), -1.0, ST.TARGET)],
            kind=ST.BINARY,
            directed=True,
            weight=1.0,
        )
    before = {eid: store.edge_slot(eid) for eid in store.live_edge_ids()}
    starts = {eid: int(store.member_start[store.edge_slot(eid)]) for eid in before}

    store.add_edge(
        'late',
        [(key('v0'), 1.0, ST.SOURCE), (key('v199'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )

    for eid, slot in before.items():
        assert store.edge_slot(eid) == slot, 'an append moved another edge'
        assert int(store.member_start[store.edge_slot(eid)]) == starts[eid], (
            'an append moved the member list of another edge'
        )


def test_an_append_advances_the_clock_by_one():
    """A batch of appends must not cost a rebuild of anything derived."""
    store = ST.CoreState(directed=True)
    store.add_entity(key('a'))
    store.add_entity(key('b'))
    before = store.structure_version
    for i in range(10):
        store.add_edge(
            f'e{i}',
            [(key('a'), 1.0, ST.SOURCE), (key('b'), -1.0, ST.TARGET)],
            kind=ST.BINARY,
            directed=True,
            weight=1.0,
        )
    assert store.structure_version == before + 10
    assert len(store.append_log) == 10, 'every append is on the log, so a cache can extend'


@pytest.mark.slow
def test_a_loop_of_single_adds_runs_in_time_proportional_to_the_count():
    """Four times the adds must cost about four times the time, not sixteen."""
    small = _build_seconds(1_000)
    large = _build_seconds(4_000)
    ratio = large / small if small else float('inf')
    assert ratio <= 8, (
        f'1000 adds cost {small:.4f}s and 4000 cost {large:.4f}s, a ratio of {ratio:.1f}. '
        'A ratio near four is linear and a ratio near sixteen is quadratic.'
    )
