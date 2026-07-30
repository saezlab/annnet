"""Reading a matrix after every append stays proportional to the appends.

The record core rebuilds the whole matrix whenever a read follows a write, so a
loop of N appends with a read each is quadratic. The slot store keeps the member
lists that the matrix is made of, and a cached matrix survives an append, so the
same loop stays linear.
"""

from __future__ import annotations

import gc
import time

import pytest

from annnet.core import _matrices as M, _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


def _append_then_read(n_edges: int, *, read_each: bool) -> float:
    store = ST.CoreState(directed=True)
    for i in range(n_edges + 1):
        store.add_entity(key(f'v{i}'))
    cache = M.MatrixCache(store)
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
        if read_each:
            cache.incidence()
    elapsed = time.perf_counter_ns() - start
    gc.enable()
    return elapsed / 1e9


def test_a_cached_matrix_survives_an_append():
    store = ST.CoreState(directed=True)
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    store.add_entity(key('C'))
    cache = M.MatrixCache(store)
    store.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    first = cache.incidence()
    assert first.matrix.shape[1] == 1
    store.add_edge(
        'e1',
        [(key('B'), 1.0, ST.SOURCE), (key('C'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    second = cache.incidence()
    assert second.matrix.shape[1] == 2
    assert cache.extends >= 1, 'the append must extend the cache, not rebuild it'
    assert cache.rebuilds == 1, 'only the first read builds the matrix'


def test_a_delete_rebuilds_the_cache_rather_than_extending_it():
    store = ST.CoreState(directed=True)
    for n in 'ABC':
        store.add_entity(key(n))
    cache = M.MatrixCache(store)
    for i, (u, v) in enumerate((('A', 'B'), ('B', 'C'))):
        store.add_edge(
            f'e{i}',
            [(key(u), 1.0, ST.SOURCE), (key(v), -1.0, ST.TARGET)],
            kind=ST.BINARY,
            directed=True,
            weight=1.0,
        )
    cache.incidence()
    store.remove_edge('e0')
    view = cache.incidence()
    assert view.matrix.shape[1] == 1
    assert cache.rebuilds == 2, 'a delete is not an append, so the cache rebuilds'


def test_a_warm_read_returns_the_same_object_until_the_next_write():
    store = ST.CoreState(directed=True)
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    store.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    cache = M.MatrixCache(store)
    assert cache.incidence() is cache.incidence()


@pytest.mark.slow
def test_appending_with_a_read_each_stays_proportional_to_the_appends():
    """The ratio between the two loops must not grow with N."""
    small_reads = _append_then_read(200, read_each=True)
    large_reads = _append_then_read(800, read_each=True)
    # Four times the appends may cost more than four times the time, but not the
    # sixteen times that a rebuild-per-read would give. The bound is loose so a
    # busy machine does not fail the run.
    assert large_reads <= small_reads * 8, (
        f'200 appends with a read each cost {small_reads:.4f}s and 800 cost {large_reads:.4f}s'
    )
