"""Reading a matrix after every append stays proportional to the appends.

The record core rebuilds the whole matrix whenever a read follows a write, so a
loop of N appends with a read each is quadratic. The slot store keeps the member
lists that the matrix is made of, and a cached matrix survives an append, so the
same loop stays linear.
"""

from __future__ import annotations

import gc
import time
from unittest import mock

import pytest

from annnet.core import _matrices as M, _store as ST
from annnet.core.graph import AnnNet

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


def _awkward_store():
    """A store holding every shape whose column is not one entry per member.

    A self-loop names one entity twice, so its column holds one row twice. A
    hyperedge holds a wide side. A boundary edge holds one side only.
    """
    store = ST.CoreState(directed=True)
    for name in 'ABCD':
        store.add_entity(key(name))
    edges = [
        ('binary', [(key('A'), 2.5, ST.SOURCE), (key('B'), -2.5, ST.TARGET)], ST.BINARY),
        ('loop', [(key('C'), 1.0, ST.SOURCE), (key('C'), -1.0, ST.TARGET)], ST.BINARY),
        (
            'hyper',
            [
                (key('A'), 1.0, ST.SOURCE),
                (key('B'), 1.0, ST.SOURCE),
                (key('D'), -2.0, ST.TARGET),
            ],
            ST.HYPER,
        ),
        ('boundary', [(key('D'), 1.0, ST.SOURCE)], ST.BINARY),
    ]
    return store, edges


@pytest.mark.parametrize('signed', [True, False])
def test_an_extended_matrix_says_what_a_rebuilt_one_says(signed):
    """The two build paths differ, so they are pinned against each other.

    A build places every member entry of every edge at once and lets the
    conversion sum the entries that share a cell. An append adds one column at a
    time and sums them itself. A self-loop is where the two summations have to
    agree, because it is the only shape that puts two entries in one cell.
    """
    store, edges = _awkward_store()
    cache = M.MatrixCache(store)
    for edge_id, members, kind in edges:
        store.add_edge(edge_id, members, kind=kind, directed=True, weight=1.0)
        cache.incidence(signed=signed)
    extended = cache.incidence(signed=signed)
    assert cache.extends >= 1, 'the appends must extend the cache, not rebuild it'

    rebuilt = M.MatrixCache(store).incidence(signed=signed)
    assert (extended.matrix.toarray() == rebuilt.matrix.toarray()).all()
    assert extended.edge_of_column == rebuilt.edge_of_column
    assert extended.column_of_edge == rebuilt.column_of_edge
    assert extended.entity_of_row == rebuilt.entity_of_row

    # And against the build that answers without a cache at all.
    uncached = M.incidence(store, signed=signed)
    assert (extended.matrix.toarray() == uncached.matrix.toarray()).all()


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


# ---------------------------------------------------------------------------
# The same, reached through the graph
# ---------------------------------------------------------------------------


def _chain(n_edges: int) -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices([f'v{i}' for i in range(n_edges + 2)])
    for i in range(n_edges):
        G.add_edges(f'v{i}', f'v{i + 1}', edge_id=f'e{i}')
    return G


def test_the_graph_extends_its_matrix_after_an_append_rather_than_rebuilding():
    """``G.X()`` is the cached matrix, which is what makes a read after a write cheap."""
    G = _chain(4)
    G.X()
    cache = G.matrices.cache
    rebuilds, extends = cache.rebuilds, cache.extends

    G.add_edges('v0', 'v4', edge_id='extra')
    assert G.X().shape[1] == 5
    assert cache.rebuilds == rebuilds, 'an append must not rebuild the matrix'
    assert cache.extends == extends + 1


def test_reading_the_matrix_does_not_name_the_positions_in_it():
    """The maps cost more than the matrix, and arithmetic does not read them.

    ``G.X()``, ``G.S`` and the rest answer with the matrix, so nothing on that
    path may build the map from an entity to its row or from an edge to its
    column. ``G.matrices.signed()`` is what asks for those.
    """
    G = _chain(4)
    with mock.patch.object(M, '_view', wraps=M._view) as named:
        for name in ('S', 'B', 'H', 'A', 'L'):
            assert getattr(G, name) is not None
        assert G.X() is not None
        assert named.call_count == 0
        assert G.matrices.signed().column_of_edge['e0'] == 0
        assert named.call_count == 1


def test_the_maps_of_a_view_follow_the_appends_that_extended_it():
    """A view asked for before an append names the columns the append added."""
    G = _chain(4)
    view = G.matrices.signed()
    G.add_edges('v0', 'v4', edge_id='extra')
    grown = G.matrices.signed()
    assert grown.column_of_edge['extra'] == 4
    assert grown.edge_of_column == ['e0', 'e1', 'e2', 'e3', 'extra']
    assert grown.matrix.shape[1] == 5
    assert view.column_of_edge is grown.column_of_edge, 'the map grows in place'


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
