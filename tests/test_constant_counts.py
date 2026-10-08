"""Counts are maintained, never enumerated.

``len(G.N)`` and ``len(G.E)`` answer from counters the store keeps as elements
arrive, change kind, are rekeyed, copied or removed. Nothing here walks the
entity list, and the tests forbid the walk rather than time it.
"""

from __future__ import annotations

import time

import pytest

from annnet import AnnNet
from annnet.core import _store as st


@pytest.mark.parametrize('layered', [False, True])
def test_full_axis_len_does_not_enumerate(monkeypatch, layered):
    graph = AnnNet(aspects={'condition': ['a', 'b']} if layered else None)
    if layered:
        graph.layers.place(['x', 'y'], [('a',), ('b',)])
    else:
        graph.add_nodes(['x', 'y'])
    graph.add_edges(
        ('x', ('a',)) if layered else 'x', ('y', ('a',)) if layered else 'y', edge_id='e'
    )
    expected_supra = graph.nv_supra

    def forbidden(*args, **kwargs):
        raise AssertionError('a count enumerated the graph')

    monkeypatch.setattr(st.CoreState, 'live_entities', forbidden)
    monkeypatch.setattr(st.CoreState, 'live_edges', forbidden)
    monkeypatch.setattr(type(graph.N), '_all_ids', forbidden)
    monkeypatch.setattr(type(graph.E), '_all_ids', forbidden)
    assert len(graph.N) == 2
    assert len(graph.E) == 1
    assert graph.nv_supra == expected_supra
    assert graph.shape == (2, 1)
    assert graph.supra_shape == (expected_supra, 1)
    assert len(graph) == 2


def test_counts_follow_kind_changes_rekey_copy_and_reuse():
    store = st.CoreState(aspects=('condition',))
    store.add_entities([('x', ('a',)), ('x', ('b',)), ('y', ('a',))])
    assert store.node_count == 2
    assert store.node_layer_count == 3
    store.set_entity_kind(store.entity_slot(('x', ('a',))), st.EDGE_ENTITY)
    assert store.node_count == 2
    assert store.node_layer_count == 2
    store.remove_entity(('x', ('b',)))
    assert store.node_count == 1
    assert store.node_layer_count == 1
    store.add_entity(('z', ('b',)))
    assert store.node_count == 2
    store.rekey({('z', ('b',)): ('y', ('b',))})
    assert store.node_count == 1
    assert store.node_layer_count == 2
    copy = store.copy()
    copy.remove_entity(('y', ('b',)))
    assert copy.node_count == store.node_count == 1
    assert copy.node_layer_count == 1
    assert store.node_layer_count == 2
    # A freed slot is reused; the count still follows the live elements.
    store.add_entity(('w', ('a',)))
    assert store.node_count == 2
    assert store.node_layer_count == 3


def test_flat_promotion_preserves_counts():
    store = st.CoreState()
    store.add_entity(('x', ('_',)))
    assert store.node_count == 1
    store.aspects = ('condition',)
    store.add_entity(('x', ('a',)))
    assert store.node_count == 1
    assert store.node_layer_count == 2


def test_counts_through_the_graph_mutation_api():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.add_edges('a', 'b', edge_id='e1')
    G.add_edges('b', 'c', edge_id='e2', as_entity=True)
    assert (len(G.N), len(G.E)) == (3, 2)
    # An edge entity is not a node.
    G.add_edges('e2', 'a', edge_id='meta')
    assert (len(G.N), len(G.E)) == (3, 3)
    G.remove_edges('e1')
    assert (len(G.N), len(G.E)) == (3, 2)
    G.remove_nodes('c')
    assert len(G.N) == 2
    assert len(G.E) == 0, 'removing c drops e2 and the edge that named e2'
    G.add_nodes('c')
    assert len(G.N) == 3
    H = G.ops.copy()
    H.add_nodes('d')
    assert len(H.N) == 4 and len(G.N) == 3
    with pytest.warns(UserWarning, match='placeholder'):
        G.layers.set_aspects(['t'], {'t': ['t1', 't2']})
    # The three flat nodes moved to the placeholder coordinate; a gains two
    # more placements.
    G.layers.place(['a'], [('t1',), ('t2',)])
    assert len(G.N) == 3
    assert G.nv_supra == 5
    G.remove_nodes('a')
    assert len(G.N) == 2 and G.nv_supra == 2, 'a bare id removes every placement'


def test_count_of_a_large_graph_costs_what_a_small_one_costs():
    """The count must not scale with the graph. Measured coarsely, asserted loosely."""
    small = AnnNet()
    small.add_nodes([f'n{i}' for i in range(10)])
    big = AnnNet()
    big.add_nodes([f'n{i}' for i in range(200_000)])

    def clock(graph):
        best = float('inf')
        for _ in range(5):
            start = time.perf_counter_ns()
            for _ in range(200):
                len(graph.N)
            best = min(best, time.perf_counter_ns() - start)
        return best

    assert clock(big) < 20 * clock(small) + 2_000_000
