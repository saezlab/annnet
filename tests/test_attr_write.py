"""A single attribute write costs one cell, not a whole table.

The record core keeps attributes in a dataframe, so one write rewrites a frame.
The slot store keeps one typed array per attribute, indexed by slot, so one write
lands in one cell. The table is derived and materializes only when a reader asks
for it.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import pytest

from annnet.core import _attrs as A, _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


def filled(n_nodes: int):
    store = ST.CoreState(directed=True)
    attrs = A.AttributeStore(store)
    for i in range(n_nodes):
        store.add_entity(key(f'v{i}'))
    attrs.set_node_column('score', np.arange(n_nodes, dtype=np.float64))
    return store, attrs


def test_a_single_write_changes_one_cell():
    store, attrs = filled(8)
    attrs.set_node(key('v3'), 'score', 99.0)
    column = attrs.node_column('score')
    assert column[store.entity_slot(key('v3'))] == 99.0
    assert column[store.entity_slot(key('v4'))] == 4.0


def test_a_single_write_does_not_rebuild_the_table():
    store, attrs = filled(8)
    attrs.obs()
    builds = attrs.table_builds
    attrs.set_node(key('v3'), 'score', 1.0)
    assert attrs.table_builds == builds, 'a write must not build a table'


def test_the_table_rebuilds_once_after_a_write():
    store, attrs = filled(8)
    attrs.obs()
    builds = attrs.table_builds
    attrs.set_node(key('v3'), 'score', 1.0)
    attrs.obs()
    attrs.obs()
    assert attrs.table_builds == builds + 1, 'one build serves every read until the next write'


def test_a_warm_read_returns_the_same_table():
    _store, attrs = filled(8)
    assert attrs.obs() is attrs.obs()


def test_a_new_column_starts_null_for_every_node():
    store, attrs = filled(4)
    attrs.set_node(key('v1'), 'label', 'x')
    column = attrs.node_column('label')
    assert column[store.entity_slot(key('v1'))] == 'x'
    assert column[store.entity_slot(key('v2'))] is None


def test_a_freed_slot_holds_a_null_after_reuse():
    store, attrs = filled(4)
    store.remove_entity(key('v2'))
    store.add_entity(key('w'))
    assert attrs.node_column('score')[store.entity_slot(key('w'))] != 2.0


def test_the_node_table_holds_one_row_per_live_node():
    store, attrs = filled(5)
    store.remove_entity(key('v2'))
    rows = attrs.obs_rows()
    assert len(rows) == 4
    assert {row['node_id'] for row in rows} == {'v0', 'v1', 'v3', 'v4'}


def test_an_edge_column_works_the_same_way():
    store = ST.CoreState(directed=True)
    attrs = A.AttributeStore(store)
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    store.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    attrs.set_edge('e0', 'confidence', 0.75)
    assert attrs.edge_column('confidence')[store.edge_slot('e0')] == 0.75
    assert attrs.var_rows() == [{'edge_id': 'e0', 'confidence': 0.75}]


@pytest.mark.slow
def test_the_cost_of_one_write_does_not_grow_with_the_node_count():
    def median_write(n_nodes: int, samples: int = 7) -> float:
        times = []
        for _ in range(samples):
            store, attrs = filled(n_nodes)
            target = key(f'v{n_nodes // 2}')
            gc.collect()
            gc.disable()
            start = time.perf_counter_ns()
            attrs.set_node(target, 'score', 1.0)
            elapsed = time.perf_counter_ns() - start
            gc.enable()
            times.append(elapsed / 1e9)
        return sorted(times)[len(times) // 2]

    small = median_write(200)
    large = median_write(3_200)
    assert large <= max(small * 4, 5e-5), (
        f'one write cost {small:.3e}s at 200 nodes and {large:.3e}s at 3200 nodes'
    )
