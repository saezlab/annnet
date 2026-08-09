"""A borrowing read gives what the walk gives, on every graph.

`FR-009`. The fast path is an optimization and not a change of meaning, so the
values and their order must be identical to what the read gave before it — on a
flat graph and a multilayer one, with and without freed slots, with and without
an edge-entity, and with a column that carries nothing for some elements.

The reference here is the walk itself, called directly. Comparing the public read
against the public read would compare the fast path with itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet
from annnet.core._attrs import EDGE_AXIS, NODE_AXIS


def _walked_node_column(graph, name):
    """What the read gave before the fast path: a gather over the live slots."""
    store = graph._attr_store
    return store._vector(
        store.node_columns.get(name), store._rows_of(NODE_AXIS, store._built_node_rows)[1]
    )


def _walked_edge_column(graph, name):
    store = graph._attr_store
    return store._vector(
        store.edge_columns.get(name), store._rows_of(EDGE_AXIS, store._built_edge_rows)[1]
    )


def _same(left, right) -> bool:
    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape:
        return False
    if left.dtype.kind == 'f' and right.dtype.kind == 'f':
        return np.array_equal(left, right, equal_nan=True)
    return all(
        (a is None and b is None) or a == b or (a != a and b != b)
        for a, b in zip(left.tolist(), right.tolist(), strict=True)
    )


# ---------------------------------------------------------------------------
# The matrix of cases
# ---------------------------------------------------------------------------


def flat() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(12)])
    graph.add_edges(
        [
            {'source': f'v{i}', 'target': f'v{i + 1}', 'edge_id': f'e{i}', 'w2': float(i)}
            for i in range(8)
        ]
    )
    return graph


def flat_with_freed_slots() -> AnnNet:
    graph = flat()
    graph.remove_edge('e3')
    graph.remove_node('v9')
    return graph


def multilayer() -> AnnNet:
    graph = AnnNet(directed=True, aspects={'condition': ['healthy', 'treated']})
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(6)], layer=('healthy',))
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(6)], layer=('treated',))
    graph.add_edges(
        [{'source': 'v0', 'target': 'v1', 'edge_id': 'e0', 'w2': 1.0}],
        layer=('healthy',),
    )
    return graph


def with_an_edge_entity() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': name, 'score': 1.0} for name in ('A', 'B', 'C')])
    graph.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
    graph.add_edges('ee_ab', 'C', edge_id='e_meta')
    return graph


def with_a_sparse_column() -> AnnNet:
    """Half the nodes carry the attribute and half were added after it existed."""
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(6)])
    graph.add_nodes([{'node_id': f'w{i}'} for i in range(6)])
    graph.add_nodes([{'node_id': f'u{i}', 'score': float(i)} for i in range(6)])
    return graph


def with_an_empty_axis() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': 'only', 'score': 1.0}])
    return graph


CASES = {
    'flat': flat,
    'flat_with_freed_slots': flat_with_freed_slots,
    'multilayer': multilayer,
    'with_an_edge_entity': with_an_edge_entity,
    'with_a_sparse_column': with_a_sparse_column,
    'with_an_empty_axis': with_an_empty_axis,
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_node_column_matches_the_walk(case):
    graph = CASES[case]()
    borrowed = graph._attr_store.node_vector('score')
    assert _same(borrowed, _walked_node_column(graph, 'score')), case


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_node_column_is_aligned_with_the_node_ids(case):
    graph = CASES[case]()
    column = graph.N['score']
    assert len(column) == len(graph.N.ids), case


@pytest.mark.parametrize('case', ['flat', 'flat_with_freed_slots', 'multilayer'])
def test_the_edge_column_matches_the_walk(case):
    graph = CASES[case]()
    borrowed = graph._attr_store.edge_vector('w2')
    assert _same(borrowed, _walked_edge_column(graph, 'w2')), case


@pytest.mark.parametrize('case', sorted(CASES))
def test_an_unknown_attribute_is_still_unknown(case):
    graph = CASES[case]()
    assert graph._attr_store.node_vector('nothing_carries_this') is None


class TestWhichPathAnswered:
    """The three shapes that keep the gather, and the one that borrows.

    A parity test that never reached the gather would prove nothing, so this
    states which path each case takes. A borrowed column is a view whose base is
    the array the store holds; a gathered one is a fresh array.
    """

    @staticmethod
    def _borrowed(graph, name) -> bool:
        column = graph._attr_store.node_vector(name)
        return column is not None and column.base is graph._attr_store.node_columns[name]

    def test_a_flat_graph_borrows(self):
        assert self._borrowed(flat(), 'score') is True

    @pytest.mark.parametrize('case', ['flat_with_freed_slots', 'multilayer', 'with_an_edge_entity'])
    def test_the_three_shapes_that_keep_the_gather(self, case):
        graph = CASES[case]()
        assert self._borrowed(graph, 'score') is False
        assert _same(graph._attr_store.node_vector('score'), _walked_node_column(graph, 'score'))

    def test_a_freed_edge_slot_keeps_the_gather_on_the_edge_axis(self):
        graph = flat_with_freed_slots()
        column = graph._attr_store.edge_vector('w2')
        assert column.base is not graph._attr_store.edge_columns['w2']
        assert _same(column, _walked_edge_column(graph, 'w2'))


def test_a_read_after_every_shape_of_write_matches_the_walk():
    """The order of the values follows the element ids after any write."""
    graph = flat()
    steps = [
        lambda: graph.add_nodes([{'node_id': 'new0', 'score': 99.0}]),
        lambda: graph.add_nodes([{'node_id': 'new1'}]),
        lambda: graph.remove_node('v2'),
        lambda: graph.add_nodes([{'node_id': 'new2', 'score': 7.0}]),
        lambda: graph.attrs.set_node_attrs('v0', score=-1.0),
    ]
    for step in steps:
        step()
        assert _same(graph._attr_store.node_vector('score'), _walked_node_column(graph, 'score'))
        assert list(graph.N.ids) == graph._attr_store.node_ids()
