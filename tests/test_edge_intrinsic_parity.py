"""An intrinsic edge column gives what the record-per-edge read gave.

`FR-016` asks for one pass over the arrays *and* for the same values in the same
order. The reference here is the read this replaces, called directly: one record
per edge, and the field taken off it.

Two shapes need naming. A **freed edge slot** leaves a hole, so the live slots
are no longer a range and the pass falls back. A **placeholder edge** is an id
the graph knows before the edge exists: it holds no members, occupies no column,
and `G.E.ids` leaves it out — so a slice over every live slot would be one value
too long.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet

INTRINSIC = ('weight', 'directed', 'kind')


def _per_edge(graph, name) -> list:
    """The read this replaces: one record per edge, and one field off it."""
    return [getattr(graph.get_edge(edge_id), name) for edge_id in graph.E.ids]


# ---------------------------------------------------------------------------
# The matrix of cases
# ---------------------------------------------------------------------------


def binary() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(8)])
    graph.add_edges(
        [
            {'source': f'v{i}', 'target': f'v{i + 1}', 'edge_id': f'e{i}', 'weight': float(i)}
            for i in range(6)
        ]
    )
    return graph


def undirected_graph() -> AnnNet:
    graph = AnnNet(directed=False)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(4)])
    graph.add_edges([{'source': 'v0', 'target': 'v1', 'edge_id': 'e0'}])
    graph.add_edges([{'source': 'v1', 'target': 'v2', 'edge_id': 'e1', 'directed': True}])
    return graph


def mixed_declarations() -> AnnNet:
    """Some edges declare a direction and some inherit the graph default."""
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(6)])
    graph.add_edges([{'source': 'v0', 'target': 'v1', 'edge_id': 'e0'}])
    graph.add_edges([{'source': 'v1', 'target': 'v2', 'edge_id': 'e1', 'directed': False}])
    graph.add_edges([{'source': 'v2', 'target': 'v3', 'edge_id': 'e2', 'directed': True}])
    return graph


def with_a_hyperedge() -> AnnNet:
    """A directed hyperedge and an undirected one, whose direction is not declared.

    A hyperedge resolves its direction from the roles of its members and not
    from ``edge_directed``, so it is the one shape a vectorized resolution has
    to be told about.
    """
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(6)])
    graph.add_edges([{'source': 'v0', 'target': 'v1', 'edge_id': 'e0'}])
    graph.add_edges([{'members': ['v2', 'v3', 'v4'], 'edge_id': 'h_flat'}])
    graph.add_edges([{'source': ['v0', 'v2'], 'target': ['v4'], 'edge_id': 'h_directed'}])
    return graph


def with_an_edge_entity() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C'])
    graph.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
    graph.add_edges('ee_ab', 'C', edge_id='e_meta')
    return graph


def with_a_freed_edge_slot() -> AnnNet:
    graph = binary()
    graph.remove_edge('e2')
    return graph


def with_a_placeholder() -> AnnNet:
    graph = binary()
    graph._ensure_edge_entity_placeholder('e_later')
    return graph


def with_a_placeholder_and_a_freed_slot() -> AnnNet:
    graph = with_a_placeholder()
    graph.remove_edge('e1')
    return graph


def empty_of_edges() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': 'only'}])
    return graph


CASES = {
    'binary': binary,
    'undirected_graph': undirected_graph,
    'mixed_declarations': mixed_declarations,
    'with_a_hyperedge': with_a_hyperedge,
    'with_an_edge_entity': with_an_edge_entity,
    'with_a_freed_edge_slot': with_a_freed_edge_slot,
    'with_a_placeholder': with_a_placeholder,
    'with_a_placeholder_and_a_freed_slot': with_a_placeholder_and_a_freed_slot,
    'empty_of_edges': empty_of_edges,
}


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('name', INTRINSIC)
def test_the_column_matches_the_record_per_edge_read(case, name):
    graph = CASES[case]()
    assert np.asarray(graph.E[name]).tolist() == _per_edge(graph, name), (case, name)


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('name', INTRINSIC)
def test_the_column_is_aligned_with_the_edge_ids(case, name):
    graph = CASES[case]()
    assert len(graph.E[name]) == len(graph.E.ids), (case, name)


def test_a_placeholder_never_takes_the_fast_path():
    """`G.E.ids` leaves a placeholder out, so a slice over the slots would not fit."""
    graph = with_a_placeholder()
    assert graph._store.edge_slots_contiguous is True
    assert graph._store.edge_axis_contiguous is False
    assert graph.E['weight'].base is not graph._store.edge_weight


def test_a_declared_direction_is_read_back_after_a_write():
    graph = binary()
    graph.E['directed'] = [False] * len(graph.E.ids)
    assert np.asarray(graph.E['directed']).tolist() == _per_edge(graph, 'directed')
    assert not any(graph.E['directed'])


def test_a_weight_write_shows_in_the_next_read():
    graph = binary()
    graph.E['weight'] = [float(i) * 10 for i in range(len(graph.E.ids))]
    assert np.asarray(graph.E['weight']).tolist() == _per_edge(graph, 'weight')


def test_the_graph_default_decides_an_inherited_direction():
    for default in (True, False):
        graph = AnnNet(directed=default)
        graph.add_nodes([{'node_id': f'v{i}'} for i in range(3)])
        graph.add_edges([{'source': 'v0', 'target': 'v1', 'edge_id': 'e0'}])
        assert np.asarray(graph.E['directed']).tolist() == _per_edge(graph, 'directed')
        assert bool(graph.E['directed'][0]) is default
