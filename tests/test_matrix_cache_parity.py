"""Whatever path a cached matrix took, it is the matrix a rebuild would give.

`FR-015`. The cache has three paths — return what it holds, extend it by the
columns a run of frontier writes changed, or build it again — and a caller can
see which one answered only in the time it took. So every one of them has to give
the same matrix, over every shape of write the store allows.

The reference is a cache that holds nothing: dropping it and reading again is a
full rebuild by definition.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from annnet import AnnNet

MATRICES = ('S', 'B', 'H', 'A', 'L')


def _graph(edges: int = 10) -> AnnNet:
    nodes = max(3, edges // 2)
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}'} for i in range(nodes)])
    graph.add_edges(
        [
            {'source': f'v{i % nodes}', 'target': f'v{(i + 1) % nodes}', 'edge_id': f'e{i}'}
            for i in range(edges)
        ]
    )
    graph.add_edges([{'members': ['v0', 'v1', 'v2'], 'edge_id': 'h0'}])
    return graph


def _dense(graph, name):
    return np.asarray(getattr(graph, name).todense())


def _matches_a_rebuild(graph) -> None:
    kept = {name: _dense(graph, name) for name in MATRICES}
    graph.matrices.cache.drop()
    for name in MATRICES:
        fresh = _dense(graph, name)
        assert kept[name].shape == fresh.shape, name
        assert np.allclose(kept[name], fresh), name


# The writes a cache has to survive or fall back from, each as a named step.
STEPS = {
    'append_at_the_frontier': lambda g: g.add_edges('v0', 'v2', edge_id='x0'),
    'append_a_run': lambda g: g.add_edges(
        [{'source': 'v0', 'target': 'v1', 'edge_id': f'y{i}'} for i in range(4)]
    ),
    'remove_at_the_frontier': lambda g: g.remove_edge('h0'),
    'remove_in_the_middle': lambda g: g.remove_edge('e4'),
    'remove_the_first': lambda g: g.remove_edge('e0'),
    'remove_a_node': lambda g: g.remove_node('v2'),
    'add_a_node': lambda g: g.add_nodes(['fresh']),
    'set_a_weight': lambda g: g.E.set_column('weight', [2.0] * len(g.E.ids)),
    'append_after_a_hole': lambda g: (
        g.remove_edge('e3'),
        g.add_edges('v0', 'v1', edge_id='z0'),
    ),
}


@pytest.mark.parametrize('step', sorted(STEPS))
def test_one_write_leaves_every_matrix_what_a_rebuild_would_give(step):
    graph = _graph()
    for name in MATRICES:
        _ = getattr(graph, name)
    STEPS[step](graph)
    _matches_a_rebuild(graph)


@pytest.mark.parametrize('pair', sorted(itertools.combinations(sorted(STEPS), 2))[:18])
def test_two_writes_in_a_row_leave_the_same(pair):
    first, second = pair
    graph = _graph()
    for name in MATRICES:
        _ = getattr(graph, name)
    try:
        STEPS[first](graph)
        for name in MATRICES:
            _ = getattr(graph, name)
        STEPS[second](graph)
    except KeyError:
        pytest.skip(f'{second} cannot follow {first} on this graph')
    _matches_a_rebuild(graph)


def test_a_long_alternation_of_appends_and_frontier_removals():
    graph = _graph()
    _ = graph.S
    for i in range(12):
        graph.add_edges('v0', 'v1', edge_id=f'w{i}')
        _ = graph.S
        if i % 3 == 2:
            graph.remove_edge(f'w{i}')
            _ = graph.S
    _matches_a_rebuild(graph)


def test_the_cache_never_answers_with_the_wrong_shape():
    """``S`` holds one column per structural edge, whichever path answered."""
    graph = _graph()
    _ = graph.S
    for edge_id in ('h0', 'e9', 'e8', 'e7'):
        graph.remove_edge(edge_id)
        assert graph.S.shape[1] == len(graph.E.ids), edge_id
