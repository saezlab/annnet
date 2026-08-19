"""What the supra layer must get right, pinned.

Three bugs sat here at once, and none of them raised: a directed edge came back
symmetric, an edge spelled with bare ids was invisible, and the index kept a node
count the graph had outgrown. Each was silent — wrong numbers, never an error —
so each gets a test that fails loudly if it returns.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from annnet import AnnNet
from annnet.core import _structure as S


def two_layer(directed: bool = True) -> AnnNet:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=directed)
        G.layers.set_aspects(['cond'], {'cond': ['a', 'b']})
        for layer in (('a',), ('b',)):
            G.add_nodes(['x', 'y'], layer=layer)
        G.add_edges([{'source': ('x', ('a',)), 'target': ('y', ('a',)), 'edge_id': 'e0'}])
    return G


def row(G: AnnNet, node: str, label: str) -> int:
    return G.layers.nl_to_row(node, (label,))


# ---------------------------------------------------------------------------
# A directed edge stays directed
# ---------------------------------------------------------------------------


def test_a_directed_intra_edge_occupies_one_cell():
    G = two_layer(directed=True)
    A = G.layers.supra_adjacency().toarray()
    assert A[row(G, 'x', 'a'), row(G, 'y', 'a')] == 1.0
    assert A[row(G, 'y', 'a'), row(G, 'x', 'a')] == 0.0


def test_the_supra_matrix_of_a_directed_graph_is_not_symmetric():
    A = two_layer(directed=True).layers.supra_adjacency().toarray()
    assert not np.allclose(A, A.T)


def test_an_undirected_graph_is_still_symmetric():
    A = two_layer(directed=False).layers.supra_adjacency().toarray()
    assert np.allclose(A, A.T)


def test_the_supra_projection_agrees_with_the_flat_matrix():
    """One layer holding every node is the flat graph, so the two must agree."""
    G = two_layer(directed=True)
    A = G.layers.supra_adjacency().toarray()
    assert A.sum() == G.A.toarray().sum()


def test_a_coupling_edge_is_symmetric_because_it_joins_a_node_to_itself():
    G = two_layer(directed=True)
    G.layers.add_categorical_coupling('cond', [['a', 'b']])
    A = G.layers.supra_adjacency().toarray()
    assert A[row(G, 'x', 'a'), row(G, 'x', 'b')] == 1.0
    assert A[row(G, 'x', 'b'), row(G, 'x', 'a')] == 1.0


def test_an_ordinal_coupling_can_be_asked_for_and_points_one_way():
    G = two_layer(directed=True)
    G.layers.add_categorical_coupling('cond', [['a', 'b']], directed=True)
    A = G.layers.supra_adjacency().toarray()
    assert A[row(G, 'x', 'a'), row(G, 'x', 'b')] == 1.0
    assert A[row(G, 'x', 'b'), row(G, 'x', 'a')] == 0.0


# ---------------------------------------------------------------------------
# How an endpoint was spelled cannot change what the edge is
# ---------------------------------------------------------------------------


def spelled(explicit: bool) -> AnnNet:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True)
        G.layers.set_aspects(['cond'], {'cond': ['a']})
        G.add_nodes(['x', 'y'], layer=('a',))
        if explicit:
            G.add_edges([{'source': ('x', ('a',)), 'target': ('y', ('a',))}])
        else:
            G.add_edges('x', 'y')
    return G


@pytest.mark.parametrize('explicit', [True, False])
def test_an_edge_is_classified_by_what_it_joins_not_how_it_was_written(explicit):
    G = spelled(explicit)
    assert next(iter(S.iter_edges(G))).ml_kind == 'intra'


@pytest.mark.parametrize('explicit', [True, False])
def test_a_bare_id_edge_reaches_the_supra_matrix(explicit):
    G = spelled(explicit)
    assert G.layers.supra_adjacency().nnz == 1
    assert G.layers.layer_edge_set(('a',))


def test_both_spellings_give_the_same_matrix():
    left = spelled(True).layers.supra_adjacency().toarray()
    right = spelled(False).layers.supra_adjacency().toarray()
    assert np.array_equal(left, right)


# ---------------------------------------------------------------------------
# The cache follows the graph
# ---------------------------------------------------------------------------


def test_a_node_added_after_a_read_widens_the_matrix():
    """``add_node`` advances the clock but never reached the hook that dropped
    this cache, so the index kept the old node count and the matrix the old shape.
    """
    G = two_layer(directed=False)
    before = G.layers.supra_adjacency().shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.add_nodes(['z'], layer=('a',))
    assert G.layers.supra_adjacency().shape[0] == before + 1


def test_an_edge_added_after_a_read_shows_up():
    G = two_layer(directed=False)
    before = G.layers.supra_adjacency().nnz
    G.add_edges([{'source': ('x', ('b',)), 'target': ('y', ('b',))}])
    assert G.layers.supra_adjacency().nnz > before


def test_a_weight_write_moves_the_matrix():
    G = two_layer(directed=False)
    G.layers.supra_adjacency()
    G.E['weight'] = [5.0] * G.ne
    assert G.layers.supra_adjacency().max() == 5.0


def test_a_second_read_is_the_cached_matrix():
    G = two_layer(directed=False)
    assert G.layers.supra_adjacency() is G.layers.supra_adjacency()


def test_the_derived_matrices_cost_no_second_build():
    """They are built from the adjacency, so a warm read must not rebuild it."""
    G = two_layer(directed=False)
    G.layers.supra_adjacency()
    calls = []
    accessor = type(G.layers)
    original = accessor._build_supra_adjacency
    accessor._build_supra_adjacency = lambda self, layers=None: (
        calls.append(1) or original(self, layers)
    )
    try:
        G.layers.supra_laplacian()
        G.layers.transition_matrix()
        G.layers.supra_degree()
    finally:
        accessor._build_supra_adjacency = original
    assert calls == [], f'the adjacency was rebuilt {len(calls)} times'


# ---------------------------------------------------------------------------
# The namespace shows operations, and the metrics agree on a partition
# ---------------------------------------------------------------------------


def test_the_layer_namespace_does_not_advertise_the_fields_of_the_graph():
    """``__getattr__`` forwards ``__dict__`` too, so ``dir()`` reported the
    graph's whole instance state as though it were layer operations.
    """
    G = two_layer()
    listed = dir(G.layers)
    assert 'graph_attributes' not in listed
    assert 'node_aligned' not in listed
    # the layer-aware mutation forwarders stay, because they take a layer=
    for name in ('add_nodes', 'add_edges', 'remove_node', 'remove_edge'):
        assert name in listed
    for name in ('supra_adjacency', 'versatility', 'iter_layers'):
        assert name in listed


def modularity_graph() -> AnnNet:
    G = two_layer(directed=False)
    G.layers.add_categorical_coupling('cond', [['a', 'b']])
    return G


def test_a_partition_may_be_a_mapping_or_a_sequence():
    """Every metric beside this one returns a mapping, so it accepts one."""
    G = modularity_graph()
    n = G.layers.supra_adjacency().shape[0]
    from_sequence = G.layers.multislice_modularity([0] * n)
    from_mapping = G.layers.multislice_modularity(dict.fromkeys(range(n), 0))
    assert from_sequence == from_mapping


def test_a_partition_that_does_not_cover_every_row_says_so():
    G = modularity_graph()
    with pytest.raises(ValueError, match='missing'):
        G.layers.multislice_modularity({0: 0})


def test_a_partition_of_the_wrong_length_says_so():
    G = modularity_graph()
    n = G.layers.supra_adjacency().shape[0]
    with pytest.raises(ValueError, match='partition length'):
        G.layers.multislice_modularity([0] * (n - 1))


# ---------------------------------------------------------------------------
# Every cached matrix follows the graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'call',
    [
        lambda L: L.supra_adjacency(),
        lambda L: L.supra_adjacency_scaled(),
        lambda L: L.build_intra_block(),
        lambda L: L.build_coupling_block(),
    ],
)
def test_a_cached_matrix_is_rebuilt_after_a_mutation(call):
    G = two_layer(directed=False)
    G.layers.add_categorical_coupling('cond', [['a', 'b']])
    before = call(G.layers).shape[0]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.add_nodes(['w'], layer=('a',))
    assert call(G.layers).shape[0] == before + 1


def test_a_scaled_matrix_keeps_its_parameters_apart():
    G = two_layer(directed=False)
    G.layers.add_categorical_coupling('cond', [['a', 'b']])
    assert G.layers.supra_adjacency_scaled(coupling_scale=3.0).max() == 3.0
    assert G.layers.supra_adjacency_scaled(coupling_scale=1.0).max() == 1.0


# ---------------------------------------------------------------------------
# A subgraph places every node on a real layer
# ---------------------------------------------------------------------------


def three_node_two_layer() -> AnnNet:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=False)
        G.layers.set_aspects(['cond'], {'cond': ['a', 'b']})
        for layer in (('a',), ('b',)):
            G.add_nodes(['x', 'y', 'z'], layer=layer)
        G.add_edges(
            [
                {'source': ('x', ('a',)), 'target': ('y', ('a',))},
                {'source': ('y', ('a',)), 'target': ('z', ('b',))},
                {'source': ('x', ('b',)), 'target': ('y', ('b',))},
            ]
        )
        G.layers.add_categorical_coupling('cond', [['a', 'b']])
    return G


def entity_keys(graph: AnnNet) -> list[str]:
    return sorted(str(ref.key) for ref in S.iter_entities(graph))


@pytest.mark.parametrize('include_inter', [False, True])
@pytest.mark.parametrize('include_coupling', [False, True])
def test_one_layer_reached_two_ways_gives_one_answer(include_inter, include_coupling):
    """``layer_union`` answers in bare ids, so the generic path had no layer to
    place a node on and dropped it on the placeholder instead.
    """
    kwargs = {'include_inter': include_inter, 'include_coupling': include_coupling}
    from_tuple = three_node_two_layer().layers.subgraph_from_layer_tuple(('a',), **kwargs)
    from_union = three_node_two_layer().layers.subgraph_from_layer_union([('a',)], **kwargs)
    assert entity_keys(from_tuple) == entity_keys(from_union)


@pytest.mark.parametrize(
    'call',
    [
        lambda L: L.subgraph_from_layer_tuple(('a',)),
        lambda L: L.subgraph_from_layer_union([('a',), ('b',)]),
        lambda L: L.subgraph_from_layer_intersection([('a',), ('b',)]),
        lambda L: L.subgraph_from_layer_difference(('a',), ('b',)),
    ],
)
def test_no_subgraph_leaves_a_node_on_the_placeholder_layer(call):
    sub = call(three_node_two_layer().layers)
    placeholder = [key for key in entity_keys(sub) if "'_'" in key]
    assert not placeholder, f'nodes dropped on the placeholder layer: {placeholder}'


def test_a_subgraph_keeps_the_aspects_it_came_from():
    sub = three_node_two_layer().layers.subgraph_from_layer_union([('a',)])
    assert sub.is_multilayer
    assert sub.layers.list_aspects() == ('cond',)


def test_a_node_is_only_placed_on_a_layer_it_is_really_in():
    """``_layer_keys`` asks the graph, so a union cannot invent a presence."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=False)
        G.layers.set_aspects(['cond'], {'cond': ['a', 'b']})
        G.add_nodes(['only_a'], layer=('a',))
        G.add_nodes(['both'], layer=('a',))
        G.add_nodes(['both'], layer=('b',))
    sub = G.layers.subgraph_from_layer_union([('a',), ('b',)])
    assert "('only_a', ('b',))" not in entity_keys(sub)
    assert "('only_a', ('a',))" in entity_keys(sub)
