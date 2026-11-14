"""``anet.rx``: rustworkx calls on the graph, with node ids in and out."""

from __future__ import annotations

import pytest

rx = pytest.importorskip('rustworkx')

from annnet import AnnNet


@pytest.fixture
def chain():
    g = AnnNet(directed=True)
    for u, v, w in [('a', 'b', 1.0), ('b', 'c', 2.0), ('a', 'c', 5.0), ('c', 'd', 1.0)]:
        g.add_edges(u, v, weight=w, edge_id=f'{u}{v}')
    return g


def test_results_are_keyed_by_node_id(chain):
    ranks = chain.rx.pagerank()
    assert set(ranks) == {'a', 'b', 'c', 'd'}
    assert ranks['d'] > ranks['a']


def test_node_arguments_take_ids(chain):
    assert chain.rx.has_path(source='a', target='d')
    assert chain.rx.descendants('a') == {'b', 'c', 'd'}
    assert chain.rx.ancestors(node='c') == {'a', 'b'}


def test_a_string_weight_reads_the_payload(chain):
    paths = chain.rx.dijkstra_shortest_paths(source='a', target='d', weight_fn='weight')
    assert paths == {'d': ['a', 'b', 'c', 'd']}
    lengths = chain.rx.dijkstra_shortest_path_lengths(node='a', edge_cost_fn='weight')
    assert lengths == {'b': 1.0, 'c': 3.0, 'd': 4.0}


def test_node_lists_and_groups(chain):
    assert chain.rx.topological_sort() == ['a', 'b', 'c', 'd']
    groups = chain.rx.strongly_connected_components()
    assert sorted(map(tuple, groups)) == [('a',), ('b',), ('c',), ('d',)]


def test_edge_results_are_keyed_by_edge_id(chain):
    scores = chain.rx.edge_betweenness_centrality()
    assert set(scores) == {'ab', 'bc', 'ac', 'cd'}


def test_raw_results_are_untouched(chain):
    raw = chain.rx.topological_sort(_rx_raw=True)
    assert isinstance(raw, rx.NodeIndices)


def test_a_slice_limits_the_graph(chain):
    chain.slices.add('s', edges=['ab'])
    assert set(chain.rx.pagerank(_rx_slice='s')) == {'a', 'b'}
    assert chain.rx.backend(slice='s').num_edges() == 1


def test_the_conversion_is_cached_until_the_graph_changes(chain):
    first = chain.rx.backend()
    assert chain.rx.backend() is first
    chain.add_edges('d', 'e')
    assert chain.rx.backend() is not first
    assert 'e' in chain.rx.pagerank()


def test_an_unknown_function_raises(chain):
    with pytest.raises(AttributeError, match='rustworkx has no callable'):
        chain.rx.not_a_function()


def test_a_view_points_at_materialize(chain):
    with pytest.raises(AttributeError, match='materialize'):
        _ = chain.view(nodes=['a', 'b']).rx
