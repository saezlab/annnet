"""Domain-free topology queries used by downstream method requirements."""

import annnet as an


def test_binary_projection_and_cycle():
    graph = an.Graph(directed=True)
    graph.add_edges('a', 'b', edge_id='ab')
    graph.add_edges('b', 'a', edge_id='ba')
    graph.add_edges('b', 'c', edge_id='bc', directed=False)
    graph.add_edges(['c', 'd'], ['a'], edge_id='hyper')
    assert set(an.directed_pairs(graph)) == {('a', 'b'), ('b', 'a')}
    assert an.sources(graph) == {'a', 'b'}
    assert an.targets(graph) == {'a', 'b'}
    cycle = an.directed_cycle(graph)
    assert cycle[0] == cycle[-1]
    assert set(cycle) == {'a', 'b'}


def test_long_acyclic_path_and_self_loop():
    graph = an.Graph(directed=True)
    for i in range(1500):
        graph.add_edges(str(i), str(i + 1), edge_id=f'e{i}')
    assert an.directed_cycle(graph) is None
    graph.add_edges('1500', '1500', edge_id='loop')
    assert an.directed_cycle(graph) == ['1500', '1500']


def test_edge_entities_are_not_projected_as_nodes():
    graph = an.Graph(directed=True)
    graph.add_edges('a', 'b', edge_id='edge', as_entity=True)
    graph.add_edges('edge', 'c', edge_id='meta')
    assert set(an.directed_pairs(graph)) == {('a', 'b')}
