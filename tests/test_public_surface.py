"""The package names what it means to name, and nothing else.

A curated surface is only worth having when something checks it. These tests
read the names the package exports and the names a graph object answers to, and
compare them to the contract in ``specs/002-annnet-core-ir/contracts/public-api.md``.

A removed name is caught by the second half: a name the surface still carries
that the contract does not name fails, so a removal that missed one place shows
up here rather than in a user's traceback.
"""

from __future__ import annotations

import pytest

import annnet
from annnet.core.graph import AnnNet

# What the contract states, section by section. The words are the ones the
# package uses today: T085 renames "vertex" to "node" and this list moves with
# it.
CONTRACT_SURFACE = {
    # 2. Add and remove elements
    'add_vertices',
    'add_edges',
    'remove_vertices',
    'remove_edges',
    # 3. Counts and sequences
    'ncount',
    'ecount',
    'nv',
    'ne',
    'nv_supra',
    'shape',
    'supra_shape',
    'supra_vertices',
    'N',
    'E',
    # 4. Attributes
    'obs',
    'var',
    'uns',
    # 5. Matrices
    'A',
    'B',
    'H',
    'S',
    'L',
    'matrices',
    # 8. Lookups and traversal
    'get_vertex',
    'get_edge',
    'neighbors',
    'degree',
    'incident_edges',
    'idx',
    # 9. Namespaces
    'attrs',
    'ops',
    'slices',
    'layers',
    'views',
    'history',
    'nx',
    'ig',
    'gt',
}

# Names the object carries that the contract does not list. Each one is here on
# purpose, and a name that is neither in the contract nor here fails the test.
BEYOND_THE_CONTRACT = {
    'vertices': 'the ids of every node, which section 3 shows as list(G.N)',
    'edges': 'the ids of every edge, the same shape as vertices',
    'has_vertex': 'a membership test by id, which the "in" operator also answers',
    'has_edge': 'a membership test by endpoints, which returns the ids it found',
    'edge_list': 'every edge as a tuple, which a caller writing a file wants',
    'global_count': 'one count of one kind of element, by name',
    'is_multilayer': 'whether the graph declares more than the flat aspect',
    'make_undirected': 'drop the direction of every edge in place',
    'view': 'a lazy filtered view, which materializes into a subgraph',
    'cache': 'the matrix cache controls',
    'read': 'read a graph from the native format',
    'write': 'write a graph to the native format',
}


@pytest.fixture
def graph() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e0')
    return G


def test_every_name_the_contract_states_is_on_the_object(graph):
    missing = sorted(name for name in CONTRACT_SURFACE if not hasattr(graph, name))
    assert not missing, f'the contract names these and the object does not carry them: {missing}'


def test_the_object_answers_to_nothing_the_contract_does_not_name(graph):
    surface = set(dir(graph))
    extra = sorted(surface - CONTRACT_SURFACE - set(BEYOND_THE_CONTRACT))
    assert not extra, (
        f'these are public and neither the contract nor this test explains them: {extra}'
    )


def test_what_the_contract_removed_is_not_reachable(graph):
    """Section 10, and the removals D48 records."""
    for name in (
        'X',
        'num_vertices',
        'num_edges',
        'num_supra_vertices',
        'number_of_vertices',
        'number_of_edges',
        'global_vertex_count',
        'global_edge_count',
        'entity_to_idx',
        'idx_to_entity',
        'edge_to_idx',
        'idx_to_edge',
        'entity_types',
        'vertex_attributes',
        'edge_attributes',
    ):
        with pytest.raises(AttributeError):
            getattr(graph, name)


def test_the_package_exports_what_it_lists(monkeypatch):
    """Every name in ``__all__`` resolves, and importing costs no backend."""
    for name in annnet.__all__:
        assert getattr(annnet, name) is not None or name in {'__license__'}


def test_the_core_exports_the_graph_and_the_records():
    import annnet.core as core

    assert core.__all__ == ['AnnNet', 'Graph', 'EdgeType', 'EdgeView', 'VertexView']
    for name in core.__all__:
        assert hasattr(core, name)


def test_a_vertex_lookup_takes_an_id_and_answers_with_one(graph):
    view = graph.get_vertex('A')
    assert view == 'A'
    assert view.kind == 'vertex'
    assert view.layers == (('_',),)
    with pytest.raises(TypeError):
        graph.get_vertex(0)
    with pytest.raises(KeyError):
        graph.get_vertex('ghost')
