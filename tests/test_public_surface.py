"""The package names what it means to name, and nothing else.

A curated surface is only worth having when something checks it. These tests
read the names the package exports and the names a graph object answers to,
and compare them to the contract in ``docs/reference/api-contract.md``
and to the machine-readable inventory ``docs/reference/public-surface.json``,
in both directions: a name the contract states must be on the object, and a
name the object carries must be in the contract.

A removed name is caught by the second half: a name the surface still carries
that the contract does not name fails, so a removal that missed one place shows
up here rather than in a user's traceback.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import annnet
from annnet.core.graph import AnnNet, REMOVED_GRAPH_NAMES

DOCS = Path(annnet.__file__).parent.parent / 'docs' / 'reference'
INVENTORY = DOCS / 'public-surface.json'
CONTRACT = DOCS / 'api-contract.md'


def _contract_paragraph(marker: str) -> str:
    """The paragraph of the contract that follows ``marker``."""
    text = CONTRACT.read_text()
    start = text.index(marker) + len(marker)
    rest = text[start:].lstrip('\n')
    return rest.split('\n\n', 1)[0]


def _contract_names(marker: str, holder: str) -> set:
    return set(re.findall(rf'`{holder}\.(\w+)`', _contract_paragraph(marker)))


# What the contract states, section by section.
CONTRACT_SURFACE = {
    # Add and remove elements
    'add_nodes',
    'add_edges',
    'remove_nodes',
    'remove_edges',
    # Axes, counts and identity
    'N',
    'E',
    'nv_supra',
    'shape',
    'supra_shape',
    'supra_nodes',
    'at',
    'exists',
    'has_node',
    'has_edge',
    'entity_kinds',
    # Attributes and metadata
    'attrs',
    'uns',
    'summary',
    # Matrices
    'A',
    'B',
    'H',
    'S',
    'L',
    'matrices',
    # Lookups and traversal
    'neighbors',
    'in_neighbors',
    'out_neighbors',
    'predecessors',
    'successors',
    'degree',
    'incident_edges',
    'idx',
    # Namespaces
    'ops',
    'slices',
    'layers',
    'history',
    'provenance',
    'spaces',
    'nx',
    'ig',
    'gt',
    'cache',
    # Views and IO
    'view',
    'read',
    'write',
}

# Names the object carries that the contract lists as retained operations.
BEYOND_THE_CONTRACT = {
    'edge_list': 'every binary edge as a tuple, which a caller writing a file wants',
    'global_count': 'one count of one kind of slice member, by name',
    'is_multilayer': 'whether the graph declares more than the flat aspect',
    'make_undirected': 'drop the direction of every edge in place',
}

REMOVED = (
    'X',
    'num_nodes',
    'num_edges',
    'num_supra_nodes',
    'number_of_nodes',
    'number_of_edges',
    'global_node_count',
    'global_edge_count',
    'entity_to_idx',
    'idx_to_entity',
    'edge_to_idx',
    'idx_to_edge',
    'entity_types',
    'node_attributes',
    'edge_attributes',
    # removed with the attribute/selection/view rework
    'obs',
    'var',
    'views',
    'nodes',
    'edges',
    'nv',
    'ne',
    'ncount',
    'ecount',
    'get_node',
    'get_edge',
    'get_edge_ids',
    'get_edges_by_direction',
    'in_edges',
    'out_edges',
    'remove_node',
    'remove_edge',
    'contextual_table',
)


@pytest.fixture
def graph() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
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
    """Every removed name is gone from the object."""
    for name in REMOVED:
        with pytest.raises(AttributeError):
            _ = getattr(graph, name)


def test_a_removed_name_names_its_replacement(graph):
    """A caller holding an old name meets the new one."""
    for name, replacement in REMOVED_GRAPH_NAMES.items():
        with pytest.raises(AttributeError, match='was removed') as caught:
            _ = getattr(graph, name)
        assert replacement.split('(')[0].split(',')[0] in str(caught.value)


def test_the_lazy_view_carries_no_matrix_name_the_graph_dropped(graph):
    """``X`` went from the graph and stays off the view."""
    view = graph.view(nodes=['A'])
    assert not hasattr(view, 'X')
    with pytest.raises(AttributeError):
        _ = view.X
    assert view.B.shape[0] == 1


def test_the_package_exports_what_it_lists():
    """Every name in ``__all__`` resolves, and importing costs no backend."""
    for name in annnet.__all__:
        assert getattr(annnet, name) is not None or name in {'__license__'}


def test_the_core_exports_the_graph_and_the_records():
    import annnet.core as core

    assert sorted(core.__all__) == sorted(
        [
            'AnnNet',
            'Aspect',
            'Attrs',
            'BOUNDARIES',
            'Graph',
            'GraphView',
            'ContextualValues',
            'EdgeSequence',
            'LayerSelection',
            'MatrixValues',
            'NodeSequence',
            'Provenance',
            'RowSelection',
            'Schema',
            'Set',
            'Space',
            'ValueMatrix',
            'ValueResolver',
            'OrderedLabels',
            'EdgeType',
            'EdgeView',
            'Endpoint',
            'NodeView',
            'as_aspect',
            'as_endpoint',
            'as_endpoints',
        ]
    )
    for name in core.__all__:
        assert hasattr(core, name)


def test_a_node_lookup_takes_an_id_and_answers_with_one(graph):
    view = graph.N.at('A')
    assert view == 'A'
    assert view.kind == 'node'
    assert view.layers == (('_',),)
    with pytest.raises(TypeError):
        graph.N.at(0)
    with pytest.raises(KeyError):
        graph.N.at('ghost')


def test_no_public_name_is_spelled_with_the_long_form_of_attribute(graph):
    """One spelling of *attribute*, the short one."""
    offenders = []
    for owner_name in ('attrs', 'layers', 'slices', 'ops', 'history', 'idx', 'cache', 'matrices'):
        owner = getattr(graph, owner_name)
        offenders.extend(
            f'{owner_name}.{n}' for n in dir(owner) if 'attribute' in n and not n.startswith('_')
        )
    offenders.extend(n for n in dir(graph) if 'attribute' in n)
    assert offenders == []


def test_verb_prefixed_attribute_names_fell_by_at_least_half(graph):
    """The surface once carried 66 ``get_``/``set_`` names; it holds at most half of that."""
    found = []
    for owner_name in (
        'attrs',
        'layers',
        'slices',
        'ops',
        'history',
        'idx',
        'cache',
        'matrices',
        'provenance',
    ):
        owner = getattr(graph, owner_name)
        found.extend(
            f'{owner_name}.{n}'
            for n in dir(owner)
            if n.startswith(('get_', 'set_')) and not n.startswith('_')
        )
    found.extend(n for n in dir(graph) if n.startswith(('get_', 'set_')))
    assert len(found) <= 33, found


def test_no_name_means_two_things_in_two_namespaces(graph):
    """The five table names live under ``attrs`` alone."""
    for name in ('nodes', 'edges', 'slices', 'aspects', 'layers'):
        assert name in dir(graph.attrs)
        assert not hasattr(graph, 'views')
    assert not hasattr(graph.layers, 'set_node_attrs')
    assert not hasattr(graph.slices, 'attrs')


@pytest.mark.skipif(
    not CONTRACT.is_file(), reason='docs/reference/api-contract.md is not beside the package'
)
def test_the_contract_states_the_graph_surface_both_ways(graph):
    """Section 2.9 of the contract is exactly what the object answers to."""
    stated = _contract_names(
        'Every public name an `AnnNet` object answers to, and nothing else:', 'G'
    )
    assert stated == set(dir(graph)), {
        'stated but absent': sorted(stated - set(dir(graph))),
        'present but unstated': sorted(set(dir(graph)) - stated),
    }


@pytest.mark.skipif(
    not CONTRACT.is_file(), reason='docs/reference/api-contract.md is not beside the package'
)
def test_the_contract_states_the_view_surface_both_ways(graph):
    """The view paragraph of section 5 is exactly what a view answers to."""
    view = graph.view()
    stated = _contract_names('The surface of a view:', 'V')
    assert stated == set(dir(view)), {
        'stated but absent': sorted(stated - set(dir(view))),
        'present but unstated': sorted(set(dir(view)) - stated),
    }


@pytest.mark.skipif(
    not CONTRACT.is_file(), reason='docs/reference/api-contract.md is not beside the package'
)
def test_the_contract_states_the_core_exports():
    import annnet.core as core

    paragraph = _contract_paragraph('`annnet.core.__all__` is exactly:')
    stated = set(re.findall(r'`(\w+)`', paragraph))
    assert stated == set(core.__all__)


@pytest.mark.skipif(
    not INVENTORY.is_file(), reason='docs/reference/public-surface.json is not beside the package'
)
def test_public_inventory_matches_package_and_namespaces(graph):
    """The machine-readable inventory, checked both ways."""
    import annnet.core as core

    inventory = json.loads(INVENTORY.read_text())
    assert sorted(annnet.__all__) == inventory['annnet']
    assert sorted(core.__all__) == inventory['core']
    assert sorted(dir(graph)) == inventory['graph']
    for name, expected in inventory['namespaces'].items():
        actual = sorted(n for n in dir(getattr(graph, name)) if not n.startswith('_'))
        assert actual == expected, name
    view = graph.view()
    assert sorted(dir(view)) == inventory['view']
    assert (
        sorted(n for n in dir(view.attrs) if not n.startswith('_'))
        == inventory['namespaces']['attrs']
    )


CHANGELOG = Path(annnet.__file__).parent.parent / 'CHANGELOG.md'
MIGRATION = DOCS / 'migration.md'

# Every public name the attribute/selection/view rework removed, by holder. The
# graph ones are the keys of REMOVED_GRAPH_NAMES; the rest were read off the
# surface before the rework and are pinned here so the record cannot drift.
REMOVED_NAMES_BY_HOLDER = {
    'G': tuple(REMOVED_GRAPH_NAMES),
    'G.attrs': (
        'audit_attributes',
        'edge_slice',
        'get_attr_edge',
        'get_attr_edges',
        'get_attr_from_edges',
        'get_attr_node',
        'get_attr_nodes',
        'get_edge_attrs',
        'get_edge_slice_attr',
        'get_edges_by_attr',
        'get_effective_edge_weight',
        'get_graph_attribute',
        'get_graph_attributes',
        'get_node_attrs',
        'get_slice_attr',
        'set_edge_attrs',
        'set_edge_attrs_bulk',
        'set_edge_slice_attrs',
        'set_edge_slice_attrs_bulk',
        'set_graph_attribute',
        'set_node_attrs',
        'set_node_attrs_bulk',
        'set_slice_attrs',
        'set_slice_edge_weight',
    ),
    'G.layers': (
        'aspect_attrs',
        'attrs',
        'elementary_attrs',
        'node_attrs',
        'set_aspect_attrs',
        'set_attrs',
        'set_elementary_attrs',
        'set_node_attrs',
        'set_node_attrs_bulk',
    ),
    'G.slices': ('attrs',),
    'G.cache': ('get_adjacency', 'get_csc', 'get_csr'),
    'V': (
        'obs',
        'var',
        'nodes_df',
        'edges_df',
        'node_count',
        'edge_count',
        'node_ids',
        'edge_ids',
        'subview',
    ),
}


def test_every_removed_name_is_gone(graph):
    holders = {
        'G': graph,
        'G.attrs': graph.attrs,
        'G.layers': graph.layers,
        'G.slices': graph.slices,
        'G.cache': graph.cache,
        'V': graph.view(),
    }
    still = [
        f'{holder}.{name}'
        for holder, names in REMOVED_NAMES_BY_HOLDER.items()
        for name in names
        if hasattr(holders[holder], name)
    ]
    assert still == []


@pytest.mark.skipif(not CHANGELOG.is_file(), reason='CHANGELOG.md is not beside the package')
def test_every_removed_name_is_in_the_changelog_and_the_migration_page():
    """A removed name is findable with its replacement."""
    changelog = CHANGELOG.read_text()
    section = changelog[
        changelog.index('### One way to reach every attribute') : changelog.index(
            '### Changed, and a caller can see it\n\n- **A column read'
        )
    ]
    migration = MIGRATION.read_text()
    missing = []
    for holder, names in REMOVED_NAMES_BY_HOLDER.items():
        for name in names:
            if re.search(rf'\b{name}\b', section) is None:
                missing.append(f'{holder}.{name} (changelog)')
            if re.search(rf'\b{name}\b', migration) is None:
                missing.append(f'{holder}.{name} (migration page)')
    assert missing == []
