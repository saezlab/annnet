"""Every public name the package removed, and the expression that replaces it.

The package is before its first stable release, so a removed name carries no
deprecation and no alias. The cost of that choice falls on a caller who holds
the old name, so the old name has to say what to write instead. A plain
``AttributeError`` does not: it says that something is absent, and leaves the
caller to search for what took its place.

One table per holder, and ``dev/migration.md`` carries the same map in prose
for a reader. ``tests/test_removed_names.py`` holds the two together, so a
removal that reaches only one of them fails the build.
"""

from __future__ import annotations

# The graph object.
GRAPH = {
    'obs': 'G.attrs.nodes',
    'var': 'G.attrs.edges',
    'views': 'G.attrs.table(address, derived=True, ...)',
    'nodes': 'list(G.N) or G.N.ids',
    'edges': 'list(G.E) or G.E.ids',
    'nv': 'len(G.N)',
    'ne': 'len(G.E)',
    'ncount': 'len(G.N), or G.nv_supra for ncount(supra=True)',
    'ecount': 'len(G.E)',
    'get_node': 'G.N.at(node_id)',
    'get_edge': 'G.E.at(edge_id)',
    'get_edge_ids': 'G.has_edge(source, target)[1]',
    'get_edges_by_direction': 'G.E.select(directed=True).ids',
    'in_edges': "G.incident_edges(nodes, direction='in')",
    'out_edges': "G.incident_edges(nodes, direction='out')",
    'remove_node': 'G.remove_nodes(node_id)',
    'remove_edge': 'G.remove_edges(edge_id)',
    'contextual_table': 'G.attrs.table(address)',
}

# The namespaces, by the spelling a caller writes.
NAMESPACES = {
    'G.attrs': {
        'audit_attributes': 'G.attrs.audit()',
        'edge_slice': "G.attrs.row('edge_slices', (slice_id, edge_id))",
        'get_attr_edge': "G.attrs.row('edges', edge_id).get(name, default)",
        'get_attr_edges': "G.attrs.rows('edges', ids)",
        'get_attr_from_edges': 'dict(zip(G.E.ids, G.E.column(name, default)))',
        'get_attr_node': "G.attrs.row('nodes', node_id).get(name, default)",
        'get_attr_nodes': "G.attrs.rows('nodes', ids)",
        'get_edge_attrs': "G.attrs.row('edges', edge_id)",
        'get_edge_slice_attr': (
            "G.attrs.row('edge_slices', (slice_id, edge_id)).get(name, default)"
        ),
        'get_edges_by_attr': 'G.E.select(**{name: value}).ids',
        'get_effective_edge_weight': 'G.E.effective_weight(edge_id, slice)',
        'get_graph_attribute': 'G.uns.get(key, default)',
        'get_graph_attributes': 'dict(G.uns)',
        'get_node_attrs': "G.attrs.row('nodes', node_id)",
        'get_slice_attr': "G.attrs.row('slices', slice_id).get(name, default)",
        'set_edge_attrs': "G.attrs.update('edges', {edge_id: attrs})",
        'set_edge_attrs_bulk': "G.attrs.update('edges', rows)",
        'set_edge_slice_attrs': "G.attrs.update('edge_slices', {(slice_id, edge_id): attrs})",
        'set_edge_slice_attrs_bulk': (
            "G.attrs.update('edge_slices', "
            '{(slice_id, edge_id): attrs for edge_id, attrs in items})'
        ),
        'set_graph_attribute': 'G.uns[key] = value',
        'set_node_attrs': "G.attrs.update('nodes', {node_id: attrs})",
        'set_node_attrs_bulk': "G.attrs.update('nodes', rows)",
        'set_slice_attrs': "G.attrs.update('slices', {slice_id: attrs})",
        'set_slice_edge_weight': (
            "G.attrs.update('edge_slices', {(slice_id, edge_id): {'weight': w}})"
        ),
    },
    'G.layers': {
        'aspect_attrs': "G.attrs.row('aspects', aspect)",
        'attrs': "G.attrs.row('layers', layer)",
        'elementary_attrs': "G.attrs.row('elementary_layers', (aspect, label))",
        'node_attrs': "G.attrs.row('node_layers', (node_id, layer))",
        'set_aspect_attrs': "G.attrs.update('aspects', {aspect: attrs})",
        'set_attrs': "G.attrs.update('layers', {layer: attrs})",
        'set_elementary_attrs': "G.attrs.update('elementary_layers', {(aspect, label): attrs})",
        'set_node_attrs': "G.attrs.update('node_layers', {(node_id, layer): attrs})",
        'set_node_attrs_bulk': "G.attrs.update('node_layers', values, layer=, key=)",
    },
    'G.slices': {
        'attrs': "G.attrs.row('slices', slice_id)",
    },
    'G.cache': {
        'get_adjacency': 'G.cache.adjacency',
        'get_csc': 'G.cache.csc',
        'get_csr': 'G.cache.csr',
    },
    # A view. The last five are names the graph carries and a view does not,
    # which a caller reaches through an independent copy.
    'V': {
        'obs': "V.attrs.table('nodes')",
        'var': "V.attrs.table('edges')",
        'nodes_df': "V.attrs.table('nodes', derived=True)",
        'edges_df': "V.attrs.table('edges', derived=True)",
        'node_count': 'len(V.N)',
        'edge_count': 'len(V.E)',
        'node_ids': 'V.N.ids',
        'edge_ids': 'V.E.ids',
        'subview': 'V.view(...)',
        'X': 'V.B',
        'nx': 'V.materialize().nx',
        'ig': 'V.materialize().ig',
        'gt': 'V.materialize().gt',
        'cache': 'V.materialize().cache',
    },
}


def refuse(holder: str, name: str) -> None:
    """Raise for a name this holder removed, and say what replaces it.

    ``holder`` is the spelling a caller writes, such as ``G.attrs``. The call
    returns when the name is not a removed one, so the caller raises its own
    error for a name that never existed.
    """
    replacement = NAMESPACES.get(holder, {}).get(name)
    if replacement is not None:
        raise AttributeError(f'{holder}.{name} was removed; use {replacement}')
