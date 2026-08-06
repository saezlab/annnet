from __future__ import annotations

from .common import (
    annnet,
    make_nodes,
    scale_note,
    time_record,
    capped_scale,
    make_edge_pairs,
    make_edge_records,
    make_node_records,
    build_annnet_graph,
)


def primitive_dimensions(
    scale,
    *,
    backend: str = 'auto',
    samples: int = 3,
    max_nodes: int = 2_500,
    max_edges: int = 10_000,
    max_accessor_repeats: int = 5,
) -> list[dict]:
    original = scale
    scale = capped_scale(
        scale,
        max_nodes=max_nodes,
        max_edges=max_edges,
        max_accessor_repeats=max_accessor_repeats,
    )
    AnnNet = annnet()
    nodes = make_nodes(scale.nodes)
    pairs = make_edge_pairs(scale.nodes, scale.edges)
    records = make_edge_records(pairs)
    suffix = scale_note(original, scale)

    def note(text: str) -> str:
        return f'{text}; {suffix}' if suffix else text

    def empty():
        return AnnNet(directed=True, annotations_backend=backend)

    def add_nodes_bulk():
        graph = empty()
        graph.add_nodes(make_node_records(nodes), slice='base')
        return graph

    def add_nodes_repeated():
        graph = empty()
        for node_id in nodes:
            graph.add_nodes(node_id, slice='base')
        return graph

    def add_edges_bulk():
        graph = empty()
        graph.add_nodes(nodes, slice='base')
        graph.add_edges(records, slice='base')
        return graph

    def add_edges_repeated():
        graph = empty()
        graph.add_nodes(nodes, slice='base')
        for row in records:
            graph.add_edges(
                row['source'],
                row['target'],
                edge_id=row['edge_id'],
                weight=row['weight'],
                slice='base',
            )
        return graph

    def remove_edges_fraction():
        graph, _nodes, _pairs, edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_edges(edge_ids[: scale.remove_edges], errors='raise')
        return graph

    def remove_nodes_fraction():
        graph, nodes, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_nodes(nodes[: scale.remove_nodes], errors='raise')
        return graph

    cases = (
        ('create_empty', empty, note('construct an empty AnnNet graph')),
        ('add_nodes_bulk', add_nodes_bulk, note('bulk node insertion')),
        ('add_nodes_repeated', add_nodes_repeated, note('one public call per node')),
        ('add_edges_bulk', add_edges_bulk, note('bulk edge insertion with explicit edge ids')),
        ('add_edges_repeated', add_edges_repeated, note('one public call per edge')),
        (
            'remove_edges_fraction',
            remove_edges_fraction,
            note(f'remove {scale.remove_edges} edges'),
        ),
        (
            'remove_nodes_fraction',
            remove_nodes_fraction,
            note(f'remove {scale.remove_nodes} nodes'),
        ),
    )
    return [
        time_record('annnet', 'primitives', op, scale, fn, backend, samples, note_)
        for op, fn, note_ in cases
    ]
