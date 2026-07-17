from __future__ import annotations

from .common import (
    annnet,
    scale_note,
    time_record,
    capped_scale,
    make_vertices,
    make_edge_pairs,
    make_edge_records,
    build_annnet_graph,
    make_vertex_records,
)


def primitive_dimensions(
    scale,
    *,
    backend: str = 'auto',
    samples: int = 3,
    max_vertices: int = 2_500,
    max_edges: int = 10_000,
    max_accessor_repeats: int = 5,
) -> list[dict]:
    original = scale
    scale = capped_scale(
        scale,
        max_vertices=max_vertices,
        max_edges=max_edges,
        max_accessor_repeats=max_accessor_repeats,
    )
    AnnNet = annnet()
    vertices = make_vertices(scale.vertices)
    pairs = make_edge_pairs(scale.vertices, scale.edges)
    records = make_edge_records(pairs)
    suffix = scale_note(original, scale)

    def note(text: str) -> str:
        return f'{text}; {suffix}' if suffix else text

    def empty():
        return AnnNet(directed=True, annotations_backend=backend)

    def add_vertices_bulk():
        graph = empty()
        graph.add_vertices(make_vertex_records(vertices), slice='base')
        return graph

    def add_vertices_repeated():
        graph = empty()
        for vertex_id in vertices:
            graph.add_vertices(vertex_id, slice='base')
        return graph

    def add_edges_bulk():
        graph = empty()
        graph.add_vertices(vertices, slice='base')
        graph.add_edges(records, slice='base')
        return graph

    def add_edges_repeated():
        graph = empty()
        graph.add_vertices(vertices, slice='base')
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
        graph, _vertices, _pairs, edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_edges(edge_ids[: scale.remove_edges], errors='raise')
        return graph

    def remove_vertices_fraction():
        graph, vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_vertices(vertices[: scale.remove_vertices], errors='raise')
        return graph

    cases = (
        ('create_empty', empty, note('construct an empty AnnNet graph')),
        ('add_vertices_bulk', add_vertices_bulk, note('bulk vertex insertion')),
        ('add_vertices_repeated', add_vertices_repeated, note('one public call per vertex')),
        ('add_edges_bulk', add_edges_bulk, note('bulk edge insertion with explicit edge ids')),
        ('add_edges_repeated', add_edges_repeated, note('one public call per edge')),
        ('remove_edges_fraction', remove_edges_fraction, note(f'remove {scale.remove_edges} edges')),
        (
            'remove_vertices_fraction',
            remove_vertices_fraction,
            note(f'remove {scale.remove_vertices} vertices'),
        ),
    )
    return [
        time_record('annnet', 'primitives', op, scale, fn, backend, samples, note_)
        for op, fn, note_ in cases
    ]
