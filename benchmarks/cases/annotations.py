from __future__ import annotations

from .common import (
    annnet,
    scale_note,
    time_record,
    capped_scale,
    make_vertices,
    build_annnet_graph,
    make_edge_attr_updates,
    make_vertex_attr_updates,
)


def annotation_update_dimensions(
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
    suffix = scale_note(original, scale)

    def note(text: str) -> str:
        return f'{text}; {suffix}' if suffix else text

    def set_vertex_attrs_bulk_initial():
        graph = AnnNet(directed=True, annotations_backend=backend)
        graph.add_vertices(vertices, slice='base')
        graph.attrs.set_vertex_attrs_bulk(
            make_vertex_attr_updates(
                vertices,
                attr_count=scale.node_attrs,
                sparse_every=scale.sparse_every,
                annotation_density=scale.annotation_density,
            )
        )
        return graph

    def set_vertex_attrs_bulk_update():
        graph, _vertices, _pairs, _edge_ids = build_annnet_graph(
            scale, backend=backend, vertex_attrs=True
        )
        graph.attrs.set_vertex_attrs_bulk(
            make_vertex_attr_updates(
                vertices,
                attr_count=scale.node_attrs,
                sparse_every=scale.sparse_every,
                annotation_density=scale.annotation_density,
                prefix='node_attr_update',
            )
        )
        return graph

    def set_edge_attrs_bulk_initial():
        graph, _vertices, _pairs, edge_ids = build_annnet_graph(scale, backend=backend)
        graph.attrs.set_edge_attrs_bulk(
            make_edge_attr_updates(
                edge_ids,
                attr_count=scale.edge_attrs,
                sparse_every=scale.sparse_every,
                annotation_density=scale.annotation_density,
            )
        )
        return graph

    def set_edge_attrs_bulk_update():
        graph, _vertices, _pairs, edge_ids = build_annnet_graph(
            scale, backend=backend, edge_attrs=True
        )
        graph.attrs.set_edge_attrs_bulk(
            make_edge_attr_updates(
                edge_ids,
                attr_count=scale.edge_attrs,
                sparse_every=scale.sparse_every,
                annotation_density=scale.annotation_density,
                prefix='edge_attr_update',
            )
        )
        return graph

    def set_slice_attrs_repeated():
        graph, _vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        for idx in range(scale.slices):
            graph.attrs.set_slice_attrs(
                f'slice_{idx}',
                label=f'condition_{idx}',
                replicate=idx,
                score=float(idx),
            )
        return graph

    def set_edge_slice_attrs_bulk():
        graph, _vertices, _pairs, edge_ids = build_annnet_graph(scale, backend=backend)
        updates = {
            edge_id: {
                'confidence': float(idx % 7) / 7.0,
                'condition': f'c{idx % max(1, scale.slices)}',
            }
            for idx, edge_id in enumerate(edge_ids)
        }
        graph.attrs.set_edge_slice_attrs_bulk('base', updates)
        return graph

    cases = (
        (
            'set_vertex_attrs_bulk_initial',
            set_vertex_attrs_bulk_initial,
            note(f'{scale.node_attrs} generated vertex attributes'),
        ),
        (
            'set_vertex_attrs_bulk_update',
            set_vertex_attrs_bulk_update,
            note('update existing vertex annotation rows'),
        ),
        (
            'set_edge_attrs_bulk_initial',
            set_edge_attrs_bulk_initial,
            note(f'{scale.edge_attrs} generated edge attributes'),
        ),
        (
            'set_edge_attrs_bulk_update',
            set_edge_attrs_bulk_update,
            note('update existing edge annotation rows'),
        ),
        (
            'set_slice_attrs_repeated',
            set_slice_attrs_repeated,
            note(f'write attributes on {scale.slices} slices'),
        ),
        (
            'set_edge_slice_attrs_bulk',
            set_edge_slice_attrs_bulk,
            note('bulk edge-slice annotation rows for the base slice'),
        ),
    )
    return [
        time_record('annnet', 'annotation_updates', op, scale, fn, backend, samples, note_)
        for op, fn, note_ in cases
    ]
