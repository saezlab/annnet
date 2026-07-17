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
    make_edge_attr_updates,
    graphtool_property_type,
    make_numeric_edge_pairs,
    graphtool_property_value,
    make_vertex_attr_updates,
    optional_module_for_engine,
)


def backend_operation_dimensions(
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
    suffix = scale_note(original, scale)
    recs: list[dict] = []
    for engine, ops in (
        ('annnet', _annnet_ops(scale, backend)),
        ('networkx', _networkx_ops(scale)),
        ('igraph', _igraph_ops(scale)),
        ('graph-tool', _graphtool_ops(scale)),
    ):
        optional = optional_module_for_engine(engine)
        for op, fn, note in ops:
            if suffix:
                note = f'{note}; {suffix}'
            recs.append(
                time_record(
                    engine,
                    'backend_operations',
                    op,
                    scale,
                    fn,
                    backend if engine == 'annnet' else None,
                    samples,
                    note,
                    optional_module=optional,
                )
            )
    return recs


def _annnet_ops(scale, backend: str):
    AnnNet = annnet()
    vertices = make_vertices(scale.vertices)
    pairs = make_edge_pairs(scale.vertices, scale.edges)
    records = make_edge_records(pairs)

    def empty():
        return AnnNet(directed=True, annotations_backend=backend)

    def add_vertices_bulk():
        graph = empty()
        graph.add_vertices(vertices, slice='base')
        return graph

    def add_edges_bulk():
        graph = empty()
        graph.add_vertices(vertices, slice='base')
        graph.add_edges(records, slice='base')
        return graph

    def remove_edges_fraction():
        graph, _vertices, _pairs, edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_edges(edge_ids[: scale.remove_edges], errors='raise')
        return graph

    def remove_vertices_fraction():
        graph, vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        graph.remove_vertices(vertices[: scale.remove_vertices], errors='raise')
        return graph

    def set_vertex_attrs_bulk():
        graph = empty()
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

    def set_edge_attrs_bulk():
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

    return _operation_cases(
        add_vertices_bulk,
        add_edges_bulk,
        remove_edges_fraction,
        remove_vertices_fraction,
        set_vertex_attrs_bulk,
        set_edge_attrs_bulk,
    )


def _networkx_ops(scale):
    vertices = make_vertices(scale.vertices)
    pairs = make_edge_pairs(scale.vertices, scale.edges)

    def build_graph():
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(vertices)
        graph.add_edges_from(pairs)
        return graph

    def add_vertices_bulk():
        import networkx as nx

        graph = nx.DiGraph()
        graph.add_nodes_from(vertices)
        return graph

    def add_edges_bulk():
        return build_graph()

    def remove_edges_fraction():
        graph = build_graph()
        graph.remove_edges_from(pairs[: scale.remove_edges])
        return graph

    def remove_vertices_fraction():
        graph = build_graph()
        graph.remove_nodes_from(vertices[: scale.remove_vertices])
        return graph

    def set_vertex_attrs_bulk():
        import networkx as nx

        graph = build_graph()
        nx.set_node_attributes(
            graph,
            make_vertex_attr_updates(
                vertices,
                attr_count=scale.node_attrs,
                sparse_every=scale.sparse_every,
                annotation_density=scale.annotation_density,
            ),
        )
        return graph

    def set_edge_attrs_bulk():
        import networkx as nx

        graph = build_graph()
        updates = dict(
            zip(
                pairs,
                make_edge_attr_updates(
                    [f'e{idx}' for idx in range(len(pairs))],
                    attr_count=scale.edge_attrs,
                    sparse_every=scale.sparse_every,
                    annotation_density=scale.annotation_density,
                ).values(),
                strict=False,
            )
        )
        nx.set_edge_attributes(graph, updates)
        return graph

    return _operation_cases(
        add_vertices_bulk,
        add_edges_bulk,
        remove_edges_fraction,
        remove_vertices_fraction,
        set_vertex_attrs_bulk,
        set_edge_attrs_bulk,
    )


def _igraph_ops(scale):
    vertices = make_vertices(scale.vertices)
    pairs = make_numeric_edge_pairs(scale.vertices, scale.edges)

    def build_graph():
        import igraph as ig

        graph = ig.Graph(directed=True)
        graph.add_vertices(scale.vertices)
        graph.vs['name'] = vertices
        graph.add_edges(pairs)
        return graph

    def add_vertices_bulk():
        import igraph as ig

        graph = ig.Graph(directed=True)
        graph.add_vertices(scale.vertices)
        graph.vs['name'] = vertices
        return graph

    def add_edges_bulk():
        return build_graph()

    def remove_edges_fraction():
        graph = build_graph()
        graph.delete_edges(list(range(min(scale.remove_edges, graph.ecount()))))
        return graph

    def remove_vertices_fraction():
        graph = build_graph()
        graph.delete_vertices(list(range(min(scale.remove_vertices, graph.vcount()))))
        return graph

    def set_vertex_attrs_bulk():
        graph = add_vertices_bulk()
        updates = make_vertex_attr_updates(
            vertices,
            attr_count=scale.node_attrs,
            sparse_every=scale.sparse_every,
            annotation_density=scale.annotation_density,
        )
        for attr_idx in range(scale.node_attrs):
            attr = f'node_attr_{attr_idx}'
            graph.vs[attr] = [updates[vertex_id].get(attr) for vertex_id in vertices]
        return graph

    def set_edge_attrs_bulk():
        graph = build_graph()
        updates = make_edge_attr_updates(
            [f'e{idx}' for idx in range(graph.ecount())],
            attr_count=scale.edge_attrs,
            sparse_every=scale.sparse_every,
            annotation_density=scale.annotation_density,
        )
        for attr_idx in range(scale.edge_attrs):
            attr = f'edge_attr_{attr_idx}'
            graph.es[attr] = [updates[f'e{idx}'].get(attr) for idx in range(graph.ecount())]
        return graph

    return _operation_cases(
        add_vertices_bulk,
        add_edges_bulk,
        remove_edges_fraction,
        remove_vertices_fraction,
        set_vertex_attrs_bulk,
        set_edge_attrs_bulk,
    )


def _graphtool_ops(scale):
    vertices = make_vertices(scale.vertices)
    pairs = make_numeric_edge_pairs(scale.vertices, scale.edges)

    def build_graph():
        import graph_tool.all as gt

        graph = gt.Graph(directed=True)
        graph.add_vertex(scale.vertices)
        graph.add_edge_list(pairs)
        return graph

    def add_vertices_bulk():
        import graph_tool.all as gt

        graph = gt.Graph(directed=True)
        graph.add_vertex(scale.vertices)
        return graph

    def add_edges_bulk():
        return build_graph()

    def remove_edges_fraction():
        graph = build_graph()
        for edge in list(graph.edges())[: scale.remove_edges]:
            graph.remove_edge(edge)
        return graph

    def remove_vertices_fraction():
        graph = build_graph()
        for idx in reversed(range(min(scale.remove_vertices, graph.num_vertices()))):
            graph.remove_vertex(graph.vertex(idx))
        return graph

    def set_vertex_attrs_bulk():
        graph = add_vertices_bulk()
        updates = make_vertex_attr_updates(
            vertices,
            attr_count=scale.node_attrs,
            sparse_every=scale.sparse_every,
            annotation_density=scale.annotation_density,
        )
        for attr_idx in range(scale.node_attrs):
            attr = f'node_attr_{attr_idx}'
            prop = graph.new_vertex_property(graphtool_property_type(attr_idx))
            for vertex in graph.vertices():
                prop[vertex] = graphtool_property_value(
                    updates[f'v{int(vertex)}'].get(attr),
                    attr_idx,
                )
            graph.vp[attr] = prop
        return graph

    def set_edge_attrs_bulk():
        graph = build_graph()
        updates = make_edge_attr_updates(
            [f'e{idx}' for idx in range(graph.num_edges())],
            attr_count=scale.edge_attrs,
            sparse_every=scale.sparse_every,
            annotation_density=scale.annotation_density,
        )
        for attr_idx in range(scale.edge_attrs):
            attr = f'edge_attr_{attr_idx}'
            prop = graph.new_edge_property(graphtool_property_type(attr_idx))
            for idx, edge in enumerate(graph.edges()):
                prop[edge] = graphtool_property_value(
                    updates[f'e{idx}'].get(attr),
                    attr_idx,
                )
            graph.ep[attr] = prop
        return graph

    return _operation_cases(
        add_vertices_bulk,
        add_edges_bulk,
        remove_edges_fraction,
        remove_vertices_fraction,
        set_vertex_attrs_bulk,
        set_edge_attrs_bulk,
    )


def _operation_cases(
    add_vertices_bulk,
    add_edges_bulk,
    remove_edges_fraction,
    remove_vertices_fraction,
    set_vertex_attrs_bulk,
    set_edge_attrs_bulk,
):
    return (
        ('add_vertices_bulk', add_vertices_bulk, 'bulk vertex insertion'),
        ('add_edges_bulk', add_edges_bulk, 'bulk edge insertion'),
        ('remove_edges_fraction', remove_edges_fraction, 'remove edge fraction'),
        ('remove_vertices_fraction', remove_vertices_fraction, 'remove vertex fraction'),
        ('set_vertex_attrs_bulk', set_vertex_attrs_bulk, 'bulk vertex annotation write'),
        ('set_edge_attrs_bulk', set_edge_attrs_bulk, 'bulk edge annotation write'),
    )
