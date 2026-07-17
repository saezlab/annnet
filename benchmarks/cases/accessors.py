from __future__ import annotations

from .common import scale_note, time_record, capped_scale, build_annnet_graph


def accessor_dimensions(
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
    graph, _vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
    repeats = max(1, scale.accessor_repeats)
    suffix = scale_note(original, scale)

    def note(text: str) -> str:
        return f'{text}; {suffix}' if suffix else text

    def nx_cold_number_of_nodes():
        graph.nx.clear()
        return graph.nx.number_of_nodes(graph)

    def nx_warm_number_of_nodes():
        graph.nx.clear()
        graph.nx.number_of_nodes(graph)
        return sum(graph.nx.number_of_nodes(graph) for _ in range(repeats))

    def nx_reconvert_number_of_nodes():
        import networkx as nx

        from annnet.adapters.networkx_adapter import to_nx

        total = 0
        for _ in range(repeats):
            nx_graph, _manifest = to_nx(
                graph, directed=True, hyperedge_mode='skip', public_only=True
            )
            total += nx.number_of_nodes(nx_graph)
        return total

    def nx_roundtrip_number_of_nodes():
        import networkx as nx

        from annnet.adapters.networkx_adapter import to_nx, from_nx

        total = 0
        for _ in range(repeats):
            nx_graph, manifest = to_nx(
                graph, directed=True, hyperedge_mode='skip', public_only=True
            )
            total += nx.number_of_nodes(nx_graph)
            from_nx(nx_graph, manifest)
        return total

    def nx_after_mutation_number_of_nodes():
        fresh, _vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        fresh.nx.clear()
        before = fresh.nx.number_of_nodes(fresh)
        fresh.add_vertices('cache_invalidation_vertex')
        return before, fresh.nx.number_of_nodes(fresh)

    def ig_cold_vcount():
        graph.ig.clear()
        return graph.ig.vcount()

    def ig_warm_vcount():
        graph.ig.clear()
        graph.ig.vcount()
        return sum(graph.ig.vcount() for _ in range(repeats))

    def ig_reconvert_vcount():
        from annnet.adapters.igraph_adapter import to_igraph

        total = 0
        for _ in range(repeats):
            ig_graph, _manifest = to_igraph(
                graph, directed=True, hyperedge_mode='skip', public_only=True
            )
            total += ig_graph.vcount()
        return total

    def ig_roundtrip_vcount():
        from annnet.adapters.igraph_adapter import to_igraph, from_igraph

        total = 0
        for _ in range(repeats):
            ig_graph, manifest = to_igraph(
                graph, directed=True, hyperedge_mode='skip', public_only=True
            )
            total += ig_graph.vcount()
            from_igraph(ig_graph, manifest)
        return total

    def ig_after_mutation_vcount():
        fresh, _vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        fresh.ig.clear()
        before = fresh.ig.vcount()
        fresh.add_vertices('cache_invalidation_vertex')
        return before, fresh.ig.vcount()

    def gt_cold_components():
        graph.gt.clear()
        _labels, hist = graph.gt.topology.label_components()
        return len(hist)

    def gt_warm_components():
        graph.gt.clear()
        graph.gt.topology.label_components()
        total = 0
        for _ in range(repeats):
            _labels, hist = graph.gt.topology.label_components()
            total += len(hist)
        return total

    def gt_reconvert_components():
        import graph_tool.topology as gt_topology

        from annnet.adapters.graphtool_adapter import to_graphtool

        total = 0
        for _ in range(repeats):
            gt_graph, _manifest = to_graphtool(graph)
            _labels, hist = gt_topology.label_components(gt_graph)
            total += len(hist)
        return total

    def gt_roundtrip_components():
        import graph_tool.topology as gt_topology

        from annnet.adapters.graphtool_adapter import to_graphtool, from_graphtool

        total = 0
        for _ in range(repeats):
            gt_graph, manifest = to_graphtool(graph)
            _labels, hist = gt_topology.label_components(gt_graph)
            total += len(hist)
            from_graphtool(gt_graph, manifest)
        return total

    def gt_after_mutation_components():
        fresh, _vertices, _pairs, _edge_ids = build_annnet_graph(scale, backend=backend)
        fresh.gt.clear()
        _labels, before_hist = fresh.gt.topology.label_components()
        fresh.add_vertices('cache_invalidation_vertex')
        _labels, after_hist = fresh.gt.topology.label_components()
        return len(before_hist), len(after_hist)

    cases = (
        (
            'networkx',
            'nx_accessor_cold_number_of_nodes',
            nx_cold_number_of_nodes,
            note('cold accessor: clear cache, lazily convert to NetworkX once, then read'),
            'networkx',
        ),
        (
            'networkx',
            'nx_accessor_warm_repeated_number_of_nodes',
            nx_warm_number_of_nodes,
            note(
                f'warm accessor: convert once, then {repeats} reads from the cached NetworkX graph'
            ),
            'networkx',
        ),
        (
            'networkx',
            'nx_explicit_reconvert_repeated_number_of_nodes',
            nx_reconvert_number_of_nodes,
            note(
                f'explicit reconvert: bypass the accessor cache and rebuild NetworkX {repeats} times'
            ),
            'networkx',
        ),
        (
            'networkx',
            'nx_explicit_roundtrip_repeated_number_of_nodes',
            nx_roundtrip_number_of_nodes,
            note(f'explicit round-trip: convert to NetworkX and rebuild AnnNet {repeats} times'),
            'networkx',
        ),
        (
            'networkx',
            'nx_accessor_after_mutation_number_of_nodes',
            nx_after_mutation_number_of_nodes,
            note(
                'after mutation: prime accessor cache, mutate AnnNet, then refresh the cached view'
            ),
            'networkx',
        ),
        (
            'igraph',
            'ig_accessor_cold_vcount',
            ig_cold_vcount,
            note('cold accessor: clear cache, lazily convert to igraph once, then read'),
            'igraph',
        ),
        (
            'igraph',
            'ig_accessor_warm_repeated_vcount',
            ig_warm_vcount,
            note(f'warm accessor: convert once, then {repeats} reads from the cached igraph graph'),
            'igraph',
        ),
        (
            'igraph',
            'ig_explicit_reconvert_repeated_vcount',
            ig_reconvert_vcount,
            note(
                f'explicit reconvert: bypass the accessor cache and rebuild igraph {repeats} times'
            ),
            'igraph',
        ),
        (
            'igraph',
            'ig_explicit_roundtrip_repeated_vcount',
            ig_roundtrip_vcount,
            note(f'explicit round-trip: convert to igraph and rebuild AnnNet {repeats} times'),
            'igraph',
        ),
        (
            'igraph',
            'ig_accessor_after_mutation_vcount',
            ig_after_mutation_vcount,
            note(
                'after mutation: prime accessor cache, mutate AnnNet, then refresh the cached view'
            ),
            'igraph',
        ),
        (
            'graph-tool',
            'gt_accessor_cold_components',
            gt_cold_components,
            note('cold accessor: clear cache, lazily convert to graph-tool once, then read'),
            'graph_tool',
        ),
        (
            'graph-tool',
            'gt_accessor_warm_repeated_components',
            gt_warm_components,
            note(
                f'warm accessor: convert once, then {repeats} reads from the cached graph-tool graph'
            ),
            'graph_tool',
        ),
        (
            'graph-tool',
            'gt_explicit_reconvert_repeated_components',
            gt_reconvert_components,
            note(
                f'explicit reconvert: bypass the accessor cache and rebuild graph-tool {repeats} times'
            ),
            'graph_tool',
        ),
        (
            'graph-tool',
            'gt_explicit_roundtrip_repeated_components',
            gt_roundtrip_components,
            note(f'explicit round-trip: convert to graph-tool and rebuild AnnNet {repeats} times'),
            'graph_tool',
        ),
        (
            'graph-tool',
            'gt_accessor_after_mutation_components',
            gt_after_mutation_components,
            note(
                'after mutation: prime accessor cache, mutate AnnNet, then refresh the cached view'
            ),
            'graph_tool',
        ),
    )
    return [
        time_record(
            engine, 'accessors', op, scale, fn, backend, samples, note_, optional_module=mod
        )
        for engine, op, fn, note_, mod in cases
    ]
