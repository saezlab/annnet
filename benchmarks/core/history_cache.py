import random

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    G = AnnNet(directed=True)

    # Build base graph
    vertices = [f"v{i}" for i in range(scale.vertices)]
    G.add_vertices_bulk(
        ({"vertex_id": vid} for vid in vertices),
        slice="base",
    )

    edges = []
    for i in range(scale.edges):
        edges.append(
            {
                "source": f"v{i % scale.vertices}",
                "target": f"v{(i * 37) % scale.vertices}",
                "weight": float(i % 7),
                "edge_type": "regular",
            }
        )
    G.add_edges_bulk(edges)

    hyperedges = []
    for _ in range(scale.hyperedges):
        hyperedges.append(
            {
                "members": random.sample(vertices, min(5, len(vertices))),
                "weight": 1.0,
            }
        )
    G.add_hyperedges_bulk(hyperedges)

    # Add slices
    for i in range(3):
        slice_name = f"slice_{i}"
        G.add_slice(slice_name)
        sample_edges = random.sample(list(G.edges()), min(scale.edges // 4, G.number_of_edges()))
        G.add_edges_to_slice_bulk(slice_name, sample_edges)

    # Benchmark operations

    with measure() as m_copy:
        G_copy = G.copy()

    with measure() as m_copy_history:
        G_copy_hist = G.copy(history=True)

    with measure() as m_subgraph_vertices:
        sample_v = random.sample(vertices, scale.vertices // 2)
        G_sub_v = G.subgraph(sample_v)

    with measure() as m_subgraph_edges:
        sample_e = random.sample(list(G.edges()), scale.edges // 2)
        G_sub_e = G.edge_subgraph(sample_e)

    with measure() as m_extract_subgraph:
        G_extract = G.extract_subgraph(vertices=sample_v, edges=sample_e)

    with measure() as m_reverse:
        G_rev = G.reverse()

    with measure() as m_subgraph_from_slice:
        G_slice = G.subgraph_from_slice("slice_0", resolve_slice_weights=True)

    with measure() as m_hash:
        h = hash(G)

    with measure() as m_memory_usage:
        mem = G.memory_usage()

    with measure() as m_vertex_incidence_sparse:
        M_sparse = G.vertex_incidence_matrix(values=True, sparse=True)

    _dense_entries = G._matrix.shape[0] * G._matrix.shape[1]
    _dense_limit = 500_000_000  # 500M float32 entries ≈ 2GB
    if _dense_entries <= _dense_limit:
        with measure() as m_vertex_incidence_dense:
            M_dense = G.vertex_incidence_matrix(values=True, sparse=False)
    else:
        m_vertex_incidence_dense = {
            "skipped": f"matrix too large ({_dense_entries:,} entries > {_dense_limit:,} limit)"
        }

    with measure() as m_vertex_incidence_lists:
        inc_lists = G.get_vertex_incidence_matrix_as_lists(values=False)

    # ------------------------------------------------------------------
    # Snapshot + diff
    # ------------------------------------------------------------------
    with measure() as m_snapshot:
        G.snapshot("before_mutation")
        G.add_vertices_bulk(
            ({"vertex_id": f"snap_v{i}"} for i in range(500)),
            slice="base",
        )
        G.snapshot("after_mutation")

    with measure() as m_diff:
        d = G.diff("before_mutation", "after_mutation")

    # ------------------------------------------------------------------
    # Deletions (bulk paths)
    # ------------------------------------------------------------------
    edges_all = list(G.edges())
    edges_to_remove = edges_all[: len(edges_all) // 10]  # drop 10%

    with measure() as m_remove_edges_bulk:
        G._remove_edges_bulk(edges_to_remove)

    with measure() as m_remove_orphans:
        G.remove_orphans()

    verts_to_remove = [v for v in list(G.vertices()) if v.startswith("snap_v")]

    with measure() as m_remove_vertices_bulk:
        G._remove_vertices_bulk(verts_to_remove)

    return {
        "copy": m_copy,
        "copy_with_history": m_copy_history,
        "subgraph_vertices": m_subgraph_vertices,
        "subgraph_edges": m_subgraph_edges,
        "extract_subgraph": m_extract_subgraph,
        "reverse": m_reverse,
        "subgraph_from_slice": m_subgraph_from_slice,
        "hash": m_hash,
        "memory_usage": m_memory_usage,
        "vertex_incidence_sparse": m_vertex_incidence_sparse,
        "vertex_incidence_dense": m_vertex_incidence_dense,
        "vertex_incidence_lists": m_vertex_incidence_lists,
        "snapshot": m_snapshot,
        "diff": {
            "metrics": m_diff,
            "added_vertices": len(d.vertices_added),
            "removed_vertices": len(d.vertices_removed),
            "added_edges": len(d.edges_added),
            "removed_edges": len(d.edges_removed),
        },
        "remove_edges_bulk": {
            "metrics": m_remove_edges_bulk,
            "count": len(edges_to_remove),
        },
        "remove_orphans": m_remove_orphans,
        "remove_vertices_bulk": {
            "metrics": m_remove_vertices_bulk,
            "count": len(verts_to_remove),
        },
        "total_vertices": G.number_of_vertices(),
        "total_edges": G.number_of_edges(),
        "subgraph_v_vertices": G_sub_v.number_of_vertices(),
        "subgraph_v_edges": G_sub_v.number_of_edges(),
        "subgraph_e_vertices": G_sub_e.number_of_vertices(),
        "subgraph_e_edges": G_sub_e.number_of_edges(),
        "memory_bytes": mem,
        "graph_hash": h,
    }
