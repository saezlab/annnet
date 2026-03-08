from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    G = AnnNet()
    vertices = [f"v{i}" for i in range(scale.vertices)]
    G.add_vertices_bulk(({"vertex_id": v} for v in vertices), slice="base")
    G.add_edges_bulk(
        {
            "source": f"v{i % scale.vertices}",
            "target": f"v{(i + 1) % scale.vertices}",
            "weight": 1.0,
        }
        for i in range(scale.edges)
    )

    # ------------------------------------------------------------------
    # DataFrame views (on the full graph)
    # ------------------------------------------------------------------
    with measure() as m_edges_view:
        _ = G.edges_view()

    with measure() as m_vertices_view:
        _ = G.vertices_view()

    # ------------------------------------------------------------------
    # GraphView API: create a view, then edges_df / vertices_df / materialize
    # ------------------------------------------------------------------
    sample_v = vertices[: scale.vertices // 2]

    with measure() as m_view_create:
        gv = G.view(vertices=sample_v)

    with measure() as m_edges_df:
        _ = gv.edges_df()

    with measure() as m_vertices_df:
        _ = gv.vertices_df()

    with measure() as m_materialize:
        _ = gv.materialize()

    # ------------------------------------------------------------------
    # Point queries — no bulk alternative, always single-call in user code
    # ------------------------------------------------------------------
    with measure() as m_has_vertex:
        for v in vertices:
            G.has_vertex(v)

    with measure() as m_has_edge:
        for i in range(scale.vertices):
            G.has_edge(f"v{i}", f"v{(i + 1) % scale.vertices}")

    with measure() as m_degree:
        for v in vertices:
            G.degree(v)

    with measure() as m_in_edges:
        _ = list(G.in_edges(vertices))

    with measure() as m_out_edges:
        _ = list(G.out_edges(vertices))

    with measure() as m_neighbors:
        _ = G.neighbors("v0")

    return {
        "edges_view": m_edges_view,
        "vertices_view": m_vertices_view,
        "view_create": m_view_create,
        "edges_df": m_edges_df,
        "vertices_df": m_vertices_df,
        "materialize": m_materialize,
        "has_vertex": {
            "metrics": m_has_vertex,
            "n": scale.vertices,
            "per_call_us": m_has_vertex["wall_time_s"] / scale.vertices * 1e6,
        },
        "has_edge": {
            "metrics": m_has_edge,
            "n": scale.vertices,
            "per_call_us": m_has_edge["wall_time_s"] / scale.vertices * 1e6,
        },
        "degree": {
            "metrics": m_degree,
            "n": scale.vertices,
            "per_call_us": m_degree["wall_time_s"] / scale.vertices * 1e6,
        },
        "in_edges": m_in_edges,
        "out_edges": m_out_edges,
        "neighbors": m_neighbors,
    }
