from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    G = AnnNet()
    G.add_vertices_bulk(({"vertex_id": f"v{i}"} for i in range(scale.vertices)), slice="base")
    G.add_edges_bulk(
        {
            "source": f"v{i % scale.vertices}",
            "target": f"v{(i + 1) % scale.vertices}",
            "weight": 1.0,
        }
        for i in range(scale.edges)
    )

    with measure() as m_edges_view:
        _ = G.edges_view()

    with measure() as m_neighbors:
        _ = G.neighbors("v0")

    return {
        "edges_view": m_edges_view,
        "neighbors": m_neighbors,
    }
