import random

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    G = AnnNet(directed=True)

    with measure() as m_vertices:
        G.add_vertices(
            ({"vertex_id": f"v{i}"} for i in range(scale.vertices)),
            slice="base",
        )

    with measure() as m_edges:
        G.add_edges(
            {
                "source": f"v{i % scale.vertices}",
                "target": f"v{(i * 37) % scale.vertices}",
                "weight": float(i % 7),
                "edge_type": "regular",
            }
            for i in range(scale.edges)
        )

    _all_verts = [f"v{i}" for i in range(scale.vertices)]
    with measure() as m_hypero:
        G.add_edges(
            {
                "members": random.sample(_all_verts, 3),
                "weight": 1.0,
            }
            for _ in range(scale.hyperedges)
        )

    return {
        "vertices": m_vertices,
        "edges": m_edges,
        "hyperedges": m_hypero,
        "total_vertices": G.number_of_vertices(),
        "total_edges": G.number_of_edges(),
    }
