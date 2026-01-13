import tempfile

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    G = AnnNet()
    G.add_vertices_bulk(({"vertex_id": f"v{i}"} for i in range(scale.vertices)), slice="base")
    G.add_edges_bulk(
        {
            "source": f"v{i % scale.vertices}",
            "target": f"v{(i * 13) % scale.vertices}",
            "weight": 1.0,
        }
        for i in range(scale.edges)
    )

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/graph.annnet"

        with measure() as m_write:
            G.write(path, overwrite=True)

        with measure() as m_read:
            _ = AnnNet.read(path)

    return {
        "write_annnet": m_write,
        "read_annnet": m_read,
    }
