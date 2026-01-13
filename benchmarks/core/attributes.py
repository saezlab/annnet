"""
Full attribute benchmark.

Compares:
- per-entity attribute mutation (slow path)
- bulk attribute mutation (new fast path)

Measures:
- wall time
- RSS delta
- rows/sec
"""

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    results = {}
    G = AnnNet(directed=True)

    # ------------------------------------------------------------------
    # Setup graph
    # ------------------------------------------------------------------
    G.add_vertices_bulk(
        ({"vertex_id": f"v{i}"} for i in range(scale.vertices)),
        slice="base",
    )

    G.add_edges_bulk(
        {
            "source": f"v{i % scale.vertices}",
            "target": f"v{(i + 1) % scale.vertices}",
            "weight": 1.0,
        }
        for i in range(scale.edges)
    )

    # ------------------------------------------------------------------
    # Baseline: per-vertex attribute mutation (SLOW)
    # ------------------------------------------------------------------
    with measure() as m_vertex_single:
        for i in range(scale.vertices):
            G.set_vertex_attrs(
                f"v{i}",
                kind="gene",
                score=i % 7,
            )

    results["vertex_attrs_single"] = {
        "metrics": m_vertex_single,
        "rows": scale.vertices,
        "rows_per_sec": scale.vertices / m_vertex_single["wall_time_s"],
    }

    # ------------------------------------------------------------------
    # Bulk vertex attributes (FASTER)
    # ------------------------------------------------------------------
    items = [
        (f"v{i}", {"kind": "gene", "score": i % 7})
        for i in range(scale.vertices)
    ]

    with measure() as m_vertex_bulk:
        G.set_vertex_attrs_bulk(items)

    results["vertex_attrs_bulk"] = {
        "metrics": m_vertex_bulk,
        "rows": scale.vertices,
        "rows_per_sec": scale.vertices / m_vertex_bulk["wall_time_s"],
    }

    # ------------------------------------------------------------------
    # Bulk edge attributes
    # ------------------------------------------------------------------
    edge_items = [
        (eid, {"weight": float(i % 5), "etype": "regular"})
        for i, eid in enumerate(G.edges())
    ]

    with measure() as m_edge_bulk:
        G.set_edge_attrs_bulk(edge_items)

    results["edge_attrs_bulk"] = {
        "metrics": m_edge_bulk,
        "rows": len(edge_items),
        "rows_per_sec": len(edge_items) / m_edge_bulk["wall_time_s"],
    }

    # ------------------------------------------------------------------
    # Slice-level edge attributes (already existed)
    # ------------------------------------------------------------------
    slice_id = "base"

    slice_items = [
        (eid, {"confidence": float(i % 3)})
        for i, eid in enumerate(G.edges())
    ]

    with measure() as m_edge_slice_bulk:
        G.set_edge_slice_attrs_bulk(slice_id, slice_items)

    results["edge_slice_attrs_bulk"] = {
        "metrics": m_edge_slice_bulk,
        "rows": len(slice_items),
        "rows_per_sec": len(slice_items) / m_edge_slice_bulk["wall_time_s"],
    }

    # ------------------------------------------------------------------
    # Bulk reads (control)
    # ------------------------------------------------------------------
    with measure() as m_read_vertices:
        _ = G.get_attr_vertices()

    with measure() as m_read_edges:
        _ = G.get_attr_edges()

    results["bulk_reads"] = {
        "vertices": m_read_vertices,
        "edges": m_read_edges,
    }

    return results
