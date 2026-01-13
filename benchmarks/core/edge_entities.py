import tracemalloc

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def _measure_mem(fn):
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    fn()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = after.compare_to(before, "filename")
    peak_bytes = sum(stat.size_diff for stat in stats)
    return peak_bytes


def run(scale):
    results = {}
    G = AnnNet(directed=True)

    # ------------------------------------------------------------
    # 0) Base graph construction (baseline cost)
    # ------------------------------------------------------------
    with measure() as m_vertices:
        mem_vertices = _measure_mem(
            lambda: G.add_vertices_bulk(
                ({"vertex_id": f"v{i}"} for i in range(scale.vertices)),
                slice="base",
            )
        )

    with measure() as m_edges:
        mem_edges = _measure_mem(
            lambda: G.add_edges_bulk(
                {
                    "source": f"v{i % scale.vertices}",
                    "target": f"v{(i + 1) % scale.vertices}",
                    "weight": 1.0,
                }
                for i in range(scale.edges)
            )
        )

    base_edges = list(G.edges())

    results["base_vertices"] = {
        "count": scale.vertices,
        "wall_time_s": m_vertices["wall_time_s"],
        "bytes": mem_vertices,
        "bytes_per_item": mem_vertices / max(1, scale.vertices),
    }

    results["base_edges"] = {
        "count": scale.edges,
        "wall_time_s": m_edges["wall_time_s"],
        "bytes": mem_edges,
        "bytes_per_item": mem_edges / max(1, scale.edges),
    }

    # ------------------------------------------------------------
    # 1) Node -> Edge edge creation (no reification)
    # ------------------------------------------------------------
    node_edge_ids = []

    def _node_edge_create():
        for i, eid in enumerate(base_edges):
            u = f"v{i % scale.vertices}"
            node_edge_ids.append(
                G.add_edge(u, eid, weight=1.0, as_entity=False)
            )

    with measure() as m_node_edge_create:
        mem_node_edge_create = _measure_mem(_node_edge_create)

    results["node_to_edge_create"] = {
        "count": len(node_edge_ids),
        "wall_time_s": m_node_edge_create["wall_time_s"],
        "bytes": mem_node_edge_create,
        "bytes_per_item": mem_node_edge_create / max(1, len(node_edge_ids)),
    }

    # ------------------------------------------------------------
    # 2) Edge reification (bulk)
    # ------------------------------------------------------------
    with measure() as m_reify:
        mem_reify = _measure_mem(
            lambda: G.add_edge_entities_bulk(node_edge_ids, slice="edge_entities")
        )

    results["edge_reification_bulk"] = {
        "count": len(node_edge_ids),
        "wall_time_s": m_reify["wall_time_s"],
        "bytes": mem_reify,
        "bytes_per_item": mem_reify / max(1, len(node_edge_ids)),
    }

    # ------------------------------------------------------------
    # 3) Edge -> Edge creation + reification
    # ------------------------------------------------------------
    edge_entities = [
        e for e in node_edge_ids
        if e in G.entity_types and G.entity_types[e] == "edge"
    ]

    ee_ids = []

    def _edge_edge_create():
        for i in range(len(edge_entities) - 1):
            ee_ids.append(
                G.add_edge(
                    edge_entities[i],
                    edge_entities[i + 1],
                    weight=1.0,
                    as_entity=False,
                )
            )

    with measure() as m_edge_edge_create:
        mem_edge_edge_create = _measure_mem(_edge_edge_create)

    with measure() as m_edge_edge_reify:
        mem_edge_edge_reify = _measure_mem(
            lambda: G.add_edge_entities_bulk(ee_ids, slice="edge_edge_entities")
        )

    results["edge_to_edge_create"] = {
        "count": len(ee_ids),
        "wall_time_s": m_edge_edge_create["wall_time_s"],
        "bytes": mem_edge_edge_create,
        "bytes_per_item": mem_edge_edge_create / max(1, len(ee_ids)),
    }

    results["edge_to_edge_reify"] = {
        "count": len(ee_ids),
        "wall_time_s": m_edge_edge_reify["wall_time_s"],
        "bytes": mem_edge_edge_reify,
        "bytes_per_item": mem_edge_edge_reify / max(1, len(ee_ids)),
    }

    # ------------------------------------------------------------
    # 4) Attribute mutation on edge-entities
    # ------------------------------------------------------------
    attr_items = [
        (
            eid,
            {
                "weight": float(i % 5),
                "label": f"type_{i % 3}",
            },
        )
        for i, eid in enumerate(ee_ids)
    ]

    with measure() as m_attr:
        mem_attr = _measure_mem(
            lambda: G.add_edge_entities_bulk(attr_items, slice="edge_edge_entities")
        )

    results["edge_entity_attr_update"] = {
        "count": len(attr_items),
        "wall_time_s": m_attr["wall_time_s"],
        "bytes": mem_attr,
        "bytes_per_item": mem_attr / max(1, len(attr_items)),
    }

    # ------------------------------------------------------------
    # Final counters
    # ------------------------------------------------------------
    results["final_counts"] = {
        "vertices": G.number_of_vertices(),
        "edges_total": G.number_of_edges(),
        "edge_entities": sum(
            1 for t in G.entity_types.values() if t == "edge"
        ),
    }

    return results
