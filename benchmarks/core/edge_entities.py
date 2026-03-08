from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    results = {}
    G = AnnNet(directed=True)

    # ------------------------------------------------------------
    # 0) Base graph construction
    # ------------------------------------------------------------
    with measure() as m_vertices:
        G.add_vertices_bulk(
            ({"vertex_id": f"v{i}"} for i in range(scale.vertices)),
            slice="base",
        )

    with measure() as m_edges:
        G.add_edges_bulk(
            {
                "source": f"v{i % scale.vertices}",
                "target": f"v{(i + 1) % scale.vertices}",
                "weight": 1.0,
            }
            for i in range(scale.edges)
        )

    base_edges = list(G.edges())

    results["base_vertices"] = {
        "count": scale.vertices,
        "wall_time_s": m_vertices["wall_time_s"],
        "rss_delta_mb": m_vertices["rss_delta_mb"],
    }

    results["base_edges"] = {
        "count": scale.edges,
        "wall_time_s": m_edges["wall_time_s"],
        "rss_delta_mb": m_edges["rss_delta_mb"],
    }

    # ------------------------------------------------------------
    # 1) Node -> Edge edge creation
    # ------------------------------------------------------------
    edges = [
        {"source": f"v{i % scale.vertices}", "target": eid, "weight": 1.0}
        for i, eid in enumerate(base_edges)
    ]

    with measure() as m_node_edge_create:
        node_edge_ids = G.add_edges_bulk(edges)

    results["node_to_edge_create"] = {
        "count": len(node_edge_ids),
        "wall_time_s": m_node_edge_create["wall_time_s"],
        "rss_delta_mb": m_node_edge_create["rss_delta_mb"],
    }

    # ------------------------------------------------------------
    # 2) Edge reification (bulk)
    # ------------------------------------------------------------
    with measure() as m_reify:
        G.add_edge_entities_bulk(node_edge_ids, slice="edge_entities")

    results["edge_reification_bulk"] = {
        "count": len(node_edge_ids),
        "wall_time_s": m_reify["wall_time_s"],
        "rss_delta_mb": m_reify["rss_delta_mb"],
    }

    # ------------------------------------------------------------
    # 3) Edge -> Edge creation + reification
    # ------------------------------------------------------------
    edge_entities = [
        e for e in node_edge_ids if e in G.entity_types and G.entity_types[e] == "edge"
    ]

    ee_edges = [
        {"source": edge_entities[i], "target": edge_entities[i + 1], "weight": 1.0}
        for i in range(len(edge_entities) - 1)
    ]

    with measure() as m_edge_edge_create:
        ee_ids = G.add_edges_bulk(ee_edges)

    with measure() as m_edge_edge_reify:
        G.add_edge_entities_bulk(ee_ids, slice="edge_edge_entities")

    results["edge_to_edge_create"] = {
        "count": len(ee_ids),
        "wall_time_s": m_edge_edge_create["wall_time_s"],
        "rss_delta_mb": m_edge_edge_create["rss_delta_mb"],
    }

    results["edge_to_edge_reify"] = {
        "count": len(ee_ids),
        "wall_time_s": m_edge_edge_reify["wall_time_s"],
        "rss_delta_mb": m_edge_edge_reify["rss_delta_mb"],
    }

    # ------------------------------------------------------------
    # 4) Attribute mutation on edge-entities
    # ------------------------------------------------------------
    attr_items = [
        (eid, {"weight": float(i % 5), "label": f"type_{i % 3}"}) for i, eid in enumerate(ee_ids)
    ]

    with measure() as m_attr:
        G.add_edge_entities_bulk(attr_items, slice="edge_edge_entities")

    results["edge_entity_attr_update"] = {
        "count": len(attr_items),
        "wall_time_s": m_attr["wall_time_s"],
        "rss_delta_mb": m_attr["rss_delta_mb"],
    }

    # ------------------------------------------------------------
    # Final counters
    # ------------------------------------------------------------
    results["final_counts"] = {
        "vertices": G.number_of_vertices(),
        "edges_total": G.number_of_edges(),
        "edge_entities": sum(1 for t in G.entity_types.values() if t == "edge"),
    }

    return results
