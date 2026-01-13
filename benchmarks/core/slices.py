from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure


def run(scale):
    results = {}
    G = AnnNet(directed=True)

    slice_ids = [f"s{i}" for i in range(scale.slices)]
    for sid in slice_ids:
        G.add_slice(sid)

    with measure() as m_vertex_presence:
        for sid in slice_ids:
            G.add_vertices_bulk(
                (
                    {"vertex_id": f"{sid}_v{i}"}
                    for i in range(scale.vertices // scale.slices)
                ),
                slice=sid,
            )

    vertex_counts = {
        sid: len(list(G.get_slice_vertices(sid)))
        for sid in slice_ids
    }

    results["vertex_presence"] = {
        "metrics": m_vertex_presence,
        "vertex_counts": vertex_counts,
    }

    with measure() as m_edge_presence:
        for sid in slice_ids:
            verts = list(G.get_slice_vertices(sid))
            if not verts:
                continue
            G.set_active_slice(sid)
            G.add_edges_bulk(
                {
                    "source": verts[i % len(verts)],
                    "target": verts[(i + 1) % len(verts)],
                    "weight": 1.0,
                }
                for i in range(scale.edges // scale.slices)
            )

    edge_counts = {
        sid: len(G.get_slice_edges(sid))
        for sid in slice_ids
    }

    results["edge_presence"] = {
        "metrics": m_edge_presence,
        "edge_counts": edge_counts,
    }

    with measure() as m_shared:
        for i in range(50):
            G.add_vertex(
                f"shared_v{i}",
                slice=slice_ids[0],
                propagate="shared",
            )

    with measure() as m_all:
        for i in range(50):
            G.add_vertex(
                f"all_v{i}",
                slice=slice_ids[0],
                propagate="all",
            )

    shared_presence = {
        sid: sum(
            v.startswith("shared_")
            for v in list(G.get_slice_vertices(sid))
        )
        for sid in slice_ids
    }

    all_presence = {
        sid: sum(
            v.startswith("all_")
            for v in list(G.get_slice_vertices(sid))
        )
        for sid in slice_ids
    }

    results["propagation"] = {
        "shared": m_shared,
        "all": m_all,
        "shared_presence": shared_presence,
        "all_presence": all_presence,
    }

    with measure() as m_slice_attrs:
        for sid in slice_ids:
            G.set_slice_attrs(sid, label=sid)

    slice_attrs = {
        sid: G.get_slice_attr(sid, "label")
        for sid in slice_ids
    }

    results["slice_attributes"] = {
        "metrics": m_slice_attrs,
        "values": slice_attrs,
    }

    with measure() as m_union:
        union_result = G.slice_union(slice_ids)
        union_sid = G.create_slice_from_operation("union_slice", union_result)

    with measure() as m_intersection:
        inter_result = G.slice_intersection(slice_ids)
        inter_sid = G.create_slice_from_operation("inter_slice", inter_result)

    with measure() as m_difference:
        if len(slice_ids) >= 2:
            diff_result = G.slice_difference(slice_ids[0], slice_ids[1])
            diff_sid = G.create_slice_from_operation("diff_slice", diff_result)
        else:
            diff_sid = None

    results["slice_ops"] = {
        "union": m_union,
        "intersection": m_intersection,
        "difference": m_difference,
        "union_vertices": len(union_result["vertices"]),
        "intersection_vertices": len(inter_result["vertices"]),
        "difference_vertices": len(diff_result["vertices"]) if diff_sid else 0,
    }

    with measure() as m_derived:
        tmp_sid = "tmp_even"
        G.add_slice(tmp_sid)
        for v in G.vertices():
            if v.endswith("0"):
                G.add_vertex_to_slice(tmp_sid, v)

        derived_result = {"vertices": G.get_slice_vertices(tmp_sid), "edges": set()}
        derived_sid = G.create_slice_from_operation("derived_even", derived_result)

    results["derived_slice"] = {
        "metrics": m_derived,
        "vertex_count": len(G.get_slice_vertices(derived_sid)),
    }

    with measure() as m_aggregated:
        agg_sid = G.create_aggregated_slice(
            source_slice_ids=slice_ids,
            target_slice_id="aggregated",
            method="union",
        )

    results["aggregated_slice"] = {
        "metrics": m_aggregated,
        "vertex_count": len(G.get_slice_vertices(agg_sid)),
    }

    with measure() as m_conserved:
        conserved = G.conserved_edges(min_slices=2)

    slice_local = {
        sid: len(G.slice_specific_edges(sid))
        for sid in slice_ids
    }

    results["edge_scope"] = {
        "conserved": {
            "metrics": m_conserved,
            "count": len(conserved),
        },
        "slice_local": slice_local,
    }

    if G.vertices():
        sample_vertex = next(iter(G.vertices()))
        sample_edge = next(iter(G.edges())) if G.edges() else None

        results["presence_queries"] = {
            "vertex": G.vertex_presence_across_slices(sample_vertex),
            "edge": G.edge_presence_across_slices(sample_edge) if sample_edge else None,
        }
    else:
        results["presence_queries"] = {"vertex": [], "edge": None}

    return results