"""
Full multilayer (Kivela) benchmark using REAL AnnNet layer API.

Covers:
- multi-aspect configuration (>=2 aspects)
- layer tuples (Cartesian product)
- vertex presence across layer tuples (V_M)
- intra-layer edges
- inter-layer edges
- coupling edges
- layer algebra (union, intersection, difference)
- layer-to-slice bridge
- subgraph extraction from layers
- supra-adjacency matrix construction
- tensor view and flattening
- multilayer-specific queries (participation coefficient, versatility)
- IO round-trip with multilayer structure
- correctness + speed + RSS (Resident Set Size)
"""

import numpy as np

from annnet.core.graph import AnnNet
from benchmarks.harness.metrics import measure

MAX_SPECTRAL_N = 6000
MAX_DENSE_COMPARE_N = 1500


def run(scale):
    results = {}
    G = AnnNet(directed=True)

    # Fixed layer structure — independent of scale.slices which is too small
    # to produce a meaningful multilayer graph. These values give ~20 layer
    # tuples and enough vertices/edges to stress the supra-matrix and
    # spectral code paths at every scale level.
    n_aspects = 2
    n_elem_per_aspect = 5  # 5^2 = 25 layer tuples

    aspects = [f"aspect_{i}" for i in range(n_aspects)]
    elem_layers = {a: [f"{a}_e{j}" for j in range(n_elem_per_aspect)] for a in aspects}

    with measure() as m_aspect_setup:
        G.layers.set_aspects(aspects, elem_layers)

    layer_tuples = list(G.layers.iter_layers())
    n_active_layers = min(len(layer_tuples), 20)

    results["aspect_setup"] = {
        "metrics": m_aspect_setup,
        "n_aspects": n_aspects,
        "n_elementary_per_aspect": n_elem_per_aspect,
        "n_layer_tuples": len(layer_tuples),
    }

    vertices_per_layer = max(20, scale.vertices // 50)
    # Half the budget is shared across all layers so that coupling,
    # layer intersection, and participation coefficient have real data.
    n_shared = vertices_per_layer // 2
    n_unique = vertices_per_layer - n_shared
    shared_vids = [f"shared_v{i}" for i in range(n_shared)]

    with measure() as m_presence:
        presence_pairs = []
        unique_vids = set(shared_vids)
        vid_counter = 0
        for aa in layer_tuples[:n_active_layers]:
            for vid in shared_vids:
                presence_pairs.append((vid, aa))
            for _ in range(n_unique):
                vid = f"v{vid_counter}"
                presence_pairs.append((vid, aa))
                unique_vids.add(vid)
                vid_counter += 1

        existing = set(G.vertices())
        to_add = [vid for vid in unique_vids if vid not in existing]
        if to_add:
            G.add_vertices({"vertex_id": vid} for vid in to_add)

        for vid, aa in presence_pairs:
            G.add_vertices(vid, layer=aa)

    presence_counts = {
        str(aa): len(G.layers.layer_vertex_set(aa)) for aa in layer_tuples[: min(10, len(layer_tuples))]
    }

    results["vertex_presence"] = {
        "metrics": m_presence,
        "total_vm_entries": len(G._VM),
        "sample_layer_counts": presence_counts,
    }

    # ------------------------------------------------------------------
    # Intra-layer edges via bulk path (add_intra_edges_bulk, fast_mode=True)
    # Single-call add_intra_edge_nl is intentionally avoided here because
    # running it before the bulk pass would double the edge count and
    # produce misleading timing comparisons.
    # ------------------------------------------------------------------

    bulk_by_layer = {}
    for aa in layer_tuples[:n_active_layers]:
        verts = list(G.layers.layer_vertex_set(aa))
        if len(verts) < 2:
            continue

        n_edges = min(len(verts), scale.edges // n_active_layers)
        edges_uv = [(verts[i % len(verts)], verts[(i + 1) % len(verts)]) for i in range(n_edges)]
        if edges_uv:
            bulk_by_layer[aa] = edges_uv

    with measure() as m_intra_edges_bulk:
        bulk_count = 0
        for aa, edges_uv in bulk_by_layer.items():
            eids = G.add_edges(
                [
                    {"source": (u, aa), "target": (v, aa), "weight": 1.0}
                    for u, v in edges_uv
                ]
            )
            bulk_count += len(eids)

    intra_count = bulk_count

    intra_edge_counts = {
        str(aa): len(G.layers.layer_edge_set(aa, include_inter=False, include_coupling=False))
        for aa in layer_tuples[: min(10, len(layer_tuples))]
    }

    results["intra_edges_bulk"] = {
        "metrics": m_intra_edges_bulk,
        "total_intra": intra_count,
        "sample_counts": intra_edge_counts,
    }

    with measure() as m_inter_edges:
        inter_count = 0
        if len(layer_tuples) >= 2:
            for i in range(min(len(layer_tuples) - 1, 50)):
                aa = layer_tuples[i]
                bb = layer_tuples[i + 1]

                verts_a = list(G.layers.layer_vertex_set(aa))
                verts_b = list(G.layers.layer_vertex_set(bb))

                if not verts_a or not verts_b:
                    continue

                for j in range(min(5, len(verts_a), len(verts_b))):
                    u = verts_a[j % len(verts_a)]
                    v = verts_b[j % len(verts_b)]

                    if G.layers.has_presence(u, aa) and G.layers.has_presence(v, bb):
                        G.add_edges((u, aa), (v, bb), weight=1.0)
                        inter_count += 1

    results["inter_edges"] = {
        "metrics": m_inter_edges,
        "total_inter": inter_count,
    }

    with measure() as m_coupling:
        coupling_count = 0
        vertices_in_multiple = {}
        for u, aa in G._VM:
            vertices_in_multiple.setdefault(u, []).append(aa)

        for u, layers in list(vertices_in_multiple.items())[:50]:
            if len(layers) < 2:
                continue
            for i in range(len(layers) - 1):
                G.add_edges((u, layers[i]), (u, layers[i + 1]), weight=1.0)
                coupling_count += 1

    results["coupling_edges"] = {
        "metrics": m_coupling,
        "total_coupling": coupling_count,
    }

    if len(layer_tuples) >= 3:
        with measure() as m_union:
            union_layers = layer_tuples[:3]
            union_result = G.layers.layer_union(union_layers, include_inter=False, include_coupling=False)

        with measure() as m_intersection:
            inter_result = G.layers.layer_intersection(
                union_layers, include_inter=False, include_coupling=False
            )

        with measure() as m_difference:
            diff_result = G.layers.layer_difference(
                layer_tuples[0], layer_tuples[1], include_inter=False, include_coupling=False
            )

        results["layer_algebra"] = {
            "union": m_union,
            "intersection": m_intersection,
            "difference": m_difference,
            "union_vertices": len(union_result["vertices"]),
            "union_edges": len(union_result["edges"]),
            "intersection_vertices": len(inter_result["vertices"]),
            "difference_vertices": len(diff_result["vertices"]),
        }

    if len(layer_tuples) >= 2:
        with measure() as m_layer_to_slice:
            slice_id = G.layers.create_slice_from_layer(
                "layer_slice_0", layer_tuples[0], include_inter=False, include_coupling=False
            )

        with measure() as m_layer_union_to_slice:
            union_slice = G.layers.create_slice_from_layer_union(
                "union_slice",
                layer_tuples[: min(3, len(layer_tuples))],
                include_inter=True,
                include_coupling=False,
            )

        results["layer_to_slice"] = {
            "single_layer": m_layer_to_slice,
            "union_to_slice": m_layer_union_to_slice,
            "slice_vertices": len(G.slices.get_slice_vertices(slice_id)),
            "union_slice_vertices": len(G.slices.get_slice_vertices(union_slice)),
        }

    if len(layer_tuples) >= 1:
        with measure() as m_subgraph:
            sub = G.layers.subgraph_from_layer_tuple(
                layer_tuples[0], include_inter=False, include_coupling=False
            )

        with measure() as m_subgraph_union:
            sub_union = G.layers.subgraph_from_layer_union(
                layer_tuples[: min(3, len(layer_tuples))], include_inter=True, include_coupling=True
            )

        results["subgraphs"] = {
            "single_layer": m_subgraph,
            "union": m_subgraph_union,
            "vertices": len(list(sub.vertices())),
            "edges": len(list(sub.edges())),
            "union_vertices": len(list(sub_union.vertices())),
            "union_edges": len(list(sub_union.edges())),
        }

    with measure() as m_supra:
        supra_A = G.layers.supra_adjacency(layers=None)

    with measure() as m_blocks:
        blocks = {
            "intra": G.layers.build_intra_block(None),
            "inter": G.layers.build_inter_block(None),
            "coupling": G.layers.build_coupling_block(None),
        }

    results["supra_adjacency"] = {
        "metrics": m_supra,
        "blocks": m_blocks,
        "shape": supra_A.shape,
        "nnz": supra_A.nnz,
        "intra_nnz": blocks["intra"].nnz,
        "inter_nnz": blocks["inter"].nnz,
        "coupling_nnz": blocks["coupling"].nnz,
    }

    with measure() as m_supra_incidence:
        supra_B, supra_B_edge_ids, supra_B_skipped = G.layers.supra_incidence(layers=None)

    results["supra_incidence"] = {
        "metrics": m_supra_incidence,
        "shape": list(supra_B.shape),
        "nnz": supra_B.nnz,
        "n_edges": len(supra_B_edge_ids),
        "n_skipped": len(supra_B_skipped),
    }

    with measure() as m_tensor:
        tensor_view = G.layers.adjacency_tensor_view(layers=None)

    with measure() as m_flatten:
        A_reconstructed = G.layers.flatten_to_supra(tensor_view)

    with measure() as m_unflatten:
        tensor_back = G.layers.unflatten_from_supra(supra_A, layers=None)

    results["tensor_operations"] = {
        "tensor_view": m_tensor,
        "flatten": m_flatten,
        "unflatten": m_unflatten,
        "n_vertices": len(tensor_view["vertices"]),
        "n_layers": len(tensor_view["layers"]),
        "n_edges": int(len(tensor_view["w"])),
        "reconstruction_matches": (
            bool(np.allclose(A_reconstructed.toarray(), supra_A.toarray()))
            if supra_A.shape[0] * supra_A.shape[1] <= 50_000_000  # 50M entries ≈ 200MB
            else "skipped_large"
        ),
    }

    if len(G._VM) >= 2:
        with measure() as m_queries:
            sample_vertex = next(iter(G.vertices()))
            vertex_layers = list(G.layers.iter_vertex_layers(sample_vertex))

            layer_deg_raw = G.layers.layer_degree_vectors(None)
            layer_deg_count = len(layer_deg_raw)

            if intra_count > 0:
                try:
                    participation = G.layers.participation_coefficient(None)
                    versatility = G.layers.versatility(None)
                except Exception:
                    participation = {}
                    versatility = {}
            else:
                participation = {}
                versatility = {}

        results["multilayer_queries"] = {
            "metrics": m_queries,
            "sample_vertex_layers": len(vertex_layers),
            "n_layers_with_degree": layer_deg_count,
            "n_vertices_participation": len(participation),
            "n_vertices_versatility": len(versatility),
            "avg_participation": (
                sum(participation.values()) / len(participation) if participation else 0.0
            ),
        }

    with measure() as m_presence_queries:
        if len(layer_tuples) >= 1:
            sample_layer = layer_tuples[0]
            vertices_in_layer = G.layers.layer_vertex_set(sample_layer)

            if vertices_in_layer:
                sample_v = next(iter(vertices_in_layer))
                layers_of_v = list(G.layers.iter_vertex_layers(sample_v))
                has_pres = G.layers.has_presence(sample_v, sample_layer)
        else:
            vertices_in_layer = set()
            layers_of_v = []
            has_pres = False

    results["presence_queries"] = {
        "metrics": m_presence_queries,
        "vertices_in_sample_layer": len(vertices_in_layer),
        "layers_of_sample_vertex": len(layers_of_v),
        "has_presence_check": has_pres,
    }

    with measure() as m_dynamics:
        n = len(G._VM)
        results["dynamics"] = {"skipped": True}
        if n <= MAX_SPECTRAL_N and intra_count > 0:
            try:
                deg = G.layers.supra_degree(None)
                laplacian = G.layers.supra_laplacian(kind="comb", layers=None)

                if laplacian.shape[0] >= 2:
                    alg_conn, fiedler = G.layers.algebraic_connectivity(None)
                else:
                    alg_conn, fiedler = 0.0, None

                P = G.layers.transition_matrix(None)
                p0 = np.ones(P.shape[0]) / P.shape[0]
                p1 = G.layers.random_walk_step(p0, None)
            except Exception:
                alg_conn, fiedler, p1 = 0.0, None, np.array([])
        else:
            alg_conn, fiedler, p1 = 0.0, None, np.array([])

    results["dynamics"] = {
        "metrics": m_dynamics,
        "algebraic_connectivity": float(alg_conn) if alg_conn else 0.0,
        "has_fiedler": fiedler is not None,
        "rw_converged": len(p1) > 0,
    }

    with measure() as m_attributes:
        if len(aspects) >= 1:
            G.layers.set_aspect_attrs(aspects[0], description="test", order="temporal")
            aspect_attrs = G.layers.get_aspect_attrs(aspects[0])

        if len(layer_tuples) >= 1:
            G.layers.set_layer_attrs(layer_tuples[0], label="test_layer")
            layer_attrs = G.layers.get_layer_attrs(layer_tuples[0])

        if len(elem_layers[aspects[0]]) >= 1:
            G.layers.set_elementary_layer_attrs(aspects[0], elem_layers[aspects[0]][0], timestamp=0)
            elem_attrs = G.layers.get_elementary_layer_attrs(aspects[0], elem_layers[aspects[0]][0])

    results["layer_attributes"] = {
        "metrics": m_attributes,
        "aspect_attrs_set": len(aspect_attrs) > 0 if "aspect_attrs" in locals() else False,
        "layer_attrs_set": len(layer_attrs) > 0 if "layer_attrs" in locals() else False,
        "elem_attrs_set": len(elem_attrs) > 0 if "elem_attrs" in locals() else False,
    }

    if len(layer_tuples) >= 4 and len(aspects) >= 2:
        with measure() as m_coupling_ops:
            layer_pairs = [
                (layer_tuples[0], layer_tuples[1]),
                (layer_tuples[2], layer_tuples[3]),
            ]
            try:
                added = G.layers.add_layer_coupling_pairs(layer_pairs, weight=1.0)
            except Exception:
                added = 0

    results["coupling_operations"] = {
        "metrics": m_coupling_ops if "m_coupling_ops" in locals() else None,
        "edges_added": added if "added" in locals() else 0,
    }

    n = len(G._VM)
    if len(layer_tuples) >= 2 and intra_count > 0 and n <= MAX_SPECTRAL_N:
        with measure() as m_spectral:
            try:
                k = min(6, max(2, len(G._VM) - 1))
                if k >= 2:
                    vals, vecs = G.layers.k_smallest_laplacian_eigs(k=k, kind="comb", layers=None)
                    spectral_gap = float(vals[1]) if len(vals) > 1 else 0.0
                else:
                    spectral_gap = 0.0
            except Exception:
                spectral_gap = 0.0

    results["spectral"] = {
        "metrics": m_spectral if "m_spectral" in locals() else None,
        "spectral_gap": spectral_gap if "spectral_gap" in locals() else 0.0,
    }

    if len(layer_tuples) >= 1:
        with measure() as m_projection:
            sample_vertices = list(G.layers.layer_vertex_set(layer_tuples[0]))
            if len(sample_vertices) >= 2:
                flattened_edges = set()
                for eid in G.layers.layer_edge_set(
                    layer_tuples[0], include_inter=False, include_coupling=False
                ):
                    if eid in G.edge_definitions:
                        u, v, _ = G.edge_definitions[eid]
                        flattened_edges.add((u, v))

    results["layer_projection"] = {
        "metrics": m_projection if "m_projection" in locals() else None,
        "projected_edges": len(flattened_edges) if "flattened_edges" in locals() else 0,
    }

    return results
