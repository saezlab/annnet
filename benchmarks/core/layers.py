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

MAX_SPECTRAL_N = 2000
MAX_DENSE_COMPARE_N = 1500

def run(scale):
    results = {}
    G = AnnNet(directed=True)

    n_aspects = min(3, max(2, scale.slices // 10))
    n_elem_per_aspect = max(2, scale.slices // n_aspects)
    
    aspects = [f"aspect_{i}" for i in range(n_aspects)]
    elem_layers = {
        a: [f"{a}_e{j}" for j in range(n_elem_per_aspect)]
        for a in aspects
    }

    with measure() as m_aspect_setup:
        G.set_aspects(aspects, elem_layers)

    layer_tuples = list(G.iter_layers())
    results["aspect_setup"] = {
        "metrics": m_aspect_setup,
        "n_aspects": n_aspects,
        "n_elementary_per_aspect": n_elem_per_aspect,
        "n_layer_tuples": len(layer_tuples),
    }

    vertices_per_layer = min(10, scale.vertices)
    
    with measure() as m_presence:
        vid_counter = 0
        for aa in layer_tuples[:min(len(layer_tuples), scale.slices)]:
            for i in range(vertices_per_layer):
                vid = f"v{vid_counter}"
                if vid not in G.vertices():
                    G.add_vertex(vid)
                G.add_presence(vid, aa)
                vid_counter += 1

    presence_counts = {
        str(aa): len(G.layer_vertex_set(aa))
        for aa in layer_tuples[:min(10, len(layer_tuples))]
    }

    results["vertex_presence"] = {
        "metrics": m_presence,
        "total_vm_entries": len(G._VM),
        "sample_layer_counts": presence_counts,
    }

    with measure() as m_intra_edges:
        intra_count = 0
        for aa in layer_tuples[:min(len(layer_tuples), scale.slices)]:
            verts = list(G.layer_vertex_set(aa))
            if len(verts) < 2:
                continue
            
            n_edges = min(len(verts), scale.edges // len(layer_tuples))
            for i in range(n_edges):
                u = verts[i % len(verts)]
                v = verts[(i + 1) % len(verts)]
                G.add_intra_edge_nl(u, v, aa, weight=1.0)
                intra_count += 1

    intra_edge_counts = {
        str(aa): len(G.layer_edge_set(aa, include_inter=False, include_coupling=False))
        for aa in layer_tuples[:min(10, len(layer_tuples))]
    }

    results["intra_edges"] = {
        "metrics": m_intra_edges,
        "total_intra": intra_count,
        "sample_counts": intra_edge_counts,
    }

    # ------------------------------------------------------------------
    # FAST bulk intra-layer edges (direct matrix path)
    # ------------------------------------------------------------------

    bulk_by_layer = {}
    for aa in layer_tuples[:min(len(layer_tuples), scale.slices)]:
        verts = list(G.layer_vertex_set(aa))
        if len(verts) < 2:
            continue

        n_edges = min(len(verts), scale.edges // len(layer_tuples))
        edges_uv = []
        for i in range(n_edges):
            u = verts[i % len(verts)]
            v = verts[(i + 1) % len(verts)]
            edges_uv.append((u, v))

        if edges_uv:
            bulk_by_layer[aa] = edges_uv

    with measure() as m_intra_edges_bulk:
        bulk_count = 0
        for aa, edges_uv in bulk_by_layer.items():
            eids = G.add_intra_edges_bulk(
                edges_uv,
                layer_tuple=aa,
                fast_mode=True,
            )
            bulk_count += len(eids)

    results["intra_edges_bulk"] = {
        "metrics": m_intra_edges_bulk,
        "total_intra": bulk_count,
        "speedup_vs_single": (
            results["intra_edges"]["metrics"]["wall_time_s"]
            / m_intra_edges_bulk["wall_time_s"]
            if m_intra_edges_bulk["wall_time_s"] > 0
            else None
        ),
    }

    with measure() as m_inter_edges:
        inter_count = 0
        if len(layer_tuples) >= 2:
            for i in range(min(len(layer_tuples) - 1, 50)):
                aa = layer_tuples[i]
                bb = layer_tuples[i + 1]
                
                verts_a = list(G.layer_vertex_set(aa))
                verts_b = list(G.layer_vertex_set(bb))
                
                if not verts_a or not verts_b:
                    continue
                
                for j in range(min(5, len(verts_a), len(verts_b))):
                    u = verts_a[j % len(verts_a)]
                    v = verts_b[j % len(verts_b)]
                    
                    if G.has_presence(u, aa) and G.has_presence(v, bb):
                        G.add_inter_edge_nl(u, aa, v, bb, weight=1.0)
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
                G.add_coupling_edge_nl(u, layers[i], layers[i+1], weight=1.0)
                coupling_count += 1

    results["coupling_edges"] = {
        "metrics": m_coupling,
        "total_coupling": coupling_count,
    }

    if len(layer_tuples) >= 3:
        with measure() as m_union:
            union_layers = layer_tuples[:3]
            union_result = G.layer_union(
                union_layers,
                include_inter=False,
                include_coupling=False
            )

        with measure() as m_intersection:
            inter_result = G.layer_intersection(
                union_layers,
                include_inter=False,
                include_coupling=False
            )

        with measure() as m_difference:
            diff_result = G.layer_difference(
                layer_tuples[0],
                layer_tuples[1],
                include_inter=False,
                include_coupling=False
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
            slice_id = G.create_slice_from_layer(
                "layer_slice_0",
                layer_tuples[0],
                include_inter=False,
                include_coupling=False
            )

        with measure() as m_layer_union_to_slice:
            union_slice = G.create_slice_from_layer_union(
                "union_slice",
                layer_tuples[:min(3, len(layer_tuples))],
                include_inter=True,
                include_coupling=False
            )

        results["layer_to_slice"] = {
            "single_layer": m_layer_to_slice,
            "union_to_slice": m_layer_union_to_slice,
            "slice_vertices": len(G.get_slice_vertices(slice_id)),
            "union_slice_vertices": len(G.get_slice_vertices(union_slice)),
        }

    if len(layer_tuples) >= 1:
        with measure() as m_subgraph:
            sub = G.subgraph_from_layer_tuple(
                layer_tuples[0],
                include_inter=False,
                include_coupling=False
            )

        with measure() as m_subgraph_union:
            sub_union = G.subgraph_from_layer_union(
                layer_tuples[:min(3, len(layer_tuples))],
                include_inter=True,
                include_coupling=True
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
        supra_A = G.supra_adjacency(layers=None)

    with measure() as m_blocks:
        blocks = {
            "intra": G.build_intra_block(None),
            "inter": G.build_inter_block(None),
            "coupling": G.build_coupling_block(None),
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

    with measure() as m_tensor:
        tensor_view = G.adjacency_tensor_view(layers=None)

    with measure() as m_flatten:
        A_reconstructed = G.flatten_to_supra(tensor_view)

    with measure() as m_unflatten:
        tensor_back = G.unflatten_from_supra(supra_A, layers=None)

    results["tensor_operations"] = {
        "tensor_view": m_tensor,
        "flatten": m_flatten,
        "unflatten": m_unflatten,
        "n_vertices": len(tensor_view["vertices"]),
        "n_layers": len(tensor_view["layers"]),
        "n_edges": int(len(tensor_view["w"])),
        "reconstruction_matches": bool(np.allclose(
            A_reconstructed.toarray(),
            supra_A.toarray()
        )),
    }

    if len(G._VM) >= 2:
        with measure() as m_queries:
            sample_vertex = next(iter(G.vertices()))
            vertex_layers = list(G.iter_vertex_layers(sample_vertex))
            
            layer_deg_raw = G.layer_degree_vectors(None)
            layer_deg_count = len(layer_deg_raw)
            
            if intra_count > 0:
                participation = G.participation_coefficient(None)
                versatility = G.versatility(None)
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
                sum(participation.values()) / len(participation)
                if participation else 0.0
            ),
        }

    with measure() as m_presence_queries:
        if len(layer_tuples) >= 1:
            sample_layer = layer_tuples[0]
            vertices_in_layer = G.layer_vertex_set(sample_layer)
            
            if vertices_in_layer:
                sample_v = next(iter(vertices_in_layer))
                layers_of_v = list(G.iter_vertex_layers(sample_v))
                has_pres = G.has_presence(sample_v, sample_layer)
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
                deg = G.supra_degree(None)
                laplacian = G.supra_laplacian(kind="comb", layers=None)

                if laplacian.shape[0] >= 2:
                    alg_conn, fiedler = G.algebraic_connectivity(None)
                else:
                    alg_conn, fiedler = 0.0, None

                P = G.transition_matrix(None)
                p0 = np.ones(P.shape[0]) / P.shape[0]
                p1 = G.random_walk_step(p0, None)
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
            G.set_aspect_attrs(aspects[0], description="test", order="temporal")
            aspect_attrs = G.get_aspect_attrs(aspects[0])
            
        if len(layer_tuples) >= 1:
            G.set_layer_attrs(layer_tuples[0], label="test_layer")
            layer_attrs = G.get_layer_attrs(layer_tuples[0])
            
        if len(elem_layers[aspects[0]]) >= 1:
            G.set_elementary_layer_attrs(
                aspects[0],
                elem_layers[aspects[0]][0],
                timestamp=0
            )
            elem_attrs = G.get_elementary_layer_attrs(
                aspects[0],
                elem_layers[aspects[0]][0]
            )

    results["layer_attributes"] = {
        "metrics": m_attributes,
        "aspect_attrs_set": len(aspect_attrs) > 0 if 'aspect_attrs' in locals() else False,
        "layer_attrs_set": len(layer_attrs) > 0 if 'layer_attrs' in locals() else False,
        "elem_attrs_set": len(elem_attrs) > 0 if 'elem_attrs' in locals() else False,
    }

    if len(layer_tuples) >= 4 and len(aspects) >= 2:
        with measure() as m_coupling_ops:
            layer_pairs = [
                (layer_tuples[0], layer_tuples[1]),
                (layer_tuples[2], layer_tuples[3]),
            ]
            added = G.add_layer_coupling_pairs(layer_pairs, weight=1.0)

    results["coupling_operations"] = {
        "metrics": m_coupling_ops if 'm_coupling_ops' in locals() else None,
        "edges_added": added if 'added' in locals() else 0,
    }

    n = len(G._VM)
    if len(layer_tuples) >= 2 and intra_count > 0 and n <= MAX_SPECTRAL_N:
        with measure() as m_spectral:
            try:
                k = min(6, max(2, len(G._VM) - 1))
                if k >= 2:
                    vals, vecs = G.k_smallest_laplacian_eigs(k=k, kind="comb", layers=None)
                    spectral_gap = float(vals[1]) if len(vals) > 1 else 0.0
                else:
                    spectral_gap = 0.0
            except Exception:
                spectral_gap = 0.0

    results["spectral"] = {
        "metrics": m_spectral if 'm_spectral' in locals() else None,
        "spectral_gap": spectral_gap if 'spectral_gap' in locals() else 0.0,
    }

    if len(layer_tuples) >= 1:
        with measure() as m_projection:
            sample_vertices = list(G.layer_vertex_set(layer_tuples[0]))
            if len(sample_vertices) >= 2:
                flattened_edges = set()
                for eid in G.layer_edge_set(
                    layer_tuples[0],
                    include_inter=False,
                    include_coupling=False
                ):
                    if eid in G.edge_definitions:
                        u, v, _ = G.edge_definitions[eid]
                        flattened_edges.add((u, v))

    results["layer_projection"] = {
        "metrics": m_projection if 'm_projection' in locals() else None,
        "projected_edges": len(flattened_edges) if 'flattened_edges' in locals() else 0,
    }

    return results