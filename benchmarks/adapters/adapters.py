import os
import sys

sys.path.insert(0, os.path.abspath(".."))
import json
import tempfile

from annnet.adapters.graphtool_adapter import from_graphtool, to_graphtool
from annnet.adapters.igraph_adapter import from_igraph, to_igraph
from annnet.adapters.networkx_adapter import from_nx, to_nx
from annnet.adapters.pyg_adapter import to_pyg
from annnet.core.graph import AnnNet
from annnet.io.cx2_io import from_cx2, to_cx2
from annnet.io.dataframe_io import from_dataframes, to_dataframes
from annnet.io.GraphDir_Parquet_io import from_parquet_graphdir, to_parquet_graphdir
from annnet.io.GraphML_io import from_gexf, from_graphml, to_gexf, to_graphml
from annnet.io.json_io import from_json, to_json
from annnet.io.SBML_io import from_sbml
from annnet.io.SIF_io import from_sif, to_sif
from benchmarks.harness.metrics import measure


def run(scale):
    out = {}
    graph = AnnNet(directed=True)
    
    with measure() as m_vertices:
        graph.add_vertices_bulk(
            ({"vertex_id": f"v{i}"} for i in range(scale.vertices)),
            slice="base",
        )
    
    with measure() as m_edges:
        graph.add_edges_bulk(
            {
                "source": f"v{i % scale.vertices}",
                "target": f"v{(i * 37) % scale.vertices}",
                "weight": float(i % 7),
                "edge_type": "regular",
            }
            for i in range(scale.edges)
        )
    
    out["total_vertices"] = graph.number_of_vertices()
    out["total_edges"] = graph.number_of_edges()    
    out["vertices"] = m_vertices
    out["edges"] = m_edges
    
    # NetworkX adapter
    with measure() as m_nx_export:
        nxG, manifest = to_nx(graph)
    with measure() as m_nx_import:
        G2 = from_nx(nxG, manifest)
    out["nx_export"] = m_nx_export
    out["nx_import"] = m_nx_import
    out["nx_vertices"] = G2.number_of_vertices()
    out["nx_edges"] = G2.number_of_edges()
    
    # JSON adapter
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "graph.json")
        with measure() as m_json_export:
            to_json(graph, path)
        size_bytes = os.path.getsize(path)
        with measure() as m_json_import:
            G3 = from_json(path)
    out["json_export"] = m_json_export
    out["json_import"] = m_json_import
    out["json_size_bytes"] = size_bytes
    out["json_vertices"] = G3.number_of_vertices()
    out["json_edges"] = G3.number_of_edges()
    
    # PyG (PyTorch Geometric) adapter
    with measure() as m_pyg_export:
        data = to_pyg(graph)
    tensor_bytes = sum(
        t.element_size() * t.nelement()
        for t in data.to_dict().values()
        if hasattr(t, "nelement")
    )
    out["pyg_export"] = m_pyg_export
    out["pyg_tensor_bytes"] = tensor_bytes
    
    # CX2 adapter
    with tempfile.TemporaryDirectory() as td:
        sk = to_cx2(graph)
        cx2_path = os.path.join(td, "graph.cx2")
        with measure() as m_cx2_export:
            with open(cx2_path, "w") as f:
                json.dump(sk, f)
        cx2_size = os.path.getsize(cx2_path)
        with measure() as m_cx2_import:
            G4 = from_cx2(cx2_path)
    out["cx2_export"] = m_cx2_export
    out["cx2_import"] = m_cx2_import
    out["cx2_size_bytes"] = cx2_size
    out["cx2_vertices"] = G4.number_of_vertices()
    out["cx2_edges"] = G4.number_of_edges()
    
    # SIF adapter
    with tempfile.TemporaryDirectory() as td:
        sif_path = os.path.join(td, "graph.sif")
        with measure() as m_sif_export:
            to_sif(graph, sif_path)
        sif_size = os.path.getsize(sif_path)
        with measure() as m_sif_import:
            G5 = from_sif(sif_path)
    out["sif_export"] = m_sif_export
    out["sif_import"] = m_sif_import
    out["sif_size_bytes"] = sif_size
    out["sif_vertices"] = G5.number_of_vertices()
    out["sif_edges"] = G5.number_of_edges()
    
    # DataFrame adapter
    with measure() as m_df_export:
        dfs = to_dataframes(graph)
    df_size = 0
    for df in dfs.values():
        if hasattr(df, 'memory_usage'):
            df_size += df.memory_usage(deep=True).sum()
        elif hasattr(df, 'estimated_size'):
            df_size += df.estimated_size()
    with measure() as m_df_import:
        G6 = from_dataframes(
            nodes=dfs.get('nodes'),
            edges=dfs.get('edges'),
            hyperedges=dfs.get('hyperedges'),
            slices=dfs.get('slices'),
            slice_weights=dfs.get('slice_weights')
        )
    out["df_export"] = m_df_export
    out["df_import"] = m_df_import
    out["df_memory_bytes"] = df_size
    out["df_vertices"] = G6.number_of_vertices()
    out["df_edges"] = G6.number_of_edges()
    
    # GraphML adapter
    with tempfile.TemporaryDirectory() as td:
        graphml_path = os.path.join(td, "graph.graphml")
        with measure() as m_graphml_export:
            to_graphml(graph, graphml_path)
        graphml_size = os.path.getsize(graphml_path)
        with measure() as m_graphml_import:
            G7 = from_graphml(graphml_path)
    out["graphml_export"] = m_graphml_export
    out["graphml_import"] = m_graphml_import
    out["graphml_size_bytes"] = graphml_size
    out["graphml_vertices"] = G7.number_of_vertices()
    out["graphml_edges"] = G7.number_of_edges()
    
    # GEXF adapter
    with tempfile.TemporaryDirectory() as td:
        gexf_path = os.path.join(td, "graph.gexf")
        with measure() as m_gexf_export:
            to_gexf(graph, gexf_path)
        gexf_size = os.path.getsize(gexf_path)
        with measure() as m_gexf_import:
            G8 = from_gexf(gexf_path)
    out["gexf_export"] = m_gexf_export
    out["gexf_import"] = m_gexf_import
    out["gexf_size_bytes"] = gexf_size
    out["gexf_vertices"] = G8.number_of_vertices()
    out["gexf_edges"] = G8.number_of_edges()
    
    # Parquet GraphDir adapter
    with tempfile.TemporaryDirectory() as td:
        parquet_dir = os.path.join(td, "graph_parquet")
        with measure() as m_parquet_export:
            to_parquet_graphdir(graph, parquet_dir)
        parquet_size = sum(
            os.path.getsize(os.path.join(parquet_dir, f))
            for f in os.listdir(parquet_dir)
            if os.path.isfile(os.path.join(parquet_dir, f))
        )
        with measure() as m_parquet_import:
            G9 = from_parquet_graphdir(parquet_dir)
    out["parquet_export"] = m_parquet_export
    out["parquet_import"] = m_parquet_import
    out["parquet_size_bytes"] = parquet_size
    out["parquet_vertices"] = G9.number_of_vertices()
    out["parquet_edges"] = G9.number_of_edges()
    
    # graph-tool adapter
    try:
        with measure() as m_gt_export:
            gt_graph, gt_manifest = to_graphtool(graph)
        with measure() as m_gt_import:
            G10 = from_graphtool(gt_graph, gt_manifest)
        out["graphtool_export"] = m_gt_export
        out["graphtool_import"] = m_gt_import
        out["graphtool_vertices"] = G10.number_of_vertices()
        out["graphtool_edges"] = G10.number_of_edges()
    except Exception as e:
        out["graphtool_error"] = str(e)
    
    # igraph adapter
    try:
        with measure() as m_ig_export:
            ig_graph, ig_manifest = to_igraph(graph)
        with measure() as m_ig_import:
            G11 = from_igraph(ig_graph, ig_manifest)
        out["igraph_export"] = m_ig_export
        out["igraph_import"] = m_ig_import
        out["igraph_vertices"] = G11.number_of_vertices()
        out["igraph_edges"] = G11.number_of_edges()
    except Exception as e:
        out["igraph_error"] = str(e)
    
    return out