import os
import shutil
import tempfile
import unittest

import polars as pl

from annnet.io.GraphDir_Parquet_io import from_parquet_graphdir, to_parquet_graphdir
from annnet.adapters.igraph_adapter import from_igraph, to_igraph
from annnet.io.json_io import from_json, to_json
from annnet.adapters.networkx_adapter import from_nx, to_nx
from annnet.io.SIF_io import from_sif, to_sif
from annnet.core.graph import AnnNet

try:
    import networkx as nx

    HAS_NX = True
except ImportError:
    HAS_NX = False

try:
    import igraph as ig

    HAS_IG = True
except ImportError:
    HAS_IG = False


class TestMultilayerAdapters(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _build_multilayer_graph(self):
        G = AnnNet()
        # 1. Aspects & Elementary Layers
        G.aspects = ["time", "transport"]
        G.elem_layers = {"time": ["t1", "t2"], "transport": ["bus", "train"]}
        G._rebuild_all_layers_cache()

        # 2. Vertices & VM (mapping nodes to layers)
        G.add_vertex("u")
        G.add_vertex("v")
        # u is present in (t1, bus) and (t2, train)
        G._VM = {
            ("u", ("t1", "bus")),
            ("u", ("t2", "train")),
            ("v", ("t1", "bus")),
        }

        # 3. Edges & Edge Layers
        # e1: u -> v in layer (t1, bus)
        e1 = G.add_edge("u", "v")
        G.edge_layers[e1] = ("t1", "bus")
        G.edge_kind[e1] = "intra"

        # e2: u(t1,bus) -> u(t2,train) (coupling)
        e2 = G.add_edge("u", "u")
        G.edge_layers[e2] = (("t1", "bus"), ("t2", "train"))
        G.edge_kind[e2] = "coupling"

        # 4. Attributes
        # Node-layer attribute
        G._vertex_layer_attrs[("u", ("t1", "bus"))] = {"cost": 10.0}

        # Layer-tuple attribute
        G._layer_attrs[("t1", "bus")] = {"freq": "high"}

        # Layer attribute table (elementary layers)
        G.layer_attributes = pl.DataFrame(
            {
                "layer": ["t1", "t2", "bus", "train"],
                "desc": ["Morning", "Evening", "Bus Line", "Train Line"],
            }
        )

        return G

    def _assert_multilayer_equal(self, G1, G2):
        # Aspects
        self.assertEqual(G1.aspects, G2.aspects)
        self.assertEqual(G1.elem_layers, G2.elem_layers)

        # VM
        self.assertEqual(G1._VM, G2._VM)

        # Edge Layers & Kinds
        # Note: Edge IDs might change in some adapters if not careful, but here we expect them to be preserved or mapped
        # For simplicity, we check if the sets of (u, v, layers, kind) match
        def get_edge_specs(g):
            specs = []
            for eid in g.edge_layers:
                u, v = g.edge_definitions[eid][:2]
                layers = g.edge_layers[eid]
                kind = g.edge_kind.get(eid)
                specs.append((u, v, layers, kind))
            return sorted(specs, key=lambda x: str(x))

        self.assertEqual(get_edge_specs(G1), get_edge_specs(G2))

        # Node-layer attrs
        self.assertEqual(G1._vertex_layer_attrs, G2._vertex_layer_attrs)

        # Layer-tuple attrs
        self.assertEqual(G1._layer_attrs, G2._layer_attrs)

        # Layer attributes table
        # Sort by layer to ensure comparison works
        df1 = G1.layer_attributes.sort("layer")
        df2 = G2.layer_attributes.sort("layer")
        # Polars equality
        self.assertTrue(df1.equals(df2))

    @unittest.skipUnless(HAS_NX, "networkx not installed")
    def test_networkx_roundtrip(self):
        G = self._build_multilayer_graph()
        nxG, manifest = to_nx(G)
        G2 = from_nx(nxG, manifest)
        self._assert_multilayer_equal(G, G2)

    @unittest.skipUnless(HAS_IG, "igraph not installed")
    def test_igraph_roundtrip(self):
        G = self._build_multilayer_graph()
        igG, manifest = to_igraph(G)
        G2 = from_igraph(igG, manifest)
        self._assert_multilayer_equal(G, G2)

    def test_json_roundtrip(self):
        G = self._build_multilayer_graph()
        path = os.path.join(self.test_dir, "graph.json")
        to_json(G, path)
        G2 = from_json(path)
        self._assert_multilayer_equal(G, G2)

    def test_graphdir_roundtrip(self):
        G = self._build_multilayer_graph()
        path = os.path.join(self.test_dir, "graph_dir")
        to_parquet_graphdir(G, path)
        G2 = from_parquet_graphdir(path)
        self._assert_multilayer_equal(G, G2)

    def test_sif_roundtrip(self):
        G = self._build_multilayer_graph()
        path = os.path.join(self.test_dir, "graph.sif")
        # SIF requires lossless=True to generate manifest
        _, manifest = to_sif(G, path, lossless=True)
        # from_sif needs the manifest to restore multilayer attributes
        G2 = from_sif(path, manifest=manifest)
        self._assert_multilayer_equal(G, G2)


if __name__ == "__main__":
    unittest.main()
