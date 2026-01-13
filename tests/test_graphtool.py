# tests/test_graphtool_adapter.py
import os
import sys
import unittest
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from annnet.core.graph import AnnNet

try:
    import graph_tool.all as gt

    HAS_GT = True
except Exception:
    HAS_GT = False

from annnet.adapters.graphtool_adapter import from_graphtool, to_graphtool


def _build_graph() -> AnnNet:
    g = AnnNet(directed=True)

    g.add_vertex("A", label="alpha", kind="src")
    g.add_vertex("B", label="beta")
    g.add_vertex("C", label="gamma", kind="sink")

    e1 = g.add_edge("A", "B", weight=2.0, interaction=+1, tag="ab")
    e2 = g.add_edge("B", "C", weight=1.0, edge_directed=False, interaction=-1)
    e3 = g.add_hyperedge(head=["A", "B"], tail=["C"], weight=0.5, interaction=+1)

    g.add_slice("Lw", region="EMEA")
    g.set_edge_slice_attrs("Lw", e1, weight=5.0)

    g.add_slice("L0")

    assert g.number_of_edges() >= 3
    return g


_BUILD_GRAPH = _build_graph


@unittest.skipUnless(HAS_GT, "graph-tool adapter or dependency not available")
class TestGraphToolAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _BUILD_GRAPH is None:
            raise unittest.SkipTest("No _build_graph() found")

    def test_to_gt_export_basic(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIsNotNone(gtG)
        self.assertIsInstance(manifest, dict)

        self.assertIn("version", manifest)
        self.assertIn("graph", manifest)
        self.assertIn("vertices", manifest)
        self.assertIn("edges", manifest)
        self.assertIn("slices", manifest)
        self.assertIn("multilayer", manifest)

        self.assertEqual(gtG.num_vertices(), 3)
        self.assertGreaterEqual(gtG.num_edges(), 2)

    def test_roundtrip_preserves_structure(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        g2 = from_graphtool(gtG, manifest)

        self.assertEqual(g2.number_of_vertices(), g.number_of_vertices())

        self.assertIn("A", g2.vertices())
        self.assertIn("B", g2.vertices())
        self.assertIn("C", g2.vertices())

    def test_manifest_preserves_slices(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        slices_data = manifest.get("slices", {})
        self.assertIn("data", slices_data)

        slice_names = list(slices_data.get("data", {}).keys())
        self.assertIn("Lw", slice_names)
        self.assertIn("L0", slice_names)

    def test_manifest_preserves_hyperedges(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        edges_meta = manifest.get("edges", {})
        hyperedges = edges_meta.get("hyperedges", {})

        self.assertGreater(len(hyperedges), 0)

    def test_roundtrip_preserves_weights(self):
        g = _BUILD_GRAPH()

        e1_eid = None
        for eid in g.edge_weights.keys():
            if g.edge_weights[eid] == 2.0:
                e1_eid = eid
                break

        self.assertIsNotNone(e1_eid)

        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        self.assertIn(e1_eid, g2.edge_weights)
        self.assertEqual(g2.edge_weights[e1_eid], 2.0)

    def test_roundtrip_preserves_slice_weights(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        slices = list(g2.list_slices(include_default=True))
        self.assertIn("Lw", slices)

        edges_in_lw = list(g2.get_slice_edges("Lw"))
        self.assertGreater(len(edges_in_lw), 0)

        eid = edges_in_lw[0]
        w_eff = g2.get_effective_edge_weight(eid, slice="Lw")
        self.assertEqual(w_eff, 5.0)

    def test_vertex_properties_in_graph(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIn("id", gtG.vp)

        vertex_ids = [gtG.vp["id"][v] for v in gtG.vertices()]
        self.assertIn("A", vertex_ids)
        self.assertIn("B", vertex_ids)
        self.assertIn("C", vertex_ids)

    def test_edge_properties_in_graph(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIn("id", gtG.ep)
        self.assertIn("weight", gtG.ep)

        for e in gtG.edges():
            weight = gtG.ep["weight"][e]
            self.assertIsInstance(weight, float)
            self.assertGreater(weight, 0)

    def test_directed_flag_preserved(self):
        g_dir = AnnNet(directed=True)
        g_dir.add_vertex("X")
        g_dir.add_vertex("Y")
        g_dir.add_edge("X", "Y")

        gtG_dir, manifest_dir = to_graphtool(g_dir)
        self.assertTrue(gtG_dir.is_directed())
        self.assertTrue(manifest_dir["graph"]["directed"])

        g_undir = AnnNet(directed=False)
        g_undir.add_vertex("X")
        g_undir.add_vertex("Y")
        g_undir.add_edge("X", "Y")

        gtG_undir, manifest_undir = to_graphtool(g_undir)
        self.assertFalse(gtG_undir.is_directed())
        self.assertFalse(manifest_undir["graph"]["directed"])

    def test_vertex_attributes_roundtrip(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        if hasattr(g2, "vertex_attributes") and g2.vertex_attributes is not None:
            v_attrs = g2.vertex_attributes
            if hasattr(v_attrs, "to_dicts"):
                rows = list(v_attrs.to_dicts())
                vertex_ids = [r.get("vertex_id") for r in rows]
                self.assertIn("A", vertex_ids)

    def test_edge_attributes_roundtrip(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        if hasattr(g2, "edge_attributes") and g2.edge_attributes is not None:
            e_attrs = g2.edge_attributes
            self.assertGreater(len(e_attrs), 0)

    def test_without_manifest_loses_hyperedges(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        g2 = from_graphtool(gtG, manifest=None)

        self.assertEqual(g2.number_of_vertices(), 3)
        self.assertLess(g2.number_of_edges(), g.number_of_edges())


if __name__ == "__main__":
    unittest.main()
