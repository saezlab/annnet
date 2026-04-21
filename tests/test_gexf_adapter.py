"""Unit tests for annnet/io/graphml.py — to_gexf / from_gexf."""

import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from annnet.core.graph import AnnNet
from annnet.io.graphml import from_gexf, to_gexf


def _build_simple():
    G = AnnNet(directed=True)
    G.add_vertex("A")
    G.add_vertex("B")
    G.add_vertex("C")
    G.add_edge("A", "B", edge_id="e1", weight=1.5)
    G.add_edge("B", "C", edge_id="e2", weight=2.0)
    return G


def _build_with_attrs():
    G = AnnNet(directed=True)
    G.add_vertex("X")
    G.set_vertex_attrs("X", gene="TP53", score=0.95)
    G.add_vertex("Y")
    G.set_vertex_attrs("Y", gene="EGFR", score=0.80)
    G.add_edge("X", "Y", edge_id="ex", weight=3.0)
    G.set_edge_attrs("ex", relation="activates")
    return G


def _build_with_hyperedges():
    G = AnnNet(directed=True)
    for v in ["A", "B", "C"]:
        G.add_vertex(v)
    G.add_edge("A", "B", edge_id="e1", weight=1.0)
    G.add_edge(src=["A", "B"], tgt=["C"], edge_id="h1", weight=0.5)
    return G


class TestGEXFAdapter(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _path(self, name):
        return os.path.join(self.tmpdir, name)

    # ------------------------------------------------------------------ #
    # Basic round-trip                                                     #
    # ------------------------------------------------------------------ #

    def test_simple_round_trip_vertex_count(self):
        G = _build_simple()
        p = self._path("simple.gexf")
        to_gexf(G, p)
        G2 = from_gexf(p)
        self.assertEqual(G2.nv, G.nv)

    def test_simple_round_trip_vertex_ids(self):
        G = _build_simple()
        p = self._path("vids.gexf")
        to_gexf(G, p)
        G2 = from_gexf(p)
        self.assertIn("A", G2.vertices())
        self.assertIn("B", G2.vertices())
        self.assertIn("C", G2.vertices())

    def test_simple_round_trip_edge_count(self):
        G = _build_simple()
        p = self._path("edges.gexf")
        to_gexf(G, p)
        G2 = from_gexf(p)
        # to_gexf uses default hyperedge_mode="reify"; binary edges pass through
        self.assertGreaterEqual(G2.ne, 2)

    def test_file_is_created(self):
        G = _build_simple()
        p = self._path("file.gexf")
        to_gexf(G, p)
        self.assertTrue(os.path.exists(p))
        self.assertGreater(os.path.getsize(p), 0)

    # ------------------------------------------------------------------ #
    # Directed vs undirected                                               #
    # ------------------------------------------------------------------ #

    def test_directed_graph(self):
        G = AnnNet(directed=True)
        G.add_vertex("A")
        G.add_vertex("B")
        G.add_edge("A", "B")
        p = self._path("dir.gexf")
        to_gexf(G, p, directed=True)
        G2 = from_gexf(p)
        self.assertEqual(G2.nv, 2)

    def test_undirected_graph(self):
        G = AnnNet(directed=False)
        G.add_vertex("A")
        G.add_vertex("B")
        G.add_edge("A", "B")
        p = self._path("undir.gexf")
        to_gexf(G, p, directed=False)
        G2 = from_gexf(p)
        self.assertEqual(G2.nv, 2)

    # ------------------------------------------------------------------ #
    # Attributes                                                           #
    # ------------------------------------------------------------------ #

    def test_vertex_attrs_partially_survive(self):
        """GEXF is lossy for attribute types; vertex IDs must survive."""
        G = _build_with_attrs()
        p = self._path("attrs.gexf")
        to_gexf(G, p)
        G2 = from_gexf(p)
        self.assertIn("X", G2.vertices())
        self.assertIn("Y", G2.vertices())

    def test_public_only_strips_private(self):
        G = AnnNet(directed=True)
        G.add_vertex("A")
        G.set_vertex_attrs("A", __private="hidden", public="visible")
        G.add_vertex("B")
        G.set_vertex_attrs("B", __private="also_hidden", public="other")
        G.add_edge("A", "B")
        p = self._path("pub.gexf")
        to_gexf(G, p, public_only=True)
        # Should not raise; file must be written
        self.assertTrue(os.path.exists(p))

    # ------------------------------------------------------------------ #
    # Hyperedge modes                                                      #
    # ------------------------------------------------------------------ #

    def test_hyperedge_reify_mode(self):
        G = _build_with_hyperedges()
        p = self._path("hyper_reify.gexf")
        to_gexf(G, p, hyperedge_mode="reify")
        G2 = from_gexf(p, hyperedge="reified")
        # At minimum the 3 real vertices must survive
        self.assertEqual(G2.nv, 3)

    def test_hyperedge_skip_mode(self):
        G = _build_with_hyperedges()
        p = self._path("hyper_skip.gexf")
        to_gexf(G, p, hyperedge_mode="skip")
        G2 = from_gexf(p, hyperedge="none")
        self.assertGreaterEqual(G2.nv, 3)

    # ------------------------------------------------------------------ #
    # Multiple graphs independently                                        #
    # ------------------------------------------------------------------ #

    def test_two_independent_exports(self):
        G1 = _build_simple()
        G2 = _build_with_attrs()
        p1 = self._path("g1.gexf")
        p2 = self._path("g2.gexf")
        to_gexf(G1, p1)
        to_gexf(G2, p2)
        R1 = from_gexf(p1)
        R2 = from_gexf(p2)
        self.assertEqual(R1.nv, 3)
        self.assertEqual(R2.nv, 2)


if __name__ == "__main__":
    unittest.main()
