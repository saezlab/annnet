import unittest

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from annnet.core.graph import Graph
import polars as pl
from graph_tool import topology, centrality, clustering, flow, generation, search, util


def gt_backend(G):
    return G.gt.backend()


# ---- Simple graph builder ----
def build_small():
    G = Graph()
    G.add_vertex("a")
    G.add_vertex("b")
    G.add_edge("a", "b", weight=3.0)
    return G


class TestLazyGTProxy(unittest.TestCase):

    # --- topology: shortest distance ---

    def test_shortest_distance_basic(self):
        G = build_small()
        d = G.gt.topology.shortest_distance(G, source="a", target="b", weights="weight")
        self.assertAlmostEqual(float(d), 3.0)

    # --- topology: label_components ---

        def test_label_components(self):
            G = Graph()
            G.add_vertex("x")
            G.add_vertex("y")
            G.add_edge("x", "y")

            # directed=False ensures we look for Weakly Connected Components (1 component)
            comp, hist = G.gt.topology.label_components(G, directed=False)
            
            # Check that the histogram has exactly 1 entry (meaning 1 component found)
            self.assertEqual(len(hist), 1)

            vp = comp
            gtg = G.gt.backend()

            # Check both vertices are in the same component
            comp_id_0 = int(vp[gtg.vertex(0)])
            comp_id_1 = int(vp[gtg.vertex(1)])
            self.assertEqual(comp_id_0, comp_id_1)

    # --- centrality: betweenness ---

    def test_betweenness(self):
        G = Graph()
        for v in ["a", "b", "c"]:
            G.add_vertex(v)
        G.add_edge("a", "b")
        G.add_edge("b", "c")

        vc, ec = G.gt.centrality.betweenness(G)
        gtg = G.gt.backend()

        ba = float(vc[gtg.vertex(0)])
        bb = float(vc[gtg.vertex(1)])
        bc = float(vc[gtg.vertex(2)])

        self.assertEqual(ba, 0.0)
        self.assertEqual(bc, 0.0)
        self.assertEqual(bb, 0.5)

    # --- clustering: local clustering coefficients ---

    def test_clustering(self):
        G = Graph()
        for v in ["a", "b", "c"]:
            G.add_vertex(v)
        G.add_edge("a", "b")
        G.add_edge("b", "c")
        G.add_edge("c", "a")         # triangle → clustering = 1.0

        cc = G.gt.clustering.local_clustering(G)
        gtg = G.gt.backend()

        for idx in range(3):
            self.assertAlmostEqual(float(cc[gtg.vertex(idx)]), 1.0)

    # --- flow: max flow ---

    def test_max_flow(self):
        G = Graph()
        G.add_vertex("s")
        G.add_vertex("t")
        eid = G.add_edge("s", "t")
        
        # Add capacity as edge attribute
        G.edge_attributes = pl.DataFrame({
            "edge_id": [eid],
            "capacity": [5.0]
        })

        g = G.gt.backend()
        cap = g.ep["capacity"]

        # push_relabel_max_flow returns residual capacity
        # flow = capacity - residual
        res = G.gt.flow.push_relabel_max_flow(G, "s", "t", cap)
        
        # Calculate actual flow on each edge
        flow_sum = sum(cap[e] - res[e] for e in g.edges())
        self.assertAlmostEqual(flow_sum, 5.0)

    # --- generation: line graph ----

    def test_generation_line_graph(self):
        g = Graph()
        g.add_vertex("a")
        g.add_vertex("b")
        g.add_edge("a", "b")

        gtg = g.gt.backend()
        lg, *_ = g.gt.generation.line_graph(g)

        # line graph of 1 edge has 1 vertex
        self.assertEqual(lg.num_vertices(), 1)

    # --- search: BFS ---

    def test_search_bfs(self):
        G = build_small()
        g = G.gt.backend()
        order = []

        class Visitor(search.BFSVisitor):
            def discover_vertex(self, u):
                order.append(int(u))

        G.gt.search.bfs_search(G, g.vertex(0), Visitor())
        self.assertEqual(order[0], 0)

    # --- util: random_permute_vertices ---

    def test_random_permute_vertices(self):
        G = build_small()
        gtg = G.gt.backend()
        G.gt.generation.random_rewire(G)
        # graph stays isomorphic: still 2 vertices
        self.assertEqual(gtg.num_vertices(), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
