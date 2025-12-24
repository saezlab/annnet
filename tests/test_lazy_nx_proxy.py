import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import networkx as nx

from annnet.core.graph import AnnNet

# Helpers


def build_small():
    G = AnnNet()
    G.add_vertex("a")
    G.add_vertex("b")
    G.add_vertex("c")
    G.add_edge("a", "b", weight=3.0)
    G.add_edge("b", "c", weight=2.0)
    return G


def build_parallel():
    G = AnnNet()
    G.add_vertex("x")
    G.add_vertex("y")
    G.add_edge("x", "y", weight=10.0)
    G.add_edge("x", "y", weight=1.0)
    return G


def build_directed():
    G = AnnNet(directed=True)
    G.add_vertex("s")
    G.add_vertex("t")
    G.add_edge("s", "t", capacity=5.0)
    return G


# TEST SUITE


class TestLazyNXProxy(unittest.TestCase):
    # ---- backend basics ----

    def test_backend_basic(self):
        G = build_small()
        nxG = G.nx.backend()
        self.assertIsInstance(nxG, nx.DiGraph)
        self.assertEqual(nxG.number_of_nodes(), 3)
        self.assertEqual(nxG.number_of_edges(), 2)

    # ---- simple shortest path ----

    def test_shortest_path_length_labels(self):
        G = build_small()
        d = G.nx.shortest_path_length(G, source="a", target="c", weight="weight")
        self.assertAlmostEqual(d, 5.0)

    # ---- integer vertex ID coercion ----

    def test_shortest_path_int_ids(self):
        G = build_small()
        a_id = G.idx.entity_to_row("a")
        c_id = G.idx.entity_to_row("c")
        d = G.nx.shortest_path_length(G, source=a_id, target=c_id, weight="weight")
        self.assertAlmostEqual(d, 5.0)

    # ---- error for bad vertex label ----

    def test_bad_vertex_label(self):
        G = build_small()
        with self.assertRaises(nx.NodeNotFound):
            G.nx.shortest_path_length(G, source="ZZZ", target="a")

    # ---- caching correctness ----

    def test_backend_cache_reuse(self):
        G = build_small()
        nxG1 = G.nx.backend()
        nxG2 = G.nx.backend()
        self.assertIs(nxG1, nxG2)  # same object

    def test_backend_cache_invalidate_on_version(self):
        G = build_small()
        nxG1 = G.nx.backend()
        # mutate graph -> _version increments
        G.add_edge("a", "c")
        nxG2 = G.nx.backend()
        self.assertIsNot(nxG1, nxG2)

    # ---- simple-mode collapse of parallel edges ----

    def test_simple_graph_edge_collapse(self):
        G = build_parallel()

        nxG = G.nx.backend(simple=True, edge_aggs={"weight": "min"})
        self.assertEqual(nxG.number_of_edges(), 1)

        # weight should be min of [10, 1]
        w = list(nxG.edges(data=True))[0][2].get("weight")
        self.assertEqual(w, 1.0)

    # ---- default aggregation for min(weight) ----

    def test_simple_graph_default_minimum_weight(self):
        G = build_parallel()
        nxG = G.nx.backend(simple=True)
        w = list(nxG.edges(data=True))[0][2].get("weight")
        self.assertEqual(w, 1.0)

    # ---- capacity for flow algorithms ----

    def test_max_flow_capacity(self):
        G = build_directed()
        fvalue, _ = G.nx.maximum_flow(G, "s", "t", capacity="capacity", _nx_simple=True)
        self.assertEqual(fvalue, 5.0)

    # ---- needed edge attrs slimming ----

    def test_needed_attrs_slimming(self):
        G = build_small()

        # shortest path needs "weight"
        nxG = G.nx.backend(needed_attrs={"weight"})
        for _, _, d in nxG.edges(data=True):
            self.assertIn("weight", d)
            self.assertEqual(len(d), 1)

    def test_needed_attrs_drop_weight(self):
        G = build_small()

        nxG = G.nx.backend(needed_attrs=set())
        for _, _, d in nxG.edges(data=True):
            self.assertEqual(len(d), 0)  # fully stripped

    # ---- verify dynamic dispatch resolves deep NX APIs ----

    def test_louvain_communities_dispatch(self):
        G = build_small()
        # Add a small extra edge
        G.add_edge("c", "a")
        comms = list(G.nx.louvain_communities(G, weight="weight"))
        self.assertGreaterEqual(len(comms), 1)

    def test_degree_centrality(self):
        G = build_small()
        dc = G.nx.degree_centrality(G)
        self.assertIn(G.idx.entity_to_row("a"), dc)
        self.assertAlmostEqual(dc[G.idx.entity_to_row("b")], 1.0)

    # ---- coercion of list/tuple/set of vertices ----

    def test_vertex_iterable_coercion(self):
        G = build_small()
        res = G.nx.single_source_dijkstra_path_length(G, source="a", weight="weight")
        self.assertEqual(res[G.idx.entity_to_row("c")], 5.0)

        res2 = G.nx.shortest_path(G, source="a", target="c", weight="weight")
        self.assertTrue(
            res2
            in (
                [G.idx.entity_to_row("a"), G.idx.entity_to_row("b"), G.idx.entity_to_row("c")],
                ["a", "b", "c"],
            )
        )

    # ---- warnings on lossy conversions ----

    def test_hyperedge_warning(self):
        G = AnnNet()
        G.add_vertex("a")
        G.add_vertex("b")
        G.add_hyperedge(members=["a", "b"])  # hyperedge

        with self.assertWarns(RuntimeWarning):
            G.nx.backend(hyperedge_mode="skip")

    # ---- slice flattening warning ----

    def test_slice_flatten_warning(self):
        G = AnnNet()
        G.add_vertex("a", slice="s1")
        G.add_vertex("b", slice="s2")
        G.add_edge("a", "b")

        with self.assertWarns(RuntimeWarning):
            G.nx.backend()

    # ---- verify peek_vertices gives valid vert IDs ----

    def test_peek_vertices(self):
        G = build_small()
        out = G.nx.peek_vertices(2)
        self.assertEqual(len(out), 2)
        nxG = G.nx.backend()
        for v in out:
            self.assertIn(v, nxG)

    # ---- ensure private _nx_* args do not leak ----

    def test_no_leak_of_private_args(self):
        G = build_small()
        d = G.nx.shortest_path_length(
            G,
            source="a",
            target="c",
            weight="weight",
            _nx_simple=True,
            _nx_directed=True,
        )
        self.assertAlmostEqual(d, 5.0)  # still correct

class TestLazyNXProxyEdgeCases:
    """Challenge the NX proxy with non-trivial scientific edge cases."""

    def test_pagerank_on_reified_hyperedges(self):
        """Verify centrality algorithms handle reified hyperedges accurately."""
        G = AnnNet()
        # A 4-way complex represented as a hyperedge
        G.add_hyperedge(members=["A", "B", "C", "D"], edge_id="complex_1", weight=2.0)
        
        # 'reify' converts the hyperedge to a node; 'skip' ignores it
        # Algorithms like PageRank should see the reified node as a hub
        pr = G.nx.pagerank(G, hyperedge_mode="reify")
        
        assert "complex_1" in pr, "Reified hyperedge missing from algorithm results"
        # The reified node should have a higher score than the members
        assert pr["complex_1"] > pr["A"]

    def test_betweenness_centrality_with_mixed_directions(self):
        """Verify centrality on a graph with conflicting local/global directedness."""
        G = AnnNet(directed=False)  # Global: Undirected
        G.add_vertices(["A", "B", "C"])
        # Path: A -> B (Directed) - C (Undirected)
        G.add_edge("A", "B", edge_id="e1", edge_directed=True)
        G.add_edge("B", "C", edge_id="e2", edge_directed=False)
        
        # If directed=True is passed to proxy, B-C becomes B<->C
        bc = G.nx.betweenness_centrality(G, directed=True)
        
        # In a directed A->B<->C path, 'B' is the only path from A to C
        assert bc["B"] > 0
        assert bc["A"] == 0
        assert bc["C"] == 0

    def test_weight_precision_loss_in_shortest_path(self):
        """Ensure high-precision weights (e.g. log-probabilities) are not truncated."""
        G = AnnNet()
        # Small difference in weights that matters for Dijkstra
        G.add_edge("A", "B", weight=1.000000000001, edge_id="e1")
        G.add_edge("A", "C", weight=1.000000000002, edge_id="e2")
        G.add_edge("C", "B", weight=0.000000000001, edge_id="e3")
        
        # Shortest path A -> B
        # Path A-B: 1.000000000001
        # Path A-C-B: 1.000000000003
        path = G.nx.shortest_path(G, source="A", target="B", weight="weight")
        
        assert path == ["A", "B"], "Precision loss led to incorrect shortest path selection"

    def test_empty_slice_algorithm_safety(self):
        """Verify algorithm behavior on slices with nodes but no edges."""
        G = AnnNet()
        G.add_vertices(["A", "B", "C"])
        G.add_slice("empty_slice")
        G.set_active_slice("empty_slice")
        
        # Running degree centrality on 0 edges
        deg = G.nx.degree_centrality(G)
        assert all(val == 0 for val in deg.values())

if __name__ == "__main__":
    unittest.main(verbosity=2)
