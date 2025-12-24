import unittest

import pandas as pd
import polars as pl
import pyarrow as pa

from annnet.adapters.dataframe_adapter import from_dataframes, to_dataframes


class TestDataFrameAdapter(unittest.TestCase):
    """Test suite for Narwhals-based dataframe adapter."""

    @classmethod
    def setUpClass(cls):
        """Set up test data once for all tests."""
        cls.nodes_data = {
            "vertex_id": ["user_001", "user_002", "user_003", "user_004", "user_005"],
            "username": ["alice", "bob", "charlie", "diana", "eve"],
            "verified": [True, False, True, False, True],
            "followers_count": [15000, 230, 89000, 1200, 45000],
        }

        cls.edges_data = {
            "source": ["user_001", "user_002", "user_003", "user_001", "user_005", "user_004"],
            "target": ["user_002", "user_003", "user_001", "user_005", "user_003", "user_001"],
            "weight": [1.0, 1.0, 1.0, 0.8, 0.5, 1.0],
            "directed": [True, True, True, True, True, True],
            "edge_type": ["follow", "follow", "follow", "close_friend", "follow", "follow"],
            "created_at": [
                "2023-01-15",
                "2023-02-20",
                "2023-01-10",
                "2024-06-01",
                "2024-03-15",
                "2023-11-30",
            ],
        }

        cls.hyperedges_exploded_data = {
            "edge_id": [
                "group_tech",
                "group_tech",
                "group_tech",
                "group_books",
                "group_books",
                "group_books",
            ],
            "vertex_id": ["user_001", "user_003", "user_005", "user_002", "user_004", "user_005"],
            "role": ["member", "member", "member", "member", "member", "member"],
            "weight": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "directed": [False, False, False, False, False, False],
        }

    def _assert_graph_structure(
        self,
        G,
        expected_vertices: int,
        expected_edges: int | None = None,
        expected_hyperedges: int | None = None,
    ):
        """Helper to verify graph structure."""
        self.assertEqual(len(list(G.vertices())), expected_vertices)
        if expected_edges is not None:
            binary_edges = sum(
                1 for eid, (_, _, etype) in G.edge_definitions.items() if etype != "hyper"
            )
            self.assertEqual(binary_edges, expected_edges)
        if expected_hyperedges is not None:
            self.assertEqual(len(G.hyperedge_definitions), expected_hyperedges)

    # -------------------------------------------------------------------------
    # Backend input tests
    # -------------------------------------------------------------------------

    def test_pandas_input(self):
        """Pandas DataFrames should be accepted as input."""
        nodes = pd.DataFrame(self.nodes_data)
        edges = pd.DataFrame(self.edges_data)
        hyperedges = pd.DataFrame(self.hyperedges_exploded_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            directed=True,
            exploded_hyperedges=True,
        )

        self._assert_graph_structure(
            G, expected_vertices=5, expected_edges=6, expected_hyperedges=2
        )

    def test_polars_input(self):
        """Polars DataFrames should be accepted as input."""
        nodes = pl.DataFrame(self.nodes_data)
        edges = pl.DataFrame(self.edges_data)
        hyperedges = pl.DataFrame(self.hyperedges_exploded_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            directed=True,
            exploded_hyperedges=True,
        )

        self._assert_graph_structure(
            G, expected_vertices=5, expected_edges=6, expected_hyperedges=2
        )

    def test_pyarrow_input(self):
        """PyArrow Tables should be accepted as input."""
        nodes = pa.table(self.nodes_data)
        edges = pa.table(self.edges_data)
        hyperedges = pa.table(self.hyperedges_exploded_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            directed=True,
            exploded_hyperedges=True,
        )

        self._assert_graph_structure(
            G, expected_vertices=5, expected_edges=6, expected_hyperedges=2
        )

    def test_mixed_backend_inputs(self):
        """Mixed DataFrame types should work together."""
        nodes = pd.DataFrame(self.nodes_data)
        edges = pl.DataFrame(self.edges_data)
        hyperedges = pa.table(self.hyperedges_exploded_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            directed=True,
            exploded_hyperedges=True,
        )

        self._assert_graph_structure(
            G, expected_vertices=5, expected_edges=6, expected_hyperedges=2
        )

    # -------------------------------------------------------------------------
    # Round-trip tests
    # -------------------------------------------------------------------------

    def test_roundtrip_polars(self):
        """Import -> export -> import should preserve graph structure."""
        nodes = pl.DataFrame(self.nodes_data)
        edges = pl.DataFrame(self.edges_data)
        hyperedges = pl.DataFrame(self.hyperedges_exploded_data)

        G1 = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            directed=True,
            exploded_hyperedges=True,
        )

        exported = to_dataframes(G1, explode_hyperedges=True)

        G2 = from_dataframes(
            nodes=exported["nodes"],
            edges=exported["edges"],
            hyperedges=exported["hyperedges"],
            exploded_hyperedges=True,
        )

        self.assertEqual(len(list(G1.vertices())), len(list(G2.vertices())))
        self.assertEqual(len(G1.hyperedge_definitions), len(G2.hyperedge_definitions))

    def test_roundtrip_pandas_to_polars(self):
        """Pandas input -> Polars export -> Polars reimport."""
        nodes = pd.DataFrame(self.nodes_data)
        edges = pd.DataFrame(self.edges_data)

        G1 = from_dataframes(nodes=nodes, edges=edges, directed=True)
        exported = to_dataframes(G1)
        G2 = from_dataframes(nodes=exported["nodes"], edges=exported["edges"])

        self.assertEqual(len(list(G1.vertices())), len(list(G2.vertices())))

    # -------------------------------------------------------------------------
    # Hyperedge format tests
    # -------------------------------------------------------------------------

    def test_compact_hyperedges(self):
        """Compact hyperedge format (list columns) should work."""
        hyperedges_compact = pl.DataFrame(
            {
                "edge_id": ["group_tech", "group_books"],
                "directed": [False, False],
                "weight": [1.0, 1.0],
                "head": [None, None],
                "tail": [None, None],
                "members": [
                    ["user_001", "user_003", "user_005"],
                    ["user_002", "user_004", "user_005"],
                ],
            }
        )

        nodes = pl.DataFrame(self.nodes_data)
        edges = pl.DataFrame(self.edges_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges_compact,
            directed=True,
            exploded_hyperedges=False,
        )

        self._assert_graph_structure(G, expected_vertices=5, expected_hyperedges=2)

    def test_export_compact_hyperedges(self):
        """Export should produce valid compact hyperedge format."""
        nodes = pl.DataFrame(self.nodes_data)
        edges = pl.DataFrame(self.edges_data)
        hyperedges = pl.DataFrame(self.hyperedges_exploded_data)

        G = from_dataframes(
            nodes=nodes,
            edges=edges,
            hyperedges=hyperedges,
            exploded_hyperedges=True,
        )

        exported = to_dataframes(G, explode_hyperedges=False)

        self.assertIn("members", exported["hyperedges"].columns)
        self.assertIn("head", exported["hyperedges"].columns)
        self.assertIn("tail", exported["hyperedges"].columns)

    def test_export_exploded_hyperedges(self):
        """Export should produce valid exploded hyperedge format."""
        nodes = pl.DataFrame(self.nodes_data)
        hyperedges = pl.DataFrame(self.hyperedges_exploded_data)

        G = from_dataframes(nodes=nodes, hyperedges=hyperedges, exploded_hyperedges=True)

        exported = to_dataframes(G, explode_hyperedges=True)

        self.assertIn("vertex_id", exported["hyperedges"].columns)
        self.assertIn("role", exported["hyperedges"].columns)
        self.assertEqual(exported["hyperedges"].height, 6)  # 3 + 3 members

    # -------------------------------------------------------------------------
    # Attribute tests
    # -------------------------------------------------------------------------

    def test_vertex_attributes_preserved(self):
        """Vertex attributes should survive round-trip."""
        nodes = pd.DataFrame(self.nodes_data)

        G = from_dataframes(nodes=nodes)
        exported = to_dataframes(G)

        self.assertIn("username", exported["nodes"].columns)
        self.assertIn("verified", exported["nodes"].columns)
        self.assertIn("followers_count", exported["nodes"].columns)

    def test_edge_attributes_preserved(self):
        """Custom edge attributes should survive round-trip."""
        edges = pd.DataFrame(
            {
                "source": ["user_001", "user_002"],
                "target": ["user_002", "user_003"],
                "weight": [0.9, 0.7],
                "interaction_type": ["dm", "reply"],
                "sentiment_score": [0.85, -0.2],
            }
        )

        nodes = pd.DataFrame(self.nodes_data)
        G = from_dataframes(nodes=nodes, edges=edges, directed=True)
        exported = to_dataframes(G)

        self.assertIn("interaction_type", exported["edges"].columns)
        self.assertIn("sentiment_score", exported["edges"].columns)

    def test_public_only_filter(self):
        """public_only=True should filter out __prefixed attributes."""
        nodes = pl.DataFrame(
            {
                "vertex_id": ["a", "b"],
                "name": ["Alice", "Bob"],
                "__internal_id": [123, 456],
            }
        )

        G = from_dataframes(nodes=nodes)
        exported = to_dataframes(G, public_only=True)

        self.assertIn("name", exported["nodes"].columns)
        self.assertNotIn("__internal_id", exported["nodes"].columns)

    # -------------------------------------------------------------------------
    # Edge case tests
    # -------------------------------------------------------------------------

    def test_empty_dataframes(self):
        """Empty DataFrames should not crash."""
        empty_nodes = pd.DataFrame({"vertex_id": []})
        empty_edges = pd.DataFrame({"source": [], "target": []})

        G = from_dataframes(nodes=empty_nodes, edges=empty_edges)

        self.assertEqual(len(list(G.vertices())), 0)
        self.assertEqual(len(G.edge_definitions), 0)

    def test_none_inputs(self):
        """None inputs should be handled gracefully."""
        G = from_dataframes(nodes=None, edges=None)

        self.assertEqual(len(list(G.vertices())), 0)

    def test_nodes_only(self):
        """AnnNet with only nodes, no edges."""
        nodes = pd.DataFrame(self.nodes_data)

        G = from_dataframes(nodes=nodes)

        self._assert_graph_structure(G, expected_vertices=5, expected_edges=0)

    def test_edges_only(self):
        """AnnNet with edges but no explicit nodes (vertices created implicitly)."""
        edges = pd.DataFrame(
            {
                "source": ["a", "b"],
                "target": ["b", "c"],
            }
        )

        G = from_dataframes(edges=edges)

        vertices = list(G.vertices())
        self.assertGreaterEqual(len(vertices), 2)

    # -------------------------------------------------------------------------
    # Validation tests
    # -------------------------------------------------------------------------

    def test_missing_vertex_id_column_raises(self):
        """Nodes DataFrame without vertex_id should raise ValueError."""
        bad_nodes = pd.DataFrame({"name": ["alice", "bob"]})

        with self.assertRaises(ValueError) as ctx:
            from_dataframes(nodes=bad_nodes)

        self.assertIn("vertex_id", str(ctx.exception))

    def test_missing_source_target_columns_raises(self):
        """Edges DataFrame without source/target should raise ValueError."""
        bad_edges = pd.DataFrame({"from": ["a"], "to": ["b"]})

        with self.assertRaises(ValueError) as ctx:
            from_dataframes(edges=bad_edges)

        self.assertIn("source", str(ctx.exception))

    def test_missing_edge_id_in_hyperedges_raises(self):
        """Hyperedges DataFrame without edge_id should raise ValueError."""
        bad_hyperedges = pd.DataFrame({"vertex_id": ["a", "b"], "role": ["member", "member"]})

        with self.assertRaises(ValueError) as ctx:
            from_dataframes(hyperedges=bad_hyperedges, exploded_hyperedges=True)

        self.assertIn("edge_id", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Export options tests
    # -------------------------------------------------------------------------

    def test_exclude_hyperedges(self):
        """include_hyperedges=False should omit hyperedges table."""
        nodes = pl.DataFrame(self.nodes_data)
        hyperedges = pl.DataFrame(self.hyperedges_exploded_data)

        G = from_dataframes(nodes=nodes, hyperedges=hyperedges, exploded_hyperedges=True)
        exported = to_dataframes(G, include_hyperedges=False)

        self.assertNotIn("hyperedges", exported)

    def test_exclude_slices(self):
        """include_slices=False should omit slice tables."""
        nodes = pl.DataFrame(self.nodes_data)

        G = from_dataframes(nodes=nodes)
        exported = to_dataframes(G, include_slices=False)

        self.assertNotIn("slices", exported)
        self.assertNotIn("slice_weights", exported)


if __name__ == "__main__":
    unittest.main()
