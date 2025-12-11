# tests/test_networkx_adapter.py
import os
import sys
import unittest
import polars as pl
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from annnet.core.graph import Graph

from annnet.adapters.cx2_adapter import from_cx2, to_cx2


class TestCX2Adapter(unittest.TestCase):

    def setUp(self):
        """Set up a complex Graph object for testing."""
        self.G = Graph(directed=True)
        
        # 1. Setup Vertices
        # We use a mix of types to test type inference
        self.G.entity_types = {
            "n1": "vertex",
            "n2": "vertex",
            "n3": "vertex",
            "h1": "hyperedge" # Should NOT be a CX2 node, but saved in manifest
        }
        
        # 2. Setup Edges
        # n1->n2 (binary)
        self.G.edge_definitions = {
            "e1": ("n1", "n2", "type_a"),
            "he1": ("n1", "n2", "n3") # Malformed/Hyperedge definition
        }
        self.G.edge_weights = {"e1": 1.5, "he1": 0.5}
        
        # 3. Setup Attributes (Polars)
        # Vertex Attributes
        self.G.vertex_attributes = pl.DataFrame({
            "id": ["n1", "n2", "n3"],
            "score": [10, 20, 30],       # Integer
            "active": [True, False, True], # Boolean
            "desc": ["A", "B", "C"]      # String
        })
        
        # Edge Attributes
        self.G.edge_attributes = pl.DataFrame({
            "edge_id": ["e1"],
            "confidence": [0.99]         # Double
        })
        
        # 4. Complex/Hidden Attributes (For Manifest testing)
        self.G._slices = {"slice1": {"vertices": {1, 2, 3}}}
        self.G.hyperedge_definitions = {"he1": ["n1", "n2", "n3"]}

    def test_cx2_structure_basics(self):
        """Test that to_cx2 produces the mandatory CX2 list structure."""
        cx2_data = to_cx2(self.G)
        
        self.assertIsInstance(cx2_data, list)
        
        # Check for mandatory aspects
        aspect_names = [list(item.keys())[0] for item in cx2_data if item]
        self.assertIn("CXVersion", aspect_names)
        self.assertIn("metaData", aspect_names)
        self.assertIn("attributeDeclarations", aspect_names)
        self.assertIn("nodes", aspect_names)
        self.assertIn("edges", aspect_names)
        self.assertIn("networkAttributes", aspect_names)

    def test_node_mapping(self):
        """Test that only 'vertex' entities become CX2 nodes."""
        cx2_data = to_cx2(self.G)
        
        # Extract nodes aspect
        nodes_aspect = next(item["nodes"] for item in cx2_data if "nodes" in item)
        
        # Should contain n1, n2, n3 (3 nodes). h1 (hyperedge) should be ignored.
        self.assertEqual(len(nodes_aspect), 3)
        
        node_names = [n["v"]["name"] for n in nodes_aspect]
        self.assertIn("n1", node_names)
        self.assertNotIn("h1", node_names)

    def test_edge_mapping(self):
        """Test that only valid binary edges become CX2 edges."""
        cx2_data = to_cx2(self.G)
        
        edges_aspect = next(item["edges"] for item in cx2_data if "edges" in item)
        
        # Should only contain 'e1'. 'he1' is a hyperedge and should be skipped.
        self.assertEqual(len(edges_aspect), 1)
        self.assertEqual(edges_aspect[0]["v"]["interaction"], "e1")
        self.assertEqual(edges_aspect[0]["v"]["weight"], 1.5)

    def test_attribute_declarations_and_types(self):
        """Test that Polars types are correctly mapped to CX2 types."""
        cx2_data = to_cx2(self.G)
        
        decls = next(item["attributeDeclarations"][0] for item in cx2_data if "attributeDeclarations" in item)
        
        # Check Node Attributes
        node_decls = decls["nodes"]
        self.assertEqual(node_decls["score"]["d"], "long")
        self.assertEqual(node_decls["active"]["d"], "boolean")
        self.assertEqual(node_decls["desc"]["d"], "string")
        
        # Check Edge Attributes
        edge_decls = decls["edges"]
        self.assertEqual(edge_decls["confidence"]["d"], "double")

    def test_manifest_preservation(self):
        """Test that complex data is stored in the manifest JSON string."""
        cx2_data = to_cx2(self.G)
        
        net_attrs = next(item["networkAttributes"][0] for item in cx2_data if "networkAttributes" in item)
        
        self.assertIn("__AnnNet_Manifest__", net_attrs)
        
        manifest_json = net_attrs["__AnnNet_Manifest__"]
        manifest = json.loads(manifest_json)
        
        # Check that our "hidden" slices data survived
        # Note: The adapter uses _serialize_slices. We assume the mock/implementation 
        # preserves the data structure, or at least the key exists.
        self.assertIn("slices", manifest)
        # Check hyperedges
        self.assertIn("hyperedges", manifest["edges"])
        self.assertIn("he1", manifest["edges"]["hyperedges"])

    def test_round_trip_basic(self):
            """Test G -> CX2 -> G matches basic structure."""
            cx2_data = to_cx2(self.G)
            G_new = from_cx2(cx2_data)
            
            # Check Vertices
            self.assertEqual(len(G_new.entity_types), len(self.G.entity_types))
            self.assertIn("n1", G_new.entity_types)
            
            # Check Edges
            self.assertIn("e1", G_new.edge_definitions)
            self.assertEqual(G_new.edge_weights["e1"], 1.5)
            
            # Check Attributes (Polars)
            df_new = G_new.vertex_attributes
            
            # This works reliably across Polars versions
            score_n1 = df_new.filter(pl.col("vertex_id") == "n1").get_column("score").item()
            
            self.assertEqual(score_n1, 10)

    def test_from_cx2_external_no_manifest(self):
        """
        Test importing a 'raw' CX2 file (e.g. from Cytoscape) that lacks 
        the AnnNet manifest. It should still build a graph.
        """
        # Manually construct a minimal CX2
        raw_cx2 = [
            {"CXVersion": "2.0"},
            {"attributeDeclarations": [{"nodes": {"importance": {"d": "integer"}}}]},
            {"nodes": [
                {"id": 100, "v": {"name": "EXT_NODE", "importance": 5}}
            ]},
            {"edges": []}
        ]
        
        G_ext = from_cx2(raw_cx2)
        
        # Should have 1 vertex
        self.assertEqual(len(G_ext.entity_types), 1)
        self.assertIn("EXT_NODE", G_ext.entity_types)
        
        # Attributes should be loaded
        df = G_ext.vertex_attributes
        self.assertFalse(df.is_empty())
        self.assertEqual(
            df.filter(pl.col("vertex_id") == "EXT_NODE")["importance"].item(),
            5,
        )

if __name__ == '__main__':
    unittest.main()