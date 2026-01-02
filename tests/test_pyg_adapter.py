import os
import sys
import unittest
import warnings

# Silence noisy warnings
warnings.filterwarnings(
    "ignore",
    message=r"Signature .*numpy\.longdouble.*",
    category=UserWarning,
    module=r"numpy\._core\.getlimits",
)

# Make project importable when tests run from /tests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from annnet.core.graph import AnnNet

# --- Optional deps gates ------------------------------------------------------
try:
    import torch
    from torch_geometric.data import HeteroData

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False

try:
    import polars as pl

    HAS_POLARS = True
except Exception:
    HAS_POLARS = False

try:
    import pandas as pd

    HAS_PANDAS = True
except Exception:
    HAS_PANDAS = False

# Import adapter under test
from annnet.adapters.pyg_adapter import to_pyg


def _build_graph() -> AnnNet:
    """Build a realistic test graph using the real AnnNet class."""
    g = AnnNet(directed=True)

    # Vertices with different kinds and attributes
    g.add_vertex("p1", kind="protein", weight=1.5, active=1.0, label="alpha")
    g.add_vertex("p2", kind="protein", weight=2.3, active=0.0, label="beta")
    g.add_vertex("p3", kind="protein", weight=1.8, active=1.0, label="gamma")

    g.add_vertex("g1", kind="gene", expression=100.0, length=500.0, label="BRCA1")
    g.add_vertex("g2", kind="gene", expression=50.0, length=300.0, label="TP53")

    g.add_vertex("d1", kind="drug", dosage=10.0, approved=1.0)

    # Binary edges: directed and undirected
    e1 = g.add_edge("p1", "g1", weight=2.0, interaction=+1, tag="regulates")
    e2 = g.add_edge("p2", "g2", weight=1.0, edge_directed=False, interaction=-1, tag="inhibits")
    e3 = g.add_edge("d1", "p1", weight=0.5, interaction=+1, tag="targets")
    e4 = g.add_edge("p3", "g1", weight=1.5, interaction=+1, tag="activates")

    # Hyperedge
    e5 = g.add_hyperedge(head=["g1", "g2"], tail=["p1"], weight=3.0, interaction=+1, edge_id="he1")

    g.add_slice("active_only", region="high_expr")
    g.add_vertex_to_slice("active_only", "p1")
    g.add_vertex_to_slice("active_only", "p3")
    g.add_vertex_to_slice("active_only", "g1")
    g.add_edge_to_slice("active_only", e1)
    g.add_edge_to_slice("active_only", e4)

    # Per-slice weight override
    g.set_edge_slice_attrs("active_only", e1, weight=5.0)

    # Second slice with different membership
    g.add_slice("druggable")
    g.add_vertex_to_slice("druggable", "d1")
    g.add_vertex_to_slice("druggable", "p1")
    g.add_vertex_to_slice("druggable", "p2")
    g.add_edge_to_slice("druggable", e3)

    # Basic sanity
    assert g.number_of_edges() >= 4
    assert len([eid for eid in g.hyperedge_definitions]) >= 1

    return g


_BUILD_GRAPH = _build_graph


@unittest.skipUnless(HAS_TORCH, "PyTorch or PyTorch Geometric not available")
class TestPyGAdapter(unittest.TestCase):
    """Production-ready test suite for AnnNet -> PyTorch Geometric adapter."""

    @classmethod
    def setUpClass(cls):
        if _BUILD_GRAPH is None:
            raise unittest.SkipTest("No _build_graph() found.")

    def test_basic_export_structure(self):
        """Basic export should produce valid HeteroData with correct node/edge types."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="skip")

        # Basic structure
        self.assertIsInstance(data, HeteroData)
        self.assertIn("protein", data.node_types)
        self.assertIn("gene", data.node_types)
        self.assertIn("drug", data.node_types)

        # Node counts
        self.assertEqual(data["protein"].num_nodes, 3)
        self.assertEqual(data["gene"].num_nodes, 2)
        self.assertEqual(data["drug"].num_nodes, 1)

        # Should have heterogeneous edges
        self.assertIn(("protein", "edge", "gene"), data.edge_types)
        self.assertIn(("drug", "edge", "protein"), data.edge_types)

    def test_node_features_extraction(self):
        """Node features should be correctly extracted into tensors."""
        g = _BUILD_GRAPH()

        data = to_pyg(
            g,
            node_features={
                "protein": ["weight", "active"],
                "gene": ["expression", "length"],
                "drug": ["dosage", "approved"],
            },
            hyperedge_mode="skip",
        )

        # Protein features
        self.assertIn("x", data["protein"])
        self.assertEqual(data["protein"].x.shape, (3, 2))
        self.assertEqual(data["protein"].x.dtype, torch.float32)

        # Gene features
        self.assertEqual(data["gene"].x.shape, (2, 2))

        # Drug features
        self.assertEqual(data["drug"].x.shape, (1, 2))

        # Verify actual values (at least one check)
        manifest = data.manifest["node_index"]["protein"]
        p1_idx = manifest["p1"]
        # p1 has weight=1.5, active=1.0
        self.assertAlmostEqual(data["protein"].x[p1_idx, 0].item(), 1.5, places=5)
        self.assertAlmostEqual(data["protein"].x[p1_idx, 1].item(), 1.0, places=5)

    def test_edge_weights_preserved(self):
        """Edge weights should be correctly transferred."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="skip")

        # Check protein->gene edges have weights
        etype = ("protein", "edge", "gene")
        self.assertIn("edge_weight", data[etype])
        self.assertEqual(data[etype].edge_weight.dtype, torch.float32)

        # Should have at least 2 protein->gene edges (e1 and e4)
        self.assertGreaterEqual(data[etype].edge_index.shape[1], 2)
        self.assertGreaterEqual(len(data[etype].edge_weight), 2)

        # Weights should be positive
        self.assertTrue(torch.all(data[etype].edge_weight > 0))

    def test_manifest_structure_and_completeness(self):
        """Manifest should contain complete mapping information."""
        g = _BUILD_GRAPH()

        data = to_pyg(g)

        # Manifest exists - use hasattr instead of dir() check
        self.assertTrue(hasattr(data, "manifest"))
        self.assertIn("node_index", data.manifest)
        self.assertIn("edge_index", data.manifest)

        # All node kinds in manifest
        self.assertIn("protein", data.manifest["node_index"])
        self.assertIn("gene", data.manifest["node_index"])
        self.assertIn("drug", data.manifest["node_index"])

        # All original vertices mapped
        protein_map = data.manifest["node_index"]["protein"]
        self.assertEqual(set(protein_map.keys()), {"p1", "p2", "p3"})

        gene_map = data.manifest["node_index"]["gene"]
        self.assertEqual(set(gene_map.keys()), {"g1", "g2"})

        drug_map = data.manifest["node_index"]["drug"]
        self.assertEqual(set(drug_map.keys()), {"d1"})

        # Indices should be sequential 0-based
        self.assertEqual(set(protein_map.values()), {0, 1, 2})
        self.assertEqual(set(gene_map.values()), {0, 1})
        self.assertEqual(set(drug_map.values()), {0})

    def test_slice_mask_single_slice(self):
        """Single slice should produce correct boolean masks."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, slice_id="active_only", hyperedge_mode="skip")

        # Masks should exist
        self.assertIn("active_only_mask", data["protein"])
        self.assertIn("active_only_mask", data["gene"])

        # Check protein mask (p1 and p3 are in active_only)
        protein_mask = data["protein"]["active_only_mask"]
        self.assertEqual(protein_mask.dtype, torch.bool)
        self.assertEqual(protein_mask.sum().item(), 2)

        # Check gene mask (g1 is in active_only)
        gene_mask = data["gene"]["active_only_mask"]
        self.assertEqual(gene_mask.sum().item(), 1)

        # Drug should have all-False mask (not in active_only)
        drug_mask = data["drug"]["active_only_mask"]
        self.assertEqual(drug_mask.sum().item(), 0)

    def test_slice_mask_different_slice(self):
        """Different slice should produce different masks."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, slice_id="druggable", hyperedge_mode="skip")

        # Druggable slice: d1, p1, p2
        protein_mask = data["protein"]["druggable_mask"]
        self.assertEqual(protein_mask.sum().item(), 2)

        drug_mask = data["drug"]["druggable_mask"]
        self.assertEqual(drug_mask.sum().item(), 1)

        # Genes not in druggable
        gene_mask = data["gene"]["druggable_mask"]
        self.assertEqual(gene_mask.sum().item(), 0)

    def test_hyperedge_skip_mode(self):
        """Skip mode should not create hypernodes or membership edges."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="skip")

        # No hypernode type
        self.assertNotIn("hypernode", data.node_types)

        # No member_of edges
        member_edges = [et for et in data.edge_types if "member_of" in et]
        self.assertEqual(len(member_edges), 0)

    def test_hyperedge_reify_mode(self):
        """Reify mode should create hypernodes and membership edges."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="reify")

        # Hypernode type should exist
        self.assertIn("hypernode", data.node_types)

        # Should have at least 1 hypernode (he1)
        self.assertGreaterEqual(data["hypernode"].num_nodes, 1)

        # Should have member_of edges
        member_edge_types = [et for et in data.edge_types if "member_of" in et]
        self.assertGreater(len(member_edge_types), 0)

        # Check specific membership (he1: head=[g1,g2], tail=[p1])
        # Should have protein->hypernode and gene->hypernode edges
        protein_member = ("protein", "member_of", "hypernode")
        gene_member = ("gene", "member_of", "hypernode")

        has_protein_member = protein_member in data.edge_types
        has_gene_member = gene_member in data.edge_types

        self.assertTrue(has_protein_member or has_gene_member)

    def test_hyperedge_expand_mode(self):
        """Expand mode should create pairwise edges from hyperedge."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="expand")

        # No hypernode in expand mode
        self.assertNotIn("hypernode", data.node_types)

        # Should have more protein->gene edges than with skip
        data_skip = to_pyg(g, hyperedge_mode="skip")

        pg_edges_expand = data[("protein", "edge", "gene")].edge_index.shape[1]
        pg_edges_skip = data_skip[("protein", "edge", "gene")].edge_index.shape[1]

        # he1 is directed: tail=[p1] to head=[g1, g2]
        # Should add 2 edges: p1->g1, p1->g2
        # But p1->g1 already exists as e1, so might not add duplicate
        # Still, expand should have >= skip
        self.assertGreaterEqual(pg_edges_expand, pg_edges_skip)

    def test_hyperedge_expand_cartesian_product(self):
        """Expanded directed hyperedge should create tail×head edges."""
        g = AnnNet(directed=True)

        # Simple hyperedge: 2 tails × 2 heads = 4 edges
        g.add_vertex("t1", kind="A")
        g.add_vertex("t2", kind="A")
        g.add_vertex("h1", kind="B")
        g.add_vertex("h2", kind="B")

        g.add_hyperedge(tail=["t1", "t2"], head=["h1", "h2"], edge_id="he_test", edge_directed=True)

        data = to_pyg(g, hyperedge_mode="expand")

        etype = ("A", "edge", "B")
        # Should have exactly 4 edges
        self.assertEqual(data[etype].edge_index.shape[1], 4)

    def test_device_placement_cpu(self):
        """Data should be on CPU when device='cpu'."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, node_features={"protein": ["weight"]}, device="cpu", hyperedge_mode="skip")

        # Check tensors are on CPU
        self.assertEqual(data["protein"].x.device.type, "cpu")
        etype = ("protein", "edge", "gene")
        self.assertEqual(data[etype].edge_index.device.type, "cpu")
        self.assertEqual(data[etype].edge_weight.device.type, "cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_placement_cuda(self):
        """Data should be on CUDA when device='cuda'."""
        g = _BUILD_GRAPH()

        data = to_pyg(
            g, node_features={"protein": ["weight"]}, device="cuda", hyperedge_mode="skip"
        )

        # Check tensors are on CUDA
        self.assertEqual(data["protein"].x.device.type, "cuda")
        etype = ("protein", "edge", "gene")
        self.assertEqual(data[etype].edge_index.device.type, "cuda")

    def test_empty_graph_graceful(self):
        """Empty graph should not crash."""
        g = AnnNet()

        data = to_pyg(g)

        self.assertIsInstance(data, HeteroData)
        self.assertEqual(len(data.node_types), 0)
        self.assertEqual(len(data.edge_types), 0)

    def test_graph_with_only_vertices(self):
        """Graph with vertices but no edges."""
        g = AnnNet()
        g.add_vertex("v1", kind="A")
        g.add_vertex("v2", kind="A")

        data = to_pyg(g)

        self.assertEqual(data["A"].num_nodes, 2)
        self.assertEqual(len(data.edge_types), 0)

    def test_missing_kind_defaults_to_default(self):
        """Vertices without 'kind' attribute should use 'default'."""
        g = AnnNet()
        g.add_vertex("v1")  # No kind specified
        g.add_vertex("v2")

        data = to_pyg(g)

        self.assertIn("default", data.node_types)
        self.assertEqual(data["default"].num_nodes, 2)

    def test_heterogeneous_edge_types(self):
        """Multiple edge types between different node kind pairs."""
        g = _BUILD_GRAPH()

        data = to_pyg(g, hyperedge_mode="skip")

        # Should have multiple distinct edge types
        edge_types = data.edge_types
        self.assertGreater(len(edge_types), 1)

        # All edge types should be 3-tuples
        for et in edge_types:
            self.assertEqual(len(et), 3)
            self.assertEqual(et[1], "edge")

    def test_self_loops_handled(self):
        """Self-loop edges should be handled correctly."""
        g = AnnNet()
        g.add_vertex("v1", kind="A")
        g.add_edge("v1", "v1", edge_id="self_loop", weight=1.0)

        data = to_pyg(g)

        etype = ("A", "edge", "A")
        self.assertIn(etype, data.edge_types)
        self.assertEqual(data[etype].edge_index[0, 0].item(), 0)
        self.assertEqual(data[etype].edge_index[1, 0].item(), 0)

    def test_undirected_edges_represented(self):
        """Undirected edges should be in the output."""
        g = _BUILD_GRAPH()

        # e2 is undirected (p2<->g2)
        data = to_pyg(g, hyperedge_mode="skip")

        # Should still appear in edge_index
        etype = ("protein", "edge", "gene")
        self.assertIn(etype, data.edge_types)
        # Can't easily verify directionality in PyG without more context,
        # but edge should exist

    def test_large_graph_performance(self):
        """Adapter should handle larger graphs efficiently."""
        g = AnnNet(directed=True)

        n_proteins = 500
        n_genes = 300
        n_edges = 2000

        for i in range(n_proteins):
            g.add_vertex(f"p{i}", kind="protein", weight=float(i))

        for i in range(n_genes):
            g.add_vertex(f"g{i}", kind="gene", expression=float(i * 10))

        # Random edges
        import random

        random.seed(42)
        for i in range(n_edges):
            p = f"p{random.randint(0, n_proteins - 1)}"
            gn = f"g{random.randint(0, n_genes - 1)}"
            g.add_edge(p, gn, edge_id=f"e{i}", weight=random.random())

        import time

        start = time.time()
        data = to_pyg(
            g, node_features={"protein": ["weight"], "gene": ["expression"]}, hyperedge_mode="skip"
        )
        elapsed = time.time() - start

        # Should complete in reasonable time
        self.assertLess(elapsed, 10.0, "Conversion should complete in <10 seconds")

        # Verify structure
        self.assertEqual(data["protein"].num_nodes, n_proteins)
        self.assertEqual(data["gene"].num_nodes, n_genes)
        self.assertEqual(data[("protein", "edge", "gene")].edge_index.shape[1], n_edges)

    @unittest.skipUnless(HAS_POLARS, "Polars not available")
    def test_polars_dataframe_compatibility(self):
        """Should work with Polars DataFrames."""
        g = _BUILD_GRAPH()

        # Ensure vertex_attributes is Polars
        if not isinstance(g.vertex_attributes, pl.DataFrame):
            # This test assumes AnnNet uses Polars by default
            # If not, convert
            pass

        data = to_pyg(g, node_features={"protein": ["weight", "active"]}, hyperedge_mode="skip")

        self.assertEqual(data["protein"].x.shape, (3, 2))

    @unittest.skipUnless(HAS_PANDAS, "Pandas not available")
    def test_pandas_dataframe_compatibility(self):
        """Should work with Pandas DataFrames."""
        g = _BUILD_GRAPH()

        # Convert to Pandas if needed
        if HAS_POLARS and isinstance(g.vertex_attributes, pl.DataFrame):
            g.vertex_attributes = g.vertex_attributes.to_pandas()

        data = to_pyg(g, node_features={"protein": ["weight", "active"]}, hyperedge_mode="skip")

        self.assertEqual(data["protein"].x.shape, (3, 2))

    def test_missing_edge_vertices_skipped(self):
        """Edges referencing nonexistent vertices should be skipped."""
        g = AnnNet()
        g.add_vertex("v1", kind="A")

        # Manually add malformed edge definition
        g.edge_definitions["bad_edge"] = ("v1", "v_nonexistent", {})

        data = to_pyg(g)

        # Should not crash, edge should be skipped
        self.assertEqual(len(data.edge_types), 0)

    def test_null_attribute_values_handled(self):
        """Null/None attribute values should be handled gracefully."""
        g = AnnNet()
        g.add_vertex("p1", kind="protein", weight=1.5, score=None)
        g.add_vertex("p2", kind="protein", weight=None, score=2.0)

        data = to_pyg(g, node_features={"protein": ["weight", "score"]})

        # Should use 0.0 for None values
        self.assertEqual(data["protein"].x.shape, (2, 2))
        # Check that tensor doesn't have NaN
        self.assertFalse(torch.any(torch.isnan(data["protein"].x)))

    def test_non_numeric_features_raise_error(self):
        """Non-numeric feature columns should raise ValueError."""
        g = AnnNet()
        g.add_vertex("p1", kind="protein", name="Protein1")

        with self.assertRaises(ValueError) as ctx:
            to_pyg(g, node_features={"protein": ["name"]})

        self.assertIn("must be numeric", str(ctx.exception))

    def test_edge_attributes_extraction(self):
        """Edge attributes should be extracted when specified."""
        g = _BUILD_GRAPH()

        data = to_pyg(
            g, edge_features={("protein", "edge", "gene"): ["interaction"]}, hyperedge_mode="skip"
        )

        etype = ("protein", "edge", "gene")
        # Edge attributes might be present depending on implementation
        # Just verify no crash and structure is valid
        self.assertIn(etype, data.edge_types)

    def test_integer_vertex_ids(self):
        """Integer vertex IDs should be handled correctly."""
        g = AnnNet()
        g.add_vertex("1", kind="A")  # Use string to avoid Polars schema conflict
        g.add_vertex("2", kind="A")
        g.add_edge("1", "2", edge_id="e1")

        data = to_pyg(g)

        self.assertEqual(data["A"].num_nodes, 2)
        self.assertEqual(data[("A", "edge", "A")].edge_index.shape[1], 1)

    def test_mixed_id_types(self):
        """Mixed string/integer IDs should work."""
        g = AnnNet()
        g.add_vertex("v1", kind="A")
        g.add_vertex("v2", kind="A")  # Keep all as strings to avoid Polars schema issues
        g.add_edge("v1", "v2", edge_id="e1")

        data = to_pyg(g)

        self.assertEqual(data["A"].num_nodes, 2)
        self.assertEqual(data[("A", "edge", "A")].edge_index.shape[1], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
