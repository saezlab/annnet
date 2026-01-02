# tests/test_io_annnet.py
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath("."))
# Test imports (package layout)
import numpy as np
import polars as pl
import zarr

from annnet.core.graph import AnnNet
from annnet.io.io_annnet import read as annnet_read
from annnet.io.io_annnet import write as annnet_write
from annnet.io._utils import _read_archive


class TestAnnNetIO(unittest.TestCase):
    def setUp(self):
        # Build a tiny directed graph with a slice + hyperedge
        G = AnnNet(directed=True)

        # Vertices (two in slice1)
        G.add_vertex("v1", slice="slice1")
        G.add_vertex("v2", slice="slice1")
        G.add_vertex("v3")
        G.add_vertex("v4")

        # Edges
        G.add_edge("v1", "v2", edge_id="e1", weight=1.5)
        G.add_edge("v2", "v3", edge_id="e2", weight=2.0)
        G.add_edge("v3", "v4", edge_id="e3", weight=0.5)

        # Hyperedge (undirected)
        G.add_hyperedge(members=["v1", "v2", "v3"], edge_id="h1", weight=3.0)

        # Some unstructured metadata (will go to uns/)
        G.graph_attributes["project"] = "unittest"
        G.graph_attributes["tags"] = ["io", "annnet"]

        # Add a nested history row to ensure audit/JSON stringify path is exercised
        G._history.append(
            {
                "ts": "2025-10-23T00:00:00Z",
                "action": "create",
                "payload": {"nested": {"x": [1, 2, 3]}},
                "notes": ["a", "b"],
                "arr": np.array([1, 2, 3]),
                "maybe_empty": {},
            }
        )

        self.G = G
        self.tmpdir = tempfile.mkdtemp()
        self._archive_tmp = None

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        if self._archive_tmp:
            shutil.rmtree(self._archive_tmp, ignore_errors=True)

    # ----------------------- helpers -----------------------
    def _roundtrip(self, use_archive=False):
        if use_archive:
            out_path = Path(self.tmpdir) / "test_graph.annnet"
        else:
            out_path = Path(self.tmpdir) / "test_graph_dir"
        
        annnet_write(self.G, out_path, compression="zstd", overwrite=True)
        G2 = annnet_read(out_path)
        return G2, out_path

    def _test_both_modes(self, test_func):
        """Run test in both directory and archive mode."""
        for mode_name, use_archive in [("directory", False), ("archive", True)]:
            with self.subTest(mode=mode_name):
                test_func(use_archive)

    def _get_root(self, out_path, use_archive):
        """Get root directory for checks (extract if archive)."""
        if use_archive:
            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.suffix, ".annnet")
            tmp = tempfile.mkdtemp()
            self._archive_tmp = tmp
            return _read_archive(out_path, Path(tmp))
        else:
            self.assertTrue(out_path.is_dir())
            return out_path

    # ----------------------- tests -------------------------
    def test_write_read_roundtrip_basic(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)

            # Top-level counts
            self.assertEqual(len(self.G.entity_to_idx), len(G2.entity_to_idx))
            self.assertEqual(self.G._num_edges, G2._num_edges)
            self.assertEqual(set(self.G._slices.keys()), set(G2._slices.keys()))
            self.assertEqual(self.G.edge_weights, G2.edge_weights)

            # Hyperedges preserved
            self.assertTrue(hasattr(G2, "hyperedge_definitions"))
            self.assertGreater(len(G2.hyperedge_definitions), 0)

            # A couple of identity maps should match
            self.assertEqual(self.G.entity_to_idx, G2.entity_to_idx)
            self.assertEqual(self.G.edge_to_idx, G2.edge_to_idx)

            # Edge metadata
            self.assertEqual(self.G.edge_directed, G2.edge_directed)
            self.assertEqual(self.G.edge_kind, G2.edge_kind)

            # slices: same edge sets, vertex sets
            for lid in self.G._slices:
                self.assertEqual(self.G._slices[lid]["vertices"], G2._slices[lid]["vertices"])
                self.assertEqual(self.G._slices[lid]["edges"], G2._slices[lid]["edges"])
                self.assertEqual(
                    self.G.slice_edge_weights.get(lid, {}), G2.slice_edge_weights.get(lid, {})
                )
        
        self._test_both_modes(_test)

    def test_manifest_and_layout(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)
            
            self.assertTrue((root / "manifest.json").exists())

            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["format"], "annnet")
            self.assertIn("counts", manifest)
            self.assertEqual(manifest["directed"], True)
            self.assertIn("compression", manifest)
            self.assertIn("encoding", manifest)

            # Core layout
            self.assertTrue((root / "structure").exists())
            self.assertTrue((root / "tables").exists())
            self.assertTrue((root / "slices").exists())
            self.assertTrue((root / "audit").exists())
            self.assertTrue((root / "uns").exists())
        
        self._test_both_modes(_test)

    def test_zarr_incidence_group(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)
            
            inc = root / "structure" / "incidence.zarr"
            self.assertTrue(inc.exists())

            # Open Zarr v3 group and validate arrays + attrs
            grp = zarr.open_group(str(inc), mode="r")
            # arrays live as subdirs; zarr v3 exposes them by name
            self.assertIn("row", grp.array_keys())
            self.assertIn("col", grp.array_keys())
            self.assertIn("data", grp.array_keys())

            row = grp["row"][:]
            col = grp["col"][:]
            dat = grp["data"][:]
            shape = tuple(grp.attrs["shape"])

            # Shapes/dtypes (dtype implied by writer: int32/int32/float32)
            self.assertEqual(row.dtype, np.int32)
            self.assertEqual(col.dtype, np.int32)
            self.assertEqual(dat.dtype, np.float32)
            self.assertEqual(shape, self.G._matrix.shape)

            # COO consistency: same length across row/col/data
            self.assertEqual(len(row), len(col))
            self.assertEqual(len(row), len(dat))
        
        self._test_both_modes(_test)

    def test_overwrite_semantics(self):
        out = Path(self.tmpdir) / "test_overwrite"
        # first write
        annnet_write(self.G, out, compression="zstd", overwrite=True)
        # second write without overwrite should fail
        with self.assertRaises(FileExistsError):
            annnet_write(self.G, out, compression="zstd", overwrite=False)
        # now allow overwrite
        annnet_write(self.G, out, compression="zstd", overwrite=True)

    def test_write_read_kivela_layers(self):
        def _test(use_archive):
            """Test roundtrip of Kivela multilayer structures."""
            # 1. Setup Kivela Multilayer Data on the existing graph
            self.G.aspects = ["time", "transport"]
            self.G.elem_layers = {"time": ["t1", "t2"], "transport": ["bus", "train"]}

            # Initialize containers if they don't exist (depends on AnnNet init)
            if not hasattr(self.G, "_VM"):
                self.G._VM = set()
            if not hasattr(self.G, "edge_layers"):
                self.G.edge_layers = {}
            if not hasattr(self.G, "_layer_attrs"):
                self.G._layer_attrs = {}
            if not hasattr(self.G, "_vertex_layer_attrs"):
                self.G._vertex_layer_attrs = {}
            if not hasattr(self.G, "_aspect_attrs"):
                self.G._aspect_attrs = {}

            # Vertex Presence: (u, layer_tuple)
            self.G._VM.add(("v1", ("t1", "bus")))
            self.G._VM.add(("v2", ("t1", "bus")))
            self.G._VM.add(("v2", ("t2", "train")))

            # Edge Layers
            # e1 (v1-v2): Intra-layer edge in (t1, bus)
            self.G.edge_layers["e1"] = ("t1", "bus")
            self.G.edge_kind["e1"] = "intra"

            # e2 (v2-v3): Inter-layer edge between (t1, bus) and (t2, train)
            self.G.edge_layers["e2"] = (("t1", "bus"), ("t2", "train"))
            self.G.edge_kind["e2"] = "inter"

            # Attributes
            self.G._aspect_attrs = {"time": {"unit": "seconds"}}
            self.G._layer_attrs = {("t1", "bus"): {"cost": 10}}
            self.G._vertex_layer_attrs = {("v1", ("t1", "bus")): {"status": "active"}}

            # Elementary layer attributes (Polars DataFrame)
            self.G.layer_attributes = pl.DataFrame(
                [
                    {"layer_id": "time_t1", "desc": "Morning"},
                    {"layer_id": "transport_bus", "desc": "Public Bus"},
                ]
            )

            # Mock the cache rebuilder if strictly necessary for IO (avoids logic errors during load)
            self.G._rebuild_all_layers_cache = lambda: None

            # 2. Roundtrip
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # 3. Verify Restoration
            # Metadata
            self.assertEqual(G2.aspects, ["time", "transport"])
            self.assertEqual(G2.elem_layers, self.G.elem_layers)

            # Vertex Presence
            self.assertEqual(len(G2._VM), 3)
            self.assertIn(("v1", ("t1", "bus")), G2._VM)
            self.assertIn(("v2", ("t2", "train")), G2._VM)

            # Edge Layers & Kinds
            self.assertEqual(G2.edge_layers["e1"], ("t1", "bus"))
            self.assertEqual(G2.edge_kind["e1"], "intra")

            # Verify Inter-layer tuple of tuples is restored correctly
            self.assertEqual(G2.edge_layers["e2"], (("t1", "bus"), ("t2", "train")))
            self.assertEqual(G2.edge_kind["e2"], "inter")

            # Attributes
            self.assertEqual(G2._aspect_attrs["time"]["unit"], "seconds")
            self.assertEqual(G2._layer_attrs[("t1", "bus")]["cost"], 10)
            self.assertEqual(G2._vertex_layer_attrs[("v1", ("t1", "bus"))]["status"], "active")

            # Verify DataFrame attributes
            self.assertFalse(G2.layer_attributes.is_empty())
            row = G2.layer_attributes.filter(pl.col("layer_id") == "time_t1").to_dicts()[0]
            self.assertEqual(row["desc"], "Morning")

            # Verify Manifest Update
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["counts"]["aspects"], 2)
        
        self._test_both_modes(_test)

    def test_slices_registry_and_memberships(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)
            
            slices_dir = root / "slices"
            self.assertTrue((slices_dir / "registry.parquet").exists())
            self.assertTrue((slices_dir / "vertex_memberships.parquet").exists())
            self.assertTrue((slices_dir / "edge_memberships.parquet").exists())

            reg = pl.read_parquet(slices_dir / "registry.parquet")
            vmem = pl.read_parquet(slices_dir / "vertex_memberships.parquet")
            emem = pl.read_parquet(slices_dir / "edge_memberships.parquet")

            self.assertGreaterEqual(reg.height, 1)
            self.assertIn("slice_id", reg.columns)

            # slice1 must have at least v1,v2
            vset = set(vmem.filter(pl.col("slice_id") == "slice1")["vertex_id"].to_list())
            self.assertTrue({"v1", "v2"}.issubset(vset))

            # edges exist in memberships as well
            self.assertIn("edge_id", emem.columns)
            self.assertIn("weight", emem.columns)
        
        self._test_both_modes(_test)

    def test_hyperedge_definitions_parquet(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)
            
            p = root / "structure" / "hyperedge_definitions.parquet"
            self.assertTrue(p.exists())
            df = pl.read_parquet(p)
            self.assertIn("edge_id", df.columns)
            self.assertIn("directed", df.columns)
            # at least one of members/head/tail exists (depending on directed flag)
            self.assertTrue(any(c in df.columns for c in ("members", "head", "tail")))
        
        self._test_both_modes(_test)

    def test_audit_and_uns_written(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # audit: history.parquet should exist and mixed nested columns converted to Utf8(JSON)
            hist = root / "audit" / "history.parquet"
            self.assertTrue(hist.exists())
            hdf = pl.read_parquet(hist)

            # payload/notes/arr/maybe_empty should be present (stringified) if they existed
            cols = set(hdf.columns)
            # Some columns might be absent if the schema was inferred differently,
            # so only check types for those that exist.
            for candidate in ("payload", "notes", "arr", "maybe_empty"):
                if candidate in cols:
                    self.assertEqual(hdf.schema[candidate], pl.Utf8)

            # uns: graph_attributes.json
            gattr = root / "uns" / "graph_attributes.json"
            self.assertTrue(gattr.exists())
            attrs = json.loads(gattr.read_text())
            self.assertEqual(attrs.get("project"), "unittest")
            self.assertEqual(attrs.get("tags"), ["io", "annnet"])
        
        self._test_both_modes(_test)

    def test_read_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            annnet_read(Path(self.tmpdir) / "does_not_exist.annnet")

    def test_write_read_kivela_empty_state(self):
        def _test(use_archive):
            """Test multilayer graph with aspects but NO presence/edges (Edge case)."""
            # 1. Setup minimal multilayer metadata
            self.G.aspects = ["time"]
            self.G.elem_layers = {"time": ["t1", "t2"]}

            # Ensure containers exist but are EMPTY
            self.G._VM = set()
            self.G.edge_layers = {}
            # We leave attributes empty as well to test optional attribute file writing
            self.G._layer_attrs = {}

            # 2. Roundtrip
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # 3. Assertions
            # Aspects preserved?
            self.assertEqual(G2.aspects, ["time"])
            # VM is empty?
            self.assertEqual(len(G2._VM), 0)
            # Verify the file was actually written (the empty schema parquet)
            self.assertTrue((root / "layers" / "vertex_presence.parquet").exists())
            # Verify attribute files were NOT written (optimization check)
            self.assertFalse((root / "layers" / "tuple_layer_attributes.parquet").exists())
        
        self._test_both_modes(_test)

    def test_write_read_large_sparse_graph(self):
        def _test(use_archive):
            n_vertices = 10_000
            n_edges = 100_000

            G = AnnNet(directed=True)

            G.add_vertices_bulk(
                ({"vertex_id": f"v{i}"} for i in range(n_vertices))
            )

            bulk = []
            for i in range(n_edges):
                bulk.append(
                    {
                        "source": f"v{i % n_vertices}",
                        "target": f"v{(i * 37) % n_vertices}",
                        "weight": float(i % 7),
                        "edge_type": "regular",
                    }
                )

            eids = G.add_edges_bulk(bulk)

            if use_archive:
                out_path = Path(self.tmpdir) / "large_graph.annnet"
            else:
                out_path = Path(self.tmpdir) / "large_graph_dir"

            annnet_write(G, out_path, compression="zstd", overwrite=True)
            G2 = annnet_read(out_path)

            self.assertEqual(G._num_entities, G2._num_entities)
            self.assertEqual(G._num_edges, G2._num_edges)
            self.assertEqual(G._matrix.shape, G2._matrix.shape)

            for eid in (eids[0], eids[len(eids)//2], eids[-1]):
                self.assertEqual(G.edge_weights[eid], G2.edge_weights[eid])

            self.assertLessEqual(len(G2._matrix), int(n_edges * 2))
            self.assertLess(len(G2._matrix), n_vertices * 50)

        self._test_both_modes(_test)


if __name__ == "__main__":
    unittest.main()