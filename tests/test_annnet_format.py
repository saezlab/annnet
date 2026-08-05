# tests/test_annnet_format.py
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath('.'))
# Test imports (package layout)
import numpy as np
import polars as pl
import zarr

from annnet.core.graph import AnnNet
from annnet.io._archive import _read_archive
from annnet.io.annnet_format import read as annnet_read
from annnet.io.annnet_format import write as annnet_write
from annnet.core import _structure as S


class TestAnnNetIO(unittest.TestCase):
    def setUp(self):
        # Build a tiny directed graph with a slice + hyperedge
        G = AnnNet(directed=True)

        # Vertices (two in slice1)
        G.add_vertices('v1', slice='slice1')
        G.add_vertices('v2', slice='slice1')
        G.add_vertices('v3')
        G.add_vertices('v4')

        # Edges
        G.add_edges('v1', 'v2', edge_id='e1', weight=1.5)
        G.add_edges('v2', 'v3', edge_id='e2', weight=2.0)
        G.add_edges('v3', 'v4', edge_id='e3', weight=0.5)

        # Hyperedge (undirected)
        G.add_edges(src=['v1', 'v2', 'v3'], edge_id='h1', weight=3.0)

        # Some unstructured metadata (will go to uns/)
        G.graph_attributes['project'] = 'unittest'
        G.graph_attributes['tags'] = ['io', 'annnet']

        # Add a nested history row to ensure audit/JSON stringify path is exercised
        G._history.append(
            {
                'ts': '2025-10-23T00:00:00Z',
                'action': 'create',
                'payload': {'nested': {'x': [1, 2, 3]}},
                'notes': ['a', 'b'],
                'arr': np.array([1, 2, 3]),
                'maybe_empty': {},
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
    def _roundtrip(self, use_archive=False, **kwargs):
        if use_archive:
            out_path = Path(self.tmpdir) / 'test_graph.annnet'
        else:
            out_path = Path(self.tmpdir) / 'test_graph_dir'

        annnet_write(self.G, out_path, compression='zstd', overwrite=True, **kwargs)
        G2 = annnet_read(out_path)
        return G2, out_path

    def _test_both_modes(self, test_func):
        """Run test in both directory and archive mode."""
        for mode_name, use_archive in [('directory', False), ('archive', True)]:
            with self.subTest(mode=mode_name):
                test_func(use_archive)

    def _get_root(self, out_path, use_archive):
        """Get root directory for checks (extract if archive)."""
        if use_archive:
            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.suffix, '.annnet')
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
            self.assertEqual(S.entity_count(self.G), S.entity_count(G2))
            self.assertEqual(len(S.edge_ids(self.G)), len(S.edge_ids(G2)))
            self.assertEqual(set(self.G._slices.keys()), set(G2._slices.keys()))
            # edge weights preserved
            for ref in S.iter_edges(self.G, include_placeholders=True):
                self.assertAlmostEqual(ref.weight, S.edge_ref(G2, ref.id).weight, places=5)

            # Hyperedges preserved
            hyper_count = sum(1 for ref in S.iter_edges(G2) if ref.kind == S.HYPER)
            self.assertGreater(hyper_count, 0)

            # Rows and columns match
            self.assertEqual(S.entity_keys(self.G), S.entity_keys(G2))
            self.assertEqual(S.edge_ids(self.G), S.edge_ids(G2))

            # Edge directedness preserved
            for ref in S.iter_edges(self.G, include_placeholders=True):
                self.assertEqual(ref.directed, S.edge_ref(G2, ref.id).directed)

            # Multilayer edge kind dict preserved
            self.assertEqual(self.G.edge_kind, G2.edge_kind)

            # slices: same edge sets, vertex sets
            for lid in self.G._slices:
                self.assertEqual(self.G._slices[lid]['vertices'], G2._slices[lid]['vertices'])
                self.assertEqual(self.G._slices[lid]['edges'], G2._slices[lid]['edges'])
                self.assertEqual(
                    self.G.slice_edge_weights.get(lid, {}), G2.slice_edge_weights.get(lid, {})
                )

        self._test_both_modes(_test)

    def test_manifest_and_layout(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            self.assertTrue((root / 'manifest.json').exists())

            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['format'], 'annnet')
            self.assertIn('counts', manifest)
            self.assertEqual(manifest['directed'], True)
            self.assertIn('compression', manifest)
            self.assertIn('encoding', manifest)

            # Core layout
            self.assertTrue((root / 'structure').exists())
            self.assertTrue((root / 'tables').exists())
            self.assertTrue((root / 'slices').exists())
            self.assertTrue((root / 'audit').exists())
            self.assertTrue((root / 'uns').exists())

        self._test_both_modes(_test)

    def test_write_exposes_matrix_choice(self):
        """`matrix` is a user-facing choice, so it must stay an explicit keyword on
        the method people call — not absorbed into **kwargs, where it is invisible."""
        import inspect

        sig = inspect.signature(AnnNet.write)
        self.assertIn('matrix', sig.parameters)
        param = sig.parameters['matrix']
        self.assertEqual(param.default, False)
        self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIn('matrix : bool', inspect.getdoc(AnnNet.write))

        # ...and it reaches the writer from the method, not just the io function.
        for matrix, expect_zarr in ((False, False), (True, True)):
            out = Path(self.tmpdir) / f'choice_{matrix}.annnet'
            self.G.write(out, matrix=matrix, overwrite=True)
            root = self._get_root(out, True)
            self.assertEqual((root / 'structure' / 'incidence.zarr').exists(), expect_zarr)

    def test_coeffs_are_stored_data(self):
        """An explicit column is not derivable from the weight, so it has to
        round-trip without the matrix — exactly, and repeatedly."""
        coeffs = {'v1': -2.0, 'v2': 3.0}
        self.G.set_edge_coeffs('e1', coeffs)
        want = dict(S.edge_coefficients(self.G, 'e1'))

        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # Persisted as its own table, and the derived cache is not written.
            self.assertTrue((root / 'structure' / 'edge_coeffs.parquet').exists())
            self.assertFalse((root / 'structure' / 'incidence.zarr').exists())

            self.assertEqual(S.edge_coefficients(G2, 'e1'), want)
            # ...and the matrix the store rebuilds is the one we started with.
            self.assertEqual(
                sorted(G2._matrix.tocoo().data.tolist()),
                sorted(self.G._matrix.tocoo().data.tolist()),
            )

        self._test_both_modes(_test)

    def test_coeffs_survive_repeated_roundtrips(self):
        """Regression: read() once dropped coeffs, so the *second* write silently
        emitted a matrix built from the +/- weight default."""
        self.G.set_edge_coeffs('e1', {'v1': -2.0, 'v2': 3.0})
        want = dict(S.edge_coefficients(self.G, 'e1'))

        G = self.G
        for i in range(3):
            out = Path(self.tmpdir) / f'cycle{i}.annnet'
            annnet_write(G, out, overwrite=True)
            G = annnet_read(out)
            self.assertEqual(S.edge_coefficients(G, 'e1'), want)

    def test_coeffs_keep_float64_precision(self):
        """The matrix is float32 and the stored column is not. Persisting the
        column must not round-trip it through the precision of the cache."""
        self.G.set_edge_coeffs('e1', {'v1': -0.1, 'v2': 1.0 / 3.0})
        want = dict(S.edge_coefficients(self.G, 'e1'))
        G2, _ = self._roundtrip()
        self.assertEqual(S.edge_coefficients(G2, 'e1'), want)

    def test_plain_graph_stores_neither_matrix_nor_coeffs(self):
        """No explicit coeffs anywhere: nothing to persist, nothing to restore.
        0.1 is not float32-exact, which previously made every column look like
        stoichiometry."""
        G = AnnNet(directed=True)
        G.add_vertices('a')
        G.add_vertices('b')
        G.add_edges('a', 'b', edge_id='e', weight=0.1)
        out = Path(self.tmpdir) / 'plain.annnet'
        annnet_write(G, out, overwrite=True)
        G2 = annnet_read(out)
        self.assertIsNone(S.edge_coefficients(G2, 'e'))
        self.assertFalse((self._get_root(out, True) / 'structure' / 'incidence.zarr').exists())

    def test_a_file_older_than_the_coefficient_table_recovers_them_from_its_matrix(self):
        """The one thing a persisted matrix is still read for.

        A file written before ``edge_coeffs.parquet`` existed records an
        explicit column nowhere else, so the load takes it from the matrix the
        file holds. Deleting that table is what makes a file of this age.
        """
        coeffs = {'v1': -2.0, 'v2': 3.0}
        self.G.set_edge_coeffs('e1', coeffs)
        out = Path(self.tmpdir) / 'legacy'
        annnet_write(self.G, out, overwrite=True, matrix=True)
        (out / 'structure' / 'edge_coeffs.parquet').unlink()

        G2 = annnet_read(out)
        self.assertEqual(S.edge_coefficients(G2, 'e1'), coeffs)

    def test_matrix_true_writes_a_matrix_the_graph_does_not_read_back(self):
        """The matrix is derived, so a persisted one changes nothing on load.

        It is written because a file older than the coefficient table records an
        explicit column nowhere else, and the load recovers those from it. A
        graph read from a file with one says what a graph read from a file
        without one says.
        """
        self.G.set_edge_coeffs('e1', {'v1': -2.0, 'v2': 3.0})
        want = dict(S.edge_coefficients(self.G, 'e1'))
        loaded = {}
        for matrix in (False, True):
            out = Path(self.tmpdir) / f'cached_{matrix}.annnet'
            annnet_write(self.G, out, overwrite=True, matrix=matrix)
            root = self._get_root(out, True)
            self.assertEqual((root / 'structure' / 'incidence.zarr').exists(), matrix)
            G2 = annnet_read(out)
            self.assertEqual(S.edge_coefficients(G2, 'e1'), want)
            loaded[matrix] = G2.X().toarray()
        self.assertTrue((loaded[False] == loaded[True]).all())

    def test_zarr_incidence_group(self):
        # The matrix is a derived cache, persisted only on matrix=True; this test is
        # about the zarr encoding, so ask for it explicitly rather than relying on
        # coeffs to trigger the write.
        self.G.set_edge_coeffs('e1', {'v1': -1.0, 'v2': 1.0})

        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive, matrix=True)
            root = self._get_root(out_path, use_archive)

            inc = root / 'structure' / 'incidence.zarr'
            self.assertTrue(inc.exists())

            # Open Zarr v3 group and validate arrays + attrs
            grp = zarr.open_group(str(inc), mode='r')
            # arrays live as subdirs; zarr v3 exposes them by name
            self.assertIn('row', grp.array_keys())
            self.assertIn('col', grp.array_keys())
            self.assertIn('data', grp.array_keys())

            row = grp['row'][:]
            col = grp['col'][:]
            dat = grp['data'][:]
            shape = tuple(grp.attrs['shape'])

            # Shapes/dtypes (dtype implied by writer: int32/int32/float32)
            self.assertEqual(row.dtype, np.int32)
            self.assertEqual(col.dtype, np.int32)
            self.assertEqual(dat.dtype, np.float32)
            self.assertEqual(shape, self.G._matrix.shape)

            # COO consistency: same length across row/col/data
            self.assertEqual(len(row), len(col))
            self.assertEqual(len(row), len(dat))

        self._test_both_modes(_test)

    def test_incidence_omitted_without_coeffs(self):
        # A graph with no explicit coeffs must NOT persist the incidence matrix
        # (it is rebuilt from records on load) yet round-trip it exactly.
        def _test(use_archive):
            expected = self.G._matrix.tocsr()
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)
            self.assertFalse((root / 'structure' / 'incidence.zarr').exists())
            got = G2._matrix.tocsr()
            self.assertEqual(expected.shape, got.shape)
            self.assertEqual((expected != got).nnz, 0)

        self._test_both_modes(_test)

    def test_overwrite_semantics(self):
        out = Path(self.tmpdir) / 'test_overwrite'
        # first write
        annnet_write(self.G, out, compression='zstd', overwrite=True)
        # second write without overwrite should fail
        with self.assertRaises(FileExistsError):
            annnet_write(self.G, out, compression='zstd', overwrite=False)
        # now allow overwrite
        annnet_write(self.G, out, compression='zstd', overwrite=True)

    def test_write_read_kivela_layers(self):
        def _test(use_archive):
            """Test roundtrip of Kivela multilayer structures."""
            # 1. Setup Kivela Multilayer Data on the existing graph
            self.G.aspects = ['time', 'transport']
            self.G.elem_layers = {'time': ['t1', 't2'], 'transport': ['bus', 'train']}

            # Vertex Presence: (u, layer_tuple)
            self.G._restore_supra_nodes(
                {
                    ('v1', ('t1', 'bus')),
                    ('v2', ('t1', 'bus')),
                    ('v2', ('t2', 'train')),
                }
            )

            # Edge layers and kinds, through the maps the graph exposes so that
            # every store of the graph learns about the change.
            if self.G.has_edge(edge_id='e1'):
                self.G.edge_layers['e1'] = ('t1', 'bus')
                self.G.edge_kind['e1'] = 'intra'
            if self.G.has_edge(edge_id='e2'):
                self.G.edge_layers['e2'] = (('t1', 'bus'), ('t2', 'train'))
                self.G.edge_kind['e2'] = 'inter'

            # Attributes
            self.G.layers._aspect_attrs = {'time': {'unit': 'seconds'}}
            self.G.layers._layer_attrs = {('t1', 'bus'): {'cost': 10}}
            self.G.layers._state_attrs = {('v1', ('t1', 'bus')): {'status': 'active'}}

            # Elementary layer attributes (Polars DataFrame)
            self.G.layer_attributes = pl.DataFrame(
                [
                    {'layer_id': 'time_t1', 'desc': 'Morning'},
                    {'layer_id': 'transport_bus', 'desc': 'Public Bus'},
                ]
            )

            # 2. Roundtrip
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # 3. Verify Restoration
            # Metadata
            self.assertEqual(G2.aspects, ['time', 'transport'])
            self.assertEqual(G2.elem_layers, self.G.elem_layers)

            # Vertex Presence — _VM now includes basal flat-graph entries too
            self.assertGreaterEqual(len(G2._VM), 3)
            self.assertIn(('v1', ('t1', 'bus')), G2._VM)
            self.assertIn(('v2', ('t2', 'train')), G2._VM)
            self.assertEqual(S.entity_keys(self.G), S.entity_keys(G2))

            # Edge Layers & Kinds
            self.assertEqual(G2.edge_layers['e1'], ('t1', 'bus'))
            self.assertEqual(G2.edge_kind['e1'], 'intra')

            # Verify Inter-layer tuple of tuples is restored correctly
            self.assertEqual(G2.edge_layers['e2'], (('t1', 'bus'), ('t2', 'train')))
            self.assertEqual(G2.edge_kind['e2'], 'inter')

            # Attributes
            self.assertEqual(G2.layers._aspect_attrs['time']['unit'], 'seconds')
            self.assertEqual(G2.layers._layer_attrs[('t1', 'bus')]['cost'], 10)
            self.assertEqual(G2.layers._state_attrs[('v1', ('t1', 'bus'))]['status'], 'active')

            # Verify DataFrame attributes
            self.assertFalse(G2.layer_attributes.is_empty())
            row = G2.layer_attributes.filter(pl.col('layer_id') == 'time_t1').to_dicts()[0]
            self.assertEqual(row['desc'], 'Morning')

            # Verify Manifest Update
            manifest = json.loads((root / 'manifest.json').read_text())
            self.assertEqual(manifest['counts']['aspects'], 2)

        self._test_both_modes(_test)

    def test_slices_registry_and_memberships(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            slices_dir = root / 'slices'
            self.assertTrue((slices_dir / 'registry.parquet').exists())
            self.assertTrue((slices_dir / 'vertex_memberships.parquet').exists())
            self.assertTrue((slices_dir / 'edge_memberships.parquet').exists())

            reg = pl.read_parquet(slices_dir / 'registry.parquet')
            vmem = pl.read_parquet(slices_dir / 'vertex_memberships.parquet')
            emem = pl.read_parquet(slices_dir / 'edge_memberships.parquet')

            self.assertGreaterEqual(reg.height, 1)
            self.assertIn('slice_id', reg.columns)
            self.assertNotIn('attributes', reg.columns)

            # slice1 must have at least v1,v2
            vset = set(vmem.filter(pl.col('slice_id') == 'slice1')['vertex_id'].to_list())
            self.assertTrue({'v1', 'v2'}.issubset(vset))

            # edges exist in memberships as well
            self.assertIn('edge_id', emem.columns)
            self.assertIn('weight', emem.columns)

        self._test_both_modes(_test)

    def test_hyperedge_definitions_parquet(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            p = root / 'structure' / 'hyperedge_definitions.parquet'
            self.assertTrue(p.exists())
            df = pl.read_parquet(p)
            self.assertIn('edge_id', df.columns)
            self.assertIn('directed', df.columns)
            # at least one of members/head/tail exists (depending on directed flag)
            self.assertTrue(any(c in df.columns for c in ('members', 'head', 'tail')))

        self._test_both_modes(_test)

    def test_audit_and_uns_written(self):
        def _test(use_archive):
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # audit: history.parquet should exist and mixed nested columns converted to Utf8(JSON)
            hist = root / 'audit' / 'history.parquet'
            self.assertTrue(hist.exists())
            hdf = pl.read_parquet(hist)

            # payload/notes/arr/maybe_empty should be present (stringified) if they existed
            cols = set(hdf.columns)
            # Some columns might be absent if the schema was inferred differently,
            # so only check types for those that exist.
            for candidate in ('payload', 'notes', 'arr', 'maybe_empty'):
                if candidate in cols:
                    self.assertEqual(hdf.schema[candidate], pl.Utf8)

            # uns: graph_attributes.json
            gattr = root / 'uns' / 'graph_attributes.json'
            self.assertTrue(gattr.exists())
            attrs = json.loads(gattr.read_text())
            self.assertEqual(attrs.get('project'), 'unittest')
            self.assertEqual(attrs.get('tags'), ['io', 'annnet'])

        self._test_both_modes(_test)

    def test_read_missing_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            annnet_read(Path(self.tmpdir) / 'does_not_exist.annnet')

    def test_write_read_kivela_empty_state(self):
        def _test(use_archive):
            """Test multilayer graph with aspects but NO presence/edges (Edge case)."""
            # 1. Setup minimal multilayer metadata
            self.G.aspects = ['time']
            self.G.elem_layers = {'time': ['t1', 't2']}

            # Ensure multilayer state is empty for this test
            self.G.entity_to_idx = {}
            self.G.edge_layers = dict.fromkeys(self.G.edge_layers)
            self.G.layers._layer_attrs = {}

            # 2. Roundtrip
            G2, out_path = self._roundtrip(use_archive=use_archive)
            root = self._get_root(out_path, use_archive)

            # 3. Assertions
            # Aspects preserved?
            self.assertEqual(G2.aspects, ['time'])
            # VM is empty?
            self.assertEqual(len(G2._VM), 0)
            # Verify the file was actually written (the empty schema parquet)
            self.assertTrue((root / 'layers' / 'vertex_presence.parquet').exists())
            # Verify attribute files were NOT written (optimization check)
            self.assertFalse((root / 'layers' / 'tuple_layer_attributes.parquet').exists())

        self._test_both_modes(_test)

    def test_write_read_large_sparse_graph(self):
        def _test(use_archive):
            n_vertices = 10_000
            n_edges = 100_000

            G = AnnNet(directed=True)

            G.add_vertices({'vertex_id': f'v{i}'} for i in range(n_vertices))

            bulk = []
            for i in range(n_edges):
                bulk.append(
                    {
                        'source': f'v{i % n_vertices}',
                        'target': f'v{(i * 37) % n_vertices}',
                        'weight': float(i % 7),
                        'edge_type': 'regular',
                    }
                )

            eids = G.add_edges(bulk)

            if use_archive:
                out_path = Path(self.tmpdir) / 'large_graph.annnet'
            else:
                out_path = Path(self.tmpdir) / 'large_graph_dir'

            annnet_write(G, out_path, compression='zstd', overwrite=True)
            G2 = annnet_read(out_path)

            self.assertEqual(S.entity_count(G), S.entity_count(G2))
            self.assertEqual(len(S.edge_ids(G)), len(S.edge_ids(G2)))
            self.assertEqual(G._matrix.shape, G2._matrix.shape)

            for eid in (eids[0], eids[len(eids) // 2], eids[-1]):
                self.assertEqual(S.edge_ref(G, eid).weight, S.edge_ref(G2, eid).weight)

            # nnz, not len(): the incidence cache is CSR (as for a freshly-built
            # graph) — len() is only defined on the legacy DOK format.
            self.assertLessEqual(G2._matrix.nnz, int(n_edges * 2))
            self.assertLess(G2._matrix.nnz, n_vertices * 50)

        self._test_both_modes(_test)

    def test_slice_attributes_roundtrip_from_dataframe_ssot(self):
        def _test(use_archive):
            self.G.attrs.set_slice_attrs('slice1', region='EMEA', cohort='A')
            G2, _ = self._roundtrip(use_archive=use_archive)
            info = G2.slices.info('slice1')
            self.assertEqual(info['attributes'], {'region': 'EMEA', 'cohort': 'A'})

        self._test_both_modes(_test)

    def test_roundtrip_repairs_pair_index_queries(self):
        def _test(use_archive):
            self.G.add_edges('v1', 'v2', edge_id='e_parallel', parallel='parallel')
            G2, _ = self._roundtrip(use_archive=use_archive)
            G2.add_edges('v1', 'v2', edge_id='e_after_read', parallel='parallel')

            self.assertCountEqual(G2.get_edge_ids('v1', 'v2'), ['e1', 'e_parallel', 'e_after_read'])
            found, ids = G2.has_edge('v1', 'v2')
            self.assertTrue(found)
            self.assertCountEqual(ids, ['e1', 'e_parallel', 'e_after_read'])

        self._test_both_modes(_test)


if __name__ == '__main__':
    unittest.main()
