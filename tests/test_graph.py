# test_graph.py
import os
import sys
import unittest

import polars as pl

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import warnings

from annnet.core._records import SliceRecord
from annnet.core.graph import AnnNet

warnings.filterwarnings(
    'ignore',
    message=r'Signature .*numpy\.longdouble.*',
    category=UserWarning,
    module=r'numpy\._core\.getlimits',
)


class TestGraphBasics(unittest.TestCase):
    def setUp(self):
        self.g = AnnNet(directed=True)  # default directed

    def test_add_vertex_and_attributes(self):
        self.g.add_vertex('v1', color='red', value=3)
        self.g.add_vertex('v2')  # no attrs
        self.assertEqual(self.g.nv, 2)
        # row exists even if no attrs were passed
        self.assertIn('v2', self.g.vertex_attributes.select('vertex_id').to_series().to_list())
        # attribute accessible
        self.assertEqual(self.g.attrs.get_attr_vertex('v1', 'color'), 'red')
        self.assertEqual(self.g.attrs.get_attr_vertex('v1', 'value'), 3)

    def test_add_edge_directed_default_and_matrix_signs(self):
        eid = self.g.add_edge('a', 'b', weight=2.5, label='eab')
        self.assertTrue(self.g._is_directed_edge(eid))
        self.assertIn(eid, self.g.get_edges_by_direction(True))
        # get_edge canonical form
        S, T = self.g.get_edge(self.g.edge_to_idx[eid])
        self.assertEqual(S, frozenset({'a'}))
        self.assertEqual(T, frozenset({'b'}))
        # matrix signs: +w at source, -w at target (directed)
        ai = self.g.entity_to_idx['a']
        bi = self.g.entity_to_idx['b']
        col = self.g.edge_to_idx[eid]
        self.assertAlmostEqual(self.g._matrix[ai, col], 2.5, places=7)
        self.assertAlmostEqual(self.g._matrix[bi, col], -2.5, places=7)
        # attribute purity for edges (structural keys stripped)
        self.assertEqual(self.g.attrs.get_attr_edge(eid, 'label'), 'eab')
        self.assertIsNone(self.g.attrs.get_attr_edge(eid, 'source'))  # structural, not stored

    def test_add_edge_undirected_override(self):
        eid = self.g.add_edge('c', 'd', weight=1.0, directed=False)
        self.assertIn(eid, self.g.get_edges_by_direction(False))
        S, T = self.g.get_edge(self.g.edge_to_idx[eid])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({'c', 'd'}))
        # matrix signs: +w on both endpoints (undirected)
        ci = self.g.entity_to_idx['c']
        di = self.g.entity_to_idx['d']
        col = self.g.edge_to_idx[eid]
        self.assertAlmostEqual(self.g._matrix[ci, col], 1.0, places=7)
        self.assertAlmostEqual(self.g._matrix[di, col], 1.0, places=7)

    def test_parallel_edges_and_lookup(self):
        e1 = self.g.add_edge('p', 'q', weight=1.0)
        e2 = self.g.add_edge('p', 'q', weight=3.0, parallel='parallel')
        self.assertNotEqual(e1, e2)
        ids = self.g.get_edge_ids('p', 'q')
        self.assertCountEqual(ids, [e1, e2])
        self.assertTrue(self.g.has_edge('p', 'q'))
        self.assertTrue(self.g.has_edge('p', 'q', edge_id=e1))

    def test_copy_repairs_pair_index_queries(self):
        e1 = self.g.add_edge('p', 'q', edge_id='e1')
        e2 = self.g.add_edge('p', 'q', edge_id='e2', parallel='parallel')
        g2 = self.g.ops.copy()
        e3 = g2.add_edge('p', 'q', edge_id='e3', parallel='parallel')
        self.assertCountEqual(g2.get_edge_ids('p', 'q'), [e1, e2, e3])
        self.assertEqual(g2.has_edge('p', 'q'), (True, ['e1', 'e2', 'e3']))

    def test_has_vertex_flat_graph(self):
        self.g.add_vertex('v1')
        self.assertTrue(self.g.has_vertex('v1'))
        self.assertFalse(self.g.has_vertex('missing'))
        self.assertEqual(self.g.layers.list_aspects(), ())
        self.assertEqual(self.g.layers.list_layers(), {})

    def test_grow_helpers_are_real_methods(self):
        self.assertTrue(callable(self.g._grow_rows_to))
        self.assertTrue(callable(self.g._grow_cols_to))
        self.assertEqual(self.g._grow_rows_to.__self__, self.g)
        self.assertEqual(self.g._grow_cols_to.__self__, self.g)
        self.assertIsInstance(self.g._slices['default'], SliceRecord)

    def test_all_stored_slices_use_slice_record(self):
        self.g.slices.add_slice('L1')
        self.g.add_vertex('a', slice='L1')
        self.g.add_edge('a', 'b', edge_id='e1', slice='L1')
        self.g.slices.union_create(['default', 'L1'], 'L2')
        g2 = self.g.ops.copy()

        for data in self.g._slices.values():
            self.assertIsInstance(data, SliceRecord)
        for data in g2._slices.values():
            self.assertIsInstance(data, SliceRecord)

    def test_edge_direction_policy_is_derived_from_edge_records(self):
        eid = self.g.add_edge('a', 'b', flexible={'var': 'score', 'threshold': 1.0})
        self.assertEqual(self.g.edge_direction_policy[eid], self.g._edges[eid].direction_policy)

    def test_edge_direction_policy_setter_updates_records(self):
        eid = self.g.add_edge('a', 'b')
        policy = {'var': 'score', 'threshold': 2.0}
        self.g.edge_direction_policy = {eid: policy}
        self.assertEqual(self.g._edges[eid].direction_policy, policy)
        self.assertEqual(self.g.edge_direction_policy, {eid: policy})

    def test_edge_entity_and_vertex_edge_mode(self):
        # Create an edge that can itself be an endpoint
        e = self.g.add_edge(
            'x', 'y', edge_id='edge_ghost', as_entity=True, weight=1.2, slice='Lx', label='meta'
        )
        # edge_ghost is registered as an entity (can be endpoint)
        self.assertIn('edge_ghost', self.g.entity_to_idx)
        self.assertEqual(self.g.entity_types['edge_ghost'], 'edge')
        # edge exists and has the right weight
        self.assertAlmostEqual(self.g.edge_weights[e], 1.2, places=7)
        # attributes stored as edge attrs
        self.assertEqual(self.g.attrs.get_attr_edge('edge_ghost', 'label'), 'meta')
        # can connect another edge TO this edge
        e2 = self.g.add_edge('z', 'edge_ghost', edge_id='meta_link')
        self.assertIn('meta_link', self.g.edge_to_idx)
        self.assertEqual(self.g._edges[e].etype, 'vertex_edge')

    def test_edge_entity_placeholder_has_distinct_etype(self):
        eid = self.g.add_edge(edge_id='edge_stub', as_entity=True)
        rec = self.g._edges[eid]
        self.assertEqual(rec.etype, 'edge_placeholder')
        self.assertEqual(rec.col_idx, -1)
        self.assertIsNone(rec.src)
        self.assertIsNone(rec.tgt)

        self.g.add_edge('a', 'b', edge_id=eid, as_entity=True)
        upgraded = self.g._edges[eid]
        self.assertEqual(upgraded.etype, 'vertex_edge')
        self.assertGreaterEqual(upgraded.col_idx, 0)
        self.assertEqual(upgraded.src, 'a')
        self.assertEqual(upgraded.tgt, 'b')

    def test_hyperedge_undirected(self):
        hid = self.g.add_edge(src=['h1', 'h2', 'h3'], weight=2.0, tag='tri')
        self.assertEqual(self.g.edge_kind[hid], 'hyper')
        S, T = self.g.get_edge(self.g.edge_to_idx[hid])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({'h1', 'h2', 'h3'}))
        # matrix entries are +2.0 on all three members
        col = self.g.edge_to_idx[hid]
        for v in ['h1', 'h2', 'h3']:
            self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx[v], col], 2.0, places=7)
        # attribute present
        self.assertEqual(self.g.attrs.get_attr_edge(hid, 'tag'), 'tri')

    def test_hyperedge_directed(self):
        hid = self.g.add_edge(src=['s1', 's2'], tgt=['t1'], weight=4.0, category='flow')
        self.assertTrue(self.g.edge_directed[hid])
        S, T = self.g.get_edge(self.g.edge_to_idx[hid])
        self.assertEqual(S, frozenset({'s1', 's2'}))
        self.assertEqual(T, frozenset({'t1'}))
        col = self.g.edge_to_idx[hid]
        for v in ['s1', 's2']:
            self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx[v], col], 4.0, places=7)
        self.assertAlmostEqual(self.g._matrix[self.g.entity_to_idx['t1'], col], -4.0, places=7)
        self.assertEqual(self.g.attrs.get_attr_edge(hid, 'category'), 'flow')

    def test_slices_and_activation_and_propagation(self):
        # add slices
        self.g.slices.add_slice('L1', purpose='left')
        self.g.slices.add_slice('L2')
        self.g.slices.set_active_slice('L1')
        self.assertEqual(self.g.slices.get_active_slice(), 'L1')
        # add some vertices into current slice
        self.g.add_vertex('A')
        self.g.add_vertex('B')
        # switch slice and add C
        self.g.slices.set_active_slice('L2')
        self.g.add_vertex('C')
        # add edge with propagate=shared (only slices that have both endpoints A,B -> L1)
        e1 = self.g.add_edge(
            'A', 'B', slice='L2', propagate='shared'
        )  # placed in L2, but should propagate to L1?
        # L1 has both A,B so edge should be present in L1 as well
        self.assertIn(e1, self.g._slices['L1']['edges'])
        self.assertIn(e1, self.g._slices['L2']['edges'])
        # add edge with propagate=all for A-C (A in L1, C in L2) -> should appear in both and pull missing endpoint
        e2 = self.g.add_edge('A', 'C', slice='L2', propagate='all')
        self.assertIn(e2, self.g._slices['L1']['edges'])
        self.assertIn(e2, self.g._slices['L2']['edges'])
        self.assertIn('C', self.g._slices['L1']['vertices'])  # pulled across
        self.assertIn('A', self.g._slices['L2']['vertices'])  # pulled across

    def test_set_and_get_slice_attrs(self):
        self.g.slices.add_slice('Geo', region='EMEA')
        self.assertEqual(self.g.attrs.get_slice_attr('Geo', 'region'), 'EMEA')
        # upsert to new dtype
        self.g.attrs.set_slice_attrs('Geo', region='APAC')
        self.assertEqual(self.g.attrs.get_slice_attr('Geo', 'region'), 'APAC')

    def test_slice_info_reads_attributes_from_dataframe_ssot(self):
        self.g.add_vertex('v1', slice='Geo')
        self.g.attrs.set_slice_attrs('Geo', region='EMEA')
        info = self.g.slices.get_slice_info('Geo')
        self.assertEqual(info['attributes'], {'region': 'EMEA'})

    def test_add_slice_does_not_mutate_layer_registry(self):
        self.g.layers.set_aspects(['time'])
        self.g.layers.set_elementary_layers({'time': ['t0', 't1']})
        self.g.slices.add_slice('analysis')
        self.assertEqual(self.g.layers.list_layers('time'), ['t0', 't1'])

    def test_flatten_layers_preserves_pair_index_queries(self):
        g = AnnNet(aspects={'time': ['t1', 't2']}, directed=True)
        g.add_vertex('u', layer=('t1',))
        g.add_vertex('u', layer=('t2',))
        g.add_vertex('v', layer=('t1',))
        g.add_edge(('u', ('t1',)), ('v', ('t1',)), edge_id='e1')
        g.add_edge(('u', ('t1',)), ('v', ('t1',)), edge_id='e2', parallel='parallel')

        out = g.layers.flatten_layers()
        e3 = out.add_edge('u', 'v', edge_id='e3', parallel='parallel')

        self.assertCountEqual(out.get_edge_ids('u', 'v'), ['e1', 'e2', e3])
        found, ids = out.has_edge('u', 'v')
        self.assertTrue(found)
        self.assertCountEqual(ids, ['e1', 'e2', e3])

    def test_reverse_rebuilds_pair_indexes(self):
        self.g.add_edge('a', 'b', edge_id='e1')
        out = self.g.ops.reverse()

        self.assertFalse(out.has_edge('a', 'b')[0])
        found, ids = out.has_edge('b', 'a')
        self.assertTrue(found)
        self.assertEqual(ids, ['e1'])

    def test_subgraph_from_slice_flat_fast_path_preserves_slice_state(self):
        self.g.slices.add_slice('L1', region='EMEA')
        eid = self.g.add_edge('u', 'v', edge_id='e1', weight=5.0, slice='L1')
        self.g.attrs.set_edge_slice_attrs('L1', eid, weight=1.25)

        out = self.g.subgraph_from_slice('L1', resolve_slice_weights=True)

        self.assertEqual(out.slices.get_active_slice(), 'L1')
        self.assertEqual(set(out.vertices()), {'u', 'v'})
        self.assertEqual(set(out.edges()), {'e1'})
        self.assertAlmostEqual(out._edges['e1'].weight, 1.25)
        self.assertEqual(out.attrs.get_slice_attr('L1', 'region'), 'EMEA')
        self.assertEqual(out._slices['default']['edges'], set())
        self.assertEqual(out._slices['L1']['edges'], {'e1'})

    def test_attrs_namespace_matches_flat_api(self):
        self.g.attrs.set_graph_attribute('source', 'unit-test')
        self.g.add_vertex('v1')
        self.g.attrs.set_vertex_attrs('v1', color='blue')
        eid = self.g.add_edge('v1', 'v2', edge_id='e1')
        self.g.attrs.set_edge_attrs(eid, relation='binds')
        self.g.slices.add_slice('Lw')
        self.g.attrs.set_edge_slice_attrs('Lw', eid, weight=2.0)

        self.assertEqual(self.g.attrs.get_graph_attribute('source'), 'unit-test')
        self.assertEqual(self.g.attrs.get_attr_vertex('v1', 'color'), 'blue')
        self.assertEqual(self.g.attrs.get_attr_edge('e1', 'relation'), 'binds')
        self.assertEqual(self.g.attrs.get_edge_slice_attr('Lw', 'e1', 'weight'), 2.0)
        self.assertEqual(self.g.attrs.get_effective_edge_weight('e1', slice='Lw'), 2.0)

    def test_per_slice_weight_and_effective_weight(self):
        # ensure the slice exists first
        self.g.slices.add_slice('Lw')
        # create the edge inside slice "Lw" so per-slice attrs apply
        eid = self.g.add_edge('u', 'v', weight=5.0, slice='Lw')
        # override via edge_slice_attributes table using the EDGE ID (string)
        self.g.attrs.set_edge_slice_attrs('Lw', eid, weight=1.25, note='downweighted')
        # effective weight in Lw reflects the override
        self.assertAlmostEqual(
            self.g.attrs.get_effective_edge_weight(eid, slice='Lw'), 1.25, places=7
        )
        # asking for a non-existent slice should fall back to the global weight
        self.assertAlmostEqual(
            self.g.attrs.get_effective_edge_weight(eid, slice='NonExistent'), 5.0, places=7
        )

    def test_incident_edges(self):
        e1 = self.g.add_edge('i1', 'i2', weight=1)
        e2 = self.g.add_edge('i2', 'i3', weight=1)
        # undirected also counts on both sides
        e3 = self.g.add_edge('i2', 'i4', weight=1, directed=False)
        inc = self.g.incident_edges('i2')
        ids = {self.g.idx_to_edge[j] for j, _edge in inc}
        self.assertSetEqual(ids, {e1, e2, e3})

    def test_remove_edge_then_vertex(self):
        e = self.g.add_edge('r1', 'r2', weight=1.0, tag='tmp')
        self.g.remove_edge(e)
        self.assertNotIn(e, self.g.edge_to_idx)
        # removing a vertex also removes incident edges
        e2 = self.g.add_edge('r1', 'r3', weight=2.0)
        self.g.remove_vertex('r1')
        self.assertNotIn('r1', self.g.entity_to_idx)
        self.assertNotIn(e2, self.g.edge_to_idx)

    def test_remove_slice_and_default_slice_guard(self):
        self.g.slices.add_slice('Z')
        self.g.slices.remove_slice('Z')
        self.assertFalse(self.g.slices.has_slice('Z'))
        with self.assertRaises(ValueError):
            self.g.slices.remove_slice('default')

    def test_audit_attributes(self):
        # create mismatch intentionally
        self.g.add_vertex('a1')
        self.g.add_edge('a1', 'a2', weight=1.0)
        # add stray row in vertex_attributes (keep schema identical: only 'vertex_id')
        self.g.vertex_attributes = pl.concat(
            [
                self.g.vertex_attributes,
                pl.DataFrame({'vertex_id': ['ghost']}),
            ],
            how='vertical',
        )
        audit = self.g.attrs.audit_attributes()
        self.assertIn('ghost', audit['extra_vertex_rows'])
        self.assertIsInstance(audit['missing_edge_rows'], list)
        self.assertIsInstance(audit['invalid_edge_slice_rows'], list)

    def test_edges_views_and_counts(self):
        e = self.g.add_edge('x1', 'x2', weight=7.0, directed=False)
        self.assertEqual(self.g.ne, len(self.g.edges()))
        elist = self.g.edge_list()
        found = [row for row in elist if row[2] == e]
        self.assertEqual(len(found), 1)
        src, tgt, eid, w = found[0]
        self.assertEqual((src, tgt, eid, w), ('x1', 'x2', e, 7.0))
        # degree uses non-zero entries count
        d = self.g.degree('x1')
        self.assertGreaterEqual(d, 1)

    def test_update_existing_edge(self):
        e = self.g.add_edge('u1', 'u2', weight=2.0, directed=True)
        # Update same edge_id: flip direction flag and endpoints
        self.g.add_edge('u2', 'u3', weight=3.5, edge_id=e, directed=False)
        # Now undirected, between u2 and u3, weight 3.5
        S, T = self.g.get_edge(self.g.edge_to_idx[e])
        self.assertEqual(S, T)
        self.assertEqual(S, frozenset({'u2', 'u3'}))
        col = self.g.edge_to_idx[e]
        u2i = self.g.entity_to_idx['u2']
        u3i = self.g.entity_to_idx['u3']
        self.assertAlmostEqual(self.g._matrix[u2i, col], 3.5, places=7)
        self.assertAlmostEqual(self.g._matrix[u3i, col], 3.5, places=7)

    def test_flatten_layers_makes_graph_flat_and_preserves_structure(self):
        g = AnnNet(aspects={'condition': ['healthy', 'treated'], 'time': ['t0', 't1']})
        g.add_vertex('v1', layer=('healthy', 't0'))
        g.add_vertex('v1', layer=('treated', 't1'))
        g.add_vertex('v2', layer=('healthy', 't0'))
        g.add_edge(('v1', ('healthy', 't0')), ('v2', ('healthy', 't0')), edge_id='e_intra')
        g.add_edge(
            ('v1', ('healthy', 't0')),
            ('v1', ('treated', 't1')),
            edge_id='e_couple',
        )
        g.add_edge(src=[('v1', ('healthy', 't0')), ('v1', ('treated', 't1'))], edge_id='h1')

        out = g.layers.flatten_layers()

        self.assertIs(out, g)
        self.assertEqual(g.aspects, [])
        self.assertEqual(g.elem_layers, {})
        self.assertIn('v1', set(g.vertices()))
        self.assertIn('v2', set(g.vertices()))
        self.assertEqual(g._resolve_entity_key('v1'), ('v1', ('_',)))
        self.assertIn('e_intra', g.edge_to_idx)
        self.assertIn('e_couple', g.edge_to_idx)
        self.assertIn('h1', g.edge_to_idx)
        self.assertEqual(g._edges['e_intra'].ml_kind, None)
        self.assertEqual(g._edges['e_couple'].ml_kind, None)
        self.assertEqual(g._edges['h1'].ml_layers, None)
        g.add_vertex('isolated')
        self.assertIn('isolated', set(g.vertices()))

    def test_make_undirected_returns_self_for_chaining(self):
        g = AnnNet(directed=True)
        g.add_edge('a', 'b')
        out = g.make_undirected()
        self.assertIs(out, g)
        self.assertFalse(g.directed)

    def test_remove_orphans_after_flatten_layers(self):
        g = AnnNet(aspects={'condition': ['healthy'], 'time': ['t0']})
        g.add_vertex('v1', layer=('healthy', 't0'))
        g.add_vertex('v2', layer=('healthy', 't0'))
        g.add_edge(('v1', ('healthy', 't0')), ('v2', ('healthy', 't0')))
        g.layers.flatten_layers()
        g.add_vertex('isolated')
        removed = g.remove_orphans()
        self.assertEqual(removed, 1)
        self.assertNotIn('isolated', set(g.vertices()))

    def test_layer_edge_set_can_select_inter_hyperedge(self):
        g = AnnNet(aspects={'condition': ['healthy', 'treated']})
        g.add_vertex('a', layer=('healthy',))
        g.add_vertex('b', layer=('treated',))
        hid = g.add_edge(
            src=[('a', ('healthy',))],
            tgt=[('b', ('treated',))],
            edge_id='h_inter',
        )
        g._edges[hid].ml_kind = 'inter'
        g._edges[hid].ml_layers = (('healthy',), ('treated',))

        self.assertNotIn(hid, g.layers.layer_edge_set(('healthy',)))
        self.assertIn(hid, g.layers.layer_edge_set(('healthy',), include_inter=True))
        self.assertIn(hid, g.layers.layer_edge_set(('treated',), include_inter=True))

    def test_supra_incidence_includes_coupling_hyperedge_when_requested(self):
        g = AnnNet(aspects={'condition': ['healthy', 'treated']})
        g.add_vertex('a', layer=('healthy',))
        g.add_vertex('a', layer=('treated',))
        hid = g.add_edge(
            src=[('a', ('healthy',))],
            tgt=[('a', ('treated',))],
            edge_id='h_couple',
        )
        g._edges[hid].ml_kind = 'coupling'
        g._edges[hid].ml_layers = (('healthy',), ('treated',))

        B0, eids0, skipped0 = g.layers.supra_incidence(include_coupling=False)
        self.assertNotIn(hid, eids0)
        self.assertNotIn(hid, skipped0)

        B1, eids1, skipped1 = g.layers.supra_incidence(include_coupling=True)
        self.assertIn(hid, eids1)
        self.assertNotIn(hid, skipped1)
        self.assertEqual(B1.shape[1], B0.shape[1] + 1)


class TestErrorPaths(unittest.TestCase):
    """Error handling and boundary conditions."""

    def setUp(self):
        self.g = AnnNet(directed=True)

    def test_set_aspects_lifts_existing_flat_vertices_to_placeholder(self):
        self.g.add_vertex('v1')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.g.layers.set_aspects(['condition', 'time'])
        self.assertEqual(self.g._resolve_entity_key('v1'), ('v1', ('_', '_')))
        self.assertTrue(any('placeholder layer' in str(w.message) for w in caught))

    def test_add_vertex_without_layer_in_multilayer_uses_placeholder_and_warns(self):
        self.g.layers.set_aspects(['condition', 'time'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.g.add_vertex('v1')
        self.assertIn(('v1', ('_', '_')), self.g._entities)
        self.assertTrue(any('placeholder layer' in str(w.message) for w in caught))

    def test_has_vertex_multilayer_with_bare_id(self):
        self.g.layers.set_aspects(['condition', 'time'])
        self.g.layers.set_elementary_layers({'condition': ['healthy'], 'time': ['t0']})
        self.g.add_vertex('v1', layer=('healthy', 't0'))
        self.assertTrue(self.g.has_vertex('v1'))
        self.assertTrue(self.g.has_vertex(('v1', ('healthy', 't0'))))
        self.assertFalse(self.g.has_vertex('missing'))

    def test_resolve_entity_key_multilayer_bare_id_is_explicitly_ambiguous(self):
        self.g.layers.set_aspects(['condition', 'time'])
        self.g.layers.set_elementary_layers(
            {'condition': ['healthy', 'treated'], 'time': ['t0', 't1']}
        )
        self.g.add_vertex('v1', layer=('healthy', 't0'))
        self.g.add_vertex('v1', layer=('treated', 't1'))
        with self.assertRaisesRegex(ValueError, 'Ambiguous bare vertex_id'):
            self.g._resolve_entity_key('v1')

    def test_degree_multilayer_bare_id_requires_explicit_supra_node(self):
        self.g.layers.set_aspects(['condition', 'time'])
        self.g.layers.set_elementary_layers(
            {'condition': ['healthy', 'treated'], 'time': ['t0', 't1']}
        )
        self.g.add_vertex('v1', layer=('healthy', 't0'))
        self.g.add_vertex('v1', layer=('treated', 't1'))
        self.g.add_vertex('v2', layer=('healthy', 't0'))
        self.g.add_edge(('v1', ('healthy', 't0')), ('v2', ('healthy', 't0')))
        with self.assertRaisesRegex(ValueError, 'Ambiguous bare vertex_id'):
            self.g.degree('v1')

    def test_remove_vertex_multilayer_bare_id_requires_explicit_supra_node(self):
        self.g.layers.set_aspects(['condition', 'time'])
        self.g.layers.set_elementary_layers(
            {'condition': ['healthy', 'treated'], 'time': ['t0', 't1']}
        )
        self.g.add_vertex('v1', layer=('healthy', 't0'))
        self.g.add_vertex('v1', layer=('treated', 't1'))
        with self.assertRaisesRegex(ValueError, 'Ambiguous bare vertex_id'):
            self.g.remove_vertex('v1')

    def test_add_vertices_bulk_without_layer_in_multilayer_warns_once(self):
        self.g.layers.set_aspects(['condition', 'time'])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.g.add_vertices_bulk(['v1', 'v2', 'v3'])
        self.assertIn(('v1', ('_', '_')), self.g._entities)
        self.assertIn(('v2', ('_', '_')), self.g._entities)
        self.assertIn(('v3', ('_', '_')), self.g._entities)
        placeholder_warnings = [w for w in caught if 'placeholder layer' in str(w.message)]
        self.assertEqual(len(placeholder_warnings), 1)

    def test_add_edge_with_bare_ids_raises_on_multilayer_graph(self):
        self.g.layers.set_aspects(['condition', 'time'])
        with self.assertRaisesRegex(ValueError, 'explicit supra-node endpoints'):
            self.g.add_edge('a', 'b')

    def test_direct_aspects_property_rebuilds_all_layers_cache(self):
        self.g.aspects = ['condition', 'time']
        self.g.elem_layers = {'condition': ['healthy'], 'time': ['t0', 't1']}
        self.assertEqual(
            self.g.layers._all_layers,
            (('healthy', 't0'), ('healthy', 't1')),
        )

    def test_direct_elem_layers_property_rebuilds_all_layers_cache(self):
        self.g.aspects = ['condition', 'time']
        self.g.elem_layers = {'condition': ['healthy'], 'time': ['t0']}
        self.assertEqual(self.g.layers._all_layers, (('healthy', 't0'),))
        self.g.elem_layers = {'condition': ['treated'], 'time': ['t2', 't3']}
        self.assertEqual(
            self.g.layers._all_layers,
            (('treated', 't2'), ('treated', 't3')),
        )

    def test_sparse_cache_invalidation_keeps_get_csr_and_cache_csr_in_sync(self):
        self.g.add_vertex('a')
        self.g.add_vertex('b')
        self.g.add_edge('a', 'b', edge_id='e1')

        csr0 = self.g._get_csr()
        cache0 = self.g.cache.csr
        self.assertEqual(csr0.shape, cache0.shape)
        nnz0 = csr0.nnz

        self.g.add_vertex('c')
        self.g.add_edge('b', 'c', edge_id='e2')

        csr1 = self.g._get_csr()
        cache1 = self.g.cache.csr
        self.assertEqual(csr1.shape, cache1.shape)
        self.assertGreater(csr1.nnz, nnz0)
        self.assertIs(csr1, cache1)

    # ------------------------------------------------------------------ #
    # remove_vertex                                                        #
    # ------------------------------------------------------------------ #

    def test_remove_nonexistent_vertex_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.g.remove_vertex('does_not_exist')

    def test_remove_vertex_twice_raises_on_second(self):
        self.g.add_vertex('A')
        self.g.remove_vertex('A')
        with self.assertRaises(KeyError):
            self.g.remove_vertex('A')

    # ------------------------------------------------------------------ #
    # remove_edge                                                          #
    # ------------------------------------------------------------------ #

    def test_remove_nonexistent_edge_raises_key_error(self):
        with self.assertRaises(KeyError):
            self.g.remove_edge('ghost_edge')

    def test_remove_edge_twice_raises_on_second(self):
        self.g.add_vertex('A')
        self.g.add_vertex('B')
        eid = self.g.add_edge('A', 'B', edge_id='e1')
        self.g.remove_edge(eid)
        with self.assertRaises(KeyError):
            self.g.remove_edge(eid)

    # ------------------------------------------------------------------ #
    # add_vertex — upsert semantics                                        #
    # ------------------------------------------------------------------ #

    def test_duplicate_vertex_is_upsert_not_error(self):
        self.g.add_vertex('A', score=1.0)
        self.g.add_vertex('A', score=2.0)  # should update, not raise
        self.assertEqual(self.g.nv, 1)
        self.assertEqual(self.g.attrs.get_attr_vertex('A', 'score'), 2.0)

    # ------------------------------------------------------------------ #
    # add_edge — auto-creates missing vertices                             #
    # ------------------------------------------------------------------ #

    def test_add_edge_auto_creates_missing_vertices(self):
        # Neither X nor Y exist yet
        eid = self.g.add_edge('X', 'Y', edge_id='auto_e')
        self.assertIn('X', self.g.entity_to_idx)
        self.assertIn('Y', self.g.entity_to_idx)
        self.assertIn('auto_e', self.g.edge_to_idx)

    # ------------------------------------------------------------------ #
    # add_edge — invalid arguments                                         #
    # ------------------------------------------------------------------ #

    def test_add_edge_non_numeric_weight_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.g.add_edge('A', 'B', weight='heavy')

    def test_add_edge_invalid_propagate_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.g.add_edge('A', 'B', propagate='invalid_mode')

    # ------------------------------------------------------------------ #
    # remove_slice guard                                                   #
    # ------------------------------------------------------------------ #

    def test_remove_default_slice_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.g.slices.remove_slice('default')


if __name__ == '__main__':
    unittest.main()
