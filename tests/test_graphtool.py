# tests/test_graphtool_adapter.py
import os
import sys
import unittest
import warnings

warnings.filterwarnings(
    'ignore',
    message=r'Signature .*numpy\.longdouble.*',
    category=UserWarning,
    module=r'numpy\._core\.getlimits',
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from annnet.core.graph import AnnNet
from annnet.adapters import graphtool_adapter
from annnet.adapters.graphtool_adapter import from_graphtool, to_graphtool

from .conftest import build_adapter_graph as _build_graph

HAS_GT = graphtool_adapter.gt is not None
_BUILD_GRAPH = _build_graph


@unittest.skipUnless(HAS_GT, 'graph-tool adapter or dependency not available')
class TestGraphToolAdapter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if _BUILD_GRAPH is None:
            raise unittest.SkipTest('No _build_graph() found')

    def test_to_gt_export_basic(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIsNotNone(gtG)
        self.assertIsInstance(manifest, dict)

        self.assertIn('version', manifest)
        self.assertIn('graph', manifest)
        self.assertIn('vertices', manifest)
        self.assertIn('edges', manifest)
        self.assertIn('slices', manifest)
        self.assertIn('multilayer', manifest)

        self.assertEqual(gtG.num_vertices(), 3)
        self.assertGreaterEqual(gtG.num_edges(), 2)

    def test_roundtrip_preserves_structure(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        g2 = from_graphtool(gtG, manifest)

        self.assertEqual(g2.nv, g.nv)

        self.assertIn('A', g2.vertices())
        self.assertIn('B', g2.vertices())
        self.assertIn('C', g2.vertices())

    def test_manifest_preserves_slices(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        slices_data = manifest.get('slices', {})
        self.assertIn('data', slices_data)

        slice_names = list(slices_data.get('data', {}).keys())
        self.assertIn('Lw', slice_names)
        self.assertIn('L0', slice_names)

    def test_manifest_preserves_hyperedges(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        edges_meta = manifest.get('edges', {})
        hyperedges = edges_meta.get('hyperedges', {})

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
        self.assertIn('Lw', slices)

        edges_in_lw = list(g2.get_slice_edges('Lw'))
        self.assertGreater(len(edges_in_lw), 0)

        eid = edges_in_lw[0]
        w_eff = g2.get_effective_edge_weight(eid, slice='Lw')
        self.assertEqual(w_eff, 5.0)

    def test_vertex_properties_in_graph(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIn('id', gtG.vp)

        vertex_ids = [gtG.vp['id'][v] for v in gtG.vertices()]
        self.assertIn('A', vertex_ids)
        self.assertIn('B', vertex_ids)
        self.assertIn('C', vertex_ids)

    def test_edge_properties_in_graph(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        self.assertIn('id', gtG.ep)
        self.assertIn('weight', gtG.ep)

        for e in gtG.edges():
            weight = gtG.ep['weight'][e]
            self.assertIsInstance(weight, float)
            self.assertGreater(weight, 0)

    def test_directed_flag_preserved(self):
        g_dir = AnnNet(directed=True)
        g_dir.add_vertex('X')
        g_dir.add_vertex('Y')
        g_dir.add_edge('X', 'Y')

        gtG_dir, manifest_dir = to_graphtool(g_dir)
        self.assertTrue(gtG_dir.is_directed())
        self.assertTrue(manifest_dir['graph']['directed'])

        g_undir = AnnNet(directed=False)
        g_undir.add_vertex('X')
        g_undir.add_vertex('Y')
        g_undir.add_edge('X', 'Y')

        gtG_undir, manifest_undir = to_graphtool(g_undir)
        self.assertFalse(gtG_undir.is_directed())
        self.assertFalse(manifest_undir['graph']['directed'])

    def test_vertex_attributes_roundtrip(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        if hasattr(g2, 'vertex_attributes') and g2.vertex_attributes is not None:
            v_attrs = g2.vertex_attributes
            if hasattr(v_attrs, 'to_dicts'):
                rows = list(v_attrs.to_dicts())
                vertex_ids = [r.get('vertex_id') for r in rows]
                self.assertIn('A', vertex_ids)

    def test_edge_attributes_roundtrip(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)
        g2 = from_graphtool(gtG, manifest)

        if hasattr(g2, 'edge_attributes') and g2.edge_attributes is not None:
            e_attrs = g2.edge_attributes
            self.assertGreater(len(e_attrs), 0)

    def test_without_manifest_loses_hyperedges(self):
        g = _BUILD_GRAPH()
        gtG, manifest = to_graphtool(g)

        g2 = from_graphtool(gtG, manifest=None)

        self.assertEqual(g2.nv, 3)
        self.assertLess(g2.ne, g.ne)

    def test_to_gt_projects_supra_endpoints_and_exports_edge_attr_types(self):
        g = AnnNet(directed=True)
        g.add_vertex('A')
        g.add_vertex('B')
        g.add_edge('A', 'B', edge_id='e_supra', weight=1.25)
        g.add_edge(edge_id='edge_entity', as_entity=True)
        g.set_edge_attrs(
            'e_supra',
            count=7,
            score=2.5,
            label='edge-label',
            missing=None,
        )

        g._edges['e_supra'].src = ('A', ('t1',))
        g._edges['e_supra'].tgt = ('B', ('t1',))

        gtG, manifest = to_graphtool(g)

        self.assertEqual(gtG.num_vertices(), 2)
        self.assertEqual(gtG.num_edges(), 1)
        self.assertIn('count', gtG.ep)
        self.assertIn('score', gtG.ep)
        self.assertIn('label', gtG.ep)

        edge = next(gtG.edges())
        self.assertEqual(gtG.ep['id'][edge], 'e_supra')
        self.assertEqual(gtG.ep['count'][edge], 7)
        self.assertAlmostEqual(gtG.ep['score'][edge], 2.5)
        self.assertEqual(gtG.ep['label'][edge], 'edge-label')
        self.assertEqual(manifest['edges']['definitions']['e_supra'][0], ('A', ('t1',)))

    def test_from_gt_without_properties_uses_numeric_ids_and_defaults(self):
        gt = graphtool_adapter.gt
        gtG = gt.Graph(directed=False)
        v0 = gtG.add_vertex()
        v1 = gtG.add_vertex()
        gtG.add_edge(v0, v1)

        g = from_graphtool(gtG, manifest=None)

        self.assertFalse(g.directed)
        self.assertEqual(set(g.vertices()), {'0', '1'})
        self.assertEqual(g.ne, 1)
        eid = next(iter(g.edge_weights.keys()))
        self.assertEqual(g.edge_weights[eid], 1.0)

    def test_from_gt_manifest_rehydrates_optional_sections(self):
        gt = graphtool_adapter.gt
        gtG = gt.Graph(directed=True)
        v0 = gtG.add_vertex()
        v1 = gtG.add_vertex()

        vp_id = gtG.new_vertex_property('string')
        vp_id[v0] = 'A'
        vp_id[v1] = 'B'
        gtG.vp['id'] = vp_id

        edge = gtG.add_edge(v0, v1)
        ep_id = gtG.new_edge_property('string')
        ep_weight = gtG.new_edge_property('double')
        ep_id[edge] = 'e1'
        ep_weight[edge] = 1.0
        gtG.ep['id'] = ep_id
        gtG.ep['weight'] = ep_weight

        manifest = {
            'graph': {'attributes': {'name': 'manifested'}},
            'vertices': {
                'attributes': [{'vertex_id': 'A', 'color': 'red'}],
                'types': {'A': 'vertex', 'B': 'vertex'},
            },
            'edges': {
                'attributes': [{'edge_id': 'e1', 'kind': 'binary'}],
                'weights': {'e1': 4.0, 'missing': 7.0},
                'directed': {'e1': False, 'missing': True},
                'direction_policy': {'e1': {'mode': 'fixed'}},
                'hyperedges': {
                    'e1': {'directed': True, 'head': ['A'], 'tail': ['B']},
                    'missing_h': {'directed': False, 'members': ['A', 'B']},
                },
                'kivela': {
                    'edge_kind': {'e1': 'inter', 'missing': 'intra'},
                    'edge_layers': {'e1': {'kind': 'single', 'layers': [['t1']]}},
                },
            },
            'slices': {
                'data': {
                    'sdata': {
                        'vertices': ['A'],
                        'edges': ['e1'],
                        'attributes': {'label': 'slice'},
                    }
                },
                'slice_attributes': [{'slice_id': 'sattr', 'kind': 'manual'}],
                'edge_slice_attributes': [
                    {
                        'slice_id': 'edge_slice_new',
                        'edge_id': 'e1',
                        'lid': None,
                        'edge': None,
                        'weight': 2.0,
                    },
                    {
                        'slice_id': None,
                        'edge_id': None,
                        'lid': 'edge_slice_lid',
                        'edge': 'e1',
                        'weight': None,
                    },
                    {
                        'slice_id': None,
                        'edge_id': 'e1',
                        'lid': None,
                        'edge': None,
                        'weight': None,
                    },
                ],
            },
            'multilayer': {
                'aspects': ['time'],
                'elem_layers': {'time': ['t1']},
                'aspect_attrs': {'time': {'unit': 'day'}},
                'VM': [{'node': 'A', 'layer': ['t1']}],
                'edge_kind': {'e1': 'intra', 'missing': 'inter'},
                'edge_layers': {'e1': {'kind': 'single', 'layers': [['t1']]}},
                'node_layer_attrs': [{'node': 'A', 'layer': ['t1'], 'attrs': {'state': 'on'}}],
                'layer_tuple_attrs': [{'layer': ['t1'], 'attrs': {'color': 'blue'}}],
                'layer_attributes': [{'aspect': 'time', 'layer': 't1', 'note': 'primary'}],
            },
        }

        g = from_graphtool(gtG, manifest)

        self.assertEqual(g.graph_attributes['name'], 'manifested')
        self.assertEqual(g._edges['e1'].weight, 4.0)
        self.assertTrue(g._edges['e1'].directed)
        self.assertEqual(g._edges['e1'].etype, 'hyper')
        self.assertEqual(g._edges['e1'].src, ['A'])
        self.assertEqual(g._edges['e1'].tgt, ['B'])
        self.assertIn('e1', g.edge_direction_policy)
        self.assertEqual(g.edge_layers['e1'], ('t1',))

        self.assertIn('sdata', g._slices)
        self.assertIn('edge_slice_new', g._slices)
        self.assertIn('edge_slice_lid', g._slices)
        self.assertIn('e1', g._slices['edge_slice_new']['edges'])
        self.assertIn('e1', g._slices['edge_slice_lid']['edges'])

        self.assertEqual(g.aspects, ['time'])
        self.assertEqual(g.elem_layers, {'time': ['t1']})
        self.assertEqual(g._aspect_attrs['time']['unit'], 'day')
        self.assertIn(('A', ('t1',)), g._VM)
        self.assertEqual(g._state_attrs[('A', ('t1',))]['state'], 'on')
        self.assertEqual(g._layer_attrs[('t1',)]['color'], 'blue')
        self.assertGreater(g.layer_attributes.height, 0)


if __name__ == '__main__':
    unittest.main()
