import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from annnet.core.graph import AnnNet

HAS_IG = True
try:
    import igraph as ig  # noqa: F401
except Exception:
    HAS_IG = False


def build_small():
    G = AnnNet()
    G.add_vertex('a')
    G.add_vertex('b')
    G.add_vertex('c')
    G.add_edge('a', 'b', weight=3.0)
    G.add_edge('b', 'c', weight=2.0)
    return G


def build_parallel():
    G = AnnNet()
    G.add_vertex('x')
    G.add_vertex('y')
    G.add_edge('x', 'y', weight=10.0)
    G.add_edge('x', 'y', weight=1.0)
    return G


class TestIGBackendAccessor(unittest.TestCase):
    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_backend_basic(self):
        G = build_small()
        igG = G.ig.backend()
        self.assertIsInstance(igG, ig.Graph)
        self.assertEqual(igG.vcount(), 3)
        self.assertEqual(igG.ecount(), 2)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_shortest_path_length_labels(self):
        G = build_small()
        d = G.ig.distances(source='a', target='c', weights='weight')
        self.assertAlmostEqual(d[0][0], 5.0)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_shortest_path_int_ids(self):
        G = build_small()
        igG = G.ig.backend()
        names = igG.vs['name']
        a_id = names.index('a')
        c_id = names.index('c')
        d = G.ig.distances(source=a_id, target=c_id, weights='weight')
        self.assertAlmostEqual(d[0][0], 5.0)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_bad_vertex_label(self):
        G = build_small()
        with self.assertRaises((KeyError, ValueError)):
            G.ig.distances(source='ZZZ', target='a', weights='weight')

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_backend_cache_reuse(self):
        G = build_small()
        igG1 = G.ig.backend()
        igG2 = G.ig.backend()
        self.assertIs(igG1, igG2)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_backend_cache_invalidate_on_version(self):
        G = build_small()
        igG1 = G.ig.backend()
        G.add_edge('a', 'c')
        igG2 = G.ig.backend()
        self.assertIsNot(igG1, igG2)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_simple_graph_edge_collapse(self):
        G = build_parallel()
        igG = G.ig.backend(simple=True, edge_aggs={'weight': 'min'}, needed_attrs={'weight'})
        self.assertEqual(igG.ecount(), 1)
        self.assertEqual(igG.es['weight'][0], 1.0)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_simple_graph_default_minimum_weight(self):
        G = build_parallel()
        igG = G.ig.backend(simple=True, needed_attrs={'weight'})
        self.assertEqual(igG.es['weight'][0], 1.0)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_needed_attrs_slimming(self):
        G = build_small()
        igG = G.ig.backend(needed_attrs={'weight'})
        self.assertEqual(set(igG.es.attributes()), {'weight'})

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_needed_attrs_drop_weight(self):
        G = build_small()
        igG = G.ig.backend(needed_attrs=set())
        self.assertEqual(len(igG.es.attributes()), 0)

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_vertex_iterable_coercion(self):
        G = build_small()
        paths = G.ig.get_shortest_paths(v='a', to=['c'], weights='weight')
        igG = G.ig.backend()
        names = igG.vs['name']
        path_entities = [names[i] for i in paths[0]]
        self.assertEqual(path_entities, ['a', 'b', 'c'])

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_reify_warning(self):
        G = build_small()
        with self.assertWarns(RuntimeWarning):
            G.ig.distances(source='a', target='c', weights='weight', _ig_hyperedge='reify')

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_dir_exposes_algorithms(self):
        G = build_small()
        names = dir(G.ig)
        self.assertIn('distances', names)
        self.assertIn('get_shortest_paths', names)


if __name__ == '__main__':
    unittest.main(verbosity=2)
