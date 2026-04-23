# tests/test_adapters.py
import unittest
import warnings

# Silence noisy NumPy longdouble warning seen on some builds
warnings.filterwarnings(
    'ignore',
    message=r'Signature .*numpy\.longdouble.*',
    category=UserWarning,
    module=r'numpy\._core\.getlimits',
)

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Optional deps presence
HAS_IG = True
try:
    import igraph as ig  # noqa: F401
except Exception:
    HAS_IG = False


from .conftest import build_adapter_graph as _build_graph


class TestIgraphAdapter(unittest.TestCase):
    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_to_igraph_export_and_roundtrip(self):
        from annnet.adapters.igraph_adapter import from_igraph, to_igraph  # adapter under test

        g = _build_graph()

        igG, manifest = to_igraph(
            g,
            directed=True,
            hyperedge_mode='skip',  # if supported similarly
            public_only=True,
        )

        # --- Export checks
        self.assertGreaterEqual(igG.vcount(), 3)
        self.assertGreaterEqual(igG.ecount(), 2)  # hyperedge skipped
        self.assertIn('weights', manifest)
        self.assertIn('slices', manifest)
        self.assertIn('Lw', manifest['slices'])

        # --- Round-trip back to AnnNet
        g2 = from_igraph(igG, manifest)
        self.assertEqual(set(g2.vertices()), set(g.vertices()))
        for eid in g.edge_weights:
            self.assertIn(eid, g2.edge_weights)
        self.assertAlmostEqual(
            g2.attrs.get_effective_edge_weight(list(manifest['slices']['Lw'])[0], slice='Lw'),
            5.0,
            places=7,
        )

    @unittest.skipUnless(HAS_IG, 'python-igraph not installed')
    def test_to_igraph_labels_and_attrs(self):
        from annnet.adapters.igraph_adapter import to_igraph

        g = _build_graph()
        igG, manifest = to_igraph(g, directed=True, hyperedge_mode='skip', public_only=True)

        # Vertex names present
        self.assertTrue(set(igG.vs['name']) >= {'A', 'B', 'C'})
        # Edge count >= 2 (hyperedge skipped)
        self.assertGreaterEqual(igG.ecount(), 2)
        self.assertIn('weights', manifest)


if __name__ == '__main__':
    unittest.main()
