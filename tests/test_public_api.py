import inspect
import pathlib
import sys
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import annnet as an


class TestPublicAPI:
    def test_top_level_exports_resolve(self):
        assert an.AnnNet is not None
        assert an.EdgeType is not None
        assert an.Traversal is not None
        assert an.__version__
        assert an.__license__
        assert callable(an.get_metadata)
        assert callable(an.get_latest_version)
        assert callable(an.info)

        for name in [
            'available_backends',
            'write',
            'read',
            'to_json',
            'from_json',
            'write_ndjson',
            'to_graphml',
            'from_graphml',
            'to_gexf',
            'from_gexf',
            'to_sif',
            'from_sif',
            'to_cx2',
            'from_cx2',
            'show_cx2',
            'to_parquet',
            'from_parquet',
            'to_dataframes',
            'from_dataframes',
            'to_nx',
            'from_nx',
            'to_igraph',
            'from_igraph',
            'to_graphtool',
            'from_graphtool',
            'to_pyg',
            'from_csv',
            'from_dataframe',
            'edges_to_csv',
            'hyperedges_to_csv',
            'from_excel',
            'from_sbml',
            'from_cobra_model',
            'from_sbml_cobra',
            'from_omnipath',
        ]:
            assert callable(getattr(an, name))

    def test_metadata_exports_resolve(self):
        meta = an.get_metadata()
        summary = an.info()

        assert meta['version'] == an.__version__
        assert meta['license'] == an.__license__
        assert isinstance(str(summary), str)
        assert 'graph backends' in str(summary).lower()

    def test_get_latest_version_rejects_non_http_schemes(self):
        with patch('urllib.request.urlopen') as urlopen:
            assert an.get_latest_version('file:///tmp/pyproject.toml') is None
            urlopen.assert_not_called()

    def test_top_level_submodules_resolve(self):
        assert an.core.AnnNet is an.AnnNet
        assert an.algorithms.Traversal is an.Traversal
        assert an.io.write.__name__ == an.write.__name__
        assert an.io.to_json.__name__ == an.to_json.__name__
        assert an.io.to_parquet.__name__ == an.to_parquet.__name__
        assert an.io.from_sbml.__name__ == an.from_sbml.__name__
        assert an.io.from_csv.__name__ == an.from_csv.__name__
        assert an.io.edges_to_csv.__name__ == an.edges_to_csv.__name__
        assert an.io.hyperedges_to_csv.__name__ == an.hyperedges_to_csv.__name__
        assert an.io.from_excel.__name__ == an.from_excel.__name__

    def test_namespace_accessors_resolve(self):
        G = an.AnnNet()
        G.add_vertices(['A', 'B'])
        G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1'}])

        assert G.views is not None
        assert G.ops is not None
        assert G.attrs is not None
        assert callable(G.history)

        sub = G.ops.subgraph(['A', 'B'])
        assert sub.num_vertices == 2

        edges_df = G.views.edges()
        assert len(edges_df) == 1
        assert G.attrs.get_edge_attrs('e1') == {}

    def test_dir_exposes_compact_annnet_api(self):
        G = an.AnnNet()

        public = {name for name in dir(G) if not name.startswith('_')}
        class_public = {
            name for name, _ in inspect.getmembers(an.AnnNet) if not name.startswith('_')
        }

        for required in {
            'add_vertices',
            'add_edges',
            'remove_vertices',
            'remove_edges',
            'num_vertices',
            'num_edges',
            'layers',
            'slices',
            'attrs',
            'views',
            'history',
            'ops',
            'view',
            'read',
            'write',
        }:
            assert required in public
            assert required in class_public

        for hidden in {
            'add_vertex',
            'add_edge',
            'add_edges_bulk',
            'add_vertices_bulk',
            'vertices_view',
            'edges_view',
            'number_of_vertices',
            'number_of_edges',
            'nv',
            'ne',
            'V',
            'E',
            'set_aspects',
            'add_slice',
            'subgraph',
            'mark',
        }:
            assert hidden not in public
            assert hidden not in class_public
