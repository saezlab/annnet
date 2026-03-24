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
        assert an.Attr is not None
        assert an.Attributes is not None
        assert an.Edge is not None
        assert an.__version__
        assert an.__license__
        assert callable(an.get_metadata)
        assert callable(an.get_latest_version)
        assert callable(an.info)

        for name in [
            "available_backends",
            "load_adapter",
            "write",
            "read",
            "to_json",
            "from_json",
            "write_ndjson",
            "to_graphml",
            "from_graphml",
            "to_gexf",
            "from_gexf",
            "to_sif",
            "from_sif",
            "to_cx2",
            "from_cx2",
            "show_cx2",
            "to_parquet",
            "from_parquet",
            "to_dataframes",
            "from_dataframes",
            "to_nx",
            "from_nx",
            "from_nx_only",
            "to_igraph",
            "from_igraph",
            "from_ig_only",
            "to_graphtool",
            "from_graphtool",
            "to_pyg",
            "load_csv_to_graph",
            "from_dataframe",
            "export_edge_list_csv",
            "export_hyperedge_csv",
            "load_excel_to_graph",
            "from_sbml",
            "from_cobra_model",
            "from_sbml_cobra",
            "read_omnipath",
            "canonicalize",
            "obj_canonicalized_hash",
            "unique_iter",
            "build_vertex_labels",
            "build_edge_labels",
            "edge_style_from_weights",
            "to_graphviz",
            "to_pydot",
            "plot",
            "render",
        ]:
            assert callable(getattr(an, name))

    def test_metadata_exports_resolve(self):
        meta = an.get_metadata()
        summary = an.info()

        assert meta["version"] == an.__version__
        assert meta["license"] == an.__license__
        assert isinstance(str(summary), str)
        assert "graph backends" in str(summary).lower()

    def test_get_latest_version_rejects_non_http_schemes(self):
        with patch("urllib.request.urlopen") as urlopen:
            assert an.get_latest_version("file:///tmp/pyproject.toml") is None
            urlopen.assert_not_called()

    def test_top_level_submodules_resolve(self):
        assert an.core.AnnNet is an.AnnNet
        assert an.algorithms.Traversal is an.Traversal
        assert an.utils.Attr is an.Attr
        assert an.io.write.__name__ == an.write.__name__
        assert an.io.to_json.__name__ == an.to_json.__name__
        assert an.io.to_parquet.__name__ == an.to_parquet.__name__
        assert an.io.from_sbml.__name__ == an.from_sbml.__name__
        assert an.io.load_csv_to_graph.__name__ == an.load_csv_to_graph.__name__
