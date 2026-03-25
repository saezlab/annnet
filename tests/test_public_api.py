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
            "available_backends",
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
            "to_igraph",
            "from_igraph",
            "to_graphtool",
            "from_graphtool",
            "to_pyg",
            "from_csv",
            "from_dataframe",
            "edges_to_csv",
            "hyperedge_to_csv",
            "from_excel",
            "from_sbml",
            "from_cobra_model",
            "from_sbml_cobra",
            "read_omnipath",
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
        assert an.io.write.__name__ == an.write.__name__
        assert an.io.to_json.__name__ == an.to_json.__name__
        assert an.io.to_parquet.__name__ == an.to_parquet.__name__
        assert an.io.from_sbml.__name__ == an.from_sbml.__name__
        assert an.io.from_csv.__name__ == an.from_csv.__name__
        assert an.io.edges_to_csv.__name__ == an.edges_to_csv.__name__
        assert an.io.hyperedge_to_csv.__name__ == an.hyperedge_to_csv.__name__
        assert an.io.from_excel.__name__ == an.from_excel.__name__
