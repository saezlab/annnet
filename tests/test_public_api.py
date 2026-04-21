import pathlib
import sys
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import annnet as an
from annnet._dataframe_backend import DATAFRAME_BACKEND_PRIORITY
from annnet._optional_components import (
    DATAFRAME_BACKENDS,
    IO_MODULES,
    PLOT_BACKENDS,
    component_names,
)
from annnet._plotting_backend import PLOT_BACKEND_PRIORITY


class TestPublicAPI:
    def test_top_level_exports_resolve(self):
        assert an.AnnNet is not None
        assert an.Graph is an.AnnNet
        assert an.EdgeType is not None
        assert an.Traversal is not None
        assert an.__version__
        assert an.__license__
        assert callable(an.get_metadata)
        assert callable(an.get_latest_version)
        assert callable(an.info)

        for name in [
            'available_backends',
            'available_dataframe_backends',
            'available_plot_backends',
            'get_default_dataframe_backend',
            'get_default_plot_backend',
            'select_dataframe_backend',
            'select_plot_backend',
            'set_default_dataframe_backend',
            'set_default_plot_backend',
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

    def test_component_registries_drive_backend_order_and_info(self):
        summary = an.info()

        assert DATAFRAME_BACKEND_PRIORITY == component_names(DATAFRAME_BACKENDS)
        assert PLOT_BACKEND_PRIORITY == component_names(PLOT_BACKENDS)
        assert tuple(summary.tabular_backends) == component_names(DATAFRAME_BACKENDS)
        assert tuple(summary.plot_backends) == component_names(PLOT_BACKENDS)
        assert tuple(summary.io_modules) == component_names(IO_MODULES)

    def test_public_backend_defaults_are_configurable(self):
        original_dataframe_backend = an.get_default_dataframe_backend()
        original_plot_backend = an.get_default_plot_backend()

        try:
            an.set_default_dataframe_backend('auto')
            assert an.get_default_dataframe_backend() == 'auto'
            assert an.Graph()._annotations_backend == an.select_dataframe_backend(None)

            dataframe_backends = an.available_dataframe_backends()
            explicit_dataframe_backend = next(
                (name for name, available in dataframe_backends.items() if available),
                None,
            )
            if explicit_dataframe_backend is not None:
                an.set_default_dataframe_backend(explicit_dataframe_backend)
                assert an.get_default_dataframe_backend() == explicit_dataframe_backend
                assert an.Graph()._annotations_backend == explicit_dataframe_backend

            an.set_default_plot_backend('auto')
            assert an.get_default_plot_backend() == 'auto'
            plot_backends = an.available_plot_backends()
            if any(plot_backends.values()):
                assert an.select_plot_backend(None) == an.select_plot_backend('auto')
        finally:
            an.set_default_dataframe_backend(original_dataframe_backend)
            an.set_default_plot_backend(original_plot_backend)

    def test_get_latest_version_rejects_non_http_schemes(self):
        with patch('urllib.request.urlopen') as urlopen:
            assert an.get_latest_version('file:///tmp/pyproject.toml') is None
            urlopen.assert_not_called()

    def test_top_level_submodules_resolve(self):
        assert an.core.AnnNet is an.AnnNet
        assert an.core.Graph is an.AnnNet
        assert an.algorithms.Traversal is an.Traversal
        assert an.io.write.__name__ == an.write.__name__
        assert an.io.to_json.__name__ == an.to_json.__name__
        assert an.io.to_parquet.__name__ == an.to_parquet.__name__
        assert an.io.from_sbml.__name__ == an.from_sbml.__name__
        assert an.io.from_csv.__name__ == an.from_csv.__name__
        assert an.io.edges_to_csv.__name__ == an.edges_to_csv.__name__
        assert an.io.hyperedges_to_csv.__name__ == an.hyperedges_to_csv.__name__
        assert an.io.from_excel.__name__ == an.from_excel.__name__
        assert an.io.from_omnipath.__name__ == an.from_omnipath.__name__
        assert an.io.show_cx2.__name__ == an.show_cx2.__name__

    def test_adapter_exports_do_not_import_backend_modules_on_attribute_resolution(self):
        import annnet.adapters as adapters

        for modname in [
            'annnet.adapters.networkx_adapter',
            'annnet.adapters.igraph_adapter',
            'annnet.adapters.graphtool_adapter',
            'annnet.adapters.pyg_adapter',
        ]:
            sys.modules.pop(modname, None)

        assert callable(adapters.to_nx)
        assert callable(adapters.from_nx)
        assert callable(adapters.to_igraph)
        assert callable(adapters.from_igraph)
        assert callable(adapters.to_graphtool)
        assert callable(adapters.from_graphtool)
        assert callable(adapters.to_pyg)

        assert 'annnet.adapters.networkx_adapter' not in sys.modules
        assert 'annnet.adapters.igraph_adapter' not in sys.modules
        assert 'annnet.adapters.graphtool_adapter' not in sys.modules
        assert 'annnet.adapters.pyg_adapter' not in sys.modules
