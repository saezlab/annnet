from __future__ import annotations

import pytest
from scipy import sparse

from annnet import AnnNet
from annnet.io.scverse import (
    from_anndata,
    from_mudata,
    from_spatialdata,
    to_anndata,
    to_mudata,
    to_spatialdata,
)

from .conftest import assert_edge_attrs_equal, assert_graphs_equal, assert_vertex_attrs_equal

ad = pytest.importorskip('anndata')
mudata = pytest.importorskip('mudata')


def _build_multilayer_graph() -> AnnNet:
    g = AnnNet(aspects={'condition': ['healthy', 'treated']}, directed=True)
    g.add_vertices('A', layer=('healthy',))
    g.add_vertices('A', layer=('treated',))
    g.add_vertices('B', layer=('healthy',))
    g.add_vertices('B', layer=('treated',))
    g.add_edges(('A', ('healthy',)), ('B', ('healthy',)), edge_id='e_h')
    g.add_edges(('A', ('treated',)), ('B', ('treated',)), edge_id='e_t')
    g.layers.set_vertex_layer_attrs('A', ('healthy',), abundance=3.5)
    g.layers.set_layer_attrs(('healthy',), cohort='ctrl')
    return g


def test_to_anndata_roundtrip_complex_graph(complex_graph):
    adata = to_anndata(complex_graph)

    assert adata.n_obs == complex_graph.nv
    assert adata.n_vars == complex_graph.ne
    assert sparse.issparse(adata.X)
    assert '__annnet__' in adata.uns
    assert {'source', 'target', 'weight', 'directed', 'edge_type'} <= set(adata.var.columns)

    g2 = from_anndata(adata)

    assert_graphs_equal(complex_graph, g2, check_slices=True, check_hyperedges=True)
    assert_vertex_attrs_equal(complex_graph, g2, 'A')
    assert_edge_attrs_equal(complex_graph, g2, 'e1', ignore_private=False)
    assert g2.uns == complex_graph.uns


def test_from_anndata_generic_binary_graph():
    obs_df = ad.AnnData(
        X=sparse.csr_matrix(
            [
                [-1.0, 0.0],
                [1.0, -2.0],
                [0.0, 2.0],
            ]
        ),
        obs={'score': [1.0, 2.0, 3.0]},
        var={
            'source': ['A', 'B'],
            'target': ['B', 'C'],
            'weight': [1.0, 2.0],
            'directed': [True, True],
            'relation': ['ab', 'bc'],
        },
    )
    obs_df.obs_names = ['A', 'B', 'C']
    obs_df.var_names = ['e1', 'e2']

    g = from_anndata(obs_df)

    assert set(g.vertices()) == {'A', 'B', 'C'}
    assert set(g.edges()) == {'e1', 'e2'}
    assert g.edge_weights['e2'] == 2.0
    assert g.attrs.get_attr_edge('e1', 'relation') == 'ab'
    assert g.attrs.get_attr_vertex('B', 'score') == 2.0


def test_multilayer_anndata_roundtrip_preserves_supra_vertices():
    g = _build_multilayer_graph()

    adata = to_anndata(g)

    assert adata.n_obs == 4
    assert 'annnet_vertex_id' in adata.obs.columns
    assert 'annnet_layer_condition' in adata.obs.columns

    g2 = from_anndata(adata)

    assert g2.aspects == g.aspects
    assert g2.elem_layers == g.elem_layers
    assert g._VM.issubset(g2._VM)
    assert dict(g2.edge_layers) == dict(g.edge_layers)
    assert dict(g2.edge_kind) == dict(g.edge_kind)


def test_mudata_roundtrip_complex_graph(complex_graph):
    mdata = to_mudata(complex_graph)

    assert list(mdata.mod.keys()) == ['graph']
    assert '__annnet__' in mdata.uns

    g2 = from_mudata(mdata)

    assert_graphs_equal(complex_graph, g2, check_slices=True, check_hyperedges=True)
    assert_edge_attrs_equal(complex_graph, g2, 'h1', ignore_private=False)


def test_spatialdata_bridge_requires_dependency_or_roundtrips(simple_graph):
    spatialdata = pytest.importorskip(
        'spatialdata', reason='spatialdata not installed', exc_type=ImportError
    )
    sdata = to_spatialdata(simple_graph)
    assert isinstance(sdata, spatialdata.SpatialData)
    g2 = from_spatialdata(sdata)
    assert_graphs_equal(simple_graph, g2, check_slices=False, check_hyperedges=False)


def test_to_spatialdata_raises_clear_import_error_when_missing(simple_graph, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == 'spatialdata':
            raise ImportError('forced missing dependency')
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', _fake_import)
    with pytest.raises(ImportError, match='spatialdata is required'):
        to_spatialdata(simple_graph)
