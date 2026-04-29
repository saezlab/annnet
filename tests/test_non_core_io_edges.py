from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from annnet.core.graph import AnnNet
from annnet.io import dataframes
from annnet.io import graphml
from annnet.io import sif
from annnet.adapters import graphtool_adapter
from annnet._support.dataframe_backend import dataframe_from_rows, dataframe_to_rows


def test_dataframe_export_options_and_private_attr_filtering():
    graph = AnnNet(directed=None)
    graph.add_vertices('A', label='alpha', __private='hidden')
    graph.add_vertices('B')
    graph.add_vertices('C')
    graph.add_edges('A', 'B', edge_id='e1', directed=None, relation='activates')
    graph._edges['e1'].weight = None
    graph.add_edges(src=['A', 'B'], tgt=['C'], edge_id='h1', weight=2.5, directed=True)
    graph.attrs.set_edge_attrs('h1', pathway='p1', __internal='secret')
    graph.slices.add_slice('s1')
    graph.slices.add_edge_to_slice('s1', 'e1')
    graph.attrs.set_edge_slice_attrs('s1', 'e1', weight=7.0)

    exported = dataframes.to_dataframes(
        graph,
        include_slices=False,
        include_hyperedges=False,
        public_only=True,
    )
    assert set(exported) == {'nodes', 'edges'}
    assert '__private' not in dataframe_to_rows(exported['nodes'])[0]

    exploded = dataframes.to_dataframes(graph, explode_hyperedges=True, public_only=True)
    hyper_rows = dataframe_to_rows(exploded['hyperedges'])
    assert {row['role'] for row in hyper_rows} == {'head', 'tail'}
    assert all('__internal' not in row for row in hyper_rows)
    assert {'slice_id': 's1', 'edge_id': 'e1', 'weight': 7.0} in dataframe_to_rows(
        exploded['slice_weights']
    )


def test_from_dataframes_validation_and_slice_weight_edge_cases():
    with pytest.raises(ValueError, match='vertex_id'):
        dataframes.from_dataframes(nodes=dataframe_from_rows([{'id': 'A'}]))
    with pytest.raises(ValueError, match='source.*target'):
        dataframes.from_dataframes(edges=dataframe_from_rows([{'source': 'A'}]))
    with pytest.raises(ValueError, match='edge_id'):
        dataframes.from_dataframes(hyperedges=dataframe_from_rows([{'members': ['A']}]))
    with pytest.raises(ValueError, match='edge_id.*vertex_id'):
        dataframes.from_dataframes(
            hyperedges=dataframe_from_rows([{'edge_id': 'h1'}]),
            exploded_hyperedges=True,
        )
    with pytest.raises(ValueError, match='slice_id.*edge_id'):
        dataframes.from_dataframes(slices=dataframe_from_rows([{'slice_id': 's1'}]))

    graph = dataframes.from_dataframes(
        nodes=dataframe_from_rows([{'vertex_id': 'A'}, {'vertex_id': 'B'}, {'vertex_id': 'C'}]),
        edges=dataframe_from_rows(
            [{'source': 'A', 'target': 'B', 'edge_id': 'e1', 'directed': False}]
        ),
        hyperedges=dataframe_from_rows(
            [
                {'edge_id': 'h1', 'vertex_id': 'A', 'role': 'member', 'directed': False},
                {'edge_id': 'h1', 'vertex_id': 'C', 'role': 'member', 'directed': False},
            ]
        ),
        slices=dataframe_from_rows(
            [{'slice_id': 'kept', 'edge_id': 'e1'}, {'slice_id': 'missing', 'edge_id': 'nope'}]
        ),
        slice_weights=dataframe_from_rows(
            [
                {'slice_id': 'kept', 'edge_id': 'e1', 'weight': 3.0},
                {'slice_id': 'kept'},
            ]
        ),
        exploded_hyperedges=True,
    )

    assert 'h1' in graph.hyperedge_definitions
    assert 'kept' in graph.slices.list_slices(include_default=True)
    assert graph.attrs.get_effective_edge_weight('e1', slice='kept') == 3.0


def test_graphml_sanitize_restore_and_gexf_smoke(tmp_path):
    import networkx as nx

    nx_graph = nx.MultiDiGraph()
    nx_graph.graph['payload'] = {'x': 1}
    nx_graph.graph['skip'] = None
    nx_graph.add_node('A', truth=True, number='42', payload=['x'], missing=float('nan'))
    nx_graph.add_edge('A', 'A', key='e1', payload={'kind': 'loop'})

    graphml._sanitize_graphml_inplace(nx_graph)
    assert nx_graph.graph['payload'] == '{"x": 1}'
    assert 'skip' not in nx_graph.graph
    assert 'missing' not in nx_graph.nodes['A']

    graphml._restore_types_graphml_inplace(nx_graph)
    assert nx_graph.graph['payload'] == {'x': 1}
    assert nx_graph.nodes['A']['truth'] is True
    assert nx_graph.nodes['A']['number'] == 42
    assert nx_graph.nodes['A']['payload'] == ['x']

    ann = AnnNet()
    ann.add_vertices('A')
    ann.add_vertices('B')
    ann.add_edges('A', 'B', edge_id='e1')
    out = tmp_path / 'graph.gexf'
    graphml.to_gexf(ann, out)
    restored = graphml.from_gexf(out)
    assert set(restored.vertices()) == {'A', 'B'}


def test_sif_helpers_and_manifest_without_file(tmp_path):
    assert sif._split_sif_line('A rel B\n', None) == ['A', 'rel', 'B']
    assert sif._split_sif_line('A|rel||B\n', '|') == ['A', 'rel', 'B']

    graph = AnnNet()
    graph.add_vertices('A', label='alpha')
    graph.add_vertices('B')
    graph.add_edges('A', 'B', edge_id='e1', weight=2.0, relation='binds')
    graph.add_edges(src=['A', 'B'], edge_id='h1', directed=False, weight=4.0)
    graph._restore_supra_nodes({('A', ('layer',))})

    assert sif._safe_vertex_attr_rows(SimpleNamespace(vertex_attributes=None)) == []
    assert (
        sif._get_edge_weight(SimpleNamespace(_edges={'bad': SimpleNamespace(weight='x')}), 'bad')
        == 1.0
    )
    assert sif._build_edge_attr_map(SimpleNamespace(edge_attributes=None)) is None

    _none, manifest = sif.to_sif(graph, path=None, lossless=True)
    assert manifest['hyperedges']['h1']['members']
    assert {'node': 'A', 'layer': ['layer']} in manifest['multilayer']['VM']

    sif_path = tmp_path / 'manual.sif'
    sif_path.write_text('# comment\nA\tbinds\tB\nC binds D\nnone\tbad\tE\n')
    nodes_path = tmp_path / 'manual.nodes'
    nodes_path.write_text('# comment\nA\tactive=true\tscore=2.5\nB\tlabel=bee\n')
    restored = sif.from_sif(
        sif_path,
        nodes_path=nodes_path,
        read_nodes_sidecar=True,
        relation_attr='interaction',
    )
    assert {'A', 'B'}.issubset(set(restored.vertices()))
    assert restored.attrs.get_attr_vertex('A', 'active') is True


def test_sif_from_manifest_restores_hyperedges_slices_and_multilayer(tmp_path):
    sif_path = tmp_path / 'lossless.sif'
    sif_path.write_text('A\tbinds\tB\n')

    manifest = {
        'binary_edges': {
            'e1': {
                'source': 'A',
                'target': 'B',
                'source_endpoint': 'A',
                'target_endpoint': 'B',
                'directed': False,
            }
        },
        'edge_metadata': {'e1': {'weight': 2.0, 'attrs': {'relation': 'binds'}}},
        'hyperedges': {
            'h1': {
                'directed': True,
                'head': ['A'],
                'tail': ['B'],
                'weight': 5.0,
                'attrs': {'kind': 'complex'},
            }
        },
        'slices': {'s1': {'edges': ['e1'], 'weights': {'e1': 9.0}}},
        'vertex_attrs': {'A': {'active': True}},
        'multilayer': {
            'aspects': ['time'],
            'elem_layers': {'time': ['t1']},
            'aspect_attrs': {'time': {'unit': 'day'}},
            'VM': [{'node': 'A', 'layer': ['t1']}],
            'edge_kind': {'e1': 'intra'},
            'edge_layers': {'e1': {'kind': 'single', 'layers': [['t1']]}},
            'node_layer_attrs': [{'node': 'A', 'layer': ['t1'], 'attrs': {'state': 'on'}}],
            'layer_tuple_attrs': [{'layer': ['t1'], 'attrs': {'color': 'red'}}],
            'layer_attributes': [{'aspect': 'time', 'layer': 't1'}],
        },
    }

    restored = sif.from_sif(sif_path, manifest=manifest, relation_attr='relation')
    assert restored._edges['e1'].directed is False
    assert restored._edges['e1'].weight == 2.0
    assert 'h1' in restored.hyperedge_definitions
    assert 's1' in restored.slices.list_slices(include_default=True)
    assert 'e1' in restored.slices.get_slice_edges('s1')
    assert restored._aspect_attrs['time']['unit'] == 'day'
    assert ('A', ('t1',)) in restored._VM
    assert restored._state_attrs[('A', ('t1',))]['state'] == 'on'


def test_graphtool_adapter_missing_dependency_paths(monkeypatch):
    monkeypatch.setattr(graphtool_adapter, 'gt', None)

    with pytest.raises(RuntimeError, match='graph-tool is not installed'):
        graphtool_adapter.to_graphtool(AnnNet())
    with pytest.raises(RuntimeError, match='graph-tool is not installed'):
        graphtool_adapter.from_graphtool(SimpleNamespace())


def test_json_helper_edge_cases():
    from annnet.io import json_format

    assert json_format._coerce_coeff_mapping(None) == {}
    assert json_format._coerce_coeff_mapping('not-json') == {}
    assert json_format._coerce_coeff_mapping([{'vertex': 'A', '__value': 2}, ['B', 3], 4]) == {
        'A': {'__value': 2},
        'B': 3,
    }
    assert json_format._endpoint_coeff_map(
        {'coeff': {'A': {'__value': object()}}}, 'coeff', {'A'}
    ) == {'A': 1.0}
    assert json_format._attrs_by_id(
        dataframe_from_rows([{'edge_id': None}, {'edge_id': 'e1', '__x': 1, 'v': 2}]),
        'edge_id',
        public_only=True,
    ) == {'e1': {'v': 2}}

    assert math.isfinite(json_format._endpoint_coeff_map({}, 'missing', {'A'})['A'])


def test_json_multilayer_and_malformed_entries_roundtrip(tmp_path):
    from annnet.io import json_format

    doc = {
        'nodes': [
            {'id': 'A', 'label': 'alpha'},
            {'id': 'B'},
            {'label': 'missing id'},
        ],
        'edges': [
            {
                'id': 'e1',
                'source': {'kind': 'supra', 'vertex': 'A', 'layer': ['t1']},
                'target': {'kind': 'supra', 'vertex': 'B', 'layer': ['t1']},
                'directed': True,
                'weight': 2.0,
                'kind': 'observed',
            },
            {'source': 'skip', 'target': 'missing id'},
        ],
        'x-extensions': {
            'slices': [{'slice_id': 's1'}, {'bad': 'row'}],
            'edge_slices': [{'slice_id': 's1', 'edge_id': 'e1', 'weight': 4.0}, {'slice_id': None}],
            'hyperedges': [],
            'multilayer': {
                'aspects': ['time'],
                'elem_layers': {'time': ['t1']},
                'aspect_attrs': {'time': {'unit': 'day'}},
                'VM': [{'node': 'A', 'layer': ['t1']}],
                'edge_kind': {'e1': 'intra', 'h1': 'hyper'},
                'edge_layers': {'e1': {'kind': 'single', 'layers': [['t1']]}},
                'node_layer_attrs': [{'node': 'A', 'layer': ['t1'], 'attrs': {'state': 'on'}}],
                'layer_tuple_attrs': [{'layer': ['t1'], 'attrs': {'color': 'red'}}],
                'layer_attributes': [{'aspect': 'time', 'layer': 't1'}],
            },
        },
    }
    path = tmp_path / 'multilayer.json'
    path.write_text(json_format.json.dumps(doc))

    restored = json_format.from_json(path)
    assert restored._aspect_attrs['time']['unit'] == 'day'
    assert restored._state_attrs[('A', ('t1',))]['state'] == 'on'
    assert restored.attrs.get_effective_edge_weight('e1', slice='s1') == 4.0
    assert restored._edges['e1'].ml_kind == 'intra'
