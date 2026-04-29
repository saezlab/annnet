from __future__ import annotations

import json
import tarfile
import urllib.request
from enum import Enum
from pathlib import Path
from unittest.mock import Mock

import pytest

import annnet
from annnet import _metadata
from annnet import _optional_components as optional_components
from annnet import _plotting_backend
from annnet import _dataframe_backend as df_backend
from annnet.adapters import _utils as adapter_utils
from annnet.io._utils import _read_archive, _write_archive


class ExampleEnum(Enum):
    ONE = 'one'
    TWO = 'two'


class CursorLike:
    columns = ['a', 'b']

    def fetchall(self):
        return [(1, 2), (3, 4)]


class FakeGraph:
    def __init__(self):
        self.ne = 4
        self.idx_to_edge = {0: 'e_pos', 1: 'e_neg', 2: 'h1', 3: 'loop'}
        self._weights = {'e_pos': 2.0, 'e_neg': -3.0, 'h1': 0.0}
        self._attrs = {'e_pos': {'kind': 'activation'}, 'e_neg': {'kind': 'repression'}}

    def vertices(self):
        return ['A', 'B', 'C']

    def get_edge(self, index):
        return [
            (frozenset({'A'}), frozenset({'B'})),
            (frozenset({'B'}), frozenset({'C'})),
            (frozenset({'A', 'B'}), frozenset({'C'})),
            (frozenset({'C'}), frozenset({'C'})),
        ][index]

    def get_attr_vertex(self, vertex, key, default=None):
        return {'A': 'alpha'}.get(vertex, default)

    def get_attr_edge(self, edge_id, key, default=None):
        return self._attrs.get(edge_id, {}).get(key, default)

    def get_effective_edge_weight(self, edge_id, slice=None):
        if edge_id == 'loop':
            raise KeyError(edge_id)
        return self._weights[edge_id]


def test_optional_component_selection_with_mocked_availability(monkeypatch):
    specs = {
        'first': optional_components.OptionalComponent('missing_first', 'install first'),
        'second': optional_components.OptionalComponent('present_second'),
    }
    monkeypatch.setattr(
        optional_components,
        'is_component_available',
        lambda component: component.module == 'present_second',
    )

    assert optional_components.component_names(specs) == ('first', 'second')
    assert optional_components.available_optional_components(specs) == {
        'first': False,
        'second': True,
    }
    assert optional_components.component_status(specs)['first'] == {
        'available': 'no',
        'install': 'install first',
    }
    assert (
        optional_components.select_component(
            specs, 'auto', kind='test', install_message='install something'
        )
        == 'second'
    )
    assert (
        optional_components.select_component(
            specs, 'second', kind='test', install_message='install something'
        )
        == 'second'
    )

    with pytest.raises(ValueError, match='Unknown test backend'):
        optional_components.select_component(
            specs, 'unknown', kind='test', install_message='install something'
        )
    with pytest.raises(RuntimeError, match='not installed'):
        optional_components.select_component(
            specs, 'first', kind='test', install_message='install something'
        )


def test_plot_backend_default_roundtrip_and_validation(monkeypatch):
    original = _plotting_backend.get_default_plot_backend()
    monkeypatch.setattr(_plotting_backend, 'select_component', lambda *args, **kwargs: 'matplotlib')
    try:
        assert _plotting_backend.set_default_plot_backend(None) == 'auto'
        assert _plotting_backend.select_plot_backend(None) == 'matplotlib'
        assert _plotting_backend.set_default_plot_backend('matplotlib') == 'matplotlib'
        assert _plotting_backend.select_plot_backend(None) == 'matplotlib'
    finally:
        _plotting_backend.set_default_plot_backend(original)


def test_dataframe_backend_helpers_preserve_rows_and_schema(tmp_path):
    backend = df_backend.select_dataframe_backend('auto')
    rows = [{'id': 'a', 'score': 1.5, 'flag': False}, {'id': 'b', 'score': 2.5, 'flag': True}]
    table = df_backend.dataframe_from_rows(rows, backend=backend)

    assert df_backend.dataframe_height(table) == 2
    table_rows = df_backend.dataframe_to_rows(table)
    assert table_rows[0]['id'] == 'a'
    assert table_rows[0]['score'] == 1.5
    assert table_rows[1]['id'] == 'b'
    assert table_rows[1]['flag'] is True
    assert set(df_backend.dataframe_columns(table)) == {'id', 'score', 'flag'}
    assert df_backend.dataframe_filter_eq(table, 'id', 'a') is not table
    assert df_backend.dataframe_to_rows(df_backend.dataframe_filter_eq(table, 'missing', 'x')) == []
    assert df_backend.dataframe_to_rows(df_backend.dataframe_filter_in(table, 'id', [])) == []
    assert df_backend.dataframe_to_rows(df_backend.dataframe_filter_not_in(table, 'id', ['a'])) == [
        {'id': 'b', 'score': 2.5, 'flag': True}
    ]

    renamed = df_backend.rename_dataframe_columns(table, {'id': 'name'})
    assert 'name' in df_backend.dataframe_columns(renamed)

    upserted = df_backend.dataframe_upsert_rows(
        table, [{'id': 'a', 'score': 9.0, 'flag': True}], 'id', backend=backend
    )
    assert {row['id']: row.get('score') for row in df_backend.dataframe_to_rows(upserted)}[
        'a'
    ] == 9.0

    empty = df_backend.empty_dataframe(
        {'name': 'text', 'value': 'float', 'enabled': 'bool', 'tags': 'list_text'},
        backend=backend,
    )
    assert df_backend.dataframe_columns(empty) == ['name', 'value', 'enabled', 'tags']
    assert df_backend.dataframe_memory_usage(None) == 0
    assert df_backend.dataframe_backend(None) == backend

    csv_path = tmp_path / 'rows.csv'
    df_backend._dataframe_write_csv(table, csv_path)
    assert csv_path.exists()

    parquet_path = tmp_path / 'rows.parquet'
    df_backend._dataframe_write_parquet(table, parquet_path)
    assert parquet_path.exists()


def test_dataframe_backend_edge_paths_and_backend_detection(monkeypatch):
    backend = df_backend.select_dataframe_backend('auto')

    assert df_backend.dataframe_to_rows(None) == []
    assert df_backend.dataframe_height(None) == 0
    assert df_backend.dataframe_columns(None) == []
    assert df_backend.clone_dataframe(None) is None
    assert df_backend.dataframe_filter_ne(None, 'missing', 'x') is None
    assert df_backend.dataframe_filter_not_in(None, 'missing', ['x']) is None
    assert df_backend.dataframe_append_rows(None, [], backend=backend) is None
    assert df_backend.dataframe_upsert_rows(None, [], 'id', backend=backend) is None
    assert df_backend.rename_dataframe_columns(None, {'a': 'b'}) is None
    assert df_backend.rename_dataframe_columns('unchanged', {}) == 'unchanged'
    assert df_backend._schema_from_df(None) is None
    assert df_backend._schema_names(None) == []
    assert df_backend._text_schema(['a', 'b']) == {'a': 'text', 'b': 'text'}

    empty_like_none = df_backend._empty_like(None)
    assert df_backend.dataframe_to_rows(empty_like_none) == []

    appended = df_backend.dataframe_append_rows(None, [{'id': 'new', 'value': 1}], backend=backend)
    assert df_backend.dataframe_to_rows(appended) == [{'id': 'new', 'value': 1}]

    table = df_backend.dataframe_from_rows(
        [{'id': 'a', 'value': 1}, {'id': 'b', 'value': 2}], backend=backend
    )
    filtered = df_backend._rows_filter(table, lambda row: row['value'] > 1)
    assert df_backend.dataframe_to_rows(filtered) == [{'id': 'b', 'value': 2}]
    assert df_backend.dataframe_to_rows(df_backend.dataframe_drop_rows(table, 'id', ['a'])) == [
        {'id': 'b', 'value': 2}
    ]
    assert df_backend.dataframe_to_rows(df_backend.dataframe_filter_ne(table, 'id', 'a')) == [
        {'id': 'b', 'value': 2}
    ]
    assert df_backend.dataframe_to_rows(df_backend.dataframe_filter_in(table, 'id', ['a'])) == [
        {'id': 'a', 'value': 1}
    ]
    assert df_backend.dataframe_filter_not_in(table, 'id', []) is not table
    assert df_backend.dataframe_append_rows(table, [], backend=backend) is not table
    assert df_backend.dataframe_upsert_rows(table, [], 'id', backend=backend) is not table

    empty_columns = df_backend.dataframe_from_columns(
        {},
        schema={'id': 'text', 'value': 'float', 'enabled': 'bool'},
        backend=backend,
    )
    assert df_backend.dataframe_columns(empty_columns) == ['id', 'value', 'enabled']
    assert df_backend.dataframe_from_columns({'id': []}, backend=backend) is not None

    class UnknownFrame:
        pass

    assert df_backend.dataframe_backend(UnknownFrame(), default=backend) == backend

    try:
        import pandas as pd

        assert df_backend.dataframe_backend(pd.DataFrame({'a': [1]})) == 'pandas'
    except Exception:
        pass

    try:
        import pyarrow as pa

        assert df_backend.dataframe_backend(pa.table({'a': [1]})) == 'pyarrow'
    except Exception:
        pass

    original = df_backend.get_default_dataframe_backend()
    try:
        assert df_backend.set_default_dataframe_backend(None) == 'auto'
        assert df_backend.set_default_dataframe_backend(backend) == backend
    finally:
        df_backend.set_default_dataframe_backend(original)

    monkeypatch.setattr(df_backend, 'available_dataframe_backends', lambda: {'x': False})
    monkeypatch.setattr(df_backend, 'select_dataframe_backend', Mock(return_value='fallback'))
    assert df_backend.set_default_dataframe_backend('auto') == 'auto'


def test_adapter_utils_serialization_roundtrips(tmp_path):
    supra = ('nodeA', ('layer1', 'layer2'))
    assert adapter_utils._deserialize_endpoint(adapter_utils._serialize_endpoint(supra)) == supra
    assert adapter_utils._deserialize_endpoint(json.dumps(['nodeB', ['x']])) == ('nodeB', ('x',))
    assert adapter_utils._deserialize_endpoint("('nodeC', ('y',))") == ('nodeC', ('y',))
    assert adapter_utils._deserialize_endpoint('plain') == 'plain'

    layers = {
        'intra': ('t1', 'bus'),
        'inter': (('t1', 'bus'), ('t2', 'train')),
        'raw': {'not': 'a tuple'},
        'none': None,
    }
    encoded_layers = adapter_utils._serialize_edge_layers(layers)
    assert adapter_utils._deserialize_edge_layers(encoded_layers) == {
        'intra': ('t1', 'bus'),
        'inter': (('t1', 'bus'), ('t2', 'train')),
    }

    vm = {('A', ('x',)), ('B', ('y',))}
    assert adapter_utils._deserialize_VM(adapter_utils._serialize_VM(vm)) == vm

    nl_attrs = {('A', ('x',)): {'color': 'red'}}
    assert (
        adapter_utils._deserialize_node_layer_attrs(
            adapter_utils._serialize_node_layer_attrs(nl_attrs)
        )
        == nl_attrs
    )

    slices = {'s1': {'vertices': {'A'}, 'edges': {'e1'}, 'attributes': {'kind': 'test'}}}
    assert adapter_utils._deserialize_slices(adapter_utils._serialize_slices(slices)) == slices

    layer_attrs = {('x', 'y'): {'speed': 10}}
    assert (
        adapter_utils._deserialize_layer_tuple_attrs(
            adapter_utils._serialize_layer_tuple_attrs(layer_attrs)
        )
        == layer_attrs
    )

    assert adapter_utils._coerce_coeff_mapping('[["A", 2], {"vertex": "B", "__value": 3}]') == {
        'A': 2,
        'B': {'__value': 3},
    }
    assert adapter_utils._coerce_coeff_mapping('not json') == {}
    assert adapter_utils._endpoint_coeff_map(
        {'coeffs': {'A': {'__value': 'bad'}}}, 'coeffs', {'A'}
    ) == {'A': 1.0}
    assert adapter_utils._serialize_value(ExampleEnum.ONE) == 'ONE'
    assert adapter_utils._attrs_to_dict(
        {'outer': {'x': ExampleEnum.TWO}, 'enum': ExampleEnum.ONE}
    ) == {
        'outer': {'x': 'TWO'},
        'enum': 'ONE',
    }

    table = df_backend.dataframe_from_rows(
        [{'a': 1}], backend=df_backend.select_dataframe_backend()
    )
    assert adapter_utils._rows_like(table) == [{'a': 1}]
    assert adapter_utils._rows_like(CursorLike()) == [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}]
    assert adapter_utils._rows_like({'a': [1, 2], 'b': [3, 4]}) == [
        {'a': 1, 'b': 3},
        {'a': 2, 'b': 4},
    ]
    assert adapter_utils._rows_like([{'x': 1}]) == [{'x': 1}]
    assert adapter_utils._safe_df_to_rows(None) == []

    manifest_path = tmp_path / 'manifest.json'
    adapter_utils.save_manifest({'ok': True}, manifest_path)
    assert adapter_utils.load_manifest(manifest_path) == {'ok': True}


def test_archive_utils_roundtrip_and_invalid_archive(tmp_path):
    src = tmp_path / 'graph.annnet'
    src.mkdir()
    (src / 'manifest.json').write_text('{"format": "annnet"}')
    archive = tmp_path / 'graph.tar.gz'

    _write_archive(src, archive)
    extract_dir = tmp_path / 'extract'
    extract_dir.mkdir()
    root = _read_archive(archive, extract_dir)
    assert root.name == src.name
    assert (root / 'manifest.json').read_text() == '{"format": "annnet"}'

    bad_archive = tmp_path / 'bad.tar.gz'
    with tarfile.open(bad_archive, 'w:gz') as tar:
        for name in ['a', 'b']:
            folder = tmp_path / name
            folder.mkdir()
            tar.add(folder, arcname=name)
    bad_extract = tmp_path / 'bad_extract'
    bad_extract.mkdir()
    with pytest.raises(ValueError, match='expected single root'):
        _read_archive(bad_archive, bad_extract)


def test_metadata_helpers_and_info_rendering(monkeypatch):
    assert _metadata._normalize_people(
        [{'name': 'Ada', 'email': 'ada@example.org'}, {'email': 'team@example.org'}, 'Raw']
    ) == ['Ada <ada@example.org>', 'team@example.org', 'Raw']
    assert _metadata._normalize_people({'name': 'not a list'}) == []
    assert _metadata._normalize_license({'text': 'BSD'}) == 'BSD'
    assert _metadata._normalize_license({'file': 'LICENSE'}) == 'LICENSE'
    assert _metadata._normalize_license(123) is None
    assert _metadata._author_links(['Ada <ada@example.org>', 'A & B']) == (
        'Ada <a href="mailto:ada@example.org" style="text-decoration:none" '
        'title="ada@example.org">&#9993;</a>, A &amp; B'
    )

    meta = {
        'version': '9.9.9',
        'license': 'BSD',
        'authors': ['Ada <ada@example.org>'],
        'author': 'Ada <ada@example.org>',
        'urls': {
            'Repository': 'https://example.org/repo',
            'Documentation': 'https://example.org/docs',
        },
        'full_metadata': {
            'project': {
                'optional-dependencies': {
                    'io': ['pyarrow'],
                    'dev': ['pytest'],
                    'ignored': ['x'],
                }
            }
        },
    }
    info = _metadata.AnnNetInfo(
        metadata=meta,
        graph_backends={'networkx': {'available': 'yes', 'install': 'annnet[networkx]'}},
        plot_backends={'matplotlib': {'available': 'no', 'install': 'annnet[plot]'}},
        tabular_backends={'polars': {'available': 'yes', 'install': 'annnet[polars]'}},
        io_modules={'json': {'available': 'yes', 'install': 'built in'}},
    )
    text = str(info)
    html = info.to_html()
    assert 'Installed version: v9.9.9' in text
    assert 'https://example.org/repo' in html
    assert 'Installable bundles' in html
    assert info._mime_()[0] == 'text/html'

    monkeypatch.setitem(_metadata.sys.modules, 'marimo', object())
    assert _metadata.supports_html() is True
    monkeypatch.delitem(_metadata.sys.modules, 'marimo')
    monkeypatch.delitem(_metadata.sys.modules, 'IPython', raising=False)
    monkeypatch.delitem(_metadata.sys.modules, 'IPython.display', raising=False)
    assert _metadata.supports_html() is False

    fake_response = Mock()
    fake_response.read.return_value = b'version = "1.2.3"'
    fake_response.__enter__ = Mock(return_value=fake_response)
    fake_response.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(urllib.request, 'urlopen', Mock(return_value=fake_response))
    assert _metadata.get_latest_version(url='https://example.org/pyproject.toml') == '1.2.3'
    assert _metadata.get_latest_version(url='ftp://example.org/pyproject.toml') is None


def test_plotting_helpers_with_fake_graph(tmp_path):
    from annnet.utils import plotting

    graph = FakeGraph()
    assert plotting._normalize([float('nan')], lo=float('nan'), hi=float('nan')).shape == (1,)
    assert plotting._is_true_hyperedge(frozenset({'A', 'B'}), frozenset({'A', 'B'})) is False
    assert plotting._is_true_hyperedge(frozenset({'A', 'B'}), frozenset({'C'})) is True

    assert plotting.build_vertex_labels(graph, key='label')['A'] == 'alpha'
    edge_labels = plotting.build_edge_labels(graph, extra_keys=['kind'], layer='slice1')
    assert 'kind=activation' in edge_labels[0]
    styles = plotting.edge_style_from_weights(graph, color_mode='signed')
    assert styles[0]['color'] == 'firebrick4'
    assert styles[1]['color'] == 'dodgerblue4'
    assert styles[2]['color'] == 'black'

    matplotlib = pytest.importorskip('matplotlib')
    matplotlib.use('Agg', force=True)
    fig, ax = plotting.to_matplotlib(
        graph,
        edge_indexes=[0, 1, 2, 3],
        show_edge_labels=True,
        edge_label_keys=['kind'],
        vertex_label_key='label',
    )
    assert fig is ax.figure
    assert ax.collections
    rendered = plotting.render((fig, ax), str(tmp_path / 'fake_graph'), format='png')
    assert Path(rendered).exists()

    class ReprNoisy:
        def __init__(self):
            self.calls = 0

        def _repr_svg_(self):
            self.calls += 1
            return '<svg />'

    noisy = ReprNoisy()
    plotting._suppress_repr_warnings(noisy)
    assert noisy._repr_svg_() == '<svg />'
    assert noisy.calls == 1


def test_lazy_module_exports_and_unknown_attributes():
    assert annnet.adapters.available_backends()
    assert 'plot' in dir(annnet.utils)

    with pytest.raises(AttributeError):
        annnet.adapters.__getattr__('missing_adapter')
    with pytest.raises(AttributeError):
        annnet.io.__getattr__('missing_io')
    with pytest.raises(AttributeError):
        annnet.utils.__getattr__('missing_util')
