# tests/io/test_omnipath.py

from __future__ import annotations

import io
import sys
import types

import numpy as np
import pandas as pd
import pytest

import annnet.io.omnipath as mod


class FakeAnnNet:
    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs
        self._history_enabled = True
        self._num_entities = 0
        self._num_edges = 0
        self._entities: set[str] = set()
        self.entity_types: dict[str, str] = {}
        self.added_edges = None
        self.added_vertices_calls = []

    def add_edges_bulk(self, bulk):
        self.added_edges = bulk
        verts = set()
        for edge in bulk:
            verts.add(edge['source'])
            verts.add(edge['target'])
        self._entities |= verts
        self.entity_types.update(dict.fromkeys(verts, 'vertex'))
        self._num_entities = len([k for k, v in self.entity_types.items() if v == 'vertex'])
        self._num_edges = len(bulk)

    def add_vertices_bulk(self, items):
        self.added_vertices_calls.append(items)
        for item in items:
            if isinstance(item, tuple):
                vid = item[0]
            else:
                vid = item
            self._entities.add(vid)
            self.entity_types[vid] = 'vertex'
        self._num_entities = len([k for k, v in self.entity_types.items() if v == 'vertex'])

    def _resolve_entity_key(self, key):
        return key


@pytest.fixture(autouse=True)
def patch_annnet(monkeypatch):
    monkeypatch.setattr(mod, 'AnnNet', FakeAnnNet)


@pytest.fixture
def edge_df():
    return pd.DataFrame(
        {
            'source': ['EGFR', 'TP53'],
            'target': ['STAT3', 'MDM2'],
            'is_directed': [True, False],
            'weight': [2.5, 1.0],
            'edge_id': ['e1', 'e2'],
            'slice': ['s1', 's2'],
            'evidence': ['lit', 'screen'],
        }
    )


def test_builds_from_dataframe_and_infers_columns(edge_df):
    g = mod.from_omnipath(df=edge_df, load_vertex_annotations=False)

    assert isinstance(g, FakeAnnNet)
    assert g.init_kwargs['directed'] is True
    assert g.init_kwargs['n'] == 2
    assert g.init_kwargs['e'] == 2
    assert g._history_enabled is True
    assert g._num_edges == 2
    assert g._num_entities == 4

    assert g.added_edges == [
        {
            'source': 'EGFR',
            'target': 'STAT3',
            'weight': 2.5,
            'edge_id': 'e1',
            'edge_directed': True,
            'slice': 's1',
            'attributes': {'evidence': 'lit'},
        },
        {
            'source': 'TP53',
            'target': 'MDM2',
            'weight': 1.0,
            'edge_id': 'e2',
            'edge_directed': False,
            'slice': 's2',
            'attributes': {'evidence': 'screen'},
        },
    ]

    # First vertex call is registration of all discovered vertices.
    assert len(g.added_vertices_calls) == 1
    registered = set(g.added_vertices_calls[0])
    assert registered == {'EGFR', 'STAT3', 'TP53', 'MDM2'}


def test_uses_explicit_source_and_target_columns():
    df = pd.DataFrame(
        {
            'src_gene': ['A'],
            'dst_gene': ['B'],
            'score': [7.0],
            'misc': ['x'],
        }
    )

    g = mod.from_omnipath(
        df=df,
        source_col='src_gene',
        target_col='dst_gene',
        weight_col='score',
        load_vertex_annotations=False,
    )

    assert g.added_edges == [
        {
            'source': 'A',
            'target': 'B',
            'weight': 7.0,
            'edge_id': None,
            'edge_directed': True,
            'slice': None,
            'attributes': {'misc': 'x'},
        }
    ]


def test_raises_when_source_target_cannot_be_inferred():
    df = pd.DataFrame({'a': ['x'], 'b': ['y']})

    with pytest.raises(
        ValueError,
        match='Could not infer source/target columns',
    ):
        mod.from_omnipath(df=df, load_vertex_annotations=False)


def test_dropna_true_silently_drops_rows():
    df = pd.DataFrame(
        {
            'source': ['A', None, 'C', np.nan],
            'target': ['B', 'X', None, 'Y'],
            'evidence': ['ok1', 'drop1', 'drop2', 'drop3'],
        }
    )

    g = mod.from_omnipath(df=df, dropna=True, load_vertex_annotations=False)

    assert g._num_edges == 1
    assert g.added_edges == [
        {
            'source': 'A',
            'target': 'B',
            'weight': 1.0,
            'edge_id': None,
            'edge_directed': True,
            'slice': None,
            'attributes': {'evidence': 'ok1'},
        }
    ]


def test_dropna_false_raises_on_first_null_endpoint():
    df = pd.DataFrame(
        {
            'source': ['A', None],
            'target': ['B', 'C'],
        }
    )

    with pytest.raises(ValueError, match='Found null source/target with dropna=False'):
        mod.from_omnipath(df=df, dropna=False, load_vertex_annotations=False)


@pytest.mark.parametrize(
    ('raw', 'default_directed', 'expected'),
    [
        (True, True, True),
        (False, True, False),
        (1, True, True),
        (0, True, False),
        (np.int64(1), False, True),
        (np.int64(0), True, False),
        ('true', False, True),
        ('TRUE', False, True),
        ('t', False, True),
        ('yes', False, True),
        ('y', False, True),
        ('directed', False, True),
        ('dir', False, True),
        ('false', True, False),
        ('f', True, False),
        ('no', True, False),
        ('n', True, False),
        ('undirected', True, False),
        ('undir', True, False),
        ('u', True, False),
        (None, True, True),
        (None, False, False),
        ('garbage', True, True),
        ('garbage', False, False),
    ],
)
def test_directed_value_coercion(raw, default_directed, expected):
    df = pd.DataFrame(
        {
            'source': ['A'],
            'target': ['B'],
            'is_directed': [raw],
        }
    )

    g = mod.from_omnipath(
        df=df,
        default_directed=default_directed,
        load_vertex_annotations=False,
    )

    assert g.added_edges[0]['edge_directed'] is expected


def test_missing_directed_column_falls_back_to_default():
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g_true = mod.from_omnipath(df=df, default_directed=True, load_vertex_annotations=False)
    g_false = mod.from_omnipath(df=df, default_directed=False, load_vertex_annotations=False)

    assert g_true.added_edges[0]['edge_directed'] is True
    assert g_false.added_edges[0]['edge_directed'] is False


def test_weight_defaults_to_1_when_missing():
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)
    assert g.added_edges[0]['weight'] == 1.0


def test_weight_defaults_to_1_when_null():
    df = pd.DataFrame(
        {
            'source': ['A', 'C'],
            'target': ['B', 'D'],
            'weight': [None, np.nan],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)
    assert [edge['weight'] for edge in g.added_edges] == [1.0, 1.0]


def test_weight_is_cast_to_float():
    df = pd.DataFrame(
        {
            'source': ['A'],
            'target': ['B'],
            'weight': ['3.25'],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)
    assert g.added_edges[0]['weight'] == 3.25


def test_edge_id_is_stringified_and_null_becomes_none():
    df = pd.DataFrame(
        {
            'source': ['A', 'C', 'E'],
            'target': ['B', 'D', 'F'],
            'edge_id': [101, None, np.nan],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)
    assert [edge['edge_id'] for edge in g.added_edges] == ['101', None, None]


def test_edge_id_preserves_non_integral_float():
    df = pd.DataFrame(
        {
            'source': ['A'],
            'target': ['B'],
            'edge_id': [101.5],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)
    assert [edge['edge_id'] for edge in g.added_edges] == ['101.5']


def test_slice_column_overrides_global_slice():
    df = pd.DataFrame(
        {
            'source': ['A', 'C', 'E'],
            'target': ['B', 'D', 'F'],
            'slice': ['local1', None, np.nan],
        }
    )

    g = mod.from_omnipath(df=df, slice='global', load_vertex_annotations=False)
    assert [edge['slice'] for edge in g.added_edges] == ['local1', 'global', 'global']


def test_global_slice_used_when_no_slice_column():
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g = mod.from_omnipath(df=df, slice='global', load_vertex_annotations=False)
    assert g.added_edges[0]['slice'] == 'global'


def test_edge_attr_cols_none_excludes_structural_fields(edge_df):
    g = mod.from_omnipath(df=edge_df, load_vertex_annotations=False)

    attrs0 = g.added_edges[0]['attributes']
    attrs1 = g.added_edges[1]['attributes']

    assert attrs0 == {'evidence': 'lit'}
    assert attrs1 == {'evidence': 'screen'}


def test_edge_attr_cols_empty_skips_attributes(edge_df):
    g = mod.from_omnipath(df=edge_df, edge_attr_cols=[], load_vertex_annotations=False)
    assert g.added_edges[0]['attributes'] == {}
    assert g.added_edges[1]['attributes'] == {}


def test_edge_attr_cols_subset_uses_only_requested_columns():
    df = pd.DataFrame(
        {
            'source': ['A'],
            'target': ['B'],
            'foo': [1],
            'bar': [2],
            'baz': [3],
        }
    )

    g = mod.from_omnipath(
        df=df,
        edge_attr_cols=['foo', 'baz', 'missing'],
        load_vertex_annotations=False,
    )

    assert g.added_edges[0]['attributes'] == {'foo': 1, 'baz': 3}


def test_registers_vertices_after_adding_edges():
    df = pd.DataFrame(
        {
            'source': ['A', 'A', 'B'],
            'target': ['B', 'C', 'D'],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)

    assert len(g.added_vertices_calls) == 1
    assert set(g.added_vertices_calls[0]) == {'A', 'B', 'C', 'D'}


def test_graph_kwargs_are_forwarded():
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g = mod.from_omnipath(
        df=df,
        load_vertex_annotations=False,
        custom_flag=123,
        another='x',
    )

    assert g.init_kwargs['custom_flag'] == 123
    assert g.init_kwargs['another'] == 'x'


def test_load_vertex_annotations_false_skips_all_annotation_loading():
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)

    # Only vertex registration call should happen.
    assert len(g.added_vertices_calls) == 1
    assert set(g.added_vertices_calls[0]) == {'A', 'B'}


def test_vertex_annotations_df_is_loaded_and_pivoted():
    edges = pd.DataFrame(
        {
            'source': ['EGFR', 'TP53'],
            'target': ['STAT3', 'MDM2'],
        }
    )
    ann = pd.DataFrame(
        [
            {'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'},
            {'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'},
            {'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'name', 'value': 'EGFR'},
            {'genesymbol': 'TP53', 'source': 'IntOGen', 'label': 'driver', 'value': 'yes'},
            {'genesymbol': 'NOT_IN_GRAPH', 'source': 'HGNC', 'label': 'x', 'value': 'y'},
            {'genesymbol': 'STAT3', 'source': None, 'label': 'x', 'value': 'y'},
            {'genesymbol': 'STAT3', 'source': 'HGNC', 'label': None, 'value': 'y'},
            {'genesymbol': 'STAT3', 'source': 'HGNC', 'label': 'x', 'value': None},
        ]
    )

    g = mod.from_omnipath(df=edges, vertex_annotations_df=ann)

    assert len(g.added_vertices_calls) == 2
    annotated = g.added_vertices_calls[1]

    got = dict(annotated)
    assert got == {
        'EGFR': {
            'HGNC:family': 'RTK',
            'HGNC:name': 'EGFR',
        },
        'TP53': {
            'IntOGen:driver': 'yes',
        },
    }


def test_vertex_annotation_sources_filter_is_applied():
    edges = pd.DataFrame({'source': ['EGFR'], 'target': ['TP53']})
    ann = pd.DataFrame(
        [
            {'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'},
            {'genesymbol': 'EGFR', 'source': 'Other', 'label': 'x', 'value': 'y'},
            {'genesymbol': 'TP53', 'source': 'IntOGen', 'label': 'driver', 'value': 'yes'},
        ]
    )

    g = mod.from_omnipath(
        df=edges,
        vertex_annotations_df=ann,
        vertex_annotation_sources=['HGNC'],
    )

    annotated = dict(g.added_vertices_calls[1])
    assert annotated == {
        'EGFR': {'HGNC:family': 'RTK'},
    }


def test_vertex_annotations_df_read_failure_warns_and_continues(capsys):
    class BadNative:
        pass

    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    g = mod.from_omnipath(df=df, vertex_annotations_df=BadNative())
    captured = capsys.readouterr()

    assert '[warning] vertex_annotations_df could not be read:' in captured.out
    assert len(g.added_vertices_calls) == 1


def test_vertex_annotations_path_uses_read_tsv(monkeypatch):
    edges = pd.DataFrame({'source': ['EGFR'], 'target': ['TP53']})
    ann = pd.DataFrame(
        [
            {'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'},
        ]
    )

    seen = {}

    def fake_read_csv(path, sep='\t'):
        seen['path'] = path
        seen['sep'] = sep
        return ann

    monkeypatch.setattr(pd, 'read_csv', fake_read_csv)

    g = mod.from_omnipath(
        df=edges,
        vertex_annotations_path='/tmp/ann.tsv',
        annotations_backend='pandas',
    )

    assert seen['path'] == '/tmp/ann.tsv'
    assert seen['sep'] == '\t'

    annotated = dict(g.added_vertices_calls[1])
    assert annotated == {'EGFR': {'HGNC:family': 'RTK'}}


def test_vertex_annotations_path_failure_warns_and_continues(monkeypatch, capsys):
    edges = pd.DataFrame({'source': ['A'], 'target': ['B']})

    def boom(*args, **kwargs):
        raise OSError('bad file')

    monkeypatch.setattr(pd, 'read_csv', boom)

    g = mod.from_omnipath(
        df=edges,
        vertex_annotations_path='/tmp/missing.tsv',
        annotations_backend='pandas',
    )
    captured = capsys.readouterr()

    assert '[warning] vertex_annotations_path failed:' in captured.out
    assert len(g.added_vertices_calls) == 1


def test_cached_annotation_path_is_used(monkeypatch):
    edges = pd.DataFrame({'source': ['EGFR'], 'target': ['TP53']})
    ann = pd.DataFrame(
        [{'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'}]
    )

    monkeypatch.setitem(sys.modules, 'requests', types.SimpleNamespace())

    fake_os = types.SimpleNamespace(
        path=types.SimpleNamespace(
            exists=lambda p: True,
            join=lambda *parts: '/'.join(parts),
            expanduser=lambda p: '/home/test',
            dirname=lambda p: '/home/test/.cache/annnet',
        ),
        makedirs=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, 'os', fake_os)

    seen = {}

    def fake_read_tsv(path, *, backend='auto'):
        seen['path'] = path
        seen['backend'] = backend
        return ann

    monkeypatch.setattr(mod, 'dataframe_read_tsv', fake_read_tsv)

    g = mod.from_omnipath(df=edges, annotations_backend='pandas')

    assert seen['path'].endswith('/home/test/.cache/annnet/omnipath_annotations.tsv.gz')
    assert seen['backend'] == 'pandas'

    annotated = dict(g.added_vertices_calls[1])
    assert annotated == {'EGFR': {'HGNC:family': 'RTK'}}


def test_download_annotation_path_is_used_when_cache_missing(monkeypatch):
    edges = pd.DataFrame({'source': ['EGFR'], 'target': ['TP53']})
    ann = pd.DataFrame(
        [{'genesymbol': 'EGFR', 'source': 'HGNC', 'label': 'family', 'value': 'RTK'}]
    )

    writes = {}
    seen = {}

    class FakeResp:
        content = b'dummy gz bytes'

        def raise_for_status(self):
            return None

    class DummyFile(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            writes['data'] = self.getvalue()
            return False

    def fake_open(path, mode):
        writes['path'] = path
        writes['mode'] = mode
        return DummyFile()

    fake_requests = types.SimpleNamespace(get=lambda *args, **kwargs: FakeResp())

    fake_os = types.SimpleNamespace(
        path=types.SimpleNamespace(
            exists=lambda p: False,
            join=lambda *parts: '/'.join(parts),
            expanduser=lambda p: '/home/test',
            dirname=lambda p: '/home/test/.cache/annnet',
        ),
        makedirs=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, 'requests', fake_requests)
    monkeypatch.setitem(sys.modules, 'os', fake_os)
    monkeypatch.setattr(mod, 'dataframe_height', lambda df: len(df))

    def fake_read_tsv(source, *, backend='auto'):
        assert isinstance(source, io.BytesIO)
        seen['backend'] = backend
        return ann

    monkeypatch.setattr(mod, 'dataframe_read_tsv', fake_read_tsv)
    monkeypatch.setattr('builtins.open', fake_open)

    g = mod.from_omnipath(df=edges, annotations_backend='pandas')

    assert writes['mode'] == 'wb'
    assert writes['path'].endswith('/home/test/.cache/annnet/omnipath_annotations.tsv.gz')
    assert writes['data'] == b'dummy gz bytes'
    assert seen['backend'] == 'pandas'

    annotated = dict(g.added_vertices_calls[1])
    assert annotated == {'EGFR': {'HGNC:family': 'RTK'}}


def test_annotation_download_failure_warns_and_continues(monkeypatch, capsys):
    edges = pd.DataFrame({'source': ['A'], 'target': ['B']})

    class BoomRequests:
        @staticmethod
        def get(*args, **kwargs):
            raise RuntimeError('network down')

    fake_os = types.SimpleNamespace(
        path=types.SimpleNamespace(
            exists=lambda p: False,
            join=lambda *parts: '/'.join(parts),
            expanduser=lambda p: '/home/test',
            dirname=lambda p: '/home/test/.cache/annnet',
        ),
        makedirs=lambda *args, **kwargs: None,
    )

    monkeypatch.setitem(sys.modules, 'requests', BoomRequests)
    monkeypatch.setitem(sys.modules, 'os', fake_os)

    g = mod.from_omnipath(df=edges)
    captured = capsys.readouterr()

    assert '[warning] vertex annotations download failed:' in captured.out
    assert len(g.added_vertices_calls) == 1


def test_annotation_pivot_failure_warns_and_continues(monkeypatch, capsys):
    edges = pd.DataFrame({'source': ['A'], 'target': ['B']})
    ann = pd.DataFrame([{'genesymbol': 'A', 'source': 'HGNC', 'label': 'x', 'value': 'y'}])

    real_dataframe_to_rows = mod.dataframe_to_rows
    calls = {'n': 0}

    def flaky_dataframe_to_rows(df):
        calls['n'] += 1
        if calls['n'] == 1:
            return real_dataframe_to_rows(df)
        raise RuntimeError('pivot failed')

    monkeypatch.setattr(mod, 'dataframe_to_rows', flaky_dataframe_to_rows)

    g = mod.from_omnipath(df=edges, vertex_annotations_df=ann)
    captured = capsys.readouterr()

    assert '[warning] vertex annotation pivot/load failed:' in captured.out
    assert len(g.added_vertices_calls) == 1


def test_annotation_rows_require_existing_resolved_entity():
    class FakeAnnNetStrict(FakeAnnNet):
        def _resolve_entity_key(self, key):
            return f'resolved::{key}'

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mod, 'AnnNet', FakeAnnNetStrict)

    try:
        edges = pd.DataFrame({'source': ['A'], 'target': ['B']})
        ann = pd.DataFrame(
            [
                {'genesymbol': 'A', 'source': 'HGNC', 'label': 'ok', 'value': '1'},
                {'genesymbol': 'B', 'source': 'HGNC', 'label': 'ok', 'value': '2'},
            ]
        )

        g = mod.from_omnipath(df=edges, vertex_annotations_df=ann)

        # Because _resolve_entity_key returns keys not present in _entities, the filtered list is empty.
        assert len(g.added_vertices_calls) == 2
        assert g.added_vertices_calls[1] == []
    finally:
        monkeypatch.undo()


def install_fake_omnipath(monkeypatch, returned_df):
    calls = {}

    def make_cls(name):
        class _C:
            @classmethod
            def get(cls, **kwargs):
                calls[name] = kwargs
                return returned_df

        _C.__name__ = name
        return _C

    fake_interactions = types.SimpleNamespace(
        OmniPath=make_cls('OmniPath'),
        AllInteractions=make_cls('AllInteractions'),
        PostTranslational=make_cls('PostTranslational'),
        PathwayExtra=make_cls('PathwayExtra'),
        KinaseExtra=make_cls('KinaseExtra'),
        LigRecExtra=make_cls('LigRecExtra'),
        Dorothea=make_cls('Dorothea'),
        TFtarget=make_cls('TFtarget'),
        Transcriptional=make_cls('Transcriptional'),
        TFmiRNA=make_cls('TFmiRNA'),
        miRNA=make_cls('miRNA'),
        lncRNAmRNA=make_cls('lncRNAmRNA'),
        CollecTRI=make_cls('CollecTRI'),
    )

    fake_omnipath = types.SimpleNamespace(interactions=fake_interactions)
    monkeypatch.setitem(sys.modules, 'omnipath', fake_omnipath)
    return calls


@pytest.mark.parametrize(
    ('dataset', 'expected_class'),
    [
        ('omnipath', 'OmniPath'),
        ('pathwayextra', 'PathwayExtra'),
        ('kinaseextra', 'KinaseExtra'),
        ('ligrecextra', 'LigRecExtra'),
        ('dorothea', 'Dorothea'),
        ('tftarget', 'TFtarget'),
        ('transcriptional', 'Transcriptional'),
        ('tfmirna', 'TFmiRNA'),
        ('mirna', 'miRNA'),
        ('lncrnamrna', 'lncRNAmRNA'),
        ('collectri', 'CollecTRI'),
    ],
)
def test_dataset_dispatch_default_signature(monkeypatch, dataset, expected_class):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    calls = install_fake_omnipath(monkeypatch, returned)

    g = mod.from_omnipath(
        dataset=dataset,
        query={'organism': 'human', 'genesymbols': True},
        load_vertex_annotations=False,
    )

    assert calls[expected_class] == {'organism': 'human', 'genesymbols': True}
    assert g._num_edges == 1


def test_dataset_dispatch_all_passes_include_and_exclude(monkeypatch):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    calls = install_fake_omnipath(monkeypatch, returned)

    mod.from_omnipath(
        dataset='all',
        include=['x'],
        exclude=['y'],
        query={'organism': 'human'},
        load_vertex_annotations=False,
    )

    assert calls['AllInteractions'] == {
        'include': ['x'],
        'exclude': ['y'],
        'organism': 'human',
    }


def test_dataset_dispatch_posttranslational_passes_exclude_only(monkeypatch):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    calls = install_fake_omnipath(monkeypatch, returned)

    mod.from_omnipath(
        dataset='posttranslational',
        include=['ignored'],
        exclude=['y'],
        query={'organism': 'mouse'},
        load_vertex_annotations=False,
    )

    assert calls['PostTranslational'] == {
        'exclude': ['y'],
        'organism': 'mouse',
    }


def test_dataset_name_is_normalized(monkeypatch):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    calls = install_fake_omnipath(monkeypatch, returned)

    mod.from_omnipath(
        dataset='post_translational',
        query={'organism': 'human'},
        load_vertex_annotations=False,
    )

    assert calls['PostTranslational'] == {'exclude': None, 'organism': 'human'}


def test_unknown_dataset_raises_value_error(monkeypatch):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    install_fake_omnipath(monkeypatch, returned)

    with pytest.raises(ValueError, match='Unknown dataset'):
        mod.from_omnipath(dataset='does-not-exist', load_vertex_annotations=False)


def test_missing_omnipath_import_raises_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, 'omnipath', raising=False)

    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == 'omnipath':
            raise ImportError('missing package')
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr('builtins.__import__', fake_import)

    with pytest.raises(ImportError, match='omnipath package is required'):
        mod.from_omnipath(df=None, load_vertex_annotations=False)


def test_query_none_becomes_empty_dict(monkeypatch):
    returned = pd.DataFrame({'source': ['A'], 'target': ['B']})
    calls = install_fake_omnipath(monkeypatch, returned)

    mod.from_omnipath(dataset='omnipath', query=None, load_vertex_annotations=False)
    assert calls['OmniPath'] == {}


def test_common_alternate_column_names_are_detected():
    df = pd.DataFrame(
        {
            'source_genesymbol': ['SRC'],
            'target_genesymbol': ['STAT3'],
            'consensus_direction': ['true'],
            'score': [4.0],
            'interaction_id': [42],
            'slice_id': ['sliceX'],
            'extra': ['meta'],
        }
    )

    g = mod.from_omnipath(df=df, load_vertex_annotations=False)

    assert g.added_edges == [
        {
            'source': 'SRC',
            'target': 'STAT3',
            'weight': 4.0,
            'edge_id': '42',
            'edge_directed': True,
            'slice': 'sliceX',
            'attributes': {'extra': 'meta'},
        }
    ]


def test_prints_timing_lines(capsys):
    df = pd.DataFrame({'source': ['A'], 'target': ['B']})

    mod.from_omnipath(df=df, load_vertex_annotations=False)
    out = capsys.readouterr().out

    assert '[timing] fetch/receive df:' in out
    assert '[timing] column resolution:' in out
    assert '[timing] AnnNet init:' in out
    assert '[timing] _to_dicts:' in out
    assert '[timing] bulk list build:' in out
    assert '[timing] add_edges_bulk:' in out
    assert 'vertices=' in out
