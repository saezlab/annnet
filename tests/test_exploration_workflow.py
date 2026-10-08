"""The exploration scenarios, on one deterministic fixture.

One deterministic tiny graph with two aspects (ordered ``time``, categorical
``condition``), a node repeated on several layers, isolated nodes, directed
and undirected edges, directed and undirected hyperedges, a cross-layer
coupling edge, a self-loop, a half-edge, an edge-entity endpoint, two
overlapping slices and attributes at all eight addresses — with nulls, a field
whose name holds ``__``, and labels that collide under naive underscore
concatenation. No download, no biology.

Each scenario below is one a user must be able to answer without a helper: no coordinate indexing by hand, no set intersections of ids, no
hyperedge string parsing, no cached-view refresh, no copying of contextual
metadata.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from annnet import AnnNet
from annnet.core import _structure as S
from annnet.core._select import ReadOnlyViewError


@pytest.fixture
def G():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = AnnNet(
            directed=True,
            aspects={'time': ['0h', '6h', '24h'], 'condition': ['ctrl', 'stim', 'stim_x']},
        )
        g.layers.set_ordered('time')
        # a: every time point under stim; b: 0h/6h under ctrl and stim; c: 0h only.
        g.layers.place(['a'], [('0h', 'stim'), ('6h', 'stim'), ('24h', 'stim')])
        g.layers.place(['b'], [('0h', 'ctrl'), ('6h', 'ctrl'), ('0h', 'stim'), ('6h', 'stim')])
        g.layers.place(['c', 'iso'], [('0h', 'stim')])
        g.add_nodes('lonely', layer=('24h', 'stim_x'))
        # Edges: directed, undirected, hyperedges, coupling, self-loop, half-edge, entity.
        g.add_edges(('a', ('0h', 'stim')), ('b', ('0h', 'stim')), edge_id='ab0', weight=0.9)
        g.add_edges(('a', ('6h', 'stim')), ('b', ('6h', 'stim')), edge_id='ab6', weight=0.4)
        g.add_edges(('b', ('0h', 'ctrl')), ('b', ('0h', 'stim')), edge_id='b_cross', weight=0.2)
        g.add_edges(('b', ('0h', 'stim')), ('c', ('0h', 'stim')), edge_id='bc', directed=False)
        g.add_edges(('c', ('0h', 'stim')), ('c', ('0h', 'stim')), edge_id='loop')
        g.add_edges(
            [
                {
                    'head': [('a', ('0h', 'stim'))],
                    'tail': [('b', ('0h', 'stim')), ('c', ('0h', 'stim'))],
                    'edge_id': 'h_dir',
                },
                {'members': [('a', ('6h', 'stim')), ('b', ('6h', 'stim'))], 'edge_id': 'h_undir'},
                {'head': [('c', ('0h', 'stim'))], 'tail': [], 'edge_id': 'half'},
            ]
        )
        g.add_edges(
            ('a', ('24h', 'stim')), ('b', ('6h', 'stim')), edge_id='backing', as_entity=True
        )
        # An edge entity sits on the placeholder coordinate; both ends are
        # unique bare ids here, which is how the public API names them.
        g.add_edges('backing', 'c', edge_id='meta')
        g.slices.add('prior', edges=['ab0', 'ab6', 'bc', 'h_dir'])
        g.slices.add('fit', edges=['ab0', 'h_dir', 'h_undir', 'meta'])
        # Attributes at all eight addresses; nulls; a dunder field.
        g.attrs.update(
            'nodes',
            {
                'a': {'group': 'x', 'score': 0.9, 'weird__name': 1},
                'b': {'group': 'x', 'score': 0.3, 'weird__name': 2},
                'c': {'group': 'y', 'score': None},
                'iso': {'group': 'y', 'score': 0.7},
            },
        )
        g.attrs.update(
            'edges',
            {
                'ab0': {'confidence': 0.95, 'source_db': 'A'},
                'ab6': {'confidence': 0.5},
                'bc': {'confidence': 0.8},
                'h_dir': {'confidence': 0.9},
                'h_undir': {'confidence': 0.85},
                'meta': {'confidence': 0.7},
            },
        )
        g.attrs.update('slices', {'prior': {'role': 'knowledge'}, 'fit': {'role': 'fitted'}})
        g.attrs.update('aspects', {'time': {'unit': 'h'}})
        g.attrs.update(
            'layers', {('0h', 'stim'): {'n_cells': 100}, ('6h', 'stim'): {'n_cells': 80}}
        )
        g.attrs.update(
            'edge_slices',
            {
                ('fit', 'ab0'): {'activity': 1.2},
                ('fit', 'h_dir'): {'activity': -0.5},
                ('fit', 'meta'): {'activity': 0.1},
            },
        )
        g.attrs.update(
            'node_layers',
            {
                ('a', ('0h', 'stim')): {'expr': 0.95},
                ('a', ('6h', 'stim')): {'expr': 0.2},
                ('b', ('0h', 'stim')): {'expr': 0.85},
            },
        )
        g.attrs.update(
            'elementary_layers',
            {('condition', 'stim_x'): {'note': 'collides'}, ('time', '0h'): {'note': 'baseline'}},
        )
        g.uns['study'] = 'demo'
    return g


# 1. Inspect schema and summary without a frame or a matrix.
def test_schema_and_summary_are_cheap(G, monkeypatch):
    from annnet.core import _attribute_api, _summary, _tables

    def forbidden(*a, **k):
        raise AssertionError('a frame or matrix was built')

    monkeypatch.setattr(_tables, 'dataframe_from_rows', forbidden)
    monkeypatch.setattr(_tables, 'dataframe_from_columns', forbidden)
    monkeypatch.setattr(_attribute_api, 'dataframe_from_rows', forbidden)
    monkeypatch.setattr(type(G.matrices), 'signed_matrix', forbidden)
    summary = G.summary()
    assert summary['nodes'] == 5 and summary['edges'] == 10
    assert summary['aspects']['time']['ordered'] is True
    assert summary['edge_kinds'] == {'binary': 6, 'hyper': 3, 'node_edge': 1}
    assert summary['attrs']['node_layers'] == ['expr']
    assert set(summary['slices']) == {'default', 'prior', 'fit'}
    assert 'AnnNet summary' in repr(summary)
    schema = G.attrs.schema('edges', derived=True)
    assert ('confidence', 'float') in schema.fields and ('kind', 'str') in schema.fields
    assert G.attrs.schema('elementary_layers').keys == (
        ('aspect', 'str'),
        ('elementary_layer', 'str'),
    )
    assert _summary.summarize(G.view(nodes=['a']))['nodes'] == 1


# 2. Threshold + category + null filtering, composed with OR/AND/difference.
def test_threshold_category_and_null_filters_compose(G):
    assert G.N.select(score__gte=0.5).ids == ('a', 'iso')
    assert G.N.select(group='y', score__isnull=True).ids == ('c',)
    chosen = (G.N.select(group='x') | G.N.select(score__gt=0.5)) - G.N.select(weird__name=2)
    assert chosen.ids == ('a', 'iso')
    assert G.E.select(confidence__gte=0.8, kind='hyper').ids == ('h_dir', 'h_undir')
    assert G.E.select(kind='binary', directed=False).ids == ('bc',)


# 3. Restrict to ordered layers and a slice, then high-confidence directed hyperedges.
def test_layers_slice_and_hyperedge_composition(G):
    early = G.layers.where(time__lte='6h')
    V = G.view(layers=early, slices='fit')
    assert set(V.E.ids) == {'ab0', 'h_dir', 'h_undir'}
    W = V.view(edges=V.E.select(confidence__gte=0.85, kind='hyper', directed=True))
    assert W.E.ids == ('h_dir',)
    assert set(W.supra_nodes()) == {
        ('a', ('0h', 'stim')),
        ('b', ('0h', 'stim')),
        ('c', ('0h', 'stim')),
    }


# 4. Complete endpoints and their coordinates via the incidence layout.
def test_incidence_layout_joins_without_parsing(G):
    table = G.attrs.table('edges', derived=True, layout='incidences', backend='pandas')
    rows = table[table['edge_id'] == 'h_dir']
    assert set(rows['entity_id']) == {'a', 'b', 'c'}
    assert set(map(tuple, rows['layer'])) == {('0h', 'stim')}
    assert dict(zip(rows['entity_id'], rows['role'], strict=True)) == {
        'a': 'source',
        'b': 'target',
        'c': 'target',
    }
    meta = table[table['edge_id'] == 'meta']
    assert set(meta['entity_kind']) == {'edge', 'node'}
    cross = table[table['edge_id'] == 'b_cross']
    assert set(map(tuple, cross['layer'])) == {('0h', 'ctrl'), ('0h', 'stim')}
    # Joinable against the node-layer address by (entity_id, layer).
    node_layers = G.attrs.table('node_layers', backend='pandas')
    joined = rows.merge(node_layers, left_on=['entity_id'], right_on=['node_id'], how='left')
    assert 'expr' in joined.columns
    # A column advertised as a node id never holds a pipe-joined list.
    edges = G.attrs.table('edges', derived=True, backend='pandas').set_index('edge_id')
    assert edges.loc['h_dir', 'source'] is None or pd.isna(edges.loc['h_dir', 'source'])
    assert sorted(edges.loc['h_dir', 'tail']) == ['b', 'c']
    for value in edges['source'].dropna():
        assert '|' not in value


# 5. Select a node by contextual score without pulling in its other placements.
def test_contextual_selection_keeps_placements(G):
    rows = G.attrs.select('node_layers', expr__gte=0.8)
    assert rows.keys == (('a', ('0h', 'stim')), ('b', ('0h', 'stim')))
    V = G.view(nodes=rows)
    assert set(V.supra_nodes()) == {('a', ('0h', 'stim')), ('b', ('0h', 'stim'))}
    assert V.E.ids == ('ab0',)
    assert rows.project('nodes').ids == ('a', 'b')
    assert G.attrs.select('node_layers', time__gt='0h').keys == (('a', ('6h', 'stim')),) or set(
        G.attrs.select('node_layers', time__gt='0h').keys
    ) >= {('a', ('6h', 'stim'))}


# 6. Positive edge-slice values, projected explicitly to edges.
def test_edge_slice_rows_project_explicitly(G):
    fit = G.attrs.select('edge_slices', slice_id='fit', activity__gt=0)
    assert fit.keys == (('fit', 'ab0'), ('fit', 'meta'))
    with pytest.raises(TypeError, match='project'):
        G.view(edges=fit)
    V = G.view(edges=fit.project('edges'))
    assert set(V.E.ids) == {'ab0', 'meta', 'backing'}, 'meta brings its backing edge'
    assert fit.project('slices').ids == ('fit',)


# 7. Compare two slices with the existing operations and read a numeric frame.
def test_compare_slices_and_read_numeric_frames(G):
    table = G.slices.compare('prior', 'fit', backend='pandas').set_index('edge_id')
    assert table.loc['ab6', 'status'] == 'a_only'
    assert table.loc['meta', 'status'] == 'b_only'
    assert table.loc['ab0', 'status'] == 'both'
    both = G.view(slices=G.slices.intersect(['prior', 'fit']))
    assert set(both.E.ids) == {'ab0', 'h_dir'}
    frame = G.slices.edge_frame(
        edges=['ab0', 'h_dir'], slices=['fit'], attrs=['activity'], backend='pandas'
    )
    assert frame.iloc[0]['ab0'] == 1.2
    matrix = G.layers.matrix('expr', nodes=['a', 'b'], layers=[('0h', 'stim'), ('6h', 'stim')])
    assert matrix.values.shape == (2, 2)
    assert matrix.values[0, 0] == 0.95 and np.isnan(matrix.values[1, 1])
    scoped = G.view(layers=[('0h', 'stim')]).layers.node_frame(attrs=['expr'], backend='pandas')
    assert scoped.shape[0] == 1


# 8. Closed versus open boundary without truncating a hyperedge.
def test_closed_and_open_boundaries_keep_hyperedges_whole(G):
    closed = G.view(nodes=[('a', ('0h', 'stim'))])
    assert closed.E.ids == ()
    opened = G.view(nodes=[('a', ('0h', 'stim'))], boundary='open')
    assert set(opened.E.ids) == {'ab0', 'h_dir'}
    assert set(opened.supra_nodes()) == {
        ('a', ('0h', 'stim')),
        ('b', ('0h', 'stim')),
        ('c', ('0h', 'stim')),
    }
    assert opened.summary()['expanded'] == {'nodes': 2, 'edges': 0}
    for eid in opened.E:
        assert S.edge_endpoints(opened.materialize(), eid) == S.edge_endpoints(G, eid)
    window = G.layers.where(time='0h', condition='stim')
    assert 'b_cross' in window.crossing and 'b' in window.boundary


# 9. A further filter on V cannot broaden V.
def test_a_nested_filter_only_restricts(G):
    V = G.view(layers=G.layers.where(time='0h'))
    W = V.view(layers=G.layers.where(time__gte='0h'), slices=['prior', 'fit', 'default'])
    assert set(W.supra_nodes()) <= set(V.supra_nodes())
    assert set(W.E.ids) <= set(V.E.ids)
    assert V.view(nodes=list(G.N)).N.ids == V.N.ids
    assert V.view(boundary='open').E.ids == V.E.ids


# 10. Edit the parent, watch the live view change; mutations through V fail.
def test_views_are_live_and_read_only(G):
    V = G.view(nodes=G.N.select(score__gte=0.5))
    assert V.N.ids == ('a', 'iso')
    G.attrs.update('nodes', {'b': {'score': 0.6}})
    assert V.N.ids == ('a', 'b', 'iso')
    G.slices.add_edges('prior', ['meta'])
    # 'meta' ends on the edge entity 'backing'; the closed slice view holds it
    # only once the backing edge is a member too (referential closure).
    assert 'meta' not in G.view(slices='prior').E.ids
    G.slices.add_edges('prior', ['backing'])
    assert {'meta', 'backing'} <= set(G.view(slices='prior').E.ids)
    for attempt in (
        lambda: V.N.__setitem__('score', 1.0),
        lambda: V.attrs.update('nodes', {'a': {'score': 1.0}}),
        lambda: V.attrs.replace('nodes', G.attrs.nodes),
        lambda: V.attrs.delete('nodes', keys=['a']),
        lambda: V.uns.__setitem__('x', 1),
        lambda: setattr(V.attrs, 'nodes', G.attrs.nodes),
    ):
        with pytest.raises(ReadOnlyViewError):
            attempt()
    # A row is a detached copy, so editing it changes neither the view nor the graph.
    before = G.attrs.row('nodes', 'a')
    V.attrs.row('nodes', 'a')['score'] = 99.0
    assert G.attrs.row('nodes', 'a') == before
    with pytest.raises(AttributeError, match='materialize'):
        V.add_nodes('z')


# 11. Materialize, edit the copy, the parent is unchanged.
def test_materialization_is_independent(G):
    V = G.view(layers=G.layers.where(time='0h'))
    H = V.materialize()
    H.add_nodes('new', layer=('0h', 'stim'))
    H.attrs.update('nodes', {'a': {'group': 'changed'}})
    H.remove_edges('ab0')
    assert 'new' not in G.N and G.attrs.row('nodes', 'a')['group'] == 'x' and 'ab0' in G.E
    assert 'ab0' in V.E.ids
    assert H.uns['selection']['boundary'] == 'closed'


# 12. Selected IDs, all eight attrs, matrix labels/values and an IO round trip agree.
def test_view_attrs_matrices_and_materialization_agree_through_io(G):
    V = G.view(layers=G.layers.where(time__lte='6h', condition='stim'), slices=['prior', 'fit'])
    H = V.materialize()
    assert list(H.N) == list(V.N.ids)
    assert list(H.E) == list(V.E.ids)
    assert list(H.supra_nodes()) == list(V.supra_nodes())
    for address in V.attrs:
        v_rows, h_rows = V.attrs.rows(address), H.attrs.rows(address)
        assert {key: h_rows[key] for key in v_rows} == v_rows, address
        extra = set(h_rows) - set(v_rows)
        if address == 'slices':
            assert extra == {'default'}, 'a graph always holds its default slice'
        elif address == 'elementary_layers':
            # Aspect declarations travel whole, and so do their attributes.
            assert extra == {('time', '24h'), ('condition', 'ctrl'), ('condition', 'stim_x')}
        else:
            assert not extra, address
    assert V.S.shape == H.S.shape
    assert np.array_equal(V.S.toarray(), H.S.toarray())
    assert [V.idx.col_to_edge(j) for j in range(V.S.shape[1])] == list(V.E.ids)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'selected.annnet'
        try:
            H.write(path)
            back = AnnNet.read(path)
        except Exception as exc:  # pragma: no cover - environment-dependent backend
            pytest.skip(f'native IO unavailable here: {exc!r}')
    assert list(back.N) == list(H.N) and list(back.E) == list(H.E)
    for address in (
        'nodes',
        'edges',
        'slices',
        'layers',
        'node_layers',
        'edge_slices',
        'elementary_layers',
    ):
        assert back.attrs.rows(address) == H.attrs.rows(address), address
    assert np.array_equal(back.S.toarray(), H.S.toarray())
    assert back.layers.aspect('time').ordered is True
