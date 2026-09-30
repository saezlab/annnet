"""One address, one write model, at all eight attribute levels.

Every test states a semantic: the same expression at every address, a batch
that is all-or-nothing, null as deletion, detached snapshots, structured
elementary-layer keys, and parity across pandas, Polars and PyArrow.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from annnet import AnnNet
from annnet.core._attribute_api import ADDRESSES, KEY_COLUMNS


@pytest.fixture
def graph():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = AnnNet(aspects={'condition': ['a', 'b']})
        g.add_nodes(['n', 'm'], layer=('a',))
        g.add_edges(('n', ('a',)), ('m', ('a',)), edge_id='e')
        g.slices.add('s')
        g.slices.add_edges('s', ['e'])
    return g


ADDRESS_KEYS = [
    ('nodes', 'n'),
    ('edges', 'e'),
    ('slices', 's'),
    ('aspects', 'condition'),
    ('layers', ('a',)),
    ('edge_slices', ('s', 'e')),
    ('node_layers', ('n', ('a',))),
    ('elementary_layers', ('condition', 'a')),
]

BACKENDS = ['pandas', 'polars', 'pyarrow']


def _rows(frame, backend):
    """Rows of a frame in any backend, as dicts."""
    if backend == 'pandas':
        return frame.to_dict(orient='records')
    if backend == 'polars':
        return frame.to_dicts()
    return frame.to_pylist()


# ---------------------------------------------------------------------------
# one shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_one_row_read_merge_and_delete(graph, address, key):
    assert graph.attrs.row(address, key) == {}
    graph.attrs.update(address, {key: {'score': 3}})
    assert graph.attrs.row(address, key)['score'] == 3
    graph.attrs.update(address, {key: {'score': 4, 'note': 'kept'}})
    assert graph.attrs.row(address, key) == {'score': 4, 'note': 'kept'}
    graph.attrs.delete(address, keys=[key], names=['note'])
    assert graph.attrs.row(address, key) == {'score': 4}
    # Clear the row: nothing left, and the element stays.
    graph.attrs.delete(address, keys=[key])
    assert graph.attrs.row(address, key) == {}
    graph.attrs.update(address, {key: {'score': 1}})
    assert graph.attrs.row(address, key) == {'score': 1}


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_a_row_is_a_detached_dict(graph, address, key):
    graph.attrs.update(address, {key: {'score': 3}})
    row = graph.attrs.row(address, key)
    assert type(row) is dict
    row['score'] = 99
    row['extra'] = 1
    assert graph.attrs.row(address, key) == {'score': 3}
    graph.attrs.update(address, {key: {'score': 5}})
    assert row['score'] == 99, 'a row read earlier does not follow the graph'
    assert row.get('missing', 'default') == 'default'


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_update_merges_and_null_deletes(graph, address, key):
    assert graph.attrs.update(address, {key: {'a': 1, 'b': 2}}) == 1
    assert graph.attrs.update(address, {key: {'a': None, 'c': math.nan}}) == 1
    assert graph.attrs.row(address, key) == {'b': 2}
    assert graph.attrs.rows(address, key) == {graph.attrs._key(address, key): {'b': 2}}


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
@pytest.mark.parametrize('backend', BACKENDS)
def test_whole_table_read_and_replace(graph, address, key, backend):
    graph.attrs.backend = backend
    graph.attrs.update(address, {key: {'score': 3}})
    frame = graph.attrs.table(address, backend='pandas')
    assert list(KEY_COLUMNS[address]) == [c for c in frame.columns if c in KEY_COLUMNS[address]]
    assert 3 in frame['score'].tolist()
    frame['score'] = 7
    graph.attrs.replace(address, frame)
    assert graph.attrs.row(address, key)['score'] == 7
    rows = _rows(graph.attrs.table(address), backend)
    assert all(row['score'] == 7 for row in rows)
    # The property is the same read as the dynamic spelling.
    same = _rows(getattr(graph.attrs, address), backend)
    assert same == rows


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_an_omitted_table_row_clears_that_rows_attributes(graph, address, key):
    graph.attrs.update(address, {key: {'score': 3}})
    empty = graph.attrs.table(address, backend='pandas').iloc[0:0]
    graph.attrs.replace(address, empty)
    assert graph.attrs.row(address, key) == {}
    # The element itself is still there: membership is structural.
    graph.attrs.update(address, {key: {'score': 1}})
    assert graph.attrs.row(address, key) == {'score': 1}


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_tables_and_rows_are_detached_snapshots(graph, address, key):
    graph.attrs.update(address, {key: {'score': 3}})
    frame = graph.attrs.table(address, backend='pandas')
    frame['score'] = 99
    assert graph.attrs.row(address, key)['score'] == 3
    for attrs in graph.attrs.rows(address).values():
        attrs['score'] = 100
    assert graph.attrs.row(address, key)['score'] == 3


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_removed_subscription_forms_name_their_replacement(graph, address, key):
    with pytest.raises(TypeError, match='row'):
        graph.attrs[address, key]
    with pytest.raises(TypeError, match=f'table|{address}'):
        graph.attrs[address]
    with pytest.raises(TypeError, match='update'):
        graph.attrs[address, key] = {'x': 1}
    with pytest.raises(TypeError, match='replace'):
        graph.attrs[address] = graph.attrs.table(address)
    with pytest.raises(TypeError, match='delete'):
        del graph.attrs[address, key]
    with pytest.raises(TypeError, match='replace'):
        setattr(graph.attrs, address, graph.attrs.table(address))
    assert graph.attrs.row(address, key) == {}, 'a refused form writes nothing'


def test_every_address_is_listed_and_the_namespace_is_small(graph):
    assert tuple(graph.attrs) == ADDRESSES
    assert len(graph.attrs) == 8
    assert set(dir(graph.attrs)) == {
        *ADDRESSES,
        'audit',
        'backend',
        'delete',
        'from_frame',
        'replace',
        'row',
        'rows',
        'schema',
        'select',
        'table',
        'update',
    }
    for old in ('set_node_attrs', 'get_node_attrs', 'set_graph_attribute', 'get_attr_node', 'get'):
        with pytest.raises(AttributeError):
            getattr(graph.attrs, old)


def test_unknown_address_and_bad_keys_raise(graph):
    with pytest.raises(KeyError, match='unknown attribute address'):
        graph.attrs.row('nowhere', 'x')
    with pytest.raises(KeyError, match='unknown attribute address'):
        graph.attrs.table('nowhere')
    with pytest.raises(KeyError, match='unknown node'):
        graph.attrs.row('nodes', 'ghost')
    with pytest.raises(KeyError, match='unknown edge'):
        graph.attrs.row('edge_slices', ('s', 'ghost'))
    with pytest.raises(KeyError, match='unknown slice'):
        graph.attrs.row('edge_slices', ('nope', 'e'))
    with pytest.raises(KeyError, match='not placed'):
        graph.attrs.row('node_layers', ('n', ('b',)))
    with pytest.raises(TypeError, match='bare node id needs layer'):
        graph.attrs.row('node_layers', 'n')
    with pytest.raises(KeyError, match='unknown elementary layer'):
        graph.attrs.row('elementary_layers', ('condition', 'zz'))
    with pytest.raises(KeyError, match='unknown aspect'):
        graph.attrs.row('aspects', 'zz')


# ---------------------------------------------------------------------------
# atomicity
# ---------------------------------------------------------------------------


def test_unknown_final_key_leaves_no_partial_write(graph):
    graph.attrs.update('nodes', {'n': {'score': 1}})
    with pytest.raises(KeyError):
        graph.attrs.update('nodes', {'n': {'score': 2}, 'absent': {'score': 3}})
    assert graph.attrs.row('nodes', 'n')['score'] == 1
    with pytest.raises(KeyError):
        graph.attrs.update('node_layers', {('n', ('a',)): {'x': 1}, ('n', ('b',)): {'x': 1}})
    assert dict(graph.attrs.row('node_layers', ('n', ('a',)))) == {}


def test_normalized_duplicate_keys_are_rejected_before_writing(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update('layers', {('a',): {'x': 1}, 'a': {'x': 2}})
    assert dict(graph.attrs.row('layers', ('a',))) == {}
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update('nodes', [('n', {'x': 1}), ('n', {'x': 2})])
    assert dict(graph.attrs.row('nodes', 'n')) == {}


def test_incompatible_value_types_are_rejected(graph):
    with pytest.raises(TypeError, match='mapping'):
        graph.attrs.update('nodes', {'n': 3})
    with pytest.raises(TypeError, match='strings'):
        graph.attrs.update('nodes', {'n': {1: 'x'}})
    with pytest.raises(TypeError, match='weight is a number'):
        graph.attrs.update('edge_slices', {('s', 'e'): {'weight': 'heavy'}})
    assert dict(graph.attrs.row('edge_slices', ('s', 'e'))) == {}


def test_reserved_fields_are_rejected_at_every_address(graph):
    for address, key in ADDRESS_KEYS:
        for name in KEY_COLUMNS[address]:
            with pytest.raises(ValueError, match='reserved'):
                graph.attrs.update(address, {key: {name: 'x'}})
    with pytest.raises(ValueError, match='reserved'):
        graph.attrs.update('edges', {'e': {'source': 'x'}})
    # weight is allowed only where an override lives.
    with pytest.raises(ValueError, match='reserved'):
        graph.attrs.update('edges', {'e': {'weight': 2.0}})
    graph.attrs.update('edge_slices', {('s', 'e'): {'weight': 2.5}})
    assert graph.E.effective_weight('e', slice='s') == 2.5


def test_composite_key_collision_is_preflighted():
    G = AnnNet()
    G.add_nodes(['a', 'b'])
    G.set_node_key('symbol')
    G.attrs.update('nodes', {'a': {'symbol': 'X'}, 'b': {'symbol': 'Y'}})
    with pytest.raises(ValueError, match='Composite key collision'):
        G.attrs.update('nodes', {'b': {'symbol': 'X'}})
    assert G.attrs.row('nodes', 'a')['symbol'] == 'X'
    assert G.attrs.row('nodes', 'b')['symbol'] == 'Y'
    assert G._node_key_index == {('X',): 'a', ('Y',): 'b'}
    # A batch is judged by its final state: a swap is not a collision.
    G.attrs.update('nodes', {'a': {'symbol': 'Y'}, 'b': {'symbol': 'X'}})
    assert G._node_key_index == {('Y',): 'a', ('X',): 'b'}
    with pytest.raises(ValueError, match='Composite key collision'):
        G.attrs.update('nodes', {'a': {'symbol': 'Q'}, 'b': {'symbol': 'Q'}})
    assert G._node_key_index == {('Y',): 'a', ('X',): 'b'}


def test_a_failing_flexible_direction_callback_rolls_the_batch_back(monkeypatch):
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    G.add_edges('a', 'b', edge_id='e', flexible={'var': 'x', 'threshold': 0.5, 'scope': 'node'})
    G.attrs.update('nodes', {'a': {'x': 0.9}, 'b': {'x': 0.1}})
    before = dict(G.attrs.row('nodes', 'a')), dict(G.attrs.row('nodes', 'b'))
    version = G._store.structure_version

    def boom(edge_id):
        raise RuntimeError('policy failed')

    monkeypatch.setattr(type(G), '_apply_flexible_direction', lambda self, eid: boom(eid))
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.update('nodes', {'a': {'x': 0.1}, 'b': {'x': 0.9}})
    assert (dict(G.attrs.row('nodes', 'a')), dict(G.attrs.row('nodes', 'b'))) == before
    assert G._store.structure_version == version


def test_an_invalid_table_does_not_destroy_existing_attrs(graph):
    graph.attrs.update('nodes', {'n': {'score': 1}})
    with pytest.raises(ValueError, match='duplicate'):
        graph.attrs.replace('nodes', pd.DataFrame({'node_id': ['n', 'n'], 'score': [2, 3]}))
    assert graph.attrs.row('nodes', 'n')['score'] == 1
    with pytest.raises(KeyError):
        graph.attrs.replace('nodes', pd.DataFrame({'node_id': ['n', 'ghost'], 'score': [2, 3]}))
    assert graph.attrs.row('nodes', 'n')['score'] == 1
    with pytest.raises(ValueError, match='key column'):
        graph.attrs.replace('nodes', pd.DataFrame({'score': [2]}))
    assert graph.attrs.row('nodes', 'n')['score'] == 1


def test_a_filtered_axis_write_resolves_its_targets_once(graph):
    G = AnnNet()
    G.add_nodes(['a', 'b', 'c'])
    G.attrs.update('nodes', {'a': {'group': 'x'}, 'b': {'group': 'x'}, 'c': {'group': 'y'}})
    chosen = G.N.select(group='x')
    chosen['group'] = 'y'
    assert G.N['group'].tolist() == ['y', 'y', 'y']
    assert chosen.ids == (), 'the selection is live and now matches nothing'
    with pytest.raises(KeyError, match='node_id'):
        G.N.select(group='y')['node_id'] = 'z'


# ---------------------------------------------------------------------------
# semantics
# ---------------------------------------------------------------------------


def test_nested_values_round_trip(graph):
    payload = {'list': [1, 2], 'nested': {'k': 'v'}}
    graph.attrs.update('nodes', {'n': {'meta': payload}})
    graph.attrs.update('edge_slices', {('s', 'e'): {'meta': payload}})
    assert graph.attrs.row('nodes', 'n')['meta'] == payload
    assert graph.attrs.row('edge_slices', ('s', 'e'))['meta'] == payload


@pytest.mark.parametrize('backend', BACKENDS)
def test_backend_null_and_dtype_parity(graph, backend):
    graph.attrs.backend = backend
    graph.attrs.update('nodes', {'n': {'f': 1.5, 'i': 2, 's': 'x', 'b': True}})
    frame = graph.attrs.table('nodes', backend='pandas').set_index('node_id')
    assert frame.loc['n', 'f'] == 1.5 and frame.loc['n', 'i'] == 2
    assert frame.loc['n', 's'] == 'x' and bool(frame.loc['n', 'b']) is True
    missing = frame.loc['m']
    assert all(pd.isna(missing[name]) for name in ('f', 'i', 's', 'b'))
    # Writing the table back in this backend keeps the nulls as absence.
    graph.attrs.replace('nodes', graph.attrs.table('nodes'))
    assert dict(graph.attrs.row('nodes', 'm')) == {}
    assert graph.attrs.row('nodes', 'n')['i'] == 2


def test_elementary_layers_use_structured_keys(graph):
    graph.attrs.update('elementary_layers', {('condition', 'a'): {'label': 'my label'}})
    frame = graph.attrs.table('elementary_layers', backend='pandas')
    assert list(frame.columns[:2]) == ['aspect', 'elementary_layer']
    assert 'layer_id' not in frame.columns
    assert frame.iloc[0]['label'] == 'my label'
    # The legacy id still addresses a row when it is unambiguous.
    assert graph.attrs.row('elementary_layers', 'condition_a')['label'] == 'my label'


def test_ambiguous_legacy_elementary_ids_are_rejected():
    G = AnnNet(aspects={'time': ['1', 'x_1'], 'time_x': ['1']})
    G.attrs.update('elementary_layers', {('time', 'x_1'): {'v': 1}})
    G.attrs.update('elementary_layers', {('time_x', '1'): {'v': 2}})
    with pytest.raises(ValueError, match='ambiguous'):
        G.attrs.row('elementary_layers', 'time_x_1')
    assert G.attrs.row('elementary_layers', ('time', 'x_1'))['v'] == 1
    assert G.attrs.row('elementary_layers', ('time_x', '1'))['v'] == 2
    frame = G.layer_attributes
    assert set(frame.columns) >= {'layer_id', 'aspect', 'elementary_layer'}


def test_layers_and_elementary_layers_are_different_addresses(graph):
    graph.attrs.update('layers', {('a',): {'x': 1}})
    graph.attrs.update('elementary_layers', {('condition', 'a'): {'x': 2}})
    assert graph.attrs.row('layers', ('a',))['x'] == 1
    assert graph.attrs.row('elementary_layers', ('condition', 'a'))['x'] == 2
    assert list(graph.attrs.table('layers', backend='pandas').columns) == ['layer', 'x']


def test_node_layer_shorthands(graph):
    assert (
        graph.attrs.update('node_layers', {'n': 0.9, 'm': 0.1}, layer=('a',), key='observed') == 2
    )
    assert graph.attrs.row('node_layers', ('n', ('a',)))['observed'] == 0.9
    with pytest.raises(ValueError, match='layer='):
        graph.attrs.update('node_layers', {'n': {'observed': 1}})
    with pytest.raises(TypeError, match='node_layers only'):
        graph.attrs.update('nodes', {'n': 1}, key='x')


def test_generic_node_attrs_are_shared_across_placements(graph):
    graph.layers.place(['n'], [('b',)])
    graph.attrs.update('nodes', {'n': {'shared': 1}})
    graph.attrs.update('node_layers', {('n', ('b',)): {'own': 2}})
    assert graph.attrs.row('nodes', 'n')['shared'] == 1
    assert dict(graph.attrs.row('node_layers', ('n', ('a',)))) == {}
    assert dict(graph.attrs.row('node_layers', ('n', ('b',)))) == {'own': 2}
    assert graph.N['shared'].tolist() == [1, None]


def test_user_attribute_named_kind_is_not_overwritten():
    G = AnnNet()
    G.add_nodes(['a'], kind='gene')
    assert G.attrs.row('nodes', 'a')['kind'] == 'gene'
    assert G.N.at('a').kind == 'node'
    assert G.attrs.table('nodes', derived=True, backend='pandas').loc[0, 'kind'] == 'gene'


def test_rows_accepts_scalars_collections_and_selections(graph):
    graph.attrs.update('nodes', {'n': {'v': 1}, 'm': {'v': 2}})
    assert graph.attrs.rows('nodes', 'n') == {'n': {'v': 1}}
    assert graph.attrs.rows('nodes', ['m', 'n']) == {'m': {'v': 2}, 'n': {'v': 1}}
    assert graph.attrs.rows('nodes', graph.N.select(v__gt=1)) == {'m': {'v': 2}}
    assert graph.attrs.rows('node_layers', ('n', ('a',))) == {('n', ('a',)): {}}


def test_schema_reports_keys_and_fields_without_a_frame(graph, monkeypatch):
    graph.attrs.update('nodes', {'n': {'score': 0.5, 'label': 'x'}})
    from annnet.core import _attribute_api, _tables

    monkeypatch.setattr(
        _attribute_api,
        'dataframe_from_rows',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('frame')),
    )
    monkeypatch.setattr(
        _tables,
        'dataframe_from_rows',
        lambda *a, **k: (_ for _ in ()).throw(AssertionError('frame')),
    )
    schema = graph.attrs.schema('nodes')
    assert schema.keys == (('node_id', 'str'),)
    assert dict(schema.fields) == {'score': 'float', 'label': 'str'}
    assert schema.rows == 2
    derived = graph.attrs.schema('edges', derived=True)
    assert ('kind', 'str') in derived.fields and ('directed', 'bool') in derived.fields
    assert 'Schema' in repr(schema)


def test_np_scalars_count_as_values(graph):
    graph.attrs.update('nodes', {'n': {'v': np.float64(1.5)}})
    assert graph.attrs.row('nodes', 'n')['v'] == 1.5
    graph.attrs.update('nodes', {'n': {'v': np.nan}})
    assert dict(graph.attrs.row('nodes', 'n')) == {}


# The ordered flag of an aspect is a declaration the aspect registry keeps
# beside the aspect's attributes. The attribute address neither shows nor
# writes it, and a row or table replacement leaves it where it is.
def test_the_ordered_declaration_is_not_an_aspect_attribute():
    import pandas as pd

    G = AnnNet(directed=True, aspects={'time': ['0h', '6h'], 'cond': ['a', 'b']})
    G.layers.set_ordered('time')
    G.attrs.update('aspects', {'time': {'unit': 'h'}})
    assert dict(G.attrs.row('aspects', 'time')) == {'unit': 'h'}
    assert G.attrs.schema('aspects').fields == (('unit', 'str'),)
    assert list(G.attrs.table('aspects', backend='pandas').columns) == ['aspect', 'unit']
    assert G.summary()['attrs']['aspects'] == ['unit']
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.update('aspects', {'time': {'__ordered__': False}})
    G.attrs.delete('aspects', keys=['time'])
    G.attrs.update('aspects', {'time': {'unit': 'hours'}})
    assert G.layers.aspect('time').ordered is True
    G.attrs.replace('aspects', pd.DataFrame([{'aspect': 'cond', 'unit': 'x'}]))
    assert G.layers.aspect('time').ordered is True
    assert G.attrs.rows('aspects') == {'time': {}, 'cond': {'unit': 'x'}}
    with pytest.raises(KeyError):
        G.attrs.update('aspects', {'time': {'unit': 'h'}, 'nope': {'unit': 'x'}})
    assert G.layers.aspect('time').ordered is True and G.attrs.rows('aspects')['time'] == {}
    # The derived table is where the declaration is read as a column.
    derived = G.attrs.table('aspects', derived=True, backend='pandas').set_index('aspect')
    assert (
        bool(derived.loc['time', 'ordered']) is True
        and bool(derived.loc['cond', 'ordered']) is False
    )
