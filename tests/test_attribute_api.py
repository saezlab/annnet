"""The write and selection verbs of ``G.attrs``: ``delete``, ``replace`` and ``from_frame``.

``update`` merges what it names; ``replace`` swaps a whole address for a table;
``delete`` removes attributes and never the structure that carries them.
``from_frame`` turns the rows of a table the caller has filtered into a fixed
selection of the graph's own keys. Every verb behaves the same at all eight
addresses, and a view reads them and refuses to write.
"""

from __future__ import annotations

import warnings

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from annnet import AnnNet
from annnet.core._select import ReadOnlyViewError, RowSelection
from annnet.core._attribute_api import KEY_COLUMNS

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


@pytest.fixture
def flat():
    g = AnnNet(directed=True)
    g.add_nodes(['a', 'b', 'c', 'd'])
    g.attrs.update(
        'nodes',
        {
            'a': {'score': 0.9, 'group': 'x', 'note': 'first'},
            'b': {'score': 0.4, 'group': 'x'},
            'c': {'score': 0.7, 'group': 'y', 'note': 'third'},
            'd': {'group': 'y'},
        },
    )
    return g


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


def test_delete_with_only_keys_clears_those_rows(flat):
    assert flat.attrs.delete('nodes', keys=['a', 'b']) == 2
    assert flat.attrs.rows('nodes') == {
        'a': {},
        'b': {},
        'c': {'score': 0.7, 'group': 'y', 'note': 'third'},
        'd': {'group': 'y'},
    }
    assert list(flat.N) == ['a', 'b', 'c', 'd'], 'the nodes themselves stay'


def test_delete_with_only_names_drops_the_fields_everywhere(flat):
    assert flat.attrs.delete('nodes', names='note') == 2
    assert all('note' not in row for row in flat.attrs.rows('nodes').values())
    assert flat.attrs.row('nodes', 'a') == {'score': 0.9, 'group': 'x'}
    assert 'note' not in flat.attrs.schema('nodes').names()
    assert flat.attrs.delete('nodes', names=['score', 'group']) == 4
    assert flat.attrs.schema('nodes').fields == ()


def test_delete_with_keys_and_names_removes_those_cells(flat):
    assert flat.attrs.delete('nodes', keys=['a', 'b'], names=['group']) == 2
    assert flat.attrs.row('nodes', 'a') == {'score': 0.9, 'note': 'first'}
    assert flat.attrs.row('nodes', 'b') == {'score': 0.4}
    assert flat.attrs.row('nodes', 'c')['group'] == 'y', 'rows outside keys are untouched'


def test_delete_counts_only_rows_that_changed(flat):
    assert flat.attrs.delete('nodes', keys=['a', 'd'], names=['note']) == 1


def test_delete_with_neither_keys_nor_names_is_refused(flat):
    before = flat.attrs.rows('nodes')
    with pytest.raises(ValueError, match='neither keys nor names'):
        flat.attrs.delete('nodes')
    assert flat.attrs.rows('nodes') == before


@pytest.mark.parametrize(
    'keys,names', [([], None), (None, []), ([], []), ((), 'note'), ([], ['note'])]
)
def test_an_explicit_empty_collection_is_a_no_op_never_everything(flat, keys, names):
    before = flat.attrs.rows('nodes')
    if keys is None:
        assert flat.attrs.delete('nodes', names=names) == 0
    elif names is None:
        assert flat.attrs.delete('nodes', keys=keys) == 0
    else:
        assert flat.attrs.delete('nodes', keys=keys, names=names) == 0
    assert flat.attrs.rows('nodes') == before


def test_delete_validates_before_it_writes(flat):
    before = flat.attrs.rows('nodes')
    with pytest.raises(KeyError, match='ghost'):
        flat.attrs.delete('nodes', keys=['a', 'ghost'])
    with pytest.raises(KeyError, match='nope'):
        flat.attrs.delete('nodes', names=['note', 'nope'])
    with pytest.raises(ValueError, match='reserved'):
        flat.attrs.delete('nodes', names=['node_id'])
    with pytest.raises(TypeError, match='strings'):
        flat.attrs.delete('nodes', names=[1])
    assert flat.attrs.rows('nodes') == before


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_delete_works_the_same_at_every_address(graph, address, key):
    graph.attrs.update(address, {key: {'x': 1, 'y': 2}})
    assert graph.attrs.delete(address, names='x') == 1
    assert graph.attrs.row(address, key) == {'y': 2}
    assert graph.attrs.delete(address, keys=[key]) == 1
    assert graph.attrs.row(address, key) == {}
    assert graph.attrs.delete(address, keys=[key]) == 0


def test_delete_accepts_a_selection_as_keys(flat):
    flat.attrs.delete('nodes', keys=flat.attrs.select('nodes', group='y'))
    assert flat.attrs.row('nodes', 'c') == {} and flat.attrs.row('nodes', 'a')['group'] == 'x'


def test_delete_never_removes_structure(graph):
    graph.attrs.update('edges', {'e': {'x': 1}})
    graph.attrs.delete('edges', keys=['e'])
    graph.attrs.delete('slices', keys=['s'])
    graph.attrs.delete('node_layers', keys=[('n', ('a',))])
    assert list(graph.E) == ['e'] and list(graph.N) == ['n', 'm']
    assert graph.slices.list(include_default=False) == ['s']
    assert 'e' in graph.slices.edges('s')


def test_delete_fires_the_same_policies_as_update_and_rolls_back(monkeypatch):
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    G.add_edges('a', 'b', edge_id='e', flexible={'var': 'x', 'threshold': 0.5, 'scope': 'node'})
    G.attrs.update('nodes', {'a': {'x': 0.9}, 'b': {'x': 0.1}})

    def boom(self, edge_id):
        raise RuntimeError('policy failed')

    monkeypatch.setattr(type(G), '_apply_flexible_direction', boom)
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.delete('nodes', names=['x'])
    assert G.attrs.row('nodes', 'a') == {'x': 0.9}


# ---------------------------------------------------------------------------
# replace
# ---------------------------------------------------------------------------


def test_replace_swaps_the_whole_address_and_keeps_the_structure(flat):
    flat.attrs.replace('nodes', pd.DataFrame({'node_id': ['a', 'd'], 'fresh': [1, 2]}))
    assert flat.attrs.rows('nodes') == {'a': {'fresh': 1}, 'b': {}, 'c': {}, 'd': {'fresh': 2}}
    assert list(flat.N) == ['a', 'b', 'c', 'd']
    assert flat.attrs.schema('nodes').names() == ['node_id', 'fresh']


@pytest.mark.parametrize('make', [pd.DataFrame, pl.DataFrame, lambda d: pa.table(d)])
def test_replace_takes_a_table_of_any_backend(flat, make):
    flat.attrs.replace('nodes', make({'node_id': ['b', 'c'], 'v': [10, 20]}))
    assert flat.attrs.rows('nodes') == {'a': {}, 'b': {'v': 10}, 'c': {'v': 20}, 'd': {}}


def test_replace_round_trips_a_table_read_from_the_graph(flat):
    before = flat.attrs.rows('nodes')
    flat.attrs.replace('nodes', flat.attrs.table('nodes'))
    assert flat.attrs.rows('nodes') == before


def test_replace_treats_null_cells_as_absent(flat):
    frame = pd.DataFrame({'node_id': ['a', 'b'], 'v': [1.0, None]})
    flat.attrs.replace('nodes', frame)
    assert flat.attrs.row('nodes', 'b') == {}


def test_a_failed_replace_leaves_the_attributes_as_they_were(flat):
    before = flat.attrs.rows('nodes')
    for frame, error in (
        (pd.DataFrame({'node_id': ['a', 'a'], 'v': [1, 2]}), ValueError),
        (pd.DataFrame({'node_id': ['a', 'ghost'], 'v': [1, 2]}), KeyError),
        (pd.DataFrame({'v': [1]}), ValueError),
        (pd.DataFrame({'node_id': [None], 'v': [1]}), ValueError),
    ):
        with pytest.raises(error):
            flat.attrs.replace('nodes', frame)
        assert flat.attrs.rows('nodes') == before
        assert flat.attrs.schema('nodes').names() == ['node_id', 'score', 'group', 'note']


# ---------------------------------------------------------------------------
# from_frame
# ---------------------------------------------------------------------------


def test_from_frame_selects_the_rows_of_a_filtered_frame(flat):
    table = flat.attrs.table('nodes', backend='pandas')
    kept = table[table['score'] > 0.5]
    chosen = flat.attrs.from_frame('nodes', kept)
    assert isinstance(chosen, RowSelection)
    assert chosen.address == 'nodes'
    assert chosen.keys == ('a', 'c')
    assert list(chosen.project('nodes')) == ['a', 'c']


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
def test_from_frame_reads_key_columns_from_any_backend(flat, backend):
    frame = flat.attrs.table('nodes', backend=backend)
    assert flat.attrs.from_frame('nodes', frame).keys == ('a', 'b', 'c', 'd')


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_from_frame_keeps_composite_keys_at_every_address(graph, address, key):
    graph.attrs.update(address, {key: {'score': 1}})
    frame = graph.attrs.table(address, backend='polars')
    chosen = graph.attrs.from_frame(address, frame)
    assert chosen.address == address
    assert graph.attrs._key(address, key) in chosen.keys
    assert chosen.rows()[graph.attrs._key(address, key)] == {'score': 1}
    keyed = frame.select(list(KEY_COLUMNS[address]))
    assert graph.attrs.from_frame(address, keyed).keys == chosen.keys


def test_from_frame_returns_the_graphs_order_without_repeats(flat):
    frame = pd.DataFrame({'node_id': ['c', 'a', 'c', 'a']})
    assert flat.attrs.from_frame('nodes', frame).keys == ('a', 'c')


def test_an_empty_frame_is_an_empty_selection_not_everything(flat):
    table = flat.attrs.table('nodes', backend='pandas')
    empty = flat.attrs.from_frame('nodes', table.iloc[0:0])
    assert empty.keys == () and len(empty) == 0
    assert flat.view(nodes=empty.project('nodes')).N.ids == ()


def test_from_frame_is_fixed_not_the_condition_that_made_the_frame(flat):
    table = flat.attrs.table('nodes', backend='pandas')
    chosen = flat.attrs.from_frame('nodes', table[table['score'] > 0.5])
    flat.add_nodes(['late'])
    flat.attrs.update('nodes', {'late': {'score': 1.0}, 'b': {'score': 0.99}})
    assert chosen.keys == ('a', 'c'), 'a node added later is not picked up, and neither is b'


def test_from_frame_identities_do_not_follow_a_reused_storage_slot(flat):
    chosen = flat.attrs.from_frame('nodes', pd.DataFrame({'node_id': ['b']}))
    flat.remove_nodes(['b'])
    assert chosen.keys == ()
    flat.add_nodes(['fresh'])
    assert chosen.keys == (), 'a new node does not inherit the slot the removed one held'
    flat.add_nodes(['b'])
    assert chosen.keys == ('b',), 'the same identity is the same key'


def test_from_frame_validates_keys(flat):
    with pytest.raises(ValueError, match='key column'):
        flat.attrs.from_frame('nodes', pd.DataFrame({'score': [1]}))
    with pytest.raises(ValueError, match='null'):
        flat.attrs.from_frame('nodes', pd.DataFrame({'node_id': [None]}))
    with pytest.raises(KeyError, match='ghost'):
        flat.attrs.from_frame('nodes', pd.DataFrame({'node_id': ['a', 'ghost']}))
    with pytest.raises(KeyError, match='unknown attribute address'):
        flat.attrs.from_frame('nowhere', pd.DataFrame({'node_id': ['a']}))


def test_from_frame_composes_with_live_selections(flat):
    table = flat.attrs.table('nodes', backend='pandas')
    fixed = flat.attrs.from_frame('nodes', table[table['group'] == 'x'])
    live = flat.attrs.select('nodes', score__gt=0.5)
    assert (fixed & live).keys == ('a',)
    assert (fixed | live).keys == ('a', 'b', 'c')
    assert (fixed - live).keys == ('b',)
    flat.attrs.update('nodes', {'b': {'score': 0.8}})
    assert (fixed & live).keys == ('a', 'b'), (
        'the live side follows the graph, the fixed side does not'
    )


def test_from_frame_node_layer_rows_project_to_nodes(graph):
    graph.layers.place(['n'], [('b',)])
    graph.attrs.update('node_layers', {('n', ('a',)): {'score': 1}, ('n', ('b',)): {'score': 5}})
    table = graph.attrs.table('node_layers', backend='pandas')
    chosen = graph.attrs.from_frame('node_layers', table[table['score'] > 2])
    assert chosen.keys == (('n', ('b',)),)
    assert list(chosen.project('nodes')) == ['n']


def test_a_from_frame_selection_drives_a_view(flat):
    flat.add_edges('a', 'b', edge_id='ab')
    flat.add_edges('b', 'c', edge_id='bc')
    table = flat.attrs.table('nodes', backend='pandas')
    chosen = flat.attrs.from_frame('nodes', table[table['group'] == 'x'])
    view = flat.view(nodes=chosen.project('nodes'))
    assert view.N.ids == ('a', 'b') and view.E.ids == ('ab',)


# ---------------------------------------------------------------------------
# a view reads and never writes
# ---------------------------------------------------------------------------


def test_a_view_reads_its_own_rows(flat):
    view = flat.view(nodes=flat.N.select(group='x'))
    assert view.attrs.row('nodes', 'a')['score'] == 0.9
    assert set(view.attrs.rows('nodes')) == {'a', 'b'}
    assert view.attrs.table('nodes', backend='pandas')['node_id'].tolist() == ['a', 'b']
    with pytest.raises(KeyError, match='outside this view'):
        view.attrs.row('nodes', 'c')


def test_from_frame_on_a_view_is_scoped_to_the_view(flat):
    view = flat.view(nodes=flat.N.select(group='x'))
    inside = view.attrs.from_frame('nodes', view.attrs.table('nodes', backend='pandas'))
    assert inside.keys == ('a', 'b')
    with pytest.raises(KeyError, match='outside this view'):
        view.attrs.from_frame('nodes', flat.attrs.table('nodes', backend='pandas'))


def test_every_writer_on_a_view_names_materialize(flat):
    view = flat.view(nodes=flat.N.select(group='x'))
    frame = flat.attrs.table('nodes', backend='pandas')
    writers = [
        lambda: view.attrs.update('nodes', {'a': {'score': 1}}),
        lambda: view.attrs.replace('nodes', frame),
        lambda: view.attrs.delete('nodes', keys=['a']),
        lambda: view.attrs.delete('nodes', names=['score']),
        lambda: setattr(view.attrs, 'nodes', frame),
        lambda: setattr(view.attrs, 'backend', 'polars'),
        lambda: view.attrs.__setitem__('nodes', frame),
        lambda: view.attrs.__delitem__(('nodes', 'a')),
    ]
    before = flat.attrs.rows('nodes')
    for write in writers:
        with pytest.raises(
            (ReadOnlyViewError, TypeError), match='materialize|not supported|read-only'
        ):
            write()
    assert flat.attrs.rows('nodes') == before
    for write in writers[:6]:
        with pytest.raises(ReadOnlyViewError, match='materialize'):
            write()


# ---------------------------------------------------------------------------
# nested values are detached too
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_editing_a_nested_value_read_from_a_row_changes_nothing(graph, address, key):
    graph.attrs.update(address, {key: {'meta': {'tags': ['a']}, 'items': [1, 2]}})
    row = graph.attrs.row(address, key)
    row['meta']['tags'].append('changed')
    row['items'].append(3)
    graph.attrs.rows(address)[graph.attrs._key(address, key)]['meta']['tags'].append('x')
    assert graph.attrs.row(address, key) == {'meta': {'tags': ['a']}, 'items': [1, 2]}


@pytest.mark.parametrize('address,key', ADDRESS_KEYS)
def test_editing_a_nested_value_read_from_a_table_changes_nothing(graph, address, key):
    graph.attrs.update(address, {key: {'meta': {'tags': ['a']}}})
    frame = graph.attrs.table(address, backend='pandas')
    for cell in frame['meta']:
        if isinstance(cell, dict):
            cell['tags'].append('changed')
    assert graph.attrs.row(address, key) == {'meta': {'tags': ['a']}}


def test_editing_a_nested_value_read_through_a_view_changes_nothing(flat):
    flat.attrs.update('nodes', {'a': {'meta': {'tags': ['a']}}})
    view = flat.view(nodes=['a'])
    view.attrs.row('nodes', 'a')['meta']['tags'].append('changed')
    assert flat.attrs.row('nodes', 'a')['meta'] == {'tags': ['a']}


def test_editing_a_nested_value_in_a_column_read_changes_nothing(flat):
    flat.attrs.update('nodes', {'a': {'meta': {'tags': ['a']}}})
    for cell in flat.N['meta']:
        if isinstance(cell, dict):
            cell['tags'].append('changed')
    flat.add_edges('a', 'b', edge_id='e')
    flat.attrs.update('edges', {'e': {'meta': {'tags': ['a']}}})
    for cell in flat.E['meta']:
        if isinstance(cell, dict):
            cell['tags'].append('changed')
    assert flat.attrs.row('nodes', 'a')['meta'] == {'tags': ['a']}
    assert flat.attrs.row('edges', 'e')['meta'] == {'tags': ['a']}
