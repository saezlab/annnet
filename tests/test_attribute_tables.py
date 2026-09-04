"""The eight attribute tables, under one namespace and one convention.

They used to carry three spellings: ``G.obs``/``G.var`` for the two generic
axes, ``G.slice_attributes`` and two siblings for some of the contextual ones,
and ``G.contextual_table(level)`` for all six. Same concept — attributes keyed
by an address — reached three different ways, with the read side in a different
namespace from the setter that writes it.

They are ``G.attrs.<address>`` now, beside those setters.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet
from annnet.core._Annotation import TABLE_NAMES
from annnet.core._contextual import LEVELS
from annnet._support.dataframe_backend import (
    dataframe_backend,
    dataframe_column_values,
    dataframe_schema,
)


# The key column each table is addressed by, which is what its name refers to.
KEY_COLUMNS = {
    'nodes': ('node_id',),
    'edges': ('edge_id',),
    'slices': ('slice_id',),
    'aspects': ('aspect',),
    'layers': ('layer',),
    'edge_slices': ('slice_id', 'edge_id'),
    'node_layers': ('node_id', 'layer'),
    'elementary_layers': ('layer_id',),
}


@pytest.fixture
def graph() -> AnnNet:
    """A graph carrying something at every one of the eight addresses."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True)
        G.layers.set_aspects(['cond'], {'cond': ['ctl', 'trt']})
        G.add_nodes(['a', 'b'], layer=('ctl',), kind='gene')
        G.add_edges('a', 'b', edge_id='e0', weight=2.0)
        G.slices.add('core')
        G.slices.add_edges('core', ['e0'])
        G.attrs.set_slice_attrs('core', curated=True)
        G.attrs.set_edge_slice_attrs('core', 'e0', confidence=0.9)
        G.layers.set_aspect_attrs('cond', kind='experimental')
        G.layers.set_elementary_attrs('cond', 'trt', dose=5)
        G.layers.set_node_attrs('a', ('ctl',), depth=2)
        G.layers.set_attrs(('ctl',), note='baseline')
    return G


@pytest.mark.parametrize('name', TABLE_NAMES)
def test_every_table_is_reachable_by_its_address(name, graph):
    table = getattr(graph.attrs, name)
    assert table is not None
    for key in KEY_COLUMNS[name]:
        assert key in table.columns, f'{name} is not keyed by {key}'


@pytest.mark.parametrize('name', TABLE_NAMES)
def test_every_table_carries_what_was_written_to_it(name, graph):
    assert len(getattr(graph.attrs, name)) >= 1, f'{name} came back empty'


def test_the_new_names_answer_what_the_old_spellings_answered(graph):
    """Purely additive: every old spelling still resolves to the same table."""
    assert graph.attrs.nodes.columns == graph.obs.columns
    assert graph.attrs.edges.columns == graph.var.columns
    assert graph.attrs.slices.columns == graph.slice_attributes.columns
    assert graph.attrs.edge_slices.columns == graph.edge_slice_attributes.columns
    assert graph.attrs.elementary_layers.columns == graph.layer_attributes.columns


def test_a_layer_coordinate_is_not_an_elementary_layer(graph):
    """``G.layer_attributes`` reads the elementary level, despite its name.

    ``attrs.layers`` is the whole coordinate across every aspect and
    ``attrs.elementary_layers`` is one label inside one aspect. Naming them apart
    is the point of the rename, so nothing may quietly make them one table.
    """
    assert 'layer' in graph.attrs.layers.columns
    assert 'layer_id' in graph.attrs.elementary_layers.columns
    assert graph.attrs.layers.columns != graph.attrs.elementary_layers.columns


def test_every_contextual_level_has_a_table(graph):
    """A level the store gains must gain a name here too."""
    reached = {tuple(KEY_COLUMNS[name]) for name in TABLE_NAMES}
    assert len(TABLE_NAMES) == len(LEVELS) + 2, (
        f'{len(LEVELS)} contextual levels plus nodes and edges should be '
        f'{len(LEVELS) + 2} tables, but TABLE_NAMES holds {len(TABLE_NAMES)}'
    )
    assert len(reached) == len(TABLE_NAMES), 'two tables share a key shape'


@pytest.mark.parametrize('backend', ['polars', 'pandas', 'pyarrow'])
def test_the_backend_is_ambient_and_every_table_follows_it(backend, graph):
    """It picks the container, never the content, so it is set once."""
    graph.attrs.backend = backend
    assert graph.attrs.backend == backend
    for name in TABLE_NAMES:
        assert dataframe_backend(getattr(graph.attrs, name)) == backend, name


def test_one_table_can_be_asked_for_in_a_backend_of_its_own(graph):
    assert dataframe_backend(graph.attrs.table('nodes', backend='pandas')) == 'pandas'
    # and the ambient one is left where it was
    assert graph.attrs.backend != 'pandas'
    assert dataframe_backend(graph.attrs.slices) == graph.attrs.backend


def test_table_without_a_backend_is_the_property(graph):
    assert graph.attrs.table('slices').columns == graph.attrs.slices.columns


def test_an_unknown_table_names_the_ones_that_exist(graph):
    with pytest.raises(KeyError, match='unknown attribute table'):
        graph.attrs.table('nodez')


def test_a_write_shows_up_in_the_next_read(graph):
    before = len(graph.attrs.slices)
    graph.slices.add('second')
    graph.attrs.set_slice_attrs('second', curated=False)
    assert len(graph.attrs.slices) == before + 1


# ---------------------------------------------------------------------------
# What a table promises when it is empty, and when it is asked for elsewhere
# ---------------------------------------------------------------------------
#
# The fixture above carries a value at every address, which is the case the
# eight properties were written against. A graph that carries none is the case a
# caller meets first, and the promises below have to hold there too: a table is
# keyed by its address whether or not anything sits at it, and asking for one in
# another backend changes the container and nothing else.


@pytest.fixture
def blank() -> AnnNet:
    """A graph carrying nothing at any of the eight addresses."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True)
        G.layers.set_aspects(['cond'], {'cond': ['ctl', 'trt']})
        G.add_nodes(['a', 'b'], layer=('ctl',))
        G.add_edges('a', 'b', edge_id='e0')
    return G


@pytest.mark.parametrize('name', TABLE_NAMES)
def test_an_empty_table_is_keyed_by_its_address_in_any_backend(name, blank):
    """A backend picks the container and never the content — no rows included.

    A table with no rows still states what it is addressed by. Handing back a
    frame with no columns loses that, and the loss is silent: the caller gets an
    object that concatenates, joins and writes as though the address were never
    part of the table.
    """
    table = blank.attrs.table(name, backend='pandas')
    assert dataframe_backend(table) == 'pandas'
    for key in KEY_COLUMNS[name]:
        assert key in table.columns, f'empty {name} lost its {key} column'


# ``nodes`` and ``edges`` render through ``G.obs`` and ``G.var``, which clone on
# every read so that a caller cannot reach the cache the store holds. The six
# contextual tables hand back the cached table itself and are read-only by
# contract. So identity is what the six promise and the two cannot.
_CLONED = ('nodes', 'edges')


@pytest.mark.parametrize('name', [n for n in TABLE_NAMES if n not in _CLONED])
def test_asking_for_the_backend_a_table_already_has_costs_nothing(name, graph):
    """Naming the ambient backend, or ``auto``, is the property itself.

    Comparing the name the caller passed against the backend of the table meant
    ``auto`` never matched a concrete backend, so asking for the backend you
    already have rebuilt the whole table.
    """
    ambient = graph.attrs.backend
    assert graph.attrs.table(name, backend=ambient) is getattr(graph.attrs, name)
    assert graph.attrs.table(name, backend='auto') is getattr(graph.attrs, name)


@pytest.mark.parametrize('name', _CLONED)
def test_a_cloned_table_asked_for_its_own_backend_is_not_converted(name, graph):
    """The two generic axes clone, so identity cannot hold — the schema still does."""
    ambient = graph.attrs.backend
    for asked in (ambient, 'auto'):
        table = graph.attrs.table(name, backend=asked)
        assert dataframe_backend(table) == ambient
        assert dataframe_schema(table) == dataframe_schema(getattr(graph.attrs, name))


@pytest.mark.parametrize('name', TABLE_NAMES)
def test_a_table_is_typed_the_same_whether_or_not_it_holds_a_row(name, blank, graph):
    """The declared type of a key column is the type a row puts there.

    A layer coordinate is a tuple, so its column is a list of strings. Declaring
    it text meant an empty table and a filled one disagreed about the type of the
    column they are both keyed by, and only the filled one was right.
    """
    empty = dataframe_schema(getattr(blank.attrs, name))
    filled = dataframe_schema(getattr(graph.attrs, name))
    for key in KEY_COLUMNS[name]:
        assert empty[key] == filled[key], (
            f'{name}.{key} is {empty[key]} when the table is empty and '
            f'{filled[key]} when it holds a row'
        )


def test_a_write_to_one_level_does_not_rebuild_the_others(blank):
    """Each level is cached against its own clock, not against a shared one.

    One counter for all six meant annotating a slice threw away the materialized
    table of every other level, so a loop that writes one level and reads another
    rebuilt on every pass.
    """
    before = {name: getattr(blank.attrs, name) for name in ('node_layers', 'aspects')}
    blank.slices.add('s')
    blank.attrs.set_slice_attrs('s', curated=True)
    for name, table in before.items():
        assert getattr(blank.attrs, name) is table, f'{name} was rebuilt by a slice write'


def test_a_table_a_caller_installed_still_answers_in_the_ambient_backend(blank):
    """``G.layer_attributes`` keeps what it was given. The namespace renders it.

    A caller may assign a table the elementary-layer API cannot address, and
    adapters round-trip it verbatim. That is a promise of the old property and it
    stays. It is not a promise of ``G.attrs``, where every table answers in one
    backend, so the two do not have to be the same object.
    """
    pd = pytest.importorskip('pandas')
    blank.attrs.backend = 'polars'
    blank.layer_attributes = pd.DataFrame({'name': ['x'], 'note': ['hi']})
    assert dataframe_backend(blank.layer_attributes) == 'pandas'
    assert dataframe_backend(blank.attrs.elementary_layers) == 'polars'


@pytest.mark.parametrize(
    ('attribute', 'rows'),
    [
        ('slice_attributes', {'slice_id': ['a', 'b'], 'curated': [False, True]}),
        (
            'edge_slice_attributes',
            {'slice_id': ['a', 'a'], 'edge_id': ['e0', 'e1'], 'weight': [0.5, 1.5]},
        ),
        ('layer_attributes', {'layer_id': ['cond_ctl'], 'dose': [7]}),
    ],
)
def test_installing_a_whole_table_is_visible_to_the_next_read(attribute, rows):
    """A table assigned to the graph replaces what the level held.

    The install writes the store directly, so nothing told the materialized table
    it was out of date and the next read answered from the stale one — with the
    values the install replaced, and without the rows it added. A caller sees the
    old table and nothing says so.
    """
    pl = pytest.importorskip('polars')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True)
        G.layers.set_aspects(['cond'], {'cond': ['ctl', 'trt']})
        G.add_nodes(['a', 'b'], layer=('ctl',))
        G.add_edges('a', 'b', edge_id='e0')
        G.add_edges('b', 'a', edge_id='e1')
        G.slices.add('a')
        G.attrs.set_slice_attrs('a', curated=True)
        G.attrs.set_edge_slice_attrs('a', 'e0', weight=9.0)
        G.layers.set_elementary_attrs('cond', 'ctl', dose=1)

    getattr(G, attribute)  # materialize, so a stale one would be there to find
    setattr(G, attribute, pl.DataFrame(rows))
    key = next(iter(rows))
    assert sorted(dataframe_column_values(getattr(G, attribute), key)) == sorted(rows[key])
