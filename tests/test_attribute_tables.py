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
from annnet._support.dataframe_backend import dataframe_backend


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
