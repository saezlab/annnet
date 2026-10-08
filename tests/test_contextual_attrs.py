"""Contextual attributes keep their own level, and there is one place per level.

A generic attribute belongs to one element and lives in a slot-indexed column. A
contextual attribute belongs to a *pair* — one edge in one slice, one node in one
layer — and almost no pair carries a value, so those stores stay keyed by the pair
instead of becoming a dense column.

These tests go through the public API on purpose. ``AttributeStore`` used to carry
a second, parallel set of the contextual stores, and an earlier version of this
file tested those directly: every assertion passed while no public write ever
reached them. The duplicate is gone, and what is asserted here is where a value
written through the API can actually be read back from.
"""

from __future__ import annotations

import pytest

from annnet import AnnNet
from annnet.core import _attrs as A, _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


@pytest.fixture
def graph():
    """A multilayer graph with one value at every contextual level."""
    G = AnnNet(directed=True)
    G.layers.set_aspects(['phase'], {'phase': ['t0', 't1']})
    G.add_nodes(['A', 'B'], layer=('t0',))
    G.add_edges(
        [{'source': ('A', ('t0',)), 'target': ('B', ('t0',)), 'edge_id': 'e0', 'weight': 2.0}],
        default_edge_directed=True,
    )
    G.slices.add('core')
    G.slices.add_edges('core', ['e0'])

    G.attrs.update('nodes', {'A': {'kind': 'protein'}})
    G.attrs.update('edges', {'e0': {'assay': 'y2h'}})
    G.attrs.update('slices', {'core': {'kind': 'curated'}})
    G.attrs.update('edge_slices', {('core', 'e0'): {'confidence': 0.9}})
    G.attrs.update('layers', {('t0',): {'when': 'baseline'}})
    G.attrs.update('node_layers', {('A', ('t0',)): {'abundance': 12.5}})
    G.attrs.update('aspects', {'phase': {'description': 'time bin'}})
    G.attrs.update('elementary_layers', {('phase', 't0'): {'colour': 'blue'}})
    G.uns['source'] = 'a file'
    return G


# ---------------------------------------------------------------------------
# Every level has exactly one entry point
# ---------------------------------------------------------------------------


def test_each_contextual_level_reads_back_what_was_written(graph):
    assert dict(graph.attrs.row('slices', 'core')) == {'kind': 'curated'}
    assert dict(graph.attrs.row('edge_slices', ('core', 'e0'))) == {'confidence': 0.9}
    assert dict(graph.attrs.row('layers', ('t0',))) == {'when': 'baseline'}
    assert dict(graph.attrs.row('node_layers', ('A', ('t0',)))) == {'abundance': 12.5}
    assert dict(graph.attrs.row('aspects', 'phase')) == {'description': 'time bin'}
    assert dict(graph.attrs.row('elementary_layers', ('phase', 't0'))) == {'colour': 'blue'}
    assert graph.uns == {'source': 'a file'}


def test_a_pair_that_carries_nothing_answers_empty(graph):
    """The level of a contextual store is the pair, not either half of it.

    An existing pair that carries nothing is an empty row; a pair naming a
    slice, placement or layer the graph does not hold is an error,
    not an empty answer.
    """
    graph.slices.add('other')
    assert dict(graph.attrs.row('edge_slices', ('other', 'e0'))) == {}
    assert dict(graph.attrs.row('node_layers', ('B', ('t0',)))) == {}
    assert dict(graph.attrs.row('layers', ('t1',))) == {}
    with pytest.raises(KeyError, match='unknown slice'):
        graph.attrs.row('edge_slices', ('nope', 'e0'))


# ---------------------------------------------------------------------------
# The two levels stay apart
# ---------------------------------------------------------------------------


def test_a_slice_attribute_never_reaches_the_edge_table(graph):
    rows = {row['edge_id']: row for row in graph.attrs.edges.to_dicts()}
    assert rows['e0']['assay'] == 'y2h'
    assert 'confidence' not in rows['e0']


def test_a_layer_attribute_never_reaches_the_node_table(graph):
    rows = {row['node_id']: row for row in graph.attrs.nodes.to_dicts()}
    assert rows['A']['kind'] == 'protein'
    assert 'abundance' not in rows['A']


# ---------------------------------------------------------------------------
# The per-slice weight override
# ---------------------------------------------------------------------------


def test_an_edge_keeps_its_own_weight_when_no_slice_overrides_it(graph):
    assert graph.E.at('e0').weight == pytest.approx(2.0)
    assert graph.slice_edge_weights.get('core', {}).get('e0') is None


def test_a_slice_overrides_the_weight_of_an_edge(graph):
    graph.attrs.update('edge_slices', {('core', 'e0'): {'weight': 10.0}})
    assert graph.slice_edge_weights['core']['e0'] == pytest.approx(10.0)
    # The override belongs to the pair, so the edge itself is untouched.
    assert graph.E.at('e0').weight == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The generic store owns the two element axes and nothing else
# ---------------------------------------------------------------------------


def test_the_attribute_store_carries_no_contextual_state(graph):
    """The parallel contextual stores are gone, so they cannot be written to.

    Their presence was the hazard: a contributor who found them here would write
    a value that no reader of the public API could ever see.
    """
    store = graph._attr_store
    for name in (
        'slice_attributes',
        'edge_slice_attributes',
        'node_layer_attributes',
        'aspect_attributes',
        'layer_attributes',
        'slice_weights',
        'graph_attrs',
    ):
        assert not hasattr(store, name), f'AttributeStore still carries {name!r}'
    for name in (
        'set_slice',
        'set_edge_slice',
        'set_node_layer',
        'set_aspect',
        'set_layer',
        'set_slice_weight',
        'effective_weight',
    ):
        assert not hasattr(store, name), f'AttributeStore still exposes {name!r}'


def test_the_attribute_store_still_owns_the_generic_axes():
    store_ = ST.CoreState(directed=True)
    attrs = A.AttributeStore(store_)
    for node_id in ('A', 'B'):
        store_.add_entity(key(node_id))
    store_.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=2.0,
    )
    attrs.set_node(key('A'), 'kind', 'protein')
    attrs.set_edge('e0', 'assay', 'y2h')
    # A cell that carries nothing is left out of the row rather than spelled None.
    assert attrs.obs_rows() == [{'node_id': 'A', 'kind': 'protein'}, {'node_id': 'B'}]
    assert attrs.var_rows() == [{'edge_id': 'e0', 'assay': 'y2h'}]


def test_forgetting_an_element_drops_the_table_that_named_it(graph):
    """The freed-slot hooks still invalidate the materialized tables."""
    _ = graph.attrs.edges
    graph._attr_store.forget_edge('e0')
    assert 'edge' not in graph._attr_store._tables
    _ = graph.attrs.nodes
    graph._attr_store.forget_node(('A', ('t0',)))
    assert 'node' not in graph._attr_store._tables
