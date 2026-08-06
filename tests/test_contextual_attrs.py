"""Contextual attributes keep their own level.

A generic attribute belongs to one element. A contextual attribute belongs to a
pair, such as one edge in one slice or one node in one layer. Almost no pair
carries a value, so these stores stay keyed by the pair rather than becoming a
dense column.

The per-slice weight override is the load-bearing case: the effective weight of
an edge depends on which slice is active.
"""

from __future__ import annotations

import pytest

from annnet.core import _attrs as A, _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


@pytest.fixture
def attrs():
    store = ST.CoreState(directed=True)
    attributes = A.AttributeStore(store)
    for node_id in ('A', 'B'):
        store.add_entity(key(node_id))
    store.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=2.0,
    )
    return attributes


# ---------------------------------------------------------------------------
# The levels
# ---------------------------------------------------------------------------


def test_a_slice_carries_its_own_attributes(attrs):
    attrs.set_slice('core', kind='curated')
    assert attrs.slice_attrs('core') == {'kind': 'curated'}
    assert attrs.slice_attrs('absent') == {}


def test_an_edge_in_a_slice_carries_its_own_attributes(attrs):
    attrs.set_edge_slice('core', 'e0', confidence=0.9)
    assert attrs.edge_slice_attrs('core', 'e0') == {'confidence': 0.9}
    assert attrs.edge_slice_attrs('other', 'e0') == {}


def test_a_node_in_a_layer_carries_its_own_attributes(attrs):
    attrs.set_node_layer(key('A'), ('t0',), abundance=12.5)
    assert attrs.node_layer_attrs(key('A'), ('t0',)) == {'abundance': 12.5}
    assert attrs.node_layer_attrs(key('A'), ('t1',)) == {}


def test_an_aspect_carries_its_own_attributes(attrs):
    attrs.set_aspect('phase', description='time bin')
    assert attrs.aspect_attrs('phase') == {'description': 'time bin'}


def test_a_layer_carries_its_own_attributes(attrs):
    attrs.set_layer(('t0',), when='baseline')
    assert attrs.layer_attrs(('t0',)) == {'when': 'baseline'}


def test_the_graph_carries_its_own_attributes(attrs):
    attrs.graph_attrs['source'] = 'a file'
    assert attrs.graph_attrs == {'source': 'a file'}


# ---------------------------------------------------------------------------
# The two levels stay apart
# ---------------------------------------------------------------------------


def test_a_slice_attribute_never_reaches_the_edge_table(attrs):
    attrs.set_edge('e0', 'kind', 'binding')
    attrs.set_edge_slice('core', 'e0', kind='curated')
    assert attrs.var_rows() == [{'edge_id': 'e0', 'kind': 'binding'}]


def test_a_layer_attribute_never_reaches_the_node_table(attrs):
    attrs.set_node(key('A'), 'kind', 'protein')
    attrs.set_node_layer(key('A'), ('t0',), kind='active')
    rows = {row['node_id']: row for row in attrs.obs_rows()}
    assert rows['A']['kind'] == 'protein'


# ---------------------------------------------------------------------------
# The per-slice weight override
# ---------------------------------------------------------------------------


def test_an_edge_takes_its_own_weight_when_no_slice_overrides_it(attrs):
    assert attrs.effective_weight('e0') == pytest.approx(2.0)


def test_a_slice_overrides_the_weight_of_an_edge(attrs):
    attrs.set_slice_weight('core', 'e0', 10.0)
    assert attrs.effective_weight('e0', slice_id='core') == pytest.approx(10.0)
    assert attrs.effective_weight('e0') == pytest.approx(2.0)
    assert attrs.effective_weight('e0', slice_id='other') == pytest.approx(2.0)


def test_removing_an_edge_drops_its_contextual_attributes(attrs):
    attrs.set_edge_slice('core', 'e0', confidence=0.9)
    attrs.set_slice_weight('core', 'e0', 10.0)
    attrs.forget_edge('e0')
    assert attrs.edge_slice_attrs('core', 'e0') == {}


def test_removing_a_node_drops_its_contextual_attributes(attrs):
    attrs.set_node_layer(key('A'), ('t0',), abundance=1.0)
    attrs.forget_node(key('A'))
    assert attrs.node_layer_attrs(key('A'), ('t0',)) == {}


# ---------------------------------------------------------------------------
# The public entry points
# ---------------------------------------------------------------------------


def public_graph():
    """A graph with one value in each contextual store, reached through the API."""
    from annnet import AnnNet

    G = AnnNet(directed=True)
    G.layers.set_aspects(['phase'], {'phase': ['t0']})
    G.add_vertices(['A', 'B'], layer=('t0',))
    G.add_edges(
        [{'source': ('A', ('t0',)), 'target': ('B', ('t0',)), 'edge_id': 'e0'}],
        default_edge_directed=True,
    )
    G.slices.add('core')
    G.slices.add_edges('core', ['e0'])
    G.attrs.set_slice_attrs('core', kind='curated')
    G.attrs.set_edge_slice_attrs('core', 'e0', confidence=0.9)
    G.layers.set_attrs(('t0',), when='baseline')
    G.layers.set_node_attrs('A', ('t0',), abundance=12.5)
    G.layers.set_aspect_attrs('phase', description='time bin')
    G.layers.set_elementary_attrs('phase', 't0', colour='blue')
    G.uns['source'] = 'a file'
    return G


def test_each_contextual_level_has_one_entry_point():
    """Every level of the contextual stores is reachable by name."""
    G = public_graph()
    assert G.slices.attrs('core') == {'kind': 'curated'}
    assert G.attrs.edge_slice('core', 'e0') == {'confidence': 0.9}
    assert G.layers.attrs(('t0',)) == {'when': 'baseline'}
    assert G.layers.node_attrs('A', ('t0',)) == {'abundance': 12.5}
    assert G.layers.aspect_attrs('phase') == {'description': 'time bin'}
    assert G.layers.elementary_attrs('phase', 't0') == {'colour': 'blue'}
    assert G.uns == {'source': 'a file'}


def test_a_pair_that_carries_nothing_answers_empty():
    """The level of a contextual store is the pair, not either half of it."""
    G = public_graph()
    assert G.attrs.edge_slice('other', 'e0') == {}
    assert G.layers.node_attrs('B', ('t0',)) == {}
