"""A placement reads the same way in one call and in a batch.

The contract says that a string is a node, that a ``(node_id, layer_coord)``
tuple is a placement, and that a bare id names a node while it sits on one
layer. These tests hold the two forms of each call to that one rule: what
``add_nodes`` does with one node it does with a list of them, and what
``add_edges`` does with one edge it does with a batch.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet


@pytest.fixture
def conditions() -> AnnNet:
    """A graph with one aspect and two layers."""
    anet = AnnNet(directed=True)
    anet.layers.set_aspects(['condition'], {'condition': ['control', 'treated']})
    return anet


@pytest.fixture
def two_aspects() -> AnnNet:
    """A graph with two aspects, so a coordinate holds two labels."""
    anet = AnnNet(directed=True)
    anet.layers.set_aspects(
        ['condition', 'time'],
        {'condition': ['control', 'treated'], 'time': ['0h', '6h']},
    )
    return anet


def _placements(anet: AnnNet, node_id: str) -> tuple:
    return anet.N.at(node_id).layers


# --- add_nodes --------------------------------------------------------------


def test_one_placement_tuple_adds_one_node(conditions):
    """``('A', ('control',))`` is one node on one layer, not two nodes."""
    returned = conditions.add_nodes(('A', ('control',)))

    assert returned == 'A'
    assert list(conditions.N) == ['A']
    assert _placements(conditions, 'A') == (('control',),)


def test_a_placement_tuple_warns_about_nothing(conditions):
    """The caller said where the node goes, so no placeholder warning fires."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        conditions.add_nodes(('A', ('control',)))


def test_a_list_of_placements_places_each_node(conditions):
    """Each item carries its own coordinate, and the batch keeps them apart."""
    returned = conditions.add_nodes([('A', ('control',)), ('B', ('treated',))])

    assert returned == ['A', 'B']
    assert _placements(conditions, 'A') == (('control',),)
    assert _placements(conditions, 'B') == (('treated',),)


def test_a_placement_in_a_list_holds_two_aspects(two_aspects):
    """A coordinate of a two-aspect graph survives the batch path."""
    two_aspects.add_nodes([('A', ('control', '0h')), ('B', ('treated', '6h'))])

    assert _placements(two_aspects, 'A') == (('control', '0h'),)
    assert _placements(two_aspects, 'B') == (('treated', '6h'),)


def test_a_batch_mixes_a_placement_and_the_layer_argument(conditions):
    """An item without a coordinate takes the one the call names."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        conditions.add_nodes(['A', ('B', ('treated',))], layer={'condition': 'control'})

    assert _placements(conditions, 'A') == (('control',),)
    assert _placements(conditions, 'B') == (('treated',),)


def test_a_placement_carries_attributes_too(conditions):
    """``(node_id, coordinate, attributes)`` says where and what."""
    conditions.add_nodes([('A', ('control',), {'kind': 'source'})])

    assert _placements(conditions, 'A') == (('control',),)
    assert conditions.attrs.row('nodes', 'A')['kind'] == 'source'


def test_a_tuple_of_a_node_and_a_mapping_is_still_attributes(conditions):
    """A mapping is attributes, in one call and in a list, as the docstring says."""
    conditions.add_nodes(('A', {'kind': 'source'}), layer={'condition': 'control'})
    conditions.add_nodes([('B', {'kind': 'relay'})], layer={'condition': 'control'})

    assert conditions.attrs.row('nodes', 'A')['kind'] == 'source'
    assert conditions.attrs.row('nodes', 'B')['kind'] == 'relay'
    assert _placements(conditions, 'A') == (('control',),)
    assert _placements(conditions, 'B') == (('control',),)


def test_a_flat_graph_takes_the_same_forms():
    """Without aspects, every node sits on the one coordinate."""
    anet = AnnNet(directed=True)
    anet.add_nodes([('A', ('_',)), 'B', ('C', {'kind': 'sink'})])

    assert list(anet.N) == ['A', 'B', 'C']
    assert anet.attrs.row('nodes', 'C')['kind'] == 'sink'


# --- add_edges --------------------------------------------------------------


def test_a_batch_edge_reads_a_bare_id_as_one_edge_does(conditions):
    """One layer holds the node, so a bare id names it in both forms."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})

    conditions.add_edges('A', 'B', edge_id='one')
    conditions.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'batch'}])

    expected = {('A', ('control',)), ('B', ('control',))}
    assert set(conditions.E.at('one').members) == expected
    assert set(conditions.E.at('batch').members) == expected


def test_a_batch_edge_of_tuples_reads_bare_ids_the_same_way(conditions):
    """The ``(source, target)`` item follows the same rule."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})

    conditions.add_edges([('A', 'B')])

    (edge_id,) = conditions.E.ids
    assert set(conditions.E.at(edge_id).members) == {
        ('A', ('control',)),
        ('B', ('control',)),
    }


def test_a_batch_edge_does_not_add_a_second_placement(conditions):
    """Adding an edge over placed nodes leaves their placements alone."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})

    conditions.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'batch'}])

    assert _placements(conditions, 'A') == (('control',),)
    assert _placements(conditions, 'B') == (('control',),)


def test_a_batch_edge_refuses_an_ambiguous_bare_id(conditions):
    """Two layers hold the node, so the bare id names neither."""
    conditions.add_nodes(['A'], layer={'condition': 'control'})
    conditions.add_nodes(['A'], layer={'condition': 'treated'})
    conditions.add_nodes(['B'], layer={'condition': 'control'})

    with pytest.raises(ValueError, match='Ambiguous bare node_id'):
        conditions.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'batch'}])


def test_a_batch_edge_takes_a_placement_tuple(conditions):
    """An explicit placement names the layer, whatever else the graph holds."""
    conditions.add_nodes(['A'], layer={'condition': 'control'})
    conditions.add_nodes(['A'], layer={'condition': 'treated'})
    conditions.add_nodes(['B'], layer={'condition': 'treated'})

    conditions.add_edges(
        [{'source': ('A', ('treated',)), 'target': ('B', ('treated',)), 'edge_id': 'batch'}]
    )

    assert set(conditions.E.at('batch').members) == {
        ('A', ('treated',)),
        ('B', ('treated',)),
    }


def test_the_layer_argument_places_the_endpoints_of_one_edge(conditions):
    """``layer=`` says where a bare endpoint goes, as it does in ``add_nodes``."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})
    conditions.add_nodes(['A', 'B'], layer={'condition': 'treated'})

    conditions.add_edges('A', 'B', edge_id='one', layer=('treated',))

    assert set(conditions.E.at('one').members) == {
        ('A', ('treated',)),
        ('B', ('treated',)),
    }


def test_the_layer_argument_places_the_endpoints_of_a_batch(conditions):
    """A batch reads ``layer=`` the way one call reads it."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})
    conditions.add_nodes(['A', 'B'], layer={'condition': 'treated'})

    conditions.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'batch'}], layer=('treated',))

    assert set(conditions.E.at('batch').members) == {
        ('A', ('treated',)),
        ('B', ('treated',)),
    }


def test_an_item_of_a_batch_names_its_own_layer(conditions):
    """A ``layer`` key in an item places that edge, as it does for a hyperedge."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})
    conditions.add_nodes(['A', 'B'], layer={'condition': 'treated'})

    conditions.add_edges(
        [
            {'source': 'A', 'target': 'B', 'edge_id': 'here', 'layer': ('control',)},
            {'source': 'A', 'target': 'B', 'edge_id': 'there', 'layer': ('treated',)},
        ]
    )

    assert set(conditions.E.at('here').members) == {('A', ('control',)), ('B', ('control',))}
    assert set(conditions.E.at('there').members) == {('A', ('treated',)), ('B', ('treated',))}


def test_the_layer_of_an_edge_is_not_an_attribute_of_it(conditions):
    """``layer=`` places the edge. It does not become a field of it."""
    conditions.add_nodes(['A', 'B'], layer={'condition': 'control'})

    conditions.add_edges('A', 'B', edge_id='one', layer=('control',))

    assert 'layer' not in conditions.attrs.row('edges', 'one')


def test_a_hyperedge_batch_reads_the_layer_the_same_way(conditions):
    """The hyperedge path already did this, and the binary path now agrees."""
    conditions.add_nodes(['A', 'B', 'C'], layer={'condition': 'treated'})

    conditions.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h'}], layer=('treated',))

    assert set(conditions.E.at('h').members) == {
        ('A', ('treated',)),
        ('B', ('treated',)),
        ('C', ('treated',)),
    }


def test_a_tuple_that_holds_one_item_is_a_batch_of_one(conditions):
    """``(item,)`` is a list of one, so the item inside it still reads as itself."""
    conditions.add_nodes((('A', ('control',)),))
    conditions.add_nodes((('B', {'kind': 'relay'}),), layer={'condition': 'control'})
    conditions.add_nodes(({'node_id': 'C', 'kind': 'sink'},), layer={'condition': 'control'})

    assert list(conditions.N) == ['A', 'B', 'C']
    assert _placements(conditions, 'A') == (('control',),)
    assert conditions.attrs.row('nodes', 'B')['kind'] == 'relay'
    assert conditions.attrs.row('nodes', 'C')['kind'] == 'sink'
