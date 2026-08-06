"""A batch item that names its endpoints and a weight takes every other default.

The bulk binary writer reads six optional fields off every item and then tests
whether the item carries an attribute. An item that names nothing but its two
endpoints, or those and a weight, carries none of the rest, and its length says
so, because every key it could hold is one the writer already knows about.

That shortcut is only worth having if it gives what the long way gives. What
these tests pin is that the two agree — on the batch defaults, on the aliases
that make a two-key item, and on the first key that takes the item off the
short path again.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def _record(graph, edge_id):
    """What an edge holds, in the terms the batch defaults are stated in."""
    view = graph.get_edge(edge_id)
    return (view.kind, sorted(view.source), sorted(view.target), view.weight, view.directed)


@pytest.mark.parametrize(
    'item',
    [
        {'source': 'A', 'target': 'B'},
        {'src': 'A', 'tgt': 'B'},
        {'source': 'A', 'tgt': 'B'},
        {'src': 'A', 'target': 'B'},
    ],
    ids=['plain', 'both aliases', 'aliased target', 'aliased source'],
)
def test_an_item_of_two_endpoints_gives_what_the_spelled_out_item_gives(item):
    short, long = AnnNet(directed=True), AnnNet(directed=True)
    spelled = {
        'source': 'A',
        'target': 'B',
        'weight': 1.0,
        'edge_type': 'regular',
        'propagate': 'none',
    }
    assert short.add_edges([item]) == long.add_edges([spelled])
    assert _record(short, 'edge_0') == _record(long, 'edge_0')


def test_the_batch_defaults_reach_an_item_that_names_only_its_endpoints():
    graph = AnnNet(directed=True)
    graph.add_edges(
        [{'source': 'A', 'target': 'B'}],
        default_weight=2.5,
        default_edge_directed=False,
    )
    view = graph.get_edge('edge_0')
    assert (view.weight, view.directed) == (2.5, False)


def test_a_third_key_that_names_the_id_is_still_read():
    graph = AnnNet(directed=True)
    assert graph.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'named'}]) == ['named']


def test_a_third_key_that_names_the_weight_is_still_read():
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B', 'weight': 3.0}])
    assert graph.get_edge('edge_0').weight == 3.0


def test_a_weight_beside_two_endpoints_leaves_the_other_defaults_standing():
    short, long = AnnNet(directed=True), AnnNet(directed=True)
    short.add_edges([{'source': 'A', 'target': 'B', 'weight': 3.0}])
    long.add_edges(
        [
            {
                'source': 'A',
                'target': 'B',
                'weight': 3.0,
                'edge_type': 'regular',
                'propagate': 'none',
            }
        ]
    )
    assert _record(short, 'edge_0') == _record(long, 'edge_0')


def test_a_weight_beside_two_endpoints_is_not_an_attribute():
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B', 'weight': 3.0}])
    assert 'weight' not in graph.attrs.get_edge_attrs('edge_0')


@pytest.mark.parametrize('field', ['edge_directed', 'directed'])
def test_a_third_key_that_names_the_direction_is_still_read(field):
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B', field: False}])
    assert graph.get_edge('edge_0').directed is False


def test_a_third_key_that_names_the_slice_is_still_read():
    graph = AnnNet(directed=True)
    graph.slices.add('other')
    graph.add_edges([{'source': 'A', 'target': 'B', 'slice': 'other'}])
    assert 'edge_0' in graph.slices.edges('other')


def test_a_third_key_that_names_the_slice_weight_is_still_read():
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B', 'slice_weight': 0.25}])
    assert graph.attrs.get_edge_slice_attr(graph.slices.active, 'edge_0', 'weight') == 0.25


def test_a_third_key_that_names_the_propagation_is_still_read():
    graph = AnnNet(directed=True)
    graph.add_vertices(['A', 'B'])
    graph.slices.add('other')
    graph.slices.add_vertex_to_slice('other', 'A')
    graph.slices.add_vertex_to_slice('other', 'B')
    graph.add_edges([{'source': 'A', 'target': 'B', 'propagate': 'shared'}])
    assert 'edge_0' in graph.slices.edges('other')


def test_a_third_key_the_writer_does_not_know_becomes_an_attribute():
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B', 'evidence': 'pubmed'}])
    assert graph.attrs.get_edge_attrs('edge_0')['evidence'] == 'pubmed'


def test_an_item_of_two_endpoints_carries_no_attribute():
    graph = AnnNet(directed=True)
    graph.add_edges([{'source': 'A', 'target': 'B'}])
    attrs = graph.attrs.get_edge_attrs('edge_0')
    assert 'source' not in attrs
    assert 'target' not in attrs
