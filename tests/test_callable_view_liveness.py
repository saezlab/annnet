"""A view built on a callable filter answers from the callable as it is now.

The callable can depend on state the graph does not own, so no clock of the graph
moves when that state changes. Everything derived from such a view has to follow
it: a selection made in the view, the union or difference of two of them, the
projection of attribute rows, and a view built from any of these. None of them may
keep the ids the first read resolved.
"""

from __future__ import annotations

import pytest

from annnet import AnnNet


@pytest.fixture
def graph():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c', 'd'])
    G.attrs.update(
        'nodes',
        {'a': {'g': 1}, 'b': {'g': 1}, 'c': {'g': 2}, 'd': {'g': 1}},
    )
    G.add_edges('a', 'b', edge_id='ab')
    G.add_edges('b', 'c', edge_id='bc')
    G.add_edges('c', 'd', edge_id='cd')
    return G


@pytest.fixture
def chosen():
    """External state the graph knows nothing about."""
    return {'a', 'b'}


def test_the_view_itself_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    assert view.N.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'c', 'd'})
    assert view.N.ids == ('c', 'd')
    assert view.E.ids == ('cd',)


def test_a_selection_made_in_the_view_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    selection = view.N.select(g=1)
    assert selection.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'c', 'd'})
    assert selection.ids == ('d',)
    assert len(selection) == 1 and 'd' in selection and 'a' not in selection


def test_a_selection_with_no_conditions_of_its_own_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    everything = view.N.select({})
    sequence = view.N
    chosen.clear()
    chosen.add('c')
    assert sequence.ids == ('c',)
    assert everything.ids == ('c',)


def test_composed_selections_follow_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    ones = view.N.select(g=1)
    twos = view.N.select(g=2)
    union, both, only_ones = ones | twos, ones & twos, ones - twos
    assert union.ids == ('a', 'b') and both.ids == () and only_ones.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'b', 'c', 'd'})
    assert union.ids == ('b', 'c', 'd')
    assert both.ids == ()
    assert only_ones.ids == ('b', 'd')


def test_a_selection_of_the_graph_combined_with_one_of_the_view_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    mixed = graph.N.select(g=1) & view.N.select(g=1)
    assert mixed.ids == ('a', 'b')
    chosen.clear()
    chosen.add('d')
    assert mixed.ids == ('d',)


def test_a_projection_of_attribute_rows_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    rows = view.attrs.select('nodes', g=1)
    projected = rows.project('nodes')
    assert rows.keys == ('a', 'b') and projected.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'c', 'd'})
    assert rows.keys == ('d',)
    assert projected.ids == ('d',)


def test_a_view_built_from_a_selection_of_a_callable_view_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    inner = graph.view(nodes=view.N.select(g=1))
    whole = graph.view(nodes=view.N)
    nested = view.view(nodes=view.N.select(g=1))
    assert inner.N.ids == ('a', 'b') and whole.N.ids == ('a', 'b') and nested.N.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'c', 'd'})
    assert inner.N.ids == ('d',)
    assert whole.N.ids == ('c', 'd')
    assert nested.N.ids == ('d',)
    assert whole.E.ids == ('cd',)


def test_a_view_built_from_a_projection_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    projected = view.attrs.select('nodes', g=1).project('nodes')
    built = graph.view(nodes=projected)
    assert built.N.ids == ('a', 'b')
    chosen.clear()
    chosen.update({'c', 'd'})
    assert built.N.ids == ('d',)


def test_a_filter_of_a_view_over_a_callable_view_follows_the_state(graph, chosen):
    view = graph.view(nodes=lambda node: node in chosen)
    narrower = view.view(edges=lambda edge: edge != 'ab')
    assert narrower.E.ids == ()
    chosen.update({'c'})
    assert narrower.E.ids == ('bc',)
    assert narrower.N.select(g=2).ids == ('c',)


def test_a_plain_view_is_still_cached_between_reads(graph):
    """Only what depends on a callable is re-evaluated; the rest keeps its cache."""
    view = graph.view(nodes=graph.N.select(g=1))
    selection = view.N.select(g=1)
    first = selection.ids
    assert selection.ids is first, 'an unchanged graph answers from the cache'
    graph.attrs.update('nodes', {'c': {'g': 1}})
    assert selection.ids == ('a', 'b', 'c', 'd')


def test_a_selection_of_a_plain_callable_is_reevaluated_on_every_read(graph):
    calls = []

    def probe(node):
        calls.append(node)
        return node == 'a'

    selection = graph.N.select(probe)
    _ = selection.ids
    seen = len(calls)
    _ = selection.ids
    assert len(calls) > seen
