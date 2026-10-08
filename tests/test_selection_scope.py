"""Selection algebra keeps each operand's scope.

A leaf made in a view is evaluated in that view and one made in the graph in
the graph; a combination is ordered by the root graph. Union, intersection,
difference, projections and liveness are checked against independent reads.
"""

from __future__ import annotations


import pytest

from annnet import AnnNet


# ---------------------------------------------------------------------------
# 3. selection algebra keeps each operand's scope
# ---------------------------------------------------------------------------


@pytest.fixture
def scoped():
    G = AnnNet()
    G.add_nodes(['a', 'b', 'c'])
    G.attrs.update('nodes', {'a': {'x': 1}, 'b': {'x': 1}, 'c': {'x': 1}})
    G.add_edges('a', 'b', edge_id='ab')
    G.add_edges('b', 'c', edge_id='bc')
    G.attrs.update('edges', {'ab': {'w': 1}, 'bc': {'w': 1}})
    return G


def test_row_selection_algebra_across_views(scoped):
    G = scoped
    a = G.view(nodes=['a']).attrs.select('nodes', x=1)
    b = G.view(nodes=['b']).attrs.select('nodes', x=1)
    assert a.keys == ('a',) and b.keys == ('b',)
    assert (a | b).keys == ('a', 'b')
    assert (b | a).keys == ('a', 'b'), 'union is in graph order whatever the operand order'
    assert (a & b).keys == ()
    assert (a - b).keys == ('a',)
    assert (b - a).keys == ('b',)
    whole = G.attrs.select('nodes', x=1)
    assert (whole - a).keys == ('b', 'c')
    assert (whole & b).keys == ('b',)
    assert ((a | b) & whole).keys == ('a', 'b')


def test_row_selection_projection_keeps_scope(scoped):
    G = scoped
    G.slices.add('s1', edges=['ab'])
    G.slices.add('s2', edges=['bc'])
    G.attrs.update('edge_slices', {('s1', 'ab'): {'act': 1}, ('s2', 'bc'): {'act': 1}})
    left = G.view(slices='s1').attrs.select('edge_slices', act=1)
    right = G.view(slices='s2').attrs.select('edge_slices', act=1)
    assert left.keys == (('s1', 'ab'),) and right.keys == (('s2', 'bc'),)
    both = left | right
    assert both.keys == (('s1', 'ab'), ('s2', 'bc'))
    assert both.project('edges').ids == ('ab', 'bc')
    assert left.project('edges').ids == ('ab',)
    assert (left & right).project('slices').ids == ()


def test_sequence_algebra_across_views(scoped):
    G = scoped
    na = G.view(nodes=['a']).N.select(x=1)
    nb = G.view(nodes=['b']).N.select(x=1)
    assert na.ids == ('a',) and nb.ids == ('b',)
    assert (na | nb).ids == ('a', 'b')
    assert (nb | na).ids == ('a', 'b')
    assert (na & nb).ids == ()
    assert (na - nb).ids == ('a',)
    assert (G.N.select(x=1) - na).ids == ('b', 'c')
    ea = G.view(nodes=['a', 'b']).E.select(w=1)
    eb = G.view(nodes=['b', 'c']).E.select(w=1)
    assert ea.ids == ('ab',) and eb.ids == ('bc',)
    assert (ea | eb).ids == ('ab', 'bc')
    assert (ea & eb).ids == ()


def test_combined_selections_stay_live_in_each_scope(scoped):
    G = scoped
    va = G.view(nodes=G.N.select(x=1) & G.view(nodes=['a', 'c']).N.select())
    vb = G.view(nodes=['b'])
    both = va.N.select(x=1) | vb.N.select(x=1)
    assert both.ids == ('a', 'b', 'c')
    G.attrs.update('nodes', {'c': {'x': 0}})
    assert both.ids == ('a', 'b'), 'the left leaf re-evaluates inside its own view'
    G.attrs.update('nodes', {'b': {'x': 0}})
    assert both.ids == ('a',)
    view_of_union = G.view(nodes=both)
    assert view_of_union.N.ids == ('a',)
    G.attrs.update('nodes', {'b': {'x': 1}})
    assert view_of_union.N.ids == ('a', 'b')


def test_a_combined_selection_can_feed_a_view_of_either_scope(scoped):
    G = scoped
    va = G.view(nodes=['a', 'b'])
    inner = va.N.select(x=1) - G.view(nodes=['b']).N.select()
    assert inner.ids == ('a',)
    assert va.view(nodes=inner).N.ids == ('a',)
    assert G.view(nodes=inner).N.ids == ('a',)
