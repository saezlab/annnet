"""Operator tests for the public AnnNet API.

An operator is the short form of a method, for the case that is common enough
to deserve one. The rule that decides what an operator does is the type of the
right operand, and nothing else:

- a string is one node, and a list of strings is many nodes
- a tuple is one edge, and a list of tuples is many edges
- a graph runs set algebra over the two element sets
- a vector runs the adjacency action

There is one rule for the element sets. Union, intersection, difference, and
symmetric difference each apply to the nodes and to the edges of the two
graphs, and an edge survives only when every node it names does. An edge that
loses a member has nothing left to connect.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def graph_of(nodes, edges, *, directed=True) -> AnnNet:
    """Build a flat graph from node ids and ``(source, target, edge_id)`` triples."""
    G = AnnNet(directed=directed)
    G.add_nodes(list(nodes))
    for source, target, edge_id in edges:
        G.add_edges(source, target, edge_id=edge_id)
    return G


@pytest.fixture
def left() -> AnnNet:
    return graph_of('abc', [('a', 'b', 'e_ab'), ('b', 'c', 'e_bc')])


@pytest.fixture
def right() -> AnnNet:
    return graph_of('bcd', [('b', 'c', 'e_bc'), ('c', 'd', 'e_cd')])


def nodes_of(graph) -> set:
    return set(graph.N)


def edges_of(graph) -> set:
    return set(graph.E)


# ---------------------------------------------------------------------------
# Add and remove
# ---------------------------------------------------------------------------


def test_a_string_added_is_one_node():
    G = AnnNet(directed=True)
    G += 'a'
    assert nodes_of(G) == {'a'}


def test_a_list_of_strings_added_is_many_nodes():
    G = AnnNet(directed=True)
    G += ['a', 'b']
    assert nodes_of(G) == {'a', 'b'}


def test_a_tuple_added_is_one_edge():
    G = AnnNet(directed=True)
    G += ('a', 'b')
    assert nodes_of(G) == {'a', 'b'}
    assert len(G.E) == 1


def test_a_list_of_tuples_added_is_many_edges():
    G = AnnNet(directed=True)
    G += [('a', 'b'), ('b', 'c')]
    assert len(G.E) == 2


def test_an_addition_returns_the_same_graph():
    """``+=`` mutates in place, so the name still points at one object."""
    G = AnnNet(directed=True)
    before = id(G)
    G += 'a'
    assert id(G) == before


def test_a_string_removed_is_one_node(left):
    left -= 'a'
    assert nodes_of(left) == {'b', 'c'}


def test_a_tuple_removed_is_one_edge(left):
    left -= ('a', 'b')
    assert nodes_of(left) == {'a', 'b', 'c'}
    assert edges_of(left) == {'e_bc'}


def test_removing_a_node_removes_the_edges_it_carried(left):
    left -= 'b'
    assert edges_of(left) == set()


# ---------------------------------------------------------------------------
# Set algebra
# ---------------------------------------------------------------------------


def test_the_union_holds_every_element_of_both(left, right):
    both = left | right
    assert nodes_of(both) == {'a', 'b', 'c', 'd'}
    assert edges_of(both) == {'e_ab', 'e_bc', 'e_cd'}


def test_the_intersection_holds_what_both_hold(left, right):
    shared = left & right
    assert nodes_of(shared) == {'b', 'c'}
    assert edges_of(shared) == {'e_bc'}


def test_the_difference_drops_what_the_right_operand_holds(left, right):
    rest = left - right
    assert nodes_of(rest) == {'a'}
    assert edges_of(rest) == set()


def test_the_symmetric_difference_holds_what_exactly_one_holds(left, right):
    either = left ^ right
    assert nodes_of(either) == {'a', 'd'}
    assert edges_of(either) == set()


def test_an_edge_needs_every_node_it_names():
    """An edge survives an operation only when both of its endpoints do."""
    first = graph_of('ab', [('a', 'b', 'e_ab')])
    second = graph_of('bc', [('b', 'c', 'e_bc')])
    assert edges_of(first | second) == {'e_ab', 'e_bc'}
    assert edges_of(first & second) == set()


def test_set_algebra_leaves_both_operands_alone(left, right):
    before = (nodes_of(left), edges_of(left), nodes_of(right), edges_of(right))
    _ = left | right
    _ = left & right
    _ = left - right
    _ = left ^ right
    assert (nodes_of(left), edges_of(left), nodes_of(right), edges_of(right)) == before


def test_a_merge_in_place_takes_the_right_operand_into_the_left(left, right):
    target = left
    target |= right
    assert target is left
    assert nodes_of(left) == {'a', 'b', 'c', 'd'}
    assert edges_of(left) == {'e_ab', 'e_bc', 'e_cd'}


def test_the_left_operand_keeps_its_attributes_in_a_union():
    """Two graphs can disagree about one element. The left one is the answer."""
    first = graph_of('ab', [('a', 'b', 'e_ab')])
    first.attrs.set_node_attrs('a', kind='left')
    second = graph_of('ab', [('a', 'b', 'e_ab')])
    second.attrs.set_node_attrs('a', kind='right')

    both = first | second
    assert both.attrs.get_attr_node('a', 'kind') == 'left'


# ---------------------------------------------------------------------------
# The adjacency action
# ---------------------------------------------------------------------------


def test_the_matrix_product_is_the_adjacency_action(left):
    x = np.ones(len(left.N))
    assert np.allclose(left @ x, left.A @ x)


def test_the_reflected_matrix_product_is_the_same_action(left):
    x = np.ones(len(left.N))
    assert np.allclose(x @ left, x @ left.A)


# ---------------------------------------------------------------------------
# The type of the operand decides
# ---------------------------------------------------------------------------


def test_a_graph_operand_never_adds_an_element(left, right):
    """``|`` on two graphs is set algebra, and it builds a third graph."""
    both = left | right
    assert both is not left
    assert both is not right


def test_an_unknown_operand_is_refused(left):
    with pytest.raises(TypeError):
        left + 3


def test_a_subtraction_of_an_unknown_operand_is_refused(left):
    with pytest.raises(TypeError):
        left - 3


# ---------------------------------------------------------------------------
# Membership, iteration, and truth
# ---------------------------------------------------------------------------


def test_a_node_id_is_a_membership_test(left):
    assert 'a' in left
    assert 'z' not in left


def test_a_pair_is_a_membership_test_of_an_edge(left):
    assert ('a', 'b') in left
    assert ('a', 'c') not in left


def test_iterating_a_graph_yields_node_ids(left):
    assert list(left) == list(left.N)


def test_an_empty_graph_is_false():
    assert not AnnNet(directed=True)


def test_a_graph_with_a_node_is_true(left):
    assert left
