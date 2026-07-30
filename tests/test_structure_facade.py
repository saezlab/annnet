"""Contract tests for the read-only structural query facade.

The facade is the one path from the rest of the package to the topology of a
graph. These tests fix the shape of every answer it gives, so the answers stay
the same when the store behind it changes.

Every address in the facade is an identity. An entity key is a
``(node_id, layer_coord)`` pair. An edge address is an edge id. No position of
any materialized matrix appears here.
"""

from __future__ import annotations

import pytest

from annnet.core import _structure as S

from ._fixtures import CASE_NAMES, build_case

FLAT = ('_',)


def key(node_id, layer=FLAT):
    """Build an entity key for a node id."""
    return (node_id, layer)


# ---------------------------------------------------------------------------
# iter_entities
# ---------------------------------------------------------------------------


def test_iter_entities_reports_every_node_with_its_kind_and_layer():
    G = build_case('binary_directed')
    refs = {ref.key: ref for ref in S.iter_entities(G)}
    assert set(refs) == {key('A'), key('B'), key('C')}
    assert all(ref.kind == S.NODE for ref in refs.values())
    assert refs[key('A')].id == 'A'
    assert refs[key('A')].layer == FLAT


def test_iter_entities_marks_an_edge_entity_with_its_own_kind():
    G = build_case('edge_entity')
    kinds = {ref.id: ref.kind for ref in S.iter_entities(G)}
    assert kinds == {'A': S.NODE, 'B': S.NODE, 'C': S.NODE, 'ee_ab': S.EDGE_ENTITY}


def test_iter_entities_separates_the_layers_of_a_multilayer_graph():
    G = build_case('multilayer')
    keys = {ref.key for ref in S.iter_entities(G)}
    assert keys == {
        ('A', ('t0',)),
        ('B', ('t0',)),
        ('A', ('t1',)),
        ('B', ('t1',)),
    }


# ---------------------------------------------------------------------------
# iter_edges
# ---------------------------------------------------------------------------


def test_iter_edges_reports_kind_directedness_and_weight():
    G = build_case('binary_directed')
    refs = {ref.id: ref for ref in S.iter_edges(G)}
    assert set(refs) == {'e_ab', 'e_bc'}
    assert refs['e_ab'].kind == S.BINARY
    assert refs['e_ab'].directed is True
    assert refs['e_ab'].weight == pytest.approx(1.5)


def test_iter_edges_reports_an_undirected_edge_as_undirected():
    G = build_case('binary_undirected')
    assert all(ref.directed is False for ref in S.iter_edges(G))


def test_iter_edges_names_the_hyperedge_and_edge_entity_kinds():
    assert {ref.kind for ref in S.iter_edges(build_case('hyper_undirected'))} == {S.HYPER}
    assert {ref.kind for ref in S.iter_edges(build_case('edge_entity'))} == {
        S.NODE_EDGE,
        S.BINARY,
    }


def test_iter_edges_keeps_two_parallel_edges_apart():
    G = build_case('parallel_edge')
    weights = {ref.id: ref.weight for ref in S.iter_edges(G)}
    assert weights == {'e_first': pytest.approx(1.0), 'e_second': pytest.approx(2.0)}


# ---------------------------------------------------------------------------
# edge_members
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('case', 'edge_id', 'expected'),
    [
        ('binary_directed', 'e_ab', {key('A'): 1.5, key('B'): -1.5}),
        ('binary_undirected', 'e_ab', {key('A'): 1.0, key('B'): 1.0}),
        ('self_loop', 'e_loop', {key('A'): -0.5}),
        ('parallel_edge', 'e_second', {key('A'): 2.0, key('B'): -2.0}),
        ('hyper_undirected', 'h_abc', {key('A'): 1.0, key('B'): 1.0, key('C'): 1.0}),
        ('hyper_directed', 'h_ab_c', {key('A'): 1.0, key('B'): 1.0, key('C'): -1.0}),
        ('edge_entity', 'ee_ab', {key('A'): 1.0, key('B'): -1.0}),
        ('edge_entity', 'e_meta', {key('ee_ab'): 1.0, key('C'): -1.0}),
        ('boundary_edge', 'b_out', {key('A'): -1.0}),
        ('boundary_edge', 'b_in', {key('A'): 1.0}),
        ('coefficient_edge', 'r_1', {key('A'): -2.0, key('B'): -1.0, key('C'): 3.0}),
        ('multilayer', 'e_couple', {('A', ('t0',)): 1.0, ('A', ('t1',)): -1.0}),
    ],
)
def test_edge_members_returns_the_member_list_of_one_edge(case, edge_id, expected):
    G = build_case(case)
    members = S.edge_members(G, edge_id)
    assert set(members) == set(expected)
    for member_key, coefficient in expected.items():
        assert members[member_key] == pytest.approx(coefficient)


@pytest.mark.parametrize('case', CASE_NAMES)
def test_edge_members_equals_the_incidence_column(case):
    """The member list is the incidence matrix, so the two must agree everywhere."""
    G = build_case(case)
    matrix = G.X().tocsc()
    for ref in S.iter_edges(G):
        column = G.idx.edge_to_col(ref.id)
        block = matrix[:, [column]].tocoo()
        from_matrix = {
            S.entity_key_of_row(G, int(row)): float(value)
            for row, value in zip(block.row, block.data, strict=False)
            if float(value) != 0.0
        }
        from_members = {k: v for k, v in S.edge_members(G, ref.id).items() if v != 0.0}
        assert set(from_members) == set(from_matrix), f'{case}/{ref.id}'
        for member_key, value in from_matrix.items():
            assert from_members[member_key] == pytest.approx(value), f'{case}/{ref.id}'


def test_edge_members_rejects_an_unknown_edge():
    G = build_case('binary_directed')
    with pytest.raises(KeyError):
        S.edge_members(G, 'no_such_edge')


# ---------------------------------------------------------------------------
# edge_endpoints
# ---------------------------------------------------------------------------


def test_edge_endpoints_splits_a_binary_edge_into_source_and_target():
    G = build_case('binary_directed')
    endpoints = S.edge_endpoints(G, 'e_ab')
    assert endpoints.source == frozenset({key('A')})
    assert endpoints.target == frozenset({key('B')})


def test_edge_endpoints_keeps_the_stored_sides_of_an_undirected_edge():
    """The sides stay as stored. Directedness is a separate fact on the edge."""
    G = build_case('binary_undirected')
    endpoints = S.edge_endpoints(G, 'e_ab')
    assert endpoints.source == frozenset({key('A')})
    assert endpoints.target == frozenset({key('B')})
    assert S.edge_ref(G, 'e_ab').directed is False


def test_edge_endpoints_splits_a_directed_hyperedge():
    G = build_case('hyper_directed')
    endpoints = S.edge_endpoints(G, 'h_ab_c')
    assert endpoints.source == frozenset({key('A'), key('B')})
    assert endpoints.target == frozenset({key('C')})


def test_edge_endpoints_puts_every_member_of_an_undirected_hyperedge_on_one_side():
    G = build_case('hyper_undirected')
    endpoints = S.edge_endpoints(G, 'h_abc')
    assert endpoints.source == frozenset({key('A'), key('B'), key('C')})
    assert endpoints.target == frozenset()


def test_edge_endpoints_leaves_the_open_side_of_a_boundary_edge_empty():
    G = build_case('boundary_edge')
    endpoints = S.edge_endpoints(G, 'b_out')
    assert endpoints.source == frozenset({key('A')})
    assert endpoints.target == frozenset()


# ---------------------------------------------------------------------------
# entity_edges
# ---------------------------------------------------------------------------


def test_entity_edges_separates_the_two_directions_of_a_binary_edge():
    G = build_case('binary_directed')
    assert set(S.entity_edges(G, 'A', 'out')) == {'e_ab'}
    assert set(S.entity_edges(G, 'A', 'in')) == set()
    assert set(S.entity_edges(G, 'B', 'in')) == {'e_ab'}
    assert set(S.entity_edges(G, 'B', 'out')) == {'e_bc'}
    assert set(S.entity_edges(G, 'B', 'both')) == {'e_ab', 'e_bc'}


def test_entity_edges_counts_an_undirected_edge_in_both_directions():
    G = build_case('binary_undirected')
    assert set(S.entity_edges(G, 'A', 'in')) == {'e_ab'}
    assert set(S.entity_edges(G, 'A', 'out')) == {'e_ab'}


def test_entity_edges_counts_a_self_loop_in_both_directions():
    G = build_case('self_loop')
    assert set(S.entity_edges(G, 'A', 'in')) == {'e_loop'}
    assert set(S.entity_edges(G, 'A', 'out')) == {'e_loop', 'e_ab'}


def test_entity_edges_reports_a_hyperedge():
    G = build_case('hyper_directed')
    assert set(S.entity_edges(G, 'A', 'out')) == {'h_ab_c'}
    assert set(S.entity_edges(G, 'C', 'in')) == {'h_ab_c'}
    assert set(S.entity_edges(G, 'C', 'out')) == set()


def test_entity_edges_reports_both_parallel_edges():
    G = build_case('parallel_edge')
    assert set(S.entity_edges(G, 'A', 'out')) == {'e_first', 'e_second'}


def test_entity_edges_returns_the_edges_in_column_order():
    G = build_case('parallel_edge')
    assert S.entity_edges(G, 'A', 'out') == ('e_first', 'e_second')


def test_entity_edges_accepts_an_entity_key_in_a_multilayer_graph():
    G = build_case('multilayer')
    assert set(S.entity_edges(G, ('A', ('t0',)), 'out')) == {'e_t0', 'e_couple'}
    assert set(S.entity_edges(G, ('A', ('t1',)), 'in')) == {'e_couple'}


def test_entity_edges_rejects_an_unknown_direction():
    G = build_case('binary_directed')
    with pytest.raises(ValueError):
        S.entity_edges(G, 'A', 'sideways')


def test_entity_edges_rejects_an_unknown_entity():
    G = build_case('binary_directed')
    with pytest.raises(KeyError):
        S.entity_edges(G, 'no_such_node')


# ---------------------------------------------------------------------------
# has_entity and has_edge
# ---------------------------------------------------------------------------


def test_has_entity_accepts_a_bare_id_and_an_entity_key():
    G = build_case('binary_directed')
    assert S.has_entity(G, 'A') is True
    assert S.has_entity(G, key('A')) is True
    assert S.has_entity(G, 'Z') is False


def test_has_entity_is_true_for_an_edge_entity():
    G = build_case('edge_entity')
    assert S.has_entity(G, 'ee_ab') is True


def test_has_entity_separates_the_layers():
    G = build_case('multilayer')
    assert S.has_entity(G, ('A', ('t0',))) is True
    assert S.has_entity(G, ('Z', ('t0',))) is False


def test_has_edge_answers_by_edge_id():
    G = build_case('binary_directed')
    assert S.has_edge(G, 'e_ab') is True
    assert S.has_edge(G, 'no_such_edge') is False


def test_has_edge_is_true_for_an_edge_entity():
    G = build_case('edge_entity')
    assert S.has_edge(G, 'ee_ab') is True


# ---------------------------------------------------------------------------
# Whole-set consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_the_facade_agrees_with_the_public_counts(case):
    G = build_case(case)
    entity_ids = {ref.id for ref in S.iter_entities(G) if ref.kind == S.NODE}
    assert entity_ids == set(G.vertices())
    assert {ref.id for ref in S.iter_edges(G)} == set(G.edges())


@pytest.mark.parametrize('case', CASE_NAMES)
def test_every_member_of_every_edge_is_a_live_entity(case):
    G = build_case(case)
    live = {ref.key for ref in S.iter_entities(G)}
    for ref in S.iter_edges(G):
        assert set(S.edge_members(G, ref.id)) <= live, f'{case}/{ref.id}'
