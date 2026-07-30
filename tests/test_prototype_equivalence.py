"""The slot store answers every structural question the record store answers.

The bridge in the store builds a slot store from a record-backed graph, reading it
only through the query facade. These tests then ask both stores the same questions
and compare the answers by identity, never by position.

One difference is intended and stated in the specification: a directed self-loop
keeps both of its roles in the slot store, so its signed incidence column sums to
zero instead of holding one negative value. Every test below either avoids that
column or accounts for it.
"""

from __future__ import annotations

import pytest

from annnet.core import _matrices as M, _store as ST, _structure as S, _validate as V

from ._fixtures import CASE_NAMES, build_case

# The one case whose signed incidence column changes on purpose.
SELF_LOOP_CASES = {'self_loop'}


@pytest.fixture(params=CASE_NAMES)
def pair(request):
    graph = build_case(request.param)
    return request.param, graph, ST.from_graph(graph)


def test_the_bridge_builds_a_consistent_store(pair):
    _case, _graph, store = pair
    assert V.validate_internal_consistency(store, strict=False) == []


def test_both_stores_hold_the_same_entities(pair):
    _case, graph, store = pair
    assert {key for _slot, key in store.live_entities()} == {
        ref.key for ref in S.iter_entities(graph)
    }


def test_both_stores_hold_the_same_entity_kinds(pair):
    _case, graph, store = pair
    for ref in S.iter_entities(graph):
        slot = store.entity_slot(ref.key)
        expected = ST.EDGE_ENTITY if ref.kind == S.EDGE_ENTITY else ST.NODE
        assert int(store.entity_kind[slot]) == expected


def test_both_stores_hold_the_same_edges(pair):
    _case, graph, store = pair
    assert set(store.live_edge_ids()) == {
        ref.id for ref in S.iter_edges(graph, include_placeholders=True)
    }


def test_both_stores_report_the_same_edge_sides(pair):
    _case, graph, store = pair
    for ref in S.iter_edges(graph, include_placeholders=True):
        expected = S.edge_sides(graph, ref.id)
        found = store.endpoints(store.edge_slot(ref.id))
        expected_source = {_bare(item) for item in expected.source}
        expected_target = {_bare(item) for item in expected.target}
        assert {key[0] for key in found.source} == expected_source
        assert {key[0] for key in found.target} == expected_target


def test_both_stores_report_the_same_directedness(pair):
    _case, graph, store = pair
    for ref in S.iter_edges(graph):
        assert store.is_directed(store.edge_slot(ref.id)) == ref.directed


def test_both_stores_report_the_same_weight(pair):
    _case, graph, store = pair
    for ref in S.iter_edges(graph):
        slot = store.edge_slot(ref.id)
        assert float(store.edge_weight[slot]) == pytest.approx(ref.weight)


def test_the_incidence_columns_agree(pair):
    """Every column matches, apart from the self-loop the specification excepts."""
    case, graph, store = pair
    view = M.incidence(store)
    matrix = view.matrix.tocsc()
    for ref in S.iter_edges(graph):
        expected = {key: value for key, value in S.edge_members(graph, ref.id).items() if value}
        column = view.column_of_edge[ref.id]
        block = matrix[:, [column]].tocoo()
        found = {
            view.entity_of_row[int(row)]: float(value)
            for row, value in zip(block.row, block.data, strict=False)
            if float(value) != 0.0
        }
        if case in SELF_LOOP_CASES and store.is_self_loop(store.edge_slot(ref.id)):
            assert found == {}, 'the two roles of a self-loop cancel in a signed incidence'
            continue
        assert set(found) == set(expected), f'{case}/{ref.id}'
        for key, value in expected.items():
            assert found[key] == pytest.approx(value), f'{case}/{ref.id}/{key}'


def test_the_member_count_of_every_edge_is_explained(pair):
    case, graph, store = pair
    for ref in S.iter_edges(graph):
        slot = store.edge_slot(ref.id)
        sides = S.edge_sides(graph, ref.id)
        expected = len(sides.source) + len(sides.target)
        assert store.member_count(slot) == expected, f'{case}/{ref.id}'


def test_the_degree_of_every_node_is_explained(pair):
    """Degree counts member entries, so a self-loop counts twice."""
    case, graph, store = pair
    for ref in S.iter_entities(graph):
        expected = sum(
            1
            for edge in S.iter_edges(graph, include_placeholders=True)
            for endpoint in _all_endpoints(graph, edge.id)
            if _names(endpoint, ref)
        )
        assert store.degree(ref.key) == expected, f'{case}/{ref.key}'


def _bare(endpoint):
    return endpoint[0] if isinstance(endpoint, tuple) else endpoint


def _names(endpoint, ref) -> bool:
    """Return True when a stored endpoint names this exact entity.

    A multilayer graph holds one entity per node and layer, so a bare id is not
    enough to tell two of them apart.
    """
    if S.is_entity_key(endpoint):
        return endpoint == ref.key
    return endpoint == ref.id


def _all_endpoints(graph, edge_id):
    sides = S.edge_sides(graph, edge_id)
    return list(sides.source) + list(sides.target)
