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


# ---------------------------------------------------------------------------
# A graph built through the public API on the slot store
# ---------------------------------------------------------------------------
# The bridge builds a slot store from a finished record graph. These build one
# through the public API instead, so the mutation gateway is what fills it. A
# difference here is a write the gateway has not routed yet.


@pytest.mark.parametrize('case', CASE_NAMES)
def test_the_gateway_fills_the_slot_store_as_the_bridge_would(case):
    built = build_case(case, store='slots')
    expected = ST.from_graph(build_case(case))

    assert S.entity_keys(built._store) == S.entity_keys(expected), 'entities'
    assert S.edge_ids(built._store) == S.edge_ids(expected), 'edges'
    for edge_id in S.edge_ids(expected):
        assert S.edge_sides(built._store, edge_id) == S.edge_sides(expected, edge_id), edge_id
        assert S.edge_members(built._store, edge_id) == pytest.approx(
            S.edge_members(expected, edge_id)
        ), edge_id


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_slot_backed_graph_holds_its_invariants(case):
    built = build_case(case, store='slots')
    assert V.validate_internal_consistency(built._store, strict=False) == []


# ---------------------------------------------------------------------------
# Every write reaches the slot store
# ---------------------------------------------------------------------------
# The gateway writes the slot store as each mutation lands. Rebuilding it from
# the records afterwards must therefore change nothing. A write that does not
# reach the store shows up here as a difference, which is how the inline entity
# registration in the bulk vertex path was found.


def _snapshot(store):
    """Everything the slot store holds, addressed by identity alone."""
    return {
        'entities': S.entity_keys(store),
        'edges': S.edge_ids(store),
        'sides': {eid: S.edge_sides(store, eid) for eid in S.edge_ids(store)},
        'members': {eid: S.edge_members(store, eid) for eid in S.edge_ids(store)},
        'policies': S.edge_policies(store),
    }


def _assert_incremental_matches_rebuild(G, note):
    from annnet.core import _mutate

    incremental = _snapshot(G._store)
    _mutate.resync(G)
    assert _snapshot(G._store) == incremental, note


@pytest.mark.parametrize('case', CASE_NAMES)
def test_building_a_shape_reaches_the_store(case):
    _assert_incremental_matches_rebuild(build_case(case, store='slots'), case)


# ---------------------------------------------------------------------------
# A copy takes the store with it
# ---------------------------------------------------------------------------
# Copying a graph copies its slot arrays instead of rebuilding the store from
# what was installed. So the copy has to hold the same graph, hold its own
# invariants, and share nothing with the graph it came from.


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_copied_graph_holds_the_same_store(case):
    G = build_case(case, store='slots')
    H = G.ops.copy()
    assert H._store is not G._store
    assert _snapshot(H._store) == _snapshot(G._store), case
    assert V.validate_internal_consistency(H._store, strict=False) == []


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_subgraph_holds_the_store_a_rebuild_would_give(case):
    """A selection numbers its own slots, so it is checked against a rebuild."""
    G = build_case(case, store='slots')
    nodes = [ref.id for ref in S.iter_entities(G) if ref.kind == S.NODE]
    for selection in (nodes, nodes[:-1]):
        _assert_incremental_matches_rebuild(G.ops.subgraph(selection), f'{case}/{selection}')


def test_a_subgraph_takes_the_weight_the_slice_gives_an_edge():
    G = build_case('binary_directed', store='slots')
    G.slices.add('heavy')
    G.slices.add_edge_to_slice('heavy', 'e_ab')
    for vertex_id in ('A', 'B'):
        G.slices.add_vertex_to_slice('heavy', vertex_id)
    G.attrs.set_edge_slice_attrs('heavy', 'e_ab', weight=10.0)
    H = G.subgraph_from_slice('heavy')
    assert S.edge_members(H._store, 'e_ab') == pytest.approx(
        {('A', ('_',)): 10.0, ('B', ('_',)): -10.0}
    )
    _assert_incremental_matches_rebuild(H, 'subgraph_from_slice')


def test_declaring_aspects_moves_the_keys_and_keeps_every_slot():
    """Every vertex changes identity and none changes address."""
    import warnings

    G = build_case('binary_directed', store='slots')
    before = {key[0]: slot for slot, key in G._store.live_entities()}
    sides_before = _snapshot(G._store)['sides']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        G.layers.set_aspects(['cond', 'time'], {'cond': ['ctrl'], 'time': ['t0']})
    after = {key[0]: slot for slot, key in G._store.live_entities()}
    assert after == before, 'a vertex changes identity, not address'
    assert {key[1] for _slot, key in G._store.live_entities()} == {('_', '_')}
    assert _snapshot(G._store)['sides'] != sides_before, 'the identities did move'
    assert V.validate_internal_consistency(G._store, strict=False) == []
    for edge_id, members in _snapshot(G._store)['members'].items():
        assert {key[0] for key in members} == {key[0] for key in sides_before[edge_id].source} | {
            key[0] for key in sides_before[edge_id].target
        }, edge_id


def test_flattening_a_multilayer_graph_installs_the_store_it_built():
    """The flat graph is built through the public API, so its store is the answer."""
    G = build_case('multilayer', store='slots')
    edge_id = S.edge_ids(G._store)[0]
    policy = {'var': 'score', 'threshold': 2.0}
    G.edge_direction_policy = {edge_id: policy}
    G.layers.flatten_layers()
    assert {key[1] for _slot, key in G._store.live_entities()} == {('_',)}
    assert S.edge_policies(G._store)[edge_id] == policy
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, 'flatten_layers')


def test_a_write_to_a_copy_leaves_the_graph_it_came_from_alone():
    G = build_case('binary_directed', store='slots')
    H = G.ops.copy()
    H.add_vertices(['Z'])
    H.remove_edges('e_ab')
    assert 'Z' not in set(G.vertices())
    assert 'e_ab' in set(S.edge_ids(G._store))


def test_removing_a_vertex_frees_its_slot_and_moves_no_other():
    """The record store renumbers its rows on a delete. The slot store does not."""
    G = build_case('binary_directed', store='slots')
    before = dict(G._store.live_entities())
    dropped = G._store.entity_slot(('A', ('_',)))
    G.remove_vertices(['A'])
    after = dict(G._store.live_entities())
    assert set(after) == set(before) - {dropped}
    assert all(after[slot] == before[slot] for slot in after)
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, 'remove_vertices')


@pytest.mark.parametrize('case', CASE_NAMES)
def test_removing_every_vertex_empties_the_store(case):
    G = build_case(case, store='slots')
    G.remove_vertices([ref.key for ref in S.iter_entities(G) if ref.kind == S.NODE])
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, case)


def test_a_sequence_of_mutations_reaches_the_store():
    G = build_case('binary_directed', store='slots')
    G.add_vertices(['D', 'E'])
    _assert_incremental_matches_rebuild(G, 'add_vertices')

    G.add_edges('C', 'D', edge_id='e_cd')
    _assert_incremental_matches_rebuild(G, 'add_edges')

    G.remove_edges('e_ab')
    _assert_incremental_matches_rebuild(G, 'remove_edges')

    G.remove_vertices(['E'])
    _assert_incremental_matches_rebuild(G, 'remove_vertices')

    G.make_undirected()
    _assert_incremental_matches_rebuild(G, 'make_undirected')
