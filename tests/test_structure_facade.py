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

from annnet.core import _build, _store as ST, _structure as S
from annnet.core.graph import AnnNet

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
        # The two roles of a directed self-loop cancel, which is the one
        # intended change the specification states.
        ('self_loop', 'e_loop', {key('A'): 0.0}),
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


def _mixed_chain(length: int) -> AnnNet:
    """A directed chain whose even edges declare themselves undirected.

    Every fixture of the operation matrix holds fewer edges than the store
    allocates up front, so none of them ever sees the edge arrays grow. This one
    is long enough to force the growth twice over, and it mixes the two ways an
    edge can answer for its directedness: a declared value, and the default of
    the graph.
    """
    G = AnnNet(directed=True)
    G.add_vertices([f'v{i}' for i in range(length + 1)])
    G.add_edges(
        [
            {
                'source': f'v{i}',
                'target': f'v{i + 1}',
                'edge_id': f'e{i}',
                **({'edge_directed': False} if i % 2 == 0 else {}),
            }
            for i in range(length)
        ]
    )
    return G


def test_entity_edges_reads_the_directedness_of_an_edge_the_arrays_have_outgrown():
    G = _mixed_chain(20)
    # ``e9`` is directed and runs into ``v10``; ``e10`` declares itself
    # undirected and so counts in both directions.
    assert set(S.entity_edges(G, 'v10', 'out')) == {'e10'}
    assert set(S.entity_edges(G, 'v10', 'in')) == {'e9', 'e10'}
    assert set(S.entity_edges(G, 'v11', 'out')) == {'e10', 'e11'}


def test_neighbors_reads_the_directedness_of_an_edge_the_arrays_have_outgrown():
    G = _mixed_chain(20)
    # An unqualified query walks an undirected edge both ways and a directed one
    # forwards only, so ``v9`` is reached from ``v10`` only when asked for.
    assert set(S.neighbors(G, 'v10', 'both')) == {'v11'}
    assert set(S.neighbors(G, 'v10', 'in')) == {'v9', 'v11'}
    assert set(S.neighbors(G, 'v11', 'both')) == {'v10', 'v12'}


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


@pytest.mark.parametrize('case', CASE_NAMES)
def test_edge_ids_lists_what_iter_edges_yields(case):
    """The cheap enumeration and the full one must agree, and in the same order."""
    G = build_case(case)
    assert S.edge_ids(G) == [ref.id for ref in S.iter_edges(G)]


@pytest.mark.parametrize('case', CASE_NAMES)
def test_entity_keys_lists_what_iter_entities_yields(case):
    G = build_case(case)
    assert S.entity_keys(G) == [ref.key for ref in S.iter_entities(G)]


@pytest.mark.parametrize('case', CASE_NAMES)
def test_node_ids_names_each_node_once(case):
    G = build_case(case)
    ids = S.node_ids(G)
    assert len(ids) == len(set(ids))
    assert set(ids) == {key[0] for key in S.node_keys(G)}


def test_node_ids_folds_the_layers_of_a_multilayer_graph():
    G = build_case('multilayer')
    assert len(S.node_keys(G)) == 4
    assert S.node_ids(G) == ['A', 'B']


@pytest.mark.parametrize('case', CASE_NAMES)
def test_node_keys_leaves_out_the_edge_entities(case):
    G = build_case(case)
    assert S.node_keys(G) == [ref.key for ref in S.iter_entities(G) if ref.kind == S.NODE]


def test_entities_by_id_groups_every_layer_under_one_id():
    G = build_case('multilayer')
    grouped = S.entities_by_id(G)
    assert set(grouped) == {'A', 'B'}
    assert {ref.layer for ref in grouped['A']} == {('t0',), ('t1',)}


def test_entities_by_id_holds_one_entry_for_a_flat_graph():
    G = build_case('binary_directed')
    grouped = S.entities_by_id(G)
    assert set(grouped) == {'A', 'B', 'C'}
    assert [ref.key for ref in grouped['A']] == [key('A')]


def test_entities_by_id_includes_an_edge_entity():
    G = build_case('edge_entity')
    assert S.entities_by_id(G)['ee_ab'][0].kind == S.EDGE_ENTITY


def test_has_entity_id_answers_for_a_bare_id_alone():
    G = build_case('binary_directed')
    assert S.has_entity_id(G, 'A') is True
    assert S.has_entity_id(G, 'Z') is False


def test_has_entity_id_can_ask_for_one_kind():
    G = build_case('edge_entity')
    assert S.has_entity_id(G, 'ee_ab') is True
    assert S.has_entity_id(G, 'ee_ab', kind=S.NODE) is False
    assert S.has_entity_id(G, 'ee_ab', kind=S.EDGE_ENTITY) is True
    assert S.has_entity_id(G, 'A', kind=S.NODE) is True


def test_has_entity_id_is_true_for_an_id_that_more_than_one_layer_carries():
    """``has_entity`` cannot answer this, because a bare id names no one entity."""
    G = build_case('multilayer')
    assert S.has_entity_id(G, 'A') is True
    assert S.has_entity(G, 'A') is False
    assert S.has_entity_id(G, 'Z') is False


@pytest.mark.parametrize('case', ['binary_directed', 'multilayer'])
def test_has_entity_id_asks_the_index_and_never_walks_the_graph(case, monkeypatch):
    """One id lookup costs one index read, whatever the graph holds.

    A walk here is the whole cost of loading a file, because a slice names every
    vertex it holds by its bare id and asks this once per name.
    """
    G = build_case(case)
    store = _build.rebuild_store(G)

    def refuse():
        raise AssertionError('has_entity_id walked every entity of the graph')

    monkeypatch.setattr(store, 'live_entities', refuse)
    for ref in S.iter_entities(G):
        assert S.has_entity_id(store, ref.id) is True
    assert S.has_entity_id(store, 'no_such_id') is False


def test_edges_between_reads_no_member_list_for_a_binary_edge(monkeypatch):
    """A binary edge names two entities, and the index already says which.

    This is what ``has_edge(source, target)`` asks, so it is a hot path. Reading
    the member list of every edge that touches the source made it twenty times
    dearer, which is the regression this pins.
    """
    G = build_case('binary_directed')

    def refuse(*_args, **_kwargs):
        raise AssertionError('edges_between read the member list of a binary edge')

    monkeypatch.setattr(G._store, 'endpoints', refuse)
    assert S.edges_between(G, 'A', 'B') == ['e_ab']
    assert S.edges_between(G, 'B', 'A') == []
    assert S.edges_between(G, 'A', 'no_such_entity') == []
    assert S.edges_between(G, 'no_such_entity', 'B') == []


def test_edges_between_leaves_out_a_hyperedge_over_the_same_two_entities():
    """A hyperedge of two members is still not an edge between them."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges([{'head': ['A'], 'tail': ['B'], 'edge_id': 'h1'}])

    assert S.edges_between(G, 'A', 'B') == []

    G.add_edges('A', 'B', edge_id='e_ab')
    assert S.edges_between(G, 'A', 'B') == ['e_ab']


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------
# A count is the size of an enumeration, so the two must never disagree. The
# facade answers a count without walking the graph, which is why it exists as
# its own question.


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_count_equals_the_length_of_its_enumeration(case):
    G = build_case(case)
    assert S.entity_count(G) == len(list(S.iter_entities(G)))
    assert S.edge_count(G) == len(list(S.iter_edges(G)))
    assert S.node_count(G) == sum(1 for ref in S.iter_entities(G) if ref.kind == S.NODE)


def test_node_count_leaves_out_an_edge_entity():
    G = build_case('edge_entity')
    assert S.entity_count(G) == 4
    assert S.node_count(G) == 3


def test_edge_count_leaves_out_an_edge_that_carries_no_structure():
    G = build_case('binary_directed')
    before = S.edge_count(G)
    G._ensure_edge_entity_placeholder('e_future')
    assert S.has_edge(G, 'e_future') is True
    assert S.edge_count(G) == before


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


# ---------------------------------------------------------------------------
# The facade answers the same for a graph and for a bare store
# ---------------------------------------------------------------------------
# A caller passes a graph or the canonical store of one and gets the same answer
# either way. That is what lets the invariant checker and the store tests ask
# about a store no graph holds. These run the contract twice, once each way, over
# a store rebuilt from the graph.


@pytest.fixture(params=CASE_NAMES)
def both_stores(request):
    graph = build_case(request.param)
    return request.param, graph, _build.rebuild_store(graph)


def test_the_facade_binds_the_store_constants_rather_than_copying_them():
    """Bound from the store itself, so no value can drift out of step with it."""
    assert (S._INHERIT, S._TARGET) == (ST.INHERIT, ST.TARGET)
    assert (S._ON_SOURCE, S._ON_TARGET, S._SLOT_NODE) == (ST.ON_SOURCE, ST.ON_TARGET, ST.NODE)


def test_the_slot_store_answers_the_facade(both_stores):
    _case, _graph, store = both_stores
    assert S.is_slot_backed(store)
    assert S.store_of(store) is store


def test_both_stores_iterate_the_same_entities(both_stores):
    _case, graph, store = both_stores
    assert [ref.key for ref in S.iter_entities(graph)] == [
        ref.key for ref in S.iter_entities(store)
    ]
    assert [ref.kind for ref in S.iter_entities(graph)] == [
        ref.kind for ref in S.iter_entities(store)
    ]


def test_both_stores_iterate_the_same_edges(both_stores):
    _case, graph, store = both_stores
    assert {ref.id for ref in S.iter_edges(graph)} == {ref.id for ref in S.iter_edges(store)}


def test_both_stores_report_the_same_edge_reference(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_edges(graph):
        found = S.edge_ref(store, ref.id)
        assert found.kind == ref.kind, f'{case}/{ref.id}'
        assert found.directed == ref.directed, f'{case}/{ref.id}'
        assert found.weight == pytest.approx(ref.weight), f'{case}/{ref.id}'


def test_both_stores_report_the_same_endpoints(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_edges(graph):
        expected = S.edge_sides(graph, ref.id)
        found = S.edge_endpoints(store, ref.id)
        assert {key[0] for key in found.source} == {_bare(e) for e in expected.source}, (
            f'{case}/{ref.id}'
        )
        assert {key[0] for key in found.target} == {_bare(e) for e in expected.target}, (
            f'{case}/{ref.id}'
        )


def test_both_stores_answer_existence_the_same(both_stores):
    _case, graph, store = both_stores
    for ref in S.iter_entities(graph):
        assert S.has_entity(store, ref.key) is True
    for ref in S.iter_edges(graph):
        assert S.has_edge(store, ref.id) is True
    assert S.has_entity(store, ('no_such_node', FLAT)) is False
    assert S.has_edge(store, 'no_such_edge') is False


def test_both_stores_reject_an_unknown_edge_the_same_way(both_stores):
    _case, _graph, store = both_stores
    with pytest.raises(KeyError):
        S.edge_members(store, 'no_such_edge')
    with pytest.raises(KeyError):
        S.edge_endpoints(store, 'no_such_edge')


def test_both_stores_list_the_same_ids_in_the_same_order(both_stores):
    case, graph, store = both_stores
    assert S.edge_ids(store) == S.edge_ids(graph), case
    assert S.entity_keys(store) == S.entity_keys(graph), case
    assert S.node_keys(store) == S.node_keys(graph), case


def test_both_stores_group_entities_by_id_the_same(both_stores):
    case, graph, store = both_stores
    expected = {
        node_id: sorted(ref.key for ref in refs)
        for node_id, refs in S.entities_by_id(graph).items()
    }
    found = {
        node_id: sorted(ref.key for ref in refs)
        for node_id, refs in S.entities_by_id(store).items()
    }
    assert found == expected, case


def test_both_stores_answer_for_a_bare_id_the_same(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_entities(graph):
        assert S.has_entity_id(store, ref.id) is True, f'{case}/{ref.id}'
    assert S.has_entity_id(store, 'no_such_id') is False


def test_both_stores_name_the_sides_in_the_same_form(both_stores):
    """``edge_sides`` answers in the form the public API shows, on either store.

    A flat graph names an entity by its bare id. Answering with a key instead
    would still name the right entity, and every caller that compares the answer
    against ids the public API gave it would silently find nothing.
    """
    case, graph, store = both_stores
    for ref in S.iter_edges(graph):
        assert S.edge_sides(store, ref.id) == S.edge_sides(graph, ref.id), f'{case}/{ref.id}'


def test_both_stores_report_the_same_counts(both_stores):
    case, graph, store = both_stores
    assert S.entity_count(store) == S.entity_count(graph), case
    assert S.node_count(store) == S.node_count(graph), case
    assert S.edge_count(store) == S.edge_count(graph), case


def test_positions_stay_contiguous_on_the_slot_store(both_stores):
    """A slot is stable, but the position it maps to is dense and starts at zero."""
    _case, _graph, store = both_stores
    rows = [S.entity_row(store, ref.key) for ref in S.iter_entities(store)]
    columns = [S.edge_column(store, ref.id) for ref in S.iter_edges(store)]
    assert rows == list(range(len(rows)))
    assert columns == list(range(len(columns)))


def test_both_stores_name_the_entity_of_a_row_the_same_way(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_entities(graph):
        row = S.entity_row(graph, ref.key)
        assert S.entity_key_of_row(store, row) == ref.key, f'{case}/{ref.key}'


def test_a_row_no_entity_occupies_is_rejected(both_stores):
    _case, _graph, store = both_stores
    with pytest.raises(KeyError):
        S.entity_key_of_row(store, S.entity_count(store))


def _bare(endpoint):
    return endpoint[0] if isinstance(endpoint, tuple) else endpoint


def test_both_stores_report_the_same_incident_edges(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_entities(graph):
        for direction in ('in', 'out', 'both'):
            expected = set(S.entity_edges(graph, ref.key, direction))
            found = set(S.entity_edges(store, ref.key, direction))
            assert found == expected, f'{case}/{ref.key}/{direction}'


def test_both_stores_report_the_same_neighbors(both_stores):
    case, graph, store = both_stores
    for ref in S.iter_entities(graph):
        for direction in ('in', 'out', 'both'):
            expected = {_bare(item) for item in S.neighbors(graph, ref.key, direction)}
            found = {_bare(item) for item in S.neighbors(store, ref.key, direction)}
            assert found == expected, f'{case}/{ref.key}/{direction}'


def test_the_slot_store_rejects_an_unknown_direction():
    store = _build.rebuild_store(build_case('binary_directed'))
    with pytest.raises(ValueError):
        S.entity_edges(store, ('A', FLAT), 'sideways')
