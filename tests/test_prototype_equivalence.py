"""Every write reaches the store, and the store still says what it describes.

There was a second store once, and these tests compared the two. There is one
now, so what is left is the two checks that a single store can be held to.

A graph is read back through the query facade as the definitions a loader hands
over, and a store is filled from those alone. The result has to hold the same
graph. That catches a write that changed one field and left a dependent one
behind, because the rebuild derives the dependent field again.

What a round trip cannot catch is a derived index that no longer matches the
member lists, because the rebuild builds the index from the same lists. The
invariant checker holds those rules, and the degree test below is the one that
walks every member entry to check the index against them.

One shape is intended and stated in the specification: a directed self-loop
keeps both of its roles, so its signed incidence column sums to zero instead of
holding one negative value. Every test below either avoids that column or
accounts for it.
"""

from __future__ import annotations

import pytest

from annnet.core import _build, _matrices as M, _structure as S, _validate as V

from ._fixtures import CASE_NAMES, build_case

# The one case whose signed incidence column changes on purpose.
SELF_LOOP_CASES = {'self_loop'}


@pytest.fixture(params=CASE_NAMES)
def pair(request):
    graph = build_case(request.param)
    return request.param, graph, graph._store


def test_a_rebuilt_store_holds_its_invariants(pair):
    _case, graph, _store = pair
    assert V.validate_internal_consistency(_build.rebuild_store(graph), strict=False) == []


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


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_slot_backed_graph_holds_its_invariants(case):
    built = build_case(case)
    assert V.validate_internal_consistency(built._store, strict=False) == []


# ---------------------------------------------------------------------------
# Every write reaches the store
# ---------------------------------------------------------------------------
# The gateway writes the store as each mutation lands. Reading the graph back as
# definitions and filling a store from those alone must therefore give the same
# graph. A write that reached one field and not the field derived from it shows
# up here as a difference, which is how four of them were found.


def _snapshot(store):
    """Everything the store holds, addressed by identity alone."""
    return {
        'entities': S.entity_keys(store),
        'kinds': {key: S.entity_ref(store, key).kind for key in S.entity_keys(store)},
        'edges': S.edge_ids(store),
        'sides': {eid: S.edge_sides(store, eid) for eid in S.edge_ids(store)},
        'members': {eid: S.edge_members(store, eid) for eid in S.edge_ids(store)},
        'policies': S.edge_policies(store),
    }


def _assert_incremental_matches_rebuild(G, note):
    assert _snapshot(_build.rebuild_store(G)) == _snapshot(G._store), note
    assert V.validate_internal_consistency(G._store, strict=False) == [], note


@pytest.mark.parametrize('case', CASE_NAMES)
def test_building_a_shape_reaches_the_store(case):
    _assert_incremental_matches_rebuild(build_case(case), case)


# ---------------------------------------------------------------------------
# A copy takes the store with it
# ---------------------------------------------------------------------------
# Copying a graph copies its slot arrays instead of rebuilding the store from
# what was installed. So the copy has to hold the same graph, hold its own
# invariants, and share nothing with the graph it came from.


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_copied_graph_holds_the_same_store(case):
    G = build_case(case)
    H = G.ops.copy()
    assert H._store is not G._store
    assert _snapshot(H._store) == _snapshot(G._store), case
    assert V.validate_internal_consistency(H._store, strict=False) == []


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_subgraph_holds_the_store_a_rebuild_would_give(case):
    """A selection numbers its own slots, so it is checked against a rebuild."""
    G = build_case(case)
    nodes = [ref.id for ref in S.iter_entities(G) if ref.kind == S.NODE]
    for selection in (nodes, nodes[:-1]):
        _assert_incremental_matches_rebuild(G.ops.subgraph(selection), f'{case}/{selection}')


def test_a_subgraph_takes_the_weight_the_slice_gives_an_edge():
    G = build_case('binary_directed')
    G.slices.add('heavy')
    G.slices.add_edge_to_slice('heavy', 'e_ab')
    for node_id in ('A', 'B'):
        G.slices.add_node_to_slice('heavy', node_id)
    G.attrs.set_edge_slice_attrs('heavy', 'e_ab', weight=10.0)
    H = G.subgraph_from_slice('heavy')
    assert S.edge_members(H._store, 'e_ab') == pytest.approx(
        {('A', ('_',)): 10.0, ('B', ('_',)): -10.0}
    )
    _assert_incremental_matches_rebuild(H, 'subgraph_from_slice')


def test_declaring_aspects_moves_the_keys_and_keeps_every_slot():
    """Every node changes identity and none changes address."""
    import warnings

    G = build_case('binary_directed')
    before = {key[0]: slot for slot, key in G._store.live_entities()}
    sides_before = _snapshot(G._store)['sides']
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        G.layers.set_aspects(['cond', 'time'], {'cond': ['ctrl'], 'time': ['t0']})
    after = {key[0]: slot for slot, key in G._store.live_entities()}
    assert after == before, 'a node changes identity, not address'
    assert {key[1] for _slot, key in G._store.live_entities()} == {('_', '_')}
    assert _snapshot(G._store)['sides'] != sides_before, 'the identities did move'
    assert V.validate_internal_consistency(G._store, strict=False) == []
    for edge_id, members in _snapshot(G._store)['members'].items():
        assert {key[0] for key in members} == {key[0] for key in sides_before[edge_id].source} | {
            key[0] for key in sides_before[edge_id].target
        }, edge_id


def test_flattening_a_multilayer_graph_installs_the_store_it_built():
    """The flat graph is built through the public API, so its store is the answer."""
    G = build_case('multilayer')
    edge_id = S.edge_ids(G._store)[0]
    policy = {'var': 'score', 'threshold': 2.0}
    G.edge_direction_policy = {edge_id: policy}
    G.layers.flatten_layers()
    assert {key[1] for _slot, key in G._store.live_entities()} == {('_',)}
    assert S.edge_policies(G._store)[edge_id] == policy
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, 'flatten_layers')


def test_the_default_direction_of_a_graph_reaches_the_store():
    """An edge that declares no direction of its own answers with the graph default."""
    from annnet.core import _mutate

    G = build_case('binary_directed')
    _mutate.set_edge_field(G, 'e_ab', 'directed', None)
    assert S.edge_ref(G, 'e_ab').directed is True, 'the default the graph was built with'
    G.directed = False
    assert S.edge_ref(G, 'e_ab').directed is False, 'the default the graph now declares'


def test_a_write_to_a_copy_leaves_the_graph_it_came_from_alone():
    G = build_case('binary_directed')
    H = G.ops.copy()
    H.add_nodes(['Z'])
    H.remove_edges('e_ab')
    assert 'Z' not in set(G.nodes())
    assert 'e_ab' in set(S.edge_ids(G._store))


def test_removing_a_node_frees_its_slot_and_moves_no_other():
    """The record store renumbers its rows on a delete. The slot store does not."""
    G = build_case('binary_directed')
    before = dict(G._store.live_entities())
    dropped = G._store.entity_slot(('A', ('_',)))
    G.remove_nodes(['A'])
    after = dict(G._store.live_entities())
    assert set(after) == set(before) - {dropped}
    assert all(after[slot] == before[slot] for slot in after)
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, 'remove_nodes')


@pytest.mark.parametrize('case', CASE_NAMES)
def test_removing_every_node_empties_the_store(case):
    G = build_case(case)
    G.remove_nodes([ref.key for ref in S.iter_entities(G) if ref.kind == S.NODE])
    assert V.validate_internal_consistency(G._store, strict=False) == []
    _assert_incremental_matches_rebuild(G, case)


def test_a_sequence_of_mutations_reaches_the_store():
    G = build_case('binary_directed')
    G.add_nodes(['D', 'E'])
    _assert_incremental_matches_rebuild(G, 'add_nodes')

    G.add_edges('C', 'D', edge_id='e_cd')
    _assert_incremental_matches_rebuild(G, 'add_edges')

    G.remove_edges('e_ab')
    _assert_incremental_matches_rebuild(G, 'remove_edges')

    G.remove_nodes(['E'])
    _assert_incremental_matches_rebuild(G, 'remove_nodes')

    G.make_undirected()
    _assert_incremental_matches_rebuild(G, 'make_undirected')


def test_setting_the_kind_of_an_entity_reaches_the_store():
    """The legacy kind setters change one field, so neither rebuilds the store."""
    G = build_case('binary_directed')
    G._set_entity_kinds_by_id({'A': 'edge'})
    assert S.entity_ref(G._store, ('A', ('_',))).kind == S.EDGE_ENTITY
    _assert_incremental_matches_rebuild(G, 'entity kinds')

    G._set_entity_kinds({('B', ('_',)): 'edge_entity'})
    assert S.entity_ref(G._store, ('B', ('_',))).kind == S.EDGE_ENTITY
    _assert_incremental_matches_rebuild(G, 'set_entity_kinds')

    G._set_entity_kinds({('Z', ('_',)): 'node'})
    assert S.has_entity(G._store, ('Z', ('_',)))
    _assert_incremental_matches_rebuild(G, 'set_entity_kinds/new')


# ---------------------------------------------------------------------------
# A file is the one check the store is not the source of
# ---------------------------------------------------------------------------
# Everything above reads the store to say what the store should hold, so a round
# trip through it cannot notice a graph that is wrong in the same way twice. A
# file can. It is written by one body of code, read by another, and the store
# that comes back is filled from the file alone. So a field lost on the way out
# or on the way in shows up here and nowhere else.


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_graph_read_from_a_file_holds_the_store_it_was_written_from(case, tmp_path):
    from annnet.io import annnet_format

    path = tmp_path / f'{case}.annnet'
    written = build_case(case)
    annnet_format.write(written, path)
    loaded = annnet_format.read(path)
    assert V.validate_internal_consistency(loaded._store, strict=False) == []
    assert _snapshot(loaded._store) == _snapshot(written._store), case
    _assert_incremental_matches_rebuild(loaded, case)
