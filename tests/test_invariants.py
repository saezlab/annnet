"""Invariant-checker tests over the operation-matrix fixtures.

The checker is the safety net of the refactor. A rule that the checker does not
enforce is a rule the refactor can break without anyone noticing. So most tests
here come in two parts. The first part shows the rule holds on every shape in
the operation matrix. The second part breaks the rule on purpose and shows the
checker reports it.

The rules come from the data model of the core. The numbering below follows
that document.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from annnet.core import _store as ST, _structure as S, _validate as V
from annnet.core._records import build_dataframe_from_rows

from ._fixtures import CASE_NAMES, build_case

FLAT = ('_',)


def problems_of(graph) -> list[str]:
    """Return the problem list without raising."""
    return V.validate_internal_consistency(graph, strict=False)


def assert_reports(graph, needle: str) -> None:
    """Assert the checker reports at least one problem naming ``needle``."""
    found = problems_of(graph)
    assert found, f'the checker reported nothing, expected a problem naming {needle!r}'
    assert any(needle in message for message in found), (
        f'no reported problem names {needle!r}. Reported: {found}'
    )


def table_rows(table) -> list[dict]:
    """Return a dataframe as a list of row dictionaries, backend independent."""
    return table.to_dicts() if hasattr(table, 'to_dicts') else list(table.rows(named=True))


# ---------------------------------------------------------------------------
# The clean baseline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_every_operation_matrix_case_is_consistent(case):
    assert problems_of(build_case(case)) == []


@pytest.mark.parametrize('case', CASE_NAMES)
def test_strict_mode_stays_quiet_on_a_consistent_graph(case):
    assert build_case(case).validate(strict=True) == []


def test_an_empty_graph_is_consistent():
    from annnet.core.graph import AnnNet

    assert problems_of(AnnNet(directed=True)) == []


# ---------------------------------------------------------------------------
# Store dispatch
# ---------------------------------------------------------------------------


def test_the_checker_picks_the_record_store_for_the_current_core():
    assert V.detect_store_kind(build_case('binary_directed')) == V.RECORD_STORE


def test_checks_for_rejects_an_unknown_store_kind():
    with pytest.raises(ValueError):
        V.checks_for('no_such_store')


def test_the_checker_picks_the_slot_store_for_a_slot_store():
    assert V.detect_store_kind(_slot_case()) == V.SLOT_STORE


def test_both_store_models_have_checks_registered():
    assert V.checks_for(V.RECORD_STORE)
    assert V.checks_for(V.SLOT_STORE)


# ---------------------------------------------------------------------------
# The slot store obeys the same rules
# ---------------------------------------------------------------------------


def _slot_case():
    """A slot store holding a self-loop, a boundary edge, and a plain edge."""
    store = ST.CoreState(directed=True)
    for node_id in ('A', 'B', 'C'):
        store.add_entity((node_id, FLAT))
    store.add_edge(
        'e_ab',
        [(('A', FLAT), 1.0, ST.SOURCE), (('B', FLAT), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.5,
    )
    store.add_edge(
        'e_loop',
        [(('A', FLAT), 0.5, ST.SOURCE), (('A', FLAT), -0.5, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=0.5,
    )
    store.add_edge(
        'b_out',
        [(('C', FLAT), -1.0, ST.SOURCE)],
        kind=ST.HYPER,
        directed=False,
        weight=1.0,
        explicit_coefficients=True,
    )
    return store


def test_a_clean_slot_store_is_consistent():
    assert problems_of(_slot_case()) == []


def test_a_slot_store_stays_consistent_through_churn():
    store = _slot_case()
    store.remove_edge('e_loop')
    assert problems_of(store) == []
    for dangling in store.remove_entity(('B', FLAT)):
        # The store reports the edges an entity removal leaves dangling, and the
        # caller has to deal with them. Keeping one would be a real problem.
        store.remove_edge(dangling)
    assert problems_of(store) == []
    store.add_entity(('D', FLAT))
    store.add_edge(
        'e_ad',
        [(('A', FLAT), 1.0, ST.SOURCE), (('D', FLAT), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )
    assert problems_of(store) == []
    store.compact_members()
    assert problems_of(store) == []


def test_the_slot_checker_reports_a_broken_bijection():
    store = _slot_case()
    store._entity_slot[('A', FLAT)] = 99
    assert_reports(store, 'slot 99')


def test_the_slot_checker_reports_a_freelist_that_holds_a_live_slot():
    store = _slot_case()
    store.entity_free.append(store.entity_slot(('A', FLAT)))
    assert_reports(store, 'freelist')


def test_the_slot_checker_reports_a_member_on_a_free_slot():
    store = _slot_case()
    slot = store.edge_slot('e_ab')
    store.member_ent[int(store.member_start[slot])] = 90
    assert_reports(store, 'holds no entity')


def test_the_slot_checker_reports_two_edges_sharing_a_member_segment():
    store = _slot_case()
    store.member_start[store.edge_slot('e_loop')] = int(store.member_start[store.edge_slot('e_ab')])
    assert_reports(store, 'shares member entry')


def test_the_slot_checker_reports_a_self_loop_that_lost_an_entry():
    """The regression that the record core has by design."""
    store = _slot_case()
    store.member_len[store.edge_slot('e_loop')] = 1
    assert_reports(store, 'self-loop')


def test_the_slot_checker_reports_a_stale_incidence_index():
    store = _slot_case()
    store._entity_edges[store.entity_slot(('A', FLAT))][99] = 1
    assert_reports(store, 'edge index')


def test_the_slot_checker_reports_an_index_that_names_the_wrong_side():
    """The index says which side an entity takes, so a wrong side is a defect.

    A traversal reads the side from the index and never from the member list, so
    nothing else would catch this.
    """
    store = _slot_case()
    index = store._entity_edges[store.entity_slot(('A', FLAT))]
    edge_slot = next(iter(index))
    index[edge_slot] = index[edge_slot] ^ 0b11
    assert_reports(store, 'edge index')


def test_the_slot_checker_reports_an_append_log_that_the_clock_denies():
    store = _slot_case()
    store.append_log_from_version = store.structure_version + 5
    assert_reports(store, 'append log')


# ---------------------------------------------------------------------------
# Rule 1 and rule 2 — identity and address agree, and no address is stale
# ---------------------------------------------------------------------------


def test_rule_1_reports_two_entities_that_claim_one_address():
    G = build_case('binary_directed')
    G._entities[('B', FLAT)].row_idx = G._entities[('A', FLAT)].row_idx
    assert_reports(G, 'row_idx')


def test_rule_1_reports_an_address_map_that_disagrees_with_the_entity():
    G = build_case('binary_directed')
    G._row_to_entity[0] = ('C', FLAT)
    assert_reports(G, '_row_to_entity')


def test_rule_2_reports_a_stale_address():
    G = build_case('binary_directed')
    G._row_to_entity[99] = ('A', FLAT)
    assert_reports(G, 'stale')


def test_rule_2_reports_a_stale_edge_address():
    G = build_case('binary_directed')
    G._col_to_edge[99] = 'e_ab'
    assert_reports(G, 'stale')


# ---------------------------------------------------------------------------
# Rule 3 — every member of an edge is a live entity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_rule_3_holds_on_every_case(case):
    G = build_case(case)
    live = {ref.key for ref in S.iter_entities(G)}
    for ref in S.iter_edges(G):
        assert set(S.edge_members(G, ref.id)) <= live


def test_rule_3_reports_a_member_that_is_not_an_entity():
    G = build_case('binary_directed')
    G._edges['e_ab'].src = 'ghost'
    assert_reports(G, 'ghost')


def test_rule_3_reports_a_hyperedge_member_that_is_not_an_entity():
    G = build_case('hyper_undirected')
    G._edges['h_abc'].src = frozenset({'A', 'ghost'})
    assert_reports(G, 'ghost')


# ---------------------------------------------------------------------------
# Rule 4 — the edge addresses form one contiguous block
# ---------------------------------------------------------------------------


def test_rule_4_reports_a_gap_in_the_edge_addresses():
    G = build_case('binary_directed')
    G._edges['e_bc'].col_idx = 7
    G._col_to_edge = {0: 'e_ab', 7: 'e_bc'}
    assert_reports(G, 'contiguous')


def test_rule_4_reports_two_edges_that_claim_one_address():
    G = build_case('binary_directed')
    G._edges['e_bc'].col_idx = 0
    assert_reports(G, 'col_idx')


# ---------------------------------------------------------------------------
# Rule 5 — the signs of a member list match the kind and the directedness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('case', 'edge_id'),
    [
        ('binary_directed', 'e_ab'),
        ('hyper_directed', 'h_ab_c'),
    ],
)
def test_rule_5_a_directed_edge_without_coefficients_has_both_signs(case, edge_id):
    members = S.edge_members(build_case(case), edge_id)
    assert {float(np.sign(value)) for value in members.values()} == {1.0, -1.0}


@pytest.mark.parametrize(
    ('case', 'edge_id'),
    [
        ('binary_undirected', 'e_ab'),
        ('hyper_undirected', 'h_abc'),
    ],
)
def test_rule_5_an_undirected_edge_without_coefficients_has_one_sign(case, edge_id):
    members = S.edge_members(build_case(case), edge_id)
    assert len({float(np.sign(value)) for value in members.values()}) == 1


def test_rule_5_reports_a_column_whose_signs_contradict_the_directedness():
    """A directly assigned matrix is the real way this rule breaks."""
    G = build_case('binary_directed')
    dense = G.X().toarray()
    column = G.idx.edge_to_col('e_ab')
    dense[:, column] = np.abs(dense[:, column])  # two positive entries on a directed edge
    G._matrix = sp.csr_array(dense)
    assert_reports(G, 'e_ab')


def test_rule_5_leaves_an_edge_with_explicit_coefficients_alone():
    """An explicit coefficient may take any value, so the sign rule does not apply."""
    assert problems_of(build_case('coefficient_edge')) == []
    assert problems_of(build_case('boundary_edge')) == []


# ---------------------------------------------------------------------------
# Rule 6 — an edge-entity lives on both axes under one id
# ---------------------------------------------------------------------------


def test_rule_6_holds_for_the_edge_entity_case():
    G = build_case('edge_entity')
    assert S.has_entity(G, 'ee_ab')
    assert S.has_edge(G, 'ee_ab')


def test_rule_6_reports_an_edge_entity_with_no_entity_side():
    G = build_case('edge_entity')
    del G._entities[('ee_ab', FLAT)]
    G._row_to_entity = {
        row: ekey for row, ekey in G._row_to_entity.items() if ekey != ('ee_ab', FLAT)
    }
    assert_reports(G, 'ee_ab')


def test_rule_6_reports_an_entity_marked_as_an_edge_with_no_edge_side():
    G = build_case('binary_directed')
    G._entities[('A', FLAT)].kind = 'edge_entity'
    assert_reports(G, 'A')


# ---------------------------------------------------------------------------
# Rule 7 and rule 8 — the materialized matrix agrees with the store
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_rule_7_every_nonzero_lands_on_a_live_row_and_a_live_column(case):
    G = build_case(case)
    live_rows = {G.idx.entity_to_row(ref.key) for ref in S.iter_entities(G)}
    live_cols = {G.idx.edge_to_col(ref.id) for ref in S.iter_edges(G)}
    block = G.X().tocoo()
    for row, col, value in zip(block.row, block.col, block.data, strict=False):
        if float(value) == 0.0:
            continue
        assert int(row) in live_rows
        assert int(col) in live_cols


def test_rule_7_reports_a_nonzero_on_a_row_that_holds_no_entity():
    G = build_case('binary_directed')
    shape = G.X().shape
    dense = np.zeros((shape[0] + 2, shape[1]), dtype=np.float32)
    dense[: shape[0], :] = G.X().toarray()
    dense[-1, 0] = 1.0
    G._matrix = sp.csr_array(dense)
    assert_reports(G, 'row')


@pytest.mark.parametrize('case', CASE_NAMES)
def test_rule_8_the_matrix_equals_the_member_lists(case):
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
        from_store = {k: v for k, v in S.edge_members(G, ref.id).items() if v != 0.0}
        assert set(from_matrix) == set(from_store)
        for member_key, value in from_matrix.items():
            assert from_store[member_key] == pytest.approx(value)


def test_rule_8_reports_a_matrix_cell_that_the_store_does_not_imply():
    G = build_case('binary_directed')
    dense = G.X().toarray()
    dense[G.idx.entity_to_row('C'), G.idx.edge_to_col('e_ab')] = 9.0
    G._matrix = sp.csr_array(dense)
    assert_reports(G, 'e_ab')


# ---------------------------------------------------------------------------
# Rule 9 — the node table stays node-level and the edge table stays edge-level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_rule_9_holds_on_every_case(case):
    G = build_case(case)
    node_ids = {ref.id for ref in S.iter_entities(G)}
    edge_ids = {ref.id for ref in S.iter_edges(G)}
    assert {row['vertex_id'] for row in table_rows(G.obs)} <= node_ids
    assert {row['edge_id'] for row in table_rows(G.var)} <= edge_ids


def test_rule_9_reports_a_node_row_for_a_node_that_does_not_exist():
    G = build_case('binary_directed')
    rows = table_rows(G.obs)
    rows.append({**rows[0], 'vertex_id': 'ghost'})
    G.vertex_attributes = build_dataframe_from_rows(rows)
    assert_reports(G, 'ghost')


def test_rule_9_reports_an_edge_row_for_an_edge_that_does_not_exist():
    G = build_case('binary_directed')
    rows = table_rows(G.var)
    rows.append({**rows[0], 'edge_id': 'ghost'})
    G.edge_attributes = build_dataframe_from_rows(rows)
    assert_reports(G, 'ghost')


# ---------------------------------------------------------------------------
# Rule 10 — a slice holds live elements only
# ---------------------------------------------------------------------------


def test_rule_10_holds_for_the_sliced_case():
    assert problems_of(build_case('sliced')) == []


def test_rule_10_reports_a_slice_that_holds_an_unknown_edge():
    G = build_case('sliced')
    G._slices['left'].edges.add('ghost')
    assert_reports(G, 'ghost')


def test_rule_10_reports_a_slice_that_holds_an_unknown_node():
    G = build_case('sliced')
    G._slices['left'].vertices.add('ghost')
    assert_reports(G, 'ghost')


def test_rule_10_reports_a_slice_that_holds_a_non_identity_node():
    G = build_case('sliced')
    G._slices['left'].vertices.add(3)
    assert_reports(G, 'bare')


# ---------------------------------------------------------------------------
# Rule 11 — a freed address never resolves to the wrong element
# ---------------------------------------------------------------------------


def test_rule_11_a_removed_node_leaves_no_address_that_resolves_to_it():
    G = build_case('binary_directed')
    removed_row = G.idx.entity_to_row('A')
    G.remove_vertex('A')
    assert problems_of(G) == []
    assert not S.has_entity(G, 'A')
    if removed_row in G._row_to_entity:
        assert G._row_to_entity[removed_row] != ('A', FLAT)


def test_rule_11_a_removed_edge_leaves_no_address_that_resolves_to_it():
    G = build_case('parallel_edge')
    G.remove_edge('e_first')
    assert problems_of(G) == []
    assert not S.has_edge(G, 'e_first')
    assert 'e_first' not in set(G._col_to_edge.values())
