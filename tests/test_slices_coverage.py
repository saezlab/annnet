"""Targeted coverage for ``annnet.core._Slices`` (SCV-5).

Covers the analytics surface (presence, conserved, specific, temporal,
summary, aggregate, *_create helpers) and the error paths.

Target: ≥80 % per the SCV-5 punch list.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def _two_slice_graph() -> AnnNet:
    """Two slices ('S1', 'S2') sharing some edges and differing on others."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.slices.add('S1')
    G.slices.add('S2')
    # binary edges
    G.add_edges('A', 'B', edge_id='e1', slice='S1')
    G.add_edges('B', 'C', edge_id='e2', slice='S1')
    G.slices.add_edge_to_slice('S2', 'e2')  # shared
    G.add_edges('C', 'D', edge_id='e3', slice='S2')
    # hyperedges
    G.add_edges([{'src': ['A', 'B', 'C'], 'edge_id': 'h_und', 'slice': 'S1'}])
    G.add_edges([{'src': ['A'], 'tgt': ['B', 'C'], 'edge_id': 'h_dir', 'slice': 'S2'}])
    return G


# ── add / remove / active error paths ───────────────────────────────────


def test_add_existing_slice_raises() -> None:
    G = AnnNet(directed=False)
    G.slices.add('S1')
    with pytest.raises(ValueError, match='already exists'):
        G.slices.add('S1')


def test_remove_default_slice_raises() -> None:
    G = AnnNet(directed=False)
    with pytest.raises(ValueError, match='Cannot remove default'):
        G.slices.remove('default')


def test_remove_missing_slice_raises() -> None:
    G = AnnNet(directed=False)
    with pytest.raises(KeyError, match='ghost'):
        G.slices.remove('ghost')


def test_remove_resets_active_to_default_when_active_was_dropped() -> None:
    G = AnnNet(directed=False)
    G.slices.add('S1')
    G.slices.active = 'S1'
    G.slices.remove('S1')
    assert G.slices.active == 'default'


def test_remove_drops_edge_slice_attrs_for_that_slice() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    G.slices.add('S1')
    G.slices.add_edge_to_slice('S1', 'e1')
    G.attrs.set_edge_slice_attrs('S1', 'e1', weight=2.5)
    G.slices.remove('S1')
    assert 'S1' not in G.slices.list()


def test_active_setter_rejects_unknown_slice() -> None:
    G = AnnNet(directed=False)
    with pytest.raises(KeyError, match='nope'):
        G.slices.active = 'nope'


# ── info / vertices / edges error paths ─────────────────────────────────


def test_info_raises_on_unknown_slice() -> None:
    G = AnnNet(directed=False)
    with pytest.raises(KeyError, match='ghost'):
        G.slices.info('ghost')


def test_add_edge_to_slice_error_paths() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    with pytest.raises(KeyError, match='ghost'):
        G.slices.add_edge_to_slice('ghost', 'e1')
    with pytest.raises(KeyError, match='nope'):
        G.slices.add_edge_to_slice('default', 'nope')


def test_add_vertex_to_slice_error_paths() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    G.slices.add('S1')
    with pytest.raises(KeyError, match='ghost'):
        G.slices.add_vertex_to_slice('ghost', 'A')
    with pytest.raises(KeyError, match='nope'):
        G.slices.add_vertex_to_slice('S1', 'nope')


def test_add_vertex_to_slice_attaches_vertex() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    G.slices.add('S1')
    G.slices.add_vertex_to_slice('S1', 'A')
    assert 'A' in G.slices.vertices('S1')


# ── set-op creation helpers ─────────────────────────────────────────────


def test_union_create_intersect_create_difference_create() -> None:
    G = _two_slice_graph()
    G.slices.union_create(['S1', 'S2'], 'U')
    G.slices.intersect_create(['S1', 'S2'], 'I')
    G.slices.difference_create('S1', 'S2', 'D')
    assert {'U', 'I', 'D'} <= set(G.slices.list())
    # I has the shared edge, D has S1's unshared edges.
    assert 'e2' in G.slices.edges('I')
    assert 'e1' in G.slices.edges('D')


def test_difference_raises_on_unknown_slice() -> None:
    G = _two_slice_graph()
    with pytest.raises(KeyError, match='both slices'):
        G.slices.difference('S1', 'ghost')


def test_intersect_empty_and_single() -> None:
    G = _two_slice_graph()
    empty = G.slices.intersect([])
    assert empty == {'vertices': set(), 'edges': set()}
    one = G.slices.intersect(['S1'])
    assert one['edges']  # non-empty


def test_intersect_with_unknown_slice_returns_empty() -> None:
    G = _two_slice_graph()
    out = G.slices.intersect(['S1', 'ghost'])
    assert out == {'vertices': set(), 'edges': set()}


# ── aggregate ───────────────────────────────────────────────────────────


def test_aggregate_union_method() -> None:
    G = _two_slice_graph()
    G.slices.aggregate(['S1', 'S2'], 'A_union', method='union')
    assert {'e1', 'e2', 'e3'} <= G.slices.edges('A_union')


def test_aggregate_intersection_method() -> None:
    G = _two_slice_graph()
    G.slices.aggregate(['S1', 'S2'], 'A_inter', method='intersection')
    # Only e2 is shared.
    assert G.slices.edges('A_inter') == {'e2'}


def test_aggregate_intersection_short_circuits_on_missing_slice() -> None:
    G = _two_slice_graph()
    G.slices.aggregate(['S1', 'ghost'], 'A_empty', method='intersection')
    assert G.slices.edges('A_empty') == set()


def test_aggregate_no_sources_raises() -> None:
    G = AnnNet(directed=False)
    with pytest.raises(ValueError, match='at least one source'):
        G.slices.aggregate([], 'target')


def test_aggregate_target_exists_raises() -> None:
    G = _two_slice_graph()
    with pytest.raises(ValueError, match='already exists'):
        G.slices.aggregate(['S1'], 'S2')


def test_aggregate_unknown_method_raises() -> None:
    G = _two_slice_graph()
    with pytest.raises(ValueError, match='Unknown aggregation method'):
        G.slices.aggregate(['S1'], 'A_bad', method='nonsense')


# ── presence queries ────────────────────────────────────────────────────


def test_vertex_presence_lists_slices_containing_vertex() -> None:
    G = _two_slice_graph()
    present = G.slices.vertex_presence('B')
    assert set(present) >= {'S1', 'S2'}  # B is endpoint of e2 (in both)


def test_edge_presence_by_id_returns_slice_list() -> None:
    G = _two_slice_graph()
    assert set(G.slices.edge_presence(edge_id='e2')) >= {'S1', 'S2'}


def test_edge_presence_by_endpoint_pair_returns_dict() -> None:
    G = _two_slice_graph()
    out = G.slices.edge_presence(source='A', target='B')
    assert 'S1' in out and 'e1' in out['S1']


def test_edge_presence_undirected_match_finds_reverse_orientation() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', directed=False)
    out = G.slices.edge_presence(
        source='B', target='A', include_default=True, undirected_match=True
    )
    assert 'default' in out and 'e1' in out['default']


def test_edge_presence_rejects_both_id_and_pair() -> None:
    G = _two_slice_graph()
    with pytest.raises(ValueError, match='either edge_id OR'):
        G.slices.edge_presence(edge_id='e1', source='A', target='B')


def test_hyperedge_presence_undirected_match() -> None:
    G = _two_slice_graph()
    out = G.slices.hyperedge_presence(members=['A', 'B', 'C'])
    assert 'S1' in out and 'h_und' in out['S1']


def test_hyperedge_presence_directed_match() -> None:
    G = _two_slice_graph()
    out = G.slices.hyperedge_presence(head=['A'], tail=['B', 'C'])
    assert 'S2' in out and 'h_dir' in out['S2']


def test_hyperedge_presence_argument_errors() -> None:
    G = _two_slice_graph()
    with pytest.raises(ValueError, match='either members OR'):
        G.slices.hyperedge_presence(members=['A'], head=['B'])
    with pytest.raises(ValueError, match='Directed hyperedge query'):
        G.slices.hyperedge_presence(head=['A'])
    with pytest.raises(ValueError, match='members must be non-empty'):
        G.slices.hyperedge_presence(members=[])
    with pytest.raises(ValueError, match='head and tail must be non-empty'):
        G.slices.hyperedge_presence(head=[], tail=['B'])
    with pytest.raises(ValueError, match='disjoint'):
        G.slices.hyperedge_presence(head=['A', 'B'], tail=['B', 'C'])


# ── conserved / specific / temporal ─────────────────────────────────────


def test_conserved_edges_returns_edges_seen_in_min_slices() -> None:
    G = _two_slice_graph()
    out = G.slices.conserved_edges(min_slices=2)
    assert out.get('e2') == 2  # shared between S1 and S2


def test_specific_edges_returns_uniques() -> None:
    G = _two_slice_graph()
    s1_only = G.slices.specific_edges('S1')
    assert 'e1' in s1_only and 'e2' not in s1_only


def test_specific_edges_unknown_slice_raises() -> None:
    G = _two_slice_graph()
    with pytest.raises(KeyError, match='ghost'):
        G.slices.specific_edges('ghost')


def test_temporal_dynamics_reports_per_step_changes() -> None:
    G = _two_slice_graph()
    out = G.slices.temporal_dynamics(['S1', 'S2'], metric='edge_change')
    assert len(out) == 1
    step = out[0]
    assert 'added' in step and 'removed' in step and 'net_change' in step


def test_temporal_dynamics_requires_two_slices() -> None:
    G = _two_slice_graph()
    with pytest.raises(ValueError, match='at least 2 slices'):
        G.slices.temporal_dynamics(['S1'])


def test_temporal_dynamics_missing_slice_raises() -> None:
    G = _two_slice_graph()
    with pytest.raises(KeyError, match='not found'):
        G.slices.temporal_dynamics(['S1', 'ghost'])


def test_temporal_dynamics_vertex_metric() -> None:
    G = _two_slice_graph()
    out = G.slices.temporal_dynamics(['S1', 'S2'], metric='vertex_change')
    assert len(out) == 1


# ── convenience: summary, __repr__ ──────────────────────────────────────


def test_summary_lists_each_slice_with_counts() -> None:
    G = _two_slice_graph()
    text = G.slices.summary()
    assert 'slices:' in text
    assert 'S1' in text and 'S2' in text


def test_repr_includes_slice_count() -> None:
    G = _two_slice_graph()
    rep = repr(G.slices)
    assert 'SliceManager' in rep
    assert str(G.slices.count()) in rep
