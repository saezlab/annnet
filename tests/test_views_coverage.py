"""Coverage of the composable view and the derived tables.

A predicate exception propagates rather than producing zero matches; a view
whose predicate matches nothing holds nothing; every table is read through
``G.attrs``. Each test states the semantic it protects.
"""

from __future__ import annotations

import pytest

from annnet.core._Views import GraphView
from annnet.core.graph import AnnNet


# ── small graph fixtures ────────────────────────────────────────────────


def _toy_directed() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    G.add_edges('C', 'D', edge_id='e3', weight=3.0)
    return G


def _toy_with_slices() -> AnnNet:
    G = _toy_directed()
    G.slices.add('s1')
    G.slices.add('s2')
    G.add_edges('A', 'C', edge_id='e_s1', slice='s1', weight=10.0)
    G.add_edges('B', 'D', edge_id='e_s2', slice='s2', weight=20.0)
    return G


def _toy_with_hyperedge() -> AnnNet:
    G = AnnNet(directed=False)
    G.add_nodes(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e1')
    G.add_edges(['A', 'B', 'C'], edge_id='h1')  # undirected hyper
    return G


# ── predicates propagate ───────────────────────────────────────────────


def test_a_node_predicate_error_propagates_with_the_id() -> None:
    G = _toy_directed()

    def hostile(vid):
        raise AttributeError('boom')

    with pytest.raises(AttributeError, match="boom.*'A'"):
        _ = G.view(nodes=hostile).N.ids


def test_an_edge_predicate_error_propagates_with_the_id() -> None:
    G = _toy_directed()

    def hostile(eid):
        raise ValueError('nope')

    with pytest.raises(ValueError, match="nope.*'e1'"):
        _ = G.view(edges=hostile).E.ids


def test_a_node_filter_intersects_a_slice_filter() -> None:
    G = _toy_with_slices()
    v = G.view(slices='s1', nodes=['A', 'C'])
    assert v.N.ids == ('A', 'C')
    assert v.E.ids == ('e_s1',)


def test_an_edge_filter_intersects_a_slice_filter() -> None:
    G = _toy_with_slices()
    v = G.view(slices='s1', edges=['e_s1', 'e1'])
    assert v.E.ids == ('e_s1',), 'e1 is not a member of s1'


# ── hyperedges ────────────────────────────────────────────────────────────


def test_a_directed_hyperedge_needs_every_member() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    assert 'h1' not in G.view(nodes={'A', 'B', 'C'}, edges={'h1'}).E.ids
    assert G.view(nodes={'A', 'B', 'C', 'D'}, edges={'h1'}).E.ids == ('h1',)


def test_an_undirected_hyperedge_is_kept_when_its_members_are() -> None:
    G = _toy_with_hyperedge()
    v = G.view(nodes={'A', 'B', 'C'}, edges={'e1', 'h1'})
    assert set(v.E.ids) == {'e1', 'h1'}


# ── scoped tables ────────────────────────────────────────────────────────


def test_the_edge_table_of_a_view_holds_its_edges() -> None:
    G = _toy_directed()
    v = G.view(edges={'e1', 'e3'})
    df = v.attrs.table('edges', derived=True)
    assert {row['edge_id'] for row in df.to_dicts()} == {'e1', 'e3'}


def test_the_node_table_of_a_view_holds_its_nodes() -> None:
    G = _toy_directed()
    v = G.view(nodes={'A', 'B'})
    df = v.attrs.table('nodes', derived=True)
    assert {row['node_id'] for row in df.to_dicts()} == {'A', 'B'}


# ── materialize ──────────────────────────────────────────────────────────


def test_materialize_without_attributes_keeps_every_node() -> None:
    G = _toy_directed()
    G.attrs.update('nodes', {'A': {'color': 'red'}})
    sub = G.view().materialize(copy_attributes=False)
    assert set(sub.N) == {'A', 'B', 'C', 'D'}
    assert dict(sub.attrs.row('nodes', 'A')) == {}


def test_materialize_keeps_an_undirected_hyperedge() -> None:
    G = _toy_with_hyperedge()
    sub = G.view().materialize(copy_attributes=False)
    assert set(sub.E) == {'e1', 'h1'}
    assert sub.E.at('h1').kind == 'hyper'
    assert sub.E.at('h1').directed is False


def test_materialize_keeps_a_directed_hyperedge() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    sub = G.view().materialize(copy_attributes=False)
    assert list(sub.E) == ['h1']
    assert sub.E.at('h1').directed is True
    assert set(sub.E.at('h1').source) == {'A', 'B'}


def test_materialize_drops_an_edge_with_an_endpoint_outside_the_view() -> None:
    G = _toy_directed()
    sub = G.view(nodes={'A', 'B'}).materialize(copy_attributes=False)
    assert set(sub.N) == {'A', 'B'}
    assert list(sub.E) == ['e1']


# ── narrowing ──────────────────────────────────────────────────────────────


def test_narrowing_with_a_node_predicate() -> None:
    G = _toy_directed()
    v = G.view(nodes={'A', 'B', 'C', 'D'})
    s = v.view(nodes=lambda vid: vid in {'A', 'B'})
    assert s.N.ids == ('A', 'B')


def test_narrowing_with_an_edge_predicate_that_keeps_everything() -> None:
    G = _toy_directed()
    v = G.view(edges={'e1', 'e2', 'e3'})
    assert v.view(edges=lambda eid: True).E.ids == v.E.ids


def test_narrowing_with_explicit_edges_intersects() -> None:
    G = _toy_directed()
    v = G.view(edges={'e1', 'e2', 'e3'})
    assert v.view(edges={'e2'}).E.ids == ('e2',)


def test_narrowing_combines_predicates_with_and() -> None:
    G = _toy_directed()
    v = G.view(predicate=lambda vid: vid.startswith('A') or vid == 'B')
    s = v.view(predicate=lambda vid: vid != 'B')
    assert s.N.ids == ('A',)


def test_narrowing_keeps_the_parent_slices() -> None:
    G = _toy_with_slices()
    v = G.view(slices=['s1'])
    s = v.view()
    assert s.slices.list() == ['s1']
    assert s.E.ids == ('e_s1',)


# ── summary and repr ───────────────────────────────────────────────────────


def test_summary_of_a_full_view_names_no_filter() -> None:
    out = GraphView(_toy_directed()).summary()
    assert out['filters'] == []
    assert out['nodes'] == 4 and out['edges'] == 3
    assert 'filters: none' in repr(out)


def test_summary_names_every_active_filter() -> None:
    G = _toy_with_slices()
    v = G.view(nodes={'A', 'B'}, edges={'e1'}, slices=['s1'], predicate=lambda vid: True)
    out = v.summary()
    assert out['filters'] == ['nodes', 'edges', 'slices', 'predicate']
    assert out['boundary'] == 'closed'


def test_repr_and_len() -> None:
    v = GraphView(_toy_directed())
    assert repr(v).startswith('GraphView(nodes=4, edges=3')
    assert len(v) == len(v.N) == 4


# ── the derived edge table ───────────────────────────────────────────────────


def test_edge_table_with_a_slice_joins_that_slices_overrides() -> None:
    G = _toy_with_slices()
    G.attrs.update('edge_slices', {('s1', 'e_s1'): {'confidence': 0.9, 'weight': 99.0}})
    df = G.attrs.table('edges', derived=True, slice='s1')
    rows = {row['edge_id']: row for row in df.to_dicts()}
    assert rows['e_s1']['slice_confidence'] == 0.9
    assert rows['e_s1']['slice_weight'] == 99.0
    assert rows['e_s1']['effective_weight'] == 99.0
    assert rows['e1']['effective_weight'] == 1.0
    assert set(rows) == {'e1', 'e2', 'e3', 'e_s1', 'e_s2'}, 'slice= joins, it does not filter'


def test_edge_table_weight_columns_follow_their_flags() -> None:
    G = _toy_directed()
    both = G.attrs.table('edges', derived=True)
    assert {'weight', 'effective_weight'} <= set(both.columns)
    assert 'global_weight' not in both.columns
    resolved_only = G.attrs.table('edges', derived=True, include_weight=False, resolved_weight=True)
    assert 'weight' not in resolved_only.columns and 'effective_weight' in resolved_only.columns
    neither = G.attrs.table('edges', derived=True, include_weight=False, resolved_weight=False)
    assert 'weight' not in neither.columns and 'effective_weight' not in neither.columns


def test_edge_table_on_an_empty_graph_keeps_its_key_columns() -> None:
    G = AnnNet(directed=True)
    df = G.attrs.table('edges', derived=True)
    assert df.height == 0
    assert {'edge_id', 'kind', 'directed', 'source', 'target'} <= set(df.columns)


def test_unknown_query_arguments_raise() -> None:
    G = _toy_directed()
    with pytest.raises(TypeError, match='unknown edge table argument'):
        G.attrs.table('edges', derived=True, nope=1)
    with pytest.raises(TypeError, match='need derived=True'):
        G.attrs.table('edges', in_slice='s1')
    with pytest.raises(TypeError, match='layout'):
        G.attrs.table('nodes', derived=True, layout='incidences')
    with pytest.raises(KeyError, match='unknown slice'):
        G.attrs.table('edges', derived=True, in_slice='nope')


# ── the other derived tables ─────────────────────────────────────────────


def test_node_table_on_an_empty_graph_keeps_its_key_column() -> None:
    G = AnnNet(directed=False)
    assert 'node_id' in G.attrs.table('nodes', derived=True).columns


def test_slice_table_includes_user_slices_with_counts() -> None:
    G = _toy_with_slices()
    df = G.attrs.table('slices', derived=True)
    rows = {row['slice_id']: row for row in df.to_dicts()}
    assert {'s1', 's2'} <= set(rows)
    assert rows['s1']['n_edges'] == 1 and rows['s1']['n_nodes'] == 2


def test_aspect_table_on_a_flat_graph_is_empty_with_its_columns() -> None:
    df = _toy_directed().attrs.table('aspects', derived=True)
    assert 'aspect' in df.columns and df.height == 0


def test_aspect_table_emits_one_row_per_aspect() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_nodes(['A'], layer={'condition': 'healthy'})
    rows = G.attrs.table('aspects', derived=True).to_dicts()
    assert [row['aspect'] for row in rows] == ['condition']
    assert rows[0]['elementary_layers'] == ['healthy', 'treated']
    assert rows[0]['ordered'] is False


def test_layer_table_on_a_flat_graph_is_empty_with_its_columns() -> None:
    df = _toy_directed().attrs.table('layers', derived=True)
    assert df.height == 0
    assert 'layer' in df.columns and 'coordinate_id' in df.columns


def test_layer_table_emits_one_row_per_declared_coordinate() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_nodes(['A'], layer={'condition': 'healthy'})
    df = G.attrs.table('layers', derived=True)
    assert df.height == 2
    assert {'layer', 'coordinate_id', 'condition'} <= set(df.columns)
    assert 'layer_id' not in df.columns, 'layer_id is the elementary-layer display id'
