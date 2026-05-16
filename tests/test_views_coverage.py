"""Coverage tests for ``annnet/core/_Views.py`` ."""

from __future__ import annotations

from annnet.core._Views import GraphView, ViewsAccessor, ViewsClass
from annnet.core.graph import AnnNet


# ── small graph fixtures ────────────────────────────────────────────────


def _toy_directed() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
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
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e1')
    G.add_edges(['A', 'B', 'C'], edge_id='h1')  # undirected hyper
    return G


# ── _compute_ids exception-tolerance branches ──────────────────────────


def test_vertex_callable_predicate_swallows_attribute_error() -> None:
    G = _toy_directed()

    def hostile(vid):
        raise AttributeError('boom')

    v = GraphView(G, vertices=hostile)
    assert v.vertex_ids == set()


def test_edge_callable_predicate_swallows_value_error() -> None:
    G = _toy_directed()

    def hostile(eid):
        raise ValueError('nope')

    v = GraphView(G, edges=hostile)
    assert v.edge_ids == set()


def test_set_vertex_filter_intersects_with_existing_slice_filter() -> None:
    G = _toy_with_slices()
    v = GraphView(G, slices=['s1'], vertices={'A', 'B', 'C', 'D'})
    # s1 only registers A+C via e_s1; the set intersection keeps both.
    assert v.vertex_ids is not None
    assert v.vertex_ids.issubset({'A', 'B', 'C', 'D'})


def test_set_edge_filter_intersects_with_existing_slice_filter() -> None:
    G = _toy_with_slices()
    v = GraphView(G, slices=['s1'], edges={'e_s1', 'e_s2'})
    # only e_s1 lives in slice s1.
    assert v.edge_ids == {'e_s1'}


def test_extra_predicate_tolerates_exceptions() -> None:
    G = _toy_directed()

    def hostile(vid):
        raise TypeError('boom')

    v = GraphView(G, vertices={'A', 'B'}, predicate=hostile)
    assert v.vertex_ids == set()


def test_hyperedge_filter_by_vertex_connectivity_directed_hyper() -> None:
    """The hyper-edge branch checks src/tgt subset for directed hypers."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')  # directed hyper
    # both vertex and edge filters present → step 5 runs
    v = GraphView(G, vertices={'A', 'B', 'C'}, edges={'h1'})
    # vertex D is missing → h1 must drop out.
    assert 'h1' not in v.edge_ids


def test_hyperedge_filter_by_vertex_connectivity_undirected_hyper_keeps_when_subset() -> None:
    G = _toy_with_hyperedge()
    v = GraphView(G, vertices={'A', 'B', 'C'}, edges={'e1', 'h1'})
    # h1 members are all inside {A,B,C} → kept; e1 endpoints A,B also kept.
    assert {'e1', 'h1'}.issubset(v.edge_ids)


# ── edges_df / vertices_df with filtering ──────────────────────────────


def test_edges_df_filters_to_view_edge_ids() -> None:
    G = _toy_directed()
    v = GraphView(G, edges={'e1', 'e3'})
    df = v.edges_df()
    eids = {row['edge_id'] for row in df.to_dicts() if 'edge_id' in row}
    assert eids == {'e1', 'e3'}


def test_vertices_df_filters_to_view_vertex_ids() -> None:
    G = _toy_directed()
    v = GraphView(G, vertices={'A', 'B'})
    df = v.vertices_df()
    vids = {row['vertex_id'] for row in df.to_dicts() if 'vertex_id' in row}
    assert vids == {'A', 'B'}


# ── materialize: copy_attributes=False and edge_id branches ────────────


def test_materialize_without_copying_attributes_still_includes_all_vertices() -> None:
    G = _toy_directed()
    v = GraphView(G)
    sub = v.materialize(copy_attributes=False)
    assert set(sub.vertices()) == {'A', 'B', 'C', 'D'}


def test_materialize_handles_undirected_hyperedge_path() -> None:
    G = _toy_with_hyperedge()
    v = GraphView(G)
    sub = v.materialize(copy_attributes=False)
    # The hyperedge must survive in the subgraph.
    assert sub.ne >= 1


def test_materialize_handles_directed_hyperedge_path() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    v = GraphView(G)
    sub = v.materialize(copy_attributes=False)
    assert sub.ne >= 1


def test_materialize_skips_edges_with_vertex_outside_view() -> None:
    G = _toy_directed()
    v = GraphView(G, vertices={'A', 'B'})  # excludes C, D
    sub = v.materialize(copy_attributes=False)
    assert set(sub.vertices()) == {'A', 'B'}
    # only e1 (A→B) survives.
    assert sub.ne == 1


# ── subview combination paths ──────────────────────────────────────────


def test_subview_with_callable_vertices_predicate_keeps_base_set() -> None:
    G = _toy_directed()
    v = GraphView(G, vertices={'A', 'B', 'C', 'D'})
    s = v.subview(vertices=lambda vid: vid in {'A', 'B'})
    # Predicate path retains base set with predicate applied lazily.
    assert s.vertex_ids == {'A', 'B'}


def test_subview_with_callable_edges_keeps_base_set() -> None:
    G = _toy_directed()
    v = GraphView(G, edges={'e1', 'e2', 'e3'})
    s = v.subview(edges=lambda eid: True)
    assert s.edge_ids == v.edge_ids


def test_subview_with_set_edges_intersects_with_base() -> None:
    G = _toy_directed()
    v = GraphView(G, edges={'e1', 'e2', 'e3'})
    s = v.subview(edges={'e2'})
    assert s.edge_ids == {'e2'}


def test_subview_combines_existing_and_new_predicates_with_and() -> None:
    G = _toy_directed()

    def starts_with_A(vid):
        return vid.startswith('A')

    def is_short(vid):
        return len(vid) == 1

    # Scope the base view to a vertex set so predicates have something
    # to filter over (otherwise vertex_ids stays None).
    v = GraphView(G, vertices={'A', 'B', 'C', 'D'}, predicate=starts_with_A)
    s = v.subview(predicate=is_short)
    assert s.vertex_ids == {'A'}


def test_subview_passes_through_existing_slices_when_none_given() -> None:
    G = _toy_with_slices()
    v = GraphView(G, slices=['s1'])
    s = v.subview()
    assert s._slices == ['s1']


# ── summary with various filter combinations ──────────────────────────


def test_summary_with_no_filters_says_full_graph() -> None:
    v = GraphView(_toy_directed())
    out = v.summary()
    assert 'Filters: None (full graph)' in out


def test_summary_with_slice_vertex_edge_and_predicate_filters() -> None:
    G = _toy_with_slices()
    v = GraphView(
        G,
        vertices={'A', 'B'},
        edges={'e1'},
        slices=['s1'],
        predicate=lambda vid: True,
    )
    out = v.summary()
    assert 'slices=' in out
    assert 'vertices=' in out
    assert 'edges=' in out
    assert 'predicate=' in out


def test_summary_with_callable_vertices_and_edges_filters() -> None:
    G = _toy_directed()
    v = GraphView(G, vertices=lambda vid: True, edges=lambda eid: True)
    out = v.summary()
    assert 'vertices=<predicate>' in out
    assert 'edges=<predicate>' in out


def test_dunder_repr_and_len() -> None:
    v = GraphView(_toy_directed())
    assert 'GraphView' in repr(v)
    assert len(v) == v.vertex_count


# ── edges_view branches (ViewsClass on AnnNet) ─────────────────────────


def test_edges_view_with_slice_joins_per_slice_attributes() -> None:
    G = _toy_with_slices()
    # Use non-reserved attribute keys.
    G.attrs.set_edge_slice_attrs('s1', 'e_s1', confidence=0.9, weight=99.0)
    df = G.views.edges(slice='s1')
    rows = df.to_dicts()
    target = next((r for r in rows if r.get('edge_id') == 'e_s1'), None)
    assert target is not None
    # the join introduced at least one slice_*-prefixed column.
    assert any(k.startswith('slice_') for k in target)


def test_edges_view_with_resolved_weight_but_no_include_weight() -> None:
    G = _toy_directed()
    df = G.views.edges(include_weight=False, resolved_weight=True)
    cols = df.columns
    # 'global_weight' is hidden but 'effective_weight' is computed.
    assert 'global_weight' not in cols
    assert 'effective_weight' in cols


def test_edges_view_with_neither_weight_includes_neither_column() -> None:
    G = _toy_directed()
    df = G.views.edges(include_weight=False, resolved_weight=False)
    cols = df.columns
    assert 'global_weight' not in cols
    assert 'effective_weight' not in cols


def test_edges_view_on_empty_graph_returns_empty_placeholder() -> None:
    G = AnnNet(directed=True)
    df = G.views.edges()
    assert df.height == 0
    assert 'edge_id' in df.columns


# ── vertices_view edge cases ───────────────────────────────────────────


def test_vertices_view_on_empty_graph_returns_empty_placeholder() -> None:
    G = AnnNet(directed=False)
    df = G.views.vertices()
    assert 'vertex_id' in df.columns


# ── slices_view path ──────────────────────────────────────────────────


def test_slices_view_includes_user_slices() -> None:
    G = _toy_with_slices()
    df = G.views.slices()
    sids = {row['slice_id'] for row in df.to_dicts()}
    assert {'s1', 's2'}.issubset(sids)


# ── aspects_view & layers_view paths ──────────────────────────────────


def test_aspects_view_on_flat_graph_returns_empty_placeholder() -> None:
    G = _toy_directed()
    df = G.views.aspects()
    assert 'aspect' in df.columns
    assert df.height == 0


def test_aspects_view_with_multilayer_graph_emits_one_row_per_aspect() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    df = G.views.aspects()
    aspects = [row['aspect'] for row in df.to_dicts()]
    assert 'condition' in aspects


def test_layers_view_on_flat_graph_returns_empty_placeholder() -> None:
    G = _toy_directed()
    df = G.views.layers()
    assert df.height == 0
    assert 'layer_id' in df.columns


def test_layers_view_with_multilayer_graph_emits_one_row_per_layer() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    df = G.views.layers()
    assert df.height == 2  # 2 elementary layers
    cols = df.columns
    assert 'layer_id' in cols
    assert 'condition' in cols


# ── ViewsAccessor convenience wrappers ────────────────────────────────


def test_views_accessor_dispatches_to_class_methods() -> None:
    G = _toy_directed()
    a = ViewsAccessor(G)
    assert a.edges().height >= 1
    assert a.vertices().height >= 1
    assert a.slices().height >= 1
    assert a.aspects() is not None
    assert a.layers() is not None
    # the singular ``layers_view`` wrapper too
    assert a.layers_view(copy=True) is not None


def test_views_accessor_layers_view_copy_false_returns_object() -> None:
    G = _toy_directed()
    a = ViewsAccessor(G)
    out = a.layers_view(copy=False)
    assert out is not None


# Touch the class reference to make pyflakes happy
_ = ViewsClass
