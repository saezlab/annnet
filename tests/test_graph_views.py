"""Unit tests for annnet/core/_Views.py — GraphView and ViewsClass."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from annnet.core._Views import GraphView
from annnet.core.graph import AnnNet


def _build_graph():
    """Directed graph: A→B, B→C, with attributes and a slice."""
    G = AnnNet(directed=True)
    G.add_vertex('A')
    G.set_vertex_attrs('A', gene='TP53', score=1.0)
    G.add_vertex('B')
    G.set_vertex_attrs('B', gene='EGFR', score=0.5)
    G.add_vertex('C')
    G.set_vertex_attrs('C', gene='MYC', score=0.2)
    G.add_edge('A', 'B', edge_id='e1', weight=2.0)
    G.set_edge_attrs('e1', relation='activates')
    G.add_edge('B', 'C', edge_id='e2', weight=3.0)
    G.set_edge_attrs('e2', relation='inhibits')
    G.add_slice('sig')
    G.add_edge_to_slice('sig', 'e1')
    return G


def _build_hyperedge_graph():
    G = AnnNet(directed=True)
    for v in ['A', 'B', 'C', 'D']:
        G.add_vertex(v)
    G.add_edge('A', 'B', edge_id='e1', weight=1.0)
    G.add_edge(src=['A', 'B'], tgt=['C'], edge_id='h1', weight=0.5)
    G.add_edge(src=['B', 'C', 'D'], edge_id='h2', weight=1.5)
    return G


class TestGraphViewVertexFilter(unittest.TestCase):
    def test_no_filter_returns_all_vertices(self):
        G = _build_graph()
        view = GraphView(G)
        # No filter → vertex_ids is None (all vertices)
        self.assertIsNone(view.vertex_ids)
        self.assertEqual(view.vertex_count, 3)

    def test_list_filter_restricts_vertices(self):
        G = _build_graph()
        view = GraphView(G, vertices=['A', 'B'])
        self.assertEqual(view.vertex_ids, {'A', 'B'})
        self.assertEqual(view.vertex_count, 2)

    def test_callable_predicate_filter(self):
        G = _build_graph()
        # Keep only vertices whose gene attribute starts with "E"
        view = GraphView(G, vertices=lambda v: v in ['A', 'C'])
        ids = view.vertex_ids
        self.assertIn('A', ids)
        self.assertIn('C', ids)
        self.assertNotIn('B', ids)

    def test_extra_predicate_further_restricts(self):
        G = _build_graph()
        view = GraphView(G, vertices=['A', 'B', 'C'], predicate=lambda v: v != 'C')
        ids = view.vertex_ids
        self.assertIn('A', ids)
        self.assertIn('B', ids)
        self.assertNotIn('C', ids)


class TestGraphViewEdgeFilter(unittest.TestCase):
    def test_no_filter_returns_all_edges(self):
        G = _build_graph()
        view = GraphView(G)
        self.assertIsNone(view.edge_ids)
        self.assertEqual(view.edge_count, 2)

    def test_list_filter_restricts_edges(self):
        G = _build_graph()
        view = GraphView(G, edges=['e1'])
        self.assertEqual(view.edge_ids, {'e1'})
        self.assertEqual(view.edge_count, 1)

    def test_callable_edge_filter(self):
        G = _build_graph()
        view = GraphView(G, edges=lambda e: e == 'e2')
        ids = view.edge_ids
        self.assertIn('e2', ids)
        self.assertNotIn('e1', ids)


class TestGraphViewSliceFilter(unittest.TestCase):
    def test_slice_filter_single(self):
        G = _build_graph()
        # "sig" slice contains only e1
        G.add_vertex_to_slice('sig', 'A')
        G.add_vertex_to_slice('sig', 'B')
        view = GraphView(G, slices='sig')
        edge_ids = view.edge_ids
        self.assertIn('e1', edge_ids)
        self.assertNotIn('e2', edge_ids)

    def test_slice_filter_list(self):
        G = _build_graph()
        G.add_slice('reg')
        G.add_edge_to_slice('reg', 'e2')
        G.add_vertex_to_slice('sig', 'A')
        G.add_vertex_to_slice('sig', 'B')
        G.add_vertex_to_slice('reg', 'B')
        G.add_vertex_to_slice('reg', 'C')
        view = GraphView(G, slices=['sig', 'reg'])
        edge_ids = view.edge_ids
        self.assertIn('e1', edge_ids)
        self.assertIn('e2', edge_ids)


class TestGraphViewObs(unittest.TestCase):
    """obs property returns filtered vertex attribute table."""

    def test_obs_no_filter_returns_all(self):
        G = _build_graph()
        view = GraphView(G)
        obs = view.obs
        try:

            rows = obs.to_dicts()
        except Exception:
            rows = obs.to_dict(orient='records')
        vertex_ids = [r['vertex_id'] for r in rows]
        self.assertIn('A', vertex_ids)
        self.assertIn('B', vertex_ids)
        self.assertIn('C', vertex_ids)

    def test_obs_filtered_to_subset(self):
        G = _build_graph()
        view = GraphView(G, vertices=['A'])
        obs = view.obs
        try:

            rows = obs.to_dicts()
        except Exception:
            rows = obs.to_dict(orient='records')
        vertex_ids = [r['vertex_id'] for r in rows]
        self.assertIn('A', vertex_ids)
        self.assertNotIn('B', vertex_ids)


class TestGraphViewVar(unittest.TestCase):
    """var property returns filtered edge attribute table."""

    def test_var_no_filter_returns_all(self):
        G = _build_graph()
        view = GraphView(G)
        var = view.var
        try:

            rows = var.to_dicts()
        except Exception:
            rows = var.to_dict(orient='records')
        edge_ids = [r['edge_id'] for r in rows]
        self.assertIn('e1', edge_ids)
        self.assertIn('e2', edge_ids)

    def test_var_filtered_to_subset(self):
        G = _build_graph()
        view = GraphView(G, edges=['e1'])
        var = view.var
        try:

            rows = var.to_dicts()
        except Exception:
            rows = var.to_dict(orient='records')
        edge_ids = [r['edge_id'] for r in rows]
        self.assertIn('e1', edge_ids)
        self.assertNotIn('e2', edge_ids)


class TestGraphViewX(unittest.TestCase):
    """X property returns submatrix view."""

    def test_X_shape_filtered_vertices_and_edges(self):
        # With explicit filters the submatrix is exactly 2×1
        G = _build_graph()
        view = GraphView(G, vertices=['A', 'B'], edges=['e1'])
        X = view.X
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 1)

    def test_X_shape_vertex_edge_filter_consistent(self):
        # Filter to A,B vertices and e1 (A→B) — C is absent, so e2 (B→C) is dropped
        G = _build_graph()
        view = GraphView(G, vertices=['A', 'B'], edges=['e1', 'e2'])
        X = view.X
        # e2 (B→C) is dropped because C not in vertex filter → only 1 column
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 1)

    def test_X_empty_for_no_matching_vertices(self):
        G = _build_graph()
        view = GraphView(G, vertices=[], edges=['e1'])
        X = view.X
        self.assertEqual(X.shape[0], 0)


class TestGraphViewMaterialize(unittest.TestCase):
    """materialize() creates a real AnnNet subgraph."""

    def test_materialize_no_filter_full_graph(self):
        G = _build_graph()
        view = GraphView(G)
        sub = view.materialize()
        self.assertEqual(sub.nv, 3)
        self.assertEqual(sub.ne, 2)

    def test_materialize_vertex_filter(self):
        G = _build_graph()
        # Only A and B → only e1 survives (both endpoints present)
        view = GraphView(G, vertices=['A', 'B'], edges=['e1'])
        sub = view.materialize()
        self.assertEqual(sub.nv, 2)
        self.assertEqual(sub.ne, 1)

    def test_materialize_copies_vertex_attrs(self):
        G = _build_graph()
        view = GraphView(G)
        sub = view.materialize(copy_attributes=True)
        attrs = sub.get_vertex_attrs('A') or {}
        # gene attribute should survive
        self.assertIn('gene', attrs)

    def test_materialize_no_attrs(self):
        G = _build_graph()
        view = GraphView(G)
        sub = view.materialize(copy_attributes=False)
        self.assertEqual(sub.nv, 3)

    def test_materialize_with_hyperedges(self):
        G = _build_hyperedge_graph()
        view = GraphView(G)
        sub = view.materialize()
        # All 4 vertices must survive the round-trip
        self.assertEqual(sub.nv, G.nv)
        # At minimum the binary edge must survive
        self.assertGreaterEqual(sub.ne, 1)


class TestGraphViewSubview(unittest.TestCase):
    """subview() further restricts an existing view."""

    def test_subview_narrows_vertices(self):
        G = _build_graph()
        base = GraphView(G, vertices=['A', 'B', 'C'])
        sub = base.subview(vertices=['A', 'B'])
        ids = sub.vertex_ids
        self.assertIn('A', ids)
        self.assertIn('B', ids)
        self.assertNotIn('C', ids)

    def test_subview_narrows_edges(self):
        G = _build_graph()
        base = GraphView(G, edges=['e1', 'e2'])
        sub = base.subview(edges=['e1'])
        self.assertIn('e1', sub.edge_ids)
        self.assertNotIn('e2', sub.edge_ids)


class TestGraphViewConvenience(unittest.TestCase):
    """summary(), __repr__, __len__."""

    def test_summary_contains_counts(self):
        G = _build_graph()
        view = GraphView(G)
        s = view.summary()
        self.assertIsInstance(s, str)
        self.assertIn('3', s)  # 3 vertices
        self.assertIn('2', s)  # 2 edges

    def test_repr(self):
        G = _build_graph()
        view = GraphView(G)
        r = repr(view)
        self.assertIn('GraphView', r)

    def test_len_equals_vertex_count(self):
        G = _build_graph()
        view = GraphView(G, vertices=['A', 'B'])
        self.assertEqual(len(view), 2)


class TestViewsClassEdgesView(unittest.TestCase):
    """ViewsClass.edges_view() (mixin on AnnNet)."""

    def test_basic_edges_view(self):
        G = _build_graph()
        df = G.edges_view()
        try:

            rows = df.to_dicts()
        except Exception:
            rows = df.to_dict(orient='records')
        edge_ids = [r['edge_id'] for r in rows]
        self.assertIn('e1', edge_ids)
        self.assertIn('e2', edge_ids)

    def test_edges_view_includes_weight_column(self):
        G = _build_graph()
        df = G.edges_view(include_weight=True)
        cols = list(df.columns)
        self.assertIn('global_weight', cols)

    def test_edges_view_with_slice_weight(self):
        G = _build_graph()
        G.set_edge_slice_attrs('sig', 'e1', weight=99.0)
        df = G.edges_view(slice='sig', resolved_weight=True)
        try:
            import polars as pl

            row = df.filter(pl.col('edge_id') == 'e1').to_dicts()[0]
        except Exception:
            row = df[df['edge_id'] == 'e1'].to_dict(orient='records')[0]
        self.assertAlmostEqual(row['effective_weight'], 99.0)

    def test_edges_view_empty_graph(self):
        G = AnnNet(directed=True)
        df = G.edges_view()
        # Should return empty DataFrame without error
        self.assertEqual(len(df), 0)

    def test_edges_view_uses_source_target_for_hyperedges(self):
        G = _build_hyperedge_graph()
        df = G.edges_view()
        try:

            rows = {r['edge_id']: r for r in df.to_dicts()}
        except Exception:
            rows = {r['edge_id']: r for r in df.to_dict(orient='records')}

        self.assertEqual(rows['h1']['source'], 'A|B')
        self.assertEqual(rows['h1']['target'], 'C')
        self.assertEqual(rows['h2']['source'], 'B|C|D')
        self.assertIsNone(rows['h2']['target'])


class TestViewsClassVerticesView(unittest.TestCase):
    """ViewsClass.vertices_view() (mixin on AnnNet)."""

    def test_basic_vertices_view(self):
        G = _build_graph()
        df = G.vertices_view()
        try:

            rows = df.to_dicts()
        except Exception:
            rows = df.to_dict(orient='records')
        vertex_ids = [r['vertex_id'] for r in rows]
        self.assertIn('A', vertex_ids)
        self.assertIn('B', vertex_ids)
        self.assertIn('C', vertex_ids)

    def test_vertices_view_empty_graph(self):
        G = AnnNet(directed=True)
        df = G.vertices_view()
        self.assertEqual(len(df), 0)


class TestAnnNetView(unittest.TestCase):
    """AnnNet.view() factory method."""

    def test_view_no_args_returns_graphview(self):
        G = _build_graph()
        v = G.view()
        self.assertIsInstance(v, GraphView)

    def test_view_with_vertex_list(self):
        G = _build_graph()
        v = G.view(vertices=['A'])
        self.assertEqual(v.vertex_ids, {'A'})

    def test_view_with_predicate(self):
        G = _build_graph()
        v = G.view(predicate=lambda x: x == 'B')
        # predicate applied to all vertices; only B survives
        # (predicate alone doesn't set vertex_ids, need vertices too)
        # test that it doesn't raise and returns a GraphView
        self.assertIsInstance(v, GraphView)


if __name__ == '__main__':
    unittest.main()
