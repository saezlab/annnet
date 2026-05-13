"""Regressions for two bugs surfaced during SCV-5 coverage work.

1. ``hash(G.ops)`` raised ``AttributeError`` because
   ``OperationsAccessor.__hash__`` was looking up ``self.vertices()`` /
   ``self.ne`` / ``self.get_edge()`` etc. on itself instead of on the
   wrapped graph ``self._G``.

2. ``G.views.layers_view()`` always returned the empty placeholder for
   multilayer graphs because it checked ``getattr(self, '_all_layers',
   ())`` on the AnnNet root, but ``_all_layers`` actually lives on
   ``self.layers``.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


# ── bug 1: hash(G.ops) -------------------------------------------------


def _toy_graph() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges('B', 'C', edge_id='e2', directed=False)
    G.uns['study'] = 'demo'
    return G


def test_hash_ops_returns_int() -> None:
    G = _toy_graph()
    h = hash(G.ops)
    assert isinstance(h, int)


def test_hash_ops_is_deterministic_for_identical_state() -> None:
    G1 = _toy_graph()
    G2 = _toy_graph()
    assert hash(G1.ops) == hash(G2.ops)


def test_hash_ops_changes_when_a_vertex_is_added() -> None:
    G = _toy_graph()
    h1 = hash(G.ops)
    G.add_vertices(['Z'])
    h2 = hash(G.ops)
    assert h1 != h2


def test_hash_ops_changes_when_an_edge_is_added() -> None:
    G = _toy_graph()
    h1 = hash(G.ops)
    G.add_edges('A', 'C', edge_id='e3')
    h2 = hash(G.ops)
    assert h1 != h2


def test_hash_ops_unaffected_by_iteration_order() -> None:
    """The hash is built from sorted vertex / edge sets — building the
    same graph in a different insertion order must yield the same hash."""
    G1 = AnnNet(directed=True)
    G1.add_vertices(['A', 'B'])
    G1.add_edges('A', 'B', edge_id='e1')

    G2 = AnnNet(directed=True)
    G2.add_vertices(['B', 'A'])  # reversed order
    G2.add_edges('A', 'B', edge_id='e1')

    assert hash(G1.ops) == hash(G2.ops)


# ── bug 2: views.layers_view on multilayer graphs ----------------------


def test_views_layers_view_returns_real_rows_on_multilayer_graph() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    df = G.views.layers_view()
    # Two elementary layers should produce two rows.
    height = df.height if hasattr(df, 'height') else len(df)
    assert height == 2

    # The aspect column should be present alongside layer_tuple / layer_id.
    cols = list(df.columns) if hasattr(df, 'columns') else list(df.schema.keys())
    assert 'layer_tuple' in cols
    assert 'layer_id' in cols
    assert 'condition' in cols


def test_views_layers_view_empty_for_flat_graph() -> None:
    """Flat (single-aspect placeholder) graphs still get the empty shape."""
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    df = G.views.layers_view()
    height = df.height if hasattr(df, 'height') else len(df)
    assert height == 0
