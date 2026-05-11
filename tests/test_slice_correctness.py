"""Slice correctness.

- Every enumeration helper must include the default slice.
- ``set_slice_edge_weight`` must work through the canonical namespace path.
- Active slice resolution must apply to weight reads.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


def _build_graph():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.slices.add_slice('treated')
    return G


# ── default slice visibility ──────────────────────────────────────────────


def test_list_slices_includes_default():
    G = _build_graph()
    names = G.slices.list_slices()
    assert 'default' in names
    assert 'treated' in names
    assert len(names) == G.slices.slice_count()


def test_get_slices_dict_includes_default():
    G = _build_graph()
    keys = set(G.slices.get_slices_dict().keys())
    assert 'default' in keys
    assert 'treated' in keys


def test_slice_statistics_includes_default():
    G = _build_graph()
    stats = G.slices.slice_statistics()
    assert 'default' in stats
    assert 'treated' in stats


def test_views_slices_populated():
    G = _build_graph()
    df = G.views.slices()
    assert df.shape[0] == G.slices.slice_count()


# ── legacy weight shim ────────────────────────────────────────────────────


def test_set_slice_edge_weight_does_not_raise():
    G = _build_graph()
    G.attrs.set_slice_edge_weight('treated', 'e1', 99.0)
    assert G.attrs.get_effective_edge_weight('e1', slice='treated') == 99.0


# ── active slice resolution on reads ──────────────────────────────────────


def test_get_effective_edge_weight_uses_active_slice():
    G = _build_graph()
    G.attrs.set_edge_slice_attrs('treated', 'e1', weight=99.0)
    G.slices.active = 'treated'
    assert G.attrs.get_effective_edge_weight('e1') == 99.0
    G.slices.active = 'default'
    assert G.attrs.get_effective_edge_weight('e1') == 1.0


def test_explicit_slice_arg_overrides_active():
    G = _build_graph()
    G.attrs.set_edge_slice_attrs('treated', 'e1', weight=99.0)
    G.slices.active = 'default'
    assert G.attrs.get_effective_edge_weight('e1', slice='treated') == 99.0


def test_edge_list_reflects_active_slice_weight():
    G = _build_graph()
    G.attrs.set_edge_slice_attrs('treated', 'e1', weight=99.0)
    G.slices.active = 'treated'
    edges = G.edge_list()
    assert any(eid == 'e1' and weight == 99.0 for _src, _tgt, eid, weight in edges)
