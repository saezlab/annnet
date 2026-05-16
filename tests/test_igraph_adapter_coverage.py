"""Coverage tests for ``annnet/adapters/igraph_adapter.py``."""

from __future__ import annotations

import pytest

igraph = pytest.importorskip('igraph')

from annnet.adapters.igraph_adapter import (
    _coeff_from_obj,
    _export_binary_graph,
    _from_ig_without_manifest,
    _ig_collect_reified,
    from_igraph,
    to_igraph,
)
from annnet.core.graph import AnnNet


# ── fixtures ────────────────────────────────────────────────────────────


def _toy_directed() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.5)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    # NB: do not set a vertex attr called ``name`` — it collides with the
    # igraph native vertex-ID column of the same name (latent bug in
    # ``_export_binary_graph``; surfaced during Wave 8 but out of scope to
    # fix here — would require a rename or skip).
    G.attrs.set_vertex_attrs('A', label='alpha')
    G.attrs.set_edge_attrs('e1', label='alpha')
    return G


def _toy_with_undirected_hyper() -> AnnNet:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    return G


def _toy_with_directed_hyper() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    return G


# ── _coeff_from_obj ────────────────────────────────────────────────────


def test_coeff_from_obj_numeric() -> None:
    assert _coeff_from_obj(2.5) == 2.5
    assert _coeff_from_obj(3) == 3.0


def test_coeff_from_obj_dict_with_value() -> None:
    assert _coeff_from_obj({'__value': 5.0}) == 5.0
    # nested __value
    assert _coeff_from_obj({'__value': {'__value': 7.0}}) == 7.0


def test_coeff_from_obj_dict_without_value_defaults_to_one() -> None:
    assert _coeff_from_obj({'foo': 'bar'}) == 1.0


def test_coeff_from_obj_dict_with_non_coercible_value() -> None:
    assert _coeff_from_obj({'__value': 'not-a-number'}) == 1.0


def test_coeff_from_obj_other_types() -> None:
    assert _coeff_from_obj('string') == 1.0
    assert _coeff_from_obj(None) == 1.0


# ── _export_binary_graph (low-level) ──────────────────────────────────


def test_export_binary_graph_directed_round_trip() -> None:
    G = _toy_directed()
    out = _export_binary_graph(G, directed=True, skip_hyperedges=True)
    assert out.vcount() == 3
    assert out.ecount() == 2


def test_export_binary_graph_undirected_emits_each_edge_once() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    out = _export_binary_graph(G, directed=False, skip_hyperedges=True)
    assert out.ecount() == 1


def test_export_binary_graph_undirected_into_directed_emits_two_edges() -> None:
    """An undirected AnnNet edge → two directed igraph edges."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    out = _export_binary_graph(G, directed=True, skip_hyperedges=True)
    # one undirected source edge → two directed copies
    assert out.ecount() == 2


def test_export_binary_graph_expands_undirected_hyperedge_to_clique() -> None:
    G = _toy_with_undirected_hyper()
    out = _export_binary_graph(G, directed=False, skip_hyperedges=False)
    # 3-member clique → 3 edges
    assert out.ecount() == 3


def test_export_binary_graph_expands_directed_hyperedge_to_cartesian() -> None:
    G = _toy_with_directed_hyper()
    out = _export_binary_graph(G, directed=True, skip_hyperedges=False)
    # 2 tail × 2 head → 4 edges
    assert out.ecount() == 4


def test_export_binary_graph_public_only_strips_double_underscore_attrs() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.attrs.set_vertex_attrs('A', __secret='nope', name='alpha')
    out = _export_binary_graph(G, directed=True, skip_hyperedges=True, public_only=True)
    # __secret was stripped; name preserved
    assert '__secret' not in out.vs.attributes()


# ── to_igraph hyperedge modes ──────────────────────────────────────────


def test_to_igraph_skip_drops_hyperedges_from_igG() -> None:
    G = _toy_with_undirected_hyper()
    igG, manifest = to_igraph(G, hyperedge_mode='skip')
    assert igG.ecount() == 0
    # but the manifest still has the hyperedge.
    assert 'h1' in manifest['edges']


def test_to_igraph_expand_emits_clique_edges() -> None:
    G = _toy_with_undirected_hyper()
    igG, _ = to_igraph(G, directed=False, hyperedge_mode='expand')
    # K3 → 3 undirected pairs
    assert igG.ecount() == 3


def test_to_igraph_reify_adds_hyperedge_node_and_membership_edges() -> None:
    G = _toy_with_undirected_hyper()
    igG, _ = to_igraph(G, directed=False, hyperedge_mode='reify', reify_prefix='he::')
    # 3 original + 1 reified HE node
    assert igG.vcount() == 4
    # In an undirected container, one edge per member is enough.
    assert igG.ecount() == 3
    assert any(bool(flag) for flag in igG.vs['is_hyperedge'])


def test_to_igraph_reify_with_directed_hyperedge_attaches_head_tail_roles() -> None:
    G = _toy_with_directed_hyper()
    igG, _ = to_igraph(G, hyperedge_mode='reify')
    roles = set(igG.es['role'])
    assert {'head', 'tail'}.issubset(roles)


def test_to_igraph_with_public_only_strips_underscore_attrs() -> None:
    G = _toy_directed()
    G.attrs.set_vertex_attrs('A', __secret='nope')
    G.attrs.set_edge_attrs('e1', __secret='nope')
    igG, manifest = to_igraph(G, public_only=True)
    # The manifest's vertex_attrs / edge_attrs sections strip __ keys.
    for v_attrs in manifest['vertex_attrs'].values():
        assert all(not str(k).startswith('__') for k in v_attrs)


def test_to_igraph_with_slice_filter_restricts_manifest_to_those_slices() -> None:
    G = _toy_directed()
    G.slices.add('s1')
    G.slices.add_edges('s1', ['e1'])
    _, manifest = to_igraph(G, slice='s1')
    # The slices section is keyed by slice id.
    assert 's1' in manifest['slices']


# ── full round-trip via manifest ──────────────────────────────────────


def test_to_igraph_from_igraph_round_trip_basic() -> None:
    G = _toy_directed()
    igG, manifest = to_igraph(G)
    H = from_igraph(igG, manifest)
    assert set(H.vertices()) == {'A', 'B', 'C'}
    assert H.ne == 2


def test_to_igraph_from_igraph_round_trip_with_undirected_hyperedge() -> None:
    G = _toy_with_undirected_hyper()
    igG, manifest = to_igraph(G, hyperedge_mode='reify')
    H = from_igraph(igG, manifest)
    assert H.ne >= 1  # hyperedge survives via manifest


def test_to_igraph_from_igraph_round_trip_with_directed_hyperedge() -> None:
    G = _toy_with_directed_hyper()
    igG, manifest = to_igraph(G, hyperedge_mode='reify')
    H = from_igraph(igG, manifest)
    assert H.ne >= 1


def test_to_igraph_from_igraph_round_trip_with_slices() -> None:
    G = _toy_directed()
    G.slices.add('s1')
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=99.0)
    igG, manifest = to_igraph(G)
    H = from_igraph(igG, manifest)
    assert 's1' in H.slices.list()


def test_to_igraph_from_igraph_round_trip_with_multilayer() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    G.add_vertices(['B'], layer={'condition': 'treated'})
    igG, manifest = to_igraph(G)
    H = from_igraph(igG, manifest)
    assert H.is_multilayer


# ── _ig_collect_reified ──────────────────────────────────────────────


def test_ig_collect_reified_returns_empty_for_unflagged_graph() -> None:
    igG = igraph.Graph(directed=False)
    igG.add_vertices(2)
    igG.add_edges([(0, 1)])
    defs, mem = _ig_collect_reified(igG)
    assert defs == []
    assert mem == set()


def test_ig_collect_reified_finds_undirected_membership() -> None:
    igG = igraph.Graph(directed=False)
    igG.add_vertices(4)
    igG.vs['name'] = ['A', 'B', 'C', 'HE']
    igG.vs['is_hyperedge'] = [False, False, False, True]
    igG.vs['eid'] = ['', '', '', 'h1']
    igG.add_edges([(0, 3), (1, 3), (2, 3)])
    defs, mem = _ig_collect_reified(igG)
    assert len(defs) == 1
    eid, directed, head, tail, *_ = defs[0]
    assert eid == 'h1'
    assert directed is False
    assert set(head) == {'A', 'B', 'C'}
    assert mem == {0, 1, 2}


def test_ig_collect_reified_finds_directed_head_tail() -> None:
    igG = igraph.Graph(directed=True)
    igG.add_vertices(5)
    igG.vs['name'] = ['A', 'B', 'C', 'D', 'HE']
    igG.vs['is_hyperedge'] = [False, False, False, False, True]
    igG.vs['eid'] = ['', '', '', '', 'h1']
    igG.add_edges([(0, 4), (1, 4), (4, 2), (4, 3)])
    igG.es['role'] = ['tail', 'tail', 'head', 'head']
    defs, _ = _ig_collect_reified(igG)
    assert len(defs) == 1
    _, directed, head, tail, *_ = defs[0]
    assert directed is True
    assert set(head) == {'C', 'D'}
    assert set(tail) == {'A', 'B'}


# ── from_igraph reified mode (rebuilds HE absent from manifest) ──────


def test_from_igraph_reified_mode_recovers_hyperedge_not_in_manifest() -> None:
    """Build an igraph with a reified hyperedge but pass an empty manifest."""
    igG = igraph.Graph(directed=False)
    igG.add_vertices(4)
    igG.vs['name'] = ['A', 'B', 'C', 'HE']
    igG.vs['is_hyperedge'] = [False, False, False, True]
    igG.vs['eid'] = ['', '', '', 'h1']
    igG.add_edges([(0, 3), (1, 3), (2, 3)])
    igG.es['role'] = ['member', 'member', 'member']

    empty_manifest = {'edges': {}, 'weights': {}, 'edge_directed': {}}
    H = from_igraph(igG, empty_manifest, hyperedge='reified')
    # h1 hyperedge gets rebuilt from the reified pattern.
    assert H.ne >= 1


# ── _from_ig_without_manifest ─────────────────────────────────────────


def test_from_ig_without_manifest_basic_round_trip() -> None:
    G = _toy_directed()
    igG, _ = to_igraph(G)
    H = _from_ig_without_manifest(igG)
    assert set(H.vertices()) == {'A', 'B', 'C'}
    assert H.ne >= 2


def test_from_ig_without_manifest_reified_mode_recovers_hyperedge() -> None:
    """Build the reified pattern manually (avoiding to_igraph's reify which
    copies the reserved ``directed`` attr onto the HE node and tripping the
    AttributesClass reserved-key barrier on round-trip)."""
    igG = igraph.Graph(directed=False)
    igG.add_vertices(4)
    igG.vs['name'] = ['A', 'B', 'C', 'HE']
    igG.vs['is_hyperedge'] = [False, False, False, True]
    igG.vs['eid'] = ['', '', '', 'h1']
    igG.add_edges([(0, 3), (1, 3), (2, 3)])
    igG.es['role'] = ['member', 'member', 'member']
    H = _from_ig_without_manifest(igG, hyperedge='reified')
    assert H.ne >= 1
