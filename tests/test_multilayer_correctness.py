"""Multilayer correctness.

Setup: 3 vertices in layer (healthy,), one intra-layer edge between them,
plus a coupling edge to (treated,). The basic layer query helpers must
return non-empty results that match the underlying state.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


def _build_graph():
    G = AnnNet(directed=True, aspects={'condition': ['healthy', 'treated']})
    G.add_vertices(['A', 'B', 'C'], layer=('healthy',))
    G.add_vertices(['A', 'B', 'C'], layer=('treated',))
    G.add_edges(('A', ('healthy',)), ('B', ('healthy',)), edge_id='intra1')
    G.add_edges(('A', ('healthy',)), ('A', ('treated',)), edge_id='couple1')
    return G


def test_iter_layers_yields_populated_layers():
    G = _build_graph()
    layers = list(G.layers.iter_layers())
    assert ('healthy',) in layers
    assert ('treated',) in layers


def test_layer_edge_set_returns_intra_edge():
    G = _build_graph()
    healthy_edges = G.layers.layer_edge_set(('healthy',))
    assert 'intra1' in healthy_edges


def test_layer_edge_set_excludes_coupling_by_default():
    G = _build_graph()
    healthy_edges = G.layers.layer_edge_set(('healthy',))
    # Coupling edges aren't included unless include_coupling=True.
    assert 'couple1' not in healthy_edges


def test_layer_edge_set_includes_coupling_when_requested():
    G = _build_graph()
    healthy_edges = G.layers.layer_edge_set(('healthy',), include_coupling=True)
    assert 'couple1' in healthy_edges


def test_subgraph_from_layer_tuple_preserves_intra_layer_edge():
    G = _build_graph()
    sg = G.layers.subgraph_from_layer_tuple(('healthy',))
    # Three vertices were placed in (healthy,); the intra edge connects A-B.
    assert sg.nv >= 2
    assert sg.ne == 1
    assert any(eid == 'intra1' for _src, _tgt, eid, _w in sg.edge_list())
