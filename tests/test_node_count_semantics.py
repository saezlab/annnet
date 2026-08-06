"""Node-count semantics: unique nodes vs supra-nodes.

``nv`` / ``ncount()`` / ``shape`` / ``nodes()`` count UNIQUE nodes
(deduplicated across layers). ``nv_supra`` / ``ncount(supra=True)`` /
``supra_shape`` / ``supra_nodes()`` count SUPRA-NODES (one per
``(node_id, layer_coord)`` row of the supra-incidence matrix).
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


def _build_multilayer(n_nodes: int = 2, n_layers: int = 3) -> AnnNet:
    G = AnnNet(directed=False)
    vids = [f'v{i}' for i in range(n_nodes)]
    layer_ids = [f'L{i}' for i in range(n_layers)]
    G.add_nodes(vids)
    G.layers.set_aspects(['cond'], {'cond': layer_ids})
    for lid in layer_ids:
        G.add_nodes(vids, layer=(lid,))
    return G


def test_nv_counts_unique_nodes_in_multilayer():
    G = _build_multilayer(n_nodes=2, n_layers=3)
    assert G.nv == 2
    assert G.ncount() == 2
    assert len(G.nodes()) == 2
    assert set(G.nodes()) == {'v0', 'v1'}


def test_nv_supra_counts_supra_nodes_in_multilayer():
    G = _build_multilayer(n_nodes=2, n_layers=3)
    # 2 nodes × 3 layers + the placeholder ('_',) carry-over from the
    # initial flat add_nodes call (still indexed as supra-nodes).
    assert G.nv_supra >= 2 * 3
    assert G.ncount(supra=True) == G.nv_supra
    assert len(G.supra_nodes()) == G.nv_supra


def test_shape_uses_unique_nodes_supra_shape_uses_supra_nodes():
    G = _build_multilayer(n_nodes=2, n_layers=3)
    G.add_edges([('v0', 'v1')])
    assert G.shape == (G.ncount(), G.ne)
    assert G.supra_shape == (G.nv_supra, G.ne)
    assert G.shape != G.supra_shape  # multilayer: V ≠ S


def test_nv_equals_nv_supra_in_flat_graph():
    G = AnnNet(directed=False)
    G.add_nodes(['A', 'B', 'C'])
    assert G.nv == G.nv_supra == 3
    assert G.shape == G.supra_shape == (3, 0)


def test_nodes_has_no_duplicates_in_multilayer():
    G = _build_multilayer(n_nodes=4, n_layers=5)
    vlist = G.nodes()
    assert len(vlist) == len(set(vlist)) == 4


def test_supra_nodes_returns_full_ekey_tuples():
    G = _build_multilayer(n_nodes=2, n_layers=2)
    sv = G.supra_nodes()
    # Every entry is a (node_id, layer_coord) pair.
    assert all(isinstance(item, tuple) and len(item) == 2 for item in sv)
    assert all(isinstance(item[0], str) and isinstance(item[1], tuple) for item in sv)
    # Nodes appearing on multiple layers appear multiple times.
    vid_counts = {}
    for vid, _coord in sv:
        vid_counts[vid] = vid_counts.get(vid, 0) + 1
    assert all(c >= 2 for c in vid_counts.values())
