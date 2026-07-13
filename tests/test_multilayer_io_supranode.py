"""IO regressions for multilayer (two-aspect) graphs whose edge endpoints and
hyperedge members are ``(vid, layer_coord)`` supra-node keys.

These paths were not previously exercised by the suite:
    - ``to_cx2`` crashed in ``reify`` / ``expand`` and silently dropped binary
      edges (supra-node endpoints were never mapped to visual node ids).
    - ``_cx2_to_cytoscapejs`` gave edges ids that collided with node ids, so
      Cytoscape.js dropped them (``show_cx2`` showed nodes but no edges).
    - ``to_parquet`` / ``from_parquet`` crashed building a String column from
      list-of-supra-node ``head`` / ``tail`` / ``members`` values.
"""

from __future__ import annotations

import warnings

import annnet


def _build_two_aspect_graph():
    G = annnet.AnnNet(directed=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.set_aspects(
            ['mechanism', 'complex'],
            {
                'mechanism': ['signaling', 'metabolic', 'regulatory'],
                'complex': ['member', 'monomer'],
            },
        )
    mem = ('signaling', 'member')
    mon = ('signaling', 'monomer')
    reg = ('regulatory', 'monomer')
    G.add_vertices(
        [{'vertex_id': 'prot:A', 'kind': 'protein'}, {'vertex_id': 'prot:B', 'kind': 'protein'}],
        layer=mem,
    )
    G.add_vertices([{'vertex_id': 'prot:C', 'kind': 'protein'}], layer=mon)
    G.add_vertices([{'vertex_id': 'gene:G', 'kind': 'gene'}], layer=reg)
    # binary edge with supra-node endpoints (crosses the complex aspect)
    G.add_edges(
        [
            {
                'source': ('prot:A', mem),
                'target': ('prot:C', mon),
                'weight': 1.0,
                'edge_kind': 'signaling',
            }
        ],
        default_edge_directed=True,
    )
    # undirected complex hyperedge over member supra-nodes
    G.add_edges(
        [
            {
                'edge_id': 'cpx:X',
                'edge_kind': 'complex',
                'weight': 1.0,
                'members': [('prot:A', mem), ('prot:B', mem)],
            }
        ]
    )
    return G


def _node_layer_count(X):
    return sum(1 for ek in X._entities if X._entities[ek].kind == 'vertex')


def _nodes_edges(cx2_data):
    nodes = next(a['nodes'] for a in cx2_data if isinstance(a, dict) and 'nodes' in a)
    edges = next(a['edges'] for a in cx2_data if isinstance(a, dict) and 'edges' in a)
    return nodes, edges


def test_to_cx2_hyperedge_modes_multilayer():
    G = _build_two_aspect_graph()
    for mode in ('skip', 'reify', 'expand'):
        cx = annnet.to_cx2(G, hyperedges=mode)
        nodes, edges = _nodes_edges(cx)
        assert nodes, f'{mode}: no nodes'
        # the binary signaling edge must survive every mode
        assert edges, f'{mode}: no edges (binary edge dropped)'
        node_ids = {n['id'] for n in nodes}
        for e in edges:
            assert e['s'] in node_ids and e['t'] in node_ids, f'{mode}: dangling edge endpoint'


def test_from_cx2_multilayer_roundtrip(tmp_path):
    # Previously from_cx2 rebuilt edges from the visual overlay and applied the
    # manifest edge-metadata before edges existed, so multilayer graphs did not
    # round-trip. Now it reconstructs from the manifest for every hyperedge mode.
    G = _build_two_aspect_graph()
    for mode in ('skip', 'reify', 'expand'):
        p = str(tmp_path / f'g_{mode}.cx2')
        annnet.to_cx2(G, path=p, hyperedges=mode)
        H = annnet.from_cx2(p)
        assert H.layers.list_aspects() == ('mechanism', 'complex'), mode
        assert H.global_count('vertices') == G.global_count('vertices'), mode
        assert H.global_count('edges') == G.global_count('edges'), mode
        # no spurious ('_', '_') placeholder node-layers reintroduced on restore
        assert _node_layer_count(H) == _node_layer_count(G), mode
        cpx = H.hyperedge_definitions['cpx:X']
        assert any(
            isinstance(m, tuple) and m[1] == ('signaling', 'member')
            for m in (cpx.get('members') or [])
        ), mode


def test_cx2_cytoscapejs_edge_ids_unique_and_present():
    from annnet.io.cx2 import _cx2_to_cytoscapejs

    G = _build_two_aspect_graph()
    cj = _cx2_to_cytoscapejs(annnet.to_cx2(G, hyperedges='reify'))
    node_ids = {n['data']['id'] for n in cj['nodes']}
    edge_ids = {e['data']['id'] for e in cj['edges']}
    assert cj['edges'], 'no edges in cytoscape.js elements'
    assert not (node_ids & edge_ids), 'edge id collides with node id (Cytoscape.js drops it)'
    for e in cj['edges']:
        assert e['data']['source'] in node_ids and e['data']['target'] in node_ids


def test_parquet_multilayer_supranode_roundtrip(tmp_path):
    G = _build_two_aspect_graph()
    p = str(tmp_path / 'g')
    annnet.to_parquet(G, p)
    H = annnet.from_parquet(p)

    assert H.layers.list_aspects() == ('mechanism', 'complex')
    assert H.global_count('vertices') == G.global_count('vertices')
    assert H.global_count('edges') == G.global_count('edges')
    assert _node_layer_count(H) == _node_layer_count(G)

    # the complex hyperedge and its supra-node members survive
    cpx = H.hyperedge_definitions['cpx:X']
    members = list(cpx.get('members') or [])
    assert any(
        isinstance(m, tuple) and m[0] in ('prot:A', 'prot:B') and m[1] == ('signaling', 'member')
        for m in members
    ), f'supra-node members not restored: {members}'


def test_multilayer_restore_preserves_legitimate_placeholder(tmp_path):
    # The placeholder cleanup must only remove spurious ('_', '_') memberships.
    # A vertex genuinely stored at the placeholder (here, connected by an edge)
    # is listed in the manifest VM and is non-orphan, so it must survive.
    G = _build_two_aspect_graph()
    G.add_vertices([{'vertex_id': 'boundary', 'kind': 'protein'}])  # -> placeholder
    G.add_edges(
        [{'source': 'boundary', 'target': ('prot:C', ('signaling', 'monomer')), 'weight': 1.0}],
        default_edge_directed=True,
    )
    ph = ('_', '_')
    assert ('boundary', ph) in G._entities

    p = str(tmp_path / 'g')
    annnet.to_parquet(G, p)
    H = annnet.from_parquet(p)
    assert ('boundary', ph) in H._entities, 'legitimate placeholder vertex was dropped'
    assert _node_layer_count(H) == _node_layer_count(G)
    assert H.global_count('edges') == G.global_count('edges')


def test_sif_multilayer_supranode_roundtrip(tmp_path):
    # Previously from_sif rebuilt structure from the SIF text, which stored
    # ``str((vid, layer_coord))`` reprs for supra-node endpoints -> one spurious
    # vertex per repr string, hyperedges lost, aspects lost. Now multilayer graphs
    # are reconstructed authoritatively from the manifest.
    G = _build_two_aspect_graph()
    p = str(tmp_path / 'g.sif')
    mp = str(tmp_path / 'g.manifest.json')
    annnet.to_sif(G, p, lossless=True, manifest_path=mp)
    H = annnet.from_sif(p, manifest=mp)

    assert H.layers.list_aspects() == ('mechanism', 'complex')
    assert H.global_count('vertices') == G.global_count('vertices')
    assert H.global_count('edges') == G.global_count('edges')
    assert _node_layer_count(H) == _node_layer_count(G)

    cpx = H.hyperedge_definitions['cpx:X']
    members = list(cpx.get('members') or [])
    assert any(
        isinstance(m, tuple) and m[0] in ('prot:A', 'prot:B') and m[1] == ('signaling', 'member')
        for m in members
    ), f'supra-node members not restored: {members}'


def test_graphml_gexf_multilayer_no_crash_and_preserve(tmp_path):
    # Previously crashed in nx.write_graphml/gexf ("keys must be str, not tuple")
    # and again json-dumping the manifest. Now exports and re-imports, preserving
    # the aspects, the complex hyperedge, and its supra-node members.
    G = _build_two_aspect_graph()
    for to, fr, ext in (
        (annnet.to_graphml, annnet.from_graphml, 'graphml'),
        (annnet.to_gexf, annnet.from_gexf, 'gexf'),
    ):
        p = str(tmp_path / f'g.{ext}')
        to(G, p)
        H = fr(p)
        assert H.layers.list_aspects() == ('mechanism', 'complex')
        assert H.global_count('edges') == G.global_count('edges')
        assert _node_layer_count(H) == _node_layer_count(G), ext
        cpx = H.hyperedge_definitions['cpx:X']
        members = list(cpx.get('members') or [])
        assert any(isinstance(m, tuple) and m[1] == ('signaling', 'member') for m in members)


def test_dataframes_multilayer_roundtrip():
    # Previously crashed building a String column from raw supra-node tuples, and
    # the format carried no aspect manifest so aspects could not be rebuilt.
    G = _build_two_aspect_graph()
    H = annnet.from_dataframes(annnet.to_dataframes(G))
    assert H.layers.list_aspects() == ('mechanism', 'complex')
    assert H.global_count('vertices') == G.global_count('vertices')
    assert H.global_count('edges') == G.global_count('edges')
    assert _node_layer_count(H) == _node_layer_count(G)
    cpx = H.hyperedge_definitions['cpx:X']
    assert any(
        isinstance(m, tuple) and m[1] == ('signaling', 'member') for m in (cpx.get('members') or [])
    )


def test_to_dataframes_bare_vid_roundtrip_unchanged():
    # Non-multilayer graphs must keep readable bare-vid endpoints and round-trip.
    G = annnet.AnnNet(directed=True)
    G.add_vertices([{'vertex_id': v} for v in 'ABCD'])
    G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1', 'weight': 2.0}])
    G.add_edges([{'edge_id': 'h1', 'members': ['B', 'C', 'D'], 'weight': 1.0}])
    H = annnet.from_dataframes(annnet.to_dataframes(G))
    assert H.global_count('vertices') == 4
    assert H.global_count('edges') == 2
    assert len(H.hyperedge_definitions) == 1
