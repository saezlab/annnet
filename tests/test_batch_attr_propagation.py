"""Batch ``add_*`` paths must not drop user attributes.

The single-vertex / single-edge forms accept and persist arbitrary user
attributes. The batch list/dict forms must behave the same: any keys that
are not structural must land in the corresponding attribute table.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


def _attr_value(graph: AnnNet, edge_id: str, key: str):
    return graph.attrs.get_edge_attrs(edge_id).get(key)


def _vertex_attr_value(graph: AnnNet, vertex_id: str, key: str):
    return graph.attrs.get_vertex_attrs(vertex_id).get(key)


# ── add_edges: batch dict form ─────────────────────────────────────────────


def test_add_edges_batch_dict_preserves_user_attrs():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(
        [
            {'source': 'A', 'target': 'B', 'edge_id': 'e1', 'confidence': 0.9},
            {'source': 'B', 'target': 'C', 'edge_id': 'e2', 'confidence': 0.7},
        ]
    )
    assert _attr_value(G, 'e1', 'confidence') == 0.9
    assert _attr_value(G, 'e2', 'confidence') == 0.7


def test_add_edges_batch_dict_with_attributes_subdict():
    """Existing 'attributes'/'attrs' subdict form must keep working."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1', 'attrs': {'confidence': 0.42}}])
    assert _attr_value(G, 'e1', 'confidence') == 0.42


def test_add_edges_batch_dict_flat_overrides_subdict():
    """Flat keys take precedence over equivalent keys nested in attrs."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        [
            {
                'source': 'A',
                'target': 'B',
                'edge_id': 'e1',
                'confidence': 0.9,
                'attrs': {'confidence': 0.1},
            }
        ]
    )
    assert _attr_value(G, 'e1', 'confidence') == 0.9


# ── add_edges: hyperedge batch (new src/tgt API) ───────────────────────────


def test_add_hyperedges_batch_undirected_via_src_list():
    """List-shaped src with no tgt → undirected hyperedge."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(
        [
            {'src': ['A', 'B', 'C'], 'edge_id': 'h1', 'confidence': 0.5},
        ]
    )
    assert _attr_value(G, 'h1', 'confidence') == 0.5
    rec = G._edges['h1']
    assert rec.etype == 'hyper'
    assert rec.directed is False
    assert set(rec.src) == {'A', 'B', 'C'}
    assert rec.tgt is None


def test_add_hyperedges_batch_directed_via_src_tgt_lists():
    """List-shaped src AND tgt → directed hyperedge (src is tail, tgt is head)."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(
        [
            {'src': ['A'], 'tgt': ['B', 'C', 'D'], 'edge_id': 'h2', 'pathway': 'tca'},
        ]
    )
    assert _attr_value(G, 'h2', 'pathway') == 'tca'
    rec = G._edges['h2']
    assert rec.etype == 'hyper'
    assert rec.directed is True
    # internal record stores head in src, tail in tgt
    assert set(rec.src) == {'B', 'C', 'D'}
    assert set(rec.tgt) == {'A'}


def test_add_hyperedges_batch_source_target_aliases():
    """`source`/`target` aliases must work the same as src/tgt."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(
        [
            {'source': ['A', 'B', 'C'], 'edge_id': 'h1', 'category': 'complex'},
        ]
    )
    assert _attr_value(G, 'h1', 'category') == 'complex'
    assert G._edges['h1'].etype == 'hyper'


# ── add_edges: entity batch (already works — regression guard) ────────────


def test_add_edges_entity_batch_preserves_user_attrs():
    G = AnnNet(directed=False)
    ids = G.add_edges(
        [
            {'edge_id': 'EE1', 'role': 'enzyme', 'pathway': 'glycolysis'},
            {'edge_id': 'EE2', 'role': 'enzyme', 'pathway': 'tca'},
        ],
        as_entity=True,
    )
    assert ids == ['EE1', 'EE2']
    assert _attr_value(G, 'EE1', 'role') == 'enzyme'
    assert _attr_value(G, 'EE2', 'pathway') == 'tca'


# ── add_vertices: list-form kwargs broadcast ──────────────────────────────


def test_add_vertices_list_kwargs_broadcasts_attrs():
    """Calling add_vertices with a list and kwargs must broadcast the
    kwargs across every listed vertex (matches single-string semantics)."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'], kind='gene', expression=1.5)
    for vid in ('A', 'B', 'C'):
        assert _vertex_attr_value(G, vid, 'kind') == 'gene'
        assert _vertex_attr_value(G, vid, 'expression') == 1.5


def test_add_vertices_dict_batch_preserves_attrs():
    G = AnnNet(directed=False)
    G.add_vertices(
        [
            {'vertex_id': 'A', 'kind': 'gene'},
            {'vertex_id': 'B', 'kind': 'protein'},
        ]
    )
    assert _vertex_attr_value(G, 'A', 'kind') == 'gene'
    assert _vertex_attr_value(G, 'B', 'kind') == 'protein'


def test_add_vertices_per_item_attrs_override_kwargs():
    """Per-item attrs in dict form take precedence over batch kwargs."""
    G = AnnNet(directed=False)
    G.add_vertices(
        [
            {'vertex_id': 'A', 'kind': 'protein'},
            {'vertex_id': 'B'},
        ],
        kind='gene',
    )
    assert _vertex_attr_value(G, 'A', 'kind') == 'protein'
    assert _vertex_attr_value(G, 'B', 'kind') == 'gene'
