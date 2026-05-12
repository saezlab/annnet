"""Targeted coverage for ``annnet.core._Matrix`` (SCV-5).

Exercises the ``G.cache`` (CacheManager) and ``G.idx`` (IndexManager)
namespaces and their error paths. The goal here is the SCV-5 threshold
(≥80 %), not exhaustive behavioural coverage.
"""

from __future__ import annotations

import pytest
import scipy.sparse as sp

from annnet.core.graph import AnnNet


def _build_graph() -> AnnNet:
    """Small directed graph with a vertex-edge entity and one binary edge."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.5)
    G.add_edges('B', 'C', edge_id='e2', weight=1.5)
    # An edge-entity (acts as an endpoint) so coverage exercises both kinds.
    G.add_edges(edge_id='EE1', as_entity=True)
    return G


# ── CacheManager ────────────────────────────────────────────────────────


def test_cache_csr_csc_adjacency_build_and_return_sparse() -> None:
    G = _build_graph()
    csr = G.cache.csr
    csc = G.cache.csc
    adj = G.cache.adjacency

    assert sp.issparse(csr) and csr.format == 'csr'
    assert sp.issparse(csc) and csc.format == 'csc'
    assert sp.issparse(adj)
    # Adjacency = B @ B.T is square on the entity dimension.
    assert adj.shape[0] == adj.shape[1] == G.cache.csr.shape[0]


def test_cache_has_flags_track_validity() -> None:
    G = _build_graph()

    # Before first access nothing is cached.
    assert G.cache.has_csr() is False
    assert G.cache.has_csc() is False
    assert G.cache.has_adjacency() is False

    G.cache.build(['csr'])
    assert G.cache.has_csr() is True
    assert G.cache.has_csc() is False

    G.cache.build(['csc', 'adjacency'])
    assert G.cache.has_csc() is True
    assert G.cache.has_adjacency() is True


def test_cache_get_methods_match_property_access() -> None:
    G = _build_graph()
    assert G.cache.get_csr() is G.cache.csr
    assert G.cache.get_csc() is G.cache.csc
    assert G.cache.get_adjacency() is G.cache.adjacency


def test_cache_invalidate_selective_and_full() -> None:
    G = _build_graph()
    G.cache.build()
    assert G.cache.has_csr() and G.cache.has_csc() and G.cache.has_adjacency()

    G.cache.invalidate(['csr'])
    assert G.cache.has_csr() is False
    assert G.cache.has_csc() is True

    G.cache.invalidate()  # full
    assert not G.cache.has_csr()
    assert not G.cache.has_csc()
    assert not G.cache.has_adjacency()


def test_cache_clear_alias_invalidates_everything() -> None:
    G = _build_graph()
    G.cache.build()
    G.cache.clear()
    assert not G.cache.has_csr()
    assert not G.cache.has_csc()
    assert not G.cache.has_adjacency()


def test_cache_invalidate_unknown_format_is_a_noop() -> None:
    G = _build_graph()
    G.cache.build(['csr'])
    G.cache.invalidate(['nonsense'])
    assert G.cache.has_csr()


def test_cache_build_unknown_format_is_a_noop() -> None:
    G = _build_graph()
    G.cache.build(['nonsense'])
    assert not G.cache.has_csr()


def test_cache_info_reports_uncached_and_cached_states() -> None:
    G = _build_graph()
    before = G.cache.info()
    assert before['csr'] == {'cached': False}
    assert before['csc'] == {'cached': False}
    assert before['adjacency'] == {'cached': False}

    G.cache.build()
    after = G.cache.info()
    for fmt in ('csr', 'csc', 'adjacency'):
        assert after[fmt]['cached'] is True
        assert after[fmt]['nnz'] >= 0
        assert isinstance(after[fmt]['shape'], tuple)
        assert after[fmt]['size_mb'] >= 0


# ── IndexManager ───────────────────────────────────────────────────────


def test_idx_entity_to_row_round_trips_vertex_and_edge_entity() -> None:
    G = _build_graph()
    a_row = G.idx.entity_to_row('A')
    assert G.idx.row_to_entity(a_row) == 'A'

    ee_row = G.idx.entity_to_row('EE1')
    assert G.idx.row_to_entity(ee_row) == 'EE1'


def test_idx_entity_to_row_raises_on_unknown() -> None:
    G = _build_graph()
    with pytest.raises(KeyError, match='not_a_vertex'):
        G.idx.entity_to_row('not_a_vertex')


def test_idx_row_to_entity_raises_on_unknown_row() -> None:
    G = _build_graph()
    with pytest.raises(KeyError, match='9999'):
        G.idx.row_to_entity(9999)


def test_idx_entities_to_rows_and_rows_to_entities_round_trip() -> None:
    G = _build_graph()
    rows = G.idx.entities_to_rows(['A', 'B', 'C'])
    assert G.idx.rows_to_entities(rows) == ['A', 'B', 'C']


def test_idx_edge_to_col_round_trips() -> None:
    G = _build_graph()
    c = G.idx.edge_to_col('e1')
    assert G.idx.col_to_edge(c) == 'e1'


def test_idx_edge_to_col_raises_on_unknown_or_placeholder() -> None:
    G = _build_graph()
    with pytest.raises(KeyError, match='nope'):
        G.idx.edge_to_col('nope')
    # An edge-entity placeholder has col_idx < 0 — same error path.
    with pytest.raises(KeyError, match='EE1'):
        G.idx.edge_to_col('EE1')


def test_idx_col_to_edge_raises_on_unknown() -> None:
    G = _build_graph()
    with pytest.raises(KeyError, match='42'):
        G.idx.col_to_edge(42)


def test_idx_edges_to_cols_and_cols_to_edges_round_trip() -> None:
    G = _build_graph()
    cols = G.idx.edges_to_cols(['e1', 'e2'])
    assert G.idx.cols_to_edges(cols) == ['e1', 'e2']


def test_idx_entity_type_distinguishes_vertex_and_edge_entity() -> None:
    G = _build_graph()
    assert G.idx.entity_type('A') == 'vertex'
    assert G.idx.entity_type('EE1') == 'edge'


def test_idx_entity_type_raises_on_unknown() -> None:
    G = _build_graph()
    with pytest.raises(KeyError, match='ghost'):
        G.idx.entity_type('ghost')


def test_idx_is_vertex_and_is_edge_entity() -> None:
    G = _build_graph()
    assert G.idx.is_vertex('A') is True
    assert G.idx.is_vertex('EE1') is False
    assert G.idx.is_edge_entity('EE1') is True
    assert G.idx.is_edge_entity('A') is False


def test_idx_has_entity_has_vertex_has_edge_id() -> None:
    G = _build_graph()
    assert G.idx.has_entity('A') is True
    assert G.idx.has_entity('EE1') is True
    assert G.idx.has_entity('missing') is False

    assert G.idx.has_vertex('A') is True
    assert G.idx.has_vertex('EE1') is False  # edge-entity, not vertex
    assert G.idx.has_vertex('missing') is False

    assert G.idx.has_edge_id('e1') is True
    assert G.idx.has_edge_id('EE1') is False  # col_idx < 0
    assert G.idx.has_edge_id('missing') is False


def test_idx_count_helpers_match_graph_shape() -> None:
    G = _build_graph()
    assert G.idx.edge_count() == G.ne
    assert G.idx.entity_count() == len(G._entities)
    assert G.idx.vertex_count() == G.nv


def test_idx_stats_includes_vertex_and_edge_entity_counts() -> None:
    G = _build_graph()
    stats = G.idx.stats()
    assert stats['n_vertices'] == 3
    assert stats['n_edge_entities'] == 1
    assert stats['n_edges'] == 2
    assert stats['n_entities'] == 4
    assert stats['max_row'] == stats['n_entities'] - 1
