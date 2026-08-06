"""Targeted coverage for ``annnet.core._Ops`` (SCV-5).

Covers ``reverse``, ``subgraph_from_slice``, ``memory_usage``,
``vertex_incidence_matrix``, ``get_vertex_incidence_matrix_as_lists``,
and the ``G.ops`` accessor's flat forwarders. Existing tests already
exercise ``copy`` / ``subgraph`` / ``edge_subgraph`` / ``extract_subgraph``;
this file fills the analytics gap.

Target: ≥80 % per SCV-5.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from annnet.core import _structure as S
from annnet.core.graph import AnnNet


def _mixed_graph() -> AnnNet:
    """Directed graph with binary, undirected, and hyperedges + a slice."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e_dir', weight=2.0)
    G.add_edges('B', 'C', edge_id='e_undir', directed=False, weight=1.0)
    G.add_edges([{'src': ['A', 'B', 'C'], 'edge_id': 'h_und'}])
    G.add_edges([{'src': ['A'], 'tgt': ['B', 'C'], 'edge_id': 'h_dir'}])
    G.slices.add('S1')
    G.slices.add_edge_to_slice('S1', 'e_dir')
    G.slices.add_edge_to_slice('S1', 'h_dir')
    return G


# ── reverse ─────────────────────────────────────────────────────────────


def test_reverse_swaps_directed_binary_endpoints() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    rec_orig = S.edge_shape(G, 'e_dir')
    rec_rev = S.edge_shape(R, 'e_dir')
    assert (rec_orig.src, rec_orig.tgt) == (rec_rev.tgt, rec_rev.src)


def test_reverse_leaves_undirected_binary_unchanged() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    a = S.edge_shape(G, 'e_undir')
    b = S.edge_shape(R, 'e_undir')
    assert (a.src, a.tgt) == (b.src, b.tgt)


def test_reverse_swaps_directed_hyperedge_head_tail() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    a = S.edge_shape(G, 'h_dir')
    b = S.edge_shape(R, 'h_dir')
    assert a.src == b.tgt and a.tgt == b.src


def test_reverse_leaves_undirected_hyperedge_unchanged() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    assert set(S.edge_shape(G, 'h_und').src) == set(S.edge_shape(R, 'h_und').src)
    assert S.edge_shape(R, 'h_und').tgt is None


def test_reverse_returns_new_graph() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    assert R is not G


# ── subgraph_from_slice ─────────────────────────────────────────────────


def test_subgraph_from_slice_contains_slice_members() -> None:
    G = _mixed_graph()
    H = G.subgraph_from_slice('S1')
    # Slice S1 holds e_dir and h_dir.
    assert S.has_edge(H, 'e_dir')
    assert S.has_edge(H, 'h_dir')


def test_subgraph_from_slice_raises_on_unknown_slice() -> None:
    G = _mixed_graph()
    with pytest.raises(KeyError, match='ghost'):
        G.subgraph_from_slice('ghost')


def test_subgraph_from_slice_resolves_slice_weight_overrides() -> None:
    G = _mixed_graph()
    G.attrs.set_edge_slice_attrs('S1', 'e_dir', weight=7.5)
    H = G.subgraph_from_slice('S1', resolve_slice_weights=True)
    assert S.has_edge(H, 'e_dir')


# ── memory_usage ────────────────────────────────────────────────────────


def test_memory_usage_returns_positive_integer() -> None:
    G = _mixed_graph()
    n = G.ops.memory_usage()
    assert isinstance(n, int)
    assert n > 0


# ── incidence matrix forms ──────────────────────────────────────────────


def test_vertex_incidence_matrix_dense_binary_mask() -> None:
    G = _mixed_graph()
    M = G.ops.vertex_incidence_matrix(values=False, sparse=False)
    assert isinstance(M, np.ndarray)
    # Binary mask: all non-zero entries are exactly 1.
    nz = M[M != 0]
    assert ((nz == 1) | (nz == -1)).all() or (nz == 1).all() or set(np.unique(nz)).issubset({0, 1})


def test_vertex_incidence_matrix_dense_values() -> None:
    G = _mixed_graph()
    M = G.ops.vertex_incidence_matrix(values=True, sparse=False)
    assert isinstance(M, np.ndarray)


def test_vertex_incidence_matrix_sparse_returns_csr() -> None:
    G = _mixed_graph()
    M = G.ops.vertex_incidence_matrix(values=False, sparse=True)
    assert sp.issparse(M) and M.format == 'csr'


def test_ops_incidence_alias_matches_vertex_incidence_matrix() -> None:
    G = _mixed_graph()
    a = G.ops.incidence(values=True, sparse=False)
    b = G.ops.vertex_incidence_matrix(values=True, sparse=False)
    assert np.array_equal(a, b)


def test_get_vertex_incidence_matrix_as_lists_returns_indices_by_default() -> None:
    G = _mixed_graph()
    out = G.ops.get_vertex_incidence_matrix_as_lists()
    assert isinstance(out, dict)
    # Every vertex should map to a list of incident column indices.
    for v in G.vertices():
        assert v in out
        assert isinstance(out[v], list)


def test_get_vertex_incidence_matrix_as_lists_can_return_values() -> None:
    G = _mixed_graph()
    out = G.ops.get_vertex_incidence_matrix_as_lists(values=True)
    for v in G.vertices():
        assert v in out
        for x in out[v]:
            assert isinstance(x, (int, float))


def test_ops_incidence_as_lists_alias_matches() -> None:
    G = _mixed_graph()
    a = G.ops.incidence_as_lists()
    b = G.ops.get_vertex_incidence_matrix_as_lists()
    assert a == b


# ── OperationsAccessor flat forwarders ──────────────────────────────────


def test_ops_subgraph_and_edge_subgraph_forwarders() -> None:
    G = _mixed_graph()
    sub = G.ops.subgraph(['A', 'B'])
    assert sub.nv == 2
    esub = G.ops.edge_subgraph(['e_dir'])
    assert S.has_edge(esub, 'e_dir')


def test_subgraph_preserves_graph_attributes() -> None:
    G = AnnNet(project='proj1', stage='draft')
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')

    H = G.ops.subgraph({'A', 'B'})

    assert H.graph_attributes == {'project': 'proj1', 'stage': 'draft'}


def test_edge_subgraph_does_not_leak_constructor_capacity_into_graph_attributes() -> None:
    G = AnnNet(aspects={'time': ['t1']}, project='demo')
    G.add_vertices('A', layer='t1')
    G.add_vertices('B', layer='t1')
    G.add_edges(('A', ('t1',)), ('B', ('t1',)), edge_id='e1')

    H = G.ops.edge_subgraph(['e1'])

    assert 'n' not in H.graph_attributes
    assert H.graph_attributes['project'] == 'demo'
    assert H.edges() == ['e1']


def test_ops_extract_and_extract_subgraph_aliases_match() -> None:
    G = _mixed_graph()
    a = G.ops.extract(vertices=['A', 'B'])
    b = G.ops.extract_subgraph(vertices=['A', 'B'])
    assert set(a.vertices()) == set(b.vertices())


def test_ops_copy_returns_independent_graph() -> None:
    G = _mixed_graph()
    H = G.ops.copy()
    assert H is not G
    assert set(H.vertices()) == set(G.vertices())
    H.add_vertices(['Z'])
    assert 'Z' not in G.vertices()


def test_ops_reverse_via_accessor() -> None:
    G = _mixed_graph()
    R = G.ops.reverse()
    assert R is not G


def test_ops_memory_usage_forwarder() -> None:
    G = _mixed_graph()
    assert G.ops.memory_usage() > 0


# ── extract_subgraph (the multi-branch entry point) ─────────────────────


def test_extract_subgraph_no_filters_returns_copy() -> None:
    G = _mixed_graph()
    H = G.ops.extract_subgraph()
    assert set(H.vertices()) == set(G.vertices())
    assert H is not G


def test_extract_subgraph_edge_indices_are_resolved() -> None:
    G = _mixed_graph()
    cols = [G.idx.edge_to_col('e_dir'), G.idx.edge_to_col('e_undir')]
    H = G.ops.extract_subgraph(edges=cols)
    assert S.has_edge(H, 'e_dir')
    assert S.has_edge(H, 'e_undir')


def test_extract_subgraph_vertex_filter_only_path() -> None:
    G = _mixed_graph()
    H = G.ops.extract_subgraph(vertices=['A', 'B'])
    assert set(H.vertices()) == {'A', 'B'}


def test_extract_subgraph_edge_filter_only_path() -> None:
    G = _mixed_graph()
    H = G.ops.extract_subgraph(edges=['e_dir'])
    assert S.has_edge(H, 'e_dir')


def test_extract_subgraph_both_filters_keeps_only_endpoint_safe_edges() -> None:
    G = _mixed_graph()
    H = G.ops.extract_subgraph(vertices={'A', 'B'}, edges={'e_dir', 'e_undir'})
    # e_dir = (A, B) — both endpoints in V → kept.
    # e_undir = (B, C) — C not in V → dropped.
    assert S.has_edge(H, 'e_dir')
    assert not S.has_edge(H, 'e_undir')


def test_extract_subgraph_both_filters_keeps_hyperedge_when_all_members_in_v() -> None:
    G = _mixed_graph()
    H = G.ops.extract_subgraph(
        vertices={'A', 'B', 'C'},
        edges={'h_und', 'h_dir', 'e_dir', 'e_undir'},
    )
    assert S.has_edge(H, 'h_und')
    assert S.has_edge(H, 'h_dir')


# ── copy modes ──────────────────────────────────────────────────────────


def test_copy_with_history_flag() -> None:
    G = _mixed_graph()
    H = G.ops.copy(history=True)
    assert H is not G
    assert set(H.vertices()) == set(G.vertices())


# ── multilayer subgraph_from_slice ──────────────────────────────────────


def _multilayer_slice_graph() -> AnnNet:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A', 'B'], layer={'condition': 'healthy'})
    G.add_vertices(['C'], layer={'condition': 'treated'})
    G.add_edges(
        ('A', ('healthy',)),
        ('B', ('healthy',)),
        edge_id='e_intra',
    )
    G.slices.add('S1')
    G.slices.add_edge_to_slice('S1', 'e_intra')
    return G


def test_subgraph_from_slice_on_multilayer_graph() -> None:
    G = _multilayer_slice_graph()
    H = G.subgraph_from_slice('S1')
    assert S.has_edge(H, 'e_intra')
    assert H.slices.active == 'S1'


def test_subgraph_from_slice_multilayer_with_binary_edges_only() -> None:
    """Cover the binary-edge branch of the multilayer subgraph_from_slice path."""
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy']})
    G.add_vertices(['A', 'B', 'C'], layer={'condition': 'healthy'})
    G.add_edges(
        ('A', ('healthy',)),
        ('B', ('healthy',)),
        edge_id='e1',
    )
    G.add_edges(
        ('B', ('healthy',)),
        ('C', ('healthy',)),
        edge_id='e2',
    )
    G.slices.add('S1')
    G.slices.add_edge_to_slice('S1', 'e1')
    G.slices.add_edge_to_slice('S1', 'e2')
    H = G.subgraph_from_slice('S1')
    assert S.has_edge(H, 'e1')
    assert S.has_edge(H, 'e2')


def test_multilayer_subgraph_from_slice_preserves_graph_attributes_and_real_supra_vertices():
    G = AnnNet(directed=True, project='sliceproj')
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A', 'B'], layer={'condition': 'healthy'}, slice='S1')
    G.add_vertices(['C'], layer={'condition': 'treated'}, slice='S1')
    G.add_edges(
        [
            {
                'members': [('A', ('healthy',)), ('B', ('healthy',)), ('C', ('treated',))],
                'edge_id': 'h1',
            }
        ],
        slice='S1',
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        H = G.subgraph_from_slice('S1')

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any('placeholder layer' in msg for msg in msgs), msgs
    assert H.graph_attributes['project'] == 'sliceproj'
    assert set(H.supra_vertices()) == {
        ('A', ('healthy',)),
        ('B', ('healthy',)),
        ('C', ('treated',)),
    }
    assert S.edge_shape(H, 'h1').src == frozenset(
        {('A', ('healthy',)), ('B', ('healthy',)), ('C', ('treated',))}
    )


# ── multilayer edge_subgraph / subgraph branches ────────────────────────


def _multilayer_with_supra_edges() -> AnnNet:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A', 'B', 'C'], layer={'condition': 'healthy'})
    G.add_vertices(['D'], layer={'condition': 'treated'})
    # Binary supra-node edges.
    G.add_edges(
        ('A', ('healthy',)),
        ('B', ('healthy',)),
        edge_id='e_intra1',
    )
    G.add_edges(
        ('B', ('healthy',)),
        ('C', ('healthy',)),
        edge_id='e_intra2',
    )
    # Inter-layer edge for the multilayer subgraph branch.
    G.add_edges(
        ('A', ('healthy',)),
        ('D', ('treated',)),
        edge_id='e_inter',
    )
    return G


def test_edge_subgraph_multilayer_preserves_supra_node_edges() -> None:
    G = _multilayer_with_supra_edges()
    H = G.ops.edge_subgraph(['e_intra1', 'e_inter'])
    assert S.has_edge(H, 'e_intra1')
    assert S.has_edge(H, 'e_inter')


def test_extract_subgraph_multilayer_with_both_filters() -> None:
    G = _multilayer_with_supra_edges()
    H = G.ops.extract_subgraph(
        vertices={'A', 'B', 'C'},
        edges={'e_intra1', 'e_intra2', 'e_inter'},
    )
    # e_intra1 (A→B) and e_intra2 (B→C) — both endpoints in V → kept.
    # e_inter (A→D) — D not in V → dropped.
    assert S.has_edge(H, 'e_intra1')
    assert S.has_edge(H, 'e_intra2')
    assert not S.has_edge(H, 'e_inter')


def test_copy_multilayer_preserves_aspects_and_layer_attrs() -> None:
    """Exercises the multilayer aspects-cloning branch of Operations.copy."""
    G = _multilayer_with_supra_edges()
    H = G.ops.copy()
    assert H.layers.aspects == G.layers.aspects
    assert dict(H.layers.elem_layers) == dict(G.layers.elem_layers)


def test_copy_multilayer_does_not_emit_placeholder_reassignment_warning() -> None:
    G = AnnNet(aspects={'time': ['t1']})
    G.add_vertices('A', layer='t1')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        H = G.ops.copy()

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any('reassigned to placeholder layer' in msg for msg in msgs), msgs
    assert H.layers.list_aspects() == ('time',)


def test_subgraph_on_multilayer_with_hyperedge_via_supra_members() -> None:
    """Hyperedge whose members are supra-node tuples must round-trip
    through ``subgraph``."""
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy']})
    G.add_vertices(['A', 'B', 'C'], layer={'condition': 'healthy'})
    G.add_edges(
        [
            {
                'src': [
                    ('A', ('healthy',)),
                    ('B', ('healthy',)),
                    ('C', ('healthy',)),
                ],
                'edge_id': 'h_und',
            }
        ]
    )
    H = G.ops.subgraph(['A', 'B', 'C'])
    assert S.has_edge(H, 'h_und')


def test_edge_subgraph_multilayer_with_hyperedge_via_supra_members() -> None:
    """Same as above but through the edge_subgraph branch."""
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy']})
    G.add_vertices(['A', 'B', 'C'], layer={'condition': 'healthy'})
    G.add_edges(
        [
            {
                'src': [('A', ('healthy',)), ('B', ('healthy',))],
                'tgt': [('C', ('healthy',))],
                'edge_id': 'h_dir',
            }
        ]
    )
    H = G.ops.edge_subgraph(['h_dir'])
    assert S.has_edge(H, 'h_dir')


def test_edge_subgraph_multilayer_hyperedge_preserves_supra_members_without_placeholder_warnings():
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A', 'B'], layer={'condition': 'healthy'})
    G.add_vertices(['C'], layer={'condition': 'treated'})
    G.add_edges(
        [
            {
                'members': [('A', ('healthy',)), ('B', ('healthy',)), ('C', ('treated',))],
                'edge_id': 'h1',
            }
        ]
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        H = G.ops.edge_subgraph(['h1'])

    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert not any('placeholder layer' in msg for msg in msgs), msgs
    assert set(H.supra_vertices()) == {
        ('A', ('healthy',)),
        ('B', ('healthy',)),
        ('C', ('treated',)),
    }
    assert S.edge_shape(H, 'h1').src == frozenset(
        {('A', ('healthy',)), ('B', ('healthy',)), ('C', ('treated',))}
    )


# ── the attributes of named elements, read out of the columns ─


def test_the_attributes_of_named_elements_come_out_of_the_columns() -> None:
    """The store answers about the elements asked for, and only those."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'], kind='gene')

    rows = G._attr_store.node_attr_rows(['A'])
    assert rows == {'A': {'kind': 'gene'}}

    # An id the graph does not hold is not a row of the result.
    assert G._attr_store.node_attr_rows(['ghost']) == {}

    # An attribute no node carries is no column, which is not the same as a
    # column every node leaves empty.
    assert G._attr_store.node_attr_map('no_such_attribute') is None
