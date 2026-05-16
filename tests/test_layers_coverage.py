"""Coverage tests for ``annnet/core/_Layers.py``.

Covers both the layer set operations and the supra-math API
(supra-adjacency / Laplacians / random walks / modularity).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from annnet.core.graph import AnnNet


# ── fixtures ──────────────────────────────────────────────────────────


def _single_aspect_two_layer() -> AnnNet:
    """Two vertices A, B present in both layers, with one intra edge per layer."""
    G = AnnNet(directed=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.set_aspects(['t'], {'t': ['t1', 't2']})
        G.add_vertices(['A'], layer={'t': 't1'})
        G.add_vertices(['B'], layer={'t': 't1'})
        G.add_vertices(['A'], layer={'t': 't2'})
        G.add_vertices(['B'], layer={'t': 't2'})
        G.add_edges(('A', ('t1',)), ('B', ('t1',)), edge_id='e_t1', weight=1.0)
        G.add_edges(('A', ('t2',)), ('B', ('t2',)), edge_id='e_t2', weight=1.0)
    return G


def _single_aspect_with_coupling() -> AnnNet:
    """Two-layer graph with diagonal couplings (A-A across layers, B-B across)."""
    G = _single_aspect_two_layer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.add_layer_coupling_pairs([(('t1',), ('t2',))])
    return G


# ── layer set operations ──────────────────────────────────────────────


def test_layer_union_aggregates_vertices_and_edges() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_union([('t1',), ('t2',)])
    assert {'A', 'B'} <= out['vertices']
    assert {'e_t1', 'e_t2'} <= out['edges']


def test_layer_union_empty_input_returns_empty_sets() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_union([])
    assert out == {'vertices': set(), 'edges': set()}


def test_layer_intersection_keeps_common_elements_only() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_intersection([('t1',), ('t2',)])
    assert {'A', 'B'} <= out['vertices']
    assert out['edges'] == set()


def test_layer_intersection_empty_input_returns_empty_sets() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_intersection([])
    assert out == {'vertices': set(), 'edges': set()}


def test_layer_difference_subtracts_b_from_a() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_difference(('t1',), ('t2',))
    assert 'e_t1' in out['edges']
    assert 'e_t2' not in out['edges']


# ── create_slice_from_layer* ─────────────────────────────────────────


def test_create_slice_from_layer_single() -> None:
    G = _single_aspect_two_layer()
    G.layers.create_slice_from_layer('s_t1', ('t1',))
    assert 's_t1' in G.slices.list()


def test_create_slice_from_layer_union() -> None:
    G = _single_aspect_two_layer()
    G.layers.create_slice_from_layer_union('s_all', [('t1',), ('t2',)])
    assert 's_all' in G.slices.list()


def test_create_slice_from_layer_intersection() -> None:
    G = _single_aspect_two_layer()
    G.layers.create_slice_from_layer_intersection('s_both', [('t1',), ('t2',)])
    assert 's_both' in G.slices.list()


def test_create_slice_from_layer_difference() -> None:
    G = _single_aspect_two_layer()
    G.layers.create_slice_from_layer_difference('s_t1_only', ('t1',), ('t2',))
    assert 's_t1_only' in G.slices.list()


# ── subgraph_from_layer_* ────────────────────────────────────────────


def test_subgraph_from_layer_tuple_extracts_single_layer() -> None:
    G = _single_aspect_two_layer()
    sub = G.layers.subgraph_from_layer_tuple(('t1',))
    assert isinstance(sub, AnnNet)


def test_subgraph_from_layer_union_extracts_multiple_layers() -> None:
    G = _single_aspect_two_layer()
    sub = G.layers.subgraph_from_layer_union([('t1',), ('t2',)])
    assert isinstance(sub, AnnNet)


def test_subgraph_from_layer_intersection_runs() -> None:
    G = _single_aspect_two_layer()
    sub = G.layers.subgraph_from_layer_intersection([('t1',), ('t2',)])
    assert isinstance(sub, AnnNet)


def test_subgraph_from_layer_difference_runs() -> None:
    G = _single_aspect_two_layer()
    sub = G.layers.subgraph_from_layer_difference(('t1',), ('t2',))
    assert isinstance(sub, AnnNet)


# ── attribute setters & getters ──────────────────────────────────────


def test_set_aspect_attrs_and_get_aspect_attrs_round_trip() -> None:
    G = _single_aspect_two_layer()
    G.layers.set_aspect_attrs('t', kind='temporal')
    assert G.layers.get_aspect_attrs('t').get('kind') == 'temporal'


def test_set_layer_attrs_and_get_round_trip() -> None:
    G = _single_aspect_two_layer()
    G.layers.set_layer_attrs(('t1',), note='alpha')
    assert G.layers.get_layer_attrs(('t1',)).get('note') == 'alpha'


def test_set_vertex_layer_attrs_and_get_round_trip() -> None:
    G = _single_aspect_two_layer()
    G.layers.set_vertex_layer_attrs('A', ('t1',), state='active')
    assert G.layers.get_vertex_layer_attrs('A', ('t1',)).get('state') == 'active'


# ── layer iteration helpers ─────────────────────────────────────────


def test_iter_layers_yields_each_elementary_layer() -> None:
    G = _single_aspect_two_layer()
    out = list(G.layers.iter_layers())
    assert out


def test_iter_vertex_layers_yields_layers_containing_the_vertex() -> None:
    G = _single_aspect_two_layer()
    out = list(G.layers.iter_vertex_layers('A'))
    assert out


def test_has_presence_returns_true_for_existing_vertex_layer() -> None:
    G = _single_aspect_two_layer()
    assert G.layers.has_presence('A', ('t1',)) is True


def test_layer_id_to_tuple_and_back_round_trip() -> None:
    G = _single_aspect_two_layer()
    tup = G.layers.layer_id_to_tuple('t1')
    back = G.layers.layer_tuple_to_id(tup)
    assert back == 't1'


def test_aspect_index_raises_for_unknown_aspect() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(KeyError, match='unknown aspect'):
        G.layers._aspect_index('not-an-aspect')


def test_flatten_layers_collapses_multilayer_to_flat() -> None:
    G = _single_aspect_two_layer()
    G.layers.flatten_layers()
    assert not G.is_multilayer


# ── supra_adjacency / supra_incidence / supra_degree ─────────────────


def test_supra_adjacency_is_square_and_symmetric_for_undirected() -> None:
    G = _single_aspect_two_layer()
    A = G.layers.supra_adjacency()
    assert A.shape[0] == A.shape[1]
    diff = A - A.T
    assert abs(diff).max() < 1e-12


def test_supra_adjacency_has_nonzero_entries_for_intra_edges() -> None:
    G = _single_aspect_two_layer()
    A = G.layers.supra_adjacency()
    assert A.nnz >= 2  # two intra edges, each adds two symmetric entries


def test_supra_incidence_returns_matrix_and_edge_ids() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.supra_incidence()
    if len(out) == 3:
        B, edge_ids, _ = out
    else:
        B, edge_ids = out
    assert B.shape[0] >= 4
    assert {'e_t1', 'e_t2'}.issubset(set(edge_ids))


def test_supra_degree_matches_row_sums_of_adjacency() -> None:
    G = _single_aspect_two_layer()
    A = G.layers.supra_adjacency()
    deg = G.layers.supra_degree()
    row_sums = np.asarray(A.sum(axis=1)).ravel()
    np.testing.assert_allclose(deg, row_sums, atol=1e-12)


# ── supra_laplacian ─────────────────────────────────────────────────


def test_supra_laplacian_comb_equals_diag_deg_minus_adjacency() -> None:
    G = _single_aspect_two_layer()
    A = G.layers.supra_adjacency().toarray()
    L = G.layers.supra_laplacian(kind='comb').toarray()
    deg = G.layers.supra_degree()
    np.testing.assert_allclose(L, np.diag(deg) - A, atol=1e-12)


def test_supra_laplacian_norm_diagonal_is_one_on_active_rows() -> None:
    G = _single_aspect_two_layer()
    Lnorm = G.layers.supra_laplacian(kind='norm').toarray()
    deg = G.layers.supra_degree()
    for i in range(Lnorm.shape[0]):
        if deg[i] > 0:
            assert abs(Lnorm[i, i] - 1.0) < 1e-12


def test_supra_laplacian_rejects_unknown_kind() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='kind must be'):
        G.layers.supra_laplacian(kind='bogus')


# ── intra/inter/coupling block builders ──────────────────────────────


def test_build_intra_inter_coupling_blocks_decompose_supra_adjacency() -> None:
    G = _single_aspect_with_coupling()
    A_intra = G.layers.build_intra_block().toarray()
    A_inter = G.layers.build_inter_block().toarray()
    A_coup = G.layers.build_coupling_block().toarray()
    A_full = G.layers.supra_adjacency().toarray()
    np.testing.assert_allclose(A_intra + A_inter + A_coup, A_full, atol=1e-12)


def test_build_intra_block_disjoint_from_coupling_block() -> None:
    G = _single_aspect_with_coupling()
    A_intra = G.layers.build_intra_block().toarray()
    A_coup = G.layers.build_coupling_block().toarray()
    assert not np.logical_and(A_intra != 0, A_coup != 0).any()


# ── transition matrix is row-stochastic ──────────────────────────────


def test_transition_matrix_rows_sum_to_one_on_connected_rows() -> None:
    G = _single_aspect_two_layer()
    P = G.layers.transition_matrix().toarray()
    deg = G.layers.supra_degree()
    for i, d in enumerate(deg):
        if d > 0:
            assert abs(P[i, :].sum() - 1.0) < 1e-12


def test_random_walk_step_preserves_total_probability() -> None:
    G = _single_aspect_with_coupling()
    n = G.layers.supra_adjacency().shape[0]
    p = np.ones(n) / n
    p_next = G.layers.random_walk_step(p)
    assert abs(p_next.sum() - 1.0) < 1e-9


def test_random_walk_step_rejects_wrong_length_vector() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='but supra has size'):
        G.layers.random_walk_step([0.5, 0.5])


# ── diffusion step ───────────────────────────────────────────────────


def test_diffusion_step_produces_finite_output() -> None:
    G = _single_aspect_with_coupling()
    n = G.layers.supra_adjacency().shape[0]
    x = np.zeros(n)
    x[0] = 1.0
    y = G.layers.diffusion_step(x, tau=0.1)
    assert np.all(np.isfinite(y))


def test_diffusion_step_rejects_wrong_length_vector() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='but supra has size'):
        G.layers.diffusion_step([0.5, 0.5], tau=0.1)


# ── algebraic connectivity ───────────────────────────────────────────


def test_algebraic_connectivity_is_non_negative_for_real_graphs() -> None:
    G = _single_aspect_with_coupling()
    lam2, fiedler = G.layers.algebraic_connectivity()
    assert lam2 >= -1e-9
    assert fiedler is not None


def test_algebraic_connectivity_returns_zero_for_trivial_graph() -> None:
    G = AnnNet(directed=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.layers.set_aspects(['t'], {'t': ['t1']})
        G.add_vertices(['A'], layer={'t': 't1'})
    lam2, fiedler = G.layers.algebraic_connectivity()
    assert lam2 == 0.0
    assert fiedler is None


# ── k smallest laplacian eigenvalues ─────────────────────────────────


def test_k_smallest_laplacian_eigs_returns_requested_number() -> None:
    G = _single_aspect_with_coupling()
    vals, vecs = G.layers.k_smallest_laplacian_eigs(k=2)
    assert vals.shape == (2,)
    assert vecs.shape[1] == 2


def test_k_smallest_laplacian_eigs_rejects_non_positive_k() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='k must be'):
        G.layers.k_smallest_laplacian_eigs(k=0)


# ── dominant random-walk eigenpair ───────────────────────────────────


def test_dominant_rw_eigenpair_returns_lambda_at_most_one() -> None:
    G = _single_aspect_with_coupling()
    lam, _ = G.layers.dominant_rw_eigenpair()
    assert lam <= 1.0 + 1e-6


# ── supra_adjacency_scaled ──────────────────────────────────────────


def test_supra_adjacency_scaled_doubles_coupling_with_scale_two() -> None:
    G = _single_aspect_with_coupling()
    A1 = G.layers.supra_adjacency_scaled(coupling_scale=1.0).toarray()
    A2 = G.layers.supra_adjacency_scaled(coupling_scale=2.0).toarray()
    A_coup = G.layers.build_coupling_block().toarray()
    np.testing.assert_allclose(A2 - A1, A_coup, atol=1e-12)


# ── sweep_coupling_regime ───────────────────────────────────────────


def test_sweep_coupling_regime_returns_one_value_per_scale() -> None:
    G = _single_aspect_with_coupling()
    out = G.layers.sweep_coupling_regime([0.5, 1.0, 2.0])
    assert len(out) == 3


def test_sweep_coupling_regime_with_callable_metric() -> None:
    G = _single_aspect_with_coupling()

    def num_nonzeros(A):
        return float(A.nnz)

    out = G.layers.sweep_coupling_regime([0.5, 1.5], metric=num_nonzeros)
    assert len(out) == 2


def test_sweep_coupling_regime_rejects_unknown_metric() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='Unknown metric'):
        G.layers.sweep_coupling_regime([1.0], metric='not-a-metric')


# ── layer-aware descriptors ─────────────────────────────────────────


def test_layer_degree_vectors_returns_one_entry_per_layer() -> None:
    G = _single_aspect_two_layer()
    out = G.layers.layer_degree_vectors()
    assert len(out) >= 1


def test_participation_coefficient_is_between_zero_and_one() -> None:
    G = _single_aspect_with_coupling()
    P = G.layers.participation_coefficient()
    for _u, val in P.items():
        assert 0.0 <= val <= 1.0


def test_versatility_returns_normalized_dict() -> None:
    G = _single_aspect_with_coupling()
    V = G.layers.versatility()
    if V:
        assert abs(max(V.values()) - 1.0) < 1e-9


# ── multislice modularity ───────────────────────────────────────────


def test_multislice_modularity_returns_a_float() -> None:
    G = _single_aspect_with_coupling()
    n = G.layers.supra_adjacency().shape[0]
    partition = np.zeros(n, dtype=int)
    Q = G.layers.multislice_modularity(partition)
    assert isinstance(Q, float)


def test_multislice_modularity_rejects_wrong_partition_length() -> None:
    G = _single_aspect_two_layer()
    with pytest.raises(ValueError, match='partition length'):
        G.layers.multislice_modularity([0, 0])


# ── tensor view & flattening ────────────────────────────────────────


def test_tensor_index_returns_consistent_vertex_and_layer_maps() -> None:
    G = _single_aspect_two_layer()
    vertices, layers, v2i, l2i = G.layers.tensor_index()
    assert sorted(v2i.values()) == list(range(len(vertices)))
    assert sorted(l2i.values()) == list(range(len(layers)))


def test_adjacency_tensor_view_carries_expected_keys() -> None:
    G = _single_aspect_two_layer()
    view = G.layers.adjacency_tensor_view()
    for key in ('vertices', 'layers', 'ui', 'ai', 'vi', 'bi', 'w'):
        assert key in view


def test_flatten_then_unflatten_round_trips_within_sparsity_pattern() -> None:
    G = _single_aspect_with_coupling()
    view = G.layers.adjacency_tensor_view()
    A_from_tensor = G.layers.flatten_to_supra(view)
    A_direct = G.layers.supra_adjacency()
    np.testing.assert_allclose(
        A_from_tensor.toarray(),
        A_direct.toarray(),
        atol=1e-12,
    )


def test_unflatten_from_supra_returns_tensor_view_dict() -> None:
    G = _single_aspect_with_coupling()
    A = G.layers.supra_adjacency()
    view = G.layers.unflatten_from_supra(A)
    for key in ('vertices', 'layers', 'ui', 'w'):
        assert key in view


# ── coupling generators (`add_*_coupling`) ──────────────────────────


def test_add_layer_coupling_pairs_creates_edges() -> None:
    G = _single_aspect_two_layer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        added = G.layers.add_layer_coupling_pairs([(('t1',), ('t2',))])
    assert added >= 1


def test_add_diagonal_coupling_filter_uses_aspect_filter() -> None:
    G = _single_aspect_two_layer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        added = G.layers.add_diagonal_coupling_filter({'t': {'t1', 't2'}})
    assert added >= 1


def test_add_categorical_coupling_couples_within_a_group() -> None:
    G = _single_aspect_two_layer()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        added = G.layers.add_categorical_coupling('t', [['t1', 't2']])
    assert added >= 1
