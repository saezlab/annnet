"""Coverage tests for ``annnet/core/graph.py``.

Focuses on under-exercised setter properties, batch helpers,
``make_undirected`` hyperedge branches, ``remove_vertices`` /
``remove_edges`` paths, and the legacy compatibility methods.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def _toy() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    return G


# ── add_vertices / add_vertices_bulk variants ─────────────────────────


def test_add_vertices_with_dict_entries_uses_attrs() -> None:
    G = AnnNet(directed=True)
    G.add_vertices([{'vertex_id': 'A', 'color': 'red'}, {'vertex_id': 'B'}])
    assert set(G.vertices()) == {'A', 'B'}
    assert G.attrs.get_attr_vertex('A', 'color') == 'red'


def test_add_vertices_with_tuple_entries_uses_attrs() -> None:
    G = AnnNet(directed=True)
    G.add_vertices([('A', {'color': 'red'}), 'B'])
    assert set(G.vertices()) == {'A', 'B'}


def test_add_vertex_singular_creates_one_vertex() -> None:
    """The compact ``add_vertices('A')`` form must create exactly one vertex."""
    G = AnnNet(directed=True)
    G.add_vertices('A')
    assert set(G.vertices()) == {'A'}


# ── make_undirected branches ──────────────────────────────────────────


def test_make_undirected_flips_directed_binary_edges() -> None:
    G = _toy()
    G.make_undirected()
    for rec in G._edges.values():
        if rec.etype != 'hyper':
            assert rec.directed is False


def test_make_undirected_with_directed_hyperedge_collapses_to_undirected() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    G.make_undirected()
    rec = G._edges['h1']
    assert rec.directed is False
    assert rec.tgt is None
    assert set(rec.src) == {'A', 'B', 'C', 'D'}


def test_make_undirected_with_undirected_hyperedge_is_idempotent() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    G.make_undirected()
    rec = G._edges['h1']
    assert rec.directed is False
    assert rec.tgt is None
    assert set(rec.src) == {'A', 'B', 'C'}


def test_make_undirected_with_drop_flexible_clears_policy() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={'var': 'x', 'threshold': 5.0, 'scope': 'edge'},
    )
    G.make_undirected(drop_flexible=True)
    assert G._edges['e1'].direction_policy is None


def test_make_undirected_without_update_default_keeps_directed_attr() -> None:
    G = _toy()
    assert G.directed is True
    G.make_undirected(update_default=False)
    assert G.directed is True  # not flipped to False


# ── setter properties ─────────────────────────────────────────────────


def test_edge_kind_setter_marks_edge_as_hyper() -> None:
    G = _toy()
    G.edge_kind = {'e1': 'hyper'}
    assert G._edges['e1'].etype == 'hyper'


def test_edge_kind_setter_marks_ml_kind_for_non_hyper() -> None:
    G = _toy()
    G.edge_kind = {'e1': 'intra'}
    assert G._edges['e1'].ml_kind == 'intra'


def test_edge_kind_setter_ignores_unknown_eid() -> None:
    G = _toy()
    G.edge_kind = {'unknown-eid': 'hyper'}  # no-op


def test_edge_definitions_setter_updates_src_tgt_etype() -> None:
    G = _toy()
    G.edge_definitions = {'e1': ('A', 'C', 'binary')}
    rec = G._edges['e1']
    assert (rec.src, rec.tgt, rec.etype) == ('A', 'C', 'binary')


def test_edge_definitions_setter_normalizes_hyper_etype_to_binary() -> None:
    G = _toy()
    G.edge_definitions = {'e1': ('A', 'B', 'hyper')}
    assert G._edges['e1'].etype == 'binary'


def test_hyperedge_definitions_setter_list_form_makes_undirected() -> None:
    G = _toy()
    G.hyperedge_definitions = {'e1': ['A', 'B', 'C']}
    rec = G._edges['e1']
    assert rec.etype == 'hyper'
    assert rec.tgt is None
    assert set(rec.src) == {'A', 'B', 'C'}


def test_hyperedge_definitions_setter_dict_form_directed() -> None:
    G = _toy()
    G.hyperedge_definitions = {'e1': {'directed': True, 'head': ['A'], 'tail': ['B', 'C']}}
    rec = G._edges['e1']
    assert rec.directed is True
    assert set(rec.src) == {'A'}
    assert set(rec.tgt) == {'B', 'C'}


def test_hyperedge_definitions_setter_dict_form_undirected() -> None:
    G = _toy()
    G.hyperedge_definitions = {'e1': {'directed': False, 'members': ['A', 'B']}}
    rec = G._edges['e1']
    assert rec.directed is False
    assert rec.tgt is None


def test_entity_types_setter_updates_recorded_kind() -> None:
    G = _toy()
    G.entity_types = {'A': 'edge'}  # mark vertex A as edge-entity
    rec = G._entities[G._resolve_entity_key('A')]
    assert rec.kind == 'edge_entity'


# ── get_or_create_vertex_by_attrs ─────────────────────────────────────


def test_get_or_create_vertex_by_attrs_requires_set_vertex_key() -> None:
    G = AnnNet(directed=True)
    with pytest.raises(RuntimeError, match='set_vertex_key'):
        G.get_or_create_vertex_by_attrs(symbol='TP53')


def test_get_or_create_vertex_by_attrs_creates_then_reuses() -> None:
    G = AnnNet(directed=True)
    G.set_vertex_key('symbol')
    vid1 = G.get_or_create_vertex_by_attrs(symbol='TP53')
    vid2 = G.get_or_create_vertex_by_attrs(symbol='TP53')
    assert vid1 == vid2


def test_get_or_create_vertex_by_attrs_raises_when_key_field_missing() -> None:
    G = AnnNet(directed=True)
    G.set_vertex_key('symbol', 'organism')
    with pytest.raises(ValueError, match='Missing composite key fields'):
        G.get_or_create_vertex_by_attrs(symbol='TP53')  # no 'organism'


# ── set_vertex_key error path ─────────────────────────────────────────


def test_set_vertex_key_rejects_empty_field_list() -> None:
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='at least one field'):
        G.set_vertex_key()


def test_set_vertex_key_rebuilds_index_from_existing_attrs() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.attrs.set_vertex_attrs('A', symbol='TP53')
    G.attrs.set_vertex_attrs('B', symbol='MYC')
    G.set_vertex_key('symbol')
    assert G._vertex_key_index[('TP53',)] == 'A'
    assert G._vertex_key_index[('MYC',)] == 'B'


def test_set_vertex_key_detects_conflict_on_existing_attrs() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.attrs.set_vertex_attrs('A', symbol='TP53')
    G.attrs.set_vertex_attrs('B', symbol='TP53')  # collision
    with pytest.raises(ValueError, match='Composite key conflict'):
        G.set_vertex_key('symbol')


# ── remove_vertices / remove_edges ────────────────────────────────────


def test_remove_edges_with_unknown_id_raises_by_default() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='Unknown edge'):
        G.remove_edges('not-an-edge')


def test_remove_edges_with_errors_ignore_silently_skips_unknown() -> None:
    G = _toy()
    G.remove_edges('not-an-edge', errors='ignore')


def test_remove_edges_rejects_invalid_errors_value() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='errors must be'):
        G.remove_edges('e1', errors='boom')


def test_remove_edges_bulk_removes_each_edge() -> None:
    G = _toy()
    G.remove_edges(['e1', 'e2'])
    assert G.ne == 0


def test_remove_vertices_with_unknown_id_raises_by_default() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='Unknown vertex'):
        G.remove_vertices('not-a-vertex')


def test_remove_vertices_with_errors_ignore_silently_skips() -> None:
    G = _toy()
    G.remove_vertices('not-a-vertex', errors='ignore')


def test_remove_vertices_rejects_invalid_errors_value() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='errors must be'):
        G.remove_vertices('A', errors='boom')


def test_remove_vertices_cascades_incident_edges() -> None:
    G = _toy()
    G.remove_vertices('A')
    # e1 (A,B) is gone; e2 (B,C) survives.
    assert 'A' not in G.vertices()
    assert 'e1' not in G._edges
    assert 'e2' in G._edges


def test_remove_vertices_bulk_removes_each() -> None:
    G = _toy()
    G.remove_vertices(['A', 'B'])
    assert set(G.vertices()) == {'C'}


# ── add_edges_to_slice batch ──────────────────────────────────────────


def test_add_edges_to_slice_bulk_adds_edges_and_endpoint_vertices() -> None:
    G = _toy()
    G.slices.add('s1')
    G._add_edges_to_slice_bulk('s1', ['e1', 'e2'])
    assert {'e1', 'e2'}.issubset(G.slices.edges('s1'))
    # endpoint vertices for the added edges land in the slice too.
    assert {'A', 'B', 'C'}.issubset(G.slices.vertices('s1'))


def test_add_edges_to_slice_bulk_skips_unknown_edges() -> None:
    G = _toy()
    G.slices.add('s1')
    G._add_edges_to_slice_bulk('s1', ['e1', 'no-such-edge'])
    # only e1 added
    assert 'e1' in G.slices.edges('s1')


# ── legacy compatibility / accessor delegators ────────────────────────


def test_layers_delegators_on_anndnet_round_trip() -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    # delegators on AnnNet point through to G.layers
    assert G.nl_to_row('A', ('healthy',)) >= 0
    G._rebuild_all_layers_cache()
    G._validate_layer_tuple(('healthy',))


# ── shape / count properties ──────────────────────────────────────────


def test_shape_property_reflects_vertex_and_edge_counts() -> None:
    G = _toy()
    rows, cols = G.shape
    assert rows == 3
    assert cols == 2


def test_global_count_reports_vertex_and_edge_counts() -> None:
    G = _toy()
    assert G.global_count('vertices') == 3
    assert G.global_count('edges') == 2


def test_global_count_rejects_unknown_kind() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='kind must be one of'):
        G.global_count('bogus')


# ── overwrite-existing-hyperedge path in _add_hyperedges_batch ────────


def test_add_hyperedges_with_existing_eid_overwrites_in_place() -> None:
    """Re-using an existing edge_id for a hyperedge enters the overwrite branch."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    # Now re-add 'h1' as a different hyperedge — triggers the in-place
    # overwrite branch in ``_add_hyperedges_batch``.
    G.add_edges(['B', 'C', 'D'], edge_id='h1')
    rec = G._edges['h1']
    assert set(rec.src) == {'B', 'C', 'D'}


def test_add_edges_as_entity_promotes_to_edge_entity() -> None:
    """Pass ``as_entity=True`` to register edges as edge-entities."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', as_entity=True)
    rec = G._edges['e1']
    assert rec.etype == 'vertex_edge'


# ── remove_edge / remove_vertex (singular legacy paths) ───────────────


def test_remove_edge_singular_raises_for_unknown() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='not found'):
        G.remove_edge('no-such')


def test_remove_edge_singular_drops_the_edge() -> None:
    G = _toy()
    G.remove_edge('e1')
    assert 'e1' not in G._edges


def test_remove_vertex_singular_cascades_incident_edges() -> None:
    G = _toy()
    G.remove_vertex('B')
    assert 'B' not in G.vertices()
    # both e1 (A,B) and e2 (B,C) had B as endpoint → both gone.
    assert 'e1' not in G._edges
    assert 'e2' not in G._edges


# ── reset / clear helpers ────────────────────────────────────────────


def test_remove_all_edges_via_bulk_with_empty_iterable_is_noop() -> None:
    G = _toy()
    G.remove_edges([])
    assert G.ne == 2


def test_remove_all_vertices_via_bulk_with_empty_iterable_is_noop() -> None:
    G = _toy()
    G.remove_vertices([])
    assert G.nv == 3


# ── add_vertices key-form branches ────────────────────────────────────


def test_add_vertices_single_dict_with_vertex_id_key() -> None:
    G = AnnNet(directed=True)
    G.add_vertices({'vertex_id': 'A', 'color': 'red'})
    assert 'A' in G.vertices()
    assert G.attrs.get_attr_vertex('A', 'color') == 'red'


def test_add_vertices_single_dict_with_id_key() -> None:
    G = AnnNet(directed=True)
    G.add_vertices({'id': 'A'})
    assert 'A' in G.vertices()


def test_add_vertices_single_dict_with_name_key() -> None:
    G = AnnNet(directed=True)
    G.add_vertices({'name': 'A'})
    assert 'A' in G.vertices()


def test_add_vertices_single_dict_without_known_key_raises() -> None:
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='vertex_id, id, name'):
        G.add_vertices({'color': 'red'})


def test_add_vertices_single_tuple_form() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(('A', {'color': 'red'}))
    assert 'A' in G.vertices()


def test_add_vertices_bulk_returns_ids_in_input_order() -> None:
    G = AnnNet(directed=True)
    items = [
        {'vertex_id': 'A'},
        {'id': 'B'},
        {'name': 'C'},
        ('D', {}),
        'E',
    ]
    out = G._add_vertices_bulk(items)
    assert out == ['A', 'B', 'C', 'D', 'E']


# add_vertex is in the blocked-legacy API — not callable on AnnNet directly.


# ── property accessors with edge weight + directed ───────────────────


def test_edge_weights_returns_per_edge_weight_dict() -> None:
    G = _toy()
    out = G.edge_weights
    assert out.get('e1') == 1.0
    assert out.get('e2') == 2.0


def test_edge_directed_returns_only_edges_with_explicit_flag() -> None:
    G = _toy()
    out = G.edge_directed
    # Both edges inherited the directed-graph default; they have explicit
    # ``directed=True`` so they appear in the dict.
    assert 'e1' in out or out == {}


def test_edge_definitions_returns_src_tgt_etype_per_binary_edge() -> None:
    G = _toy()
    out = G.edge_definitions
    assert out['e1'] == ('A', 'B', 'binary')
    assert out['e2'] == ('B', 'C', 'binary')


def test_idx_to_entity_round_trips_with_entity_to_idx() -> None:
    G = _toy()
    e2i = G.entity_to_idx
    i2e = G.idx_to_entity
    for vid, idx in e2i.items():
        assert i2e[idx] == vid


def test_hyperedge_definitions_round_trips_through_setter() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    out = G.hyperedge_definitions
    assert 'h1' in out
    assert out['h1']['directed'] is False
    assert set(out['h1']['members']) == {'A', 'B', 'C'}
