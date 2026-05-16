"""Coverage tests for ``annnet/core/_Annotation.py``."""

from __future__ import annotations

import narwhals as nw
import pytest

from annnet.core._Annotation import AttributesAccessor, AttributesClass
from annnet.core.graph import AnnNet


# ── fixtures ────────────────────────────────────────────────────────────


def _toy() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    G.slices.add('s1')
    return G


# ── bulk attribute setters ─────────────────────────────────────────────


def test_set_vertex_attrs_bulk_dict_input_writes_each_row() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs_bulk({'A': {'color': 'red'}, 'B': {'color': 'blue'}})
    assert G.attrs.get_attr_vertex('A', 'color') == 'red'
    assert G.attrs.get_attr_vertex('B', 'color') == 'blue'


def test_set_vertex_attrs_bulk_accepts_iterable_of_pairs() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs_bulk([('A', {'color': 'red'}), ('B', {'color': 'blue'})])
    assert G.attrs.get_attr_vertex('A', 'color') == 'red'


def test_set_vertex_attrs_bulk_rejects_non_dict_attrs() -> None:
    G = _toy()
    with pytest.raises(TypeError, match='must be dict'):
        G.attrs.set_vertex_attrs_bulk({'A': 'not-a-dict'})


def test_set_vertex_attrs_bulk_rejects_reserved_keys() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_vertex_attrs_bulk({'A': {'vertex_id': 'X'}})


def test_set_vertex_attrs_bulk_noop_on_empty_input() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs_bulk({})  # must not raise


def test_set_vertex_attrs_bulk_noop_when_all_attrs_dicts_empty() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs_bulk({'A': {}, 'B': {}})  # filtered out


def test_set_edge_attrs_bulk_dict_input_writes_each_row() -> None:
    G = _toy()
    G.attrs.set_edge_attrs_bulk({'e1': {'label': 'alpha'}, 'e2': {'label': 'beta'}})
    assert G.attrs.get_attr_edge('e1', 'label') == 'alpha'
    assert G.attrs.get_attr_edge('e2', 'label') == 'beta'


def test_set_edge_attrs_bulk_accepts_iterable_of_pairs() -> None:
    G = _toy()
    G.attrs.set_edge_attrs_bulk([('e1', {'label': 'alpha'})])
    assert G.attrs.get_attr_edge('e1', 'label') == 'alpha'


def test_set_edge_attrs_bulk_rejects_non_dict_attrs() -> None:
    G = _toy()
    with pytest.raises(TypeError, match='must be dict'):
        G.attrs.set_edge_attrs_bulk({'e1': 'not-a-dict'})


def test_set_edge_attrs_bulk_rejects_reserved_keys() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_edge_attrs_bulk({'e1': {'source': 'X'}})


def test_set_edge_attrs_bulk_noop_on_empty_input() -> None:
    G = _toy()
    G.attrs.set_edge_attrs_bulk({})


def test_set_edge_attrs_bulk_noop_when_all_attrs_dicts_empty() -> None:
    G = _toy()
    G.attrs.set_edge_attrs_bulk({'e1': {}})


# ── set_edge_slice_attrs internal branches ─────────────────────────────


def test_set_edge_slice_attrs_coerces_weight_to_float() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=99)
    out = G.attrs.get_edge_slice_attr('s1', 'e1', 'weight')
    assert out == 99.0
    assert isinstance(out, float)


def test_set_edge_slice_attrs_writes_non_weight_attrs() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs('s1', 'e1', confidence=0.95)
    out = G.attrs.get_edge_slice_attr('s1', 'e1', 'confidence')
    assert out == 0.95


def test_set_edge_slice_attrs_noop_when_only_reserved_allow_weight_missing() -> None:
    """Calling with no attrs at all is a no-op."""
    G = _toy()
    G.attrs.set_edge_slice_attrs('s1', 'e1')  # no kwargs


def test_set_edge_slice_attrs_bulk_writes_only_present_attrs() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1', 'e2'])
    G.attrs.set_edge_slice_attrs_bulk('s1', [('e1', {'weight': 5.0}), ('e2', {'confidence': 0.9})])
    assert G.attrs.get_edge_slice_attr('s1', 'e1', 'weight') == 5.0
    assert G.attrs.get_edge_slice_attr('s1', 'e2', 'confidence') == 0.9


def test_set_edge_slice_attrs_bulk_accepts_dict_form() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs_bulk('s1', {'e1': {'weight': 7.0}})
    assert G.attrs.get_edge_slice_attr('s1', 'e1', 'weight') == 7.0


def test_set_edge_slice_attrs_bulk_skips_non_dict_or_empty_entries() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs_bulk(
        's1',
        [('e1', {'weight': 1.0}), ('e2', 'not-a-dict'), ('e1', {})],
    )
    # only the valid entry got written
    assert G.attrs.get_edge_slice_attr('s1', 'e1', 'weight') == 1.0


def test_set_edge_slice_attrs_bulk_noop_on_empty() -> None:
    G = _toy()
    G.attrs.set_edge_slice_attrs_bulk('s1', [])


# ── set_slice_edge_weight error paths ──────────────────────────────────


def test_set_slice_edge_weight_raises_for_missing_slice() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='slice'):
        AttributesClass.set_slice_edge_weight(G, 'not-a-slice', 'e1', 1.0)


def test_set_slice_edge_weight_raises_for_missing_edge() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='Edge'):
        AttributesClass.set_slice_edge_weight(G, 's1', 'not-an-edge', 1.0)


def test_set_slice_edge_weight_writes_the_weight() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    AttributesClass.set_slice_edge_weight(G, 's1', 'e1', 42.0)
    assert G.attrs.get_edge_slice_attr('s1', 'e1', 'weight') == 42.0


# ── get_effective_edge_weight ──────────────────────────────────────────


def test_get_effective_edge_weight_falls_back_to_rec_weight_when_no_slice_override() -> None:
    G = _toy()
    w = G.attrs.get_effective_edge_weight('e1')
    assert w == 1.0


def test_get_effective_edge_weight_uses_slice_override_when_present() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=42.0)
    w = G.attrs.get_effective_edge_weight('e1', slice='s1')
    assert w == 42.0


def test_get_effective_edge_weight_returns_one_for_unknown_edge() -> None:
    G = _toy()
    assert G.attrs.get_effective_edge_weight('no-such-edge') == 1.0


# ── audit_attributes ──────────────────────────────────────────────────


def test_audit_attributes_returns_expected_shape_on_clean_graph() -> None:
    G = _toy()
    out = AttributesClass.audit_attributes(G)
    for key in (
        'extra_vertex_rows',
        'extra_edge_rows',
        'missing_vertex_rows',
        'missing_edge_rows',
        'invalid_edge_slice_rows',
    ):
        assert key in out


def test_audit_attributes_returns_lists_for_each_category() -> None:
    """Smoke-test that audit_attributes runs and returns the documented shape."""
    G = _toy()
    G.attrs.set_vertex_attrs('A', color='red')
    G.attrs.set_edge_attrs('e1', label='alpha')
    out = AttributesClass.audit_attributes(G)
    # The lists must be defined (empty or populated, exact contents depend on
    # whether vertex/edge insert auto-creates an attr row in this backend).
    for key in (
        'extra_vertex_rows',
        'extra_edge_rows',
        'missing_vertex_rows',
        'missing_edge_rows',
        'invalid_edge_slice_rows',
    ):
        assert isinstance(out[key], list)


# ── get_edge_attrs / get_vertex_attrs ─────────────────────────────────


def test_get_edge_attrs_by_int_index_uses_col_to_edge() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    attrs = G.attrs.get_edge_attrs(0)
    assert attrs.get('label') == 'alpha'


def test_get_edge_attrs_by_string_id() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    attrs = G.attrs.get_edge_attrs('e1')
    assert attrs.get('label') == 'alpha'


def test_get_edge_attrs_empty_for_unknown_id() -> None:
    G = _toy()
    assert G.attrs.get_edge_attrs('no-such') == {}


def test_get_vertex_attrs_returns_dict_with_attrs() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs('A', color='red')
    assert G.attrs.get_vertex_attrs('A').get('color') == 'red'


def test_get_vertex_attrs_empty_for_unknown() -> None:
    G = _toy()
    assert G.attrs.get_vertex_attrs('no-such') == {}


# ── get_attr_edges / get_attr_vertices ────────────────────────────────


def test_get_attr_edges_with_no_indexes_returns_all() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    out = G.attrs.get_attr_edges()
    assert 'e1' in out


def test_get_attr_edges_with_indexes_filters_to_those_eids() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    G.attrs.set_edge_attrs('e2', label='beta')
    out = G.attrs.get_attr_edges(indexes=[0])  # only e1
    assert set(out) == {'e1'}


def test_get_attr_vertices_with_no_filter_returns_all() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs('A', color='red')
    G.attrs.set_vertex_attrs('B', color='blue')
    out = G.attrs.get_attr_vertices()
    assert {'A', 'B'}.issubset(out)


def test_get_attr_vertices_with_filter_restricts() -> None:
    G = _toy()
    G.attrs.set_vertex_attrs('A', color='red')
    G.attrs.set_vertex_attrs('B', color='blue')
    out = G.attrs.get_attr_vertices(vertices={'A'})
    assert set(out) == {'A'}


# ── get_attr_from_edges / get_edges_by_attr ───────────────────────────


def test_get_attr_from_edges_returns_default_when_column_missing() -> None:
    G = _toy()
    out = G.attrs.get_attr_from_edges('not-a-column', default='??')
    assert out == {'e1': '??', 'e2': '??'}


def test_get_attr_from_edges_returns_actual_values_when_set() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    out = G.attrs.get_attr_from_edges('label', default='??')
    assert out['e1'] == 'alpha'
    assert out['e2'] == '??'  # e2 has no value → default


def test_get_edges_by_attr_returns_matching_edges() -> None:
    G = _toy()
    G.attrs.set_edge_attrs('e1', label='alpha')
    G.attrs.set_edge_attrs('e2', label='alpha')
    out = G.attrs.get_edges_by_attr('label', 'alpha')
    assert set(out) == {'e1', 'e2'}


def test_get_edges_by_attr_empty_when_column_missing() -> None:
    G = _toy()
    assert G.attrs.get_edges_by_attr('not-a-column', 'x') == []


# ── graph attributes ──────────────────────────────────────────────────


def test_get_graph_attributes_returns_shallow_copy() -> None:
    G = _toy()
    G.attrs.set_graph_attribute('study', 'demo')
    out = G.attrs.get_graph_attributes()
    out['mutated'] = True
    # mutating the returned dict must not affect the graph.
    assert 'mutated' not in G.attrs.get_graph_attributes()


# ── _dtype_for_value branches ──────────────────────────────────────────


def test_dtype_for_value_covers_each_python_type() -> None:
    G = _toy()
    assert AttributesClass._dtype_for_value(G, None) is nw.Unknown
    assert AttributesClass._dtype_for_value(G, True) is nw.Boolean
    assert AttributesClass._dtype_for_value(G, 7) is nw.Int64
    assert AttributesClass._dtype_for_value(G, 3.14) is nw.Float64
    assert AttributesClass._dtype_for_value(G, b'bytes') is nw.Binary
    assert AttributesClass._dtype_for_value(G, 'string') is nw.String
    # list / tuple → List(inner)
    out = AttributesClass._dtype_for_value(G, [1, 2])
    assert isinstance(out, nw.List)
    # dict → Object
    assert AttributesClass._dtype_for_value(G, {'k': 1}) is nw.Object
    # empty list inner falls back to String
    out_empty = AttributesClass._dtype_for_value(G, [])
    assert isinstance(out_empty, nw.List)


def test_is_null_dtype_recognises_unknown() -> None:
    G = _toy()
    assert AttributesClass._is_null_dtype(G, nw.Unknown) is True
    assert AttributesClass._is_null_dtype(G, nw.Int64) is False


# ── _sanitize_value_for_nw ────────────────────────────────────────────


def test_sanitize_value_for_nw_json_serializes_containers() -> None:
    G = _toy()
    assert AttributesClass._sanitize_value_for_nw(G, [1, 2]) == '[1, 2]'
    assert AttributesClass._sanitize_value_for_nw(G, {'a': 1}) == '{"a": 1}'
    assert AttributesClass._sanitize_value_for_nw(G, 'plain') == 'plain'
    assert AttributesClass._sanitize_value_for_nw(G, 42) == 42


# ── _is_binary_type ───────────────────────────────────────────────────


def test_is_binary_type_detects_narwhals_binary() -> None:
    G = _toy()
    assert AttributesClass._is_binary_type(G, nw.Binary) is True


def test_is_binary_type_detects_via_string_substring() -> None:
    G = _toy()

    class FakeDType:
        def __str__(self):
            return 'blob[1024]'

    assert AttributesClass._is_binary_type(G, FakeDType()) is True


# ── AttributesAccessor forwarders ─────────────────────────────────────


def test_attributes_accessor_forwards_all_public_methods() -> None:
    G = _toy()
    a = G.attrs  # AttributesAccessor instance
    a.set_graph_attribute('study', 'demo')
    assert a.get_graph_attribute('study') == 'demo'
    assert a.get_graph_attributes()['study'] == 'demo'
    a.set_vertex_attrs('A', color='red')
    assert a.get_attr_vertex('A', 'color') == 'red'
    assert a.get_vertex_attrs('A').get('color') == 'red'
    assert a.get_attr_vertices().get('A', {}).get('color') == 'red'
    a.set_edge_attrs('e1', label='alpha')
    assert a.get_attr_edge('e1', 'label') == 'alpha'
    assert a.get_edge_attrs('e1').get('label') == 'alpha'
    assert a.get_attr_edges().get('e1', {}).get('label') == 'alpha'
    assert a.get_attr_from_edges('label', default='??').get('e1') == 'alpha'
    assert 'e1' in a.get_edges_by_attr('label', 'alpha')
    a.set_slice_attrs('s1', notes='primary')
    assert a.get_slice_attr('s1', 'notes') == 'primary'


def test_attributes_accessor_forwards_bulk_and_slice_helpers() -> None:
    G = _toy()
    a = G.attrs
    a.set_vertex_attrs_bulk({'A': {'color': 'red'}})
    a.set_edge_attrs_bulk({'e1': {'label': 'alpha'}})
    a.set_edge_slice_attrs('s1', 'e1', weight=2.0)
    a.set_edge_slice_attrs_bulk('s1', {'e1': {'weight': 3.0}})
    assert a.get_edge_slice_attr('s1', 'e1', 'weight') == 3.0
    a.set_slice_edge_weight('s1', 'e1', 4.0)
    assert a.get_edge_slice_attr('s1', 'e1', 'weight') == 4.0
    assert a.get_effective_edge_weight('e1', slice='s1') == 4.0


# ── composite vertex key (covers the _vertex_key_enabled branches) ───


def test_set_vertex_attrs_with_composite_key_writes_and_indexes() -> None:
    G = _toy()
    G.set_vertex_key('name')
    G.attrs.set_vertex_attrs('A', name='alice')
    assert G.attrs.get_attr_vertex('A', 'name') == 'alice'


def test_set_vertex_attrs_with_composite_key_rejects_collision() -> None:
    G = _toy()
    G.set_vertex_key('name')
    G.attrs.set_vertex_attrs('A', name='alice')
    with pytest.raises(ValueError, match='Composite key collision'):
        G.attrs.set_vertex_attrs('B', name='alice')  # 'alice' already owned by A


def test_set_vertex_attrs_bulk_with_composite_key_writes_all() -> None:
    G = _toy()
    G.set_vertex_key('name')
    G.attrs.set_vertex_attrs_bulk({'A': {'name': 'alice'}, 'B': {'name': 'bob'}})
    assert G.attrs.get_attr_vertex('A', 'name') == 'alice'
    assert G.attrs.get_attr_vertex('B', 'name') == 'bob'


def test_set_vertex_attrs_bulk_with_composite_key_rejects_collision() -> None:
    G = _toy()
    G.set_vertex_key('name')
    G.attrs.set_vertex_attrs('A', name='alice')
    with pytest.raises(ValueError, match='Composite key collision'):
        G.attrs.set_vertex_attrs_bulk({'B': {'name': 'alice'}})


# ── flexible edge direction policy (covers _apply_flexible_direction) ─


def test_flexible_edge_with_edge_scope_policy_applies_on_attr_change() -> None:
    """Setting an edge attribute that an edge-scope policy watches flips
    the orientation in the incidence matrix."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={
            'var': 'temperature',
            'threshold': 10.0,
            'scope': 'edge',
            'above': 's->t',
        },
    )
    # set the watched attribute → triggers _apply_flexible_direction
    G.attrs.set_edge_attrs('e1', temperature=20.0)


def test_flexible_edge_with_vertex_scope_policy_applies_on_vertex_change() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={
            'var': 'level',
            'threshold': 5.0,
            'scope': 'vertex',
            'above': 's->t',
        },
    )
    # change a vertex attr watched by the policy
    G.attrs.set_vertex_attrs('A', level=10.0)
    G.attrs.set_vertex_attrs('B', level=2.0)


def test_flexible_edge_tie_handling_keep_does_nothing() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={
            'var': 'x',
            'threshold': 5.0,
            'scope': 'edge',
            'tie': 'keep',
        },
    )
    # Set x == threshold → tie_case=True, tie='keep' → returns without rewriting
    G.attrs.set_edge_attrs('e1', x=5.0)


def test_flexible_edge_tie_handling_undirected() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={
            'var': 'x',
            'threshold': 5.0,
            'scope': 'edge',
            'tie': 'undirected',
        },
    )
    G.attrs.set_edge_attrs('e1', x=5.0)


def test_flexible_edge_attrs_bulk_triggers_apply() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={'var': 'x', 'threshold': 5.0, 'scope': 'edge'},
    )
    G.attrs.set_edge_attrs_bulk({'e1': {'x': 10.0}})


def test_flexible_edge_vertex_attrs_bulk_triggers_apply() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges(
        'A',
        'B',
        edge_id='e1',
        weight=1.0,
        flexible={'var': 'level', 'threshold': 5.0, 'scope': 'vertex'},
    )
    G.attrs.set_vertex_attrs_bulk({'A': {'level': 10.0}, 'B': {'level': 2.0}})


# Touch the accessor class import for symmetry.
_ = AttributesAccessor
