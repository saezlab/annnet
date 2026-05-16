"""Coverage tests for ``annnet/core/_records.py``"""

from __future__ import annotations

import narwhals as nw

from annnet.core._records import (
    EdgeRecord,
    EdgeType,
    EdgeView,
    EntityRecord,
    SliceRecord,
    _df_filter_not_equal,
    _external_entity_kind,
    _get_numeric_supertype,
    _internal_entity_kind,
    build_dataframe_from_rows,
)


# ── _get_numeric_supertype ─────────────────────────────────────────────


def test_supertype_float64_and_float32_yields_float64() -> None:
    assert _get_numeric_supertype(nw.Float64, nw.Float32) is nw.Float64
    assert _get_numeric_supertype(nw.Float32, nw.Float64) is nw.Float64


def test_supertype_float32_and_float32_yields_float32() -> None:
    assert _get_numeric_supertype(nw.Float32, nw.Float32) is nw.Float32


def test_supertype_int_with_float_yields_a_float_type() -> None:
    out = _get_numeric_supertype(nw.Int32, nw.Float32)
    assert out is nw.Float32


def test_supertype_int_widening_picks_wider() -> None:
    assert _get_numeric_supertype(nw.Int8, nw.Int64) is nw.Int64
    assert _get_numeric_supertype(nw.Int64, nw.Int8) is nw.Int64


def test_supertype_signed_mixed_with_unsigned_fallbacks_to_float64() -> None:
    assert _get_numeric_supertype(nw.Int32, nw.UInt32) is nw.Float64


def test_supertype_two_unsigned_picks_wider() -> None:
    assert _get_numeric_supertype(nw.UInt8, nw.UInt32) is nw.UInt32


# ── build_dataframe_from_rows ──────────────────────────────────────────


def test_build_dataframe_from_rows_round_trips_simple_rows() -> None:
    rows = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    df = build_dataframe_from_rows(rows)
    # Doesn't matter what concrete backend the wrapper picks — just
    # that it's a 2-row tabular thing.
    back = nw.from_native(df, eager_only=True)
    assert back.shape[0] == 2


def test_df_filter_not_equal_drops_matching_rows() -> None:
    rows = [{'k': 1}, {'k': 2}, {'k': 1}]
    df = build_dataframe_from_rows(rows)
    out = _df_filter_not_equal(df, 'k', 1)
    back = nw.from_native(out, eager_only=True)
    assert back.shape[0] == 1


# ── EntityRecord / SliceRecord / EdgeRecord ────────────────────────────


def test_entity_record_holds_row_idx_and_kind() -> None:
    rec = EntityRecord(row_idx=3, kind='vertex')
    assert rec.row_idx == 3
    assert rec.kind == 'vertex'


def test_slice_record_dict_style_get_returns_field_value() -> None:
    sr = SliceRecord(vertices={'a'}, edges=set(), attributes={'k': 1})
    assert sr['vertices'] == {'a'}
    sr['edges'] = {'e1'}
    assert sr.edges == {'e1'}
    # default-returning .get path
    assert sr.get('vertices') == {'a'}
    assert sr.get('nonexistent', 'fallback') == 'fallback'


def test_edge_record_constructs_with_named_fields() -> None:
    rec = EdgeRecord(
        src='a',
        tgt='b',
        weight=1.0,
        directed=True,
        etype='binary',
        col_idx=0,
        ml_kind=None,
        ml_layers=None,
        direction_policy=None,
    )
    assert rec.src == 'a'
    assert rec.etype == 'binary'


# ── EdgeType ───────────────────────────────────────────────────────────


def test_edge_type_enum_members() -> None:
    assert EdgeType.DIRECTED.value == 'DIRECTED'
    assert EdgeType.UNDIRECTED.value == 'UNDIRECTED'


# ── EdgeView ───────────────────────────────────────────────────────────


def test_edge_view_unpacks_as_tuple_and_exposes_attrs() -> None:
    v = EdgeView(
        'A',
        'B',
        edge_id='e1',
        kind='binary',
        members=frozenset({'A', 'B'}),
        weight=2.5,
        directed=True,
    )
    s, t = v
    assert s == 'A'
    assert t == 'B'
    assert v.edge_id == 'e1'
    assert v.source == 'A'
    assert v.target == 'B'
    assert v.weight == 2.5
    assert v.directed is True


def test_edge_view_repr_contains_every_named_field() -> None:
    v = EdgeView(
        'X',
        'Y',
        edge_id='e42',
        kind='hyper_directed',
        members=frozenset({'X', 'Y'}),
        weight=0.0,
        directed=False,
    )
    r = repr(v)
    for needle in ('e42', 'hyper_directed', "'X'", "'Y'", '0.0', 'False'):
        assert needle in r


# ── kind translation helpers ───────────────────────────────────────────


def test_external_entity_kind_translates_edge_entity_to_edge() -> None:
    assert _external_entity_kind('edge_entity') == 'edge'
    assert _external_entity_kind('vertex') == 'vertex'


def test_internal_entity_kind_translates_edge_to_edge_entity() -> None:
    assert _internal_entity_kind('edge') == 'edge_entity'
    assert _internal_entity_kind('vertex') == 'vertex'
