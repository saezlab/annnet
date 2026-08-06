"""Coverage tests for the row helpers and the adapter view of a graph."""

from __future__ import annotations

from enum import Enum

import narwhals as nw

from annnet._support.graph_records import (
    _attrs_to_dict,
    _rows_like,
    _rows_to_df,
    _serialize_value,
)
from annnet.core._structure import (
    _is_directed_eid,
    _iter_edge_records,
    _iter_node_ids,
)
from annnet.core.graph import AnnNet


# ── _is_directed_eid ──────────────────────────────────────────────────


def test_is_directed_eid_reads_the_directedness_of_the_edge() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    assert _is_directed_eid(G, 'e1') is True


def test_is_directed_eid_falls_back_to_true_for_unknown_eid() -> None:
    G = AnnNet(directed=True)
    # No edge with this id — neither the store nor the attributes answer.
    assert _is_directed_eid(G, 'no-such-edge') is True


def test_is_directed_eid_handles_an_object_that_holds_no_graph() -> None:
    class Bare:
        pass

    assert _is_directed_eid(Bare(), 'anything') is True


# ── _iter_node_ids ───────────────────────────────────────────────────


def test_iter_node_ids_yields_only_node_entities_in_row_order() -> None:
    G = AnnNet(directed=False)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1')
    out = list(_iter_node_ids(G))
    assert out == ['A', 'B', 'C']


def test_iter_node_ids_falls_back_to_nodes_method_when_there_is_no_store() -> None:
    class Stub:
        def nodes(self):
            return ['x', 'y']

    assert list(_iter_node_ids(Stub())) == ['x', 'y']


def test_iter_node_ids_raises_when_no_adapter_surface() -> None:
    class Bare:
        pass

    import pytest

    with pytest.raises(AttributeError):
        list(_iter_node_ids(Bare()))


def test_iter_node_ids_skips_the_edge_entities() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    G.add_edges('a', 'b', edge_id='e1', as_entity=True)

    assert list(_iter_node_ids(G)) == ['a', 'b']


# ── _serialize_value ───────────────────────────────────────────────────


class _Color(Enum):
    RED = 'red'


def test_serialize_value_translates_enum_to_name() -> None:
    assert _serialize_value(_Color.RED) == 'RED'


def test_serialize_value_passes_through_plain_scalars() -> None:
    assert _serialize_value(7) == 7
    assert _serialize_value('s') == 's'


def test_serialize_value_unwraps_mapping_like_objects() -> None:
    out = _serialize_value({'a': 1})
    assert out == {'a': 1}


# ── _attrs_to_dict ─────────────────────────────────────────────────────


def test_attrs_to_dict_flattens_enums_and_nested_mappings() -> None:
    out = _attrs_to_dict(
        {
            'plain': 1,
            'color': _Color.RED,
            'nested': {'inner': _Color.RED, 'other': 2},
        }
    )
    assert out['plain'] == 1
    assert out['color'] == 'RED'
    assert out['nested'] == {'inner': 'RED', 'other': 2}


# ── _rows_like ─────────────────────────────────────────────────────────


def test_rows_like_returns_empty_for_none() -> None:
    assert _rows_like(None) == []


def test_rows_like_handles_dict_of_columns() -> None:
    out = _rows_like({'a': [1, 2], 'b': ['x', 'y']})
    assert out == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]


def test_rows_like_handles_list_of_dicts() -> None:
    rows = [{'a': 1}, {'a': 2}]
    assert _rows_like(rows) == rows


def test_rows_like_handles_cursor_like_object_with_fetchall() -> None:
    class Cursor:
        columns = ['a', 'b']

        def fetchall(self):
            return [(1, 'x'), (2, 'y')]

    out = _rows_like(Cursor())
    assert out == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]


def test_rows_like_returns_empty_for_unrecognized_shape() -> None:
    assert _rows_like(42) == []
    assert _rows_like(['not-a-dict']) == []


# ── _iter_edge_records ─────────────────────────────────────────────────


def test_iter_edge_records_yields_the_edges_in_column_order() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1')
    G.add_edges('B', 'C', edge_id='e2')
    out = list(_iter_edge_records(G))
    eids = [eid for eid, _ in out]
    assert eids == ['e1', 'e2']


def test_iter_edge_records_gives_each_edge_the_shape_an_adapter_reads() -> None:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h1'}])

    shapes = dict(_iter_edge_records(G))
    assert (shapes['e1'].src, shapes['e1'].tgt) == ('A', 'B')
    assert (shapes['e1'].etype, shapes['e1'].weight, shapes['e1'].directed) == ('binary', 2.0, True)
    assert shapes['h1'].src == frozenset({'A', 'B', 'C'})
    assert shapes['h1'].tgt is None
    assert (shapes['h1'].etype, shapes['h1'].directed) == ('hyper', False)


def test_iter_edge_records_raises_when_no_edge_store() -> None:
    class Bare:
        pass

    import pytest

    with pytest.raises(AttributeError):
        list(_iter_edge_records(Bare()))


# ── _rows_to_df ────────────────────────────────────────────────────────


def test_rows_to_df_preserves_first_seen_column_order() -> None:
    rows = [{'b': 1, 'a': 2}, {'b': 3, 'a': 4}]
    df = _rows_to_df(rows)
    cols = list(nw.from_native(df, eager_only=True).columns)
    assert cols == ['b', 'a']


def test_rows_to_df_returns_empty_dataframe_on_empty_input() -> None:
    df = _rows_to_df([])
    # whatever the backend is, it must be an empty table.
    assert nw.from_native(df, eager_only=True).shape[0] == 0
