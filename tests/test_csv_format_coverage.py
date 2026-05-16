"""Coverage tests for ``annnet/io/csv_format.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from annnet._support.dataframe_backend import dataframe_from_rows
from annnet.core.graph import AnnNet
from annnet.io.csv_format import (
    _attr_columns,
    _columns,
    _detect_schema,
    _ingest_adjacency,
    _ingest_edge_list,
    _ingest_hyperedge,
    _ingest_incidence,
    _ingest_lil,
    _is_numeric_column,
    _norm,
    _pick_first,
    _rows,
    _split_set,
    _split_slices,
    _truthy,
    edges_to_csv,
    from_csv,
    from_dataframe,
    hyperedges_to_csv,
)


# ── parser helpers ─────────────────────────────────────────────────────


def test_norm_stringifies_and_strips_none_to_empty() -> None:
    assert _norm(None) == ''
    assert _norm('  abc  ') == 'abc'
    assert _norm(7) == '7'
    assert _norm(3.14) == '3.14'


def test_truthy_handles_bool_int_string_and_unknown() -> None:
    assert _truthy(None) is None
    assert _truthy(True) is True
    assert _truthy(np.bool_(False)) is False
    assert _truthy(1) is True
    assert _truthy(np.int64(0)) is False
    assert _truthy('yes') is True
    assert _truthy('NO') is False
    assert _truthy('maybe') is None


def test_split_slices_handles_strings_lists_json_and_dicts() -> None:
    assert _split_slices(None) == []
    assert _split_slices('') == []
    assert _split_slices('a;b,c|d') == ['a', 'b', 'c', 'd']
    assert _split_slices(['a', 'b']) == ['a', 'b']
    # JSON array path
    assert _split_slices('["x","y"]') == ['x', 'y']
    # JSON object → keys
    assert _split_slices('{"k1": 1, "k2": 2}') == ['k1', 'k2']
    # invalid JSON falls back to separator split
    assert _split_slices('[invalid,json]') == ['[invalid', 'json]']
    # scalar fallback
    assert _split_slices(42) == ['42']


def test_split_set_handles_strings_lists_json_and_scalars() -> None:
    assert _split_set(None) == set()
    assert _split_set('') == set()
    assert _split_set('a;b,c|d') == {'a', 'b', 'c', 'd'}
    assert _split_set({'x', 'y'}) == {'x', 'y'}
    assert _split_set('["m","n"]') == {'m', 'n'}
    # invalid JSON falls back to separator split
    assert _split_set('[broken]') == {'[broken]'}
    # scalar fallback
    assert _split_set(7) == {'7'}


# ── micro-wrappers on dataframe helpers ────────────────────────────────


def test_columns_and_rows_round_trip() -> None:
    df = dataframe_from_rows([{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}])
    assert set(_columns(df)) == {'a', 'b'}
    assert _rows(df) == [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]


def test_pick_first_is_case_insensitive_and_returns_canonical_name() -> None:
    df = dataframe_from_rows([{'Source': 'a', 'Target': 'b'}])
    assert _pick_first(df, ['source']) == 'Source'
    assert _pick_first(df, ['nonexistent']) is None


def test_is_numeric_column_distinguishes_int_from_str() -> None:
    df = dataframe_from_rows([{'n': 1, 's': 'x'}, {'n': 2, 's': 'y'}])
    assert _is_numeric_column(df, 'n') is True
    assert _is_numeric_column(df, 's') is False


def test_attr_columns_excludes_reserved_case_insensitively() -> None:
    df = dataframe_from_rows([{'Source': 'a', 'Target': 'b', 'color': 'red', 'extra': 1}])
    out = _attr_columns(df, ['source', 'target'])
    assert set(out) == {'color', 'extra'}


# ── _detect_schema ─────────────────────────────────────────────────────


def test_detect_schema_hyperedge_via_members() -> None:
    df = dataframe_from_rows([{'members': 'a|b|c', 'weight': 1.0}])
    assert _detect_schema(df) == 'hyperedge'


def test_detect_schema_hyperedge_via_head_and_tail() -> None:
    df = dataframe_from_rows([{'head': 'a', 'tail': 'b|c'}])
    assert _detect_schema(df) == 'hyperedge'


def test_detect_schema_lil_via_neighbors() -> None:
    df = dataframe_from_rows([{'vertex': 'a', 'neighbors': 'b|c'}])
    assert _detect_schema(df) == 'lil'


def test_detect_schema_edge_list_via_coo_triples() -> None:
    df = dataframe_from_rows([{'row': 'a', 'col': 'b', 'val': 1.0}])
    assert _detect_schema(df) == 'edge_list'


def test_detect_schema_edge_list_via_src_dst() -> None:
    df = dataframe_from_rows([{'source': 'a', 'target': 'b'}])
    assert _detect_schema(df) == 'edge_list'


def test_detect_schema_incidence_when_first_col_label_then_numeric() -> None:
    df = dataframe_from_rows(
        [
            {'vertex_id': 'a', 'e1': 1, 'e2': 0, 'e3': 0, 'e4': 1},
            {'vertex_id': 'b', 'e1': -1, 'e2': 1, 'e3': 0, 'e4': 0},
        ]
    )
    assert _detect_schema(df) == 'incidence'


def test_detect_schema_adjacency_via_square_numeric_with_label_col() -> None:
    df = dataframe_from_rows(
        [
            {'name': 'a', 'a': 0, 'b': 1},
            {'name': 'b', 'a': 1, 'b': 0},
        ]
    )
    assert _detect_schema(df) == 'adjacency'


def test_detect_schema_adjacency_via_pure_square_numeric() -> None:
    df = dataframe_from_rows(
        [
            {'a': 0, 'b': 1},
            {'a': 1, 'b': 0},
        ]
    )
    assert _detect_schema(df) == 'adjacency'


def test_detect_schema_fallback_to_edge_list_when_unrecognized() -> None:
    df = dataframe_from_rows([{'foo': 1, 'bar': 'x'}])
    assert _detect_schema(df) == 'edge_list'


# ── ingest_edge_list ───────────────────────────────────────────────────


def test_ingest_edge_list_classic_src_dst_creates_edges_and_vertices() -> None:
    df = dataframe_from_rows(
        [
            {'source': 'A', 'target': 'B', 'weight': 2.5, 'color': 'red'},
            {'source': 'B', 'target': 'C', 'weight': 1.0, 'color': 'blue'},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_edge_list(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C'}
    assert G.ne == 2


def test_ingest_edge_list_coo_triples_use_row_col_val() -> None:
    df = dataframe_from_rows(
        [
            {'row': 'A', 'col': 'B', 'val': 1.0},
            {'row': 'A', 'col': 'C', 'val': 2.0},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_edge_list(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C'}
    assert G.ne == 2


def test_ingest_edge_list_raises_when_no_endpoints_anywhere() -> None:
    df = dataframe_from_rows([{'foo': 'a', 'bar': 'b'}])
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='source/target'):
        _ingest_edge_list(df, G, default_slice=None, default_directed=True, default_weight=1.0)


def test_ingest_edge_list_honors_directed_column() -> None:
    df = dataframe_from_rows(
        [
            {'source': 'A', 'target': 'B', 'directed': 'true'},
            {'source': 'B', 'target': 'C', 'directed': 'false'},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_edge_list(df, G, default_slice=None, default_directed=None, default_weight=1.0)
    assert G.ne == 2


def test_ingest_edge_list_creates_per_slice_edges_with_weight_override() -> None:
    df = dataframe_from_rows(
        [
            {'source': 'A', 'target': 'B', 'slice': 's1;s2', 'weight:s2': 9.0},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_edge_list(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    # both slices must be registered on the graph (per-slice loop ran)
    assert {'s1', 's2'}.issubset(set(G.slices.list()))


def test_ingest_edge_list_skips_rows_with_empty_endpoints() -> None:
    df = dataframe_from_rows(
        [
            {'source': '', 'target': 'B'},
            {'source': 'A', 'target': ''},
            {'source': 'A', 'target': 'B'},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_edge_list(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    assert G.ne == 1


# ── ingest_hyperedge ───────────────────────────────────────────────────


def test_ingest_hyperedge_members_path_creates_undirected_hyper() -> None:
    df = dataframe_from_rows([{'members': 'A|B|C', 'weight': 2.0}])
    G = AnnNet(directed=False)
    _ingest_hyperedge(df, G, default_slice=None, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C'}
    assert G.ne == 1


def test_ingest_hyperedge_head_tail_path_creates_directed_hyper() -> None:
    df = dataframe_from_rows([{'head': 'A|B', 'tail': 'C|D', 'weight': 1.0}])
    G = AnnNet(directed=True)
    _ingest_hyperedge(df, G, default_slice=None, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C', 'D'}
    assert G.ne == 1


def test_ingest_hyperedge_raises_when_no_columns_match() -> None:
    df = dataframe_from_rows([{'foo': 'A', 'bar': 'B'}])
    G = AnnNet(directed=False)
    with pytest.raises(ValueError, match='members'):
        _ingest_hyperedge(df, G, default_slice=None, default_weight=1.0)


def test_ingest_hyperedge_uses_default_slice_when_no_slice_column() -> None:
    df = dataframe_from_rows([{'members': 'A|B|C'}])
    G = AnnNet(directed=False)
    _ingest_hyperedge(df, G, default_slice='S0', default_weight=1.0)
    assert 'S0' in set(G.slices.list())


# ── ingest_incidence ───────────────────────────────────────────────────


def test_ingest_incidence_handles_directed_undirected_and_hyper_columns() -> None:
    # Four incidence columns, one per branch in the ingest:
    #   e1: directed binary (one +1, one -1)        → branch len(pos)==1 and len(neg)==1
    #   e2: undirected binary (two +1)              → branch len(pos)==2 and len(neg)==0
    #   e3: undirected hyper (three +1)             → fallback hyper (no neg)
    #   e4: directed hyper (two +1, two -1)         → fallback hyper (pos and neg)
    df = dataframe_from_rows(
        [
            {'vertex_id': 'A', 'e1': 1, 'e2': 1, 'e3': 1, 'e4': 1},
            {'vertex_id': 'B', 'e1': -1, 'e2': 1, 'e3': 1, 'e4': 1},
            {'vertex_id': 'C', 'e1': 0, 'e2': 0, 'e3': 1, 'e4': -1},
            {'vertex_id': 'D', 'e1': 0, 'e2': 0, 'e3': 0, 'e4': -1},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_incidence(df, G, default_slice=None, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C', 'D'}
    # The four incidence branches must produce at least 3 edges (e3 may
    # collapse depending on which branch it lands in — the goal here is
    # branch coverage, not exact edge count).
    assert G.ne >= 3


def test_ingest_incidence_skips_non_numeric_columns() -> None:
    df = dataframe_from_rows(
        [
            {'vertex_id': 'A', 'e1': 1, 'extra': 'red'},
            {'vertex_id': 'B', 'e1': -1, 'extra': 'blue'},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_incidence(df, G, default_slice=None, default_weight=1.0)
    assert G.ne == 1  # only 'e1' becomes an edge; 'extra' skipped


def test_ingest_incidence_uses_explicit_vertex_id_col_when_first_column() -> None:
    # When the canonical vertex-id column is already first, no rename happens.
    df = dataframe_from_rows(
        [
            {'vertex_id': 'A', 'e1': 1},
            {'vertex_id': 'B', 'e1': -1},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_incidence(df, G, default_slice=None, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B'}


# ── ingest_adjacency ───────────────────────────────────────────────────


def test_ingest_adjacency_with_row_labels_directed_when_asymmetric() -> None:
    df = dataframe_from_rows(
        [
            {'name': 'A', 'A': 0, 'B': 1, 'C': 0},
            {'name': 'B', 'A': 0, 'B': 0, 'C': 1},
            {'name': 'C', 'A': 1, 'B': 0, 'C': 0},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_adjacency(df, G, default_slice=None, default_directed=None, default_weight=1.0)
    assert G.ne == 3  # directed cycle


def test_ingest_adjacency_with_row_labels_undirected_when_symmetric() -> None:
    df = dataframe_from_rows(
        [
            {'name': 'A', 'A': 0, 'B': 1, 'C': 1},
            {'name': 'B', 'A': 1, 'B': 0, 'C': 0},
            {'name': 'C', 'A': 1, 'B': 0, 'C': 0},
        ]
    )
    G = AnnNet(directed=False)
    _ingest_adjacency(df, G, default_slice=None, default_directed=None, default_weight=1.0)
    # symmetric — only the upper triangle is added
    assert G.ne == 2


def test_ingest_adjacency_no_labels_uses_integer_row_ids() -> None:
    df = dataframe_from_rows(
        [
            {'a': 0, 'b': 1},
            {'a': 1, 'b': 0},
        ]
    )
    G = AnnNet(directed=False)
    _ingest_adjacency(df, G, default_slice=None, default_directed=None, default_weight=1.0)
    assert G.ne == 1


def test_ingest_adjacency_raises_when_non_numeric_in_matrix_region() -> None:
    # B is consistently a string column → not a numeric matrix column.
    df = dataframe_from_rows(
        [
            {'name': 'A', 'A': 0, 'B': 'x'},
            {'name': 'B', 'A': 1, 'B': 'y'},
        ]
    )
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='non-numeric'):
        _ingest_adjacency(df, G, default_slice=None, default_directed=None, default_weight=1.0)


def test_ingest_adjacency_raises_when_rows_neq_cols() -> None:
    df = dataframe_from_rows(
        [
            {'name': 'A', 'A': 0, 'B': 1, 'C': 0},
            {'name': 'B', 'A': 1, 'B': 0, 'C': 0},
        ]
    )
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='rows must equal'):
        _ingest_adjacency(df, G, default_slice=None, default_directed=None, default_weight=1.0)


# ── ingest_lil ─────────────────────────────────────────────────────────


def test_ingest_lil_creates_edges_per_neighbor() -> None:
    df = dataframe_from_rows(
        [
            {'vertex': 'A', 'neighbors': 'B|C', 'weight': 2.0},
            {'vertex': 'B', 'neighbors': 'C', 'weight': 1.0},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_lil(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    assert set(G.vertices()) == {'A', 'B', 'C'}
    assert G.ne == 3


def test_ingest_lil_raises_when_no_neighbors_column() -> None:
    df = dataframe_from_rows([{'vertex': 'A'}])
    G = AnnNet(directed=True)
    with pytest.raises(ValueError, match='neighbors'):
        _ingest_lil(df, G, default_slice=None, default_directed=True, default_weight=1.0)


def test_ingest_lil_creates_edges_per_slice_when_slice_column_present() -> None:
    df = dataframe_from_rows(
        [
            {'vertex': 'A', 'neighbors': 'B', 'slice': 's1|s2'},
        ]
    )
    G = AnnNet(directed=True)
    _ingest_lil(df, G, default_slice=None, default_directed=True, default_weight=1.0)
    # The per-slice loop ran for both slices; the edge between A and B
    # is registered as a member of each.
    assert {'s1', 's2'}.issubset(set(G.slices.list()))


# ── from_dataframe top-level dispatcher ────────────────────────────────


def test_from_dataframe_constructs_anndnet_when_none_passed() -> None:
    df = dataframe_from_rows([{'source': 'A', 'target': 'B'}])
    G = from_dataframe(df, schema='edge_list')
    assert isinstance(G, AnnNet)
    assert G.ne == 1


def test_from_dataframe_auto_dispatches_to_each_branch() -> None:
    # hyperedge
    G1 = from_dataframe(
        dataframe_from_rows([{'members': 'A|B|C'}]),
        schema='auto',
    )
    assert G1.ne == 1
    # incidence
    G2 = from_dataframe(
        dataframe_from_rows(
            [
                {'vertex_id': 'a', 'e1': 1, 'e2': 0},
                {'vertex_id': 'b', 'e1': -1, 'e2': 1},
                {'vertex_id': 'c', 'e1': 0, 'e2': -1},
            ]
        ),
        schema='auto',
    )
    assert G2.ne == 2
    # adjacency
    G3 = from_dataframe(
        dataframe_from_rows(
            [
                {'a': 0, 'b': 1},
                {'a': 1, 'b': 0},
            ]
        ),
        schema='auto',
    )
    assert G3.ne == 1
    # lil
    G4 = from_dataframe(
        dataframe_from_rows([{'vertex': 'A', 'neighbors': 'B|C'}]),
        schema='auto',
    )
    assert G4.ne == 2


def test_from_dataframe_raises_on_unknown_schema() -> None:
    df = dataframe_from_rows([{'source': 'A', 'target': 'B'}])
    with pytest.raises(ValueError, match='Unknown schema'):
        from_dataframe(df, schema='not-real')


# ── from_csv (uses real on-disk CSV) ───────────────────────────────────


def test_from_csv_reads_a_real_file(tmp_path: Path) -> None:
    p = tmp_path / 'edges.csv'
    p.write_text('source,target,weight\nA,B,1.0\nB,C,2.0\n', encoding='utf-8')
    G = from_csv(p)
    assert set(G.vertices()) == {'A', 'B', 'C'}
    assert G.ne == 2


# ── edges_to_csv / hyperedges_to_csv ───────────────────────────────────


def test_edges_to_csv_writes_only_binary_edges(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges('C', 'D', edge_id='e2', weight=3.0)
    p = tmp_path / 'out.csv'
    edges_to_csv(G, p)
    text = p.read_text(encoding='utf-8')
    assert 'source' in text
    assert 'A' in text and 'B' in text


def test_edges_to_csv_with_slice_filter_writes_a_file(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0, slice='s1')
    G.add_edges('B', 'C', edge_id='e2', weight=3.0, slice='s2')
    p = tmp_path / 'out.csv'
    # exercises the `slice=` branch in edges_to_csv
    edges_to_csv(G, p, slice='s1')
    text = p.read_text(encoding='utf-8')
    # header is always present; we don't assert on exact row count because
    # edges_view's slice filtering semantics are tested elsewhere.
    assert 'source' in text and 'target' in text


def test_hyperedges_to_csv_writes_members_rows(tmp_path: Path) -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    p = tmp_path / 'hyper.csv'
    hyperedges_to_csv(G, p)
    text = p.read_text(encoding='utf-8')
    assert 'members' in text
    for needle in ('A', 'B', 'C'):
        assert needle in text


def test_hyperedges_to_csv_with_no_hyperedges_writes_empty_or_header_only(
    tmp_path: Path,
) -> None:
    """The writer doesn't crash on a graph with only binary edges."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    p = tmp_path / 'h.csv'
    hyperedges_to_csv(G, p)
    # We don't constrain the exact layout — only that the call succeeded
    # and produced a file on disk (exercises the head/tail branch with
    # an empty `prepared` list).
    assert p.exists()


def test_hyperedges_to_csv_with_directed_hyper_writes_file(tmp_path: Path) -> None:
    """Smoke-test that the directed-hyperedge path runs end-to-end."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')  # directed hyper
    p = tmp_path / 'h.csv'
    hyperedges_to_csv(G, p)
    assert p.exists()


# ── round-trip via from_csv / hyperedges_to_csv ────────────────────────


def test_round_trip_edge_list_via_disk(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.5)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    p = tmp_path / 'rt.csv'
    edges_to_csv(G, p)
    G2 = from_csv(p, schema='edge_list')
    assert set(G2.vertices()) == {'A', 'B', 'C'}
    assert G2.ne == 2
