"""Coverage tests for ``annnet/io/parquet.py``"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from annnet.core.graph import AnnNet
from annnet.io.parquet import (
    _as_list_or_empty,
    _build_attr_map,
    _edge_weight,
    _empty_table,
    _is_directed_eid,
    _is_nullish,
    _strip_nulls,
    from_parquet,
    to_parquet,
)


# ── helpers ────────────────────────────────────────────────────────────


def test_empty_table_creates_text_typed_empty_dataframe() -> None:
    df = _empty_table(['a', 'b'])
    assert 'a' in df.columns
    assert 'b' in df.columns
    assert df.height == 0


def test_strip_nulls_removes_none_and_nan() -> None:
    out = _strip_nulls({'a': 1, 'b': None, 'c': float('nan'), 'd': 'x'})
    assert out == {'a': 1, 'd': 'x'}


def test_is_nullish_recognizes_none_nan_and_numpy_nan() -> None:
    assert _is_nullish(None) is True
    assert _is_nullish(float('nan')) is True
    assert _is_nullish(np.nan) is True
    assert _is_nullish(0) is False
    assert _is_nullish('x') is False


def test_as_list_or_empty_handles_every_shape() -> None:
    assert _as_list_or_empty(None) == []
    assert _as_list_or_empty(float('nan')) == []
    assert _as_list_or_empty([1, 2]) == [1, 2]
    assert _as_list_or_empty((1, 2)) == [1, 2]
    assert _as_list_or_empty('scalar') == ['scalar']

    class Listy:
        def to_list(self):
            return [10, 20]

    assert _as_list_or_empty(Listy()) == [10, 20]

    arr = np.array([1, 2, 3])
    assert _as_list_or_empty(arr) == [1, 2, 3]


def test_build_attr_map_skips_records_without_key_and_dedupes() -> None:
    class FakeDF:
        """A minimal table-like object the dataframe backend can iterate."""

    # use a real backend dataframe instead
    from annnet._support.dataframe_backend import dataframe_from_rows

    df = dataframe_from_rows(
        [
            {'vertex_id': 'A', 'name': 'alice'},
            {'vertex_id': 'B', 'name': 'bob'},
            {'vertex_id': None, 'name': 'no-key'},  # skipped
            {'vertex_id': 'A', 'name': 'duplicate'},  # 'A' already in map → skipped
        ]
    )
    out = _build_attr_map(df, 'vertex_id')
    assert out == {'A': {'name': 'alice'}, 'B': {'name': 'bob'}}


# ── _edge_weight / _is_directed_eid via tiny stub graph ────────────────


class _StubGraph:
    """Minimal duck-typed graph stand-in for the standalone helpers."""

    def __init__(
        self,
        edge_weights=None,
        edge_directed=None,
        hyperedge_definitions=None,
        edge_attrs=None,
    ):
        self.edge_weights = edge_weights or {}
        self.edge_directed = edge_directed or {}
        self.hyperedge_definitions = hyperedge_definitions or {}
        self._edge_attrs = edge_attrs or {}

    class _AttrsShim:
        def __init__(self, owner):
            self._owner = owner

        def get_attr_edge(self, eid, key, default=None):
            return self._owner._edge_attrs.get(eid, {}).get(key, default)

    @property
    def attrs(self):
        return self._AttrsShim(self)


def test_edge_weight_returns_one_when_missing_else_float() -> None:
    g = _StubGraph(edge_weights={'e1': 2.5})
    assert _edge_weight(g, 'e1') == 2.5
    assert _edge_weight(g, 'missing') == 1.0


def test_is_directed_eid_prefers_edge_directed_dict() -> None:
    g = _StubGraph(edge_directed={'e1': True})
    assert _is_directed_eid(g, 'e1') is True


def test_is_directed_eid_falls_back_to_edge_attr() -> None:
    g = _StubGraph(edge_attrs={'e1': {'directed': False}})
    assert _is_directed_eid(g, 'e1') is False


def test_is_directed_eid_for_hyperedge_with_identical_endpoints_is_undirected() -> None:
    g = _StubGraph(hyperedge_definitions={'h1': {'head': ['A'], 'tail': ['A']}})
    assert _is_directed_eid(g, 'h1') is False


def test_is_directed_eid_default_true_for_binary_false_for_hyper() -> None:
    g_binary = _StubGraph()
    assert _is_directed_eid(g_binary, 'e_unknown_binary') is True

    # When the hyperedge has no entry in edge_directed and no 'directed' attr,
    # the fallback returns `kind != 'hyper'` → False.
    g_hyper = _StubGraph(hyperedge_definitions={'h1': {'head': ['A'], 'tail': ['B']}})
    assert _is_directed_eid(g_hyper, 'h1') is False


# ── round-trips ───────────────────────────────────────────────────────


def test_round_trip_empty_graph(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    p = tmp_path / 'empty'
    to_parquet(G, p)
    H = from_parquet(p)
    assert H.nv == 0
    assert H.ne == 0


def test_round_trip_simple_directed_graph_with_attrs(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.attrs.set_vertex_attrs('A', color='red')
    G.attrs.set_vertex_attrs('B', color='blue')
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges('B', 'C', edge_id='e2', weight=3.0)
    G.attrs.set_edge_attrs('e1', label='alpha')

    p = tmp_path / 'simple'
    to_parquet(G, p)
    H = from_parquet(p)
    assert set(H.vertices()) == {'A', 'B', 'C'}
    assert H.ne == 2


def test_round_trip_with_undirected_hyperedge(tmp_path: Path) -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    p = tmp_path / 'hyper'
    to_parquet(G, p)
    H = from_parquet(p)
    assert set(H.vertices()) >= {'A', 'B', 'C'}
    assert H.ne == 1


def test_round_trip_with_directed_hyperedge(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    p = tmp_path / 'dhyper'
    to_parquet(G, p)
    H = from_parquet(p)
    assert H.ne == 1


def test_round_trip_with_slices_and_per_slice_weights(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.slices.add('s1')
    G.add_edges('A', 'B', edge_id='e1', slice='s1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', slice='s1', weight=2.0)
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=99.0)

    p = tmp_path / 'sliced'
    to_parquet(G, p)
    H = from_parquet(p)
    # slice and edges round-tripped
    assert 's1' in H.slices.list()
    assert H.ne == 2


def test_round_trip_with_multilayer_graph_via_manifest(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    G.add_vertices(['B'], layer={'condition': 'treated'})

    p = tmp_path / 'multilayer'
    to_parquet(G, p)
    H = from_parquet(p)
    assert set(H.vertices()) == {'A', 'B'}
    assert H.is_multilayer
    assert tuple(H.layers.list_aspects()) == ('condition',)


def test_round_trip_with_hyper_edge_attrs(tmp_path: Path) -> None:
    """Hyperedge attrs survive serialization and reattach in from_parquet."""
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    G.attrs.set_edge_attrs('h1', confidence=0.95, label='triple')

    p = tmp_path / 'hyper_attrs'
    to_parquet(G, p)
    H = from_parquet(p)
    assert H.ne == 1
    out = H.attrs.get_edge_attrs('h1')
    assert out.get('confidence') == 0.95
    assert out.get('label') == 'triple'


def test_round_trip_with_per_slice_weight_round_trips_weight(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.slices.add('s1')
    G.add_edges('A', 'B', edge_id='e1', slice='s1', weight=1.0)
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=42.0)

    p = tmp_path / 'sweight'
    to_parquet(G, p)
    H = from_parquet(p)
    w = H.attrs.get_edge_slice_attr('s1', 'e1', 'weight')
    assert w == 42.0


def test_build_attr_map_iterates_only_dict_records() -> None:
    """Coverage for the dict() fallback path inside ``_build_attr_map``."""
    from annnet._support.dataframe_backend import dataframe_from_rows

    df = dataframe_from_rows(
        [
            {'vertex_id': 'A', 'color': 'red'},
            {'vertex_id': 'B', 'color': 'blue'},
        ]
    )
    out = _build_attr_map(df, 'vertex_id')
    assert set(out) == {'A', 'B'}


def test_is_nullish_via_object_that_raises_in_not_equal_path() -> None:
    """Exercise the ``except (TypeError, ValueError): pass`` in ``_is_nullish``."""

    class Hostile:
        def __ne__(self, other):
            raise TypeError('cannot compare')

        __hash__ = None  # type: ignore[assignment]

    # The function must return False (not None) without propagating the
    # TypeError raised from the ``val != val`` comparison.
    assert _is_nullish(Hostile()) is False


def test_is_nullish_via_object_that_raises_value_error_on_neq() -> None:
    class Hostile2:
        def __ne__(self, other):
            raise ValueError('numpy array comparison')

        __hash__ = None  # type: ignore[assignment]

    assert _is_nullish(Hostile2()) is False


def test_build_attr_map_skips_records_missing_the_key_column() -> None:
    """Hit the ``if key_col not in rec: continue`` branch."""
    from annnet._support.dataframe_backend import dataframe_from_rows

    df = dataframe_from_rows(
        [
            {'name': 'alice'},  # no 'vertex_id' column at all
            {'name': 'bob'},
        ]
    )
    out = _build_attr_map(df, 'vertex_id')
    assert out == {}


# Silence pyflakes about the imported math module (used elsewhere transitively).
_ = math
