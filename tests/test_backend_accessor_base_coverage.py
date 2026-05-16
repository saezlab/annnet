"""Coverage tests for ``annnet/core/backend_accessors/_base.py``."""

from __future__ import annotations

import warnings

import pytest

from annnet.core._records import EdgeRecord, EntityRecord
from annnet.core.backend_accessors._base import _BackendAccessorBase
from annnet.core.graph import AnnNet


class _Accessor(_BackendAccessorBase):
    """Minimal concrete subclass to test the abstract base."""

    VERTEX_KEYS = {'source', 'target', 'u', 'v'}


def _toy_accessor(directed: bool = True) -> tuple[_Accessor, AnnNet]:
    G = AnnNet(directed=directed)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G._test_cache = {}  # type: ignore[attr-defined]
    a = _Accessor()
    a._init_backend_accessor(G, cache_attr='_test_cache')
    return a, G


# ── _freeze_cache_value ───────────────────────────────────────────────


def test_freeze_cache_value_handles_none_set_dict_list_tuple_and_scalar() -> None:
    a, _ = _toy_accessor()
    assert a._freeze_cache_value(None) is None
    assert a._freeze_cache_value({1, 2}) == frozenset({1, 2})
    assert a._freeze_cache_value({'b': 2, 'a': 1}) == (('a', 1), ('b', 2))
    assert a._freeze_cache_value([3, 1, 2]) == (1, 2, 3)
    assert a._freeze_cache_value((3, 1, 2)) == (1, 2, 3)
    assert a._freeze_cache_value(7) == 7


# ── _vertex_row_maps / _vertex_row_to_id ──────────────────────────────


def test_vertex_row_maps_round_trips_id_and_row() -> None:
    a, _ = _toy_accessor()
    id_to_row, row_to_id = a._vertex_row_maps()
    assert set(id_to_row) == {'A', 'B', 'C'}
    for vid, row in id_to_row.items():
        assert row_to_id[row] == vid


def test_vertex_row_to_id_returns_id_for_known_row() -> None:
    a, _ = _toy_accessor()
    id_to_row, _ = a._vertex_row_maps()
    row = id_to_row['A']
    assert a._vertex_row_to_id(row) == 'A'


def test_vertex_row_to_id_returns_none_for_unknown_row() -> None:
    a, _ = _toy_accessor()
    assert a._vertex_row_to_id(99_999) is None


def test_vertex_row_to_id_returns_none_for_non_vertex_row() -> None:
    """Synthesize an edge_entity row and confirm it's filtered out."""
    a, G = _toy_accessor()
    G._row_to_entity[999] = ('e1',)
    G._entities[('e1',)] = EntityRecord(row_idx=999, kind='edge_entity')
    assert a._vertex_row_to_id(999) is None


# ── _infer_label_field ────────────────────────────────────────────────


def test_infer_label_field_prefers_default_label_field_when_set() -> None:
    a, G = _toy_accessor()
    G.default_label_field = 'pretty_name'  # type: ignore[attr-defined]
    assert a._infer_label_field() == 'pretty_name'


def test_infer_label_field_falls_back_to_known_columns() -> None:
    a, G = _toy_accessor()
    G.attrs.set_vertex_attrs('A', name='alice')
    assert a._infer_label_field() == 'name'


def test_infer_label_field_returns_none_when_nothing_matches() -> None:
    a, _ = _toy_accessor()
    assert a._infer_label_field() is None


# ── _vertex_id_col ────────────────────────────────────────────────────


def test_vertex_id_col_returns_vertex_id_by_default() -> None:
    a, _ = _toy_accessor()
    assert a._vertex_id_col() == 'vertex_id'


def test_vertex_id_col_returns_default_when_attribute_access_raises() -> None:
    a = _Accessor()
    a._G = object()  # type: ignore[assignment]
    a._cache_attr = '_test_cache'
    a.cache_enabled = True
    assert a._vertex_id_col() == 'vertex_id'


# ── _lookup_vertex_id_by_label ────────────────────────────────────────


def test_lookup_vertex_id_by_label_finds_known_value() -> None:
    a, G = _toy_accessor()
    G.attrs.set_vertex_attrs('A', name='alice')
    G.attrs.set_vertex_attrs('B', name='bob')
    assert a._lookup_vertex_id_by_label('name', 'bob') == 'B'


def test_lookup_vertex_id_by_label_returns_none_for_missing_column() -> None:
    a, _ = _toy_accessor()
    assert a._lookup_vertex_id_by_label('not-a-col', 'x') is None


def test_lookup_vertex_id_by_label_returns_none_for_missing_value() -> None:
    a, G = _toy_accessor()
    G.attrs.set_vertex_attrs('A', name='alice')
    assert a._lookup_vertex_id_by_label('name', 'nope') is None


# ── _map_nested_output ────────────────────────────────────────────────


def test_map_nested_output_walks_dict_list_tuple_set_and_leaf() -> None:
    a, _ = _toy_accessor()
    payload = {
        1: [2, (3, 4), {5, 6}],
        7: {'inner': 8},
    }
    out = a._map_nested_output(payload, lambda x: x * 10 if isinstance(x, int) else x)
    assert out == {10: [20, (30, 40), {50, 60}], 70: {'inner': 80}}


# ── _callable_names ───────────────────────────────────────────────────


def test_callable_names_collects_public_callables_and_skips_none() -> None:
    a, _ = _toy_accessor()

    class Probe:
        def foo(self): ...

        bar = 5

        def _private(self): ...

    names = a._callable_names(None, Probe())
    assert 'foo' in names
    assert 'bar' not in names
    assert '_private' not in names


def test_callable_names_tolerates_attribute_access_failures() -> None:
    a, _ = _toy_accessor()

    class Hostile:
        def __dir__(self):
            return ['boom']

        def __getattr__(self, name):
            raise RuntimeError('no')

    # Must not raise; failing attribute is skipped.
    names = a._callable_names(Hostile())
    assert 'boom' not in names


# ── clear / _cache_key / _get_or_make_cached ──────────────────────────


def test_clear_resets_cache_attribute_on_graph() -> None:
    a, G = _toy_accessor()
    G._test_cache = {'k': {'v': 1}}  # type: ignore[attr-defined]
    a.clear()
    assert G._test_cache == {}


def test_cache_key_freezes_each_part() -> None:
    a, _ = _toy_accessor()
    k = a._cache_key({1, 2}, {'a': 1})
    assert k == (frozenset({1, 2}), (('a', 1),))


def test_get_or_make_cached_builds_then_reuses_then_rebuilds_on_version_change() -> None:
    a, G = _toy_accessor()
    calls = {'n': 0}

    def build():
        calls['n'] += 1
        return {'payload': calls['n']}

    entry1, rebuilt1 = a._get_or_make_cached(('k',), build)
    assert rebuilt1 is True
    assert entry1['payload'] == 1

    entry2, rebuilt2 = a._get_or_make_cached(('k',), build)
    assert rebuilt2 is False
    assert entry2 is entry1

    # Bump the graph version → forced rebuild.
    G._version = (G._version or 0) + 1
    entry3, rebuilt3 = a._get_or_make_cached(('k',), build)
    assert rebuilt3 is True
    assert entry3['payload'] == 2


def test_get_or_make_cached_skips_storing_when_cache_disabled() -> None:
    a, G = _toy_accessor()
    a.cache_enabled = False
    a._get_or_make_cached(('k',), lambda: {'x': 1})
    assert G._test_cache == {}


# ── _replace_owner_graph ──────────────────────────────────────────────


def test_replace_owner_graph_swaps_owner_in_args_and_kwargs() -> None:
    a, G = _toy_accessor()
    backend_graph = object()
    args, kwargs = a._replace_owner_graph((G, 1), {'graph': G, 'x': 2}, backend_graph)
    assert args == [backend_graph, 1]
    assert kwargs == {'graph': backend_graph, 'x': 2}


# ── _warn_on_lossy_conversion ─────────────────────────────────────────


def test_warn_on_lossy_conversion_complains_when_hyperedges_dropped() -> None:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')  # hyperedge
    a = _Accessor()
    a._init_backend_accessor(G, cache_attr='_test_cache')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a._warn_on_lossy_conversion(
            backend_name='nx',
            hyperedge_mode='skip',
            slice=None,
            slices=None,
            manifest='ok',
        )
    msgs = ' '.join(str(w.message) for w in caught)
    assert 'hyperedges dropped' in msgs


def test_warn_on_lossy_conversion_complains_when_manifest_missing() -> None:
    a, _ = _toy_accessor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a._warn_on_lossy_conversion(
            backend_name='nx',
            hyperedge_mode='expand',
            slice=None,
            slices=None,
            manifest=None,
        )
    msgs = ' '.join(str(w.message) for w in caught)
    assert 'no manifest' in msgs


def test_warn_on_lossy_conversion_silent_on_clean_path() -> None:
    a, _ = _toy_accessor()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        a._warn_on_lossy_conversion(
            backend_name='nx',
            hyperedge_mode='expand',
            slice=None,
            slices=None,
            manifest='ok',
        )
    assert not caught


# ── _adapter_export ───────────────────────────────────────────────────


def test_adapter_export_returns_callable_for_known_backend() -> None:
    a, _ = _toy_accessor()
    for backend in ('nx', 'ig'):
        out = a._adapter_export(backend)
        assert callable(out)


def test_adapter_export_raises_keyerror_for_unknown_backend() -> None:
    a, _ = _toy_accessor()
    with pytest.raises(KeyError):
        a._adapter_export('not-a-backend')


# ── _coerce_vertex_* ──────────────────────────────────────────────────


def test_coerce_vertex_iterable_preserves_container_type() -> None:
    a, _ = _toy_accessor()
    coerce = str.upper
    assert a._coerce_vertex_iterable(['a', 'b'], coerce) == ['A', 'B']
    assert a._coerce_vertex_iterable(('a', 'b'), coerce) == ('A', 'B')
    assert a._coerce_vertex_iterable({'a', 'b'}, coerce) == {'A', 'B'}
    assert a._coerce_vertex_iterable('a', coerce) == 'A'


def test_coerce_vertex_kwargs_rewrites_only_known_keys() -> None:
    a, _ = _toy_accessor()
    kwargs = {'source': 'a', 'unrelated': 'b'}
    a._coerce_vertex_kwargs(kwargs, lambda xs: xs.upper())
    assert kwargs == {'source': 'A', 'unrelated': 'b'}


def test_coerce_vertex_bound_rewrites_only_known_keys() -> None:
    a, _ = _toy_accessor()

    class Bound:
        arguments = {'u': 'x', 'irrelevant': 1}

    a._coerce_vertex_bound(Bound, lambda v: v.upper())
    assert Bound.arguments == {'u': 'X', 'irrelevant': 1}


# ── _edge_attr_aggregator ─────────────────────────────────────────────


def test_edge_attr_aggregator_callable_passthrough() -> None:
    a, _ = _toy_accessor()

    def f(xs):
        return sum(xs) * 10

    assert a._edge_attr_aggregator('k', {'k': f})([1, 2]) == 30


def test_edge_attr_aggregator_named_strategies() -> None:
    a, _ = _toy_accessor()
    assert a._edge_attr_aggregator('k', {'k': 'sum'})([1, 2, 3]) == 6
    assert a._edge_attr_aggregator('k', {'k': 'min'})([3, 1, 2]) == 1
    assert a._edge_attr_aggregator('k', {'k': 'max'})([3, 1, 2]) == 3
    assert a._edge_attr_aggregator('k', {'k': 'mean'})([2, 4]) == 3
    # empty mean returns None
    assert a._edge_attr_aggregator('k', {'k': 'mean'})([]) is None


def test_edge_attr_aggregator_key_specific_defaults() -> None:
    a, _ = _toy_accessor()
    assert a._edge_attr_aggregator('capacity', None)([1, 2, 3]) == 6
    assert a._edge_attr_aggregator('weight', None)([3, 1, 2]) == 1


def test_edge_attr_aggregator_fallback_picks_first_value_or_none() -> None:
    a, _ = _toy_accessor()
    pick = a._edge_attr_aggregator('color', None)
    assert pick(['red', 'blue']) == 'red'
    assert pick([]) is None


# Silence unused import warnings — keeps the module references that the
# test infra walks over for collection.
_ = EdgeRecord
