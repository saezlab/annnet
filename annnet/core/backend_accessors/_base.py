from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import import_module
from collections.abc import Callable

from ..._support.dataframe_backend import dataframe_columns, dataframe_to_rows, dataframe_filter_eq

if TYPE_CHECKING:
    from ..graph import AnnNet


class _BackendAccessorBase:
    VERTEX_LABEL_FIELDS = ('name', 'label', 'title', 'slug', 'external_id', 'string_id')
    _ADAPTER_EXPORTS = {
        'nx': ('annnet.adapters.networkx_adapter', 'to_nx'),
        'ig': ('annnet.adapters.igraph_adapter', 'to_igraph'),
        'gt': ('annnet.adapters.graphtool_adapter', 'to_graphtool'),
    }

    _G: AnnNet
    _cache_attr: str

    def _freeze_cache_value(self, value):
        if value is None:
            return None
        if isinstance(value, set):
            return frozenset(value)
        if isinstance(value, dict):
            return tuple(sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(sorted(value))
        return value

    def _vertex_row_maps(self):
        id_to_row = {}
        row_to_id = {}
        for ekey, rec in self._G._entities.items():
            if rec.kind != 'vertex':
                continue
            vid = ekey[0]
            id_to_row[vid] = rec.row_idx
            row_to_id[rec.row_idx] = vid
        return id_to_row, row_to_id

    def _vertex_row_to_id(self, row_idx: int):
        try:
            ekey = self._G._row_to_entity.get(row_idx)
            if ekey is None:
                return None
            rec = self._G._entities.get(ekey)
            if rec is None or rec.kind != 'vertex':
                return None
            return ekey[0]
        except Exception:  # noqa: BLE001
            return None

    def _infer_label_field(self) -> str | None:
        try:
            if getattr(self._G, 'default_label_field', None):
                return self._G.default_label_field
            va = getattr(self._G, 'vertex_attributes', None)
            cols = dataframe_columns(va) if va is not None else []
            for col in self.VERTEX_LABEL_FIELDS:
                if col in cols:
                    return col
        except Exception:  # noqa: BLE001
            pass
        return None

    def _vertex_id_col(self) -> str:
        try:
            va = self._G.vertex_attributes
            cols = dataframe_columns(va)
            for key in ('vertex_id', 'id', 'vid'):
                if key in cols:
                    return key
        except Exception:  # noqa: BLE001
            pass
        return 'vertex_id'

    def _lookup_vertex_id_by_label(self, label_field: str, value):
        try:
            va = self._G.vertex_attributes
            if va is None or label_field not in dataframe_columns(va):
                return None
            id_col = self._vertex_id_col()
            rows = dataframe_to_rows(dataframe_filter_eq(va, label_field, value))
            if rows:
                return rows[0].get(id_col)
        except Exception:  # noqa: BLE001
            return None
        return None

    def _map_nested_output(self, obj, leaf_mapper):
        if isinstance(obj, dict):
            return {
                leaf_mapper(key): self._map_nested_output(value, leaf_mapper)
                for key, value in obj.items()
            }
        if isinstance(obj, list):
            return [self._map_nested_output(value, leaf_mapper) for value in obj]
        if isinstance(obj, tuple):
            return tuple(self._map_nested_output(value, leaf_mapper) for value in obj)
        if isinstance(obj, set):
            return {self._map_nested_output(value, leaf_mapper) for value in obj}
        return leaf_mapper(obj)

    def _callable_names(self, *objects):
        names = set()
        for obj in objects:
            if obj is None:
                continue
            try:
                for name in dir(obj):
                    if name.startswith('_'):
                        continue
                    try:
                        attr = getattr(obj, name)
                    except Exception:  # noqa: BLE001
                        continue
                    if callable(attr):
                        names.add(name)
            except Exception:  # noqa: BLE001
                continue
        return names

    def _init_backend_accessor(self, owner: AnnNet, *, cache_attr: str):
        self._G = owner
        self._cache_attr = cache_attr
        self.cache_enabled = True

    def clear(self):
        setattr(self._G, self._cache_attr, {})

    def _cache_key(self, *parts):
        return tuple(self._freeze_cache_value(part) for part in parts)

    def _get_or_make_cached(
        self,
        key,
        build: Callable[[], dict],
    ):
        cache = getattr(self._G, self._cache_attr, None)
        if not isinstance(cache, dict):
            cache = {}

        version = getattr(self._G, '_version', None)
        entry = cache.get(key)
        rebuilt = False
        if (
            not self.cache_enabled
            or entry is None
            or (version is not None and entry.get('version') != version)
        ):
            entry = dict(build())
            entry['version'] = version
            if self.cache_enabled:
                cache[key] = entry
                setattr(self._G, self._cache_attr, cache)
            rebuilt = True
        return entry, rebuilt

    def _replace_owner_graph(self, args, kwargs, backend_graph):
        args = list(args)
        for idx, value in enumerate(args):
            if value is self._G:
                args[idx] = backend_graph
        for key, value in list(kwargs.items()):
            if value is self._G:
                kwargs[key] = backend_graph
        return args, kwargs

    def _warn_on_lossy_conversion(self, *, backend_name, hyperedge_mode, slice, slices, manifest):
        import warnings

        msgs = []
        if (
            any(rec.etype == 'hyper' for rec in self._G._edges.values())
            and hyperedge_mode != 'expand'
        ):
            msgs.append("hyperedges dropped (hyperedge_mode='skip')")
        slices_dict = getattr(self._G, '_slices', None)
        if (
            isinstance(slices_dict, dict)
            and len(slices_dict) > 1
            and (slice is None and not slices)
        ):
            msgs.append(f'multiple slices flattened into single {backend_name} graph')
        if manifest is None:
            msgs.append('no manifest provided; round-trip fidelity not guaranteed')
        if msgs:
            warnings.warn(
                f'AnnNet-{backend_name} conversion is lossy: ' + '; '.join(msgs) + '.',
                category=RuntimeWarning,
                stacklevel=3,
            )

    def _adapter_export(self, backend: str):
        module_name, export_name = self._ADAPTER_EXPORTS[backend]
        return getattr(import_module(module_name), export_name)

    def _coerce_vertex_iterable(self, obj, coerce_one):
        if isinstance(obj, (list, tuple, set)):
            coerced = [coerce_one(value) for value in obj]
            return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
        return coerce_one(obj)

    def _coerce_vertex_kwargs(self, kwargs: dict, coerce_many):
        for key in list(kwargs.keys()):
            if key in self.VERTEX_KEYS:
                kwargs[key] = coerce_many(kwargs[key])

    def _coerce_vertex_bound(self, bound, coerce_many):
        for key in list(bound.arguments.keys()):
            if key in self.VERTEX_KEYS:
                bound.arguments[key] = coerce_many(bound.arguments[key])

    def _edge_attr_aggregator(self, key, aggregations: dict | None):
        aggregations = aggregations or {}
        agg = aggregations.get(key)
        if callable(agg):
            return agg
        if agg == 'sum':
            return sum
        if agg == 'min':
            return min
        if agg == 'max':
            return max
        if agg == 'mean':
            return lambda values: (sum(values) / len(values)) if values else None
        if key == 'capacity':
            return sum
        if key == 'weight':
            return min
        return lambda values: next(iter(values)) if values else None
