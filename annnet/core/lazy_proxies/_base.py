from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

if TYPE_CHECKING:
    from ..graph import AnnNet


class _LazyProxyBase:
    VERTEX_LABEL_FIELDS = ("name", "label", "title", "slug", "external_id", "string_id")

    _G: AnnNet

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
            if rec.kind != "vertex":
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
            if rec is None or rec.kind != "vertex":
                return None
            return ekey[0]
        except Exception:
            return None

    def _infer_label_field(self) -> str | None:
        try:
            if getattr(self._G, "default_label_field", None):
                return self._G.default_label_field
            va = getattr(self._G, "vertex_attributes", None)
            cols = list(va.columns) if va is not None and hasattr(va, "columns") else []
            for col in self.VERTEX_LABEL_FIELDS:
                if col in cols:
                    return col
        except Exception:
            pass
        return None

    def _vertex_id_col(self) -> str:
        try:
            va = self._G.vertex_attributes
            cols = list(va.columns)
            for key in ("vertex_id", "id", "vid"):
                if key in cols:
                    return key
        except Exception:
            pass
        return "vertex_id"

    def _lookup_vertex_id_by_label(self, label_field: str, value):
        try:
            va = self._G.vertex_attributes
            if va is None or not hasattr(va, "columns") or label_field not in va.columns:
                return None
            id_col = self._vertex_id_col()
            try:
                matches = va.filter(pl.col(label_field) == value)
                if matches.height == 0:
                    return None
                try:
                    return matches.select(id_col).to_series().to_list()[0]
                except Exception:
                    return matches.select(id_col).item(0, 0)
            except Exception:
                for row in va.to_dicts():
                    if row.get(label_field) == value:
                        return row.get(id_col)
        except Exception:
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
                    if name.startswith("_"):
                        continue
                    try:
                        attr = getattr(obj, name)
                    except Exception:
                        continue
                    if callable(attr):
                        names.add(name)
            except Exception:
                continue
        return names
