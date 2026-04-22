import math
from typing import Any

import narwhals as nw

from ._helpers import _get_numeric_supertype
from .._dataframe_backend import (
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_eq,
    dataframe_filter_in,
    dataframe_upsert_rows,
)

_NUMERIC_NW_DTYPES = {
    nw.Int8,
    nw.Int16,
    nw.Int32,
    nw.Int64,
    nw.UInt8,
    nw.UInt16,
    nw.UInt32,
    nw.UInt64,
    nw.Float32,
    nw.Float64,
}

_NUMERIC_DTYPES = _NUMERIC_NW_DTYPES


class AttributesClass:
    """Attribute accessors and upsert helpers.

    This mixin owns graph-, vertex-, edge-, slice-, and edge-slice-level
    metadata. Structural topology is intentionally excluded; only non-structural
    user attributes are stored in the annotation tables.
    """

    def set_graph_attribute(self, key, value):
        """Set a graph-level attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        value : Any
            Attribute value.
        """
        self.graph_attributes[key] = value

    def get_graph_attribute(self, key, default=None):
        """Get a graph-level attribute.

        Parameters
        ----------
        key : str
            Attribute name.
        default : Any, optional
            Value to return if the attribute is missing.

        Returns
        -------
        Any
        """
        return self.graph_attributes.get(key, default)

    def _lookup_attr(self, df, row_filter: dict, key, default=None):
        """Return one attribute value from a dataframe-like annotation table."""
        if df is None or key not in dataframe_columns(df):
            return default
        rows_df = df
        for col, value in row_filter.items():
            rows_df = dataframe_filter_eq(rows_df, col, value)
        rows = dataframe_to_rows(rows_df)
        if not rows:
            return default
        val = rows[0].get(key)
        return default if val is None else val

    def set_vertex_attrs(self, vertex_id, **attrs):
        """Upsert pure vertex attributes (non-structural) into the vertex table.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored.
        """
        clean = {k: v for k, v in attrs.items() if k not in self._vertex_RESERVED}
        if not clean:
            return

        # If composite-key is active, validate prospective key BEFORE writing
        if self._vertex_key_enabled():
            old_key = self._current_key_of_vertex(vertex_id)
            # prospective values = old values overridden by incoming clean attrs
            merged = {
                f: (clean[f] if f in clean else self.get_attr_vertex(vertex_id, f, None))
                for f in self._vertex_key_fields
            }
            new_key = self._build_key_from_attrs(merged)
            if new_key is not None:
                owner = self._vertex_key_index.get(new_key)
                if owner is not None and owner != vertex_id:
                    raise ValueError(
                        f'Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}'
                    )

        # Write attributes
        self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, clean)

        watched = self._variables_watched_by_vertices()
        if watched and any(k in watched for k in clean):
            for eid in self._incident_flexible_edges(vertex_id):
                self._apply_flexible_direction(eid)

        # Update index AFTER successful write
        if self._vertex_key_enabled():
            new_key = self._current_key_of_vertex(vertex_id)
            old_key = old_key if 'old_key' in locals() else None
            if old_key != new_key:
                if old_key is not None and self._vertex_key_index.get(old_key) == vertex_id:
                    self._vertex_key_index.pop(old_key, None)
                if new_key is not None:
                    self._vertex_key_index[new_key] = vertex_id

    def set_vertex_attrs_bulk(self, updates):
        """Upsert vertex attributes in bulk.

        Parameters
        ----------
        updates : dict[str, dict] | Iterable[tuple[str, dict]]
            Mapping or iterable of `(vertex_id, attrs)` pairs.
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)

        for vid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(f'vertex bulk attrs must be dict, got {type(attrs)} for {vid}')

        clean_updates = {
            vid: {k: v for k, v in attrs.items() if k not in self._vertex_RESERVED}
            for vid, attrs in updates.items()
        }
        clean_updates = {vid: attrs for vid, attrs in clean_updates.items() if attrs}

        if not clean_updates:
            return

        if self._vertex_key_enabled():
            old_keys = {vid: self._current_key_of_vertex(vid) for vid in clean_updates}

            new_keys = {}
            for vid, attrs in clean_updates.items():
                merged = {
                    f: (attrs[f] if f in attrs else self.get_attr_vertex(vid, f, None))
                    for f in self._vertex_key_fields
                }
                new_keys[vid] = self._build_key_from_attrs(merged)

            for vid, new_key in new_keys.items():
                if new_key is not None:
                    owner = self._vertex_key_index.get(new_key)
                    if owner is not None and owner != vid:
                        raise ValueError(
                            f'Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}'
                        )

        self.vertex_attributes = self._upsert_rows_bulk(self.vertex_attributes, clean_updates)

        watched = self._variables_watched_by_vertices()
        if watched:
            affected_vertices = {
                vid for vid, attrs in clean_updates.items() if any(k in watched for k in attrs)
            }
            if affected_vertices:
                affected_edges = set()
                for vid in affected_vertices:
                    affected_edges.update(self._incident_flexible_edges(vid))
                for eid in affected_edges:
                    self._apply_flexible_direction(eid)

        if self._vertex_key_enabled():
            for vid in clean_updates:
                new_key = new_keys[vid]
                old_key = old_keys[vid]
                if old_key != new_key:
                    if old_key is not None and self._vertex_key_index.get(old_key) == vid:
                        self._vertex_key_index.pop(old_key, None)
                    if new_key is not None:
                        self._vertex_key_index[new_key] = vid

    def get_attr_vertex(self, vertex_id, key, default=None):
        """Get a single vertex attribute (scalar) or default if missing.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._lookup_attr(self.vertex_attributes, {'vertex_id': vertex_id}, key, default)

    def set_edge_attrs(self, edge_id, **attrs):
        """Upsert pure edge attributes (non-structural) into the edge DF.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored.
        """
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if clean:
            self.edge_attributes = self._upsert_row(self.edge_attributes, edge_id, clean)
        pol = self.edge_direction_policy.get(edge_id)
        if pol and pol.get('scope', 'edge') == 'edge' and pol['var'] in clean:
            self._apply_flexible_direction(edge_id)

    def set_edge_attrs_bulk(self, updates):
        """Upsert edge attributes in bulk.

        Parameters
        ----------
        updates : dict[str, dict] | Iterable[tuple[str, dict]]
            Mapping or iterable of `(edge_id, attrs)` pairs.
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)
        for eid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(f'edge bulk attrs must be dict, got {type(attrs)} for {eid}')

        clean_updates = {
            eid: {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
            for eid, attrs in updates.items()
        }
        clean_updates = {eid: attrs for eid, attrs in clean_updates.items() if attrs}

        if not clean_updates:
            return

        self.edge_attributes = self._upsert_rows_bulk(self.edge_attributes, clean_updates)

        affected_edges = set()
        for eid, attrs in clean_updates.items():
            pol = self.edge_direction_policy.get(eid)
            if pol and pol.get('scope') == 'edge' and pol['var'] in attrs:
                affected_edges.add(eid)

        for eid in affected_edges:
            self._apply_flexible_direction(eid)

    def get_attr_edge(self, edge_id, key, default=None):
        """Get a single edge attribute (scalar) or default if missing.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._lookup_attr(self.edge_attributes, {'edge_id': edge_id}, key, default)

    def set_slice_attrs(self, slice_id, **attrs):
        """Upsert pure slice attributes.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored.
        """
        clean = {k: v for k, v in attrs.items() if k not in self._slice_RESERVED}
        if clean:
            self.slice_attributes = self._upsert_row(self.slice_attributes, slice_id, clean)

    def get_slice_attr(self, slice_id, key, default=None):
        """Get a single slice attribute (scalar) or default if missing.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._lookup_attr(self.slice_attributes, {'slice_id': slice_id}, key, default)

    def set_edge_slice_attrs(self, slice_id, edge_id, **attrs):
        """Upsert per-slice attributes for a specific edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored except `weight`.
        """
        # allow 'weight' through; keep ignoring true structural keys
        clean = {
            k: v for k, v in attrs.items() if (k not in self._EDGE_RESERVED) or (k == 'weight')
        }
        if not clean:
            return

        # Normalize hot keys (intern) and avoid float dtype surprises for 'weight'
        try:
            import sys as _sys

            if isinstance(slice_id, str):
                slice_id = _sys.intern(slice_id)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except Exception:  # noqa: BLE001
            pass
        if 'weight' in clean:
            try:
                # cast once to float to reduce dtype mismatch churn inside _upsert_row
                clean['weight'] = float(clean['weight'])
            except Exception:  # noqa: BLE001
                # leave as-is if not coercible; behavior stays identical
                pass

        # Upsert via central helper (keeps exact behavior, schema handling, and caching)
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )

    def get_edge_slice_attr(self, slice_id, edge_id, key, default=None):
        """Get a per-slice attribute for an edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        key : str
            Attribute name.
        default : Any, optional
            Value to return if missing.

        Returns
        -------
        Any
        """
        return self._lookup_attr(
            self.edge_slice_attributes,
            {'slice_id': slice_id, 'edge_id': edge_id},
            key,
            default,
        )

    def set_slice_edge_weight(self, slice_id, edge_id, weight):  # legacy weight helper
        """Set a legacy per-slice weight override for an edge.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_id : str
            Edge identifier.
        weight : float
            Weight override.

        Raises
        ------
        KeyError
            If the slice or edge does not exist.

        See Also
        --------
        get_effective_edge_weight
        """
        if slice_id not in self._slices:
            raise KeyError(f'slice {slice_id} not found')
        _rec = self._edges.get(edge_id)
        if _rec is None or _rec.col_idx < 0:
            raise KeyError(f'Edge {edge_id} not found')
        self.slice_edge_weights[slice_id][edge_id] = float(weight)

    def get_effective_edge_weight(self, edge_id, slice=None):
        """Resolve the effective weight for an edge, optionally within a slice.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        slice : str, optional
            If provided, return the slice override if present; otherwise global weight.

        Returns
        -------
        float
            Effective weight.
        """
        if slice is not None:
            w = self._lookup_attr(
                self.edge_slice_attributes,
                {'slice_id': slice, 'edge_id': edge_id},
                'weight',
                None,
            )
            if w is not None and not (isinstance(w, float) and math.isnan(w)):
                return float(w)

            # fallback to legacy dict if present
            w2 = self.slice_edge_weights.get(slice, {}).get(edge_id, None)
            if w2 is not None:
                return float(w2)

        _rec = self._edges.get(edge_id)
        return float(_rec.weight if (_rec is not None and _rec.weight is not None) else 1.0)

    def audit_attributes(self):
        """Audit attribute tables for extra/missing rows and invalid edge-slice pairs.

        Returns
        -------
        dict
            Summary with keys:
            - `extra_vertex_rows`
            - `extra_edge_rows`
            - `missing_vertex_rows`
            - `missing_edge_rows`
            - `invalid_edge_slice_rows`
        """
        vertex_ids = {ekey[0] for ekey, rec in self._entities.items() if rec.kind == 'vertex'}
        edge_ids = set(self._col_to_edge.values())

        na = self.vertex_attributes
        ea = self.edge_attributes
        ela = self.edge_slice_attributes

        vertex_attr_ids = (
            {row.get('vertex_id') for row in dataframe_to_rows(na)}
            if na is not None and 'vertex_id' in dataframe_columns(na)
            else set()
        )
        vertex_attr_ids.discard(None)

        edge_attr_ids = (
            {row.get('edge_id') for row in dataframe_to_rows(ea)}
            if ea is not None and 'edge_id' in dataframe_columns(ea)
            else set()
        )
        edge_attr_ids.discard(None)

        extra_vertex_rows = [i for i in vertex_attr_ids if i not in vertex_ids]
        extra_edge_rows = [i for i in edge_attr_ids if i not in edge_ids]
        missing_vertex_rows = [i for i in vertex_ids if i not in vertex_attr_ids]
        missing_edge_rows = [i for i in edge_ids if i not in edge_attr_ids]

        bad_edge_slice = []
        if ela is not None and {'slice_id', 'edge_id'} <= set(dataframe_columns(ela)):
            for r in dataframe_to_rows(ela):
                lid = r.get('slice_id')
                eid = r.get('edge_id')
                if lid not in self._slices or eid not in edge_ids:
                    bad_edge_slice.append((lid, eid))

        return {
            'extra_vertex_rows': extra_vertex_rows,
            'extra_edge_rows': extra_edge_rows,
            'missing_vertex_rows': missing_vertex_rows,
            'missing_edge_rows': missing_edge_rows,
            'invalid_edge_slice_rows': bad_edge_slice,
        }

    def _dtype_for_value(self, v):
        """INTERNAL: Infer an appropriate dtype class for value `v`.

        Returns Narwhals dtype classes so annotation logic stays backend-neutral.
        """
        import enum

        if v is None:
            return nw.Unknown
        if isinstance(v, bool):
            return nw.Boolean
        if isinstance(v, int) and not isinstance(v, bool):
            return nw.Int64
        if isinstance(v, float):
            return nw.Float64
        if isinstance(v, enum.Enum):
            return nw.Object
        if isinstance(v, (bytes, bytearray)):
            return nw.Binary
        if isinstance(v, (list, tuple)):
            inner = self._dtype_for_value(v[0]) if len(v) else nw.String
            return nw.List(nw.String if inner == nw.Unknown else inner)
        if isinstance(v, dict):
            return nw.Object
        return nw.String

    def _is_null_dtype(self, dtype) -> bool:
        """Check if a dtype represents a null/unknown type."""
        import narwhals as nw

        if dtype == nw.Unknown:
            return True
        dt_type = type(dtype) if not isinstance(dtype, type) else dtype
        return dt_type == nw.Unknown or str(dtype).lower() in {'null', 'unknown'}

    def _ensure_attr_columns(self, df, attrs: dict) -> nw.DataFrame[Any]:
        """Create/align attribute columns and dtypes to accept ``attrs``.

        Parameters
        ----------
        df : IntoDataFrame
            Existing attribute table (any supported backend).
        attrs : dict
            Incoming key/value pairs to upsert.

        Returns
        -------
        nw.DataFrame
            Narwhals DataFrame with columns added/cast so inserts/updates work.

        Notes
        -----
        - New columns are created with the inferred dtype.
        - If a column is Unknown (null-ish) and the incoming value is not,
          it is cast to the inferred dtype.
        - If dtypes conflict, both sides upcast to String to avoid schema errors.
        """
        nw_df = nw.from_native(df, eager_only=True)
        schema = nw_df.collect_schema()

        for col, val in attrs.items():
            target = self._dtype_for_value(val)

            if col not in schema:
                try:
                    nw_df = nw_df.with_columns(nw.lit(None).cast(target).alias(col))
                except Exception:  # noqa: BLE001
                    nw_df = nw_df.with_columns(nw.lit(None).alias(col))
            else:
                # Upgrade logic: ONLY cast if the existing column is a Null/Unknown type
                cur = schema[col]
                if self._is_null_dtype(cur) and not self._is_null_dtype(target):
                    try:
                        nw_df = nw_df.with_columns(nw.col(col).cast(target))
                    except Exception:  # noqa: BLE001
                        pass
                # DELETED: The 'elif cur != target' block that forced String fallback.
                # Type conflicts are now managed lazily during the actual upsert/concat.

        return nw_df

    def _sanitize_value_for_nw(self, v):
        # narwhals.lit can't handle nested python containers yet
        if isinstance(v, (list, tuple, dict)):
            import json

            return json.dumps(v, ensure_ascii=False)
        return v

    def _is_binary_type(self, dt) -> bool:
        """INTERNAL: Robustly identify binary types across backends."""
        import narwhals as nw

        if isinstance(dt, (nw.Binary, nw.dtypes.Binary)):
            return True
        s = str(dt).lower()
        return any(kw in s for kw in ('binary', 'blob', 'byte'))

    def _safe_nw_cast(self, column_expr, target_dtype):
        """INTERNAL: Attempt cast; fallback to String on engine rejection."""
        import narwhals as nw

        try:
            # Ensure target_dtype is a Narwhals DType, not a native backend class
            if not isinstance(target_dtype, (nw.dtypes.DType, type(nw.Int64))):
                return column_expr.cast(nw.String)
            return column_expr.cast(target_dtype)
        except Exception:  # noqa: BLE001
            return column_expr.cast(nw.String)

    def _upsert_row(self, df: 'object', idx: Any, attrs: dict) -> 'object':
        if not isinstance(attrs, dict) or not attrs:
            return df

        nw_df = nw.from_native(df, eager_only=True)
        cols = set(nw_df.columns)

        # 1. Key Resolution
        if {'slice_id', 'edge_id'} <= cols:
            key_cols, key_vals = ('slice_id', 'edge_id'), {'slice_id': idx[0], 'edge_id': idx[1]}
            cache_name, df_id_name = '_edge_slice_attr_keys', '_edge_slice_attr_df_id'
        elif 'vertex_id' in cols:
            key_cols, key_vals = ('vertex_id',), {'vertex_id': idx}
            cache_name, df_id_name = '_vertex_attr_ids', '_vertex_attr_df_id'
        elif 'edge_id' in cols:
            key_cols, key_vals = ('edge_id',), {'edge_id': idx}
            cache_name, df_id_name = '_edge_attr_ids', '_edge_attr_df_id'
        elif 'slice_id' in cols:
            key_cols, key_vals = ('slice_id',), {'slice_id': idx}
            cache_name, df_id_name = '_slice_attr_ids', '_slice_attr_df_id'
        else:
            raise ValueError('Cannot infer key columns from DataFrame schema')

        nw_df = self._ensure_attr_columns(nw_df, attrs)

        schema = nw_df.collect_schema()
        upcasts = []
        for c, v in attrs.items():
            if c in schema and v is not None:
                existing_dt = schema[c]
                if not self._is_null_dtype(existing_dt):
                    v_dt = self._dtype_for_value(v)

                    def _is_num(dt):
                        s = str(dt).lower()
                        return any(x in s for x in ('float', 'int', 'decimal', 'uint'))

                    if _is_num(existing_dt) and _is_num(v_dt) and existing_dt != v_dt:
                        try:
                            sup = _get_numeric_supertype(existing_dt, v_dt)
                            if sup and sup != existing_dt:
                                upcasts.append(nw.col(c).cast(sup).alias(c))
                        except Exception:  # noqa: BLE001
                            pass
        if upcasts:
            nw_df = nw_df.with_columns(upcasts)

        cond = None
        for k in key_cols:
            c = nw.col(k) == nw.lit(key_vals[k])
            cond = c if cond is None else (cond & c)

        # 2. Existence Check
        try:
            key_cache = getattr(self, cache_name, None)
            current_df_id = id(df)
            if key_cache is None or getattr(self, df_id_name, None) != current_df_id:
                if key_cols == ('slice_id', 'edge_id'):
                    key_cache = set(
                        zip(
                            nw_df.get_column('slice_id').to_list(),
                            nw_df.get_column('edge_id').to_list(),
                            strict=False,
                        )
                    )
                else:
                    key_cache = set(nw_df.get_column(key_cols[0]).to_list())
                setattr(self, cache_name, key_cache)
                setattr(self, df_id_name, current_df_id)
            cache_key = idx
            exists = cache_key in key_cache
        except Exception:  # noqa: BLE001
            exists = nw_df.filter(cond).shape[0] > 0
            key_cache = None

        if exists:
            schema = nw_df.collect_schema()
            upds = []
            for k, v in attrs.items():
                v2 = self._sanitize_value_for_nw(v)
                tgt_dt = schema[k]
                if self._is_null_dtype(tgt_dt):
                    inf = self._dtype_for_value(v)
                    upds.append(
                        nw.when(cond)
                        .then(nw.lit(v2).cast(inf))
                        .otherwise(nw.col(k).cast(inf))
                        .alias(k)
                    )
                elif self._is_binary_type(tgt_dt):
                    # Localized cast for binary column being updated
                    upds.append(
                        nw.when(cond)
                        .then(nw.lit(v2).cast(nw.String))
                        .otherwise(nw.col(k).cast(nw.String))
                        .alias(k)
                    )
                else:
                    upds.append(
                        nw.when(cond).then(nw.lit(v2).cast(tgt_dt)).otherwise(nw.col(k)).alias(k)
                    )
            new_nw_df = nw_df.with_columns(upds)
            setattr(self, df_id_name, id(nw.to_native(new_nw_df)))
            return nw.to_native(new_nw_df)

        # 3. Insertion & Reconciliation
        schema = nw_df.collect_schema()
        new_row_dict = {**dict.fromkeys(nw_df.columns), **key_vals, **attrs}

        # Helper to detect list types (critical for nested data)
        def _is_list(dt):
            return any(x in str(dt).lower() for x in ('list', 'array'))

        coerced = {
            c: (
                [new_row_dict[c]]
                if _is_list(schema[c])
                and not isinstance(new_row_dict[c], (list, tuple, type(None)))
                else [new_row_dict[c]]
            )
            for c in nw_df.columns
        }

        to_append = nw.DataFrame.from_dict(coerced, backend=nw.get_native_namespace(nw_df))
        right_schema = to_append.collect_schema()

        df_up, app_up = [], []
        for c in nw_df.columns:
            l, r = schema[c], right_schema[c]
            l_null = self._is_null_dtype(l)
            r_null = self._is_null_dtype(r)

            if l_null and not r_null:
                df_up.append(self._safe_nw_cast(nw.col(c), r).alias(c))
            elif r_null and not l_null:
                app_up.append(self._safe_nw_cast(nw.col(c), l).alias(c))
            elif l != r:
                # 1. Binary check (priority for adapters)
                if self._is_binary_type(l) or self._is_binary_type(r):
                    df_up.append(nw.col(c).cast(nw.String).alias(c))
                    app_up.append(nw.col(c).cast(nw.String).alias(c))
                    continue

                # 2. Robust Numeric Supertype Check
                # Use string-based heuristics if the method check is brittle
                def _is_num(dt):
                    s = str(dt).lower()
                    return any(x in s for x in ('float', 'int', 'decimal', 'uint'))

                if _is_num(l) and _is_num(r):
                    try:
                        sup = _get_numeric_supertype(l, r)
                        if sup:
                            df_up.append(self._safe_nw_cast(nw.col(c), sup).alias(c))
                            app_up.append(self._safe_nw_cast(nw.col(c), sup).alias(c))
                            continue
                    except Exception:  # noqa: BLE001
                        pass  # Fall through to string if supertype fails

                # 3. Final Fallback (Incompatible types only)
                df_up.append(nw.col(c).cast(nw.String).alias(c))
                app_up.append(nw.col(c).cast(nw.String).alias(c))

        if df_up:
            nw_df = nw_df.with_columns(df_up)
        if app_up:
            to_append = to_append.with_columns(app_up)

        final_df = nw.concat([nw_df, to_append], how='vertical')
        if key_cache is not None:
            key_cache.add(cache_key)

        setattr(self, df_id_name, id(nw.to_native(final_df)))
        return nw.to_native(final_df)

    def _upsert_rows_bulk(self, df: 'object', updates: dict) -> 'object':
        if not updates:
            return df

        nw_df = nw.from_native(df, eager_only=True)
        cols = set(nw_df.columns)

        # Determine join keys
        if {'slice_id', 'edge_id'} <= cols:
            join_keys = ['slice_id', 'edge_id']
        elif 'vertex_id' in cols:
            join_keys = ['vertex_id']
        elif 'edge_id' in cols:
            join_keys = ['edge_id']
        else:
            join_keys = ['slice_id']

        native = nw.to_native(nw_df)

        # Convert existing rows to a dict keyed by join-key tuple — O(n)
        try:
            existing_rows = native.to_dicts()
        except AttributeError:
            try:
                existing_rows = native.to_dict(orient='records')
            except Exception:  # noqa: BLE001
                existing_rows = []

        def _key(row):
            return tuple(row.get(k) for k in join_keys)

        row_map = {_key(r): r for r in existing_rows}

        # Apply each update in-place or insert new row — O(updates)
        for idx, attrs in updates.items():
            if isinstance(idx, tuple):
                key_part = {'slice_id': idx[0], 'edge_id': idx[1]}
            elif 'vertex_id' in cols:
                key_part = {'vertex_id': idx}
            elif 'edge_id' in cols:
                key_part = {'edge_id': idx}
            else:
                key_part = {'slice_id': idx}

            k = tuple(key_part.get(kk) for kk in join_keys)
            if k in row_map:
                row_map[k].update(attrs)
            else:
                row_map[k] = {**key_part, **attrs}

        # Rebuild DataFrame once — O(n)
        final_records = list(row_map.values())
        backend = nw.get_native_namespace(nw_df)
        result = nw.DataFrame.from_dicts(final_records, backend=backend)
        return nw.to_native(result)

    def _variables_watched_by_vertices(self):
        # set of vertex-attribute names used by vertex-scope policies
        return {
            p['var']
            for p in self.edge_direction_policy.values()
            if p.get('scope', 'edge') == 'vertex'
        }

    def _incident_flexible_edges(self, v):
        # naive scan; optimize later with an index if needed
        out = []
        for eid, rec in self._edges.items():
            if rec.col_idx < 0 or rec.etype == 'hyper':
                continue
            s, t = rec.src, rec.tgt
            if s is None or t is None:
                continue
            if eid in self.edge_direction_policy and (s == v or t == v):
                out.append(eid)
        return out

    def _apply_flexible_direction(self, edge_id):
        pol = self.edge_direction_policy.get(edge_id)
        if not pol:
            return

        _rec = self._edges[edge_id]
        src, tgt = _rec.src, _rec.tgt
        col = _rec.col_idx
        w = float(_rec.weight if _rec.weight is not None else 1.0)

        var = pol['var']
        T = float(pol['threshold'])
        scope = pol.get('scope', 'edge')  # 'edge'|'vertex'
        above = pol.get('above', 's->t')  # 's->t'|'t->s'
        tie = pol.get('tie', 'keep')  # default behavior

        # decide condition and detect tie
        tie_case = False
        if scope == 'edge':
            x = self.get_attr_edge(edge_id, var, None)
            if x is None:
                return
            if x == T:
                tie_case = True
            cond = x > T
        else:
            xs = self.get_attr_vertex(src, var, None)
            xt = self.get_attr_vertex(tgt, var, None)
            if xs is None or xt is None:
                return
            if xs == xt:
                tie_case = True
            cond = (xs - xt) > 0

        M = self._matrix
        si = self._entities[self._resolve_entity_key(src)].row_idx
        ti = self._entities[self._resolve_entity_key(tgt)].row_idx

        if tie_case:
            if tie == 'keep':
                # do nothing - previous signs remain (default)
                return
            if tie == 'undirected':
                # force (+w,+w) while equality holds
                M[(si, col)] = +w
                if src != tgt:
                    M[(ti, col)] = +w
                return
            # force a direction at equality
            cond = True if tie == 's->t' else False

        # rewrite as directed per 'above'
        M[(si, col)] = 0
        M[(ti, col)] = 0
        src_to_tgt = cond if above == 's->t' else (not cond)
        if src_to_tgt:
            M[(si, col)] = +w
            if src != tgt:
                M[(ti, col)] = -w
        else:
            M[(si, col)] = -w
            if src != tgt:
                M[(ti, col)] = +w

    ## Full attribute dict for a single entity

    def get_edge_attrs(self, edge) -> dict:
        """Return the full attribute dict for a single edge.

        Parameters
        ----------
        edge : int | str
            Edge index or edge ID.

        Returns
        -------
        dict
            Attribute dictionary for that edge. Empty if not found.
        """
        # normalize to edge id
        if isinstance(edge, int):
            eid = self._col_to_edge[edge]
        else:
            eid = edge

        df = self.edge_attributes
        if df is None or 'edge_id' not in dataframe_columns(df):
            return {}
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'edge_id', eid))
        return rows[0] if rows else {}

    def get_vertex_attrs(self, vertex) -> dict:
        """Return the full attribute dict for a single vertex.

        Parameters
        ----------
        vertex : str
            Vertex ID.

        Returns
        -------
        dict
            Attribute dictionary for that vertex. Empty if not found.
        """
        df = self.vertex_attributes

        if df is None or 'vertex_id' not in dataframe_columns(df):
            return {}
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'vertex_id', vertex))
        return rows[0] if rows else {}

    ## Bulk attributes

    def get_attr_edges(self, indexes=None) -> dict:
        """Retrieve edge attributes as a dictionary.

        Parameters
        ----------
        indexes : Iterable[int] | None, optional
            Edge indices to retrieve. If None, returns all edges.

        Returns
        -------
        dict[str, dict]
            Mapping of `edge_id` to attribute dictionaries.
        """
        df = self.edge_attributes

        if indexes is not None:
            wanted = [self._col_to_edge[i] for i in indexes]
            df = dataframe_filter_in(df, 'edge_id', wanted)

        return {
            row.get('edge_id'): dict(row)
            for row in dataframe_to_rows(df)
            if row.get('edge_id') is not None
        }

    def get_attr_vertices(self, vertices=None) -> dict:
        """Retrieve vertex (vertex) attributes as a dictionary.

        Parameters
        ----------
        vertices : Iterable[str] | None, optional
            Vertex IDs to retrieve. If None, returns all vertices.

        Returns
        -------
        dict[str, dict]
            Mapping of `vertex_id` to attribute dictionaries.
        """
        df = self.vertex_attributes

        if vertices is not None:
            df = dataframe_filter_in(df, 'vertex_id', list(vertices))

        return {
            row.get('vertex_id'): dict(row)
            for row in dataframe_to_rows(df)
            if row.get('vertex_id') is not None
        }

    def get_attr_from_edges(self, key: str, default=None) -> dict:
        """Extract a specific attribute column for all edges.

        Parameters
        ----------
        key : str
            Attribute column name to extract.
        default : Any, optional
            Value to use if the column or value is missing.

        Returns
        -------
        dict[str, Any]
            Mapping of `edge_id` to attribute values.
        """
        df = self.edge_attributes
        if df is None or 'edge_id' not in dataframe_columns(df):
            return {}

        rows = dataframe_to_rows(df)
        if key not in dataframe_columns(df):
            return {r.get('edge_id'): default for r in rows if r.get('edge_id') is not None}
        return {
            r.get('edge_id'): (r.get(key) if r.get(key) is not None else default)
            for r in rows
            if r.get('edge_id') is not None
        }

    def get_edges_by_attr(self, key: str, value) -> list:
        """Retrieve all edges where a given attribute equals a specific value.

        Parameters
        ----------
        key : str
            Attribute column name to filter on.
        value : Any
            Value to match.

        Returns
        -------
        list[str]
            Edge IDs where the attribute equals `value`.
        """
        df = self.edge_attributes
        if df is None or key not in dataframe_columns(df):
            return []

        return [
            r.get('edge_id')
            for r in dataframe_to_rows(df)
            if r.get('edge_id') is not None and r.get(key) == value
        ]

    def get_graph_attributes(self) -> dict:
        """Return a shallow copy of the graph-level attributes dictionary.

        Returns
        -------
        dict
            Shallow copy of global graph metadata.

        Notes
        -----
        Returned value is a shallow copy to prevent external mutation.
        """
        return dict(self.graph_attributes)

    def set_edge_slice_attrs_bulk(self, slice_id, items):
        """Upsert edge-slice attributes for a single slice in bulk.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        items : Iterable[tuple[str, dict]] | dict[str, dict]
            Iterable or mapping of `(edge_id, attrs)` pairs.
        """

        # normalize
        rows = []
        if isinstance(items, dict):
            it = items.items()
        else:
            it = items
        for eid, attrs in it:
            if not isinstance(attrs, dict) or not attrs:
                continue
            r = {'slice_id': slice_id, 'edge_id': eid}
            r.update(attrs)
            if 'weight' in r:
                try:
                    r['weight'] = float(r['weight'])
                except Exception:  # noqa: BLE001
                    pass
            rows.append(r)
        if not rows:
            return

        self.edge_slice_attributes = dataframe_upsert_rows(
            self.edge_slice_attributes,
            rows,
            ['slice_id', 'edge_id'],
            backend=getattr(self, '_annotations_backend', None),
        )

        # legacy mirror
        if any('weight' in row for row in rows):
            self.slice_edge_weights.setdefault(slice_id, {})
            for r in rows:
                w = r.get('weight')
                if w is not None:
                    self.slice_edge_weights[slice_id][r['edge_id']] = float(w)
