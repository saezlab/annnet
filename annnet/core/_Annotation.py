import math
from typing import Any

import narwhals as nw

from .._dataframe_backend import (
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_eq,
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


def _check_reserved_collision(reserved, attrs, *, kind, allow=()):
    """Raise ``ValueError`` if any reserved key appears in ``attrs``.

    ``reserved`` is the set of structural / signal keys for the relevant
    table; ``allow`` lets a caller whitelist a subset (e.g. the
    edge-slice writer keeps ``weight`` even though it is reserved).
    ``kind`` is a short noun used in the error message (``"edge"``,
    ``"vertex"``, ``"slice"``, ``"edge-slice"``).
    """
    if not attrs:
        return
    allow = set(allow)
    bad = sorted(k for k in attrs if k in reserved and k not in allow)
    if bad:
        raise ValueError(
            f'{kind} attributes use reserved key(s): {bad!r}. '
            f'These names are part of the structural / dispatch contract; '
            f'rename your attribute(s) to use a different key.'
        )


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

    def set_vertex_attrs(self, vertex_id, **attrs):
        """Upsert pure vertex attributes (non-structural) into the vertex table.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.
        **attrs
            Attribute key/value pairs.

        Raises
        ------
        ValueError
            If any key is structurally reserved (e.g. ``vertex_id``).
        """
        _check_reserved_collision(self._vertex_RESERVED, attrs, kind='vertex')
        clean = dict(attrs)
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

        for _vid, attrs in updates.items():
            _check_reserved_collision(self._vertex_RESERVED, attrs, kind='vertex')

        clean_updates = {vid: dict(attrs) for vid, attrs in updates.items() if attrs}

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
        df = self.vertex_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'vertex_id', vertex_id))
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

    def set_edge_attrs(self, edge_id, **attrs):
        """Upsert pure edge attributes (non-structural) into the edge DF.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        **attrs
            Attribute key/value pairs.

        Raises
        ------
        ValueError
            If any key is structurally reserved (e.g. ``edge_id``,
            ``source``, ``target``, ``weight``, ``members``, ``head``,
            ``tail``, ``flexible``).
        """
        _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge')
        clean = dict(attrs)
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

        for _eid, attrs in updates.items():
            _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge')

        clean_updates = {eid: dict(attrs) for eid, attrs in updates.items() if attrs}

        if not clean_updates:
            return

        self.edge_attributes = self._upsert_rows_bulk(self.edge_attributes, clean_updates)

        policy_map = self.edge_direction_policy
        affected_edges = set()
        for eid, attrs in clean_updates.items():
            pol = policy_map.get(eid)
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
        df = self.edge_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'edge_id', edge_id))
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

    def set_slice_attrs(self, slice_id, **attrs):
        """Upsert pure slice attributes.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        **attrs
            Attribute key/value pairs. Structural keys are ignored.
        """
        _check_reserved_collision(self._slice_RESERVED, attrs, kind='slice')
        clean = dict(attrs)
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
        df = self.slice_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'slice_id', slice_id))
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

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
        # 'weight' is the legitimate per-slice override; everything else
        # reserved must raise so users don't silently lose data.
        _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge-slice', allow=('weight',))
        clean = dict(attrs)
        if not clean:
            return

        # Normalize hot keys (intern) and avoid float dtype surprises for 'weight'
        try:
            import sys as _sys

            if isinstance(slice_id, str):
                slice_id = _sys.intern(slice_id)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
        except (AttributeError, TypeError):
            pass
        if 'weight' in clean:
            try:
                # cast once to float to reduce dtype mismatch churn inside _upsert_row
                clean['weight'] = float(clean['weight'])
            except (TypeError, ValueError):
                # leave as-is if not coercible; behavior stays identical
                pass

        # Upsert via central helper (keeps exact behavior, schema handling, and caching)
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )
        self._sync_slice_edge_weights_for_rows(
            slice_id,
            [{'slice_id': slice_id, 'edge_id': edge_id, **clean}],
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
        df = self.edge_slice_attributes
        if df is None or key not in dataframe_columns(df):
            return default
        rows = [
            row
            for row in dataframe_to_rows(df)
            if row.get('slice_id') == slice_id and row.get('edge_id') == edge_id
        ]
        if not rows:
            return default
        val = rows[0].get(key, None)
        return default if val is None else val

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
        # Call the canonical setter on AttributesClass directly. ``self`` is the
        # graph (from the AttributesAccessor.set_slice_edge_weight shim that
        # passes ``self._G``), so going through ``self.set_edge_slice_attrs``
        # would hit the deprecated-attribute barrier.
        AttributesClass.set_edge_slice_attrs(self, slice_id, edge_id, weight=float(weight))

    def get_effective_edge_weight(self, edge_id, slice=None):
        """Resolve the effective weight for an edge, optionally within a slice.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        slice : str, optional
            Slice to read the override from. When omitted, the graph's
            currently active slice is used. Pass an explicit slice ID to
            override the active-slice resolution.

        Returns
        -------
        float
            Effective weight.
        """
        if slice is None:
            slice = self._current_slice
        if slice is not None:
            df = self.edge_slice_attributes
            if df is not None and {'slice_id', 'edge_id', 'weight'} <= set(dataframe_columns(df)):
                for row in dataframe_to_rows(df):
                    if row.get('slice_id') != slice or row.get('edge_id') != edge_id:
                        continue
                    w = row.get('weight', None)
                    if w is not None and not (isinstance(w, float) and math.isnan(w)):
                        return float(w)

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

        if na is not None and 'vertex_id' in dataframe_columns(na):
            vertex_attr_ids = {
                row.get('vertex_id')
                for row in dataframe_to_rows(na)
                if row.get('vertex_id') is not None
            }
        else:
            vertex_attr_ids = set()

        if ea is not None and 'edge_id' in dataframe_columns(ea):
            edge_attr_ids = {
                row.get('edge_id')
                for row in dataframe_to_rows(ea)
                if row.get('edge_id') is not None
            }
        else:
            edge_attr_ids = set()

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

    def _dtype_for_value(self, v, *, prefer='narwhals'):
        """INTERNAL: Infer an appropriate Narwhals dtype class for value `v`."""
        import enum

        # Narwhals dtype classes
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
            inner = self._dtype_for_value(v[0], prefer='narwhals') if len(v) else nw.String
            return nw.List(nw.String if inner == nw.Unknown else inner)
        if isinstance(v, dict):
            return nw.Object
        return nw.String

    def _is_null_dtype(self, dtype) -> bool:
        """Check if a dtype represents a null/unknown type."""
        # Catch Narwhals Unknown and equivalent null-like classes.
        if dtype == nw.Unknown:
            return True
        dt_type = type(dtype) if not isinstance(dtype, type) else dtype
        return dt_type == nw.Unknown

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

        impl = nw_df.implementation
        impl.is_polars()

        for col, val in attrs.items():
            # Use Narwhals dtypes for logic to avoid backend-mismatch pitfalls
            target = self._dtype_for_value(val, prefer='narwhals')

            if col not in schema:
                # Add new column with appropriate null-casting
                try:
                    nw_df = nw_df.with_columns(nw.lit(None).cast(target).alias(col))
                except (AttributeError, TypeError, ValueError):
                    nw_df = nw_df.with_columns(nw.lit(None).alias(col))
            else:
                # Upgrade logic: ONLY cast if the existing column is a Null/Unknown type
                cur = schema[col]
                if self._is_null_dtype(cur) and not self._is_null_dtype(target):
                    try:
                        nw_df = nw_df.with_columns(nw.col(col).cast(target))
                    except (AttributeError, TypeError, ValueError):
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
        except (AttributeError, TypeError, ValueError):
            return column_expr.cast(nw.String)

    def _upsert_row(self, df: 'object', idx: Any, attrs: dict) -> 'object':
        if not isinstance(attrs, dict) or not attrs:
            return df
        cols = set(dataframe_columns(df))
        if {'slice_id', 'edge_id'} <= cols:
            key_cols = ('slice_id', 'edge_id')
            key_vals = {'slice_id': idx[0], 'edge_id': idx[1]}
            _cache_name, df_id_name = '_edge_slice_attr_keys', '_edge_slice_attr_df_id'
        elif 'vertex_id' in cols:
            key_cols = ('vertex_id',)
            key_vals = {'vertex_id': idx}
            _cache_name, df_id_name = '_vertex_attr_ids', '_vertex_attr_df_id'
        elif 'edge_id' in cols:
            key_cols = ('edge_id',)
            key_vals = {'edge_id': idx}
            _cache_name, df_id_name = '_edge_attr_ids', '_edge_attr_df_id'
        elif 'slice_id' in cols:
            key_cols = ('slice_id',)
            key_vals = {'slice_id': idx}
            _cache_name, df_id_name = '_slice_attr_ids', '_slice_attr_df_id'
        else:
            raise ValueError('Cannot infer key columns from DataFrame schema')

        out = dataframe_upsert_rows(df, [{**key_vals, **attrs}], key_cols)
        setattr(self, df_id_name, id(out))
        return out

    def _upsert_rows_bulk(self, df: 'object', updates: dict) -> 'object':
        if not updates:
            return df
        cols = set(dataframe_columns(df))
        if {'slice_id', 'edge_id'} <= cols:
            join_keys = ('slice_id', 'edge_id')
        elif 'vertex_id' in cols:
            join_keys = ('vertex_id',)
        elif 'edge_id' in cols:
            join_keys = ('edge_id',)
        else:
            join_keys = ('slice_id',)

        update_records = []
        for idx, attrs in updates.items():
            if isinstance(idx, tuple):
                record = {'slice_id': idx[0], 'edge_id': idx[1], **attrs}
            elif 'vertex_id' in cols:
                record = {'vertex_id': idx, **attrs}
            elif 'edge_id' in cols:
                record = {'edge_id': idx, **attrs}
            else:
                record = {'slice_id': idx, **attrs}
            update_records.append(record)

        return dataframe_upsert_rows(df, update_records, join_keys)

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
            if rec.direction_policy is not None and (s == v or t == v):
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
            x = AttributesClass.get_attr_edge(self, edge_id, var, None)
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

        # Polars fast-path
        rows = dataframe_to_rows(dataframe_filter_eq(df, 'edge_id', eid))
        if not rows:
            return {}
        return {k: v for k, v in rows[0].items() if k != 'edge_id' and v is not None}

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

        rows = dataframe_to_rows(dataframe_filter_eq(df, 'vertex_id', vertex))
        if not rows:
            return {}
        return {k: v for k, v in rows[0].items() if k != 'vertex_id' and v is not None}

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

        rows = dataframe_to_rows(df)
        if indexes is not None:
            wanted = {self._col_to_edge[i] for i in indexes}
            rows = [row for row in rows if row.get('edge_id') in wanted]
        return {r.get('edge_id'): dict(r) for r in rows if r.get('edge_id') is not None}

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

        rows = dataframe_to_rows(df)
        if vertices is not None:
            wanted = set(vertices)
            rows = [row for row in rows if row.get('vertex_id') in wanted]
        return {r.get('vertex_id'): dict(r) for r in rows if r.get('vertex_id') is not None}

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
        if df is None:
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
        rows = dataframe_to_rows(df)
        return [
            r.get('edge_id') for r in rows if r.get('edge_id') is not None and r.get(key) == value
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
                except (TypeError, ValueError):
                    pass
            rows.append(r)
        if not rows:
            return

        # start from current DF
        updates = {
            (row['slice_id'], row['edge_id']): {
                k: v for k, v in row.items() if k not in {'slice_id', 'edge_id'}
            }
            for row in rows
        }
        self.edge_slice_attributes = self._upsert_rows_bulk(self.edge_slice_attributes, updates)

        self._sync_slice_edge_weights_for_rows(slice_id, rows)


class AttributesAccessor:
    """Namespace for graph, vertex, edge, and slice annotations."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def set_graph_attribute(self, *args, **kwargs):
        return AttributesClass.set_graph_attribute(self._G, *args, **kwargs)

    def get_graph_attribute(self, *args, **kwargs):
        return AttributesClass.get_graph_attribute(self._G, *args, **kwargs)

    def get_graph_attributes(self, *args, **kwargs):
        return AttributesClass.get_graph_attributes(self._G, *args, **kwargs)

    def set_vertex_attrs(self, *args, **kwargs):
        return AttributesClass.set_vertex_attrs(self._G, *args, **kwargs)

    def set_vertex_attrs_bulk(self, *args, **kwargs):
        return AttributesClass.set_vertex_attrs_bulk(self._G, *args, **kwargs)

    def get_vertex_attrs(self, *args, **kwargs):
        return AttributesClass.get_vertex_attrs(self._G, *args, **kwargs)

    def get_attr_vertex(self, *args, **kwargs):
        return AttributesClass.get_attr_vertex(self._G, *args, **kwargs)

    def get_attr_vertices(self, *args, **kwargs):
        return AttributesClass.get_attr_vertices(self._G, *args, **kwargs)

    def set_edge_attrs(self, *args, **kwargs):
        return AttributesClass.set_edge_attrs(self._G, *args, **kwargs)

    def set_edge_attrs_bulk(self, *args, **kwargs):
        return AttributesClass.set_edge_attrs_bulk(self._G, *args, **kwargs)

    def get_edge_attrs(self, *args, **kwargs):
        return AttributesClass.get_edge_attrs(self._G, *args, **kwargs)

    def get_attr_edge(self, *args, **kwargs):
        return AttributesClass.get_attr_edge(self._G, *args, **kwargs)

    def get_attr_edges(self, *args, **kwargs):
        return AttributesClass.get_attr_edges(self._G, *args, **kwargs)

    def get_attr_from_edges(self, *args, **kwargs):
        return AttributesClass.get_attr_from_edges(self._G, *args, **kwargs)

    def get_edges_by_attr(self, *args, **kwargs):
        return AttributesClass.get_edges_by_attr(self._G, *args, **kwargs)

    def set_slice_attrs(self, *args, **kwargs):
        return AttributesClass.set_slice_attrs(self._G, *args, **kwargs)

    def get_slice_attr(self, *args, **kwargs):
        return AttributesClass.get_slice_attr(self._G, *args, **kwargs)

    def set_edge_slice_attrs(self, *args, **kwargs):
        return AttributesClass.set_edge_slice_attrs(self._G, *args, **kwargs)

    def set_edge_slice_attrs_bulk(self, *args, **kwargs):
        return AttributesClass.set_edge_slice_attrs_bulk(self._G, *args, **kwargs)

    def get_edge_slice_attr(self, *args, **kwargs):
        return AttributesClass.get_edge_slice_attr(self._G, *args, **kwargs)

    def set_slice_edge_weight(self, *args, **kwargs):
        return AttributesClass.set_slice_edge_weight(self._G, *args, **kwargs)

    def get_effective_edge_weight(self, *args, **kwargs):
        return AttributesClass.get_effective_edge_weight(self._G, *args, **kwargs)

    def audit_attributes(self, *args, **kwargs):
        return AttributesClass.audit_attributes(self._G, *args, **kwargs)
