"""Attribute storage and accessors."""

import math
from typing import Any

import narwhals as nw

from .._support.dataframe_backend import (
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
    """Attribute accessors and upsert helpers (graph/vertex/edge/slice/edge-slice)."""

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

        if self._vertex_key_enabled():
            old_key = self._current_key_of_vertex(vertex_id)
            merged = {
                f: (
                    clean[f]
                    if f in clean
                    else AttributesClass.get_attr_vertex(self, vertex_id, f, None)
                )
                for f in self._vertex_key_fields
            }
            new_key = self._build_key_from_attrs(merged)
            if new_key is not None:
                owner = self._vertex_key_index.get(new_key)
                if owner is not None and owner != vertex_id:
                    raise ValueError(
                        f'Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}'
                    )

        self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, clean)

        watched = self._variables_watched_by_vertices()
        if watched and any(k in watched for k in clean):
            for eid in self._incident_flexible_edges(vertex_id):
                self._apply_flexible_direction(eid)

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
                    f: (
                        attrs[f]
                        if f in attrs
                        else AttributesClass.get_attr_vertex(self, vid, f, None)
                    )
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
                new_key, old_key = new_keys[vid], old_keys[vid]
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
        _check_reserved_collision(self._EDGE_RESERVED, attrs, kind='edge-slice', allow=('weight',))
        clean = dict(attrs)
        if not clean:
            return
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
                clean['weight'] = float(clean['weight'])
            except (TypeError, ValueError):
                pass
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )
        self._sync_slice_edge_weights_for_rows(
            slice_id, [{'slice_id': slice_id, 'edge_id': edge_id, **clean}]
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

    def set_slice_edge_weight(self, slice_id, edge_id, weight):
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
        na, ea, ela = self.vertex_attributes, self.edge_attributes, self.edge_slice_attributes

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

        bad_edge_slice = []
        if ela is not None and {'slice_id', 'edge_id'} <= set(dataframe_columns(ela)):
            for r in dataframe_to_rows(ela):
                lid, eid = r.get('slice_id'), r.get('edge_id')
                if lid not in self._slices or eid not in edge_ids:
                    bad_edge_slice.append((lid, eid))

        return {
            'extra_vertex_rows': [i for i in vertex_attr_ids if i not in vertex_ids],
            'extra_edge_rows': [i for i in edge_attr_ids if i not in edge_ids],
            'missing_vertex_rows': [i for i in vertex_ids if i not in vertex_attr_ids],
            'missing_edge_rows': [i for i in edge_ids if i not in edge_attr_ids],
            'invalid_edge_slice_rows': bad_edge_slice,
        }

    # ── dtype / schema helpers ────────────────────────────────────────────────

    def _dtype_for_value(self, v, *, prefer='narwhals'):
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
            inner = self._dtype_for_value(v[0], prefer='narwhals') if len(v) else nw.String
            return nw.List(nw.String if inner == nw.Unknown else inner)
        if isinstance(v, dict):
            return nw.Object
        return nw.String

    def _is_null_dtype(self, dtype) -> bool:
        if dtype == nw.Unknown:
            return True
        dt_type = type(dtype) if not isinstance(dtype, type) else dtype
        return dt_type == nw.Unknown

    def _ensure_attr_columns(self, df, attrs: dict) -> nw.DataFrame[Any]:
        nw_df = nw.from_native(df, eager_only=True)
        schema = nw_df.collect_schema()
        for col, val in attrs.items():
            target = self._dtype_for_value(val, prefer='narwhals')
            if col not in schema:
                try:
                    nw_df = nw_df.with_columns(nw.lit(None).cast(target).alias(col))
                except (AttributeError, TypeError, ValueError):
                    nw_df = nw_df.with_columns(nw.lit(None).alias(col))
            else:
                cur = schema[col]
                if self._is_null_dtype(cur) and not self._is_null_dtype(target):
                    try:
                        nw_df = nw_df.with_columns(nw.col(col).cast(target))
                    except (AttributeError, TypeError, ValueError):
                        pass
        return nw_df

    def _sanitize_value_for_nw(self, v):
        if isinstance(v, (list, tuple, dict)):
            import json

            return json.dumps(v, ensure_ascii=False)
        return v

    def _is_binary_type(self, dt) -> bool:
        if isinstance(dt, (nw.Binary, nw.dtypes.Binary)):
            return True
        s = str(dt).lower()
        return any(kw in s for kw in ('binary', 'blob', 'byte'))

    def _upsert_row(self, df: 'object', idx: Any, attrs: dict) -> 'object':
        if not isinstance(attrs, dict) or not attrs:
            return df
        cols = set(dataframe_columns(df))
        if {'slice_id', 'edge_id'} <= cols:
            key_cols = ('slice_id', 'edge_id')
            key_vals = {'slice_id': idx[0], 'edge_id': idx[1]}
            df_id_name = '_edge_slice_attr_df_id'
        elif 'vertex_id' in cols:
            key_cols = ('vertex_id',)
            key_vals = {'vertex_id': idx}
            df_id_name = '_vertex_attr_df_id'
        elif 'edge_id' in cols:
            key_cols = ('edge_id',)
            key_vals = {'edge_id': idx}
            df_id_name = '_edge_attr_df_id'
        elif 'slice_id' in cols:
            key_cols = ('slice_id',)
            key_vals = {'slice_id': idx}
            df_id_name = '_slice_attr_df_id'
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
        return {
            p['var']
            for p in self.edge_direction_policy.values()
            if p.get('scope', 'edge') == 'vertex'
        }

    def _incident_flexible_edges(self, v):
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
        w = float(_rec.weight if _rec.weight is not None else 1.0)

        var = pol['var']
        T = float(pol['threshold'])
        scope = pol.get('scope', 'edge')
        above = pol.get('above', 's->t')
        tie = pol.get('tie', 'keep')

        tie_case = False
        if scope == 'edge':
            x = AttributesClass.get_attr_edge(self, edge_id, var, None)
            if x is None:
                return
            if x == T:
                tie_case = True
            cond = x > T
        else:
            xs = AttributesClass.get_attr_vertex(self, src, var, None)
            xt = AttributesClass.get_attr_vertex(self, tgt, var, None)
            if xs is None or xt is None:
                return
            if xs == xt:
                tie_case = True
            cond = (xs - xt) > 0

        # Persist the resolved column into the record's coeffs so the lazily
        # rebuilt incidence matrix reflects it (records are the source of truth).
        def _resolve(sval, tval):
            coeffs = {src: sval}
            if src != tgt:
                coeffs[tgt] = tval
            _rec.coeffs = coeffs
            self._mark_matrix_dirty()
            self._invalidate_sparse_caches()

        if tie_case:
            if tie == 'keep':
                return
            if tie == 'undirected':
                _resolve(+w, +w)
                return
            cond = True if tie == 's->t' else False

        src_to_tgt = cond if above == 's->t' else (not cond)
        if src_to_tgt:
            _resolve(+w, -w)
        else:
            _resolve(-w, +w)

    # ── full / bulk reads ─────────────────────────────────────────────────────

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
        eid = self._col_to_edge[edge] if isinstance(edge, int) else edge
        rows = dataframe_to_rows(dataframe_filter_eq(self.edge_attributes, 'edge_id', eid))
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
        rows = dataframe_to_rows(dataframe_filter_eq(self.vertex_attributes, 'vertex_id', vertex))
        if not rows:
            return {}
        return {k: v for k, v in rows[0].items() if k != 'vertex_id' and v is not None}

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
        rows = dataframe_to_rows(self.edge_attributes)
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
        rows = dataframe_to_rows(self.vertex_attributes)
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
        rows = []
        it = items.items() if isinstance(items, dict) else items
        for eid, attrs in it:
            if not isinstance(attrs, dict) or not attrs:
                continue
            r = {'slice_id': slice_id, 'edge_id': eid, **attrs}
            if 'weight' in r:
                try:
                    r['weight'] = float(r['weight'])
                except (TypeError, ValueError):
                    pass
            rows.append(r)
        if not rows:
            return
        updates = {
            (row['slice_id'], row['edge_id']): {
                k: v for k, v in row.items() if k not in {'slice_id', 'edge_id'}
            }
            for row in rows
        }
        self.edge_slice_attributes = self._upsert_rows_bulk(self.edge_slice_attributes, updates)
        self._sync_slice_edge_weights_for_rows(slice_id, rows)


# Methods exposed verbatim on the ``G.attrs`` namespace.
_ATTR_DELEGATED = (
    'set_graph_attribute',
    'get_graph_attribute',
    'get_graph_attributes',
    'set_vertex_attrs',
    'set_vertex_attrs_bulk',
    'get_vertex_attrs',
    'get_attr_vertex',
    'get_attr_vertices',
    'set_edge_attrs',
    'set_edge_attrs_bulk',
    'get_edge_attrs',
    'get_attr_edge',
    'get_attr_edges',
    'get_attr_from_edges',
    'get_edges_by_attr',
    'set_slice_attrs',
    'get_slice_attr',
    'set_edge_slice_attrs',
    'set_edge_slice_attrs_bulk',
    'get_edge_slice_attr',
    'set_slice_edge_weight',
    'get_effective_edge_weight',
    'audit_attributes',
)


class AttributesAccessor:
    """Namespace for graph, vertex, edge, and slice annotations (``G.attrs``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph


def _install_attr_delegators():
    for _name in _ATTR_DELEGATED:

        def _make(name):
            target = getattr(AttributesClass, name)

            def _delegator(self, *args, **kwargs):
                return target(self._G, *args, **kwargs)

            _delegator.__name__ = name
            _delegator.__qualname__ = f'AttributesAccessor.{name}'
            _delegator.__doc__ = target.__doc__
            return _delegator

        setattr(AttributesAccessor, _name, _make(_name))


_install_attr_delegators()
