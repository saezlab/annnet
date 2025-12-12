import math

import polars as pl

from ._helpers import _get_numeric_supertype


class AttributesClass:
    # Attributes & weights

    def set_graph_attribute(self, key, value):
        """Set a graph-level attribute.

        Parameters
        --
        key : str
        value : Any

        """
        self.graph_attributes[key] = value

    def get_graph_attribute(self, key, default=None):
        """Get a graph-level attribute.

        Parameters
        --
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        return self.graph_attributes.get(key, default)

    def set_vertex_attrs(self, vertex_id, **attrs):
        """Upsert pure vertex attributes (non-structural) into the vertex DF [DataFrame]."""
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
                        f"Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}"
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
            old_key = old_key if "old_key" in locals() else None
            if old_key != new_key:
                if old_key is not None and self._vertex_key_index.get(old_key) == vertex_id:
                    self._vertex_key_index.pop(old_key, None)
                if new_key is not None:
                    self._vertex_key_index[new_key] = vertex_id

    def get_attr_vertex(self, vertex_id, key, default=None):
        """Get a single vertex attribute (scalar) or default if missing.

        Parameters
        --
        vertex_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.vertex_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("vertex_id") == vertex_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_vertex_attribute(self, vertex_id, attribute):  # legacy alias
        """(Legacy alias) Get a single vertex attribute from the Polars DF [DataFrame].

        Parameters
        --
        vertex_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        ---
        Any or None
            Scalar value if present, else ``None``.

        See Also

        get_attr_vertex

        """
        # allow Attr enums
        attribute = getattr(attribute, "value", attribute)

        df = self.vertex_attributes
        if not isinstance(df, pl.DataFrame):
            return None
        if df.height == 0 or "vertex_id" not in df.columns or attribute not in df.columns:
            return None

        rows = df.filter(pl.col("vertex_id") == vertex_id)
        if rows.height == 0:
            return None

        s = rows.get_column(attribute)
        return s.item(0) if s.len() else None

    def set_edge_attrs(self, edge_id, **attrs):
        """Upsert pure edge attributes (non-structural) into the edge DF.

        Parameters
        --
        edge_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.

        """
        # keep attributes table pure: strip structural keys
        clean = {k: v for k, v in attrs.items() if k not in self._EDGE_RESERVED}
        if clean:
            self.edge_attributes = self._upsert_row(self.edge_attributes, edge_id, clean)
        pol = self.edge_direction_policy.get(edge_id)
        if pol and pol.get("scope", "edge") == "edge" and pol["var"] in clean:
            self._apply_flexible_direction(edge_id)

    def get_attr_edge(self, edge_id, key, default=None):
        """Get a single edge attribute (scalar) or default if missing.

        Parameters
        --
        edge_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.edge_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("edge_id") == edge_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def get_edge_attribute(self, edge_id, attribute):  # legacy alias
        """(Legacy alias) Get a single edge attribute from the Polars DF [DataFrame].

        Parameters
        --
        edge_id : str
        attribute : str or enum.Enum
            Column name or Enum with ``.value``.

        Returns
        ---
        Any or None
            Scalar value if present, else ``None``.

        See Also

        get_attr_edge

        """
        # allow Attr enums
        attribute = getattr(attribute, "value", attribute)

        df = self.edge_attributes
        if not isinstance(df, pl.DataFrame):
            return None
        if df.height == 0 or "edge_id" not in df.columns or attribute not in df.columns:
            return None

        rows = df.filter(pl.col("edge_id") == edge_id)
        if rows.height == 0:
            return None

        s = rows.get_column(attribute)
        return s.item(0) if s.len() else None

    def set_slice_attrs(self, slice_id, **attrs):
        """Upsert pure slice attributes.

        Parameters
        --
        slice_id : str
        **attrs
            Key/value attributes. Structural keys are ignored.

        """
        clean = {k: v for k, v in attrs.items() if k not in self._slice_RESERVED}
        if clean:
            self.slice_attributes = self._upsert_row(self.slice_attributes, slice_id, clean)

    def get_slice_attr(self, slice_id, key, default=None):
        """Get a single slice attribute (scalar) or default if missing.

        Parameters
        --
        slice_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.slice_attributes
        if key not in df.columns:
            return default
        rows = df.filter(pl.col("slice_id") == slice_id)
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_edge_slice_attrs(self, slice_id, edge_id, **attrs):
        """Upsert per-slice attributes for a specific edge.

        Parameters
        --
        slice_id : str
        edge_id : str
        **attrs
            Pure attributes. Structural keys are ignored (except 'weight', which is allowed here).

        """
        # allow 'weight' through; keep ignoring true structural keys
        clean = {
            k: v for k, v in attrs.items() if (k not in self._EDGE_RESERVED) or (k == "weight")
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
        except Exception:
            pass
        if "weight" in clean:
            try:
                # cast once to float to reduce dtype mismatch churn inside _upsert_row
                clean["weight"] = float(clean["weight"])
            except Exception:
                # leave as-is if not coercible; behavior stays identical
                pass

        # Ensure edge_slice_attributes compares strings to strings (defensive against prior bad writes),
        # but only cast when actually needed (skip no-op with_columns).
        df = self.edge_slice_attributes
        if isinstance(df, pl.DataFrame) and df.height > 0:
            to_cast = []
            if "slice_id" in df.columns and df.schema["slice_id"] != pl.Utf8:
                to_cast.append(pl.col("slice_id").cast(pl.Utf8))
            if "edge_id" in df.columns and df.schema["edge_id"] != pl.Utf8:
                to_cast.append(pl.col("edge_id").cast(pl.Utf8))
            if to_cast:
                df = df.with_columns(*to_cast)
                self.edge_slice_attributes = df  # reassign only when changed

        # Upsert via central helper (keeps exact behavior, schema handling, and caching)
        self.edge_slice_attributes = self._upsert_row(
            self.edge_slice_attributes, (slice_id, edge_id), clean
        )

    def get_edge_slice_attr(self, slice_id, edge_id, key, default=None):
        """Get a per-slice attribute for an edge.

        Parameters
        --
        slice_id : str
        edge_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.edge_slice_attributes
        if key not in df.columns:
            return default
        rows = df.filter((pl.col("slice_id") == slice_id) & (pl.col("edge_id") == edge_id))
        if rows.height == 0:
            return default
        val = rows.select(pl.col(key)).to_series()[0]
        return default if val is None else val

    def set_slice_edge_weight(self, slice_id, edge_id, weight):  # legacy weight helper
        """Set a legacy per-slice weight override for an edge.

        Parameters
        --
        slice_id : str
        edge_id : str
        weight : float

        Raises
        --
        KeyError
            If the slice or edge does not exist.

        See Also

        get_effective_edge_weight

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")
        self.slice_edge_weights[slice_id][edge_id] = float(weight)

    def get_effective_edge_weight(self, edge_id, slice=None):
        """Resolve the effective weight for an edge, optionally within a slice.

        Parameters
        --
        edge_id : str
        slice : str, optional
            If provided, return the slice override if present; otherwise global weight.

        Returns
        ---
        float
            Effective weight.

        """
        if slice is not None:
            df = self.edge_slice_attributes
            if (
                isinstance(df, pl.DataFrame)
                and df.height > 0
                and {"slice_id", "edge_id", "weight"} <= set(df.columns)
            ):
                rows = df.filter(
                    (pl.col("slice_id") == slice) & (pl.col("edge_id") == edge_id)
                ).select("weight")
                if rows.height > 0:
                    w = rows.to_series()[0]
                    if w is not None and not (isinstance(w, float) and math.isnan(w)):
                        return float(w)

            # fallback to legacy dict if present
            w2 = self.slice_edge_weights.get(slice, {}).get(edge_id, None)
            if w2 is not None:
                return float(w2)

        return float(self.edge_weights[edge_id])

    def audit_attributes(self):
        """Audit attribute tables for extra/missing rows and invalid edge-slice pairs.

        Returns
        ---
        dict
            {
            'extra_vertex_rows': list[str],
            'extra_edge_rows': list[str],
            'missing_vertex_rows': list[str],
            'missing_edge_rows': list[str],
            'invalid_edge_slice_rows': list[tuple[str, str]],
            }

        """
        vertex_ids = {eid for eid, t in self.entity_types.items() if t == "vertex"}
        edge_ids = set(self.edge_to_idx.keys())

        na = self.vertex_attributes
        ea = self.edge_attributes
        ela = self.edge_slice_attributes

        vertex_attr_ids = (
            set(na.select("vertex_id").to_series().to_list())
            if isinstance(na, pl.DataFrame) and na.height > 0 and "vertex_id" in na.columns
            else set()
        )
        edge_attr_ids = (
            set(ea.select("edge_id").to_series().to_list())
            if isinstance(ea, pl.DataFrame) and ea.height > 0 and "edge_id" in ea.columns
            else set()
        )

        extra_vertex_rows = [i for i in vertex_attr_ids if i not in vertex_ids]
        extra_edge_rows = [i for i in edge_attr_ids if i not in edge_ids]
        missing_vertex_rows = [i for i in vertex_ids if i not in vertex_attr_ids]
        missing_edge_rows = [i for i in edge_ids if i not in edge_attr_ids]

        bad_edge_slice = []
        if (
            isinstance(ela, pl.DataFrame)
            and ela.height > 0
            and {"slice_id", "edge_id"} <= set(ela.columns)
        ):
            for lid, eid in ela.select(["slice_id", "edge_id"]).iter_rows():
                if lid not in self._slices or eid not in edge_ids:
                    bad_edge_slice.append((lid, eid))

        return {
            "extra_vertex_rows": extra_vertex_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_vertex_rows": missing_vertex_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_slice_rows": bad_edge_slice,
        }

    def _pl_dtype_for_value(self, v):
        """INTERNAL: Infer an appropriate Polars dtype for a Python value.

        Parameters
        --
        v : Any

        Returns
        ---
        polars.datatypes.DataType
            One of ``pl.Null``, ``pl.Boolean``, ``pl.Int64``, ``pl.Float64``,
            ``pl.Utf8``, ``pl.Binary``, ``pl.Object``, or ``pl.List(inner)``.

        Notes
        -
        - Enums are mapped to ``pl.Object`` (useful for categorical enums).
        - Lists/tuples infer inner dtype from the first element (defaults to ``Utf8``).

        """
        import enum

        if v is None:
            return pl.Null
        if isinstance(v, bool):
            return pl.Boolean
        if isinstance(v, int) and not isinstance(v, bool):
            return pl.Int64
        if isinstance(v, float):
            return pl.Float64
        if isinstance(v, enum.Enum):
            return pl.Object  # important for EdgeType
        if isinstance(v, (bytes, bytearray)):
            return pl.Binary
        if isinstance(v, (list, tuple)):
            inner = self._pl_dtype_for_value(v[0]) if len(v) else pl.Utf8
            return pl.List(pl.Utf8 if inner == pl.Null else inner)
        if isinstance(v, dict):
            return pl.Object
        return pl.Utf8

    def _ensure_attr_columns(self, df: pl.DataFrame, attrs: dict) -> pl.DataFrame:
        """INTERNAL: Create/align attribute columns and dtypes to accept ``attrs``.

        Parameters
        --
        df : polars.DataFrame
            Existing attribute table.
        attrs : dict
            Incoming key/value pairs to upsert.

        Returns
        ---
        polars.DataFrame
            DataFrame with columns added/cast so inserts/updates won't hit ``Null`` dtypes.

        Notes
        -
        - New columns are created with the inferred dtype.
        - If a column is ``Null`` and the incoming value is not, it is cast to the inferred dtype.
        - If dtypes conflict (mixed over time), both sides upcast to ``Utf8`` to avoid schema errors.

        """
        _NUMERIC_DTYPES = {
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
            pl.Float32,
            pl.Float64,
        }
        schema = df.schema
        for col, val in attrs.items():
            target = self._pl_dtype_for_value(val)
            if col not in schema:
                df = df.with_columns(pl.lit(None).cast(target).alias(col))
            else:
                cur = schema[col]
                if cur == pl.Null and target != pl.Null:
                    df = df.with_columns(pl.col(col).cast(target))
                # if mixed types are expected over time:
                elif cur != target and target != pl.Null:
                    if cur in _NUMERIC_DTYPES and target in _NUMERIC_DTYPES:
                        if pl.Float64 in (cur, target):
                            supertype = pl.Float64
                        else:
                            supertype = pl.Int64
                        df = df.with_columns(pl.col(col).cast(supertype))
                    else:
                        df = df.with_columns(pl.col(col).cast(pl.Utf8))

        return df

    def _upsert_row(self, df: pl.DataFrame, idx, attrs: dict) -> pl.DataFrame:
        """INTERNAL: Upsert a row in a Polars DF [DataFrame] using explicit key columns.

        Keys

        - ``vertex_attributes``           - key: ``["vertex_id"]``
        - ``edge_attributes``             - key: ``["edge_id"]``
        - ``slice_attributes``            - key: ``["slice_id"]``
        - ``edge_slice_attributes``       - key: ``["slice_id", "edge_id"]``
        """
        if not isinstance(attrs, dict) or not attrs:
            return df

        cols = set(df.columns)

        # Determine key columns + values
        if {"slice_id", "edge_id"} <= cols:
            if not (isinstance(idx, tuple) and len(idx) == 2):
                raise ValueError("idx must be a (slice_id, edge_id) tuple")
            key_cols = ("slice_id", "edge_id")
            key_vals = {"slice_id": idx[0], "edge_id": idx[1]}
            cache_name = "_edge_slice_attr_keys"  # set of (slice_id, edge_id)
            df_id_name = "_edge_slice_attr_df_id"
        elif "vertex_id" in cols:
            key_cols = ("vertex_id",)
            key_vals = {"vertex_id": idx}
            cache_name = "_vertex_attr_ids"  # set of vertex_id
            df_id_name = "_vertex_attr_df_id"
        elif "edge_id" in cols:
            key_cols = ("edge_id",)
            key_vals = {"edge_id": idx}
            cache_name = "_edge_attr_ids"  # set of edge_id
            df_id_name = "_edge_attr_df_id"
        elif "slice_id" in cols:
            key_cols = ("slice_id",)
            key_vals = {"slice_id": idx}
            cache_name = "_slice_attr_ids"  # set of slice_id
            df_id_name = "_slice_attr_df_id"
        else:
            raise ValueError("Cannot infer key columns from DataFrame schema")

        # Ensure attribute columns exist / are cast appropriately
        df = self._ensure_attr_columns(df, attrs)

        # Build the match condition (used later for updates)
        cond = None
        for k in key_cols:
            v = key_vals[k]
            c = pl.col(k) == pl.lit(v)
            cond = c if cond is None else (cond & c)

        # existence check via small per-table caches (no DF scan)
        try:
            key_cache = getattr(self, cache_name, None)
            cached_df_id = getattr(self, df_id_name, None)
            if key_cache is None or cached_df_id != id(df):
                # Rebuild cache lazily for the current df object
                if "vertex_id" in cols and key_cols == ("vertex_id",):
                    series = df.get_column("vertex_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                elif (
                    "edge_id" in cols and "slice_id" in cols and key_cols == ("slice_id", "edge_id")
                ):
                    if df.height:
                        key_cache = set(
                            zip(
                                df.get_column("slice_id").to_list(),
                                df.get_column("edge_id").to_list(),
                            )
                        )
                    else:
                        key_cache = set()
                elif "edge_id" in cols and key_cols == ("edge_id",):
                    series = df.get_column("edge_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                elif "slice_id" in cols and key_cols == ("slice_id",):
                    series = df.get_column("slice_id") if df.height else pl.Series([])
                    key_cache = set(series.to_list()) if df.height else set()
                else:
                    key_cache = set()
                setattr(self, cache_name, key_cache)
                setattr(self, df_id_name, id(df))
            # Decide existence from cache
            cache_key = (
                key_vals[key_cols[0]]
                if len(key_cols) == 1
                else (key_vals["slice_id"], key_vals["edge_id"])
            )
            exists = cache_key in key_cache
        except Exception:
            # Fallback to original behavior if caching fails
            exists = df.filter(cond).height > 0
            key_cache = None

        if exists:
            # cast literals to column dtypes; keep exact semantics
            schema = df.schema
            upds = []
            for k, v in attrs.items():
                tgt_dtype = schema[k]
                upds.append(
                    pl.when(cond).then(pl.lit(v).cast(tgt_dtype)).otherwise(pl.col(k)).alias(k)
                )
            new_df = df.with_columns(upds)

            # Keep cache pointers in sync with the new df object
            try:
                setattr(self, df_id_name, id(new_df))
                # cache contents unchanged for updates
            except Exception:
                pass

            return new_df

        # build a single row aligned to df schema
        schema = df.schema

        # Start with None for all columns, fill keys and attrs
        new_row = dict.fromkeys(df.columns)
        new_row.update(key_vals)
        new_row.update(attrs)

        to_append = pl.DataFrame([new_row])

        # 1) Ensure to_append has all df columns
        for c in df.columns:
            if c not in to_append.columns:
                to_append = to_append.with_columns(pl.lit(None).cast(schema[c]).alias(c))

        # 2) Resolve dtype mismatches:
        #    - df Null + to_append non-Null -> cast df to right
        #    - to_append Null + df non-Null -> cast to_append to left
        #    - left != right -> upcast both to Utf8
        left_schema = schema
        right_schema = to_append.schema
        df_casts = []
        app_casts = []
        for c in df.columns:
            left = left_schema[c]
            right = right_schema[c]
            if left == pl.Null and right != pl.Null:
                df_casts.append(pl.col(c).cast(right))
            elif right == pl.Null and left != pl.Null:
                app_casts.append(pl.col(c).cast(left).alias(c))
            elif left != right:
                if left.is_numeric() and right.is_numeric():
                    supertype = _get_numeric_supertype(left, right)
                    df_casts.append(pl.col(c).cast(supertype))
                    app_casts.append(pl.col(c).cast(supertype).alias(c))
                else:
                    # fallback: Utf8 for incompatible non-numeric types
                    df_casts.append(pl.col(c).cast(pl.Utf8))
                    app_casts.append(pl.col(c).cast(pl.Utf8).alias(c))

        if df_casts:
            df = df.with_columns(df_casts)
            left_schema = df.schema  # refresh for correctness
        if app_casts:
            to_append = to_append.with_columns(app_casts)

        new_df = df.vstack(to_append)

        # Update caches after insertion
        try:
            if key_cache is not None:
                if len(key_cols) == 1:
                    key_cache.add(cache_key)
                else:
                    key_cache.add(cache_key)
            setattr(self, df_id_name, id(new_df))
        except Exception:
            pass

        return new_df

    def _variables_watched_by_vertices(self):
        # set of vertex-attribute names used by vertex-scope policies
        return {
            p["var"]
            for p in self.edge_direction_policy.values()
            if p.get("scope", "edge") == "vertex"
        }

    def _incident_flexible_edges(self, v):
        # naive scan; optimize later with an index if needed
        out = []
        for eid, (s, t, _kind) in self.edge_definitions.items():
            if eid in self.edge_direction_policy and (s == v or t == v):
                out.append(eid)
        return out

    def _apply_flexible_direction(self, edge_id):
        pol = self.edge_direction_policy.get(edge_id)
        if not pol:
            return

        src, tgt, _ = self.edge_definitions[edge_id]
        col = self.edge_to_idx[edge_id]
        w = float(self.edge_weights.get(edge_id, 1.0))

        var = pol["var"]
        T = float(pol["threshold"])
        scope = pol.get("scope", "edge")  # 'edge'|'vertex'
        above = pol.get("above", "s->t")  # 's->t'|'t->s'
        tie = pol.get("tie", "keep")  # default behavior

        # decide condition and detect tie
        tie_case = False
        if scope == "edge":
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
        si = self.entity_to_idx[src]
        ti = self.entity_to_idx[tgt]

        if tie_case:
            if tie == "keep":
                # do nothing - previous signs remain (default)
                return
            if tie == "undirected":
                # force (+w,+w) while equality holds
                M[(si, col)] = +w
                if src != tgt:
                    M[(ti, col)] = +w
                return
            # force a direction at equality
            cond = True if tie == "s->t" else False

        # rewrite as directed per 'above'
        M[(si, col)] = 0
        M[(ti, col)] = 0
        src_to_tgt = cond if above == "s->t" else (not cond)
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
        --
        edge : int | str
            Edge index (int) or edge id (str).

        Returns
        ---
        dict
            Attribute dictionary for that edge. {} if not found.

        """
        # normalize to edge id
        if isinstance(edge, int):
            eid = self.idx_to_edge[edge]
        else:
            eid = edge

        df = self.edge_attributes
        # Polars-safe: iterate the (at most one) row as a dict
        try:
            for row in df.filter(pl.col("edge_id") == eid).iter_rows(named=True):
                return dict(row)
            return {}
        except Exception:
            # Fallback if df is pandas or dict-like
            try:
                row = df[df["edge_id"] == eid].to_dict(orient="records")
                return row[0] if row else {}
            except Exception:
                return {}

    def get_vertex_attrs(self, vertex) -> dict:
        """Return the full attribute dict for a single vertex.

        Parameters
        --
        vertex : str
            Vertex id.

        Returns
        ---
        dict
            Attribute dictionary for that vertex. {} if not found.

        """
        df = self.vertex_attributes
        try:
            for row in df.filter(pl.col("vertex_id") == vertex).iter_rows(named=True):
                return dict(row)
            return {}
        except Exception:
            try:
                row = df[df["vertex_id"] == vertex].to_dict(orient="records")
                return row[0] if row else {}
            except Exception:
                return {}

    ## Bulk attributes

    def get_attr_edges(self, indexes=None) -> dict:
        """Retrieve edge attributes as a dictionary.

        Parameters
        --
        indexes : Iterable[int] | None, optional
            A list or iterable of edge indices to retrieve attributes for.
            - If `None` (default), attributes for **all** edges are returned.
            - If provided, only those edges will be included in the output.

        Returns
        ---
        dict[str, dict]
            A dictionary mapping `edge_id` - `attribute_dict`, where:
            - `edge_id` is the unique string identifier of the edge.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -
        - This function reads directly from `self.edge_attributes`, which should be
        a Polars DataFrame where each row corresponds to an edge.
        - Useful for bulk inspection, serialization, or analytics without looping manually.

        """
        df = self.edge_attributes
        if indexes is not None:
            df = df.filter(pl.col("edge_id").is_in([self.idx_to_edge[i] for i in indexes]))
        return {row["edge_id"]: row.as_dict() for row in df.iter_rows(named=True)}

    def get_attr_vertices(self, vertices=None) -> dict:
        """Retrieve vertex (vertex) attributes as a dictionary.

        Parameters
        --
        vertices : Iterable[str] | None, optional
            A list or iterable of vertex IDs to retrieve attributes for.
            - If `None` (default), attributes for **all** verices are returned.
            - If provided, only those verices will be included in the output.

        Returns
        ---
        dict[str, dict]
            A dictionary mapping `vertex_id` - `attribute_dict`, where:
            - `vertex_id` is the unique string identifier of the vertex.
            - `attribute_dict` is a dictionary of attribute names and values.

        Notes
        -
        - This reads from `self.vertex_attributes`, which stores per-vertex metadata.
        - Use this for bulk data extraction instead of repeated single-vertex calls.

        """
        df = self.vertex_attributes
        if vertices is not None:
            df = df.filter(pl.col("vertex_id").is_in(vertices))
        return {row["vertex_id"]: row.as_dict() for row in df.iter_rows(named=True)}

    def get_attr_from_edges(self, key: str, default=None) -> dict:
        """Extract a specific attribute column for all edges.

        Parameters
        --
        key : str
            Attribute column name to extract from `self.edge_attributes`.
        default : Any, optional
            Default value to use if the column does not exist or if an edge
            does not have a value. Defaults to `None`.

        Returns
        ---
        dict[str, Any]
            A dictionary mapping `edge_id` - attribute value.

        Notes
        -
        - If the requested column is missing, all edges return `default`.
        - This is useful for quick property lookups (e.g., weight, label, type).

        """
        df = self.edge_attributes
        if key not in df.columns:
            return {row["edge_id"]: default for row in df.iter_rows(named=True)}
        return {
            row["edge_id"]: row[key] if row[key] is not None else default
            for row in df.iter_rows(named=True)
        }

    def get_edges_by_attr(self, key: str, value) -> list:
        """Retrieve all edges where a given attribute equals a specific value.

        Parameters
        --
        key : str
            Attribute column name to filter on.
        value : Any
            Value to match.

        Returns
        ---
        list[str]
            A list of edge IDs where the attribute `key` equals `value`.

        Notes
        -
        - If the attribute column does not exist, an empty list is returned.
        - Comparison is exact; consider normalizing types before calling.

        """
        df = self.edge_attributes
        if key not in df.columns:
            return []
        return [row["edge_id"] for row in df.iter_rows(named=True) if row[key] == value]

    def get_graph_attributes(self) -> dict:
        """Return a shallow copy of the graph-level attributes dictionary.

        Returns
        ---
        dict
            A dictionary of global metadata describing the graph as a whole.
            Typical keys might include:
            - `"name"` : Graph name or label.
            - `"directed"` : Boolean indicating directedness.
            - `"slices"` : List of slices present in the graph.
            - `"created_at"` : Timestamp of graph creation.

        Notes
        -
        - Returns a **shallow copy** to prevent external mutation of internal state.
        - Graph-level attributes are meant to store metadata not tied to individual
        verices or edges (e.g., versioning info, provenance, global labels).

        """
        return dict(self.graph_attributes)

    def set_edge_slice_attrs_bulk(self, slice_id, items):
        """items: iterable of (edge_id, attrs_dict) or dict{edge_id: attrs_dict}
        Upserts rows in edge_slice_attributes for one slice in bulk.
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
            r = {"slice_id": slice_id, "edge_id": eid}
            r.update(attrs)
            if "weight" in r:
                try:
                    r["weight"] = float(r["weight"])
                except Exception:
                    pass
            rows.append(r)
        if not rows:
            return

        # start from current DF
        df = self.edge_slice_attributes
        add_df = pl.DataFrame(rows)

        # ensure required key cols exist/correct dtype on existing df
        if not isinstance(df, pl.DataFrame) or df.is_empty():
            # create from scratch with canonical dtypes
            self.edge_slice_attributes = add_df
            # legacy mirror
            if "weight" in add_df.columns:
                self.slice_edge_weights.setdefault(slice_id, {})
                for r in add_df.iter_rows(named=True):
                    w = r.get("weight")
                    if w is not None:
                        self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)
            return

        # schema alignment using _ensure_attr_columns + Utf8 upcast rule
        need_cols = {c: None for c in add_df.columns if c not in df.columns}
        if need_cols:
            df = self._ensure_attr_columns(df, need_cols)  # adds missing columns to df
        # add missing columns to add_df
        for c in df.columns:
            if c not in add_df.columns:
                add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))
        # reconcile dtype mismatches (Null/Null, mixed -> Utf8), same policy as _upsert_row
        for c in df.columns:
            lc, rc = df.schema[c], add_df.schema[c]
            if lc == pl.Null and rc != pl.Null:
                df = df.with_columns(pl.col(c).cast(rc))
            elif rc == pl.Null and lc != pl.Null:
                add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
            elif lc != rc:
                df = df.with_columns(pl.col(c).cast(pl.Utf8))
                add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

        # drop existing keys for (slice_id, edge_id) we are about to write; then vstack new rows
        mask_keep = ~(
            (pl.col("slice_id") == slice_id) & pl.col("edge_id").is_in(add_df.get_column("edge_id"))
        )
        df = df.filter(mask_keep)
        df = df.vstack(add_df)
        self.edge_slice_attributes = df

        # legacy mirror
        if "weight" in add_df.columns:
            self.slice_edge_weights.setdefault(slice_id, {})
            for r in add_df.iter_rows(named=True):
                w = r.get("weight")
                if w is not None:
                    self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)
