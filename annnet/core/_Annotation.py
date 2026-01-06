import math
from typing import Any

import narwhals as nw

try:
    import polars as pl  # optional

    is_polars = True
except Exception:  # ModuleNotFoundError, etc.
    pl = None
    is_polars = False
from ._helpers import _get_numeric_supertype

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

_NUMERIC_PL_DTYPES = set()
if pl is not None:
    _NUMERIC_PL_DTYPES = {
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

_NUMERIC_DTYPES = _NUMERIC_PL_DTYPES if is_polars else _NUMERIC_NW_DTYPES


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

    def set_vertex_attrs_bulk(self, updates):
        """
        updates: dict[vertex_id, dict[attr, value]]
                 or iterable[(vertex_id, dict)]
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)

        for vid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(
                    f"vertex bulk attrs must be dict, got {type(attrs)} for {vid}"
                )

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
                            f"Composite key collision on {self._vertex_key_fields}: {new_key} owned by {owner}"
                        )
        
        self.vertex_attributes = self._upsert_rows_bulk(self.vertex_attributes, clean_updates)
        
        watched = self._variables_watched_by_vertices()
        if watched:
            affected_vertices = {vid for vid, attrs in clean_updates.items() if any(k in watched for k in attrs)}
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
        --
        vertex_id : str
        key : str
        default : Any, optional

        Returns
        ---
        Any

        """
        df = self.vertex_attributes
        if df is None or not hasattr(df, "columns") or key not in df.columns:
            return default

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter(pl.col("vertex_id") == vertex_id)
            if rows.height == 0:
                return default
            val = rows.select(pl.col(key)).to_series()[0]
            return default if val is None else val

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(nw.col("vertex_id") == vertex_id)
        )

        # empty?
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return default

        # pull first value of column `key`
        try:
            col = rows[key]
            val = (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            # fallback via first row dict
            r0 = (
                rows.to_dicts()[0]
                if hasattr(rows, "to_dicts")
                else rows.to_dict(orient="records")[0]
            )
            val = r0.get(key, None)

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
        df = self.vertex_attributes
        if df is None or not hasattr(df, "columns"):
            return None
        if "vertex_id" not in df.columns or attribute not in df.columns:
            return None

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return None
            rows = df.filter(pl.col("vertex_id") == vertex_id)
            if rows.height == 0:
                return None
            s = rows.get_column(attribute)
            return s.item(0) if s.len() else None

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(nw.col("vertex_id") == vertex_id)
        )
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return None

        # first scalar value of column
        try:
            col = rows[attribute]
            return (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            r0 = (
                rows.to_dict(orient="records")[0]
                if hasattr(rows, "to_dict")
                else rows.to_dicts()[0]
            )
            return r0.get(attribute, None)

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

    def set_edge_attrs_bulk(self, updates):
        """
        updates: dict[edge_id, dict[attr, value]]
        """
        if not updates:
            return
        if not isinstance(updates, dict):
            updates = dict(updates)
        for eid, attrs in updates.items():
            if not isinstance(attrs, dict):
                raise TypeError(
                    f"edge bulk attrs must be dict, got {type(attrs)} for {eid}"
                )
        
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
            if pol and pol.get("scope") == "edge" and pol["var"] in attrs:
                affected_edges.add(eid)
        
        for eid in affected_edges:
            self._apply_flexible_direction(eid)

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
        if df is None or not hasattr(df, "columns") or key not in df.columns:
            return default

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter(pl.col("edge_id") == edge_id)
            if rows.height == 0:
                return default
            val = rows.select(pl.col(key)).to_series()[0]
            return default if val is None else val

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(nw.col("edge_id") == edge_id)
        )
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return default

        try:
            col = rows[key]
            val = (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            r0 = (
                rows.to_dict(orient="records")[0]
                if hasattr(rows, "to_dict")
                else rows.to_dicts()[0]
            )
            val = r0.get(key, None)

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
        if df is None or not hasattr(df, "columns"):
            return None
        if "edge_id" not in df.columns or attribute not in df.columns:
            return None

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return None
            rows = df.filter(pl.col("edge_id") == edge_id)
            if rows.height == 0:
                return None
            s = rows.get_column(attribute)
            return s.item(0) if s.len() else None

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(nw.col("edge_id") == edge_id)
        )
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return None

        try:
            col = rows[attribute]
            return (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            r0 = (
                rows.to_dict(orient="records")[0]
                if hasattr(rows, "to_dict")
                else rows.to_dicts()[0]
            )
            return r0.get(attribute, None)

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
        if df is None or not hasattr(df, "columns") or key not in df.columns:
            return default

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter(pl.col("slice_id") == slice_id)
            if rows.height == 0:
                return default
            val = rows.select(pl.col(key)).to_series()[0]
            return default if val is None else val

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(nw.col("slice_id") == slice_id)
        )
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return default

        try:
            col = rows[key]
            val = (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            r0 = (
                rows.to_dict(orient="records")[0]
                if hasattr(rows, "to_dict")
                else rows.to_dicts()[0]
            )
            val = r0.get(key, None)

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
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame) and df.height > 0:
            to_cast = []
            if "slice_id" in df.columns and df.schema["slice_id"] != pl.Utf8:
                to_cast.append(pl.col("slice_id").cast(pl.Utf8))
            if "edge_id" in df.columns and df.schema["edge_id"] != pl.Utf8:
                to_cast.append(pl.col("edge_id").cast(pl.Utf8))
            if to_cast:
                df = df.with_columns(*to_cast)
                self.edge_slice_attributes = df

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
        if df is None or not hasattr(df, "columns") or key not in df.columns:
            return default

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter((pl.col("slice_id") == slice_id) & (pl.col("edge_id") == edge_id))
            if rows.height == 0:
                return default
            val = rows.select(pl.col(key)).to_series()[0]
            return default if val is None else val

        import narwhals as nw

        rows = nw.to_native(
            nw.from_native(df, pass_through=True).filter(
                (nw.col("slice_id") == slice_id) & (nw.col("edge_id") == edge_id)
            )
        )
        if (hasattr(rows, "__len__") and len(rows) == 0) or (getattr(rows, "height", None) == 0):
            return default

        try:
            col = rows[key]
            val = (
                col.iloc[0]
                if hasattr(col, "iloc")
                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
            )
        except Exception:
            r0 = (
                rows.to_dict(orient="records")[0]
                if hasattr(rows, "to_dict")
                else rows.to_dicts()[0]
            )
            val = r0.get(key, None)

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
                df is not None
                and hasattr(df, "columns")
                and {"slice_id", "edge_id", "weight"} <= set(df.columns)
            ):
                try:
                    import polars as pl
                except Exception:
                    pl = None

                if pl is not None and isinstance(df, pl.DataFrame) and df.height > 0:
                    rows = df.filter(
                        (pl.col("slice_id") == slice) & (pl.col("edge_id") == edge_id)
                    ).select("weight")
                    if rows.height > 0:
                        w = rows.to_series()[0]
                        if w is not None and not (isinstance(w, float) and math.isnan(w)):
                            return float(w)
                else:
                    import narwhals as nw

                    rows = nw.to_native(
                        nw.from_native(df, pass_through=True)
                        .filter((nw.col("slice_id") == slice) & (nw.col("edge_id") == edge_id))
                        .select("weight")
                    )
                    if (hasattr(rows, "__len__") and len(rows) > 0) or (
                        getattr(rows, "height", 0) > 0
                    ):
                        try:
                            col = rows["weight"]
                            w = (
                                col.iloc[0]
                                if hasattr(col, "iloc")
                                else (col.to_list()[0] if hasattr(col, "to_list") else list(col)[0])
                            )
                        except Exception:
                            r0 = (
                                rows.to_dict(orient="records")[0]
                                if hasattr(rows, "to_dict")
                                else rows.to_dicts()[0]
                            )
                            w = r0.get("weight", None)
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

        if na is not None and hasattr(na, "columns") and "vertex_id" in na.columns:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(na, pl.DataFrame) and na.height > 0:
                vertex_attr_ids = set(na.select("vertex_id").to_series().to_list())
            else:
                import narwhals as nw

                tmp = nw.to_native(nw.from_native(na, pass_through=True).select("vertex_id"))
                try:
                    s = tmp["vertex_id"]
                    vertex_attr_ids = set(s.to_list() if hasattr(s, "to_list") else list(s))
                except Exception:
                    vertex_attr_ids = set()
        else:
            vertex_attr_ids = set()

        if ea is not None and hasattr(ea, "columns") and "edge_id" in ea.columns:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(ea, pl.DataFrame) and ea.height > 0:
                edge_attr_ids = set(ea.select("edge_id").to_series().to_list())
            else:
                import narwhals as nw

                tmp = nw.to_native(nw.from_native(ea, pass_through=True).select("edge_id"))
                try:
                    s = tmp["edge_id"]
                    edge_attr_ids = set(s.to_list() if hasattr(s, "to_list") else list(s))
                except Exception:
                    edge_attr_ids = set()
        else:
            edge_attr_ids = set()

        extra_vertex_rows = [i for i in vertex_attr_ids if i not in vertex_ids]
        extra_edge_rows = [i for i in edge_attr_ids if i not in edge_ids]
        missing_vertex_rows = [i for i in vertex_ids if i not in vertex_attr_ids]
        missing_edge_rows = [i for i in edge_ids if i not in edge_attr_ids]

        bad_edge_slice = []
        if (
            ela is not None
            and hasattr(ela, "columns")
            and {"slice_id", "edge_id"} <= set(ela.columns)
        ):
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(ela, pl.DataFrame) and ela.height > 0:
                for lid, eid in ela.select(["slice_id", "edge_id"]).iter_rows():
                    if lid not in self._slices or eid not in edge_ids:
                        bad_edge_slice.append((lid, eid))
            else:
                import narwhals as nw

                tmp = nw.to_native(
                    nw.from_native(ela, pass_through=True).select(["slice_id", "edge_id"])
                )
                rows = tmp.to_dicts() if hasattr(tmp, "to_dicts") else tmp.to_dict(orient="records")
                for r in rows:
                    lid = r.get("slice_id")
                    eid = r.get("edge_id")
                    if lid not in self._slices or eid not in edge_ids:
                        bad_edge_slice.append((lid, eid))

        return {
            "extra_vertex_rows": extra_vertex_rows,
            "extra_edge_rows": extra_edge_rows,
            "missing_vertex_rows": missing_vertex_rows,
            "missing_edge_rows": missing_edge_rows,
            "invalid_edge_slice_rows": bad_edge_slice,
        }

    def _dtype_for_value(self, v, *, prefer="polars"):
        """INTERNAL: Infer an appropriate dtype class for value `v`.

        - If Polars is available and prefer='polars', returns Polars dtype objects.
        - Otherwise returns Narwhals dtype classes.
        """
        import enum

        try:
            import polars as pl
        except Exception:
            pl = None
        import narwhals as nw

        if prefer == "polars" and pl is not None:
            if v is None:
                return pl.Null
            if isinstance(v, bool):
                return pl.Boolean
            if isinstance(v, int) and not isinstance(v, bool):
                return pl.Int64
            if isinstance(v, float):
                return pl.Float64
            if isinstance(v, enum.Enum):
                return pl.Object
            if isinstance(v, (bytes, bytearray)):
                return pl.Binary
            if isinstance(v, (list, tuple)):
                inner = self._dtype_for_value(v[0], prefer="polars") if len(v) else pl.Utf8
                return pl.List(pl.Utf8 if inner == pl.Null else inner)
            if isinstance(v, dict):
                return pl.Object
            return pl.Utf8

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
            inner = self._dtype_for_value(v[0], prefer="narwhals") if len(v) else nw.String
            return nw.List(nw.String if inner == nw.Unknown else inner)
        if isinstance(v, dict):
            return nw.Object
        return nw.String

    def _is_null_dtype(self, dtype) -> bool:
        """Check if a dtype represents a null/unknown type."""
        import narwhals as nw

        try:
            import polars as pl
        except ImportError:
            pl = None

        # Catch Narwhals Unknown and backend-specific Null classes
        if dtype == nw.Unknown or (pl and dtype == pl.Null):
            return True
        # Handle instance vs class comparison
        dt_type = type(dtype) if not isinstance(dtype, type) else dtype
        return dt_type == nw.Unknown or (pl and dt_type == pl.Null)

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
        is_polars = impl.is_polars()

        for col, val in attrs.items():
            # Use Narwhals dtypes for logic to avoid backend-mismatch pitfalls
            target = self._dtype_for_value(val, prefer="narwhals")

            if col not in schema:
                # Add new column with appropriate null-casting
                if is_polars:
                    import polars as pl

                    pdf = nw.to_native(nw_df)
                    pl_target = self._dtype_for_value(val, prefer="polars")
                    pdf = pdf.with_columns(pl.lit(None).cast(pl_target).alias(col))
                    nw_df = nw.from_native(pdf, eager_only=True)
                else:
                    try:
                        nw_df = nw_df.with_columns(nw.lit(None).cast(target).alias(col))
                    except Exception:
                        nw_df = nw_df.with_columns(nw.lit(None).alias(col))
            else:
                # Upgrade logic: ONLY cast if the existing column is a Null/Unknown type
                cur = schema[col]
                if self._is_null_dtype(cur) and not self._is_null_dtype(target):
                    try:
                        nw_df = nw_df.with_columns(nw.col(col).cast(target))
                    except Exception:
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
        return any(kw in s for kw in ("binary", "blob", "byte"))

    def _safe_nw_cast(self, column_expr, target_dtype):
        """INTERNAL: Attempt cast; fallback to String on engine rejection."""
        import narwhals as nw

        try:
            # Ensure target_dtype is a Narwhals DType, not a native backend class
            if not isinstance(target_dtype, (nw.dtypes.DType, type(nw.Int64))):
                return column_expr.cast(nw.String)
            return column_expr.cast(target_dtype)
        except Exception:
            return column_expr.cast(nw.String)

    def _upsert_row(self, df: "object", idx: Any, attrs: dict) -> "object":
        if not isinstance(attrs, dict) or not attrs:
            return df

        nw_df = nw.from_native(df, eager_only=True)
        cols = set(nw_df.columns)

        # 1. Key Resolution (Unchanged)
        if {"slice_id", "edge_id"} <= cols:
            key_cols, key_vals = ("slice_id", "edge_id"), {"slice_id": idx[0], "edge_id": idx[1]}
            cache_name, df_id_name = "_edge_slice_attr_keys", "_edge_slice_attr_df_id"
        elif "vertex_id" in cols:
            key_cols, key_vals = ("vertex_id",), {"vertex_id": idx}
            cache_name, df_id_name = "_vertex_attr_ids", "_vertex_attr_df_id"
        elif "edge_id" in cols:
            key_cols, key_vals = ("edge_id",), {"edge_id": idx}
            cache_name, df_id_name = "_edge_attr_ids", "_edge_attr_df_id"
        elif "slice_id" in cols:
            key_cols, key_vals = ("slice_id",), {"slice_id": idx}
            cache_name, df_id_name = "_slice_attr_ids", "_slice_attr_df_id"
        else:
            raise ValueError("Cannot infer key columns from DataFrame schema")

        nw_df = self._ensure_attr_columns(nw_df, attrs)
        cond = None
        for k in key_cols:
            c = nw.col(k) == nw.lit(key_vals[k])
            cond = c if cond is None else (cond & c)

        # 2. Existence Check (Unchanged)
        try:
            key_cache = getattr(self, cache_name, None)
            current_df_id = id(nw.to_native(nw_df))
            if key_cache is None or getattr(self, df_id_name, None) != current_df_id:
                if key_cols == ("slice_id", "edge_id"):
                    key_cache = set(
                        zip(
                            nw_df.get_column("slice_id").to_list(),
                            nw_df.get_column("edge_id").to_list(),
                        )
                    )
                else:
                    key_cache = set(nw_df.get_column(key_cols[0]).to_list())
                setattr(self, cache_name, key_cache)
                setattr(self, df_id_name, current_df_id)
            cache_key = idx
            exists = cache_key in key_cache
        except Exception:
            exists = nw_df.filter(cond).shape[0] > 0
            key_cache = None

        if exists:
            schema = nw_df.collect_schema()
            upds = []
            for k, v in attrs.items():
                v2 = self._sanitize_value_for_nw(v)
                tgt_dt = schema[k]
                if self._is_null_dtype(tgt_dt):
                    inf = self._dtype_for_value(v2, prefer="narwhals")
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
            return any(x in str(dt).lower() for x in ("list", "array"))

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
                    return any(x in s for x in ("float", "int", "decimal", "uint"))

                if _is_num(l) and _is_num(r):
                    try:
                        sup = _get_numeric_supertype(l, r)
                        if sup:
                            df_up.append(self._safe_nw_cast(nw.col(c), sup).alias(c))
                            app_up.append(self._safe_nw_cast(nw.col(c), sup).alias(c))
                            continue
                    except Exception:
                        pass  # Fall through to string if supertype fails

                # 3. Final Fallback (Incompatible types only)
                df_up.append(nw.col(c).cast(nw.String).alias(c))
                app_up.append(nw.col(c).cast(nw.String).alias(c))

        if df_up:
            nw_df = nw_df.with_columns(df_up)
        if app_up:
            to_append = to_append.with_columns(app_up)

        final_df = nw.concat([nw_df, to_append], how="vertical")
        if key_cache is not None:
            key_cache.add(cache_key)

        setattr(self, df_id_name, id(nw.to_native(final_df)))
        return nw.to_native(final_df)

    def _upsert_rows_bulk(self, df: "object", updates: dict) -> "object":
        if not updates:
            return df
        
        nw_df = nw.from_native(df, eager_only=True)
        
        # Build complete update DataFrame
        update_records = []
        for idx, attrs in updates.items():
            if isinstance(idx, tuple):
                record = {"slice_id": idx[0], "edge_id": idx[1], **attrs}
            else:
                cols = set(nw_df.columns)
                if "vertex_id" in cols:
                    record = {"vertex_id": idx, **attrs}
                elif "edge_id" in cols:
                    record = {"edge_id": idx, **attrs}
                else:
                    record = {"slice_id": idx, **attrs}
            update_records.append(record)
        
        update_df = nw.DataFrame.from_dicts(update_records, backend=nw.get_native_namespace(nw_df))
        
        # Determine join keys
        cols = set(nw_df.columns)
        if {"slice_id", "edge_id"} <= cols:
            join_keys = ["slice_id", "edge_id"]
        elif "vertex_id" in cols:
            join_keys = ["vertex_id"]
        elif "edge_id" in cols:
            join_keys = ["edge_id"]
        else:
            join_keys = ["slice_id"]
        
        # Anti-join to get rows NOT being updated
        unchanged = nw_df.join(update_df.select(join_keys), on=join_keys, how="anti")
        
        # Combine: unchanged rows + all update rows
        result = nw.concat([unchanged, update_df], how="diagonal")
        
        return nw.to_native(result)

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
            import polars as pl
        except Exception:
            pl = None

        # Polars fast-path
        if pl is not None and isinstance(df, pl.DataFrame):
            for row in df.filter(pl.col("edge_id") == eid).iter_rows(named=True):
                return dict(row)
            return {}

        # Non-Polars path (Narwhals -> native rows)
        try:
            import narwhals as nw

            native = nw.to_native(
                nw.from_native(df, pass_through=True).filter(nw.col("edge_id") == eid)
            )
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            return rows[0] if rows else {}
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
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            for row in df.filter(pl.col("vertex_id") == vertex).iter_rows(named=True):
                return dict(row)
            return {}

        try:
            import narwhals as nw

            native = nw.to_native(
                nw.from_native(df, pass_through=True).filter(nw.col("vertex_id") == vertex)
            )
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            return rows[0] if rows else {}
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

        try:
            import polars as pl
        except Exception:
            pl = None

        if indexes is not None:
            wanted = [self.idx_to_edge[i] for i in indexes]
            if pl is not None and isinstance(df, pl.DataFrame):
                df = df.filter(pl.col("edge_id").is_in(wanted))
            else:
                import narwhals as nw

                df = nw.to_native(
                    nw.from_native(df, pass_through=True).filter(nw.col("edge_id").is_in(wanted))
                )

        if pl is not None and isinstance(df, pl.DataFrame):
            return {row["edge_id"]: dict(row) for row in df.iter_rows(named=True)}

        # non-Polars
        try:
            import narwhals as nw

            native = nw.to_native(nw.from_native(df, pass_through=True))
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            return {r.get("edge_id"): dict(r) for r in rows if r.get("edge_id") is not None}
        except Exception:
            return {}

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

        try:
            import polars as pl
        except Exception:
            pl = None

        if vertices is not None:
            if pl is not None and isinstance(df, pl.DataFrame):
                df = df.filter(pl.col("vertex_id").is_in(list(vertices)))
            else:
                import narwhals as nw

                df = nw.to_native(
                    nw.from_native(df, pass_through=True).filter(
                        nw.col("vertex_id").is_in(list(vertices))
                    )
                )

        if pl is not None and isinstance(df, pl.DataFrame):
            return {row["vertex_id"]: dict(row) for row in df.iter_rows(named=True)}

        try:
            import narwhals as nw

            native = nw.to_native(nw.from_native(df, pass_through=True))
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            return {r.get("vertex_id"): dict(r) for r in rows if r.get("vertex_id") is not None}
        except Exception:
            return {}

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
        if df is None or not hasattr(df, "columns"):
            return {}

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if key not in df.columns:
                return {row["edge_id"]: default for row in df.iter_rows(named=True)}
            return {
                row["edge_id"]: (row[key] if row[key] is not None else default)
                for row in df.iter_rows(named=True)
            }

        # non-Polars
        import narwhals as nw

        native = nw.to_native(nw.from_native(df, pass_through=True))
        rows = (
            native.to_dicts() if hasattr(native, "to_dicts") else native.to_dict(orient="records")
        )
        if key not in getattr(native, "columns", []):
            return {r.get("edge_id"): default for r in rows if r.get("edge_id") is not None}
        return {
            r.get("edge_id"): (r.get(key) if r.get(key) is not None else default)
            for r in rows
            if r.get("edge_id") is not None
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
        if df is None or not hasattr(df, "columns") or key not in df.columns:
            return []

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            return [row["edge_id"] for row in df.iter_rows(named=True) if row.get(key) == value]

        import narwhals as nw

        native = nw.to_native(nw.from_native(df, pass_through=True))
        rows = (
            native.to_dicts() if hasattr(native, "to_dicts") else native.to_dict(orient="records")
        )
        return [
            r.get("edge_id") for r in rows if r.get("edge_id") is not None and r.get(key) == value
        ]

    def get_graph_attributes(self) -> dict:
        """Return a shallow copy of the graph-level attributes dictionary.

        Returns
        ---
        dict
            A dictionary of global metadata describing the graph as a whole.
            Typical keys might include:
            - `"name"` : AnnNet name or label.
            - `"directed"` : Boolean indicating directedness.
            - `"slices"` : List of slices present in the graph.
            - `"created_at"` : Timestamp of graph creation.

        Notes
        -
        - Returns a **shallow copy** to prevent external mutation of internal state.
        - AnnNet-level attributes are meant to store metadata not tied to individual
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
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None:
            add_df = pl.DataFrame(rows)
        else:
            import pandas as pd

            add_df = pd.DataFrame.from_records(rows)

        # ensure required key cols exist/correct dtype on existing df
        try:
            import polars as pl
        except Exception:
            pl = None

        is_polars = pl is not None and isinstance(df, pl.DataFrame)

        is_empty = False
        try:
            is_empty = df.is_empty() if is_polars else (len(df) == 0)
        except Exception:
            pass

        if (not is_polars) or is_empty:
            self.edge_slice_attributes = add_df
            # legacy mirror
            if "weight" in getattr(add_df, "columns", []):
                self.slice_edge_weights.setdefault(slice_id, {})
                if pl is not None and isinstance(add_df, pl.DataFrame):
                    it = add_df.iter_rows(named=True)
                else:
                    it = add_df.to_dict(orient="records") if hasattr(add_df, "to_dict") else []
                for r in it:
                    w = r.get("weight")
                    if w is not None:
                        self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)
            return

        # schema alignment using _ensure_attr_columns + Utf8 upcast rule
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame) and isinstance(add_df, pl.DataFrame):
            # Polars fast-path
            need_cols = {c: None for c in add_df.columns if c not in df.columns}
            if need_cols:
                df = self._ensure_attr_columns(df, need_cols)
            for c in df.columns:
                if c not in add_df.columns:
                    add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))
            for c in df.columns:
                lc, rc = df.schema[c], add_df.schema[c]
                if lc == pl.Null and rc != pl.Null:
                    df = df.with_columns(pl.col(c).cast(rc))
                elif rc == pl.Null and lc != pl.Null:
                    add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
                elif lc != rc:
                    df = df.with_columns(pl.col(c).cast(pl.Utf8))
                    add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

            mask_keep = ~(
                (pl.col("slice_id") == slice_id)
                & pl.col("edge_id").is_in(add_df.get_column("edge_id"))
            )
            df = df.filter(mask_keep)
            df = df.vstack(add_df)
            self.edge_slice_attributes = df

        else:
            # pandas fallback
            import pandas as pd

            pdf = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            padd = add_df if isinstance(add_df, pd.DataFrame) else pd.DataFrame(add_df)

            # ensure all columns exist on both sides
            for c in padd.columns:
                if c not in pdf.columns:
                    pdf[c] = pd.NA
            for c in pdf.columns:
                if c not in padd.columns:
                    padd[c] = pd.NA

            # drop existing rows for the keys we’re writing: (slice_id, edge_id)
            wanted_eids = set(padd["edge_id"].tolist()) if "edge_id" in padd.columns else set()
            if "slice_id" in pdf.columns and "edge_id" in pdf.columns and wanted_eids:
                keep_mask = ~(
                    (pdf["slice_id"] == slice_id) & (pdf["edge_id"].isin(list(wanted_eids)))
                )
                pdf = pdf.loc[keep_mask]

            pdf = pd.concat([pdf, padd], ignore_index=True)
            self.edge_slice_attributes = pdf

        # legacy mirror
        if "weight" in getattr(add_df, "columns", []):
            self.slice_edge_weights.setdefault(slice_id, {})
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(add_df, pl.DataFrame):
                it = add_df.iter_rows(named=True)
            else:
                it = add_df.to_dict(orient="records") if hasattr(add_df, "to_dict") else []
            for r in it:
                w = r.get("weight")
                if w is not None:
                    self.slice_edge_weights[slice_id][r["edge_id"]] = float(w)
