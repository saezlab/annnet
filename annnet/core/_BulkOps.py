import json

import narwhals as nw

try:
    import polars as pl
except Exception:
    pl = None
import scipy.sparse as sp

from ._helpers import (
    EdgeRecord,
    EdgeType,
    EntityRecord,
    _get_numeric_supertype,
)


def _sanitize(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


def _to_polars_if_possible(df):
    import narwhals as nw

    try:
        nwd = nw.from_native(df, eager_only=True)
        if nwd.implementation.is_polars():
            return nw.to_native(nwd), True
    except Exception:
        pass
    return df, False


class BulkOps:
    """Batched mutation API for :class:`annnet.core.graph.AnnNet`.

    These methods mirror the scalar graph construction API while reducing
    repeated dataframe updates, sparse-matrix growth, and index maintenance.
    """

    def add_vertices_bulk(self, vertices, layer=None, slice=None):
        """Add many vertices in one pass.

        Parameters
        ----------
        vertices : Iterable[str] | Iterable[tuple[str, dict]] | Iterable[dict]
            Vertices to add. Each item can be:
            - `vertex_id` (str)
            - `(vertex_id, attrs)` tuple
            - dict containing `vertex_id` plus attributes
        layer : str | dict | tuple | None, optional
            Layer spec for all vertices in this batch. Forwarded to
            ``_make_layer_coord`` once — no per-vertex overhead.
            ``None`` means the flat/default layer.
        slice : str, optional
            Target slice. Defaults to the active slice.

        Returns
        -------
        None

        Notes
        -----
        This is the batched companion to :meth:`annnet.core.graph.AnnNet.add_vertex`.
        A Polars fast path is used when possible; otherwise the method falls
        back to a schema-safe row-upsert implementation.
        """
        slice = slice or self._current_slice

        # Normalize input

        norm_vids = []
        norm_attrs = []
        for it in vertices:
            if isinstance(it, dict):
                if it.get("vertex_id"):
                    vid = it["vertex_id"]
                    _id_keys = {"vertex_id"}
                elif it.get("id"):
                    vid = it["id"]
                    _id_keys = {"vertex_id", "id"}
                elif it.get("name"):
                    vid = it["name"]
                    _id_keys = {"vertex_id", "id", "name"}
                else:
                    vid = None
                if vid is None:
                    continue
                attrs = {k: v for k, v in it.items() if k not in _id_keys}
            elif isinstance(it, (tuple, list)) and it:
                vid = it[0]
                attrs = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
            else:
                vid = it
                attrs = {}
            norm_vids.append(vid)
            norm_attrs.append(attrs)

        if not norm_vids:
            return

        # Intern hot strings
        try:
            import sys as _sys

            norm_vids = [_sys.intern(v) if isinstance(v, str) else v for v in norm_vids]
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            pass

        # Entity registration — compute layer coord ONCE for the whole batch
        coord = self._resolve_vertex_insert_coord(
            layer,
            vertex_ids=norm_vids,
            context="add_vertices_bulk",
        )

        new_rows = 0
        for vid in norm_vids:
            ekey = (vid, coord)
            if ekey not in self._entities:
                idx = len(self._entities)
                self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind="vertex"))
                new_rows += 1
            # Maintain _V / _VM caches for _Layers.py compat
            self._V.add(vid)
            self._VM.add(ekey)
        if new_rows:
            self._grow_rows_to(len(self._entities))

        # Slice membership (slice tracks bare vids for _Slices.py compat)

        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(norm_vids)

        # Ensure attribute table exists

        self._ensure_vertex_table()
        df = self.vertex_attributes
        df, is_pl = _to_polars_if_possible(df)

        # If not Polars, fall back to Narwhals

        if pl is None or not isinstance(df, pl.DataFrame):
            return self._add_vertices_bulk_fallback(vertices, layer=layer, slice=slice)

        # Build ONE incoming DF
        # Collect all keys once to avoid repeated set growth in loops
        keys = set()
        for a in norm_attrs:
            keys.update(a.keys())

        cols = {"vertex_id": norm_vids}
        # build columns with a single pass per key
        for k in keys:
            col = [a.get(k, None) for a in norm_attrs]
            cols[k] = col

        incoming = pl.DataFrame(cols, nan_to_null=True, strict=False)

        # Ensure df has needed columns

        if keys:
            # _ensure_attr_columns returns a Narwhals DF; convert back to native Polars
            df_tmp = self._ensure_attr_columns(df, {k: None for k in keys})
            df, is_pl2 = _to_polars_if_possible(df_tmp)
            if not is_pl2:
                # backend drifted away from Polars -> fallback to Narwhals behavior
                return self._add_vertices_bulk_fallback(vertices, layer=layer, slice=slice)

        # Split inserts vs updates
        nrows = len(df)
        id_df = df.select("vertex_id") if ("vertex_id" in df.columns and nrows > 0) else None

        if id_df is None:
            # df empty: everything is insert
            to_insert = incoming
            to_update = None
        else:
            # anti-join = new rows, semi-join = existing rows
            to_insert = incoming.join(id_df, on="vertex_id", how="anti")
            to_update = incoming.join(id_df, on="vertex_id", how="semi")

        # Schema alignment helper

        def _align_numeric_and_string(df_left: pl.DataFrame, df_right: pl.DataFrame):
            # Cast both sides to a compatible dtype for each shared column
            left = df_left
            right = df_right
            for c in left.columns:
                if c not in right.columns:
                    right = right.with_columns(pl.lit(None).alias(c))
            for c in right.columns:
                if c not in left.columns:
                    left = left.with_columns(pl.lit(None).alias(c))

            # Now cast column-by-column
            for c in left.columns:
                lc = left.schema[c]
                rc = right.schema[c]
                if lc == pl.Null and rc != pl.Null:
                    left = left.with_columns(pl.col(c).cast(rc))
                elif rc == pl.Null and lc != pl.Null:
                    right = right.with_columns(pl.col(c).cast(lc).alias(c))
                elif lc != rc:
                    if lc.is_numeric() and rc.is_numeric():
                        supertype = _get_numeric_supertype(lc, rc)
                        left = left.with_columns(pl.col(c).cast(supertype))
                        right = right.with_columns(pl.col(c).cast(supertype).alias(c))
                    else:
                        left = left.with_columns(pl.col(c).cast(pl.Utf8))
                        right = right.with_columns(pl.col(c).cast(pl.Utf8).alias(c))
            # Keep identical column order
            right = right.select(left.columns)
            return left, right

        # Inserts: one concat

        if to_insert is not None and len(to_insert) > 0:
            df, to_insert = _align_numeric_and_string(df, to_insert)
            df = pl.concat([df, to_insert], how="vertical", rechunk=False)

        # Updates: one join + one coalesce pass

        if to_update is not None and len(to_update) > 0 and keys:
            df, to_update = _align_numeric_and_string(df, to_update)

            suffix = "__new"

            left_dupes = [c for c in df.columns if c.endswith(suffix)]
            if left_dupes:
                df = df.drop(left_dupes)

            right_dupes = [c for c in to_update.columns if c.endswith(suffix)]
            if right_dupes:
                to_update = to_update.drop(right_dupes)

            # Join once; suffix new columns
            df2 = df.join(to_update, on="vertex_id", how="left", suffix=suffix)

            # Build expressions once and update only the provided keys
            exprs = []
            drops = []
            for k in keys:
                nk = k + "__new"
                if k in df2.columns and nk in df2.columns:
                    exprs.append(pl.coalesce([pl.col(nk), pl.col(k)]).alias(k))
                    drops.append(nk)

            if exprs:
                df2 = df2.with_columns(exprs)
            if drops:
                df2 = df2.drop(drops)

            df = df2

        self.vertex_attributes = df

    def _add_vertices_bulk_fallback(self, vertices, layer=None, slice=None):
        """Fallback implementation for :meth:`add_vertices_bulk`.

        Parameters
        ----------
        vertices : Iterable[str] | Iterable[tuple[str, dict]] | Iterable[dict]
            Vertices to add.
        slice : str, optional
            Target slice. Defaults to the active slice.

        Returns
        -------
        None

        Notes
        -----
        This path is slower than the Polars fast path but preserves the same
        input contract and side effects.
        """

        slice = slice or self._current_slice

        # NORMALIZE INPUT

        norm = []
        for it in vertices:
            if isinstance(it, dict):
                if it.get("vertex_id"):
                    vid = it["vertex_id"]
                    _id_keys = {"vertex_id"}
                elif it.get("id"):
                    vid = it["id"]
                    _id_keys = {"vertex_id", "id"}
                elif it.get("name"):
                    vid = it["name"]
                    _id_keys = {"vertex_id", "id", "name"}
                else:
                    vid = None
                if vid is None:
                    continue
                attrs = {k: v for k, v in it.items() if k not in _id_keys}
                norm.append((vid, attrs))

            elif isinstance(it, (tuple, list)) and it:
                vid = it[0]
                attrs = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
                norm.append((vid, attrs))

            else:
                norm.append((it, {}))

        if not norm:
            return

        # Intern hot strings
        try:
            import sys as _sys

            norm = [
                (_sys.intern(vid) if isinstance(vid, str) else vid, attrs) for vid, attrs in norm
            ]
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            pass

        # ENTITY REGISTRATION — compute layer coord once for the whole batch
        coord = self._resolve_vertex_insert_coord(
            layer,
            vertex_ids=[vid for vid, _ in norm],
            context="add_vertices_bulk",
        )

        new_rows = 0
        for vid, _ in norm:
            ekey = (vid, coord)
            if ekey not in self._entities:
                idx = len(self._entities)
                self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind="vertex"))
                self._V.add(vid)
                self._VM.add(ekey)
                new_rows += 1

        if new_rows:
            self._grow_rows_to(len(self._entities))

        # SLICE MEMBERSHIP
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(v for v, _ in norm)

        # ATTRIBUTE TABLE PREP
        self._ensure_vertex_table()
        df = self.vertex_attributes

        # Build lookup of existing vertex_ids once
        try:
            import polars as pl
        except Exception:
            pl = None

        existing_ids = set()
        try:
            if pl is not None and isinstance(df, pl.DataFrame):
                if df.height > 0 and "vertex_id" in df.columns:
                    existing_ids = set(df.get_column("vertex_id").to_list())
            else:
                import narwhals as nw

                native = nw.to_native(nw.from_native(df).select("vertex_id"))
                col = native["vertex_id"]
                existing_ids = set(col.to_list() if hasattr(col, "to_list") else list(col))
        except Exception:
            existing_ids = set()

        # Build new rows (for vertex_ids missing in DF)

        new_rows_data = []
        new_attr_keys = set()

        for vid, attrs in norm:
            if vid not in existing_ids:
                cols = list(df.columns) if hasattr(df, "columns") else []
                row = {c: None for c in cols} if len(cols) > 0 else {"vertex_id": None}
                row["vertex_id"] = vid
                for k, v in attrs.items():
                    row[k] = _sanitize(v)
                    new_attr_keys.add(k)
                new_rows_data.append(row)
            else:
                for k in attrs.keys():
                    new_attr_keys.add(k)

        # Ensure DF has columns for all attributes used
        if new_attr_keys:
            df = self._ensure_attr_columns(df, dict.fromkeys(new_attr_keys))

        # Vectorized insert of new rows
        try:
            import polars as pl
        except Exception:
            pl = None

        if new_rows_data:
            # Polars fast-path (keep vectorized semantics)
            if pl is not None and isinstance(df, pl.DataFrame):
                add_df = pl.DataFrame(new_rows_data, nan_to_null=True, strict=False)

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
                        if lc.is_numeric() and rc.is_numeric():
                            supertype = _get_numeric_supertype(lc, rc)
                            df = df.with_columns(pl.col(c).cast(supertype))
                            add_df = add_df.with_columns(pl.col(c).cast(supertype).alias(c))
                        else:
                            df = df.with_columns(pl.col(c).cast(pl.Utf8))
                            add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

                add_df = add_df.select(df.columns)
                df = df.vstack(add_df)

            # Non-Polars fallback: do row-wise upserts (correct, slower)
            else:
                for row in new_rows_data:
                    vid = row.get("vertex_id")
                    attrs_only = {
                        k: v for k, v in row.items() if k != "vertex_id" and v is not None
                    }
                    df = self._upsert_row(df, vid, attrs_only)

        # VECTORIZED UPSERT OF EXISTING ATTRIBUTES
        try:
            import polars as pl
        except Exception:
            pl = None

        update_pairs = [(vid, attrs) for vid, attrs in norm if vid in existing_ids and attrs]

        if update_pairs:
            if pl is not None and isinstance(df, pl.DataFrame):
                update_df = pl.DataFrame(
                    {
                        "vertex_id": [vid for vid, _ in update_pairs],
                        **{
                            k: [_sanitize(attrs.get(k, None)) for _, attrs in update_pairs]
                            for k in new_attr_keys
                        },
                    },
                    nan_to_null=True,
                    strict=False,
                )

                for c in update_df.columns:
                    if c not in df.columns:
                        continue
                    lc, rc = df.schema[c], update_df.schema[c]
                    if lc == pl.Null and rc != pl.Null:
                        df = df.with_columns(pl.col(c).cast(rc))
                    elif rc == pl.Null and lc != pl.Null:
                        update_df = update_df.with_columns(pl.col(c).cast(lc).alias(c))
                    elif lc != rc:
                        if lc.is_numeric() and rc.is_numeric():
                            supertype = _get_numeric_supertype(lc, rc)
                            df = df.with_columns(pl.col(c).cast(supertype))
                            update_df = update_df.with_columns(pl.col(c).cast(supertype).alias(c))
                        else:
                            df = df.with_columns(pl.col(c).cast(pl.Utf8))
                            update_df = update_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))
                suffix = "_new"
                df_dupes = [c for c in df.columns if c.endswith(suffix)]
                if df_dupes:
                    df = df.drop(df_dupes)

                ud_dupes = [c for c in update_df.columns if c.endswith(suffix)]
                if ud_dupes:
                    update_df = update_df.drop(ud_dupes)
                df = df.join(update_df, on="vertex_id", how="left", suffix="_new")
                for c in new_attr_keys:
                    if c in df.columns and c + "_new" in df.columns:
                        df = df.with_columns(
                            pl.coalesce([pl.col(c + "_new"), pl.col(c)]).alias(c)
                        ).drop(c + "_new")

            else:
                # Non-Polars fallback: row-wise updates
                for vid, attrs in update_pairs:
                    df = self._upsert_row(df, vid, {k: _sanitize(v) for k, v in attrs.items()})

        self.vertex_attributes = df

    def add_edges_bulk(
        self,
        edges,
        *,
        slice=None,
        as_entity=False,
        default_weight=1.0,
        default_edge_type="regular",
        default_propagate="none",
        default_slice_weight=None,
        default_edge_directed=None,
    ):
        """Add many binary or vertex-edge edges in one pass.

        Parameters
        ----------
        edges : Iterable
            Batch payload. Each item may be a tuple ``(source, target)``,
            ``(source, target, weight)``, or a dict with ``source`` and
            ``target`` plus optional edge fields.
        slice : str, optional
            Default slice to place edges into.
        as_entity : bool, optional
            If True, each created edge is also registered as a connectable
            entity (gets a matrix row). Equivalent to ``as_entity=True`` on
            :meth:`add_edge`. Default False.
        default_weight : float, optional
            Default weight for edges missing an explicit weight.
        default_edge_type : str, optional
            Default edge type when not provided explicitly.
        default_propagate : {'none', 'shared', 'all'}, optional
            Default slice propagation mode.
        default_slice_weight : float, optional
            Default per-slice weight override.
        default_edge_directed : bool, optional
            Default per-edge directedness override.

        Returns
        -------
        list[str]
            Edge IDs for created/updated edges.

        Notes
        -----
        This is the batched companion to :meth:`annnet.core.graph.AnnNet.add_edge`
        for binary and vertex-edge payloads.
        """
        slice = self._current_slice if slice is None else slice
        pending_attrs = {}

        # Normalize into dicts
        norm = []
        for it in edges:
            if isinstance(it, dict):
                d = dict(it)
                # Accept src/tgt aliases for source/target
                if "src" in d and "source" not in d:
                    d["source"] = d.pop("src")
                if "tgt" in d and "target" not in d:
                    d["target"] = d.pop("tgt")
                # Accept directed alias for edge_directed
                if "directed" in d and "edge_directed" not in d:
                    d["edge_directed"] = d.pop("directed")
            elif isinstance(it, (tuple, list)):
                if len(it) == 2:
                    d = {"source": it[0], "target": it[1], "weight": default_weight}
                else:
                    d = {"source": it[0], "target": it[1], "weight": it[2]}
            else:
                continue
            d.setdefault("weight", default_weight)
            d.setdefault("edge_type", default_edge_type)
            d.setdefault("propagate", default_propagate)
            if "slice" not in d:
                d["slice"] = slice
            if "edge_directed" not in d:
                d["edge_directed"] = default_edge_directed
            norm.append(d)

        if not norm:
            return []

        # Intern hot strings & coerce weights
        try:
            import sys as _sys

            for d in norm:
                s, t = d["source"], d["target"]
                if isinstance(s, str):
                    d["source"] = _sys.intern(s)
                if isinstance(t, str):
                    d["target"] = _sys.intern(t)
                lid = d.get("slice")
                if isinstance(lid, str):
                    d["slice"] = _sys.intern(lid)
                eid = d.get("edge_id")
                if isinstance(eid, str):
                    d["edge_id"] = _sys.intern(eid)
                try:
                    d["weight"] = float(d["weight"])
                except Exception:
                    pass
        except Exception:
            pass

        M = self._matrix
        # Bypass scipy DOK __setitem__ validation chain (isintlike + ndim + asarray per write).
        # _set_intXint skips validation and writes directly to the backing store.
        # Falls back to plain __setitem__ on very old scipy that lacks this private method.
        _m_fast_set = getattr(M, "_set_intXint", None)
        _m_dtype = M.dtype.type  # e.g. np.float32

        # 1) Ensure endpoints exist — collect unique vids first, resolve coord once,
        #    then build endpoint_cache: {vid: row_idx} for O(1) lookup in inner loop.
        unique_vids: dict[str, str] = {}  # vid -> edge_type (for vertex_edge check)
        for d in norm:
            s, t = d["source"], d["target"]
            et = d.get("edge_type", "regular")
            unique_vids[s] = et
            unique_vids[t] = et

        # Flat graphs: coord is fixed for all binary endpoints
        coord = self._make_layer_coord(None)

        endpoint_cache: dict[str, int] = {}  # vid -> row_idx
        for vid, et in unique_vids.items():
            ekey = (vid, coord)
            if ekey not in self._entities:
                if et == "vertex_edge" and isinstance(vid, str) and vid.startswith("edge_"):
                    self._ensure_edge_entity_placeholder(vid)
                else:
                    idx = len(self._entities)
                    self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind="vertex"))
                    self._V.add(vid)
                    self._VM.add(ekey)
            endpoint_cache[vid] = self._entities[ekey].row_idx

        # Grow rows once if needed
        self._grow_rows_to(len(self._entities))

        # 2) Pre-size columns for new edges and pre-generate auto edge IDs
        _edges_store = self._edges
        new_count = 0
        _need_auto_id = []  # indices into norm that need auto edge_id
        for _i, d in enumerate(norm):
            eid = d.get("edge_id")
            if eid not in _edges_store or _edges_store[eid].col_idx < 0:
                new_count += 1
            if eid is None:
                _need_auto_id.append(_i)

        if new_count:
            self._grow_cols_to(len(self._col_to_edge) + new_count)

        # Pre-generate auto IDs in one pass (avoids 50k f-string format calls in loop)
        if _need_auto_id:
            _base_id = self._next_edge_id
            self._next_edge_id += len(_need_auto_id)
            _auto_ids = iter(range(_base_id, _base_id + len(_need_auto_id)))
            for _i in _need_auto_id:
                norm[_i]["edge_id"] = f"edge_{next(_auto_ids)}"

        # 3) Create/update columns
        out_ids = []
        # Batch matrix writes: collect (row, col) -> val for new/updated entries,
        # and keys to zero-out (update path only). Applied in one shot after the loop.
        _M_writes: dict = {}
        _M_zero_keys: list = []
        # Batch slice membership: collect per-slice edge IDs and vertex pairs.
        # Applied after the loop with single bulk set.update() calls.
        _slice_eids: dict = {}  # slice_id -> list[edge_id]
        _slice_vids: dict = {}  # slice_id -> list[vid]
        _slice_weights: list = []  # (slice_id, edge_id, weight) for per-slice weights

        for d in norm:
            s, t = d["source"], d["target"]
            w = d["weight"]
            etype = d.get("edge_type", "regular")
            prop = d.get("propagate", default_propagate)
            slice_local = d.get("slice", slice)
            slice_w = d.get("slice_weight", default_slice_weight)
            e_dir = d.get("edge_directed", default_edge_directed)
            edge_id = d.get("edge_id")

            if e_dir is not None:
                is_dir = bool(e_dir)
            elif self.directed is not None:
                is_dir = self.directed
            else:
                is_dir = True
            s_idx = endpoint_cache[s]
            t_idx = endpoint_cache[t]

            fw = _m_dtype(w)

            # update vs create
            if edge_id in self._edges and self._edges[edge_id].col_idx >= 0:
                rec = self._edges[edge_id]
                col = rec.col_idx
                old_s, old_t = rec.src, rec.tgt
                try:
                    _M_zero_keys.append((self._entity_row(old_s), col))
                except Exception:
                    pass
                if old_t is not None and old_t != old_s:
                    try:
                        _M_zero_keys.append((self._entity_row(old_t), col))
                    except Exception:
                        pass
                _M_writes[(s_idx, col)] = fw
                if s != t:
                    _M_writes[(t_idx, col)] = _m_dtype(-w) if is_dir else fw
                rec.src = s
                rec.tgt = t
                rec.weight = w
                rec.directed = is_dir
                if (old_s, old_t) != (s, t):
                    lst = self._adj.get((old_s, old_t))
                    if lst:
                        try:
                            lst.remove(edge_id)
                        except ValueError:
                            pass
                        if not lst:
                            del self._adj[(old_s, old_t)]
                    self._adj.setdefault((s, t), []).append(edge_id)
                    for _old, _new, _index in (
                        (old_s, s, self._src_to_edges),
                        (old_t, t, self._tgt_to_edges),
                    ):
                        if _old != _new:
                            _lst = _index.get(_old)
                            if _lst:
                                try:
                                    _lst.remove(edge_id)
                                except ValueError:
                                    pass
                                if not _lst:
                                    del _index[_old]
                            _index.setdefault(_new, []).append(edge_id)
                pending_attrs.setdefault(edge_id, {})["edge_type"] = (
                    EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED
                )
            else:
                col = len(self._col_to_edge)
                self._col_to_edge[col] = edge_id
                if edge_id in self._edges:
                    rec = self._edges[edge_id]
                    rec.src = s
                    rec.tgt = t
                    rec.weight = w
                    rec.directed = is_dir
                    rec.etype = "binary"
                    rec.col_idx = col
                else:
                    self._edges[edge_id] = EdgeRecord(
                        src=s,
                        tgt=t,
                        weight=w,
                        directed=is_dir,
                        etype="binary",
                        col_idx=col,
                        ml_kind=None,
                        ml_layers=None,
                        direction_policy=None,
                    )
                _M_writes[(s_idx, col)] = fw
                if s != t:
                    _M_writes[(t_idx, col)] = _m_dtype(-w) if is_dir else fw
                self._adj.setdefault((s, t), []).append(edge_id)
                self._src_to_edges.setdefault(s, []).append(edge_id)
                self._tgt_to_edges.setdefault(t, []).append(edge_id)

            # slice membership + optional per-slice weight (batched, applied after loop)
            if slice_local is not None:
                _lst = _slice_eids.get(slice_local)
                if _lst is None:
                    _slice_eids[slice_local] = [edge_id]
                    _slice_vids[slice_local] = [s, t]
                else:
                    _lst.append(edge_id)
                    _slice_vids[slice_local].extend((s, t))
                if slice_w is not None:
                    _slice_weights.append((slice_local, edge_id, float(slice_w)))

            # propagation
            if prop == "shared":
                self._propagate_to_shared_slices(edge_id, s, t)
            elif prop == "all":
                self._propagate_to_all_slices(edge_id, s, t)

            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                pending_attrs.setdefault(edge_id, {}).update(attrs)

            out_ids.append(edge_id)

        # Apply matrix writes in one batch (C-level dict ops, bypasses per-call Python overhead)
        for key in _M_zero_keys:
            dict.pop(M, key, None)
        if _M_writes:
            dict.update(M, _M_writes)
            self._csr_cache = None

        # Flush batched slice membership (one set.update per slice instead of E individual adds)
        for sid, eids in _slice_eids.items():
            if sid not in self._slices:
                self._slices[sid] = {"vertices": set(), "edges": set(), "attributes": {}}
            self._slices[sid]["edges"].update(eids)
        for sid, vids in _slice_vids.items():
            self._slices[sid]["vertices"].update(vids)
        for sid, eid, sw in _slice_weights:
            self.set_edge_slice_attrs(sid, eid, weight=sw)
            self.slice_edge_weights.setdefault(sid, {})[eid] = sw

        # flush all edge attribute writes in one bulk call
        if pending_attrs:
            self.set_edge_attrs_bulk(pending_attrs)

        # register all created edges as connectable entities (batch, after the loop)
        if as_entity:
            for eid in out_ids:
                self._register_edge_as_entity(eid)
                rec = self._edges[eid]
                if rec.etype == "binary":
                    rec.etype = "vertex_edge"

        return out_ids

    def add_hyperedges_bulk(
        self,
        hyperedges,
        *,
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Add many hyperedges in one pass.

        Parameters
        ----------
        hyperedges : Iterable[dict]
            Hyperedge payloads. Each item may define either ``members`` for an
            undirected hyperedge or ``head`` and ``tail`` for a directed one.
        slice : str, optional
            Default slice for hyperedges missing an explicit slice.
        default_weight : float, optional
            Default weight for hyperedges missing an explicit weight.
        default_edge_directed : bool, optional
            Default directedness override.

        Returns
        -------
        list[str]
            Hyperedge IDs for created/updated hyperedges.

        Notes
        -----
        This method is the batched hyperedge companion to
        :meth:`annnet.core.graph.AnnNet.add_edge`.
        """
        slice = self._current_slice if slice is None else slice

        items = []
        for it in hyperedges:
            if not isinstance(it, dict):
                continue
            d = dict(it)
            # Accept directed alias for edge_directed
            if "directed" in d and "edge_directed" not in d:
                d["edge_directed"] = d.pop("directed")
            d.setdefault("weight", default_weight)
            if "slice" not in d:
                d["slice"] = slice
            if "edge_directed" not in d:
                d["edge_directed"] = default_edge_directed
            items.append(d)

        if not items:
            return []

        # Intern + coerce
        try:
            import sys as _sys

            for d in items:
                if "members" in d and d["members"] is not None:
                    d["members"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d["members"]
                    ]
                else:
                    d["head"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get("head", [])
                    ]
                    d["tail"] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get("tail", [])
                    ]
                lid = d.get("slice")
                if isinstance(lid, str):
                    d["slice"] = _sys.intern(lid)
                eid = d.get("edge_id")
                if isinstance(eid, str):
                    d["edge_id"] = _sys.intern(eid)
                try:
                    d["weight"] = float(d["weight"])
                except Exception:
                    pass
        except Exception:
            pass

        # Collect ALL unique vertices first
        all_verts = set()
        for d in items:
            if "members" in d and d["members"] is not None:
                all_verts.update(d["members"])
            else:
                all_verts.update(d.get("head", []))
                all_verts.update(d.get("tail", []))

        # Single pass vertex creation — compute coord once for the flat/default layer
        coord = self._make_layer_coord(None)
        for u in all_verts:
            ekey = (u, coord)
            if ekey not in self._entities:
                idx = len(self._entities)
                self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind="vertex"))
                self._V.add(u)
                self._VM.add(ekey)

        self._grow_rows_to(len(self._entities))

        # Pre-size columns
        new_count = sum(1 for d in items if d.get("edge_id") not in self._edges)
        if new_count:
            self._grow_cols_to(len(self._col_to_edge) + new_count)

        M = self._matrix
        slices = self._slices

        # Bypass scipy DOK __setitem__ validation chain (isintlike + ndim + asarray per write).
        # _set_intXint skips validation and writes directly to the backing store (~0.65µs vs ~18µs).
        # Falls back to plain __setitem__ on very old scipy that lacks this private method.
        _m_fast_set = getattr(M, "_set_intXint", None)
        _m_dtype = M.dtype.type  # e.g. np.float32

        out_ids = []

        # Batch attribute writes
        attrs_batch = {}

        for d in items:
            members = d.get("members")
            head = d.get("head")
            tail = d.get("tail")
            slice_local = d.get("slice", slice)
            w = float(d.get("weight", default_weight))
            e_id = d.get("edge_id")

            # Decide directedness from form unless forced
            directed = d.get("edge_directed")
            if directed is None:
                directed = members is None

            # allocate/update column
            if e_id is None:
                e_id = self._get_next_edge_id()

            if e_id in self._edges:
                rec = self._edges[e_id]
                col = rec.col_idx
                # clear old cells
                if rec.etype == "hyper":
                    old_verts = rec.src if rec.tgt is None else (rec.src | rec.tgt)
                    for vid in old_verts:
                        try:
                            r = self._entity_row(vid)
                            if _m_fast_set is not None:
                                _m_fast_set(r, col, 0)
                            else:
                                M[r, col] = 0
                        except Exception:
                            pass
                else:
                    for vid in (rec.src, rec.tgt):
                        if vid is None:
                            continue
                        try:
                            r = self._entity_row(vid)
                            if _m_fast_set is not None:
                                _m_fast_set(r, col, 0)
                            else:
                                M[r, col] = 0
                        except Exception:
                            pass
            else:
                col = len(self._col_to_edge)
                self._col_to_edge[col] = e_id
                rec = EdgeRecord(
                    src=None,
                    tgt=None,
                    weight=1.0,
                    directed=False,
                    etype="hyper",
                    col_idx=col,
                    ml_kind=None,
                    ml_layers=None,
                    direction_policy=None,
                )
                self._edges[e_id] = rec

            # write new column values + metadata
            if members is not None:
                fw = _m_dtype(w)
                if _m_fast_set is not None:
                    for u in members:
                        _m_fast_set(self._entity_row(u), col, fw)
                else:
                    for u in members:
                        M[self._entity_row(u), col] = fw
                rec.src = frozenset(members)
                rec.tgt = None
                rec.directed = False
            else:
                fw = _m_dtype(w)
                neg_fw = _m_dtype(-w)
                if _m_fast_set is not None:
                    for u in head:
                        _m_fast_set(self._entity_row(u), col, fw)
                    for v in tail:
                        _m_fast_set(self._entity_row(v), col, neg_fw)
                else:
                    for u in head:
                        M[self._entity_row(u), col] = fw
                    for v in tail:
                        M[self._entity_row(v), col] = neg_fw
                rec.src = frozenset(head)
                rec.tgt = frozenset(tail)
                rec.directed = True

            rec.weight = w
            rec.etype = "hyper"

            # slice membership
            if slice_local is not None:
                if slice_local not in slices:
                    slices[slice_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                slices[slice_local]["edges"].add(e_id)
                if members is not None:
                    slices[slice_local]["vertices"].update(members)
                else:
                    slices[slice_local]["vertices"].update(head)
                    slices[slice_local]["vertices"].update(tail)

            # Collect attributes for batch write
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                attrs_batch[e_id] = attrs

            out_ids.append(e_id)

        self._csr_cache = None  # matrix was mutated; invalidate cached CSR

        # SINGLE BULK WRITE FOR ALL ATTRIBUTES
        if attrs_batch:
            self.set_edge_attrs_bulk(attrs_batch)

        return out_ids

    def add_edges_to_slice_bulk(self, slice_id, edge_ids):
        """Add many edges to a slice and attach all incident vertices.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_ids : Iterable[str]
            Edge identifiers to add.

        Returns
        -------
        None

        Notes
        -----
        No weights are changed in this operation.
        """
        slice = slice_id if slice_id is not None else self._current_slice
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        L = self._slices[slice]

        add_edges = {
            eid for eid in edge_ids if eid in self._edges and self._edges[eid].col_idx >= 0
        }
        if not add_edges:
            return

        L["edges"].update(add_edges)

        verts = set()
        for eid in add_edges:
            rec = self._edges[eid]
            if rec.etype == "hyper":
                verts.update(rec.src)
                if rec.tgt is not None:
                    verts.update(rec.tgt)
            else:
                if rec.src is not None:
                    verts.add(rec.src)
                if rec.tgt is not None:
                    verts.add(rec.tgt)

        L["vertices"].update(verts)

    def set_vertex_key(self, *fields: str):
        """Declare composite key fields and rebuild the uniqueness index.

        Parameters
        ----------
        *fields : str
            Ordered field names used to build a composite key.

        Raises
        ------
        ValueError
            If duplicates exist among already-populated vertices.

        Notes
        -----
        Vertices missing some key fields are skipped during indexing.
        """
        if not fields:
            raise ValueError("set_vertex_key requires at least one field")
        self._vertex_key_fields = tuple(str(f) for f in fields)
        self._vertex_key_index.clear()

        df = self.vertex_attributes

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return
        else:
            try:
                if df is None or len(df) == 0:
                    return
            except Exception:
                return

        missing = [f for f in self._vertex_key_fields if f not in df.columns]
        if missing:
            # ok to skip; those rows simply won't be indexable until fields appear
            pass

        # Rebuild index, enforcing uniqueness only for fully-populated tuples

        if pl is not None and isinstance(df, pl.DataFrame):
            try:
                for row in df.iter_rows(named=True):
                    vid = row.get("vertex_id")
                    key = tuple(row.get(f) for f in self._vertex_key_fields)
                    if any(v is None for v in key):
                        continue
                    owner = self._vertex_key_index.get(key)
                    if owner is not None and owner != vid:
                        raise ValueError(f"Composite key conflict for {key}: {owner} vs {vid}")
                    self._vertex_key_index[key] = vid
                return
            except Exception:
                pass  # fall through to generic
        # generic path
        try:
            import narwhals as nw

            native = nw.to_native(nw.from_native(df))
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            for row in rows:
                vid = row.get("vertex_id")
                key = tuple(row.get(f) for f in self._vertex_key_fields)
                if any(v is None for v in key):
                    continue
                owner = self._vertex_key_index.get(key)
                if owner is not None and owner != vid:
                    raise ValueError(f"Composite key conflict for {key}: {owner} vs {vid}")
                self._vertex_key_index[key] = vid
        except Exception:
            # last-resort fallback: per-vid lookups
            try:
                vids = df.get_column("vertex_id").to_list()  # polars
            except Exception:
                try:
                    vids = list(df["vertex_id"])
                except Exception:
                    vids = []
            for vid in vids:
                cur = {f: self.get_attr_vertex(vid, f, None) for f in self._vertex_key_fields}
                key = self._build_key_from_attrs(cur)
                if key is None:
                    continue
                owner = self._vertex_key_index.get(key)
                if owner is not None and owner != vid:
                    raise ValueError(f"Composite key conflict for {key}: {owner} vs {vid}")
                self._vertex_key_index[key] = vid

    # Bulk remove / mutate down

    def remove_edges(self, edge_ids):
        """Remove many edges in one pass.

        Parameters
        ----------
        edge_ids : Iterable[str]
            Edge identifiers to remove.

        Returns
        -------
        None

        Notes
        -----
        This is faster than calling `remove_edge` in a loop.
        """
        to_drop = [eid for eid in edge_ids if eid in self._edges and self._edges[eid].col_idx >= 0]
        if not to_drop:
            return
        self._remove_edges_bulk(to_drop)

    def remove_vertices(self, vertex_ids):
        """Remove many vertices (and their incident edges) in one pass.

        Parameters
        ----------
        vertex_ids : Iterable[str]
            Vertex identifiers to remove.

        Returns
        -------
        None

        Notes
        -----
        This is faster than calling `remove_vertex` in a loop.
        """
        to_drop = [vid for vid in vertex_ids if vid in self._entities]
        if not to_drop:
            return
        self._remove_vertices_bulk(to_drop)

    def _remove_edges_bulk(self, edge_ids):
        drop = set(edge_ids)
        if not drop:
            return

        # Columns to keep, old->new remap.
        # _col_to_edge is insertion-ordered (col keys are 0,1,2,...) so iterating it
        # gives columns in ascending order without sorting — O(E) vs O(E log E).
        keep_pairs = [(col, eid) for col, eid in self._col_to_edge.items() if eid not in drop]
        old_to_new = {old: new for new, (old, _eid) in enumerate(keep_pairs)}
        new_cols = len(keep_pairs)

        # Rebuild matrix once (batch dict op avoids per-entry Python __setitem__ overhead)
        M_old = self._matrix  # DOK
        rows, _cols = M_old.shape
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        new_data = {(r, old_to_new[c]): v for (r, c), v in M_old.items() if c in old_to_new}
        if new_data:
            dict.update(M_new, new_data)
        self._matrix = M_new
        self._csr_cache = None

        # Rebuild edge col mappings
        self._col_to_edge.clear()
        for new_idx, (old_idx, eid) in enumerate(keep_pairs):
            self._col_to_edge[new_idx] = eid
            self._edges[eid].col_idx = new_idx

        # Adjacency + primary record cleanup
        for eid in drop:
            rec = self._edges.pop(eid, None)
            if (
                rec is not None
                and rec.etype != "hyper"
                and rec.src is not None
                and rec.tgt is not None
            ):
                s, t = rec.src, rec.tgt
                lst = self._adj.get((s, t))
                if lst:
                    try:
                        lst.remove(eid)
                    except ValueError:
                        pass
                    if not lst:
                        del self._adj[(s, t)]
                for v, index in ((s, self._src_to_edges), (t, self._tgt_to_edges)):
                    _lst = index.get(v)
                    if _lst:
                        try:
                            _lst.remove(eid)
                        except ValueError:
                            pass
                        if not _lst:
                            del index[v]
            # Clear multilayer metadata kept on EdgeRecord.
            rec2 = self._edges.get(eid)
            if rec2 is not None:
                rec2.ml_kind = None
            self.edge_layers.pop(eid, None)
        for slice_data in self._slices.values():
            slice_data["edges"].difference_update(drop)
        for d in self.slice_edge_weights.values():
            for eid in drop:
                d.pop(eid, None)

        # DataFrames
        ea = self.edge_attributes
        if ea is not None and hasattr(ea, "columns") and "edge_id" in ea.columns:
            if pl is not None and isinstance(ea, pl.DataFrame) and ea.height:
                self.edge_attributes = ea.filter(~pl.col("edge_id").is_in(list(drop)))
            else:
                import narwhals as nw

                self.edge_attributes = nw.to_native(
                    nw.from_native(ea).filter(~nw.col("edge_id").is_in(list(drop)))
                )
        ela = self.edge_slice_attributes
        if ela is not None and hasattr(ela, "columns") and "edge_id" in ela.columns:
            if pl is not None and isinstance(ela, pl.DataFrame) and ela.height:
                self.edge_slice_attributes = ela.filter(~pl.col("edge_id").is_in(list(drop)))
            else:
                import narwhals as nw

                self.edge_slice_attributes = nw.to_native(
                    nw.from_native(ela).filter(~nw.col("edge_id").is_in(list(drop)))
                )

    def _remove_vertices_bulk(self, vertex_ids):
        drop_vs = set(vertex_ids)
        if not drop_vs:
            return

        # 1) Collect incident edges (binary + hyper) in one pass
        drop_es = set()
        for eid, rec in list(self._edges.items()):
            if rec.etype == "hyper":
                if drop_vs & set(rec.src):
                    drop_es.add(eid)
                elif rec.tgt is not None and (drop_vs & set(rec.tgt)):
                    drop_es.add(eid)
            else:
                if rec.src in drop_vs or rec.tgt in drop_vs:
                    drop_es.add(eid)

        # 2) Drop all those edges in one pass
        if drop_es:
            self._remove_edges_bulk(drop_es)

        # 3) Build row keep list and old->new map
        keep_idx = sorted(rec.row_idx for eid, rec in self._entities.items() if eid not in drop_vs)
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        new_rows = len(keep_idx)

        # 4) Rebuild matrix rows once (batch dict op avoids per-entry Python __setitem__ overhead)
        M_old = self._matrix  # DOK
        _rows, cols = M_old.shape
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        new_data = {(old_to_new[r], c): v for (r, c), v in M_old.items() if r in old_to_new}
        if new_data:
            dict.update(M_new, new_data)
        self._matrix = M_new
        self._csr_cache = None

        # 5) Rebuild entity mappings
        new_entities = {}
        new_row_to_entity = {}
        for new_i, old_i in enumerate(keep_idx):
            ent = self._row_to_entity[old_i]
            old_rec = self._entities[ent]
            new_entities[ent] = EntityRecord(row_idx=new_i, kind=old_rec.kind)
            new_row_to_entity[new_i] = ent
        # remove dropped entities
        for vid in drop_vs:
            pass  # they're not in new_entities, no action needed
        self._entities = new_entities
        self._row_to_entity = new_row_to_entity
        self._rebuild_entity_indexes()

        # 6) Clean vertex attributes and slice memberships
        drop_vertex_ids = {
            ent[0] if isinstance(ent, tuple) and len(ent) == 2 else ent for ent in drop_vs
        }
        va = self.vertex_attributes
        if va is not None and hasattr(va, "columns") and "vertex_id" in va.columns:
            if pl is not None and isinstance(va, pl.DataFrame) and va.height:
                self.vertex_attributes = va.filter(
                    ~pl.col("vertex_id").is_in(list(drop_vertex_ids))
                )
            else:
                import narwhals as nw

                self.vertex_attributes = nw.to_native(
                    nw.from_native(va).filter(~nw.col("vertex_id").is_in(list(drop_vertex_ids)))
                )

        for slice_data in self._slices.values():
            slice_data["vertices"].difference_update(drop_vertex_ids)

        self._V.difference_update(drop_vertex_ids)
        self._VM.difference_update(drop_vs)
