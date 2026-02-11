import json

import narwhals as nw

try:
    import polars as pl
except Exception:
    pl = None
import scipy.sparse as sp

from ._helpers import (
    EdgeType,
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
    # Bulk build graph

    def add_vertices_bulk(self, vertices, slice=None):
        """Bulk add vertices with a Polars fast path when available.

        Parameters
        ----------
        vertices : Iterable[str] | Iterable[tuple[str, dict]] | Iterable[dict]
            Vertices to add. Each item can be:
            - `vertex_id` (str)
            - `(vertex_id, attrs)` tuple
            - dict containing `vertex_id` plus attributes
        slice : str, optional
            Target slice. Defaults to the active slice.

        Returns
        -------
        None

        Notes
        -----
        Falls back to Narwhals-based logic when Polars is unavailable.
        """
        slice = slice or self._current_slice

        # Normalize input

        norm_vids = []
        norm_attrs = []
        for it in vertices:
            if isinstance(it, dict):
                vid = it.get("vertex_id") or it.get("id") or it.get("name")
                if vid is None:
                    continue
                attrs = {k: v for k, v in it.items() if k not in ("vertex_id", "id", "name")}
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

        # Entity registration

        new_rows = 0
        for vid in norm_vids:
            if vid not in self.entity_to_idx:
                idx = self._num_entities
                self.entity_to_idx[vid] = idx
                self.idx_to_entity[idx] = vid
                self.entity_types[vid] = "vertex"
                self._num_entities = idx + 1
                new_rows += 1
        if new_rows:
            self._grow_rows_to(self._num_entities)

        # Slice membership

        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(norm_vids)

        # Ensure attribute table exists

        self._ensure_vertex_table()
        df = self.vertex_attributes
        df, is_pl = _to_polars_if_possible(df)

        # If not Polars, fall back to Narwhals

        if pl is None or not isinstance(df, pl.DataFrame):
            return self.add_vertices_bulk_nw(vertices, slice=slice)

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
                return self.add_vertices_bulk_nw(vertices, slice=slice)

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

    def add_vertices_bulk_nw(self, vertices, slice=None):
        """Bulk add vertices using Narwhals-compatible operations.

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
        This path is slower than the Polars fast path but preserves behavior.
        """

        slice = slice or self._current_slice

        # NORMALIZE INPUT

        norm = []
        for it in vertices:
            if isinstance(it, dict):
                vid = it.get("vertex_id") or it.get("id") or it.get("name")
                if vid is None:
                    continue
                attrs = {k: v for k, v in it.items() if k not in ("vertex_id", "id", "name")}
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

        # ENTITY REGISTRATION
        new_rows = 0
        for vid, _ in norm:
            if vid not in self.entity_to_idx:
                idx = self._num_entities
                self.entity_to_idx[vid] = idx
                self.idx_to_entity[idx] = vid
                self.entity_types[vid] = "vertex"
                self._num_entities = idx + 1
                new_rows += 1

        if new_rows:
            self._grow_rows_to(self._num_entities)

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
        default_weight=1.0,
        default_edge_type="regular",
        default_propagate="none",
        default_slice_weight=None,
        default_edge_directed=None,
    ):
        """Bulk add or update binary edges (and vertex-edge edges).

        Parameters
        ----------
        edges : Iterable
            Each item can be:
            - `(source, target)`
            - `(source, target, weight)`
            - dict with keys `source`, `target`, and optional edge fields.
        slice : str, optional
            Default slice to place edges into.
        default_weight : float, optional
            Default weight for edges missing an explicit weight.
        default_edge_type : str, optional
            Default edge type when not provided.
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
        Behavior mirrors `add_edge()` but grows the matrix once to reduce overhead.
        """
        slice = self._current_slice if slice is None else slice

        # Normalize into dicts
        norm = []
        for it in edges:
            if isinstance(it, dict):
                d = dict(it)
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

        entity_to_idx = self.entity_to_idx
        M = self._matrix
        # 1) Ensure endpoints exist (global); we’ll rely on slice handling below to add membership.
        for d in norm:
            s, t = d["source"], d["target"]
            et = d.get("edge_type", "regular")
            if s not in entity_to_idx:
                # vertex or edge-entity depending on mode?
                if et == "vertex_edge" and isinstance(s, str) and s.startswith("edge_"):
                    self.add_edge_entity(s)
                else:
                    # bare global insert (no slice side-effects; membership handled later)
                    idx = self._num_entities
                    self.entity_to_idx[s] = idx
                    self.idx_to_entity[idx] = s
                    self.entity_types[s] = "vertex"
                    self._num_entities = idx + 1
            if t not in entity_to_idx:
                if et == "vertex_edge" and isinstance(t, str) and t.startswith("edge_"):
                    self.add_edge_entity(t)
                else:
                    idx = self._num_entities
                    self.entity_to_idx[t] = idx
                    self.idx_to_entity[idx] = t
                    self.entity_types[t] = "vertex"
                    self._num_entities = idx + 1

        # Grow rows once if needed
        self._grow_rows_to(self._num_entities)

        # 2) Pre-size columns for new edges
        new_count = sum(1 for d in norm if d.get("edge_id") not in self.edge_to_idx)
        if new_count:
            self._grow_cols_to(self._num_edges + new_count)

        # 3) Create/update columns
        out_ids = []
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
            s_idx = self.entity_to_idx[s]
            t_idx = self.entity_to_idx[t]

            if edge_id is None:
                edge_id = self._get_next_edge_id()

            # update vs create
            if edge_id in self.edge_to_idx:
                col = self.edge_to_idx[edge_id]
                # keep old_type on update (mimic add_edge)
                old_s, old_t, old_type = self.edge_definitions[edge_id]
                # clear only previous cells (no full column wipe)
                try:
                    M[self.entity_to_idx[old_s], col] = 0
                except Exception:
                    pass
                if old_t is not None and old_t != old_s:
                    try:
                        M[self.entity_to_idx[old_t], col] = 0
                    except Exception:
                        pass
                # write new
                M[s_idx, col] = w
                if s != t:
                    M[t_idx, col] = -w if is_dir else w
                self.edge_definitions[edge_id] = (s, t, old_type)
                self.edge_weights[edge_id] = w
                self.edge_directed[edge_id] = is_dir
                # keep attribute side-effect for directedness flag
                self.set_edge_attrs(
                    edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED)
                )
            else:
                col = self._num_edges
                self.edge_to_idx[edge_id] = col
                self.idx_to_edge[col] = edge_id
                self.edge_definitions[edge_id] = (s, t, etype)
                self.edge_weights[edge_id] = w
                self.edge_directed[edge_id] = is_dir
                self._num_edges = col + 1
                # write cells
                M[s_idx, col] = w
                if s != t:
                    M[t_idx, col] = -w if is_dir else w

            # slice membership + optional per-slice weight
            if slice_local is not None:
                if slice_local not in self._slices:
                    self._slices[slice_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._slices[slice_local]["edges"].add(edge_id)
                self._slices[slice_local]["vertices"].update((s, t))
                if slice_w is not None:
                    self.set_edge_slice_attrs(slice_local, edge_id, weight=float(slice_w))
                    self.slice_edge_weights.setdefault(slice_local, {})[edge_id] = float(slice_w)

            # propagation
            if prop == "shared":
                self._propagate_to_shared_slices(edge_id, s, t)
            elif prop == "all":
                self._propagate_to_all_slices(edge_id, s, t)

            # per-edge extra attributes
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                self.set_edge_attrs(edge_id, **attrs)

            out_ids.append(edge_id)

        return out_ids

    def add_hyperedges_bulk(
        self,
        hyperedges,
        *,
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Bulk add or update hyperedges.

        Parameters
        ----------
        hyperedges : Iterable[dict]
            Each item can be:
            - `{'members': [...], 'edge_id': ..., 'weight': ..., 'slice': ..., 'attributes': {...}}`
            - `{'head': [...], 'tail': [...], ...}`
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
        Behavior mirrors `add_hyperedge()` but grows the matrix once to reduce overhead.
        """
        slice = self._current_slice if slice is None else slice

        items = []
        for it in hyperedges:
            if not isinstance(it, dict):
                continue
            d = dict(it)
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

        entity_to_idx = self.entity_to_idx
        entity_types = self.entity_types
        idx_to_entity = self.idx_to_entity
        num_entities = self._num_entities

        # Collect ALL unique vertices first
        all_verts = set()
        for d in items:
            if "members" in d and d["members"] is not None:
                all_verts.update(d["members"])
            else:
                all_verts.update(d.get("head", []))
                all_verts.update(d.get("tail", []))

        # Single pass vertex creation
        for u in all_verts:
            if u not in entity_to_idx:
                entity_to_idx[u] = num_entities
                idx_to_entity[num_entities] = u
                entity_types[u] = "vertex"
                num_entities += 1

        self._num_entities = num_entities
        self._grow_rows_to(num_entities)

        # Pre-size columns
        edge_to_idx = self.edge_to_idx
        new_count = sum(1 for d in items if d.get("edge_id") not in edge_to_idx)
        if new_count:
            self._grow_cols_to(self._num_edges + new_count)

        M = self._matrix
        edge_definitions = self.edge_definitions
        hyperedge_definitions = self.hyperedge_definitions
        edge_weights = self.edge_weights
        edge_directed = self.edge_directed
        edge_kind = self.edge_kind
        idx_to_edge = self.idx_to_edge
        slices = self._slices
        num_edges = self._num_edges

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

            if e_id in edge_to_idx:
                col = edge_to_idx[e_id]
                # clear old cells (binary or hyper)
                if e_id in hyperedge_definitions:
                    h = hyperedge_definitions[e_id]
                    if h.get("members"):
                        rows = h["members"]
                    else:
                        rows = set(h.get("head", ())) | set(h.get("tail", ()))
                    for vid in rows:
                        try:
                            M[entity_to_idx[vid], col] = 0
                        except Exception:
                            pass
                else:
                    old = edge_definitions.get(e_id)
                    if old is not None:
                        os, ot, _ = old
                        try:
                            M[entity_to_idx[os], col] = 0
                        except Exception:
                            pass
                        if ot is not None and ot != os:
                            try:
                                M[entity_to_idx[ot], col] = 0
                            except Exception:
                                pass
            else:
                col = num_edges
                edge_to_idx[e_id] = col
                idx_to_edge[col] = e_id
                num_edges = col + 1

            # write new column values + metadata
            if members is not None:
                for u in members:
                    M[entity_to_idx[u], col] = w
                hyperedge_definitions[e_id] = {"directed": False, "members": set(members)}
                edge_directed[e_id] = False
                edge_kind[e_id] = "hyper"
                edge_definitions[e_id] = (None, None, "hyper")
            else:
                for u in head:
                    M[entity_to_idx[u], col] = w
                for v in tail:
                    M[entity_to_idx[v], col] = -w
                hyperedge_definitions[e_id] = {
                    "directed": True,
                    "head": set(head),
                    "tail": set(tail),
                }
                edge_directed[e_id] = True
                edge_kind[e_id] = "hyper"
                edge_definitions[e_id] = (None, None, "hyper")

            edge_weights[e_id] = w

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

        self._num_edges = num_edges

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

        add_edges = {eid for eid in edge_ids if eid in self.edge_to_idx}
        if not add_edges:
            return

        L["edges"].update(add_edges)

        verts = set()
        for eid in add_edges:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members") is not None:
                    verts.update(h["members"])
                else:
                    verts.update(h.get("head", ()))
                    verts.update(h.get("tail", ()))
            else:
                s, t, _ = self.edge_definitions[eid]
                verts.add(s)
                verts.add(t)

        L["vertices"].update(verts)

    def add_edge_entities_bulk(self, items, slice=None):
        """Bulk add edge-entities (vertex-edge hybrids).

        Parameters
        ----------
        items : Iterable
            Accepts:
            - iterable of string IDs
            - iterable of `(edge_entity_id, attrs)` tuples
            - iterable of dicts with key `edge_entity_id` (or `id`)
        slice : str, optional
            Target slice. Defaults to the active slice.

        Returns
        -------
        None

        Notes
        -----
        Attribute inserts are batched for efficiency.
        """
        slice = slice or self._current_slice

        # normalize -> [(eid, attrs)]
        norm = []
        for it in items:
            if isinstance(it, dict):
                eid = it.get("edge_entity_id") or it.get("id")
                if eid is None:
                    continue
                a = {k: v for k, v in it.items() if k not in ("edge_entity_id", "id")}
                norm.append((eid, a))
            elif isinstance(it, (tuple, list)) and it:
                eid = it[0]
                a = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
                norm.append((eid, a))
            else:
                norm.append((it, {}))
        if not norm:
            return

        # intern hot strings
        try:
            import sys as _sys

            norm = [
                (_sys.intern(eid) if isinstance(eid, str) else eid, attrs) for eid, attrs in norm
            ]
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            pass

        # create missing rows as type 'edge'
        new_rows = 0
        for eid, _ in norm:
            if eid not in self.entity_to_idx:
                idx = self._num_entities
                self.entity_to_idx[eid] = idx
                self.idx_to_entity[idx] = eid
                self.entity_types[eid] = "edge"
                self._num_entities = idx + 1
                new_rows += 1

            if eid not in self.edge_definitions:
                self.edge_definitions[eid] = (None, None, "edge_entity")
                self.edge_weights[eid] = 1.0
                self.edge_directed[eid] = False

        if new_rows:
            self._grow_rows_to(self._num_entities)

        # slice membership
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["edges"].update(eid for eid, _ in norm)

        # attributes go to edge table
        attrs_to_write = {eid: attrs for eid, attrs in norm if attrs}
        if attrs_to_write:
            self.set_edge_attrs_bulk(attrs_to_write)

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
        to_drop = [eid for eid in edge_ids if eid in self.edge_to_idx]
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
        to_drop = [vid for vid in vertex_ids if vid in self.entity_to_idx]
        if not to_drop:
            return
        self._remove_vertices_bulk(to_drop)

    def _remove_edges_bulk(self, edge_ids):
        drop = set(edge_ids)
        if not drop:
            return

        # Columns to keep, old->new remap
        keep_pairs = sorted(
            ((idx, eid) for eid, idx in self.edge_to_idx.items() if eid not in drop)
        )
        old_to_new = {
            old: new for new, (old, _eid) in enumerate(((old, eid) for old, eid in keep_pairs))
        }
        new_cols = len(keep_pairs)

        # Rebuild matrix once
        M_old = self._matrix  # DOK
        rows, _cols = M_old.shape
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if c in old_to_new:
                M_new[r, old_to_new[c]] = v
        self._matrix = M_new

        # Rebuild edge mappings
        self.idx_to_edge.clear()
        self.edge_to_idx.clear()
        for new_idx, (old_idx, eid) in enumerate(keep_pairs):
            self.idx_to_edge[new_idx] = eid
            self.edge_to_idx[eid] = new_idx
        self._num_edges = new_cols

        # Metadata cleanup (vectorized)
        # Dicts
        for eid in drop:
            self.edge_definitions.pop(eid, None)
            self.edge_weights.pop(eid, None)
            self.edge_directed.pop(eid, None)
            self.edge_kind.pop(eid, None)
            self.hyperedge_definitions.pop(eid, None)
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

        # 1) Collect incident edges (binary + hyper)
        drop_es = set()
        for eid, (s, t, _typ) in list(self.edge_definitions.items()):
            if s in drop_vs or t in drop_vs:
                drop_es.add(eid)
        for eid, hdef in list(self.hyperedge_definitions.items()):
            if hdef.get("members"):
                if drop_vs & set(hdef["members"]):
                    drop_es.add(eid)
            else:
                if (drop_vs & set(hdef.get("head", ()))) or (
                    drop_vs & set(hdef.get("tail", ()))
                ):  # directed
                    drop_es.add(eid)

        # 2) Drop all those edges in one pass
        if drop_es:
            self._remove_edges_bulk(drop_es)

        # 3) Build row keep list and old->new map
        keep_idx = []
        for idx in range(self._num_entities):
            ent = self.idx_to_entity[idx]
            if ent not in drop_vs:
                keep_idx.append(idx)
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        new_rows = len(keep_idx)

        # 4) Rebuild matrix rows once
        M_old = self._matrix  # DOK
        _rows, cols = M_old.shape
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if r in old_to_new:
                M_new[old_to_new[r], c] = v
        self._matrix = M_new

        # 5) Rebuild entity mappings
        new_entity_to_idx = {}
        new_idx_to_entity = {}
        for new_i, old_i in enumerate(keep_idx):
            ent = self.idx_to_entity[old_i]
            new_entity_to_idx[ent] = new_i
            new_idx_to_entity[new_i] = ent
        self.entity_to_idx = new_entity_to_idx
        self.idx_to_entity = new_idx_to_entity
        # types: drop removed
        for vid in drop_vs:
            self.entity_types.pop(vid, None)
        self._num_entities = new_rows

        # 6) Clean vertex attributes and slice memberships
        va = self.vertex_attributes
        if va is not None and hasattr(va, "columns") and "vertex_id" in va.columns:
            if pl is not None and isinstance(va, pl.DataFrame) and va.height:
                self.vertex_attributes = va.filter(~pl.col("vertex_id").is_in(list(drop_vs)))
            else:
                import narwhals as nw

                self.vertex_attributes = nw.to_native(
                    nw.from_native(va).filter(~nw.col("vertex_id").is_in(list(drop_vs)))
                )

        for slice_data in self._slices.values():
            slice_data["vertices"].difference_update(drop_vs)
