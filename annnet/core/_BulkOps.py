import polars as pl
import scipy.sparse as sp

class BulkOps():
    # Bulk build graph

    def add_vertices_bulk(self, vertices, slice=None):
        """Bulk add vertices (and edge-entities if prefixed externally)."""

        slice = slice or self._current_slice

        # -----------------------------------------------------------------------
        # NORMALIZE INPUT
        # -----------------------------------------------------------------------
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
                (_sys.intern(vid) if isinstance(vid, str) else vid, attrs)
                for vid, attrs in norm
            ]
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            pass

        # -----------------------------------------------------------------------
        # ENTITY REGISTRATION (fast, unchanged semantics)
        # -----------------------------------------------------------------------
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

        # -----------------------------------------------------------------------
        # SLICE MEMBERSHIP (unchanged behaviour)
        # -----------------------------------------------------------------------
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(v for v, _ in norm)

        # -----------------------------------------------------------------------
        # ATTRIBUTE TABLE PREP
        # -----------------------------------------------------------------------
        self._ensure_vertex_table()
        df = self.vertex_attributes

        # Build lookup of existing vertex_ids once
        try:
            if df.height > 0:
                existing_ids = set(df.get_column("vertex_id").to_list())
            else:
                existing_ids = set()
        except Exception:
            existing_ids = set()

        # -----------------------------------------------------------------------
        # Build new rows (for vertex_ids missing in DF)
        # -----------------------------------------------------------------------
        new_rows_data = []
        new_attr_keys = set()

        for vid, attrs in norm:
            if vid not in existing_ids:
                row = {c: None for c in df.columns} if df.columns else {"vertex_id": None}
                row["vertex_id"] = vid
                for k, v in attrs.items():
                    row[k] = v
                    new_attr_keys.add(k)
                new_rows_data.append(row)
            else:
                for k in attrs.keys():
                    new_attr_keys.add(k)

        # -----------------------------------------------------------------------
        # Ensure DF has columns for all attributes used
        # -----------------------------------------------------------------------
        if new_attr_keys:
            df = self._ensure_attr_columns(df, dict.fromkeys(new_attr_keys))

        # -----------------------------------------------------------------------
        # Vectorized insert of new rows
        # -----------------------------------------------------------------------
        if new_rows_data:
            add_df = pl.DataFrame(new_rows_data, nan_to_null=True, strict=False)

            # Ensure every column exists on add_df (exact order)
            for c in df.columns:
                if c not in add_df.columns:
                    add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))

            # Resolve dtype mismatches using same rules
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

        # -----------------------------------------------------------------------
        # VECTORIZED UPSERT OF EXISTING ATTRIBUTES
        # -----------------------------------------------------------------------
        update_pairs = [(vid, attrs) for vid, attrs in norm if vid in existing_ids and attrs]

        if update_pairs:
            # Build a DataFrame of updates
            update_df = pl.DataFrame(
                {
                    "vertex_id": [vid for vid, _ in update_pairs],
                    **{
                        k: [attrs.get(k, None) for _, attrs in update_pairs]
                        for k in new_attr_keys
                    }
                },
                nan_to_null=True,
                strict=False,
            )

            # Resolve dtype mismatches with df (vectorized)
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

            # LEFT JOIN updates, COALESCE new values
            df = df.join(update_df, on="vertex_id", how="left", suffix="_new")
            for c in new_attr_keys:
                if c in df.columns and c + "_new" in df.columns:
                    df = df.with_columns(
                        pl.coalesce([pl.col(c + "_new"), pl.col(c)]).alias(c)
                    ).drop(c + "_new")

        # -----------------------------------------------------------------------
        # DONE
        # -----------------------------------------------------------------------
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
        """Bulk add/update *binary* (and vertex-edge) edges.
        Accepts each item as:
        - (src, tgt)
        - (src, tgt, weight)
        - dict with keys: source, target, [weight, edge_id, edge_type, propagate, slice_weight, edge_directed, attributes]
        Behavior: identical to calling add_edge() per item (same propagation/slice/attrs), but grows columns once and avoids full-column wipes.
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
                    self._add_edge_entity(s)
                else:
                    # bare global insert (no slice side-effects; membership handled later)
                    idx = self._num_entities
                    self.entity_to_idx[s] = idx
                    self.idx_to_entity[idx] = s
                    self.entity_types[s] = "vertex"
                    self._num_entities = idx + 1
            if t not in entity_to_idx:
                if et == "vertex_edge" and isinstance(t, str) and t.startswith("edge_"):
                    self._add_edge_entity(t)
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
        """Bulk add/update hyperedges.
        Each item can be:
        - {'members': [...], 'edge_id': ..., 'weight': ..., 'slice': ..., 'attributes': {...}}
        - {'head': [...], 'tail': [...], ...}
        Behavior: identical to calling add_hyperedge() per item, but grows columns once and avoids full-column wipes.
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

        # Ensure participants exist (global)
        for d in items:
            if "members" in d and d["members"] is not None:
                for u in d["members"]:
                    if u not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[u] = idx
                        self.idx_to_entity[idx] = u
                        self.entity_types[u] = "vertex"
                        self._num_entities = idx + 1
            else:
                for u in d.get("head", []):
                    if u not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[u] = idx
                        self.idx_to_entity[idx] = u
                        self.entity_types[u] = "vertex"
                        self._num_entities = idx + 1
                for v in d.get("tail", []):
                    if v not in self.entity_to_idx:
                        idx = self._num_entities
                        self.entity_to_idx[v] = idx
                        self.entity_types[v] = "vertex"
                        self.idx_to_entity[idx] = v
                        self._num_entities = idx + 1

        # Grow rows once
        self._grow_rows_to(self._num_entities)

        # Pre-size columns
        new_count = sum(1 for d in items if d.get("edge_id") not in self.edge_to_idx)
        if new_count:
            self._grow_cols_to(self._num_edges + new_count)

        M = self._matrix
        out_ids = []

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

            if e_id in self.edge_to_idx:
                col = self.edge_to_idx[e_id]
                # clear old cells (binary or hyper)
                if e_id in self.hyperedge_definitions:
                    h = self.hyperedge_definitions[e_id]
                    if h.get("members"):
                        rows = h["members"]
                    else:
                        rows = set(h.get("head", ())) | set(h.get("tail", ()))
                    for vid in rows:
                        try:
                            M[self.entity_to_idx[vid], col] = 0
                        except Exception:
                            pass
                else:
                    old = self.edge_definitions.get(e_id)
                    if old is not None:
                        os, ot, _ = old
                        try:
                            M[self.entity_to_idx[os], col] = 0
                        except Exception:
                            pass
                        if ot is not None and ot != os:
                            try:
                                M[self.entity_to_idx[ot], col] = 0
                            except Exception:
                                pass
            else:
                col = self._num_edges
                self.edge_to_idx[e_id] = col
                self.idx_to_edge[col] = e_id
                self._num_edges = col + 1

            # write new column values + metadata
            if members is not None:
                for u in members:
                    M[self.entity_to_idx[u], col] = w
                self.hyperedge_definitions[e_id] = {"directed": False, "members": set(members)}
                self.edge_directed[e_id] = False
                self.edge_kind[e_id] = "hyper"
                self.edge_definitions[e_id] = (None, None, "hyper")
            else:
                for u in head:
                    M[self.entity_to_idx[u], col] = w
                for v in tail:
                    M[self.entity_to_idx[v], col] = -w
                self.hyperedge_definitions[e_id] = {
                    "directed": True,
                    "head": set(head),
                    "tail": set(tail),
                }
                self.edge_directed[e_id] = True
                self.edge_kind[e_id] = "hyper"
                self.edge_definitions[e_id] = (None, None, "hyper")

            self.edge_weights[e_id] = w

            # slice membership
            if slice_local is not None:
                if slice_local not in self._slices:
                    self._slices[slice_local] = {
                        "vertices": set(),
                        "edges": set(),
                        "attributes": {},
                    }
                self._slices[slice_local]["edges"].add(e_id)
                if members is not None:
                    self._slices[slice_local]["vertices"].update(members)
                else:
                    self._slices[slice_local]["vertices"].update(head)
                    self._slices[slice_local]["vertices"].update(tail)

            # per-edge attributes (optional)
            attrs = d.get("attributes") or d.get("attrs") or {}
            if attrs:
                self.set_edge_attrs(e_id, **attrs)

            out_ids.append(e_id)

        return out_ids

    def add_edges_to_slice_bulk(self, slice_id, edge_ids):
        """Bulk version of add_edge_to_slice: add many edges to a slice and attach
        all incident vertices. No weights are changed here.
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
        """Bulk add edge-entities (vertex-edge hybrids). Accepts:
        - iterable of str IDs
        - iterable of (edge_entity_id, attrs_dict)
        - iterable of dicts with key 'edge_entity_id' (or 'id')
        Behavior: identical to calling add_edge_entity() for each, but grows rows once
        and batches attribute inserts.
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

        if new_rows:
            self._grow_rows_to(self._num_entities)

        # slice membership
        if slice not in self._slices:
            self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._slices[slice]["vertices"].update(eid for eid, _ in norm)

        # attributes (edge-entities share vertex_attributes table)
        self._ensure_vertex_table()
        df = self.vertex_attributes
        to_append, existing_ids = [], set()
        try:
            if df.height and "vertex_id" in df.columns:
                existing_ids = set(df.get_column("vertex_id").to_list())
        except Exception:
            pass

        for eid, attrs in norm:
            if df.is_empty() or eid not in existing_ids:
                row = dict.fromkeys(df.columns) if not df.is_empty() else {"vertex_id": None}
                row["vertex_id"] = eid
                for k, v in attrs.items():
                    row[k] = v
                to_append.append(row)

        if to_append:
            need_cols = {k for r in to_append for k in r if k != "vertex_id"}
            if need_cols:
                df = self._ensure_attr_columns(df, dict.fromkeys(need_cols))
            add_df = pl.DataFrame(to_append)
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
                if to_append:
                    need_cols = {k for r in to_append for k in r if k != "vertex_id"}
                    if need_cols:
                        df = self._ensure_attr_columns(df, dict.fromkeys(need_cols))

                    add_df = pl.DataFrame(to_append)

                    # ensure all df columns exist on add_df
                    for c in df.columns:
                        if c not in add_df.columns:
                            add_df = add_df.with_columns(pl.lit(None).cast(df.schema[c]).alias(c))

                    # dtype reconciliation (same as before)
                    for c in df.columns:
                        lc, rc = df.schema[c], add_df.schema[c]
                        if lc == pl.Null and rc != pl.Null:
                            df = df.with_columns(pl.col(c).cast(rc))
                        elif rc == pl.Null and lc != pl.Null:
                            add_df = add_df.with_columns(pl.col(c).cast(lc).alias(c))
                        elif lc != rc:
                            df = df.with_columns(pl.col(c).cast(pl.Utf8))
                            add_df = add_df.with_columns(pl.col(c).cast(pl.Utf8).alias(c))

                    # reorder add_df columns to match df exactly
                    add_df = add_df.select(df.columns)

                    df = df.vstack(add_df)

        for eid, attrs in norm:
            if attrs and (df.is_empty() or (eid in existing_ids)):
                df = self._upsert_row(df, eid, attrs)
        self.vertex_attributes = df

    def set_vertex_key(self, *fields: str):
        """Declare composite key fields (order matters). Rebuilds the uniqueness index.

        - Raises ValueError if duplicates exist among already-populated vertices.
        - Vertices missing some key fields are skipped during indexing.
        """
        if not fields:
            raise ValueError("set_vertex_key requires at least one field")
        self._vertex_key_fields = tuple(str(f) for f in fields)
        self._vertex_key_index.clear()

        df = self.vertex_attributes
        if not isinstance(df, pl.DataFrame) or df.height == 0:
            return  # nothing to index yet

        missing = [f for f in self._vertex_key_fields if f not in df.columns]
        if missing:
            # ok to skip; those rows simply won't be indexable until fields appear
            pass

        # Rebuild index, enforcing uniqueness only for fully-populated tuples
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
        except Exception:
            # Fallback if iter_rows misbehaves
            for vid in df.get_column("vertex_id").to_list():
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
        """Remove many edges in one pass (much faster than looping)."""
        to_drop = [eid for eid in edge_ids if eid in self.edge_to_idx]
        if not to_drop:
            return
        self._remove_edges_bulk(to_drop)

    def remove_vertices(self, vertex_ids):
        """Remove many vertices (and all their incident edges) in one pass."""
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
        if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height:
            if "edge_id" in self.edge_attributes.columns:
                self.edge_attributes = self.edge_attributes.filter(
                    ~pl.col("edge_id").is_in(list(drop))
                )
        if (
            isinstance(self.edge_slice_attributes, pl.DataFrame)
            and self.edge_slice_attributes.height
        ):
            cols = set(self.edge_slice_attributes.columns)
            if {"edge_id"}.issubset(cols):
                self.edge_slice_attributes = self.edge_slice_attributes.filter(
                    ~pl.col("edge_id").is_in(list(drop))
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
        if isinstance(self.vertex_attributes, pl.DataFrame) and self.vertex_attributes.height:
            if "vertex_id" in self.vertex_attributes.columns:
                self.vertex_attributes = self.vertex_attributes.filter(
                    ~pl.col("vertex_id").is_in(list(drop_vs))
                )
        for slice_data in self._slices.values():
            slice_data["vertices"].difference_update(drop_vs)
