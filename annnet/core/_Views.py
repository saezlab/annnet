import narwhals as nw

try:
    import polars as pl
except Exception:
    pl = None
import scipy.sparse as sp


class GraphView:
    """Lazy view into a graph with deferred operations.

    Provides filtered access to graph components without copying the underlying data.
    Views can be materialized into concrete subgraphs when needed.

    Parameters
    --
    graph : AnnNet
        Parent graph instance
    vertices : list[str] | set[str] | callable | None
        vertex IDs to include, or predicate function
    edges : list[str] | set[str] | callable | None
        Edge IDs to include, or predicate function
    slices : str | list[str] | None
        slice ID(s) to include
    predicate : callable | None
        Additional filter: predicate(vertex_id) -> bool

    """

    def __init__(self, graph, vertices=None, edges=None, slices=None, predicate=None):
        self._graph = graph
        self._vertices_filter = vertices
        self._edges_filter = edges
        self._predicate = predicate

        # Normalize slices to list
        if slices is None:
            self._slices = None
        elif isinstance(slices, str):
            self._slices = [slices]
        else:
            self._slices = list(slices)

        # Lazy caches
        self._vertex_ids_cache = None
        self._edge_ids_cache = None
        self._computed = False

    # ==================== Properties ====================

    @property
    def obs(self):
        """Filtered vertex attribute table (uses AnnNet.vertex_attributes)."""
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return self._graph.vertex_attributes

        df = self._graph.vertex_attributes
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            return df.filter(pl.col("vertex_id").is_in(list(vertex_ids)))

        import narwhals as nw

        return nw.to_native(nw.from_native(df).filter(nw.col("vertex_id").is_in(list(vertex_ids))))

    @property
    def var(self):
        """Filtered edge attribute table (uses AnnNet.edge_attributes)."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return self._graph.edge_attributes

        df = self._graph.edge_attributes
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            return df.filter(pl.col("edge_id").is_in(list(edge_ids)))

        import narwhals as nw

        return nw.to_native(nw.from_native(df).filter(nw.col("edge_id").is_in(list(edge_ids))))

    @property
    def X(self):
        """Filtered incidence matrix subview."""
        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Get row and column indices
        if vertex_ids is not None:
            rows = [
                self._graph.entity_to_idx[nid]
                for nid in vertex_ids
                if nid in self._graph.entity_to_idx
            ]
        else:
            rows = list(range(self._graph._matrix.shape[0]))

        if edge_ids is not None:
            cols = [
                self._graph.edge_to_idx[eid] for eid in edge_ids if eid in self._graph.edge_to_idx
            ]
        else:
            cols = list(range(self._graph._matrix.shape[1]))

        # Return submatrix slice
        if rows and cols:
            return self._graph._matrix[rows, :][:, cols]
        else:
            return sp.dok_matrix((len(rows), len(cols)), dtype=self._graph._matrix.dtype)

    @property
    def vertex_ids(self):
        """Get filtered vertex IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._vertex_ids_cache

    @property
    def edge_ids(self):
        """Get filtered edge IDs (cached)."""
        if not self._computed:
            self._compute_ids()
        return self._edge_ids_cache

    @property
    def vertex_count(self):
        """Number of vertices in this view."""
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return sum(1 for t in self._graph.entity_types.values() if t == "vertex")
        return len(vertex_ids)

    @property
    def edge_count(self):
        """Number of edges in this view."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return len(self._graph.edge_to_idx)
        return len(edge_ids)

    # ==================== Internal Computation ====================

    def _compute_ids(self):
        """Compute and cache filtered vertex and edge IDs."""
        vertex_ids = None
        edge_ids = None

        # Step 1: Apply slice filter (uses AnnNet._slices)
        if self._slices is not None:
            vertex_ids = set()
            edge_ids = set()
            for slice_id in self._slices:
                if slice_id in self._graph._slices:
                    vertex_ids.update(self._graph._slices[slice_id]["vertices"])
                    edge_ids.update(self._graph._slices[slice_id]["edges"])

        # Step 2: Apply vertex filter
        if self._vertices_filter is not None:
            candidate_vertices = (
                vertex_ids
                if vertex_ids is not None
                else set(
                    vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
                )
            )

            if callable(self._vertices_filter):
                filtered_vertices = set()
                for vid in candidate_vertices:
                    try:
                        if self._vertices_filter(vid):
                            filtered_vertices.add(vid)
                    except Exception:
                        pass
                vertex_ids = filtered_vertices
            else:
                specified = set(self._vertices_filter)
                if vertex_ids is not None:
                    vertex_ids &= specified
                else:
                    vertex_ids = specified & candidate_vertices

        # Step 3: Apply edge filter
        if self._edges_filter is not None:
            candidate_edges = (
                edge_ids if edge_ids is not None else set(self._graph.edge_to_idx.keys())
            )

            if callable(self._edges_filter):
                filtered_edges = set()
                for eid in candidate_edges:
                    try:
                        if self._edges_filter(eid):
                            filtered_edges.add(eid)
                    except Exception:
                        pass
                edge_ids = filtered_edges
            else:
                specified = set(self._edges_filter)
                if edge_ids is not None:
                    edge_ids &= specified
                else:
                    edge_ids = specified & candidate_edges

        # Step 4: Apply additional predicate to vertices
        if self._predicate is not None and vertex_ids is not None:
            filtered_vertices = set()
            for vid in vertex_ids:
                try:
                    if self._predicate(vid):
                        filtered_vertices.add(vid)
                except Exception:
                    pass
            vertex_ids = filtered_vertices

        # Step 5: Filter edges by vertex connectivity (uses AnnNet.edge_definitions, hyperedge_definitions)
        if vertex_ids is not None and edge_ids is not None:
            filtered_edges = set()
            for eid in edge_ids:
                # Binary/vertex-edge edges
                if eid in self._graph.edge_definitions:
                    source, target, _ = self._graph.edge_definitions[eid]
                    if source in vertex_ids and target in vertex_ids:
                        filtered_edges.add(eid)
                # Hyperedges
                elif eid in self._graph.hyperedge_definitions:
                    hdef = self._graph.hyperedge_definitions[eid]
                    if hdef.get("directed", False):
                        heads = set(hdef.get("head", []))
                        tails = set(hdef.get("tail", []))
                        if heads.issubset(vertex_ids) and tails.issubset(vertex_ids):
                            filtered_edges.add(eid)
                    else:
                        members = set(hdef.get("members", []))
                        if members.issubset(vertex_ids):
                            filtered_edges.add(eid)
            edge_ids = filtered_edges

        # Cache results
        self._vertex_ids_cache = vertex_ids
        self._edge_ids_cache = edge_ids
        self._computed = True

    # ==================== View Methods (use AnnNet's existing methods) ====================

    def edges_df(self, **kwargs):
        """Get edge DataFrame view with optional filtering.
        Uses AnnNet.edges_view() and filters by edge IDs.
        """
        # Use AnnNet's existing edges_view() method
        df = self._graph.edges_view(**kwargs)

        # Filter by edge IDs in this view
        edge_ids = self.edge_ids
        if edge_ids is not None:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(df, pl.DataFrame):
                df = df.filter(pl.col("edge_id").is_in(list(edge_ids)))
            else:
                import narwhals as nw

                df = nw.to_native(
                    nw.from_native(df).filter(nw.col("edge_id").is_in(list(edge_ids)))
                )

        return df

    def vertices_df(self, **kwargs):
        """Get vertex DataFrame view.
        Uses AnnNet.vertices_view() and filters by vertex IDs.
        """
        # Use AnnNet's existing vertices_view() method
        df = self._graph.vertices_view(**kwargs)

        # Filter by vertex IDs in this view
        vertex_ids = self.vertex_ids
        if vertex_ids is not None:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(df, pl.DataFrame):
                df = df.filter(pl.col("vertex_id").is_in(list(vertex_ids)))
            else:
                import narwhals as nw

                df = nw.to_native(
                    nw.from_native(df).filter(nw.col("vertex_id").is_in(list(vertex_ids)))
                )
        return df

    # ==================== Materialization (uses AnnNet methods) ====================

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.
        Uses AnnNet.add_vertex(), add_edge(), add_hyperedge(), get_*_attrs()
        """
        # Create new AnnNet instance
        from .graph import AnnNet

        subG = AnnNet(directed=self._graph.directed)

        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Determine which vertices to copy
        if vertex_ids is not None:
            vertices_to_copy = vertex_ids
        else:
            vertices_to_copy = [
                vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
            ]

        # Copy vertices (uses AnnNet.add_vertex, get_vertex_attrs)
        for vid in vertices_to_copy:
            if copy_attributes:
                attrs = self._graph.get_vertex_attrs(vid)
                # drop structural keys
                attrs = {k: v for k, v in attrs.items() if k not in self._graph._vertex_RESERVED}
                subG.add_vertex(vid, **attrs)
            else:
                subG.add_vertex(vid)

        # Determine which edges to copy
        if edge_ids is not None:
            edges_to_copy = edge_ids
        else:
            edges_to_copy = self._graph.edge_to_idx.keys()

        # Copy edges (uses AnnNet methods)
        for eid in edges_to_copy:
            # Binary edges
            if eid in self._graph.edge_definitions:
                source, target, edge_type = self._graph.edge_definitions[eid]

                if source not in vertices_to_copy or target not in vertices_to_copy:
                    continue

                weight = self._graph.edge_weights.get(eid, 1.0)
                directed = self._graph.edge_directed.get(eid, self._graph.directed)

                if copy_attributes:
                    attrs = self._graph.get_edge_attrs(eid)
                    subG.add_edge(source, target, weight=weight, directed=directed, **attrs)
                else:
                    subG.add_edge(source, target, weight=weight, directed=directed)

            # Hyperedges
            elif eid in self._graph.hyperedge_definitions:
                hdef = self._graph.hyperedge_definitions[eid]

                if hdef.get("directed", False):
                    heads = list(hdef.get("head", []))
                    tails = list(hdef.get("tail", []))

                    if not all(h in vertices_to_copy for h in heads):
                        continue
                    if not all(t in vertices_to_copy for t in tails):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(head=heads, tail=tails, weight=weight)
                else:
                    members = list(hdef.get("members", []))

                    if not all(m in vertices_to_copy for m in members):
                        continue

                    weight = self._graph.edge_weights.get(eid, 1.0)
                    if copy_attributes:
                        attrs = self._graph.get_edge_attrs(eid)
                        subG.add_hyperedge(members=members, weight=weight, **attrs)
                    else:
                        subG.add_hyperedge(members=members, weight=weight)

        return subG

    def subview(self, vertices=None, edges=None, slices=None, predicate=None):
        """Create a new GraphView by further restricting this view.

        - vertices/edges: if a list/set is given, intersect with this view's vertex_ids/edge_ids.
        - slices: defaults to this view's slices if None.
        - predicate: applied in addition to the current filtering (AND).
        """
        # Force compute current filters
        base_vertices = self.vertex_ids
        base_edges = self.edge_ids

        # vertices
        if vertices is None:
            new_vertices = base_vertices
            vertex_pred = None
        elif callable(vertices):
            new_vertices = base_vertices  # keep current set; apply new predicate below
            vertex_pred = vertices
        else:
            to_set = set(vertices)
            new_vertices = (set(base_vertices) & to_set) if base_vertices is not None else to_set
            vertex_pred = None

        # Edges
        if edges is None:
            new_edges = base_edges
            edge_pred = None
        elif callable(edges):
            new_edges = base_edges
            edge_pred = edges
        else:
            to_set = set(edges)
            new_edges = (set(base_edges) & to_set) if base_edges is not None else to_set
            edge_pred = None

        # slices
        new_slices = slices if slices is not None else (self._slices if self._slices else None)

        # Combine predicates (AND) with existing one
        def combined_pred(v):
            ok = True
            if self._predicate:
                try:
                    ok = ok and bool(self._predicate(v))
                except Exception:
                    ok = False
            if predicate:
                try:
                    ok = ok and bool(predicate(v))
                except Exception:
                    ok = False
            if vertex_pred:
                try:
                    ok = ok and bool(vertex_pred(v))
                except Exception:
                    ok = False
            return ok

        final_pred = combined_pred if (self._predicate or predicate or vertex_pred) else None

        # Return a fresh GraphView
        return GraphView(
            self._graph,
            vertices=new_vertices,
            edges=new_edges,
            slices=new_slices,
            predicate=final_pred,
        )

    # ==================== Convenience ====================

    def summary(self):
        """Human-readable summary."""
        lines = [
            "GraphView Summary",
            "─" * 30,
            f"vertices: {self.vertex_count}",
            f"Edges: {self.edge_count}",
        ]

        filters = []
        if self._slices:
            filters.append(f"slices={self._slices}")
        if self._vertices_filter:
            if callable(self._vertices_filter):
                filters.append("vertices=<predicate>")
            else:
                filters.append(f"vertices={len(list(self._vertices_filter))} specified")
        if self._edges_filter:
            if callable(self._edges_filter):
                filters.append("edges=<predicate>")
            else:
                filters.append(f"edges={len(list(self._edges_filter))} specified")
        if self._predicate:
            filters.append("predicate=<function>")

        if filters:
            lines.append(f"Filters: {', '.join(filters)}")
        else:
            lines.append("Filters: None (full graph)")

        return "\n".join(lines)

    def __repr__(self):
        return f"GraphView(vertices={self.vertex_count}, edges={self.edge_count})"

    def __len__(self):
        return self.vertex_count


class ViewsClass:
    # Materialized views

    def edges_view(
        self,
        slice=None,
        include_directed=True,
        include_weight=True,
        resolved_weight=True,
        copy=True,
    ):
        """Build a Polars DF [DataFrame] view of edges with optional slice join.
        Same columns/semantics as before, but vectorized (no per-edge DF scans).
        """
        if not self.edge_to_idx:
            try:
                import polars as pl
                return pl.DataFrame(schema={"edge_id": pl.Utf8, "kind": pl.Utf8})
            except Exception:
                import pandas as pd
                return pd.DataFrame(
                    {"edge_id": pd.Series(dtype="string"), "kind": pd.Series(dtype="string")}
                )

        # Keep original keys for dict lookups
        eids_raw = list(self.edge_to_idx.keys())
        eids_str = [str(eid) for eid in eids_raw]  # Stringified for DataFrame
        
        kinds = [self.edge_kind.get(eid, "binary") for eid in eids_raw]

        need_global = include_weight or resolved_weight
        global_w = [self.edge_weights.get(eid, None) for eid in eids_raw] if need_global else None
        dirs = (
            [self.edge_directed.get(eid, True if self.directed is None else self.directed)
            for eid in eids_raw]
            if include_directed else None
        )

        src, tgt, etype = [], [], []
        head, tail, members = [], [], []
        
        for eid_raw, k in zip(eids_raw, kinds):
            if k == "hyper":
                h = self.hyperedge_definitions[eid_raw]
                if h.get("directed", False):
                    head.append(tuple(str(x) for x in sorted(h.get("head", ()))))
                    tail.append(tuple(str(x) for x in sorted(h.get("tail", ()))))
                    members.append(None)
                else:
                    head.append(None)
                    tail.append(None)
                    members.append(tuple(str(x) for x in sorted(h.get("members", ()))))
                src.append(None)
                tgt.append(None)
                etype.append(None)
            else:
                s, t, et = self.edge_definitions[eid_raw]
                src.append(str(s) if s is not None else None)
                tgt.append(str(t) if t is not None else None)
                etype.append(str(et) if et is not None else None)
                head.append(None)
                tail.append(None)
                members.append(None)

        # Use stringified IDs in DataFrame
        cols = {"edge_id": eids_str, "kind": kinds}
        if include_directed:
            cols["directed"] = dirs
        if include_weight:
            cols["global_weight"] = global_w
        if resolved_weight and not include_weight:
            cols["_gw_tmp"] = global_w

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None:
            base = pl.DataFrame(cols).with_columns(
                pl.Series("source", src, dtype=pl.Utf8),
                pl.Series("target", tgt, dtype=pl.Utf8),
                pl.Series("edge_type", etype, dtype=pl.Utf8),
                pl.Series("head", head, dtype=pl.List(pl.Utf8)),
                pl.Series("tail", tail, dtype=pl.List(pl.Utf8)),
                pl.Series("members", members, dtype=pl.List(pl.Utf8)),
            )

            # Normalize edge_attributes before join
            if isinstance(self.edge_attributes, pl.DataFrame) and self.edge_attributes.height > 0:
                edge_attrs = self.edge_attributes
                if "edge_id" in edge_attrs.columns:
                    edge_attrs = edge_attrs.with_columns(pl.col("edge_id").cast(pl.Utf8))
                out = base.join(edge_attrs, on="edge_id", how="left")
            else:
                out = base

            if (slice is not None and isinstance(self.edge_slice_attributes, pl.DataFrame) 
                and self.edge_slice_attributes.height > 0):
                slice_df = self.edge_slice_attributes
                if "edge_id" in slice_df.columns:
                    slice_df = slice_df.with_columns(pl.col("edge_id").cast(pl.Utf8))
                slice_slice = slice_df.filter(pl.col("slice_id") == slice).drop("slice_id")
                if slice_slice.height > 0:
                    rename_map = {c: f"slice_{c}" for c in slice_slice.columns if c != "edge_id"}
                    if rename_map:
                        slice_slice = slice_slice.rename(rename_map)
                    out = out.join(slice_slice, on="edge_id", how="left")

            if resolved_weight:
                gw_col = "global_weight" if include_weight else "_gw_tmp"
                lw_col = "slice_weight" if ("slice_weight" in out.columns) else None
                if lw_col:
                    out = out.with_columns(
                        pl.coalesce([pl.col(lw_col), pl.col(gw_col)]).alias("effective_weight")
                    )
                else:
                    out = out.with_columns(pl.col(gw_col).alias("effective_weight"))

                if not include_weight and "_gw_tmp" in out.columns:
                    out = out.drop("_gw_tmp")

            return out.clone() if copy else out

        # pandas fallback
        import pandas as pd
        base = pd.DataFrame(cols)
        base["source"] = src
        base["target"] = tgt
        base["edge_type"] = etype
        base["head"] = head
        base["tail"] = tail
        base["members"] = members
        out = base

        ea = self.edge_attributes
        if ea is not None and hasattr(ea, "columns") and len(ea) > 0 and "edge_id" in ea.columns:
            ea_df = pd.DataFrame(ea)
            ea_df["edge_id"] = ea_df["edge_id"].astype(str)
            out = out.merge(ea_df, on="edge_id", how="left")

        if slice is not None:
            esa = self.edge_slice_attributes
            if esa is not None and hasattr(esa, "columns") and len(esa) > 0:
                esa_df = pd.DataFrame(esa)
                if {"slice_id", "edge_id"}.issubset(esa_df.columns):
                    esa_df["edge_id"] = esa_df["edge_id"].astype(str)
                    slice_slice = esa_df[esa_df["slice_id"] == slice].drop(
                        columns=["slice_id"], errors="ignore"
                    )
                    if not slice_slice.empty:
                        rename_map = {c: f"slice_{c}" for c in slice_slice.columns if c != "edge_id"}
                        slice_slice = slice_slice.rename(columns=rename_map)
                        out = out.merge(slice_slice, on="edge_id", how="left")

        if resolved_weight:
            gw_col = "global_weight" if include_weight else "_gw_tmp"
            if "slice_weight" in out.columns:
                out["effective_weight"] = out["slice_weight"].where(
                    out["slice_weight"].notna(), out[gw_col]
                )
            else:
                out["effective_weight"] = out[gw_col]
            if not include_weight and "_gw_tmp" in out.columns:
                out = out.drop(columns=["_gw_tmp"], errors="ignore")

        return out.copy(deep=True) if copy else out

    def vertices_view(self, copy=True):
        """Read-only vertex attribute table.

        Parameters
        --
        copy : bool, optional
            Return a cloned DF.

        Returns
        ---
        polars.DataFrame
            Columns: ``vertex_id`` plus pure attributes (may be empty).

        """
        df = self.vertex_attributes
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return pl.DataFrame(schema={"vertex_id": pl.Utf8})
            return df.clone() if copy else df

        # fallback
        import pandas as pd

        if df is None or (hasattr(df, "__len__") and len(df) == 0):
            out = pd.DataFrame({"vertex_id": pd.Series(dtype="string")})
        else:
            out = pd.DataFrame(df)
        return out.copy(deep=True) if copy else out

    def slices_view(self, copy=True):
        """Read-only slice attribute table.

        Parameters
        --
        copy : bool, optional
            Return a cloned DF.

        Returns
        ---
        polars.DataFrame
            Columns: ``slice_id`` plus pure attributes (may be empty).

        """
        df = self.slice_attributes
        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return pl.DataFrame(schema={"slice_id": pl.Utf8})
            return df.clone() if copy else df

        import pandas as pd

        if df is None or (hasattr(df, "__len__") and len(df) == 0):
            out = pd.DataFrame({"slice_id": pd.Series(dtype="string")})
        else:
            out = pd.DataFrame(df)
        return out.copy(deep=True) if copy else out

    def aspects_view(self, copy=True):
        """
        View of Kivela aspects and their metadata.

        Columns:
        aspect : str
        elem_layers : list[str]
        <aspect_attr_keys>...
        """
        if not getattr(self, "aspects", None):
            try:
                import polars as pl

                return pl.DataFrame(schema={"aspect": pl.Utf8, "elem_layers": pl.List(pl.Utf8)})
            except Exception:
                import pandas as pd

                return pd.DataFrame(
                    {"aspect": pd.Series(dtype="string"), "elem_layers": pd.Series(dtype="object")}
                )

        rows = []
        for a in self.aspects:
            base = {
                "aspect": a,
                "elem_layers": list(self.elem_layers.get(a, [])),
            }
            # aspect attrs stored in self._aspect_attrs[a]
            for k, v in self._aspect_attrs.get(a, {}).items():
                base[k] = v
            rows.append(base)

        try:
            import polars as pl

            df = pl.DataFrame(rows)
            return df.clone() if copy else df
        except Exception:
            import pandas as pd

            df = pd.DataFrame.from_records(rows)
            return df.copy(deep=True) if copy else df

    def layers_view(self, copy=True):
        """
        Read-only table of multi-aspect layers (full Kivelä layers).

        Columns:
        layer_tuple : list[str]   # the aspect tuple
        layer_id    : str         # canonical id
        <aspect columns>          # one column per aspect
        <layer attrs>...          # metadata set by set_layer_attrs()
        <elem attrs>...           # merged elementary layer attrs (prefixed)

        For a single-aspect model:
        layer_tuple = [label]
        layer_id    = label
        aspect column = that label
        """
        # no aspects configured → no layers
        if not getattr(self, "aspects", None):
            try:
                import polars as pl

                return pl.DataFrame(schema={"layer_tuple": pl.List(pl.Utf8), "layer_id": pl.Utf8})
            except Exception:
                import pandas as pd

                return pd.DataFrame(
                    {
                        "layer_tuple": pd.Series(dtype="object"),
                        "layer_id": pd.Series(dtype="string"),
                    }
                )

        # empty product → no layers
        if not getattr(self, "_all_layers", ()):
            try:
                import polars as pl

                return pl.DataFrame(
                    schema={
                        "layer_tuple": pl.List(pl.Utf8),
                        "layer_id": pl.Utf8,
                    }
                )
            except Exception:
                import pandas as pd

                return pd.DataFrame(
                    {
                        "layer_tuple": pd.Series(dtype="object"),
                        "layer_id": pd.Series(dtype="string"),
                    }
                )

        rows = []
        for aa in self._all_layers:
            aa = tuple(aa)
            lid = self.layer_tuple_to_id(aa)

            base = {
                "layer_tuple": list(aa),
                "layer_id": lid,
            }

            # split per-aspect columns
            for i, a in enumerate(self.aspects):
                base[a] = aa[i]

            # attach multi-aspect layer metadata
            for k, v in self._layer_attrs.get(aa, {}).items():
                base[k] = v

            # attach elementary layer attrs for each aspect (prefixed)
            # using the canonical elementary id "{aspect}_{label}"
            for i, a in enumerate(self.aspects):
                lid_elem = f"{a}_{aa[i]}"
                rdict = self._row_attrs(self.layer_attributes, "layer_id", lid_elem) or {}
                for k, v in rdict.items():
                    base[f"{a}__{k}"] = v

            rows.append(base)

        try:
            import polars as pl

            df = pl.DataFrame(rows)
            return df.clone() if copy else df
        except Exception:
            import pandas as pd

            df = pd.DataFrame.from_records(rows)
            return df.copy(deep=True) if copy else df
