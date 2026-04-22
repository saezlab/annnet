import scipy.sparse as sp

from .._dataframe_backend import (
    clone_dataframe,
    empty_dataframe,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_in,
    dataframe_from_rows,
)


class GraphView:
    """Lazy view into a graph with deferred operations.

    Provides filtered access to graph components without copying the underlying data.
    Views can be materialized into concrete subgraphs when needed.

    Parameters
    ----------
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
        """Return the filtered vertex attribute table for this view.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.vertex_attributes` and filters by the view's vertex IDs.
        """
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return self._graph.vertex_attributes

        df = self._graph.vertex_attributes
        return dataframe_filter_in(df, 'vertex_id', vertex_ids)

    @property
    def var(self):
        """Return the filtered edge attribute table for this view.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.edge_attributes` and filters by the view's edge IDs.
        """
        edge_ids = self.edge_ids
        if edge_ids is None:
            return self._graph.edge_attributes

        df = self._graph.edge_attributes
        return dataframe_filter_in(df, 'edge_id', edge_ids)

    @property
    def X(self):
        """Return the filtered incidence matrix subview.

        Returns
        -------
        scipy.sparse.dok_matrix
        """
        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Get row and column indices
        if vertex_ids is not None:
            rows = []
            for nid in vertex_ids:
                ekey = self._graph._resolve_entity_key(nid)
                rec = self._graph._entities.get(ekey)
                if rec is not None:
                    rows.append(rec.row_idx)
        else:
            rows = list(range(self._graph._matrix.shape[0]))

        if edge_ids is not None:
            cols = []
            for eid in edge_ids:
                rec = self._graph._edges.get(eid)
                if rec is not None and rec.col_idx >= 0:
                    cols.append(rec.col_idx)
        else:
            cols = list(range(self._graph._matrix.shape[1]))

        # Return submatrix slice
        if rows and cols:
            return self._graph._matrix[rows, :][:, cols]
        else:
            return sp.dok_matrix((len(rows), len(cols)), dtype=self._graph._matrix.dtype)

    @property
    def vertex_ids(self):
        """Get filtered vertex IDs (cached).

        Returns
        -------
        set[str] | None
            None means no vertex filter (full graph).
        """
        if not self._computed:
            self._compute_ids()
        return self._vertex_ids_cache

    @property
    def edge_ids(self):
        """Get filtered edge IDs (cached).

        Returns
        -------
        set[str] | None
            None means no edge filter (full graph).
        """
        if not self._computed:
            self._compute_ids()
        return self._edge_ids_cache

    @property
    def vertex_count(self):
        """Return the number of vertices in this view.

        Returns
        -------
        int
        """
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return sum(1 for rec in self._graph._entities.values() if rec.kind == 'vertex')
        return len(vertex_ids)

    @property
    def edge_count(self):
        """Return the number of edges in this view.

        Returns
        -------
        int
        """
        edge_ids = self.edge_ids
        if edge_ids is None:
            return len(self._graph._col_to_edge)
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
                    vertex_ids.update(self._graph._slices[slice_id]['vertices'])
                    edge_ids.update(self._graph._slices[slice_id]['edges'])

        # Step 2: Apply vertex filter
        if self._vertices_filter is not None:
            candidate_vertices = (
                vertex_ids
                if vertex_ids is not None
                else {
                    ekey[0] for ekey, rec in self._graph._entities.items() if rec.kind == 'vertex'
                }
            )

            if callable(self._vertices_filter):
                filtered_vertices = set()
                for vid in candidate_vertices:
                    try:
                        if self._vertices_filter(vid):
                            filtered_vertices.add(vid)
                    except Exception:  # noqa: BLE001
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
                edge_ids if edge_ids is not None else set(self._graph._col_to_edge.values())
            )

            if callable(self._edges_filter):
                filtered_edges = set()
                for eid in candidate_edges:
                    try:
                        if self._edges_filter(eid):
                            filtered_edges.add(eid)
                    except Exception:  # noqa: BLE001
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
                except Exception:  # noqa: BLE001
                    pass
            vertex_ids = filtered_vertices

        # Step 5: Filter edges by vertex connectivity
        if vertex_ids is not None and edge_ids is not None:
            filtered_edges = set()
            for eid in edge_ids:
                rec = self._graph._edges.get(eid)
                if rec is None or rec.col_idx < 0:
                    continue
                if rec.etype == 'hyper':
                    if rec.tgt is not None:
                        if set(rec.src).issubset(vertex_ids) and set(rec.tgt).issubset(vertex_ids):
                            filtered_edges.add(eid)
                    else:
                        if set(rec.src).issubset(vertex_ids):
                            filtered_edges.add(eid)
                else:
                    s, t = rec.src, rec.tgt
                    if s is not None and t is not None and s in vertex_ids and t in vertex_ids:
                        filtered_edges.add(eid)
            edge_ids = filtered_edges

        # Cache results
        self._vertex_ids_cache = vertex_ids
        self._edge_ids_cache = edge_ids
        self._computed = True

    # ==================== View Methods (use AnnNet's existing methods) ====================

    def edges_df(self, **kwargs):
        """Return an edge DataFrame view filtered to this view's edges.

        Parameters
        ----------
        **kwargs
            Passed through to `AnnNet.edges_view()`.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.edges_view()` and then filters by the view's edge IDs.
        """
        # Use AnnNet's existing edges_view() method
        df = self._graph.edges_view(**kwargs)

        # Filter by edge IDs in this view
        edge_ids = self.edge_ids
        if edge_ids is not None:
            df = dataframe_filter_in(df, 'edge_id', edge_ids)

        return df

    def vertices_df(self, **kwargs):
        """Return a vertex DataFrame view filtered to this view's vertices.

        Parameters
        ----------
        **kwargs
            Passed through to `AnnNet.vertices_view()`.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.vertices_view()` and then filters by the view's vertex IDs.
        """
        # Use AnnNet's existing vertices_view() method
        df = self._graph.vertices_view(**kwargs)

        # Filter by vertex IDs in this view
        vertex_ids = self.vertex_ids
        if vertex_ids is not None:
            df = dataframe_filter_in(df, 'vertex_id', vertex_ids)
        return df

    # ==================== Materialization (uses AnnNet methods) ====================

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.

        Parameters
        ----------
        copy_attributes : bool, optional
            If True, copy vertex/edge attributes into the new graph.

        Returns
        -------
        AnnNet
            Materialized subgraph.
        """
        from .graph import AnnNet

        subG = AnnNet(directed=self._graph.directed)

        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # vset is a set for O(1) membership checks throughout
        if vertex_ids is not None:
            vset = vertex_ids  # already a set from _compute_ids
        else:
            vset = {ekey[0] for ekey, rec in self._graph._entities.items() if rec.kind == 'vertex'}

        # ---- Copy vertices in one bulk call ----
        if copy_attributes:
            # obs is the already-filtered vertex attr DataFrame — one scan replaces N individual lookups
            obs_df = self.obs
            try:
                vertex_records = dataframe_to_rows(obs_df)
            except Exception:  # noqa: BLE001
                vertex_records = [{'vertex_id': vid} for vid in vset]
            subG.add_vertices_bulk(vertex_records)
        else:
            subG.add_vertices_bulk({'vertex_id': vid} for vid in vset)

        # ---- Collect all edge attrs in one bulk scan ----
        edge_attrs_map = {}
        if copy_attributes:
            var_df = self.var  # already-filtered edge attr DataFrame
            try:
                for row in dataframe_to_rows(var_df):
                    eid = row.pop('edge_id', None)
                    if eid is not None:
                        edge_attrs_map[eid] = row
            except Exception:  # noqa: BLE001
                pass

        # ---- Determine which edges to copy ----
        if edge_ids is not None:
            eids_to_copy = edge_ids
        else:
            eids_to_copy = self._graph._col_to_edge.values()

        # ---- Partition into binary and hyperedges ----
        binary_edges = []
        hyper_edges = []

        for eid in eids_to_copy:
            rec = self._graph._edges.get(eid)
            if rec is None or rec.col_idx < 0:
                continue
            weight = rec.weight if rec.weight is not None else 1.0
            if rec.etype == 'hyper':
                if rec.tgt is not None:
                    heads = list(rec.src)
                    tails = list(rec.tgt)
                    if not all(h in vset for h in heads) or not all(t in vset for t in tails):
                        continue
                    d = {'head': heads, 'tail': tails, 'weight': weight}
                else:
                    members = list(rec.src)
                    if not all(m in vset for m in members):
                        continue
                    d = {'members': members, 'weight': weight}
                if copy_attributes and eid in edge_attrs_map:
                    d['attributes'] = edge_attrs_map[eid]
                hyper_edges.append(d)
            else:
                source, target = rec.src, rec.tgt
                if source is None or target is None:
                    continue
                if source not in vset or target not in vset:
                    continue
                directed = rec.directed if rec.directed is not None else self._graph.directed
                d = {
                    'source': source,
                    'target': target,
                    'weight': weight,
                    'edge_type': rec.etype,
                    'edge_directed': directed,
                }
                if copy_attributes and eid in edge_attrs_map:
                    d['attributes'] = edge_attrs_map[eid]
                binary_edges.append(d)

        if binary_edges:
            subG.add_edges_bulk(binary_edges)
        if hyper_edges:
            subG.add_hyperedges_bulk(hyper_edges)

        return subG

    def subview(self, vertices=None, edges=None, slices=None, predicate=None):
        """Create a new GraphView by further restricting this view.

        Parameters
        ----------
        vertices : Iterable[str] | callable | None
            Vertex IDs or predicate; intersects with current view if provided.
        edges : Iterable[str] | callable | None
            Edge IDs or predicate; intersects with current view if provided.
        slices : Iterable[str] | None
            Slice IDs to include. Defaults to current view's slices if None.
        predicate : callable | None
            Additional vertex predicate applied in conjunction with existing filters.

        Returns
        -------
        GraphView

        Notes
        -----
        Predicates are combined with logical AND.
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
        elif callable(edges):
            new_edges = base_edges
        else:
            to_set = set(edges)
            new_edges = (set(base_edges) & to_set) if base_edges is not None else to_set

        # slices
        new_slices = slices if slices is not None else (self._slices if self._slices else None)

        # Combine predicates (AND) with existing one
        def combined_pred(v):
            ok = True
            if self._predicate:
                try:
                    ok = ok and bool(self._predicate(v))
                except Exception:  # noqa: BLE001
                    ok = False
            if predicate:
                try:
                    ok = ok and bool(predicate(v))
                except Exception:  # noqa: BLE001
                    ok = False
            if vertex_pred:
                try:
                    ok = ok and bool(vertex_pred(v))
                except Exception:  # noqa: BLE001
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
        """Return a human-readable summary of this view.

        Returns
        -------
        str
        """
        lines = [
            'GraphView Summary',
            '─' * 30,
            f'vertices: {self.vertex_count}',
            f'Edges: {self.edge_count}',
        ]

        filters = []
        if self._slices:
            filters.append(f'slices={self._slices}')
        if self._vertices_filter:
            if callable(self._vertices_filter):
                filters.append('vertices=<predicate>')
            else:
                filters.append(f'vertices={len(list(self._vertices_filter))} specified')
        if self._edges_filter:
            if callable(self._edges_filter):
                filters.append('edges=<predicate>')
            else:
                filters.append(f'edges={len(list(self._edges_filter))} specified')
        if self._predicate:
            filters.append('predicate=<function>')

        if filters:
            lines.append(f'Filters: {", ".join(filters)}')
        else:
            lines.append('Filters: None (full graph)')

        return '\n'.join(lines)

    def __repr__(self):
        return f'GraphView(vertices={self.vertex_count}, edges={self.edge_count})'

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
        """Build a DataFrame view of edges with optional slice join.

        Parameters
        ----------
        slice : str, optional
            Slice ID to join per-slice attributes.
        include_directed : bool, optional
            Include directedness column.
        include_weight : bool, optional
            Include global weight column.
        resolved_weight : bool, optional
            Include effective weight (slice override if present).
        copy : bool, optional
            Return a cloned DataFrame if True.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Vectorized implementation avoids per-edge scans.
        """
        if not self._col_to_edge:
            return empty_dataframe({'edge_id': 'text', 'kind': 'text'}, backend=None)

        eids_raw = list(self._col_to_edge.values())
        eids_str = [str(eid) for eid in eids_raw]  # Stringified for DataFrame

        _default_dir = True if self.directed is None else self.directed
        _edge_recs = [self._edges[eid] for eid in eids_raw]
        kinds = [
            'hyper' if rec.etype == 'hyper' else (rec.ml_kind or 'binary') for rec in _edge_recs
        ]

        need_global = include_weight or resolved_weight
        global_w = [rec.weight for rec in _edge_recs] if need_global else None
        dirs = (
            [rec.directed if rec.directed is not None else _default_dir for rec in _edge_recs]
            if include_directed
            else None
        )

        src, tgt, etype = [], [], []
        head, tail, members = [], [], []

        for rec in _edge_recs:
            if rec.etype == 'hyper':
                if rec.tgt is not None:
                    src_vals = tuple(str(x) for x in sorted(rec.src))
                    tgt_vals = tuple(str(x) for x in sorted(rec.tgt))
                    head.append(src_vals)
                    tail.append(tgt_vals)
                    members.append(None)
                    src.append('|'.join(src_vals))
                    tgt.append('|'.join(tgt_vals))
                else:
                    src_vals = tuple(str(x) for x in sorted(rec.src))
                    head.append(None)
                    tail.append(None)
                    members.append(src_vals)
                    src.append('|'.join(src_vals))
                    tgt.append(None)
                etype.append(None)
            else:
                src.append(str(rec.src) if rec.src is not None else None)
                tgt.append(str(rec.tgt) if rec.tgt is not None else None)
                etype.append(str(rec.etype) if rec.etype is not None else None)
                head.append(None)
                tail.append(None)
                members.append(None)

        # Use stringified IDs in DataFrame
        cols = {'edge_id': eids_str, 'kind': kinds}
        if include_directed:
            cols['directed'] = dirs
        if include_weight:
            cols['global_weight'] = global_w
        if resolved_weight and not include_weight:
            cols['_gw_tmp'] = global_w

        rows = []
        for i, _eid in enumerate(eids_str):
            row = {name: values[i] for name, values in cols.items()}
            row.update(
                {
                    'source': src[i],
                    'target': tgt[i],
                    'edge_type': etype[i],
                    'head': head[i],
                    'tail': tail[i],
                    'members': members[i],
                }
            )
            rows.append(row)

        edge_attr_rows = {}
        if self.edge_attributes is not None and 'edge_id' in dataframe_columns(
            self.edge_attributes
        ):
            edge_attr_rows = {
                str(row.get('edge_id')): row
                for row in dataframe_to_rows(self.edge_attributes)
                if row.get('edge_id') is not None
            }
        if edge_attr_rows:
            for row in rows:
                row.update(edge_attr_rows.get(row['edge_id'], {}))

        if slice is not None and self.edge_slice_attributes is not None:
            slice_attr_rows = {}
            for attr_row in dataframe_to_rows(self.edge_slice_attributes):
                if attr_row.get('slice_id') != slice or attr_row.get('edge_id') is None:
                    continue
                slice_attr_rows[str(attr_row['edge_id'])] = {
                    f'slice_{key}': value
                    for key, value in attr_row.items()
                    if key not in {'slice_id', 'edge_id'}
                }
            for row in rows:
                row.update(slice_attr_rows.get(row['edge_id'], {}))

        if resolved_weight:
            gw_col = 'global_weight' if include_weight else '_gw_tmp'
            for row in rows:
                row['effective_weight'] = row.get('slice_weight', row.get(gw_col))
                if row['effective_weight'] is None:
                    row['effective_weight'] = row.get(gw_col)
                if not include_weight:
                    row.pop('_gw_tmp', None)

        out = dataframe_from_rows(rows, backend=getattr(self, '_annotations_backend', None))
        return clone_dataframe(out) if copy else out

    def vertices_view(self, copy=True):
        """Read-only vertex attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like
            Columns include `vertex_id` plus pure attributes.
        """
        df = self.vertex_attributes
        if df is None:
            return empty_dataframe({'vertex_id': 'text'}, backend=None)
        return clone_dataframe(df) if copy else df

    def slices_view(self, copy=True):
        """Read-only slice attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like
            Columns include `slice_id` plus pure attributes.
        """
        df = self.slice_attributes
        if df is None:
            return empty_dataframe({'slice_id': 'text'}, backend=None)
        return clone_dataframe(df) if copy else df

    def aspects_view(self, copy=True):
        """Return a view of Kivela aspects and their metadata.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Columns include `aspect`, `elem_layers`, and any aspect attribute keys.
        """
        if not getattr(self, 'aspects', None):
            return empty_dataframe({'aspect': 'text', 'elem_layers': 'list_text'}, backend=None)

        rows = []
        for a in self.aspects:
            base = {
                'aspect': a,
                'elem_layers': list(self.elem_layers.get(a, [])),
            }
            # aspect attrs stored in self._aspect_attrs[a]
            for k, v in self._aspect_attrs.get(a, {}).items():
                base[k] = v
            rows.append(base)

        df = dataframe_from_rows(rows, backend=None)
        return clone_dataframe(df) if copy else df

    def layers_view(self, copy=True):
        """Return a read-only table of multi-aspect layers.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Columns include `layer_tuple`, `layer_id`, aspect columns, layer attributes,
        and prefixed elementary layer attributes.
        """
        # no aspects configured → no layers
        if not getattr(self, 'aspects', None):
            return empty_dataframe(
                {'layer_tuple': 'list_text', 'layer_id': 'text'},
                backend=None,
            )

        # empty product → no layers
        if not getattr(self, '_all_layers', ()):
            return empty_dataframe(
                {'layer_tuple': 'list_text', 'layer_id': 'text'},
                backend=None,
            )

        rows = []
        for aa in self._all_layers:
            aa = tuple(aa)
            lid = self.layer_tuple_to_id(aa)

            base = {
                'layer_tuple': list(aa),
                'layer_id': lid,
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
                lid_elem = f'{a}_{aa[i]}'
                rdict = self._row_attrs(self.layer_attributes, 'layer_id', lid_elem) or {}
                for k, v in rdict.items():
                    base[f'{a}__{k}'] = v

            rows.append(base)

        df = dataframe_from_rows(rows, backend=None)
        return clone_dataframe(df) if copy else df
