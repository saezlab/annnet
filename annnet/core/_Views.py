"""Lazy graph views and materialized table builders."""

import scipy.sparse as sp

from .._support.dataframe_backend import (
    clone_dataframe,
    empty_dataframe,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_in,
    dataframe_from_rows,
)


class GraphView:
    """Lazy, filtered view into a graph; materialize() for a concrete subgraph."""

    def __init__(self, graph, vertices=None, edges=None, slices=None, predicate=None):
        self._graph = graph
        self._vertices_filter = vertices
        self._edges_filter = edges
        self._predicate = predicate
        if slices is None:
            self._slices = None
        elif isinstance(slices, str):
            self._slices = [slices]
        else:
            self._slices = list(slices)
        self._vertex_ids_cache = None
        self._edge_ids_cache = None
        self._computed = False

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
        return dataframe_filter_in(self._graph.vertex_attributes, 'vertex_id', vertex_ids)

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
        return dataframe_filter_in(self._graph.edge_attributes, 'edge_id', edge_ids)

    @property
    def X(self):
        """Return the filtered incidence matrix subview.

        Returns
        -------
        scipy.sparse.dok_matrix
        """
        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids
        if vertex_ids is not None:
            rows = []
            for nid in vertex_ids:
                rec = self._graph._entities.get(self._graph._resolve_entity_key(nid))
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
        if rows and cols:
            return self._graph._matrix[rows, :][:, cols]
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

    def _compute_ids(self):
        vertex_ids = None
        edge_ids = None

        if self._slices is not None:
            vertex_ids = set()
            edge_ids = set()
            for slice_id in self._slices:
                if slice_id in self._graph._slices:
                    vertex_ids.update(self._graph._slices[slice_id]['vertices'])
                    edge_ids.update(self._graph._slices[slice_id]['edges'])

        if self._vertices_filter is not None:
            candidate_vertices = (
                vertex_ids
                if vertex_ids is not None
                else {
                    ekey[0] for ekey, rec in self._graph._entities.items() if rec.kind == 'vertex'
                }
            )
            if callable(self._vertices_filter):
                filtered = set()
                for vid in candidate_vertices:
                    try:
                        if self._vertices_filter(vid):
                            filtered.add(vid)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                vertex_ids = filtered
            else:
                specified = set(self._vertices_filter)
                vertex_ids = (
                    (vertex_ids & specified)
                    if vertex_ids is not None
                    else (specified & candidate_vertices)
                )

        if self._edges_filter is not None:
            candidate_edges = (
                edge_ids if edge_ids is not None else set(self._graph._col_to_edge.values())
            )
            if callable(self._edges_filter):
                filtered = set()
                for eid in candidate_edges:
                    try:
                        if self._edges_filter(eid):
                            filtered.add(eid)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                edge_ids = filtered
            else:
                specified = set(self._edges_filter)
                edge_ids = (
                    (edge_ids & specified)
                    if edge_ids is not None
                    else (specified & candidate_edges)
                )

        if self._predicate is not None and vertex_ids is not None:
            filtered = set()
            for vid in vertex_ids:
                try:
                    if self._predicate(vid):
                        filtered.add(vid)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            vertex_ids = filtered

        if vertex_ids is not None and edge_ids is not None:
            filtered = set()
            for eid in edge_ids:
                rec = self._graph._edges.get(eid)
                if rec is None or rec.col_idx < 0:
                    continue
                if rec.etype == 'hyper':
                    if rec.tgt is not None:
                        if set(rec.src).issubset(vertex_ids) and set(rec.tgt).issubset(vertex_ids):
                            filtered.add(eid)
                    elif set(rec.src).issubset(vertex_ids):
                        filtered.add(eid)
                else:
                    s, t = rec.src, rec.tgt
                    if s is not None and t is not None and s in vertex_ids and t in vertex_ids:
                        filtered.add(eid)
            edge_ids = filtered

        self._vertex_ids_cache = vertex_ids
        self._edge_ids_cache = edge_ids
        self._computed = True

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
        df = self._graph.views.edges(**kwargs)
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
        df = self._graph.views.vertices(**kwargs)
        vertex_ids = self.vertex_ids
        if vertex_ids is not None:
            df = dataframe_filter_in(df, 'vertex_id', vertex_ids)
        return df

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
        subG = self._graph.ops.extract_subgraph(vertices=self.vertex_ids, edges=self.edge_ids)
        if copy_attributes:
            return subG

        def _id_only_table(df, id_col: str):
            rows = []
            if df is not None:
                for row in dataframe_to_rows(df):
                    key = row.get(id_col)
                    if key is not None:
                        rows.append({id_col: key})
            if rows:
                return dataframe_from_rows(rows)
            return empty_dataframe({id_col: 'text'}, backend=self._graph._annotations_backend)

        subG.vertex_attributes = _id_only_table(subG.vertex_attributes, 'vertex_id')
        subG.edge_attributes = _id_only_table(subG.edge_attributes, 'edge_id')
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
        base_vertices = self.vertex_ids
        base_edges = self.edge_ids

        if vertices is None:
            new_vertices, vertex_pred = base_vertices, None
        elif callable(vertices):
            new_vertices, vertex_pred = base_vertices, vertices
        else:
            to_set = set(vertices)
            new_vertices = (set(base_vertices) & to_set) if base_vertices is not None else to_set
            vertex_pred = None

        if edges is None or callable(edges):
            new_edges = base_edges
        else:
            to_set = set(edges)
            new_edges = (set(base_edges) & to_set) if base_edges is not None else to_set

        new_slices = slices if slices is not None else (self._slices if self._slices else None)

        def combined_pred(v):
            ok = True
            for pred in (self._predicate, predicate, vertex_pred):
                if pred:
                    try:
                        ok = ok and bool(pred(v))
                    except (AttributeError, TypeError, ValueError):
                        ok = False
            return ok

        final_pred = combined_pred if (self._predicate or predicate or vertex_pred) else None
        return GraphView(
            self._graph,
            vertices=new_vertices,
            edges=new_edges,
            slices=new_slices,
            predicate=final_pred,
        )

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
            filters.append(
                'vertices=<predicate>'
                if callable(self._vertices_filter)
                else f'vertices={len(list(self._vertices_filter))} specified'
            )
        if self._edges_filter:
            filters.append(
                'edges=<predicate>'
                if callable(self._edges_filter)
                else f'edges={len(list(self._edges_filter))} specified'
            )
        if self._predicate:
            filters.append('predicate=<function>')
        lines.append(f'Filters: {", ".join(filters)}' if filters else 'Filters: None (full graph)')
        return '\n'.join(lines)

    def __repr__(self):
        return f'GraphView(vertices={self.vertex_count}, edges={self.edge_count})'

    def __len__(self):
        return self.vertex_count


class ViewsClass:
    """Materialized table builders mixed into ``AnnNet``."""

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
            return empty_dataframe({'edge_id': 'text', 'kind': 'text', 'ml_kind': 'text'})

        eids_raw = list(self._col_to_edge.values())
        eids_str = [str(eid) for eid in eids_raw]

        _default_dir = True if self.directed is None else self.directed
        _edge_recs = [self._edges[eid] for eid in eids_raw]
        kinds = ['hyper' if rec.etype == 'hyper' else 'binary' for rec in _edge_recs]
        ml_kinds = [rec.ml_kind for rec in _edge_recs]

        need_global = include_weight or resolved_weight
        global_w = [rec.weight for rec in _edge_recs] if need_global else None
        dirs = (
            [rec.directed if rec.directed is not None else _default_dir for rec in _edge_recs]
            if include_directed
            else None
        )

        src, tgt, etype, head, tail, members = [], [], [], [], [], []
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

        edge_attrs_map = self._rows_attr_map(self.edge_attributes, 'edge_id')
        slice_attrs_map = {}
        if slice is not None:
            for row in dataframe_to_rows(self.edge_slice_attributes):
                if row.get('slice_id') != slice:
                    continue
                eid = row.get('edge_id')
                if eid is None:
                    continue
                slice_attrs_map[str(eid)] = {
                    f'slice_{k}': v for k, v in row.items() if k not in {'slice_id', 'edge_id'}
                }

        out_rows = []
        for idx, eid in enumerate(eids_str):
            row = {
                'edge_id': eid,
                'kind': kinds[idx],
                'ml_kind': ml_kinds[idx],
                'source': src[idx],
                'target': tgt[idx],
                'edge_type': etype[idx],
                'head': list(head[idx]) if head[idx] is not None else None,
                'tail': list(tail[idx]) if tail[idx] is not None else None,
                'members': list(members[idx]) if members[idx] is not None else None,
            }
            if include_directed:
                row['directed'] = dirs[idx]
            if include_weight:
                row['global_weight'] = global_w[idx]
            elif resolved_weight:
                row['_gw_tmp'] = global_w[idx]

            row.update(edge_attrs_map.get(eid, {}))
            row.update(slice_attrs_map.get(eid, {}))

            if resolved_weight:
                gw_col = 'global_weight' if include_weight else '_gw_tmp'
                row['effective_weight'] = row.get('slice_weight', row.get(gw_col))
                if not include_weight:
                    row.pop('_gw_tmp', None)

            out_rows.append(row)

        out = dataframe_from_rows(out_rows)
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
        if df is None or 'vertex_id' not in dataframe_columns(df):
            out = empty_dataframe({'vertex_id': 'text'})
        else:
            out = clone_dataframe(df)
        return clone_dataframe(out) if copy else out

    def slices_view(self, copy=True):
        """Read-only slice attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like
            One row per slice (including the default slice), keyed by
            ``slice_id``. User-set slice attributes appear as additional
            columns; slices without user attrs still appear, with null
            cells.
        """
        all_slice_ids = list(self.slices.list(include_default=True))
        attr_df = self.slice_attributes
        attr_rows: dict = {}
        if attr_df is not None and 'slice_id' in dataframe_columns(attr_df):
            for row in dataframe_to_rows(attr_df):
                sid = row.get('slice_id')
                if sid is not None:
                    attr_rows[sid] = {k: v for k, v in row.items() if k != 'slice_id'}
        rows = [{'slice_id': sid, **attr_rows.get(sid, {})} for sid in all_slice_ids]
        out = dataframe_from_rows(rows) if rows else empty_dataframe({'slice_id': 'text'})
        return clone_dataframe(out) if copy else out

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
            return empty_dataframe({'aspect': 'text', 'elem_layers': 'list_text'})
        rows = []
        for a in self.aspects:
            base = {'aspect': a, 'elem_layers': list(self.elem_layers.get(a, []))}
            base.update(self.layers._aspect_attrs.get(a, {}))
            rows.append(base)
        df = dataframe_from_rows(rows)
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
        if not self.aspects or not getattr(self.layers, '_all_layers', ()):
            return empty_dataframe({'layer_tuple': 'list_text', 'layer_id': 'text'})

        elem_attr_rows = {}
        if self.layer_attributes is not None and 'layer_id' in dataframe_columns(
            self.layer_attributes
        ):
            for row in dataframe_to_rows(self.layer_attributes):
                layer_id = row.get('layer_id')
                if layer_id is not None:
                    elem_attr_rows[str(layer_id)] = {
                        k: v for k, v in row.items() if k != 'layer_id'
                    }

        rows = []
        for aa in self.layers._all_layers:
            aa = tuple(aa)
            base = {'layer_tuple': list(aa), 'layer_id': self.layers.layer_tuple_to_id(aa)}
            for i, aspect in enumerate(self.aspects):
                base[aspect] = aa[i]
            base.update(self.layers._layer_attrs.get(aa, {}))
            for i, aspect in enumerate(self.aspects):
                for k, v in elem_attr_rows.get(f'{aspect}_{aa[i]}', {}).items():
                    base[f'{aspect}__{k}'] = v
            rows.append(base)
        df = dataframe_from_rows(rows)
        return clone_dataframe(df) if copy else df


class ViewsAccessor:
    """Namespace for materialized graph tables (``G.views``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def edges(self, *args, **kwargs):
        """Materialize the edge table view."""
        return ViewsClass.edges_view(self._G, *args, **kwargs)

    def vertices(self, *args, **kwargs):
        """Materialize the vertex table view."""
        return ViewsClass.vertices_view(self._G, *args, **kwargs)

    def slices(self, *args, **kwargs):
        """Materialize the slice table view."""
        return ViewsClass.slices_view(self._G, *args, **kwargs)

    def aspects(self, *args, **kwargs):
        """Materialize the aspect table view."""
        return ViewsClass.aspects_view(self._G, *args, **kwargs)

    def layers(self, *args, **kwargs):
        """Materialize the layer table view."""
        return ViewsClass.layers_view(self._G, *args, **kwargs)

    def layers_view(self, copy=True):
        """Materialize the layer table view."""
        return ViewsClass.layers_view(self._G, copy=copy)
