import polars as pl

class GraphView:
    """Lazy view into a graph with deferred operations.

    Provides filtered access to graph components without copying the underlying data.
    Views can be materialized into concrete subgraphs when needed.

    Parameters
    --
    graph : Graph
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
        """Filtered vertex attribute table (uses Graph.vertex_attributes)."""
        vertex_ids = self.vertex_ids
        if vertex_ids is None:
            return self._graph.vertex_attributes

        

        return self._graph.vertex_attributes.filter(pl.col("vertex_id").is_in(list(vertex_ids)))

    @property
    def var(self):
        """Filtered edge attribute table (uses Graph.edge_attributes)."""
        edge_ids = self.edge_ids
        if edge_ids is None:
            return self._graph.edge_attributes

        

        return self._graph.edge_attributes.filter(pl.col("edge_id").is_in(list(edge_ids)))

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

        # Step 1: Apply slice filter (uses Graph._slices)
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

        # Step 5: Filter edges by vertex connectivity (uses Graph.edge_definitions, hyperedge_definitions)
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

    # ==================== View Methods (use Graph's existing methods) ====================

    def edges_df(self, **kwargs):
        """Get edge DataFrame view with optional filtering.
        Uses Graph.edges_view() and filters by edge IDs.
        """
        # Use Graph's existing edges_view() method
        df = self._graph.edges_view(**kwargs)

        # Filter by edge IDs in this view
        edge_ids = self.edge_ids
        if edge_ids is not None:
            

            df = df.filter(pl.col("edge_id").is_in(list(edge_ids)))

        return df

    def vertices_df(self, **kwargs):
        """Get vertex DataFrame view.
        Uses Graph.vertices_view() and filters by vertex IDs.
        """
        # Use Graph's existing vertices_view() method
        df = self._graph.vertices_view(**kwargs)

        # Filter by vertex IDs in this view
        vertex_ids = self.vertex_ids
        if vertex_ids is not None:
            

            df = df.filter(pl.col("vertex_id").is_in(list(vertex_ids)))

        return df

    # ==================== Materialization (uses Graph methods) ====================

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.
        Uses Graph.add_vertex(), add_edge(), add_hyperedge(), get_*_attrs()
        """
        # Create new Graph instance
        from .graph import Graph
        subG = Graph(directed=self._graph.directed)

        vertex_ids = self.vertex_ids
        edge_ids = self.edge_ids

        # Determine which vertices to copy
        if vertex_ids is not None:
            vertices_to_copy = vertex_ids
        else:
            vertices_to_copy = [
                vid for vid, vtype in self._graph.entity_types.items() if vtype == "vertex"
            ]

        # Copy vertices (uses Graph.add_vertex, get_vertex_attrs)
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

        # Copy edges (uses Graph methods)
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
            self._graph, vertices=new_vertices, edges=new_edges, slices=new_slices, predicate=final_pred
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
