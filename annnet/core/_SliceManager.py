class SliceManager:
    """Manager for graph slice operations.

    Provides organized namespace for slice operations by delegating to Graph methods.
    All heavy lifting is done by the Graph class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Basic Operations (Delegation) ====================

    def add(self, slice_id, **attributes):
        """Create new slice.

        Delegates to Graph.add_slice()
        """
        return self._G.add_slice(slice_id, **attributes)

    def remove(self, slice_id):
        """Remove slice.

        Delegates to Graph.remove_slice()
        """
        return self._G.remove_slice(slice_id)

    def list(self, include_default=False):
        """List slice IDs.

        Delegates to Graph.list_slices()
        """
        return self._G.list_slices(include_default=include_default)

    def exists(self, slice_id):
        """Check if slice exists.

        Delegates to Graph.has_slice()
        """
        return self._G.has_slice(slice_id)

    def info(self, slice_id):
        """Get slice metadata.

        Delegates to Graph.get_slice_info()
        """
        return self._G.get_slice_info(slice_id)

    def count(self):
        """Get number of slices.

        Delegates to Graph.slice_count()
        """
        return self._G.slice_count()

    def vertices(self, slice_id):
        """Get vertices in slice.

        Delegates to Graph.get_slice_vertices()
        """
        return self._G.get_slice_vertices(slice_id)

    def edges(self, slice_id):
        """Get edges in slice.

        Delegates to Graph.get_slice_edges()
        """
        return self._G.get_slice_edges(slice_id)

    # ==================== Active slice Property ====================

    @property
    def active(self):
        """Get active slice ID.

        Delegates to Graph.get_active_slice()
        """
        return self._G.get_active_slice()

    @active.setter
    def active(self, slice_id):
        """Set active slice ID.

        Delegates to Graph.set_active_slice()
        """
        self._G.set_active_slice(slice_id)

    # ==================== Set Operations (Pure Delegation) ====================

    def union(self, slice_ids):
        """Compute union of slices (returns dict, doesn't create slice).

        Delegates to Graph.slice_union()

        Parameters
        --
        slice_ids : list[str]
            slices to union

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.slice_union(slice_ids)

    def intersect(self, slice_ids):
        """Compute intersection of slices (returns dict, doesn't create slice).

        Delegates to Graph.slice_intersection()

        Parameters
        --
        slice_ids : list[str]
            slices to intersect

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}

        """
        return self._G.slice_intersection(slice_ids)

    def difference(self, slice_a, slice_b):
        """Compute set difference (returns dict, doesn't create slice).

        Delegates to Graph.slice_difference()

        Parameters
        --
        slice_a : str
            First slice
        slice_b : str
            Second slice

        Returns
        ---
        dict
            {"vertices": set[str], "edges": set[str]}
            Elements in slice_a but not in slice_b

        """
        return self._G.slice_difference(slice_a, slice_b)

    # ==================== Creation from Operations ====================

    def union_create(self, slice_ids, name, **attributes):
        """Create new slice as union of existing slices.

        Combines Graph.slice_union() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_ids : list[str]
            slices to union
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_union(slice_ids)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def intersect_create(self, slice_ids, name, **attributes):
        """Create new slice as intersection of existing slices.

        Combines Graph.slice_intersection() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_ids : list[str]
            slices to intersect
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_intersection(slice_ids)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def difference_create(self, slice_a, slice_b, name, **attributes):
        """Create new slice as difference of two slices.

        Combines Graph.slice_difference() + Graph.create_slice_from_operation()

        Parameters
        --
        slice_a : str
            First slice
        slice_b : str
            Second slice
        name : str
            New slice name
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        result = self._G.slice_difference(slice_a, slice_b)
        return self._G.create_slice_from_operation(name, result, **attributes)

    def aggregate(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        """Create aggregated slice from multiple sources.

        Delegates to Graph.create_aggregated_slice()

        Parameters
        --
        source_slice_ids : list[str]
            Source slices
        target_slice_id : str
            Target slice name
        method : {'union', 'intersection'}
            Aggregation method
        weight_func : callable, optional
            Weight merging function (reserved)
        **attributes
            slice attributes

        Returns
        ---
        str
            Created slice ID

        """
        return self._G.create_aggregated_slice(
            source_slice_ids, target_slice_id, method, weight_func, **attributes
        )

    # ==================== Analysis & Queries ====================

    def stats(self, include_default=False):
        """Get statistics for all slices.

        Delegates to Graph.slice_statistics()

        Returns
        ---
        dict[str, dict]
            {slice_id: {'vertices': int, 'edges': int, 'attributes': dict}}

        """
        return self._G.slice_statistics(include_default=include_default)

    def vertex_presence(self, vertex_id, include_default=False):
        """Find slices containing a vertex.

        Delegates to Graph.vertex_presence_across_slices()

        Parameters
        --
        vertex_id : str
            Vertex to search for
        include_default : bool
            Include default slice

        Returns
        ---
        list[str]
            slice IDs containing the vertex

        """
        return self._G.vertex_presence_across_slices(vertex_id, include_default)

    def edge_presence(
        self, edge_id=None, source=None, target=None, include_default=False, undirected_match=None
    ):
        """Find slices containing an edge.

        Delegates to Graph.edge_presence_across_slices()

        Parameters
        --
        edge_id : str, optional
            Edge ID to search for
        source : str, optional
            Source vertex (with target)
        target : str, optional
            Target vertex (with source)
        include_default : bool
            Include default slice
        undirected_match : bool, optional
            Allow symmetric matches

        Returns
        ---
        list[str] or dict[str, list[str]]
            If edge_id: list of slice IDs
            If source/target: {slice_id: [edge_ids]}

        """
        return self._G.edge_presence_across_slices(
            edge_id,
            source,
            target,
            include_default=include_default,
            undirected_match=undirected_match,
        )

    def hyperedge_presence(self, members=None, head=None, tail=None, include_default=False):
        """Find slices containing a hyperedge.

        Delegates to Graph.hyperedge_presence_across_slices()

        Parameters
        --
        members : Iterable[str], optional
            Undirected hyperedge members
        head : Iterable[str], optional
            Directed hyperedge head
        tail : Iterable[str], optional
            Directed hyperedge tail
        include_default : bool
            Include default slice

        Returns
        ---
        dict[str, list[str]]
            {slice_id: [edge_ids]}

        """
        return self._G.hyperedge_presence_across_slices(
            members=members, head=head, tail=tail, include_default=include_default
        )

    def conserved_edges(self, min_slices=2, include_default=False):
        """Find edges present in multiple slices.

        Delegates to Graph.conserved_edges()

        Parameters
        --
        min_slices : int
            Minimum number of slices
        include_default : bool
            Include default slice

        Returns
        ---
        dict[str, int]
            {edge_id: slice_count}

        """
        return self._G.conserved_edges(min_slices, include_default)

    def specific_edges(self, slice_id):
        """Find edges unique to a slice.

        Delegates to Graph.slice_specific_edges()

        Parameters
        --
        slice_id : str
            slice to check

        Returns
        ---
        set[str]
            Edge IDs unique to this slice

        """
        return self._G.slice_specific_edges(slice_id)

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        """Analyze temporal changes across slices.

        Delegates to Graph.temporal_dynamics()

        Parameters
        --
        ordered_slices : list[str]
            slices in chronological order
        metric : {'edge_change', 'vertex_change'}
            What to track

        Returns
        ---
        list[dict]
            Per-step changes: [{'added': int, 'removed': int, 'net_change': int}]

        """
        return self._G.temporal_dynamics(ordered_slices, metric)

    # ==================== Convenience Methods ====================

    def summary(self):
        """Get human-readable summary of all slices.

        Returns
        ---
        str
            Formatted summary

        """
        stats = self.stats(include_default=True)
        lines = [f"slices: {len(stats)}"]

        for i, (slice_id, info) in enumerate(stats.items()):
            prefix = "├─" if i < len(stats) - 1 else "└─"
            lines.append(f"{prefix} {slice_id}: {info['vertices']} vertices, {info['edges']} edges")

        return "\n".join(lines)

    def __repr__(self):
        return f"SliceManager({self.count()} slices)"
