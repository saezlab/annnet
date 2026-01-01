class SliceManager:
    """Manager for graph slice operations.

    Provides organized namespace for slice operations by delegating to AnnNet methods.
    All heavy lifting is done by the AnnNet class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Basic Operations (Delegation) ====================

    def add(self, slice_id, **attributes):
        """Create new slice.

        Delegates to AnnNet.add_slice()
        """
        return self._G.add_slice(slice_id, **attributes)

    def remove(self, slice_id):
        """Remove slice.

        Delegates to AnnNet.remove_slice()
        """
        return self._G.remove_slice(slice_id)

    def list(self, include_default=False):
        """List slice IDs.

        Delegates to AnnNet.list_slices()
        """
        return self._G.list_slices(include_default=include_default)

    def exists(self, slice_id):
        """Check if slice exists.

        Delegates to AnnNet.has_slice()
        """
        return self._G.has_slice(slice_id)

    def info(self, slice_id):
        """Get slice metadata.

        Delegates to AnnNet.get_slice_info()
        """
        return self._G.get_slice_info(slice_id)

    def count(self):
        """Get number of slices.

        Delegates to AnnNet.slice_count()
        """
        return self._G.slice_count()

    def vertices(self, slice_id):
        """Get vertices in slice.

        Delegates to AnnNet.get_slice_vertices()
        """
        return self._G.get_slice_vertices(slice_id)

    def edges(self, slice_id):
        """Get edges in slice.

        Delegates to AnnNet.get_slice_edges()
        """
        return self._G.get_slice_edges(slice_id)

    # ==================== Active slice Property ====================

    @property
    def active(self):
        """Get active slice ID.

        Delegates to AnnNet.get_active_slice()
        """
        return self._G.get_active_slice()

    @active.setter
    def active(self, slice_id):
        """Set active slice ID.

        Delegates to AnnNet.set_active_slice()
        """
        self._G.set_active_slice(slice_id)

    # ==================== Set Operations (Pure Delegation) ====================

    def union(self, slice_ids):
        """Compute union of slices (returns dict, doesn't create slice).

        Delegates to AnnNet.slice_union()

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

        Delegates to AnnNet.slice_intersection()

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

        Delegates to AnnNet.slice_difference()

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

        Combines AnnNet.slice_union() + AnnNet.create_slice_from_operation()

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

        Combines AnnNet.slice_intersection() + AnnNet.create_slice_from_operation()

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

        Combines AnnNet.slice_difference() + AnnNet.create_slice_from_operation()

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

        Delegates to AnnNet.create_aggregated_slice()

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

        Delegates to AnnNet.slice_statistics()

        Returns
        ---
        dict[str, dict]
            {slice_id: {'vertices': int, 'edges': int, 'attributes': dict}}

        """
        return self._G.slice_statistics(include_default=include_default)

    def vertex_presence(self, vertex_id, include_default=False):
        """Find slices containing a vertex.

        Delegates to AnnNet.vertex_presence_across_slices()

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

        Delegates to AnnNet.edge_presence_across_slices()

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

        Delegates to AnnNet.hyperedge_presence_across_slices()

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

        Delegates to AnnNet.conserved_edges()

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

        Delegates to AnnNet.slice_specific_edges()

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

        Delegates to AnnNet.temporal_dynamics()

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


class SliceClass:
    # slice basics

    def add_slice(self, slice_id, **attributes):
        """Create a new empty slice.

        Parameters
        --
        slice_id : str
            New slice identifier (ID).
        **attributes
            Pure slice attributes to store (non-structural).

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            If the slice already exists.

        """
        if slice_id in self._slices and slice_id != "default":
            raise ValueError(f"slice {slice_id} already exists")

        self._slices[slice_id] = {"vertices": set(), "edges": set(), "attributes": attributes}
        # Persist slice metadata to DF (pure attributes, upsert)
        if attributes:
            self.set_slice_attrs(slice_id, **attributes)
        # slice_id as an elementary slice of that aspect
        if len(self.aspects) == 1:
            a = self.aspects[0]
            if a in self.elem_layers:
                if slice_id not in self.elem_layers[a]:
                    self.elem_layers[a].append(slice_id)
        return slice_id

    def set_active_slice(self, slice_id):
        """Set the active slice for subsequent operations.

        Parameters
        --
        slice_id : str
            Existing slice ID.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        self._current_slice = slice_id

    def get_active_slice(self):
        """Get the currently active slice ID.

        Returns
        ---
        str
            Active slice ID.

        """
        return self._current_slice

    def get_slices_dict(self, include_default: bool = False):
        """Get a mapping of slice IDs to their metadata.

        Parameters
        --
        include_default : bool, optional
            Include the internal ``'default'`` slice if True.

        Returns
        ---
        dict[str, dict]
            ``{slice_id: {"vertices": set, "edges": set, "attributes": dict}}``.

        """
        if include_default:
            return self._slices
        return {k: v for k, v in self._slices.items() if k != self._default_slice}

    def list_slices(self, include_default: bool = False):
        """List slice IDs.

        Parameters
        --
        include_default : bool, optional
            Include the internal ``'default'`` slice if True.

        Returns
        ---
        list[str]
            slice IDs.

        """
        return list(self.get_slices_dict(include_default=include_default).keys())

    def has_slice(self, slice_id):
        """Check whether a slice exists.

        Parameters
        --
        slice_id : str

        Returns
        ---
        bool

        """
        return slice_id in self._slices

    def slice_count(self):
        """Get the number of slices (including the internal default).

        Returns
        ---
        int

        """
        return len(self._slices)

    def get_slice_info(self, slice_id):
        """Get a slice's metadata snapshot.

        Parameters
        --
        slice_id : str

        Returns
        ---
        dict
            Copy of ``{"vertices": set, "edges": set, "attributes": dict}``.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")
        return self._slices[slice_id].copy()

    # slice set-ops & cross-slice analytics

    def get_slice_vertices(self, slice_id):
        """Vertices in a slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        """
        return self._slices[slice_id]["vertices"].copy()

    def get_slice_edges(self, slice_id):
        """Edges in a slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        """
        return self._slices[slice_id]["edges"].copy()

    def slice_union(self, slice_ids):
        """Union of multiple slices.

        Parameters
        --
        slice_ids : Iterable[str]

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not slice_ids:
            return {"vertices": set(), "edges": set()}

        union_vertices = set()
        union_edges = set()

        for slice_id in slice_ids:
            if slice_id in self._slices:
                union_vertices.update(self._slices[slice_id]["vertices"])
                union_edges.update(self._slices[slice_id]["edges"])

        return {"vertices": union_vertices, "edges": union_edges}

    def slice_intersection(self, slice_ids):
        """Intersection of multiple slices.

        Parameters
        --
        slice_ids : Iterable[str]

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        """
        if not slice_ids:
            return {"vertices": set(), "edges": set()}

        if len(slice_ids) == 1:
            slice_id = slice_ids[0]
            return {
                "vertices": self._slices[slice_id]["vertices"].copy(),
                "edges": self._slices[slice_id]["edges"].copy(),
            }

        # Start with first slice
        common_vertices = self._slices[slice_ids[0]]["vertices"].copy()
        common_edges = self._slices[slice_ids[0]]["edges"].copy()

        # Intersect with remaining slices
        for slice_id in slice_ids[1:]:
            if slice_id in self._slices:
                common_vertices &= self._slices[slice_id]["vertices"]
                common_edges &= self._slices[slice_id]["edges"]
            else:
                # slice doesn't exist, intersection is empty
                return {"vertices": set(), "edges": set()}

        return {"vertices": common_vertices, "edges": common_edges}

    def slice_difference(self, slice1_id, slice2_id):
        """Set difference: elements in ``slice1_id`` not in ``slice2_id``.

        Parameters
        --
        slice1_id : str
        slice2_id : str

        Returns
        ---
        dict
            ``{"vertices": set[str], "edges": set[str]}``

        Raises
        --
        KeyError
            If either slice is missing.

        """
        if slice1_id not in self._slices or slice2_id not in self._slices:
            raise KeyError("One or both slices not found")

        slice1 = self._slices[slice1_id]
        slice2 = self._slices[slice2_id]

        return {
            "vertices": slice1["vertices"] - slice2["vertices"],
            "edges": slice1["edges"] - slice2["edges"],
        }

    def create_slice_from_operation(self, result_slice_id, operation_result, **attributes):
        """Create a new slice from the result of a set operation.

        Parameters
        --
        result_slice_id : str
        operation_result : dict
            Output of ``slice_union``/``slice_intersection``/``slice_difference``.
        **attributes
            Pure slice attributes.

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            If the target slice already exists.

        """
        if result_slice_id in self._slices:
            raise ValueError(f"slice {result_slice_id} already exists")

        self._slices[result_slice_id] = {
            "vertices": operation_result["vertices"].copy(),
            "edges": operation_result["edges"].copy(),
            "attributes": attributes,
        }

        return result_slice_id

    def add_vertex_to_slice(self, lid, vid):
        """Attach an existing vertex to a slice.

        Parameters
        --
        lid : str
            slice ID.
        vid : str
            vertex ID.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if lid not in self._slices:
            raise KeyError(f"slice {lid} does not exist")
        self._slices[lid]["vertices"].add(vid)

    def edge_presence_across_slices(
        self,
        edge_id: str | None = None,
        source: str | None = None,
        target: str | None = None,
        *,
        include_default: bool = False,
        undirected_match: bool | None = None,
    ):
        """Locate where an edge exists across slices.

        Parameters
        --
        edge_id : str, optional
            If provided, match by ID (any kind: binary/vertex-edge/hyper).
        source : str, optional
            When used with ``target``, match only binary/vertex-edge edges by endpoints.
        target : str, optional
        include_default : bool, optional
            Include the internal default slice in the search.
        undirected_match : bool, optional
            When endpoint matching, allow undirected symmetric matches.

        Returns
        ---
        list[str] or dict[str, list[str]]
            If ``edge_id`` given: list of slice IDs.
            Else: ``{slice_id: [edge_id, ...]}``.

        Raises
        --
        ValueError
            If both modes (ID and endpoints) are provided or neither is valid.

        """
        has_id = edge_id is not None
        has_pair = (source is not None) and (target is not None)
        if has_id == has_pair:
            raise ValueError("Provide either edge_id OR (source and target), but not both.")

        slices_view = self.get_slices_dict(include_default=include_default)

        if has_id:
            return [lid for lid, ldata in slices_view.items() if edge_id in ldata["edges"]]

        if undirected_match is None:
            undirected_match = False

        out: dict[str, list[str]] = {}
        for lid, ldata in slices_view.items():
            matches = []
            for eid in ldata["edges"]:
                # skip hyper-edges for (source,target) mode
                if self.edge_kind.get(eid) == "hyper":
                    continue
                s, t, _ = self.edge_definitions[eid]
                edge_is_directed = self.edge_directed.get(
                    eid, True if self.directed is None else self.directed
                )
                if s == source and t == target:
                    matches.append(eid)
                elif undirected_match and not edge_is_directed and s == target and t == source:
                    matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def hyperedge_presence_across_slices(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        include_default: bool = False,
    ):
        """Locate slices containing a hyperedge with exactly these sets.

        Parameters
        --
        members : Iterable[str], optional
            Undirected member set (exact match).
        head : Iterable[str], optional
            Directed head set (exact match).
        tail : Iterable[str], optional
            Directed tail set (exact match).
        include_default : bool, optional

        Returns
        ---
        dict[str, list[str]]
            ``{slice_id: [edge_id, ...]}``.

        Raises
        --
        ValueError
            For invalid combinations or empty sets.

        """
        undirected = members is not None
        if undirected and (head is not None or tail is not None):
            raise ValueError("Use either members OR head+tail, not both.")
        if not undirected and (head is None or tail is None):
            raise ValueError("Directed hyperedge query requires both head and tail.")

        if undirected:
            members = set(members)
            if not members:
                raise ValueError("members must be non-empty.")
        else:
            head = set(head)
            tail = set(tail)
            if not head or not tail:
                raise ValueError("head and tail must be non-empty.")
            if head & tail:
                raise ValueError("head and tail must be disjoint.")

        slices_view = self.get_slices_dict(include_default=include_default)
        out: dict[str, list[str]] = {}

        for lid, ldata in slices_view.items():
            matches = []
            for eid in ldata["edges"]:
                if self.edge_kind.get(eid) != "hyper":
                    continue
                meta = self.hyperedge_definitions.get(eid, {})
                if undirected and (not meta.get("directed", False)):
                    if set(meta.get("members", ())) == members:
                        matches.append(eid)
                elif (not undirected) and meta.get("directed", False):
                    if set(meta.get("head", ())) == head and set(meta.get("tail", ())) == tail:
                        matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def vertex_presence_across_slices(self, vertex_id, include_default: bool = False):
        """List slices containing a specific vertex.

        Parameters
        --
        vertex_id : str
        include_default : bool, optional

        Returns
        ---
        list[str]

        """
        slices_with_vertex = []
        for slice_id, slice_data in self.get_slices_dict(include_default=include_default).items():
            if vertex_id in slice_data["vertices"]:
                slices_with_vertex.append(slice_id)
        return slices_with_vertex

    def conserved_edges(self, min_slices=2, include_default=False):
        """Edges present in at least ``min_slices`` slices.

        Parameters
        --
        min_slices : int, optional
        include_default : bool, optional

        Returns
        ---
        dict[str, int]
            ``{edge_id: count}``.

        """
        slices_to_check = self.get_slices_dict(
            include_default=include_default
        )  # hides 'default' by default
        edge_counts = {}
        for _, slice_data in slices_to_check.items():
            for eid in slice_data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_slices}

    def slice_specific_edges(self, slice_id):
        """Edges that appear **only** in the specified slice.

        Parameters
        --
        slice_id : str

        Returns
        ---
        set[str]

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        target_edges = self._slices[slice_id]["edges"]
        specific_edges = set()

        for edge_id in target_edges:
            # Count how many slices contain this edge
            count = sum(1 for slice_data in self._slices.values() if edge_id in slice_data["edges"])
            if count == 1:  # Only in target slice
                specific_edges.add(edge_id)

        return specific_edges

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        """Compute changes between consecutive slices in a temporal sequence.

        Parameters
        --
        ordered_slices : list[str]
            slice IDs in chronological order.
        metric : {'edge_change', 'vertex_change'}, optional

        Returns
        ---
        list[dict[str, int]]
            Per-step dictionaries with keys: ``'added'``, ``'removed'``, ``'net_change'``.

        Raises
        --
        ValueError
            If fewer than two slices are provided.
        KeyError
            If a referenced slice does not exist.

        """
        if len(ordered_slices) < 2:
            raise ValueError("Need at least 2 slices for temporal analysis")

        changes = []

        for i in range(len(ordered_slices) - 1):
            current_id = ordered_slices[i]
            next_id = ordered_slices[i + 1]

            if current_id not in self._slices or next_id not in self._slices:
                raise KeyError("One or more slices not found")

            current_data = self._slices[current_id]
            next_data = self._slices[next_id]

            if metric == "edge_change":
                added = len(next_data["edges"] - current_data["edges"])
                removed = len(current_data["edges"] - next_data["edges"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

            elif metric == "vertex_change":
                added = len(next_data["vertices"] - current_data["vertices"])
                removed = len(current_data["vertices"] - next_data["vertices"])
                changes.append({"added": added, "removed": removed, "net_change": added - removed})

        return changes

    def create_aggregated_slice(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        """Create a new slice by aggregating multiple source slices.

        Parameters
        --
        source_slice_ids : list[str]
        target_slice_id : str
        method : {'union', 'intersection'}, optional
        weight_func : callable, optional
            Reserved for future weight merging logic (currently unused).
        **attributes
            Pure slice attributes.

        Returns
        ---
        str
            The created slice ID.

        Raises
        --
        ValueError
            For unknown methods or missing source slices, or if target exists.

        """
        if not source_slice_ids:
            raise ValueError("Must specify at least one source slice")

        if target_slice_id in self._slices:
            raise ValueError(f"Target slice {target_slice_id} already exists")

        if method == "union":
            result = self.slice_union(source_slice_ids)
        elif method == "intersection":
            result = self.slice_intersection(source_slice_ids)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

        return self.create_slice_from_operation(target_slice_id, result, **attributes)

    def slice_statistics(self, include_default: bool = False):
        """Basic per-slice statistics.

        Parameters
        --
        include_default : bool, optional

        Returns
        ---
        dict[str, dict]
            ``{slice_id: {'vertices': int, 'edges': int, 'attributes': dict}}``.

        """
        stats = {}
        for slice_id, slice_data in self.get_slices_dict(include_default=include_default).items():
            stats[slice_id] = {
                "vertices": len(slice_data["vertices"]),
                "edges": len(slice_data["edges"]),
                "attributes": slice_data["attributes"],
            }
        return stats
