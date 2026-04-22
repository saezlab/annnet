from ._records import EdgeRecord, SliceRecord


class SliceManager:
    """Namespace for all slice operations.

    Owns the full slice implementation — SliceClass on AnnNet is a thin
    forwarding shell kept for callers that cannot be updated (e.g. adapters).
    State (_slices, _current_slice, _default_slice, slice_edge_weights) lives
    on AnnNet because it is accessed pervasively in core graph methods.
    """

    __slots__ = ("_G",)

    def __init__(self, graph):
        self._G = graph

    def _empty_slice_record(self):
        return SliceRecord()

    def _slice_attrs(self, slice_id):
        G = self._G
        df = getattr(G, "slice_attributes", None)
        if df is None or not hasattr(df, "columns") or "slice_id" not in df.columns:
            return {}

        try:
            import math

            import polars as pl
        except Exception:
            math = None
            pl = None

        def _clean(row):
            out = {}
            for k, v in row.items():
                if k == "slice_id" or v is None:
                    continue
                if math is not None and isinstance(v, float) and math.isnan(v):
                    continue
                out[k] = v
            return out

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter(pl.col("slice_id") == slice_id).to_dicts()
            return _clean(rows[0]) if rows else {}

        try:
            import narwhals as nw

            native = nw.to_native(
                nw.from_native(df, pass_through=True).filter(nw.col("slice_id") == slice_id)
            )
            rows = (
                native.to_dicts()
                if hasattr(native, "to_dicts")
                else native.to_dict(orient="records")
            )
            return _clean(rows[0]) if rows else {}
        except Exception:
            return {}

    def _ensure_slice(self, slice_id, **attributes):
        G = self._G
        if slice_id not in G._slices:
            G._slices[slice_id] = self._empty_slice_record()
        if attributes:
            G.attrs.set_slice_attrs(slice_id, **attributes)
        return G._slices[slice_id]

    # ── core mutations ────────────────────────────────────────────────────────

    def add_slice(self, slice_id, **attributes):
        """Create a new empty slice.

        Parameters
        ----------
        slice_id : str
        **attributes
            Slice attributes.

        Returns
        -------
        str
        """
        G = self._G
        if slice_id in G._slices and slice_id != "default":
            raise ValueError(f"slice {slice_id} already exists")
        self._ensure_slice(slice_id, **attributes)
        return slice_id

    # kept as a method (not just via SliceClass) so SliceManager.add() works as alias
    def add(self, slice_id, **attributes):
        return self.add_slice(slice_id, **attributes)

    def remove(self, slice_id):
        """Remove a non-default slice."""
        return self.remove_slice(slice_id)

    def remove_slice(self, slice_id):
        """Remove a non-default slice and its per-slice attributes.

        Parameters
        ----------
        slice_id : str

        Raises
        ------
        ValueError
            If attempting to remove the internal default slice.
        KeyError
            If the slice does not exist.
        """
        G = self._G
        if slice_id == G._default_slice:
            raise ValueError("Cannot remove default slice")
        if slice_id not in G._slices:
            raise KeyError(f"slice {slice_id} not found")

        ela = getattr(G, "edge_slice_attributes", None)
        if ela is not None and hasattr(ela, "columns"):
            cols = list(ela.columns)
            is_empty = (getattr(ela, "height", None) == 0) or (
                hasattr(ela, "__len__") and len(ela) == 0
            )
            if (not is_empty) and ("slice_id" in cols):
                from ._records import _df_filter_not_equal

                G.edge_slice_attributes = _df_filter_not_equal(ela, "slice_id", slice_id)

        if isinstance(G.slice_edge_weights, dict):
            G.slice_edge_weights.pop(slice_id, None)

        del G._slices[slice_id]
        G._rebuild_slice_edge_weights_cache()
        if G._current_slice == slice_id:
            G._current_slice = G._default_slice

    def add_edge_to_slice(self, lid, eid):
        """Attach an existing edge to a slice (no weight changes).

        Parameters
        ----------
        lid : str
        eid : str

        Raises
        ------
        KeyError
            If the slice does not exist.
        """
        G = self._G
        if lid not in G._slices:
            raise KeyError(f"slice {lid} does not exist")
        if eid not in G._edges:
            if eid not in G._entities:
                G._register_edge_as_entity(eid)
            G._edges[eid] = EdgeRecord(
                src=None,
                tgt=None,
                weight=1.0,
                directed=False,
                etype="edge_placeholder",
                col_idx=-1,
                ml_kind=None,
                ml_layers=None,
                direction_policy=None,
            )
        G._slices[lid]["edges"].add(eid)

    def add_edges(self, slice_id, edge_ids):
        """Attach many existing edges to a slice and include their incident vertices."""
        G = self._G
        sid = slice_id if slice_id is not None else G._current_slice
        data = self._ensure_slice(sid)

        add_edges = {eid for eid in edge_ids if eid in G._edges and G._edges[eid].col_idx >= 0}
        if not add_edges:
            return

        data["edges"].update(add_edges)
        verts: set = set()
        for eid in add_edges:
            rec = G._edges[eid]
            if rec.etype == "hyper":
                if rec.src is not None:
                    verts.update(rec.src)
                if rec.tgt is not None:
                    verts.update(rec.tgt)
            else:
                if rec.src is not None:
                    verts.add(rec.src)
                if rec.tgt is not None:
                    verts.add(rec.tgt)
        data["vertices"].update(verts)

    # ── active slice ──────────────────────────────────────────────────────────

    @property
    def active(self):
        return self._G._current_slice

    @active.setter
    def active(self, slice_id):
        self.set_active_slice(slice_id)

    def set_active_slice(self, slice_id):
        if slice_id not in self._G._slices:
            raise KeyError(f"slice {slice_id} not found")
        self._G._current_slice = slice_id

    def get_active_slice(self):
        return self._G._current_slice

    # ── queries ───────────────────────────────────────────────────────────────

    def get_slices_dict(self, include_default=False):
        G = self._G
        if include_default:
            return G._slices
        return {k: v for k, v in G._slices.items() if k != G._default_slice}

    def list_slices(self, include_default=False):
        return list(self.get_slices_dict(include_default=include_default).keys())

    def list(self, include_default=False):
        return self.list_slices(include_default=include_default)

    def has_slice(self, slice_id):
        return slice_id in self._G._slices

    def exists(self, slice_id):
        return self.has_slice(slice_id)

    def slice_count(self):
        return len(self._G._slices)

    def count(self):
        return self.slice_count()

    def get_slice_info(self, slice_id):
        G = self._G
        if slice_id not in G._slices:
            raise KeyError(f"slice {slice_id} not found")
        data = G._slices[slice_id]
        return {
            "vertices": data["vertices"].copy(),
            "edges": data["edges"].copy(),
            "attributes": self._slice_attrs(slice_id),
        }

    def info(self, slice_id):
        return self.get_slice_info(slice_id)

    def get_slice_vertices(self, slice_id):
        return self._G._slices[slice_id]["vertices"].copy()

    def vertices(self, slice_id):
        return self.get_slice_vertices(slice_id)

    def get_slice_edges(self, slice_id):
        return self._G._slices[slice_id]["edges"].copy()

    def edges(self, slice_id):
        return self.get_slice_edges(slice_id)

    # ── set operations ────────────────────────────────────────────────────────

    def slice_union(self, slice_ids):
        G = self._G
        union_vertices: set = set()
        union_edges: set = set()
        for sid in slice_ids:
            if sid in G._slices:
                union_vertices.update(G._slices[sid]["vertices"])
                union_edges.update(G._slices[sid]["edges"])
        return {"vertices": union_vertices, "edges": union_edges}

    def union(self, slice_ids):
        return self.slice_union(slice_ids)

    def slice_intersection(self, slice_ids):
        G = self._G
        if not slice_ids:
            return {"vertices": set(), "edges": set()}
        if len(slice_ids) == 1:
            sid = slice_ids[0]
            data = G._slices.get(sid, SliceRecord())
            return {"vertices": data["vertices"].copy(), "edges": data["edges"].copy()}
        common_v = G._slices[slice_ids[0]]["vertices"].copy()
        common_e = G._slices[slice_ids[0]]["edges"].copy()
        for sid in slice_ids[1:]:
            if sid in G._slices:
                common_v &= G._slices[sid]["vertices"]
                common_e &= G._slices[sid]["edges"]
            else:
                return {"vertices": set(), "edges": set()}
        return {"vertices": common_v, "edges": common_e}

    def intersect(self, slice_ids):
        return self.slice_intersection(slice_ids)

    def slice_difference(self, slice1_id, slice2_id):
        G = self._G
        if slice1_id not in G._slices or slice2_id not in G._slices:
            raise KeyError("One or both slices not found")
        s1 = G._slices[slice1_id]
        s2 = G._slices[slice2_id]
        return {
            "vertices": s1["vertices"] - s2["vertices"],
            "edges": s1["edges"] - s2["edges"],
        }

    def difference(self, slice_a, slice_b):
        return self.slice_difference(slice_a, slice_b)

    def create_slice_from_operation(self, result_slice_id, operation_result, **attributes):
        G = self._G
        if result_slice_id in G._slices:
            raise ValueError(f"slice {result_slice_id} already exists")
        data = self._ensure_slice(result_slice_id, **attributes)
        data["vertices"] = operation_result["vertices"].copy()
        data["edges"] = operation_result["edges"].copy()
        return result_slice_id

    def add_vertex_to_slice(self, lid, vid):
        G = self._G
        if lid not in G._slices:
            raise KeyError(f"slice {lid} does not exist")
        G._slices[lid]["vertices"].add(vid)

    # ── set-op creation helpers ───────────────────────────────────────────────

    def union_create(self, slice_ids, name, **attributes):
        result = self.slice_union(slice_ids)
        return self.create_slice_from_operation(name, result, **attributes)

    def intersect_create(self, slice_ids, name, **attributes):
        result = self.slice_intersection(slice_ids)
        return self.create_slice_from_operation(name, result, **attributes)

    def difference_create(self, slice_a, slice_b, name, **attributes):
        result = self.slice_difference(slice_a, slice_b)
        return self.create_slice_from_operation(name, result, **attributes)

    def aggregate(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        return self.create_aggregated_slice(
            source_slice_ids, target_slice_id, method, weight_func, **attributes
        )

    def create_aggregated_slice(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        if not source_slice_ids:
            raise ValueError("Must specify at least one source slice")
        if target_slice_id in self._G._slices:
            raise ValueError(f"Target slice {target_slice_id} already exists")
        G = self._G
        data = self._ensure_slice(target_slice_id, **attributes)
        if method == "union":
            vertices = set()
            edges = set()
            for sid in source_slice_ids:
                src = G._slices.get(sid)
                if src is None:
                    continue
                vertices.update(src["vertices"])
                edges.update(src["edges"])
            data["vertices"] = vertices
            data["edges"] = edges
            return target_slice_id
        if method == "intersection":
            first = G._slices.get(source_slice_ids[0], SliceRecord())
            vertices = first["vertices"].copy()
            edges = first["edges"].copy()
            for sid in source_slice_ids[1:]:
                src = G._slices.get(sid)
                if src is None:
                    vertices = set()
                    edges = set()
                    break
                vertices.intersection_update(src["vertices"])
                edges.intersection_update(src["edges"])
            data["vertices"] = vertices
            data["edges"] = edges
            return target_slice_id
        raise ValueError(f"Unknown aggregation method: {method}")

    # ── analytics ─────────────────────────────────────────────────────────────

    def slice_statistics(self, include_default=False):
        stats = {}
        for sid, data in self.get_slices_dict(include_default=include_default).items():
            stats[sid] = {
                "vertices": len(data["vertices"]),
                "edges": len(data["edges"]),
                "attributes": self._slice_attrs(sid),
            }
        return stats

    def stats(self, include_default=False):
        return self.slice_statistics(include_default=include_default)

    def vertex_presence(self, vertex_id, include_default=False):
        return self.vertex_presence_across_slices(vertex_id, include_default)

    def vertex_presence_across_slices(self, vertex_id, include_default=False):
        return [
            sid
            for sid, data in self.get_slices_dict(include_default=include_default).items()
            if vertex_id in data["vertices"]
        ]

    def edge_presence(
        self, edge_id=None, source=None, target=None, include_default=False, undirected_match=None
    ):
        return self.edge_presence_across_slices(
            edge_id,
            source,
            target,
            include_default=include_default,
            undirected_match=undirected_match,
        )

    def edge_presence_across_slices(
        self,
        edge_id=None,
        source=None,
        target=None,
        *,
        include_default=False,
        undirected_match=None,
    ):
        G = self._G
        has_id = edge_id is not None
        has_pair = (source is not None) and (target is not None)
        if has_id == has_pair:
            raise ValueError("Provide either edge_id OR (source and target), but not both.")
        slices_view = self.get_slices_dict(include_default=include_default)
        if has_id:
            return [lid for lid, ldata in slices_view.items() if edge_id in ldata["edges"]]
        if undirected_match is None:
            undirected_match = False
        out: dict = {}
        default_dir = True if G.directed is None else G.directed
        for lid, ldata in slices_view.items():
            matches = []
            for eid in ldata["edges"]:
                rec = G._edges.get(eid)
                if rec is None or rec.col_idx < 0 or rec.etype == "hyper":
                    continue
                s, t = rec.src, rec.tgt
                edge_is_directed = rec.directed if rec.directed is not None else default_dir
                if s == source and t == target:
                    matches.append(eid)
                elif undirected_match and not edge_is_directed and s == target and t == source:
                    matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def hyperedge_presence(self, members=None, head=None, tail=None, include_default=False):
        return self.hyperedge_presence_across_slices(
            members=members, head=head, tail=tail, include_default=include_default
        )

    def hyperedge_presence_across_slices(
        self, *, members=None, head=None, tail=None, include_default=False
    ):
        G = self._G
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
        out: dict = {}
        for lid, ldata in slices_view.items():
            matches = []
            for eid in ldata["edges"]:
                rec = G._edges.get(eid)
                if rec is None or rec.col_idx < 0 or rec.etype != "hyper":
                    continue
                if undirected and rec.tgt is None:
                    if set(rec.src) == members:
                        matches.append(eid)
                elif (not undirected) and rec.tgt is not None:
                    if set(rec.src) == head and set(rec.tgt) == tail:
                        matches.append(eid)
            if matches:
                out[lid] = matches
        return out

    def conserved_edges(self, min_slices=2, include_default=False):
        G = self._G
        edge_counts: dict = {}
        for sid, data in G._slices.items():
            if not include_default and sid == G._default_slice:
                continue
            for eid in data["edges"]:
                edge_counts[eid] = edge_counts.get(eid, 0) + 1
        return {eid: c for eid, c in edge_counts.items() if c >= min_slices}

    def specific_edges(self, slice_id):
        return self.slice_specific_edges(slice_id)

    def slice_specific_edges(self, slice_id):
        G = self._G
        if slice_id not in G._slices:
            raise KeyError(f"slice {slice_id} not found")
        target = G._slices[slice_id]["edges"]
        return {
            eid
            for eid in target
            if sum(1 for data in G._slices.values() if eid in data["edges"]) == 1
        }

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        G = self._G
        if len(ordered_slices) < 2:
            raise ValueError("Need at least 2 slices for temporal analysis")
        changes = []
        for i in range(len(ordered_slices) - 1):
            cur, nxt = ordered_slices[i], ordered_slices[i + 1]
            if cur not in G._slices or nxt not in G._slices:
                raise KeyError("One or more slices not found")
            cd, nd = G._slices[cur], G._slices[nxt]
            key = "edges" if metric == "edge_change" else "vertices"
            added = len(nd[key] - cd[key])
            removed = len(cd[key] - nd[key])
            changes.append({"added": added, "removed": removed, "net_change": added - removed})
        return changes

    # ── convenience ───────────────────────────────────────────────────────────

    def summary(self):
        stats = self.stats(include_default=True)
        lines = [f"slices: {len(stats)}"]
        for i, (sid, info) in enumerate(stats.items()):
            prefix = "├─" if i < len(stats) - 1 else "└─"
            lines.append(f"{prefix} {sid}: {info['vertices']} vertices, {info['edges']} edges")
        return "\n".join(lines)

    def __repr__(self):
        return f"SliceManager({self.count()} slices)"


class SliceClass:
    """Thin forwarding shell that keeps slice methods on AnnNet.

    All logic lives in SliceManager. These forwarders exist because
    external code (adapters etc.) calls G.add_slice() / G.list_slices()
    directly and cannot be updated.
    """

    def add_slice(self, slice_id, **attributes):
        return self.slices.add_slice(slice_id, **attributes)

    def remove_slice(self, slice_id):
        return self.slices.remove_slice(slice_id)

    def add_edge_to_slice(self, lid, eid):
        return self.slices.add_edge_to_slice(lid, eid)

    def set_active_slice(self, slice_id):
        return self.slices.set_active_slice(slice_id)

    def get_active_slice(self):
        return self.slices.get_active_slice()

    def get_slices_dict(self, include_default=False):
        return self.slices.get_slices_dict(include_default=include_default)

    def list_slices(self, include_default=False):
        return self.slices.list_slices(include_default=include_default)

    def has_slice(self, slice_id):
        return self.slices.has_slice(slice_id)

    def slice_count(self):
        return self.slices.slice_count()

    def get_slice_info(self, slice_id):
        return self.slices.get_slice_info(slice_id)

    def get_slice_vertices(self, slice_id):
        return self.slices.get_slice_vertices(slice_id)

    def get_slice_edges(self, slice_id):
        return self.slices.get_slice_edges(slice_id)

    def slice_union(self, slice_ids):
        return self.slices.slice_union(slice_ids)

    def slice_intersection(self, slice_ids):
        return self.slices.slice_intersection(slice_ids)

    def slice_difference(self, slice1_id, slice2_id):
        return self.slices.slice_difference(slice1_id, slice2_id)

    def create_slice_from_operation(self, result_slice_id, operation_result, **attributes):
        return self.slices.create_slice_from_operation(
            result_slice_id, operation_result, **attributes
        )

    def add_vertex_to_slice(self, lid, vid):
        return self.slices.add_vertex_to_slice(lid, vid)

    def edge_presence_across_slices(
        self,
        edge_id=None,
        source=None,
        target=None,
        *,
        include_default=False,
        undirected_match=None,
    ):
        return self.slices.edge_presence_across_slices(
            edge_id,
            source,
            target,
            include_default=include_default,
            undirected_match=undirected_match,
        )

    def hyperedge_presence_across_slices(
        self, *, members=None, head=None, tail=None, include_default=False
    ):
        return self.slices.hyperedge_presence_across_slices(
            members=members, head=head, tail=tail, include_default=include_default
        )

    def vertex_presence_across_slices(self, vertex_id, include_default=False):
        return self.slices.vertex_presence_across_slices(vertex_id, include_default)

    def conserved_edges(self, min_slices=2, include_default=False):
        return self.slices.conserved_edges(min_slices, include_default)

    def slice_specific_edges(self, slice_id):
        return self.slices.slice_specific_edges(slice_id)

    def temporal_dynamics(self, ordered_slices, metric="edge_change"):
        return self.slices.temporal_dynamics(ordered_slices, metric)

    def create_aggregated_slice(
        self, source_slice_ids, target_slice_id, method="union", weight_func=None, **attributes
    ):
        return self.slices.create_aggregated_slice(
            source_slice_ids, target_slice_id, method, weight_func, **attributes
        )

    def slice_statistics(self, include_default=False):
        return self.slices.slice_statistics(include_default=include_default)
