from __future__ import annotations

import time
from typing import TYPE_CHECKING

import narwhals as nw

try:
    import polars as pl
except Exception:
    pl = None
if TYPE_CHECKING:
    from .graph import AnnNet


class CacheManager:
    """Cache manager for materialized views (CSR/CSC)."""

    def __init__(self, graph):
        self._G = graph
        self._csr = None
        self._csc = None
        self._adjacency = None
        self._csr_version = None
        self._csc_version = None
        self._adjacency_version = None

    # ==================== CSR/CSC Properties ====================

    @property
    def csr(self):
        """Get CSR (Compressed Sparse Row) format.
        Builds and caches on first access.
        """
        if self._csr is None or self._csr_version != self._G._version:
            self._csr = self._G._matrix.tocsr()
            self._csr_version = self._G._version
        return self._csr

    @property
    def csc(self):
        """Get CSC (Compressed Sparse Column) format.
        Builds and caches on first access.
        """
        if self._csc is None or self._csc_version != self._G._version:
            self._csc = self._G._matrix.tocsc()
            self._csc_version = self._G._version
        return self._csc

    @property
    def adjacency(self):
        """Get adjacency matrix (computed from incidence).
        For incidence B: adjacency A = B @ B.T
        """
        if self._adjacency is None or self._adjacency_version != self._G._version:
            csr = self.csr
            # Adjacency from incidence: A = B @ B.T
            self._adjacency = csr @ csr.T
            self._adjacency_version = self._G._version
        return self._adjacency

    def has_csr(self) -> bool:
        """True if CSR cache exists and matches current graph version."""
        return self._csr is not None and self._csr_version == self._G._version

    def has_csc(self) -> bool:
        """True if CSC cache exists and matches current graph version."""
        return self._csc is not None and self._csc_version == self._G._version

    def has_adjacency(self) -> bool:
        """True if adjacency cache exists and matches current graph version."""
        return self._adjacency is not None and self._adjacency_version == self._G._version

    def get_csr(self):
        return self.csr

    def get_csc(self):
        return self.csc

    def get_adjacency(self):
        return self.adjacency

    # ==================== Cache Management ====================

    def invalidate(self, formats=None):
        """Invalidate cached formats.

        Parameters
        --
        formats : list[str], optional
            Formats to invalidate ('csr', 'csc', 'adjacency').
            If None, invalidate all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                self._csr = None
                self._csr_version = None
            elif fmt == "csc":
                self._csc = None
                self._csc_version = None
            elif fmt == "adjacency":
                self._adjacency = None
                self._adjacency_version = None

    def build(self, formats=None):
        """Pre-build specified formats (eager caching).

        Parameters
        --
        formats : list[str], optional
            Formats to build ('csr', 'csc', 'adjacency').
            If None, build all.

        """
        if formats is None:
            formats = ["csr", "csc", "adjacency"]

        for fmt in formats:
            if fmt == "csr":
                _ = self.csr
            elif fmt == "csc":
                _ = self.csc
            elif fmt == "adjacency":
                _ = self.adjacency

    def clear(self):
        """Clear all caches."""
        self.invalidate()

    def info(self):
        """Get cache status and memory usage.

        Returns
        ---
        dict
            Status of each cached format

        """

        def _format_info(matrix, version):
            if matrix is None:
                return {"cached": False}

            # Calculate size
            size_bytes = 0
            if hasattr(matrix, "data"):
                size_bytes += matrix.data.nbytes
            if hasattr(matrix, "indices"):
                size_bytes += matrix.indices.nbytes
            if hasattr(matrix, "indptr"):
                size_bytes += matrix.indptr.nbytes

            return {
                "cached": True,
                "version": version,
                "size_mb": size_bytes / (1024**2),
                "nnz": matrix.nnz if hasattr(matrix, "nnz") else 0,
                "shape": matrix.shape,
            }

        return {
            "csr": _format_info(self._csr, self._csr_version),
            "csc": _format_info(self._csc, self._csc_version),
            "adjacency": _format_info(self._adjacency, self._adjacency_version),
        }


class Operations:
    # Slicing / copying / accounting

    def edge_subgraph(self, edges) -> AnnNet:
        """Create a new graph containing only a specified subset of edges.

        Parameters
        --
        edges : Iterable[str] | Iterable[int]
            Edge identifiers (strings) or edge indices (integers) to retain
            in the subgraph.

        Returns
        ---
        AnnNet
            A new `AnnNet` instance containing only the selected edges and the
            vertices incident to them.

        Behavior

        - Copies the current graph and deletes all edges **not** in the provided set.
        - Optionally, you can prune orphaned vertices (i.e., vertices not incident
        to any remaining edge) — this is generally recommended for consistency.

        Notes
        -
        - Attributes associated with remaining edges and vertices are preserved.
        - Hyperedges are supported: if a hyperedge is in the provided set, all
        its members are retained.
        - If `edges` is empty, the resulting graph will be empty except for
        any isolated vertices that remain.

        """
        # normalize to edge_id set
        if all(isinstance(e, int) for e in edges):
            E = {self.idx_to_edge[e] for e in edges}
        else:
            E = set(edges)

        # collect incident vertices and partition edges
        V = set()
        bin_payload, hyper_payload = [], []
        for eid in E:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    V.update(h["members"])
                    hyper_payload.append(
                        {
                            "members": list(h["members"]),
                            "edge_id": eid,
                            "weight": self.edge_weights.get(eid, 1.0),
                        }
                    )
                else:
                    V.update(h.get("head", ()))
                    V.update(h.get("tail", ()))
                    hyper_payload.append(
                        {
                            "head": list(h.get("head", ())),
                            "tail": list(h.get("tail", ())),
                            "edge_id": eid,
                            "weight": self.edge_weights.get(eid, 1.0),
                        }
                    )
            else:
                s, t, etype = self.edge_definitions[eid]
                V.add(s)
                V.add(t)
                bin_payload.append(
                    {
                        "source": s,
                        "target": t,
                        "edge_id": eid,
                        "edge_type": etype,
                        "edge_directed": self.edge_directed.get(
                            eid, True if self.directed is None else self.directed
                        ),
                        "weight": self.edge_weights.get(eid, 1.0),
                    }
                )

        # new graph prealloc
        G = self.__class__
        g = G(directed=self.directed, n=len(V), e=len(E))
        # vertices with attrs
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]
        g.add_vertices_bulk(v_rows, slice=g._default_slice)

        # edges
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=g._default_slice)

        # copy slice memberships for retained edges & incident vertices
        for lid, meta in self._slices.items():
            g.add_slice(lid, **meta["attributes"])
            kept_edges = set(meta["edges"]) & E
            if kept_edges:
                g.add_edges_to_slice_bulk(lid, kept_edges)

        return g

    def subgraph(self, vertices) -> AnnNet:
        """Create a vertex-induced subgraph.

        Parameters
        --
        vertices : Iterable[str]
            A set or list of vertex identifiers to keep in the subgraph.

        Returns
        ---
        AnnNet
            A new `AnnNet` containing only the specified vertices and any edges
            for which **all** endpoints are within this set.

        Behavior

        - Copies the current graph and removes edges with any endpoint outside
        the provided vertex set.
        - Removes all vertices not listed in `vertices`.

        Notes
        -
        - For binary edges, both endpoints must be in `vertices` to be retained.
        - For hyperedges, **all** member vertices must be included to retain the edge.
        - Attributes for retained vertices and edges are preserved.

        """
        V = set(vertices)

        # collect edges fully inside V
        E_bin, E_hyper_members, E_hyper_dir = [], [], []
        for eid, (s, t, et) in self.edge_definitions.items():
            if et == "hyper":
                continue
            if s in V and t in V:
                E_bin.append(eid)
        for eid, h in self.hyperedge_definitions.items():
            if h.get("members"):
                if set(h["members"]).issubset(V):
                    E_hyper_members.append(eid)
            else:
                if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(V):
                    E_hyper_dir.append(eid)

        # payloads
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]

        bin_payload = []
        for eid in E_bin:
            s, t, etype = self.edge_definitions[eid]
            bin_payload.append(
                {
                    "source": s,
                    "target": t,
                    "edge_id": eid,
                    "edge_type": etype,
                    "edge_directed": self.edge_directed.get(
                        eid, True if self.directed is None else self.directed
                    ),
                    "weight": self.edge_weights.get(eid, 1.0),
                }
            )

        hyper_payload = []
        for eid in E_hyper_members:
            m = self.hyperedge_definitions[eid]["members"]
            hyper_payload.append(
                {"members": list(m), "edge_id": eid, "weight": self.edge_weights.get(eid, 1.0)}
            )
        for eid in E_hyper_dir:
            h = self.hyperedge_definitions[eid]
            hyper_payload.append(
                {
                    "head": list(h.get("head", ())),
                    "tail": list(h.get("tail", ())),
                    "edge_id": eid,
                    "weight": self.edge_weights.get(eid, 1.0),
                }
            )

        # build new graph
        G = self.__class__
        g = G(
            directed=self.directed, n=len(V), e=len(E_bin) + len(E_hyper_members) + len(E_hyper_dir)
        )
        g.add_vertices_bulk(v_rows, slice=g._default_slice)
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=g._default_slice)

        # slice memberships restricted to V
        for lid, meta in self._slices.items():
            g.add_slice(lid, **meta["attributes"])
            keep = set()
            for eid in meta["edges"]:
                kind = self.edge_kind.get(eid, "binary")
                if kind == "hyper":
                    h = self.hyperedge_definitions[eid]
                    if h.get("members"):
                        if set(h["members"]).issubset(V):
                            keep.add(eid)
                    else:
                        if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(
                            V
                        ):
                            keep.add(eid)
                else:
                    s, t, _ = self.edge_definitions[eid]
                    if s in V and t in V:
                        keep.add(eid)
            if keep:
                g.add_edges_to_slice_bulk(lid, keep)

        return g

    def extract_subgraph(self, vertices=None, edges=None) -> AnnNet:
        """Create a subgraph based on a combination of vertex and/or edge filters.

        Parameters
        --
        vertices : Iterable[str] | None, optional
            A set of vertex IDs to include. If provided, behaves like `subgraph()`.
            If `None`, no vertex filtering is applied.
        edges : Iterable[str] | Iterable[int] | None, optional
            A set of edge IDs or indices to include. If provided, behaves like
            `edge_subgraph()`. If `None`, no edge filtering is applied.

        Returns
        ---
        AnnNet
            A new `AnnNet` filtered according to the provided vertex and/or edge
            sets.

        Behavior

        - If both `vertices` and `edges` are provided, the resulting subgraph is
        the intersection of the two filters.
        - If only `vertices` is provided, equivalent to `subgraph(vertices)`.
        - If only `edges` is provided, equivalent to `edge_subgraph(edges)`.
        - If neither is provided, a full copy of the graph is returned.

        Notes
        -
        - This is a convenience method; it delegates to `subgraph()` and
        `edge_subgraph()` internally.

        """
        if vertices is None and edges is None:
            return self.copy()

        if edges is not None:
            if all(isinstance(e, int) for e in edges):
                E = {self.idx_to_edge[e] for e in edges}
            else:
                E = set(edges)
        else:
            E = None

        V = set(vertices) if vertices is not None else None

        # If only one filter, delegate to optimized path
        if V is not None and E is None:
            return self.subgraph(V)
        if V is None and E is not None:
            return self.edge_subgraph(E)

        # Both filters: keep only edges in E whose endpoints (or members) lie in V
        kept_edges = set()
        kept_vertices = set(V)
        for eid in E:
            kind = self.edge_kind.get(eid, "binary")
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    if set(h["members"]).issubset(V):
                        kept_edges.add(eid)
                else:
                    if set(h.get("head", ())).issubset(V) and set(h.get("tail", ())).issubset(V):
                        kept_edges.add(eid)
            else:
                s, t, _ = self.edge_definitions[eid]
                if s in V and t in V:
                    kept_edges.add(eid)

        return self.edge_subgraph(kept_edges).subgraph(kept_vertices)

    def reverse(self) -> AnnNet:
        """Return a new graph with all directed edges reversed.

        Returns
        ---
        AnnNet
            A new `AnnNet` instance with reversed directionality where applicable.

        Behavior

        - **Binary edges:** direction is flipped by swapping source and target.
        - **Directed hyperedges:** `head` and `tail` sets are swapped.
        - **Undirected edges/hyperedges:** unaffected.
        - Edge attributes and metadata are preserved.

        Notes
        -
        - This operation does not modify the original graph.
        - If the graph is undirected (`self.directed == False`), the result is
        identical to the original.
        - For mixed graphs (directed + undirected edges), only the directed
        ones are reversed.

        """
        g = self.copy()

        for eid, defn in g.edge_definitions.items():
            if not g._is_directed_edge(eid):
                continue
            # Binary edge: swap endpoints
            u, v, etype = defn
            g.edge_definitions[eid] = (v, u, etype)

        for eid, meta in g.hyperedge_definitions.items():
            if not meta.get("directed", False):
                continue
            # Hyperedge: swap head and tail sets
            meta["head"], meta["tail"] = meta["tail"], meta["head"]

        return g

    def subgraph_from_slice(self, slice_id, *, resolve_slice_weights=True):
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        slice_meta = self._slices[slice_id]
        V = set(slice_meta["vertices"])
        E = set(slice_meta["edges"])

        G = self.__class__
        g = G(directed=self.directed, n=len(V), e=len(E))
        g.add_slice(slice_id, **slice_meta["attributes"])
        g.set_active_slice(slice_id)

        # vertices with attrs (edge-entities share same table)
        v_rows = [
            {"vertex_id": v, **(self._row_attrs(self.vertex_attributes, "vertex_id", v) or {})}
            for v in V
        ]
        g.add_vertices_bulk(v_rows, slice=slice_id)

        # edge attrs
        e_attrs = {}
        ea = self.edge_attributes
        if ea is not None and hasattr(ea, "columns") and "edge_id" in ea.columns:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(ea, pl.DataFrame) and ea.height:
                for row in ea.filter(pl.col("edge_id").is_in(list(E))).to_dicts():
                    d = dict(row)
                    eid = d.pop("edge_id", None)
                    if eid is not None:
                        e_attrs[eid] = d
            else:
                import narwhals as nw

                ndf = nw.from_native(ea, pass_through=True)
                native = nw.to_native(ndf.filter(nw.col("edge_id").is_in(list(E))))

                # Polars: to_dicts(); Pandas: to_dict(orient="records")
                to_dicts_fn = getattr(type(native), "to_dicts", None)

                if callable(to_dicts_fn):
                    rows = to_dicts_fn(native)
                else:
                    try:
                        rows = native.to_dict(orient="records")
                    except TypeError:
                        rows = native.to_dict()

                for row in rows:
                    d = dict(row)
                    eid = d.pop("edge_id", None)
                    if eid is not None:
                        e_attrs[eid] = d

        # weights
        eff_w = {}
        if resolve_slice_weights:
            df = self.edge_slice_attributes
            if (
                df is not None
                and hasattr(df, "columns")
                and {"slice_id", "edge_id", "weight"}.issubset(df.columns)
            ):
                try:
                    import polars as pl
                except Exception:
                    pl = None

                if pl is not None and isinstance(df, pl.DataFrame) and df.height:
                    for r in df.filter(
                        (pl.col("slice_id") == slice_id) & (pl.col("edge_id").is_in(list(E)))
                    ).iter_rows(named=True):
                        if r.get("weight") is not None:
                            eff_w[r["edge_id"]] = float(r["weight"])
                else:
                    import narwhals as nw

                    ndf = nw.from_native(df, pass_through=True).filter(
                        (nw.col("slice_id") == slice_id) & (nw.col("edge_id").is_in(list(E)))
                    )
                    native = nw.to_native(ndf)
                    # iterate rows as dicts (works across backends via narwhals -> native)
                    to_dicts_fn = getattr(type(native), "to_dicts", None)

                    if callable(to_dicts_fn):
                        rows = to_dicts_fn(native)
                    else:
                        try:
                            rows = native.to_dict(orient="records")
                        except TypeError:
                            rows = native.to_dict()
                    for r in rows:
                        w = r.get("weight")
                        if w is not None:
                            eff_w[r["edge_id"]] = float(w)

        # partition edges
        bin_payload, hyper_payload = [], []
        for eid in E:
            w = (
                eff_w.get(eid, self.edge_weights.get(eid, 1.0))
                if resolve_slice_weights
                else self.edge_weights.get(eid, 1.0)
            )
            kind = self.edge_kind.get(eid, "binary")
            attrs = e_attrs.get(eid, {})
            if kind == "hyper":
                h = self.hyperedge_definitions[eid]
                if h.get("members"):
                    hyper_payload.append(
                        {
                            "members": list(h["members"]),
                            "edge_id": eid,
                            "weight": w,
                            "attributes": attrs,
                        }
                    )
                else:
                    hyper_payload.append(
                        {
                            "head": list(h.get("head", ())),
                            "tail": list(h.get("tail", ())),
                            "edge_id": eid,
                            "weight": w,
                            "attributes": attrs,
                        }
                    )
            else:
                s, t, et = self.edge_definitions[eid]
                bin_payload.append(
                    {
                        "source": s,
                        "target": t,
                        "edge_id": eid,
                        "edge_type": et,
                        "edge_directed": self.edge_directed.get(
                            eid, True if self.directed is None else self.directed
                        ),
                        "weight": w,
                        "attributes": attrs,
                    }
                )

        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=slice_id)
        if hyper_payload:
            g.add_hyperedges_bulk(hyper_payload, slice=slice_id)

        return g

    def _row_attrs(self, df, key_col: str, key):
        """INTERNAL: return a dict of attributes for the row in `df` where `key_col == key`,
        excluding the key column itself. If not found or df empty, return {}.
        Caches per (id(df), key_col) for speed; cache auto-refreshes when the df object changes.
        """

        # Basic guards
        if df is None or not hasattr(df, "columns") or key_col not in df.columns:
            return {}

        try:
            import polars as pl
        except Exception:
            pl = None

        if pl is not None and isinstance(df, pl.DataFrame):
            if df.height == 0:
                return {}
        else:
            # generic emptiness check
            try:
                if len(df) == 0:
                    return {}
            except Exception:
                pass

        # Cache setup
        cache = getattr(self, "_row_attr_cache", None)
        if cache is None:
            cache = {}
            self._row_attr_cache = cache

        cache_key = (id(df), key_col)
        mapping = cache.get(cache_key)

        # Build the mapping once per df object
        if mapping is None:
            mapping = {}
            # Latest write should win if duplicates exist (matches upsert semantics)
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(df, pl.DataFrame):
                for row in df.iter_rows(named=True):
                    kval = row.get(key_col)
                    if kval is None:
                        continue
                    d = dict(row)
                    d.pop(key_col, None)
                    mapping[kval] = d
            else:
                import narwhals as nw

                native = nw.to_native(nw.from_native(df, pass_through=True))
                # rows as dicts across backends
                to_dicts_fn = getattr(type(native), "to_dicts", None)

                if callable(to_dicts_fn):
                    rows = to_dicts_fn(native)
                else:
                    try:
                        rows = native.to_dict(orient="records")
                    except TypeError:
                        rows = native.to_dict()
                for row in rows:
                    kval = row.get(key_col)
                    if kval is None:
                        continue
                    d = dict(row)
                    d.pop(key_col, None)
                    mapping[kval] = d
            cache[cache_key] = mapping

        return mapping.get(key, {})

    def copy(self, history: bool = False):
        """
        Deep copy of the entire AnnNet.
        Fully structural + attribute fidelity.
        O(N) Python, O(nnz) matrix. ~100× faster than old version.

        Parameters
        ----------
        history : bool
            If True, copy the mutation history and snapshot timeline.
            If False, the new graph starts with a clean history.
        """

        # ---------------------------------------------------------------
        # 1) Construct empty graph with same capacity (fast path)
        # ---------------------------------------------------------------
        G = self.__class__
        new = G(directed=self.directed, n=self._num_entities, e=self._num_edges)

        # ---------------------------------------------------------------
        # 2) Clone incidence matrix (DOK → DOK copy is fast)
        # ---------------------------------------------------------------
        new._matrix = self._matrix.copy()

        # ---------------------------------------------------------------
        # 3) Clone entity/index mappings
        # ---------------------------------------------------------------
        new._num_entities = self._num_entities
        new.entity_to_idx = self.entity_to_idx.copy()
        new.idx_to_entity = self.idx_to_entity.copy()
        new.entity_types = self.entity_types.copy()

        # Vertex-layer presence
        new._V = self._V.copy()
        new._VM = self._VM.copy()
        new.vertex_aligned = self.vertex_aligned

        new._nl_to_row = self._nl_to_row.copy()
        new._row_to_nl = list(self._row_to_nl)

        # ---------------------------------------------------------------
        # 4) Clone edge/index mappings
        # ---------------------------------------------------------------
        new._num_edges = self._num_edges
        new.edge_to_idx = self.edge_to_idx.copy()
        new.idx_to_edge = self.idx_to_edge.copy()
        new.edge_definitions = {
            eid: (s, t, etype) for eid, (s, t, etype) in self.edge_definitions.items()
        }
        new.edge_weights = self.edge_weights.copy()
        new.edge_directed = self.edge_directed.copy()
        new.edge_kind = self.edge_kind.copy()
        new.edge_layers = self.edge_layers.copy()
        new._next_edge_id = self._next_edge_id
        new.edge_direction_policy = {k: v.copy() for k, v in self.edge_direction_policy.items()}

        # ---------------------------------------------------------------
        # 5) Clone slice structure (vertices, edges, attributes)
        # ---------------------------------------------------------------
        new._slices = {}
        for lid, meta in self._slices.items():
            new._slices[lid] = {
                "vertices": meta["vertices"].copy(),
                "edges": meta["edges"].copy(),
                "attributes": meta["attributes"].copy(),
            }

        new._default_slice = self._default_slice
        new._current_slice = self._current_slice

        # ---------------------------------------------------------------
        # 6) Clone slice_edge_weights
        # ---------------------------------------------------------------
        new.slice_edge_weights = {lid: m.copy() for lid, m in self.slice_edge_weights.items()}

        # ---------------------------------------------------------------
        # 7) Clone hyperedges
        # ---------------------------------------------------------------
        new.hyperedge_definitions = {
            eid: {k: (v.copy() if isinstance(v, (set, list, dict)) else v) for k, v in hdef.items()}
            for eid, hdef in self.hyperedge_definitions.items()
        }

        # ---------------------------------------------------------------
        # 8) Clone attribute tables (Polars DF → clone is fast / zero-copy)
        # ---------------------------------------------------------------
        new.vertex_attributes = self.vertex_attributes.clone()
        new.edge_attributes = self.edge_attributes.clone()
        new.slice_attributes = self.slice_attributes.clone()
        new.edge_slice_attributes = self.edge_slice_attributes.clone()
        new.layer_attributes = self.layer_attributes.clone()

        # ---------------------------------------------------------------
        # 9) Clone Kivela metadata
        # ---------------------------------------------------------------
        new.aspects = list(self.aspects)
        new.elem_layers = {k: list(v) for k, v in self.elem_layers.items()}
        new._all_layers = tuple(tuple(x) for x in self._all_layers)

        new._aspect_attrs = {a: m.copy() for a, m in self._aspect_attrs.items()}
        new._layer_attrs = {aa: m.copy() for aa, m in self._layer_attrs.items()}
        new._vertex_layer_attrs = {k: m.copy() for k, m in self._vertex_layer_attrs.items()}

        # ---------------------------------------------------------------
        # 10) Copy global graph attributes
        # ---------------------------------------------------------------
        new.graph_attributes = self.graph_attributes.copy()

        # ---------------------------------------------------------------
        # 11) History / versioning
        # ---------------------------------------------------------------
        if history:
            # Deep copy history, version, snapshots
            new._history_enabled = self._history_enabled
            new._history = [h.copy() for h in self._history]
            new._version = self._version
            new._snapshots = list(self._snapshots)
            # Reset the clock to avoid time regress
            new._history_clock0 = time.perf_counter_ns()
        else:
            # Reset to clean slate
            new._history_enabled = self._history_enabled
            new._history = []
            new._version = 0
            new._snapshots = []
            new._history_clock0 = time.perf_counter_ns()

        # ---------------------------------------------------------------
        # 12) Reinstall hooks (fresh)
        # ---------------------------------------------------------------
        new._install_history_hooks()

        return new

    def memory_usage(self):
        """Approximate total memory usage in bytes.

        Returns
        ---
        int
            Estimated bytes for the incidence matrix, dictionaries, and attribute DFs.

        """
        # Approximate matrix memory: each non-zero entry stores row, col, and value (4 bytes each)
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)
        # Estimate dict memory: ~100 bytes per entry
        dict_bytes = (
            len(self.entity_to_idx) + len(self.edge_to_idx) + len(self.edge_weights)
        ) * 100

        df_bytes = 0

        # vertex attributes
        va = self.vertex_attributes
        if va is not None:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(va, pl.DataFrame):
                df_bytes += va.estimated_size()
            else:
                # best-effort fallback
                try:
                    df_bytes += va.memory_usage(deep=True).sum()
                except Exception:
                    pass

        # Edge attributes
        ea = self.edge_attributes
        if ea is not None:
            try:
                import polars as pl
            except Exception:
                pl = None

            if pl is not None and isinstance(ea, pl.DataFrame):
                df_bytes += ea.estimated_size()
            else:
                try:
                    df_bytes += ea.memory_usage(deep=True).sum()
                except Exception:
                    pass
        return matrix_bytes + dict_bytes + df_bytes

    def get_vertex_incidence_matrix_as_lists(self, values: bool = False) -> dict:
        """Materialize the vertex–edge incidence structure as Python lists.

        Parameters
        --
        values : bool, optional (default=False)
            - If `False`, returns edge indices incident to each vertex.
            - If `True`, returns the **matrix values** (usually weights or 1/0) for
            each incident edge instead of the indices.

        Returns
        ---
        dict[str, list]
            A mapping from `vertex_id` - list of incident edges (indices or values),
            where:
            - Keys are vertex IDs.
            - Values are lists of edge indices (if `values=False`) or numeric values
            from the incidence matrix (if `values=True`).

        Notes
        -
        - Internally uses the sparse incidence matrix `self._matrix`, which is stored
        as a SciPy CSR (compressed sparse row) matrix or similar.
        - The incidence matrix `M` is defined as:
            - Rows: vertices
            - Columns: edges
            - Entry `M[i, j]` non-zero ⇨ vertex `i` is incident to edge `j`.
        - This is a convenient method when you want a native-Python structure for
        downstream use (e.g., exporting, iterating, or visualization).

        """
        result = {}
        csr = self._matrix.tocsr()
        for i in range(self._num_entities):
            vertex_id = self.idx_to_entity[i]
            row = csr.getrow(i)
            if values:
                result[vertex_id] = row.data.tolist()
            else:
                result[vertex_id] = row.indices.tolist()
        return result

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = False):
        """Return the vertex–edge incidence matrix in sparse or dense form.

        Parameters
        --
        values : bool, optional (default=False)
            If `True`, include the numeric values stored in the matrix
            (e.g., weights or signed incidence values). If `False`, convert the
            matrix to a binary mask (1 if incident, 0 if not).
        sparse : bool, optional (default=False)
            - If `True`, return the underlying sparse matrix (CSR).
            - If `False`, return a dense NumPy ndarray.

        Returns
        ---
        scipy.sparse.csr_matrix | numpy.ndarray
            The vertex–edge incidence matrix `M`:
            - Rows correspond to vertices.
            - Columns correspond to edges.
            - `M[i, j]` ≠ 0 indicates that vertex `i` is incident to edge `j`.

        Notes
        -
        - If `values=False`, the returned matrix is binarized before returning.
        - Use `sparse=True` for large graphs to avoid memory blowups.
        - This is the canonical low-level structure that most algorithms (e.g.,
        spectral clustering, Laplacian construction, hypergraph analytics) rely on.

        """
        M = self._matrix.tocsr()

        if not values:
            # Convert to binary mask
            M = M.copy()
            M.data[:] = 1

        if sparse:
            return M
        else:
            return M.toarray()

    def __hash__(self) -> int:
        """Return a stable hash representing the current graph structure and metadata.

        Returns
        ---
        int
            A hash value that uniquely (within high probability) identifies the graph
            based on its topology and attributes.

        Behavior

        - Includes the set of vertices, edges, and directedness in the hash.
        - Includes graph-level attributes (if any) to capture metadata changes.
        - Does **not** depend on memory addresses or internal object IDs, so the same
        graph serialized/deserialized or reconstructed with identical structure
        will produce the same hash.

        Notes
        -
        - This method enables `AnnNet` objects to be used in hash-based containers
        (like `set` or `dict` keys).
        - If the graph is **mutated** after hashing (e.g., vertices or edges are added
        or removed), the hash will no longer reflect the new state.
        - The method uses a deterministic representation: sorted vertex/edge sets
        ensure that ordering does not affect the hash.

        """
        # Core structural components
        vertex_ids = tuple(sorted(self.vertices()))
        edge_defs = []

        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            edge_defs.append((eid, tuple(sorted(S)), tuple(sorted(T)), directed))

        edge_defs = tuple(sorted(edge_defs))

        # Include high-level metadata if available
        graph_meta = (
            tuple(sorted(self.graph_attributes.items()))
            if hasattr(self, "graph_attributes")
            else ()
        )

        return hash((vertex_ids, edge_defs, graph_meta))
