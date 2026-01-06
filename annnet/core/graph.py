import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Optional, Union

import narwhals as nw
import numpy as np

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None
import scipy.sparse as sp

from ..algorithms.traversal import Traversal
from ._Annotation import AttributesClass
from ._BulkOps import BulkOps
from ._Cache import CacheManager, Operations
from ._helpers import (
    _EDGE_RESERVED,
    EdgeType,
    _df_filter_not_equal,
    _get_numeric_supertype,
    _slice_RESERVED,
    _vertex_RESERVED,
)
from ._History import GraphDiff, History
from ._Index import IndexManager, IndexMapping
from ._Layers import LayerClass, LayerManager
from ._Slices import SliceClass, SliceManager
from ._Views import GraphView, ViewsClass
from .lazy_proxies import _LazyGTProxy, _LazyIGProxy, _LazyNXProxy

# ===================================

class AnnNet(
    BulkOps,
    Operations,
    History,
    ViewsClass,
    IndexMapping,
    LayerClass,
    SliceClass,
    AttributesClass,
    Traversal,
):
    """Sparse incidence-matrix graph with slices, attributes, parallel edges, and hyperedges.

    The graph is backed by a DOK (Dictionary Of Keys) sparse matrix and exposes
    sliceed views and attribute tables stored as Polars DF (DataFrame). Supports:
    vertices, binary edges (directed/undirected), edge-entities (vertex-edge hybrids),
    k-ary hyperedges (directed/undirected), per-slice membership and weights,
    and Polars-backed attribute upserts.

    Parameters
    --
    directed : bool, optional
        Whether edges are directed by default. Individual edges can override this.

    Notes
    -
    - Incidence columns encode orientation: +w on source/head, −w on target/tail for
      directed edges; +w on all members for undirected edges/hyperedges.
    - Attributes are **pure**: structural keys are filtered out so attribute tables
      contain only user data.

    See Also

    add_vertex, add_edge, add_hyperedge, edges_view, vertices_view, slices_view

    """

    # Construction
    def __init__(
        self,
        directed=None,
        n: int = 0,
        e: int = 0,
        annotations=None,
        annotations_backend="polars",
        **kwargs,
    ):
        """Initialize an empty incidence-matrix graph.

        Parameters
        --
        directed : bool, optional
            Global default for edge directionality. Individual edges can override this.

        Notes
        -
        - Stores entities (vertices and edge-entities), edges (including parallels), and
        an incidence matrix in DOK (Dictionary Of Keys) sparse format.
        - Attribute tables are Polars DF (DataFrame) with canonical key columns:
        ``vertex_attributes(vertex_id)``, ``edge_attributes(edge_id)``,
        ``slice_attributes(slice_id)``, and
        ``edge_slice_attributes(slice_id, edge_id, weight)``.
        - A ``'default'`` slice is created and set active.

        """
        self.directed = directed

        self._vertex_RESERVED = set(_vertex_RESERVED)
        self._EDGE_RESERVED = set(_EDGE_RESERVED)
        self._slice_RESERVED = set(_slice_RESERVED)

        # Entity mappings (vertices + vertex-edge hybrids)
        self.entity_to_idx = {}  # entity_id -> row index
        self.idx_to_entity = {}  # row index -> entity_id
        self.entity_types = {}  # entity_id -> 'vertex' or 'edge'

        # Edge mappings (supports parallel edges)
        self.edge_to_idx = {}  # edge_id -> column index
        self.idx_to_edge = {}  # column index -> edge_id
        self.edge_definitions = {}  # edge_id -> (source, target, edge_type)
        self.edge_weights = {}  # edge_id -> weight
        self.edge_directed = {}  # Per-edge directedness; edge_id -> bool  (None = Mixed, True=directed, False=undirected)

        # flexible-direction behavior
        self.edge_direction_policy = {}  # eid -> policy dict
        # ensure 'flexible' isn’t stored as an attribute column
        self._EDGE_RESERVED.update({"flexible"})

        # Composite vertex key (tuple-of-attrs) support
        self._vertex_key_fields = None  # tuple[str,...] or None
        self._vertex_key_index = {}  # dict[tuple, vertex_id]

        # Sparse incidence matrix
        self._matrix = sp.dok_matrix((0, 0), dtype=np.float32)
        self._num_entities = 0
        self._num_edges = 0

        # Attribute storage using polars preferably
        self._annotations_backend = annotations_backend
        self._init_annotation_tables(annotations)
        self.edge_kind = {}
        self.hyperedge_definitions = {}
        self.graph_attributes = {}

        # Edge ID counter for parallel edges
        self._next_edge_id = 0

        # slice management - lightweight dict structure
        self._slices = {}  # slice_id -> {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_slice = None
        self._default_slice = "default"
        self.slice_edge_weights = defaultdict(dict)  # slice_id -> {edge_id: weight}

        # Initialize default slice
        self._slices[self._default_slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        self._current_slice = self._default_slice

        # counts stay logical (start empty)
        self._num_entities = 0
        self._num_edges = 0

        # pre-size the incidence matrix to capacity (no zeros allocated in DOK)
        n = int(n) if n and n > 0 else 0
        e = int(e) if e and e > 0 else 0
        self._matrix = sp.dok_matrix((n, e), dtype=np.float32)

        # grow-only helpers to avoid per-insert exact resizes
        def _grow_rows_to(target: int):
            rows, cols = self._matrix.shape
            if target > rows:
                # geometric bump; keeps behavior, reduces churn
                new_rows = max(target, rows + max(8, rows >> 1))
                self._matrix.resize((new_rows, cols))

        def _grow_cols_to(target: int):
            rows, cols = self._matrix.shape
            if target > cols:
                new_cols = max(target, cols + max(8, cols >> 1))
                self._matrix.resize((rows, new_cols))

        # bind as privates
        self._grow_rows_to = _grow_rows_to
        self._grow_cols_to = _grow_cols_to

        # History and Timeline
        self._history_enabled = True
        self._history = []  # list[dict]
        self._version = 0
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()  # wrap mutating methods
        self._snapshots = []

        # Multi-aspect definition
        self.aspects: list[str] = []  # e.g., ["time", "relation"]
        self.elem_layers: dict[str, list[str]] = {}  # aspect -> elementary labels
        self._all_layers: tuple[tuple[str, ...], ...] = ()  # cartesian product cache

        # vertex and vertex–layer presence
        self._V: set[str] = set()  # vertex ids (entities of type 'vertex')
        self._VM: set[tuple[str, tuple[str, ...]]] = set()  # {(u, aa_tuple)}
        self.vertex_aligned: bool = False  # if True, VM == V × all_layers

        # Stable indexing for supra rows
        self._nl_to_row: dict[tuple[str, tuple[str, ...]], int] = {}
        self._row_to_nl: list[tuple[str, tuple[str, ...]]] = []

        # Legacy 1-aspect shim: when aspects==1 we map layer_id "L" -> ("L",)
        self._legacy_single_aspect_enabled: bool = True

        # Multilayer edge bookkeeping (used by supra_adjacency)
        self.edge_kind = {}  # eid -> {"intra","inter","coupling"}
        self.edge_layers = {}  # eid -> aa   or -> (aa,bb) for inter/coupling

        # Aspect / layer / vertex–layer attribute tables (Kivela metadata)
        # All of this is annotation on top of the structural incidence.
        self._aspect_attrs = {}  # aspect_name -> {attr_name: value}
        self._layer_attrs = {}  # aa (tuple[str,...]) -> {attr_name: value}
        self._vertex_layer_attrs = {}  # (u, aa) -> {attr_name: value}

    def _init_annotation_tables(self, annotations):
        # 1) If user provided tables, keep them (we’ll wrap with Narwhals in ops)
        if annotations is not None:
            self.vertex_attributes = annotations.get("vertex_attributes")
            self.edge_attributes = annotations.get("edge_attributes")
            self.slice_attributes = annotations.get("slice_attributes")
            self.edge_slice_attributes = annotations.get("edge_slice_attributes")
            self.layer_attributes = annotations.get("layer_attributes")
            return

        # 2) Otherwise, create empties.
        if self._annotations_backend == "polars" and pl is not None:
            self.vertex_attributes = pl.DataFrame(schema={"vertex_id": pl.Utf8})
            self.edge_attributes = pl.DataFrame(schema={"edge_id": pl.Utf8})
            self.slice_attributes = pl.DataFrame(schema={"slice_id": pl.Utf8})
            self.edge_slice_attributes = pl.DataFrame(
                schema={"slice_id": pl.Utf8, "edge_id": pl.Utf8, "weight": pl.Float64}
            )
            self.layer_attributes = pl.DataFrame(schema={"layer_id": pl.Utf8})
            return

        # 3) No polars: need a fallback engine, or force user to pass tables.
        # Picked pandas fallback since it is common.
        try:
            import pandas as pd
        except Exception:
            raise RuntimeError(
                "Polars is not installed, and no annotation tables were provided. "
                "Install polars OR pass annotation tables (pandas/pyarrow/etc.) to AnnNet(..., annotations=...)."
            )

        self.vertex_attributes = pd.DataFrame({"vertex_id": pd.Series(dtype="string")})
        self.edge_attributes = pd.DataFrame({"edge_id": pd.Series(dtype="string")})
        self.slice_attributes = pd.DataFrame({"slice_id": pd.Series(dtype="string")})
        self.edge_slice_attributes = pd.DataFrame(
            {
                "slice_id": pd.Series(dtype="string"),
                "edge_id": pd.Series(dtype="string"),
                "weight": pd.Series(dtype="float64"),
            }
        )
        self.layer_attributes = pd.DataFrame({"layer_id": pd.Series(dtype="string")})

    # Build graph

    def add_vertex(self, vertex_id, slice=None, **attributes):
        """Add (or upsert) a vertex and optionally attach it to a slice.

        Parameters
        --
        vertex_id : str
            vertex ID (must be unique across entities).
        slice : str, optional
            Target slice. Defaults to the active slice.
        **attributes
            Pure vertex attributes to store.

        Returns
        ---
        str
            The vertex ID (echoed).

        Notes
        -
        - Ensures a row exists in the Polars DF [DataFrame] for attributes.
        - Resizes the incidence matrix if needed.

        """
        # Fast normalize to cut hashing/dup costs in dicts.
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
            if slice is None:
                slice = self._current_slice
            elif isinstance(slice, str):
                slice = _sys.intern(slice)
        except Exception:
            slice = slice or self._current_slice

        entity_to_idx = self.entity_to_idx
        idx_to_entity = self.idx_to_entity
        entity_types = self.entity_types
        M = self._matrix  # DOK

        # Add to global superset if new
        if vertex_id not in entity_to_idx:
            idx = self._num_entities
            entity_to_idx[vertex_id] = idx
            idx_to_entity[idx] = vertex_id
            entity_types[vertex_id] = "vertex"
            self._num_entities = idx + 1

            rows, cols = M.shape
            if self._num_entities > rows:
                # geometric growth (≈1.5x), minimum step 8 to avoid frequent resizes
                new_rows = max(self._num_entities, rows + max(8, rows >> 1))
                M.resize((new_rows, cols))

        # Add to specified slice (create if needed)
        slices = self._slices
        if slice not in slices:
            slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
        slices[slice]["vertices"].add(vertex_id)

        # Multislice presence sync:
        # - Track V (true vertices only)
        # - If vertex_aligned: ensure presence across all slices
        # - Else, if we are in 1-aspect shim mode and a slice was given, add (u, (slice,))
        if self.entity_types.get(vertex_id) == "vertex":
            self._V.add(vertex_id)
            if self.vertex_aligned and self._all_slices:
                for aa in self._all_slices:
                    self._VM.add((vertex_id, aa))
            elif (
                slice is not None and len(self.aspects) == 1 and self._legacy_single_aspect_enabled
            ):
                self._VM.add((vertex_id, (slice,)))

        # Ensure vertex_attributes has a row for this vertex (even with no attrs)
        self._ensure_vertex_table()
        self._ensure_vertex_row(vertex_id)

        # Upsert passed attributes (if any)
        if attributes:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vertex_id, attributes)

        return vertex_id

    def add_vertices(self, vertices, slice=None, **attributes):
        # normalize to [(vertex_id, per_attrs), ...]
        it = []
        for item in vertices:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                it.append({"vertex_id": item[0], **item[1], **attributes})
            elif isinstance(item, dict):
                d = dict(item)
                d.update(attributes)
                it.append(d)
            else:
                it.append({"vertex_id": item, **attributes})
        self.add_vertices_bulk(
            [(d["vertex_id"], {k: v for k, v in d.items() if k != "vertex_id"}) for d in it],
            slice=slice,
        )
        return [d["vertex_id"] for d in it]

    def add_edge_entity(self, edge_entity_id, slice=None, **attributes):
        """DEPRECATED: Use add_edge(..., as_entity=True) instead."""
        self._register_edge_as_entity(edge_entity_id)
        slice = slice or self._current_slice
        if slice is not None:
            if slice not in self._slices:
                self._slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
            self._slices[slice]["edges"].add(edge_entity_id)
        if attributes:
            self.set_edge_attrs(edge_entity_id, **attributes)
        return edge_entity_id

    def _register_edge_as_entity(self, edge_id):
        """Make an existing edge connectable as an endpoint."""
        if edge_id in self.entity_to_idx:
            return
        idx = self._num_entities
        self.entity_to_idx[edge_id] = idx
        self.idx_to_entity[idx] = edge_id
        self.entity_types[edge_id] = "edge"
        self._num_entities = idx + 1
        M = self._matrix
        rows, cols = M.shape
        if idx >= rows:
            new_rows = max(idx + 1, rows + max(8, rows >> 1))
            M.resize((new_rows, cols))

    def add_edge(
        self,
        source,
        target,
        slice=None,
        weight=1.0,
        edge_id=None,
        as_entity=False,
        propagate="none",
        slice_weight=None,
        directed=None,
        edge_directed=None,
        **attributes,
    ):
        """Add or update a binary edge between two entities.

        Parameters
        --
        source : str
            Source entity ID (vertex or edge).
        target : str
            Target entity ID (vertex or edge).
        slice : str, optional
            slice to place the edge into. Defaults to the active slice.
        weight : float, optional
            Global edge weight stored in the incidence column (default 1.0).
        edge_id : str, optional
            Explicit edge ID. If omitted, a fresh ID is generated.
        as_entity : bool, optional
            If True, this edge can itself be an endpoint of other edges.
        propagate : {'none', 'shared', 'all'}, optional
            slice propagation:
            - ``'none'`` : only the specified slice
            - ``'shared'`` : all slices that already contain **both** endpoints
            - ``'all'`` : all slices that contain **either** endpoint (and add the other)
        slice_weight : float, optional
            Per-slice weight override for this edge (stored in edge-slice DF).
        edge_directed : bool, optional
            Override default directedness for this edge. If None, uses graph default.
        **attributes
            Pure edge attributes to upsert.

        Returns
        ---
        str
            The edge ID (new or updated).

        Raises
        --
        ValueError
            If ``propagate`` is invalid.
        TypeError
            If ``weight`` is not numeric.

        Notes
        -
        - Directed edges write ``+weight`` at source row and ``-weight`` at target row.
        - Undirected edges write ``+weight`` at both endpoints.
        - When ``as_entity=True``, the edge gets a row so other edges can connect to it.

        """
        edge_type = attributes.pop("edge_type", "regular")

        # Resolve dict endpoints via composite key (if enabled)
        if self._vertex_key_enabled():
            if isinstance(source, dict):
                source = self.get_or_create_vertex_by_attrs(slice=slice, **source)
            if isinstance(target, dict):
                target = self.get_or_create_vertex_by_attrs(slice=slice, **target)

        flexible = attributes.pop("flexible", None)
        if flexible is not None:
            if (
                not isinstance(flexible, dict)
                or "var" not in flexible
                or "threshold" not in flexible
            ):
                raise ValueError(
                    "flexible must be a dict with keys {'var','threshold'[,'scope','above','tie']}"
                )
            tie = flexible.get("tie", "keep")
            if tie not in {"keep", "undirected", "s->t", "t->s"}:
                raise ValueError(
                    "flexible['tie'] must be one of {'keep','undirected','s->t','t->s'}"
                )

        # normalize endpoints: accept str OR iterable; route hyperedges
        def _to_tuple(x):
            if isinstance(x, (str, bytes)):
                return (x,), False
            try:
                xs = tuple(x)
            except TypeError:
                return (x,), False
            return xs, (len(xs) != 1)

        S, src_multi = _to_tuple(source)
        T, tgt_multi = _to_tuple(target)

        # Hyperedge delegation
        if src_multi or tgt_multi:
            if edge_directed:
                return self.add_hyperedge(
                    head=S,
                    tail=T,
                    edge_directed=True,
                    slice=slice,
                    weight=weight,
                    edge_id=edge_id,
                    **attributes,
                )
            else:
                members = tuple(set(S) | set(T))
                return self.add_hyperedge(
                    members=members,
                    edge_directed=False,
                    slice=slice,
                    weight=weight,
                    edge_id=edge_id,
                    **attributes,
                )

        # Binary case: unwrap singletons to plain IDs
        source, target = S[0], T[0]

        # validate inputs
        if propagate not in {"none", "shared", "all"}:
            raise ValueError(f"propagate must be one of 'none'|'shared'|'all', got {propagate!r}")
        if not isinstance(weight, (int, float)):
            raise TypeError(f"weight must be numeric, got {type(weight).__name__}")

        # resolve slice + whether to touch sliceing at all
        slice = self._current_slice if slice is None else slice
        touch_slice = slice is not None

        entity_to_idx = self.entity_to_idx
        idx_to_edge = self.idx_to_edge
        edge_to_idx = self.edge_to_idx
        edge_defs = self.edge_definitions
        edge_w = self.edge_weights
        edge_dir = self.edge_directed
        slices = self._slices
        M = self._matrix  # DOK

        # ensure endpoints exist (creates vertex if not already registered)
        for ep in (source, target):
            if ep not in entity_to_idx:
                self.add_vertex(ep, slice=slice)

        # indices (after potential vertex creation)
        source_idx = entity_to_idx[source]
        target_idx = entity_to_idx[target]

        # edge id
        if edge_id is None:
            edge_id = self._get_next_edge_id()

        # determine direction
        if edge_directed is not None:
            is_dir = bool(edge_directed)
        elif self.directed is not None:
            is_dir = self.directed
        else:
            is_dir = True

        if edge_id in edge_to_idx:
            # UPDATE existing column

            col_idx = edge_to_idx[edge_id]

            # allow explicit direction change; otherwise keep existing
            if edge_directed is None:
                is_dir = edge_dir.get(edge_id, is_dir)
            edge_dir[edge_id] = is_dir

            # keep edge_type attr write
            self.set_edge_attrs(
                edge_id, edge_type=(EdgeType.DIRECTED if is_dir else EdgeType.UNDIRECTED)
            )

            # if source/target changed, update definition
            old_src, old_tgt, old_type = edge_defs[edge_id]
            edge_defs[edge_id] = (source, target, old_type)  # keep old_type by default

            # ensure matrix has enough rows (in case vertices were added since creation)
            self._grow_rows_to(self._num_entities)

            # clear only the cells that were previously set, not the whole column
            try:
                old_src_idx = entity_to_idx[old_src]
                M[old_src_idx, col_idx] = 0
            except KeyError:
                pass
            if old_src != old_tgt:
                try:
                    old_tgt_idx = entity_to_idx[old_tgt]
                    M[old_tgt_idx, col_idx] = 0
                except KeyError:
                    pass

            # write new endpoints
            M[source_idx, col_idx] = weight
            if source != target:
                M[target_idx, col_idx] = -weight if is_dir else weight

            edge_w[edge_id] = weight

        else:
            # CREATE new column

            col_idx = self._num_edges
            edge_to_idx[edge_id] = col_idx
            idx_to_edge[col_idx] = edge_id
            edge_defs[edge_id] = (source, target, edge_type)
            edge_w[edge_id] = weight
            edge_dir[edge_id] = is_dir
            self._num_edges = col_idx + 1

            # grow-only to current logical capacity
            self._grow_rows_to(self._num_entities)
            self._grow_cols_to(self._num_edges)
            M[source_idx, col_idx] = weight
            if source != target:
                M[target_idx, col_idx] = -weight if is_dir else weight

        # slice handling
        if touch_slice:
            if slice not in slices:
                slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
            slices[slice]["edges"].add(edge_id)
            slices[slice]["vertices"].update((source, target))

            if slice_weight is not None:
                w = float(slice_weight)
                self.set_edge_slice_attrs(slice, edge_id, weight=w)
                self.slice_edge_weights.setdefault(slice, {})[edge_id] = w

        # propagation
        if propagate == "shared":
            self._propagate_to_shared_slices(edge_id, source, target)
        elif propagate == "all":
            self._propagate_to_all_slices(edge_id, source, target)

        if flexible is not None:
            self.edge_directed[edge_id] = True  # always directed; orientation is controlled
            self.edge_direction_policy[edge_id] = flexible

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        if flexible is not None:
            self._apply_flexible_direction(edge_id)

        if as_entity:
            self._register_edge_as_entity(edge_id)

        return edge_id

    def add_parallel_edge(self, source, target, weight=1.0, **attributes):
        """Add a parallel edge (same endpoints, different ID).

        Parameters
        --
        source : str
        target : str
        weight : float, optional
        **attributes
            Pure edge attributes.

        Returns
        ---
        str
            The new edge ID.

        """

        _add_edge = self.add_edge
        return _add_edge(source, target, weight=weight, edge_id=None, **attributes)

    def add_hyperedge(
        self,
        *,
        members=None,
        head=None,
        tail=None,
        slice=None,
        weight=1.0,
        edge_id=None,
        edge_directed=None,  # bool or None (None -> infer from params)
        **attributes,
    ):
        """Create a k-ary hyperedge as a single incidence column.

        Modes
        -
        - **Undirected**: pass ``members`` (>=2). Each member gets ``+weight``.
        - **Directed**: pass ``head`` and ``tail`` (both non-empty, disjoint).
        Head gets ``+weight``; tail gets ``-weight``.
        """
        # Map dict endpoints to vertex_id when composite keys are enabled
        if self._vertex_key_enabled():

            def _map(x):
                return (
                    self.get_or_create_vertex_by_attrs(slice=slice, **x)
                    if isinstance(x, dict)
                    else x
                )

            if members is not None:
                members = [_map(u) for u in members]
            else:
                head = [_map(u) for u in head]
                tail = [_map(v) for v in tail]

        # validate form
        if members is None and (head is None or tail is None):
            raise ValueError("Provide members (undirected) OR head+tail (directed).")
        if members is not None and (head is not None or tail is not None):
            raise ValueError("Use either members OR head+tail, not both.")

        if members is not None:
            members = list(members)
            if len(members) < 2:
                raise ValueError("Hyperedge needs >=2 members.")
            directed = False if edge_directed is None else bool(edge_directed)
            if directed:
                raise ValueError("Directed=True requires head+tail, not members.")
        else:
            head = list(head)
            tail = list(tail)
            if not head or not tail:
                raise ValueError("Directed hyperedge needs non-empty head and tail.")
            if set(head) & set(tail):
                raise ValueError("head and tail must be disjoint.")
            directed = True if edge_directed is None else bool(edge_directed)
            if not directed:
                raise ValueError("Undirected=False conflicts with head/tail.")

        # set slice
        slice = self._current_slice if slice is None else slice

        # Intern frequently-used strings for cheaper dict ops
        try:
            import sys as _sys

            if isinstance(slice, str):
                slice = _sys.intern(slice)
            if isinstance(edge_id, str):
                edge_id = _sys.intern(edge_id)
            if members is not None:
                members = [_sys.intern(u) if isinstance(u, str) else u for u in members]
            else:
                head = [_sys.intern(u) if isinstance(u, str) else u for u in head]
                tail = [_sys.intern(v) if isinstance(v, str) else v for v in tail]
        except Exception:
            pass

        # locals for hot paths
        entity_to_idx = self.entity_to_idx
        slices = self._slices
        M = self._matrix  # DOK

        # ensure participants exist globally
        def _ensure_entity(x):
            if x in entity_to_idx:
                return
            if (
                isinstance(x, str)
                and x.startswith("edge_")
                and x in self.entity_types
                and self.entity_types[x] == "edge"
            ):
                return
            self.add_vertex(x, slice=slice)

        if members is not None:
            for u in members:
                _ensure_entity(u)
        else:
            for u in head:
                _ensure_entity(u)
            for v in tail:
                _ensure_entity(v)

        # allocate edge id + column
        if edge_id is None:
            edge_id = self._get_next_edge_id()

        is_new = edge_id not in self.edge_to_idx
        if is_new:
            col_idx = self._num_edges
            self.edge_to_idx[edge_id] = col_idx
            self.idx_to_edge[col_idx] = edge_id
            self._num_edges += 1
            self._grow_rows_to(self._num_entities)
            self._grow_cols_to(self._num_edges)
        else:
            col_idx = self.edge_to_idx[edge_id]
            # clear: delete only previously set cells instead of zeroing whole column
            # handle prior hyperedge or binary edge reuse
            prev_h = self.hyperedge_definitions.get(edge_id)
            if prev_h is not None:
                if prev_h.get("directed", False):
                    rows_to_clear = prev_h["head"] | prev_h["tail"]
                else:
                    rows_to_clear = prev_h["members"]
                for vid in rows_to_clear:
                    try:
                        M[entity_to_idx[vid], col_idx] = 0
                    except KeyError:
                        # vertex may not exist anymore; ignore
                        pass
            else:
                # maybe it was a binary edge before
                prev = self.edge_definitions.get(edge_id)
                if prev is not None:
                    src, tgt, _ = prev
                    if src is not None:
                        try:
                            M[entity_to_idx[src], col_idx] = 0
                        except KeyError:
                            pass
                    if tgt is not None and tgt != src:
                        try:
                            M[entity_to_idx[tgt], col_idx] = 0
                        except KeyError:
                            pass

        self._grow_rows_to(self._num_entities)

        # write column entries
        w = float(weight)
        if members is not None:
            # undirected: +w at each member
            for u in members:
                M[entity_to_idx[u], col_idx] = w
            self.hyperedge_definitions[edge_id] = {
                "directed": False,
                "members": set(members),
            }
        else:
            # directed: +w on head, -w on tail
            for u in head:
                M[entity_to_idx[u], col_idx] = w
            mw = -w
            for v in tail:
                M[entity_to_idx[v], col_idx] = mw
            self.hyperedge_definitions[edge_id] = {
                "directed": True,
                "head": set(head),
                "tail": set(tail),
            }

        # bookkeeping shared with binary edges
        self.edge_weights[edge_id] = w
        self.edge_directed[edge_id] = bool(directed)
        self.edge_kind[edge_id] = "hyper"
        # keep a sentinel in edge_definitions so old code won't crash
        self.edge_definitions[edge_id] = (None, None, "hyper")

        # slice membership + per-slice vertices
        if slice is not None:
            if slice not in slices:
                slices[slice] = {"vertices": set(), "edges": set(), "attributes": {}}
            slices[slice]["edges"].add(edge_id)
            if members is not None:
                slices[slice]["vertices"].update(members)
            else:
                slices[slice]["vertices"].update(self.hyperedge_definitions[edge_id]["head"])
                slices[slice]["vertices"].update(self.hyperedge_definitions[edge_id]["tail"])

        # attributes
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)

        return edge_id

    def set_hyperedge_coeffs(self, edge_id: str, coeffs: dict[str, float]) -> None:
        """Write per-vertex coefficients into the incidence column (DOK [dictionary of keys])."""
        col = self.edge_to_idx[edge_id]
        for vid, coeff in coeffs.items():
            row = self.entity_to_idx[vid]
            self._matrix[row, col] = float(coeff)

    def add_edge_to_slice(self, lid, eid):
        """Attach an existing edge to a slice (no weight changes).

        Parameters
        --
        lid : str
            slice ID.
        eid : str
            Edge ID.

        Raises
        --
        KeyError
            If the slice does not exist.

        """
        if lid not in self._slices:
            raise KeyError(f"slice {lid} does not exist")
        self._slices[lid]["edges"].add(eid)

    def _propagate_to_shared_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to all slices that already contain **both** endpoints.

        Parameters
        --
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data["vertices"] and target in slice_data["vertices"]:
                slice_data["edges"].add(edge_id)

    def _propagate_to_all_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to any slice containing **either** endpoint and
        insert the missing endpoint into that slice.

        Parameters
        --
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data["vertices"] or target in slice_data["vertices"]:
                slice_data["edges"].add(edge_id)
                # Only add missing endpoint if both vertices should be in slice
                if source in slice_data["vertices"]:
                    slice_data["vertices"].add(target)
                if target in slice_data["vertices"]:
                    slice_data["vertices"].add(source)

    def _normalize_vertices_arg(self, vertices):
        """Normalize a single vertex or an iterable of vertices into a set.

        This internal utility function standardizes input for methods like
        `in_edges()` and `out_edges()` by converting the argument into a set
        of vertex identifiers.

        Parameters
        --
        vertices : str | Iterable[str] | None
            - A single vertex ID (string).
            - An iterable of vertex IDs (e.g., list, tuple, set).
            - `None` is allowed and will return an empty set.

        Returns
        ---
        set[str]
            A set of vertex identifiers. If `vertices` is `None`, returns an
            empty set. If a single vertex is provided, returns a one-element set.

        Notes
        -
        - Strings are treated as **single vertex IDs**, not iterables.
        - If the argument is neither iterable nor a string, it is wrapped in a set.
        - Used internally by API methods that accept flexible vertex arguments.

        """
        if vertices is None:
            return set()
        if isinstance(vertices, (str, bytes)):
            return {vertices}
        try:
            return set(vertices)
        except TypeError:
            return {vertices}

    def make_undirected(self, *, drop_flexible: bool = True, update_default: bool = True):
        """
        Force the whole graph to be undirected in-place.

        - Binary edges: rewrite incidence to (+w, +w) at (source, target).
        - Hyperedges: rewrite incidence to undirected (+w on all members).
        - Keeps weights, endpoints, slices, layer roles and attributes.
        """

        M = self._matrix
        entity_to_idx = self.entity_to_idx

        # 1) Binary edges (skip hyperedge sentinel rows)
        for eid, (src, tgt, kind) in list(self.edge_definitions.items()):
            if kind == "hyper":
                # handled via hyperedge_definitions below
                continue
            if src is None or tgt is None:
                continue  # just in case

            col = self.edge_to_idx.get(eid)
            if col is None:
                continue

            w = float(self.edge_weights.get(eid, 1.0))

            # Write +w at both endpoints
            si = entity_to_idx.get(src)
            ti = entity_to_idx.get(tgt)

            if si is not None:
                M[si, col] = w
            if ti is not None and ti != si:
                M[ti, col] = w

            # Mark metadata as undirected
            self.edge_directed[eid] = False
            try:
                self.set_edge_attrs(eid, edge_type=EdgeType.UNDIRECTED)
            except Exception:
                pass  # attribute table might be missing in some minimal setups

        # 2) Hyperedges
        for eid, h in list(self.hyperedge_definitions.items()):
            col = self.edge_to_idx.get(eid)
            if col is None:
                continue

            w = float(self.edge_weights.get(eid, 1.0))

            if h.get("directed", False):
                head = set(h.get("head", ()))
                tail = set(h.get("tail", ()))
                members = head | tail

                # Clear old (+w on head, -w on tail)
                for u in members:
                    idx = entity_to_idx.get(u)
                    if idx is not None:
                        try:
                            M[idx, col] = 0
                        except KeyError:
                            pass
            else:
                members = set(h.get("members", ()))

            # Rewrite as undirected: +w for all members
            for u in members:
                idx = entity_to_idx.get(u)
                if idx is not None:
                    M[idx, col] = w

            # Update hyperedge metadata
            self.hyperedge_definitions[eid] = {
                "directed": False,
                "members": set(members),
            }
            self.edge_directed[eid] = False

        # 3) Optional: drop flexible-direction policies (they make no sense if everything is undirected)
        if drop_flexible:
            self.edge_direction_policy.clear()

        # 4) Optional: set global default to undirected for future edges
        if update_default:
            self.directed = False

    # Remove / mutate down

    def remove_edge(self, edge_id):
        """Remove an edge (binary or hyperedge) from the graph.

        Parameters
        --
        edge_id : str

        Raises
        --
        KeyError
            If the edge is not found.

        Notes
        -
        - Physically removes the incidence column (no CSR round-trip).
        - Cleans edge attributes, slice memberships, and per-slice entries.

        """
        if edge_id not in self.edge_to_idx:
            raise KeyError(f"Edge {edge_id} not found")

        col_idx = self.edge_to_idx[edge_id]

        # column removal without CSR (single pass over nonzeros)
        M_old = self._matrix
        rows, cols = M_old.shape
        new_cols = cols - 1
        # Rebuild DOK with columns > col_idx shifted left by 1
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if c == col_idx:
                continue  # drop this column
            elif c > col_idx:
                M_new[r, c - 1] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new

        # mappings (preserve relative order of remaining edges)
        # Remove the deleted edge id
        del self.edge_to_idx[edge_id]
        # Shift indices for edges after the removed column
        for old_idx in range(col_idx + 1, self._num_edges):
            eid = self.idx_to_edge.pop(old_idx)
            self.idx_to_edge[old_idx - 1] = eid
            self.edge_to_idx[eid] = old_idx - 1
        # Drop the last stale entry (now shifted)
        self.idx_to_edge.pop(self._num_edges - 1, None)
        self._num_edges -= 1

        # Metadata cleanup
        # Edge definitions / weights / directedness
        self.edge_definitions.pop(edge_id, None)
        self.edge_weights.pop(edge_id, None)
        if edge_id in self.edge_directed:
            self.edge_directed.pop(edge_id, None)

        # Remove from edge attributes
        ea = self.edge_attributes
        if ea is not None and hasattr(ea, "columns"):
            cols = list(ea.columns)
            # generic emptiness check
            is_empty = (getattr(ea, "height", None) == 0) or (
                hasattr(ea, "__len__") and len(ea) == 0
            )
            if (not is_empty) and ("edge_id" in cols):
                self.edge_attributes = _df_filter_not_equal(ea, "edge_id", edge_id)

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data["edges"].discard(edge_id)

        # Remove from edge-slice attributes
        esa = self.edge_slice_attributes
        if esa is not None and hasattr(esa, "columns"):
            cols = list(esa.columns)
            is_empty = (getattr(esa, "height", None) == 0) or (
                hasattr(esa, "__len__") and len(esa) == 0
            )
            if (not is_empty) and ("edge_id" in cols):
                self.edge_slice_attributes = _df_filter_not_equal(esa, "edge_id", edge_id)

        # Legacy / auxiliary dicts
        for d in self.slice_edge_weights.values():
            d.pop(edge_id, None)

        self.edge_kind.pop(edge_id, None)
        self.hyperedge_definitions.pop(edge_id, None)

    def remove_vertex(self, vertex_id):
        """Remove a vertex and all incident edges (binary + hyperedges).

        Parameters
        --
        vertex_id : str

        Raises
        --
        KeyError
            If the vertex is not found.

        Notes
        -
        - Rebuilds entity indexing and shrinks the incidence matrix accordingly.

        """
        if vertex_id not in self.entity_to_idx:
            raise KeyError(f"vertex {vertex_id} not found")

        entity_idx = self.entity_to_idx[vertex_id]

        # Collect incident edges (set to avoid duplicates)
        edges_to_remove = set()

        # Binary edges: edge_definitions {eid: (source, target, ...)}
        for eid, edef in list(self.edge_definitions.items()):
            try:
                source, target = edef[0], edef[1]
            except Exception:
                source, target = edef.get("source"), edef.get("target")
            if source == vertex_id or target == vertex_id:
                edges_to_remove.add(eid)

        # Hyperedges: hyperedge_definitions {eid: {"head":[...], "tail":[...]}} or {"members":[...]}
        def _vertex_in_hyperdef(hdef: dict, vertex: str) -> bool:
            # Common keys first
            for key in ("head", "tail", "members", "vertices", "vertices"):
                seq = hdef.get(key)
                if isinstance(seq, (list, tuple, set)) and vertex in seq:
                    return True
            # Safety net: scan any list/tuple/set values
            for v in hdef.values():
                if isinstance(v, (list, tuple, set)) and vertex in v:
                    return True
            return False

        hdefs = getattr(self, "hyperedge_definitions", {})
        if isinstance(hdefs, dict):
            for heid, hdef in list(hdefs.items()):
                if isinstance(hdef, dict) and _vertex_in_hyperdef(hdef, vertex_id):
                    edges_to_remove.add(heid)

        # Remove all collected edges
        for eid in edges_to_remove:
            self.remove_edge(eid)

        # row removal without CSR: rebuild DOK with rows-1 and shift indices
        M_old = self._matrix
        rows, cols = M_old.shape
        new_rows = rows - 1
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if r == entity_idx:
                continue  # drop this row
            elif r > entity_idx:
                M_new[r - 1, c] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new

        # Update entity mappings
        del self.entity_to_idx[vertex_id]
        del self.entity_types[vertex_id]

        # Shift indices for entities after the removed row; preserve relative order
        for old_idx in range(entity_idx + 1, self._num_entities):
            ent_id = self.idx_to_entity.pop(old_idx)
            self.idx_to_entity[old_idx - 1] = ent_id
            self.entity_to_idx[ent_id] = old_idx - 1
        # Drop last stale entry and shrink count
        self.idx_to_entity.pop(self._num_entities - 1, None)
        self._num_entities -= 1

        # Remove from vertex attributes
        va = self.vertex_attributes
        if va is not None and hasattr(va, "columns"):
            cols = list(va.columns)
            is_empty = (getattr(va, "height", None) == 0) or (
                hasattr(va, "__len__") and len(va) == 0
            )
            if (not is_empty) and ("vertex_id" in cols):
                self.vertex_attributes = _df_filter_not_equal(va, "vertex_id", vertex_id)

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data["vertices"].discard(vertex_id)

    def remove_slice(self, slice_id):
        """Remove a non-default slice and its per-slice attributes.

        Parameters
        --
        slice_id : str

        Raises
        --
        ValueError
            If attempting to remove the internal default slice.
        KeyError
            If the slice does not exist.

        Notes
        -
        - Does not delete vertices/edges globally; only membership and slice metadata.

        """
        if slice_id == self._default_slice:
            raise ValueError("Cannot remove default slice")
        if slice_id not in self._slices:
            raise KeyError(f"slice {slice_id} not found")

        # Purge per-slice attributes
        ela = getattr(self, "edge_slice_attributes", None)
        if ela is not None and hasattr(ela, "columns"):
            cols = list(ela.columns)
            is_empty = (getattr(ela, "height", None) == 0) or (
                hasattr(ela, "__len__") and len(ela) == 0
            )
            if (not is_empty) and ("slice_id" in cols):
                self.edge_slice_attributes = _df_filter_not_equal(ela, "slice_id", slice_id)

        # Drop legacy dict slice if present
        if isinstance(getattr(self, "slice_edge_weights", None), dict):
            self.slice_edge_weights.pop(slice_id, None)

        # Remove the slice and reset current if needed
        del self._slices[slice_id]
        if self._current_slice == slice_id:
            self._current_slice = self._default_slice

    # Basic queries & metrics

    def get_vertex(self, index: int) -> str:
        """Return the vertex ID corresponding to a given internal index.

        Parameters
        --
        index : int
            The internal vertex index.

        Returns
        ---
        str
            The vertex ID.

        """
        return self.idx_to_entity[index]

    def get_edge(self, index: int):
        """Return edge endpoints in a canonical form.

        Parameters
        --
        index : int
            Internal edge index.

        Returns
        ---
        tuple[frozenset, frozenset]
            (S, T) where S and T are frozensets of vertex IDs.
            - For directed binary edges: ({u}, {v})
            - For undirected binary edges: (M, M)
            - For directed hyperedges: (head_set, tail_set)
            - For undirected hyperedges: (members, members)

        """
        if isinstance(index, str):
            eid = index
            try:
                index = self.edge_to_idx[eid]
            except KeyError:
                raise KeyError(f"Unknown edge id: {eid}") from None
        else:
            eid = self.idx_to_edge[index]

        kind = self.edge_kind.get(eid)

        eid = self.idx_to_edge[index]
        kind = self.edge_kind.get(eid)

        if kind == "hyper":
            meta = self.hyperedge_definitions[eid]
            if meta.get("directed", False):
                return (frozenset(meta["head"]), frozenset(meta["tail"]))
            else:
                M = frozenset(meta["members"])
                return (M, M)
        else:
            u, v, _etype = self.edge_definitions[eid]
            directed = self.edge_directed.get(eid, True if self.directed is None else self.directed)
            if directed:
                return (frozenset([u]), frozenset([v]))
            else:
                M = frozenset([u, v])
                return (M, M)

    def incident_edges(self, vertex_id) -> list[int]:
        """Return all edge indices incident to a given vertex.

        Parameters
        --
        vertex_id : str
            vertex identifier.

        Returns
        ---
        list[int]
            List of edge indices incident to the vertex.

        """
        incident = []
        # Fast path: direct matrix row lookup if available
        if vertex_id in self.entity_to_idx:
            row_idx = self.entity_to_idx[vertex_id]
            try:
                incident.extend(self._matrix.tocsr().getrow(row_idx).indices.tolist())
                return incident
            except Exception:
                # fallback if matrix is not in CSR (compressed sparse row) format
                pass

        # Fallback: scan edge definitions
        for j in range(self.number_of_edges()):
            eid = self.idx_to_edge[j]
            kind = self.edge_kind.get(eid)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if (
                    meta.get("directed", False)
                    and (vertex_id in meta["head"] or vertex_id in meta["tail"])
                ) or (not meta.get("directed", False) and vertex_id in meta["members"]):
                    incident.append(j)
            else:
                u, v, _etype = self.edge_definitions[eid]
                if vertex_id == u or vertex_id == v:
                    incident.append(j)

        return incident

    def _is_directed_edge(self, edge_id):
        """Check if an edge is directed (per-edge flag overrides graph default).

        Parameters
        --
        edge_id : str

        Returns
        ---
        bool

        """
        return bool(self.edge_directed.get(edge_id, self.directed))

    def has_edge(
        self,
        source: str | None = None,
        target: str | None = None,
        edge_id: str | None = None,
    ) -> bool | tuple[bool, list[str]]:
        """
        Edge existence check with three modes.

        Modes
        -----
        1) edge_id only:
            has_edge(edge_id="e1")
            -> bool    (True if e1 exists anywhere)

        2) source + target only:
            has_edge(source="u", target="v")
            -> (bool, [edge_ids...])
               bool = True if at least one edge u->v exists

        3) source + target + edge_id:
            has_edge(source="u", target="v", edge_id="e1")
            -> bool    (True only if e1 exists AND is u->v)

        Any other combination is invalid and raises ValueError.
        """

        # ---- Mode 1: edge_id only ----
        if edge_id is not None and source is None and target is None:
            return edge_id in self.edge_definitions

        # ---- Mode 2: source + target only ----
        if edge_id is None and source is not None and target is not None:
            eids: list[str] = []
            for eid, (src, tgt, _) in self.edge_definitions.items():
                if src == source and tgt == target:
                    eids.append(eid)
            return (len(eids) > 0, eids)

        # ---- Mode 3: edge_id + source + target ----
        if edge_id is not None and source is not None and target is not None:
            data = self.edge_definitions.get(edge_id)
            if data is None:
                return False
            src, tgt, _ = data
            return src == source and tgt == target

        # ---- Anything else is ambiguous / invalid ----
        raise ValueError(
            "Invalid argument combination: use either "
            "(edge_id), (source,target), or (source,target,edge_id)."
        )

    def has_vertex(self, vertex_id: str) -> bool:
        """Test for the existence of a vertex.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        bool

        """
        return vertex_id in self.entity_to_idx and self.entity_types.get(vertex_id) == "vertex"

    def get_edge_ids(self, source, target):
        """List all edge IDs between two endpoints.

        Parameters
        --
        source : str
        target : str

        Returns
        ---
        list[str]
            Edge IDs (may be empty).

        """
        edge_ids = []
        for eid, (src, tgt, _) in self.edge_definitions.items():
            if src == source and tgt == target:
                edge_ids.append(eid)
        return edge_ids

    def degree(self, entity_id):
        """Degree of a vertex or edge-entity (number of incident non-zero entries).

        Parameters
        --
        entity_id : str

        Returns
        ---
        int

        """
        if entity_id not in self.entity_to_idx:
            return 0

        entity_idx = self.entity_to_idx[entity_id]
        row = self._matrix.getrow(entity_idx)
        return len(row.nonzero()[1])

    def vertices(self):
        """Get all vertex IDs (excluding edge-entities).

        Returns
        ---
        list[str]

        """
        return [eid for eid, etype in self.entity_types.items() if etype == "vertex"]

    def edges(self):
        """Get all edge IDs.

        Returns
        ---
        list[str]

        """
        return list(self.edge_to_idx.keys())

    def edge_list(self):
        """Materialize (source, target, edge_id, weight) for binary/vertex-edge edges.

        Returns
        ---
        list[tuple[str, str, str, float]]

        """
        edges = []
        for edge_id, (source, target, edge_type) in self.edge_definitions.items():
            weight = self.edge_weights[edge_id]
            edges.append((source, target, edge_id, weight))
        return edges

    def get_directed_edges(self):
        """List IDs of directed edges.

        Returns
        ---
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [eid for eid in self.edge_to_idx.keys() if self.edge_directed.get(eid, default_dir)]

    def get_undirected_edges(self):
        """List IDs of undirected edges.

        Returns
        ---
        list[str]

        """
        default_dir = True if self.directed is None else self.directed
        return [
            eid for eid in self.edge_to_idx.keys() if not self.edge_directed.get(eid, default_dir)
        ]

    def number_of_vertices(self):
        """Count vertices (excluding edge-entities).

        Returns
        ---
        int

        """
        return len([e for e in self.entity_types.values() if e == "vertex"])

    def number_of_edges(self):
        """Count edges (columns in the incidence matrix).

        Returns
        ---
        int

        """
        return self._num_edges

    def global_entity_count(self):
        """Count unique entities present across all slices (union of memberships).

        Returns
        ---
        int

        """
        all_vertices = set()
        for slice_data in self._slices.values():
            all_vertices.update(slice_data["vertices"])
        return len(all_vertices)

    def global_edge_count(self):
        """Count unique edges present across all slices (union of memberships).

        Returns
        ---
        int

        """
        all_edges = set()
        for slice_data in self._slices.values():
            all_edges.update(slice_data["edges"])
        return len(all_edges)

    def in_edges(self, vertices):
        """Iterate over all edges that are **incoming** to one or more vertices.

        Parameters
        --
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **target set** intersects with this set will be yielded.

        Yields
        --
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior

        - **Directed binary edges**: returned if any vertex is in the target (`T`).
        - **Directed hyperedges**: returned if any vertex is in the tail set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -
        - Works with binary and hyperedges.
        - Undirected edges appear in both `in_edges()` and `out_edges()`.
        - The returned `(S, T)` is the canonical form from `get_edge()`.

        """
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return
        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            if directed:
                if T & V:
                    yield j, (S, T)
            else:
                if (S | T) & V:
                    yield j, (S, T)

    def out_edges(self, vertices):
        """Iterate over all edges that are **outgoing** from one or more vertices.

        Parameters
        --
        vertices : str | Iterable[str]
            A single vertex ID or an iterable of vertex IDs. All edges whose
            **source set** intersects with this set will be yielded.

        Yields
        --
        tuple[int, tuple[frozenset, frozenset]]
            Tuples of the form `(edge_index, (S, T))`, where:
            - `edge_index` : int — internal integer index of the edge.
            - `S` : frozenset[str] — set of source/head verices.
            - `T` : frozenset[str] — set of target/tail verices.

        Behavior

        - **Directed binary edges**: returned if any vertex is in the source (`S`).
        - **Directed hyperedges**: returned if any vertex is in the head set.
        - **Undirected edges/hyperedges**: returned if any vertex is in
        the edge's member set (`S ∪ T`).

        Notes
        -
        - Works with binary and hyperedges.
        - Undirected edges appear in both `out_edges()` and `in_edges()`.
        - The returned `(S, T)` is the canonical form from `get_edge()`.

        """
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return
        for j in range(self.number_of_edges()):
            S, T = self.get_edge(j)
            eid = self.idx_to_edge[j]
            directed = self._is_directed_edge(eid)
            if directed:
                if S & V:
                    yield j, (S, T)
            else:
                if (S | T) & V:
                    yield j, (S, T)

    def get_or_create_vertex_by_attrs(self, slice=None, **attrs) -> str:
        """Return vertex_id for the given composite-key attributes, creating the vertex if needed.

        - Requires set_vertex_key(...) to have been called.
        - All key fields must be present and non-null in attrs.
        """
        if not self._vertex_key_fields:
            raise RuntimeError(
                "Call set_vertex_key(...) before using get_or_create_vertex_by_attrs"
            )

        key = self._build_key_from_attrs(attrs)
        if key is None:
            missing = [f for f in self._vertex_key_fields if f not in attrs or attrs[f] is None]
            raise ValueError(f"Missing composite key fields: {missing}")

        # Existing?
        owner = self._vertex_key_index.get(key)
        if owner is not None:
            return owner

        # Create new vertex
        vid = self._gen_vertex_id_from_key(key)
        # No need to pre-check entity_to_idx here; ids are namespaced by 'cid:' prefix
        self.add_vertex(vid, slice=slice, **attrs)

        # Index ownership
        self._vertex_key_index[key] = vid
        return vid

    def vertex_key_tuple(self, vertex_id) -> tuple | None:
        """Return the composite-key tuple for vertex_id (None if incomplete or no key set)."""
        return self._current_key_of_vertex(vertex_id)

    @property
    def V(self):
        """All vertices as a tuple.

        Returns
        ---
        tuple
            Tuple of all vertex IDs in the graph.

        """
        return tuple(self.vertices())

    @property
    def E(self):
        """All edges as a tuple.

        Returns
        ---
        tuple
            Tuple of all edge identifiers (whatever `self.edges()` yields).

        """
        return tuple(self.edges())

    @property
    def num_vertices(self):
        """Total number of vertices (vertices) in the graph."""
        return self.number_of_vertices()

    @property
    def num_edges(self):
        """Total number of edges in the graph."""
        return self.number_of_edges()

    @property
    def nv(self):
        """Shorthand for num_vertices."""
        return self.num_vertices

    @property
    def ne(self):
        """Shorthand for num_edges."""
        return self.num_edges

    @property
    def shape(self):
        """AnnNet shape as a tuple: (num_vertices, num_edges).
        Useful for quick inspection.
        """
        return (self.num_vertices, self.num_edges)

    # Lazy proxies
    ## Lazy NetworkX proxy

    @property
    def nx(self):
        """Accessor for the lazy NX proxy.
        Usage: G.nx.algorithm(); e.g: G.nx.louvain_communities(G), G.nx.shortest_path_length(G, weight="weight")
        """
        if not hasattr(self, "_nx_proxy"):
            self._nx_proxy = _LazyNXProxy(self)
        return self._nx_proxy

    ## Lazy iGraph proxy

    @property
    def ig(self):
        """Accessor for the lazy igraph proxy.
        Usage: G.ig.community_multilevel(G, weights="weight"), G.ig.shortest_paths_dijkstra(G, source="a", target="z", weights="weight")
        (same idea as NX: pass G; proxy swaps it with the backend igraph.AnnNet lazily)
        """
        if not hasattr(self, "_ig_proxy"):
            self._ig_proxy = _LazyIGProxy(self)
        return self._ig_proxy

    ## Lazy AnnNet-tool proxy

    @property
    def gt(self):
        """Lazy graph-tool proxy.
        Usage:
            G.gt.module.algorithm(...)
            G.gt.backend()
        """
        if not hasattr(self, "_gt_proxy"):
            self._gt_proxy = _LazyGTProxy(self)
        return self._gt_proxy

    # AnnNet API

    def X(self):
        """Sparse incidence matrix."""
        return self._matrix

    @property
    def obs(self):
        """vertex attribute table (observations)."""
        return self.vertex_attributes

    @property
    def var(self):
        """Edge attribute table (variables)."""
        return self.edge_attributes

    @property
    def uns(self):
        """Unstructured metadata."""
        return self.graph_attributes

    @property
    def slices(self):
        """slice operations (add, remove, union, intersect)."""
        if not hasattr(self, "_slice_manager"):
            self._slice_manager = SliceManager(self)
        return self._slice_manager

    @property
    def layers(self):
        """Layer operations."""
        if not hasattr(self, "_layer_manager"):
            self._layer_manager = LayerManager(self)
        return self._layer_manager

    @property
    def idx(self):
        """Index lookups (entity_id↔row, edge_id↔col)."""
        if not hasattr(self, "_index_manager"):
            self._index_manager = IndexManager(self)
        return self._index_manager

    @property
    def cache(self):
        """Cache management (CSR/CSC materialization)."""
        if not hasattr(self, "_cache_manager"):
            self._cache_manager = CacheManager(self)
        return self._cache_manager

    # I/O
    def write(self, path, **kwargs):
        """Save to .annnet format (zero loss)."""
        from ..io.io_annnet import write

        write(self, path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        """Load from .annnet format."""
        from ..io.io_annnet import read

        return read(path, **kwargs)

    # View API
    def view(self, vertices=None, edges=None, slices=None, predicate=None):
        """Create lazy view/subgraph."""
        return GraphView(self, vertices, edges, slices, predicate)

    # Audit
    def snapshot(self, label=None):
        """Create a named snapshot of current graph state.

        Uses existing AnnNet attributes: entity_types, edge_to_idx, _slices, _version

        Parameters
        --
        label : str, optional
            Human-readable label for snapshot (auto-generated if None)

        Returns
        ---
        dict
            Snapshot metadata

        """
        if label is None:
            label = f"snapshot_{len(self._snapshots)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        snapshot = {
            "label": label,
            "version": self._version,
            "timestamp": datetime.now(UTC).isoformat(),
            "counts": {
                "vertices": self.number_of_vertices(),
                "edges": self.number_of_edges(),
                "slices": len(self._slices),
            },
            # Store minimal state for comparison (uses existing AnnNet attributes)
            "vertex_ids": set(v for v, t in self.entity_types.items() if t == "vertex"),
            "edge_ids": set(self.edge_to_idx.keys()),
            "slice_ids": set(self._slices.keys()),
        }

        self._snapshots.append(snapshot)
        return snapshot

    def diff(self, a, b=None):
        """Compare two snapshots or compare snapshot with current state.

        Parameters
        --
        a : str | dict | AnnNet
            First snapshot (label, snapshot dict, or AnnNet instance)
        b : str | dict | AnnNet | None
            Second snapshot. If None, compare with current state.

        Returns
        ---
        GraphDiff
            Difference object with added/removed entities

        """
        snap_a = self._resolve_snapshot(a)
        snap_b = self._resolve_snapshot(b) if b is not None else self._current_snapshot()

        return GraphDiff(snap_a, snap_b)

    def _resolve_snapshot(self, ref):
        """Resolve snapshot reference (label, dict, or AnnNet)."""
        if isinstance(ref, dict):
            return ref
        elif isinstance(ref, str):
            # Find by label
            for snap in self._snapshots:
                if snap["label"] == ref:
                    return snap
            raise ValueError(f"Snapshot '{ref}' not found")
        elif isinstance(ref, AnnNet):
            # Create snapshot from another graph (uses AnnNet attributes)
            return {
                "label": "external",
                "version": ref._version,
                "vertex_ids": set(v for v, t in ref.entity_types.items() if t == "vertex"),
                "edge_ids": set(ref.edge_to_idx.keys()),
                "slice_ids": set(ref._slices.keys()),
            }
        else:
            raise TypeError(f"Invalid snapshot reference: {type(ref)}")

    def _current_snapshot(self):
        """Create snapshot of current state (uses AnnNet attributes)."""
        return {
            "label": "current",
            "version": self._version,
            "vertex_ids": set(v for v, t in self.entity_types.items() if t == "vertex"),
            "edge_ids": set(self.edge_to_idx.keys()),
            "slice_ids": set(self._slices.keys()),
        }

    def list_snapshots(self):
        """List all snapshots.

        Returns
        ---
        list[dict]
            Snapshot metadata

        """
        return [
            {
                "label": snap["label"],
                "timestamp": snap["timestamp"],
                "version": snap["version"],
                "counts": snap["counts"],
            }
            for snap in self._snapshots
        ]
