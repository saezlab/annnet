import time
from datetime import UTC, datetime
import warnings
from collections import defaultdict

import numpy as np

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None
import scipy.sparse as sp

from ._Cache import Operations, CacheManager
from ._Index import IndexManager, IndexMapping
from ._Views import GraphView, ViewsClass
from ._Layers import LayerClass, LayerManager
from ._Slices import SliceClass, SliceManager
from ._BulkOps import BulkOps
from ._helpers import (
    _EDGE_RESERVED,
    EdgeType,
    EdgeRecord,
    EntityRecord,
    EdgeKindCompat,
    EdgeToIdxCompat,
    IdxToEdgeCompat,
    EdgeLayersCompat,
    EdgeWeightsCompat,
    EntityToIdxCompat,
    EntityTypesCompat,
    IdxToEntityCompat,
    EdgeDirectedCompat,
    EdgeDefinitionsCompat,
    EdgeDirectionPolicyCompat,
    HyperedgeDefinitionsCompat,
    _slice_RESERVED,
    _vertex_RESERVED,
    _df_filter_not_equal,
)
from ._History import History, GraphDiff
from ._Annotation import AttributesClass
from .backend_accessors import _GTBackendAccessor, _IGBackendAccessor, _NXBackendAccessor
from .._dataframe_backend import empty_dataframe, select_dataframe_backend
from ..algorithms.traversal import Traversal

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
    """Incidence-based graph with slices, multilayer coordinates, and rich edge types.

    AnnNet stores topology in a sparse incidence matrix backed by canonical
    entity and edge registries. A row represents an entity, typically a vertex
    or an edge-entity, and a column represents an edge. The class supports:

    - binary directed and undirected edges
    - hyperedges, including directed head/tail hyperedges
    - edge-entities that can themselves participate as endpoints
    - slice membership and per-slice edge weights
    - optional multilayer coordinates on vertices and edges
    - dataframe-backed attribute storage

    Parameters
    ----------
    directed : bool | None, optional
        Default directedness for newly created binary edges. If ``None``,
        methods fall back to directed semantics unless a per-edge flag is set.
    v : int, optional
        Initial row capacity for the incidence matrix.
    e : int, optional
        Initial column capacity for the incidence matrix.
    annotations : dict | None, optional
        Pre-built annotation tables to use instead of creating empty tables.
    annotations_backend : {"auto", "polars", "pandas", "pyarrow"} | None, optional
        Preferred backend for newly initialized annotation tables. ``"auto"``
        prefers Polars, then pandas, then PyArrow. ``None`` uses AnnNet's
        configured dataframe default.
    aspects : dict[str, list[str]] | None, optional
        Initial multilayer aspect declaration. If omitted, the graph starts
        flat with a single placeholder aspect ``"_"``.
    **kwargs
        Initial graph-level attributes stored in :attr:`graph_attributes`.

    Notes
    -----
    Directed incidence columns use positive values for sources or heads and
    negative values for targets or tails. Undirected binary edges and
    undirected hyperedges use positive values for all incident entities.

    See Also
    --------
    add_vertex
    add_edge
    add_vertices_bulk
    add_edges_bulk
    view
    """

    # Construction
    def __init__(
        self,
        directed=None,
        v: int = 0,
        e: int = 0,
        annotations=None,
        annotations_backend=None,
        aspects: dict | None = None,
        **kwargs,
    ):
        """Initialize an empty :class:`AnnNet` graph.

        Parameters
        ----------
        directed : bool | None, optional
            Default directedness for newly created binary edges.
        v : int, optional
            Initial row capacity for the sparse incidence matrix.
        e : int, optional
            Initial column capacity for the sparse incidence matrix.
        annotations : dict | None, optional
            Existing annotation tables keyed by table name.
        annotations_backend : {"auto", "polars", "pandas", "pyarrow"} | None, optional
            Backend used when empty annotation tables need to be created.
            ``"auto"`` prefers Polars, then pandas, then PyArrow. ``None``
            uses AnnNet's configured dataframe default.
        aspects : dict[str, list[str]] | None, optional
            Initial multilayer aspect registry.
        **kwargs
            Initial graph-level attributes.

        Notes
        -----
        A default slice named ``"default"`` is always created and made active.
        """
        self.directed = directed

        self._vertex_RESERVED = set(_vertex_RESERVED)
        self._EDGE_RESERVED = set(_EDGE_RESERVED)
        self._slice_RESERVED = set(_slice_RESERVED)

        # --- Aspect/layer registry (immutable aspect count after init) ---
        if aspects is None:
            self._aspects: tuple[str, ...] = ('_',)
            self._layers: dict[str, set[str]] = {'_': {'_'}}
        else:
            if not aspects:
                raise ValueError('aspects dict must not be empty')
            for asp, vals in aspects.items():
                if not vals:
                    raise ValueError(f'Aspect {asp!r} must have at least one layer value')
            self._aspects = tuple(aspects.keys())
            self._layers = {k: set(v) for k, v in aspects.items()}

        # --- Entity store (vertices + edge-entities, things with matrix rows) ---
        # Keys are (vertex_id, layer_coord) tuples; flat graphs use ("_",) as basal layer coord.
        self._entities: dict[tuple, EntityRecord] = {}
        self._row_to_entity: dict[int, tuple] = {}
        self._vid_to_ekeys: dict[str, list[tuple]] = {}

        # --- Edge store (all edges with matrix columns) ---
        self._edges: dict[str, EdgeRecord] = {}  # edge_id   -> EdgeRecord
        self._col_to_edge: dict[int, str] = {}  # col_idx   -> edge_id

        # --- Adjacency indices (derived from _edges, kept updated incrementally) ---
        self._adj: dict = {}  # (src, tgt) -> [edge_id, ...]
        self._src_to_edges: dict = {}  # src -> [edge_id, ...]
        self._tgt_to_edges: dict = {}  # tgt -> [edge_id, ...]

        # --- Composite vertex key support ---
        self._vertex_key_fields = None  # tuple[str, ...] | None
        self._vertex_key_index: dict = {}  # key_tuple -> vertex_id

        # --- Sparse incidence matrix ---
        v = int(v) if v and v > 0 else 0
        e = int(e) if e and e > 0 else 0
        self._matrix = sp.dok_matrix((v, e), dtype=np.float32)
        self._csr_cache = None

        # Grow-only helpers (avoid per-insert exact resize)
        def _grow_rows_to(target: int):
            rows, cols = self._matrix.shape
            if target > rows:
                new_rows = max(target, rows + max(8, rows >> 1))
                self._matrix.resize((new_rows, cols))

        def _grow_cols_to(target: int):
            rows, cols = self._matrix.shape
            if target > cols:
                new_cols = max(target, cols + max(8, cols >> 1))
                self._matrix.resize((rows, new_cols))

        self._grow_rows_to = _grow_rows_to
        self._grow_cols_to = _grow_cols_to

        # --- Attribute storage ---
        self._annotations_backend = select_dataframe_backend(annotations_backend)
        self._init_annotation_tables(annotations)
        self.graph_attributes: dict = {}
        self.graph_attributes.update(kwargs)

        # --- Edge ID counter ---
        self._next_edge_id = 0

        # --- Slice state ---
        self._slices: dict = {}  # slice_id -> {"vertices": set, "edges": set, "attributes": dict}
        self._current_slice: str | None = None
        self._default_slice: str = 'default'
        self.slice_edge_weights: defaultdict = defaultdict(dict)  # slice_id -> {edge_id: weight}
        self._slices[self._default_slice] = {'vertices': set(), 'edges': set(), 'attributes': {}}
        self._current_slice = self._default_slice

        # --- History ---
        self._history_enabled = True
        self._history: list = []
        self._version = 0
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()
        self._snapshots: list = []

        # --- Multilayer state ---
        self._all_layers: tuple = ()
        self.vertex_aligned: bool = False
        self._legacy_single_aspect_enabled: bool = True

        # _V: cached set of pure vertex IDs (derived from _entities on demand)
        self._V: set[str] = set()
        # _VM and _nl_to_row/_row_to_nl are derived from _entities; kept for _Layers.py compat
        self._VM: set = set()
        self._nl_to_row: dict = {}
        self._row_to_nl: list = []

        # Multilayer edge metadata — stored in EdgeRecord.ml_kind / ml_layers (canonical).
        # edge_kind / edge_layers properties are thin compat accessors over EdgeRecord.

        # Aspect / layer / state attribute tables (Kivelä metadata)
        # _aspect_attrs : {aspect_name: {attr: val}}
        # _layer_attrs  : {aspect_name: {layer_value: {attr: val}}}
        # _state_attrs  : {(vid, layer_coord): {attr: val}}  ← supra-node-level attrs
        self._aspect_attrs: dict = {}
        self._layer_attrs: dict = {}
        self._state_attrs: dict = {}

    def _init_annotation_tables(self, annotations):
        # 1) If user provided tables, keep them (we’ll wrap with Narwhals in ops)
        if annotations is not None:
            self.vertex_attributes = annotations.get('vertex_attributes')
            self.edge_attributes = annotations.get('edge_attributes')
            self.slice_attributes = annotations.get('slice_attributes')
            self.edge_slice_attributes = annotations.get('edge_slice_attributes')
            self.layer_attributes = annotations.get('layer_attributes')
            return

        # 2) Otherwise, create empty tables with the centrally selected backend.
        backend = self._annotations_backend
        self.vertex_attributes = empty_dataframe({'vertex_id': 'text'}, backend=backend)
        self.edge_attributes = empty_dataframe({'edge_id': 'text'}, backend=backend)
        self.slice_attributes = empty_dataframe({'slice_id': 'text'}, backend=backend)
        self.edge_slice_attributes = empty_dataframe(
            {'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
            backend=backend,
        )
        self.layer_attributes = empty_dataframe({'layer_id': 'text'}, backend=backend)

    def _entity_row(self, vid) -> int:
        """Return incidence matrix row index for a vertex (resolves bare vid to supra-node key)."""
        return self._entities[self._resolve_entity_key(vid)].row_idx

    def _placeholder_layer_coord(self) -> tuple:
        """Canonical placeholder coordinate for the current aspect rank."""
        if self._aspects == ('_',):
            return ('_',)
        return tuple('_' for _ in self._aspects)

    def _ensure_placeholder_layers_declared(self) -> tuple:
        """Ensure the placeholder value '_' exists for every declared aspect."""
        coord = self._placeholder_layer_coord()
        if self._aspects == ('_',):
            self._layers.setdefault('_', {'_'})
            return coord
        for aspect in self._aspects:
            self._layers.setdefault(aspect, set()).add('_')
        self._rebuild_all_layers_cache()
        return coord

    def _warn_placeholder_vertex_assignment(self, vertex_ids, *, context: str) -> None:
        """Warn that vertices were assigned to the placeholder layer tuple."""
        coord = self._placeholder_layer_coord()
        if isinstance(vertex_ids, str):
            warnings.warn(
                f'{context}: vertex {vertex_ids!r} was assigned to placeholder layer '
                f'{coord!r}. Pass layer= to place it explicitly.',
                UserWarning,
                stacklevel=3,
            )
            return

        vids = list(vertex_ids)
        if not vids:
            return
        sample = ', '.join(repr(v) for v in vids[:3])
        suffix = '' if len(vids) <= 3 else ', ...'
        warnings.warn(
            f'{context}: {len(vids)} vertices were assigned to placeholder layer '
            f'{coord!r} ({sample}{suffix}). Pass layer= to place them explicitly.',
            UserWarning,
            stacklevel=3,
        )

    def _resolve_vertex_insert_coord(
        self, layer_spec, *, vertex_ids=None, context='add_vertex'
    ) -> tuple:
        """Resolve layer placement for vertex insertion, using placeholder fallback when needed."""
        if layer_spec is not None:
            return self._make_layer_coord(layer_spec)
        if self._aspects == ('_',):
            return ('_',)
        coord = self._ensure_placeholder_layers_declared()
        if vertex_ids is not None:
            self._warn_placeholder_vertex_assignment(vertex_ids, context=context)
        return coord

    def _make_layer_coord(self, layer_spec) -> tuple:
        """Normalize any layer specification to a canonical layer_coord tuple.

        Accepted forms
        --------------
        None              flat graph → ("_",); multilayer → raises
        str               single-aspect shorthand, e.g. "t1" → ("t1",)
        dict              {aspect: value} ordered by self._aspects
        tuple             already canonical — validated and returned as-is

        Raises ValueError if a layer value is not declared for its aspect.
        """
        is_flat = self._aspects == ('_',)

        if layer_spec is None:
            if is_flat:
                return ('_',)
            raise ValueError(
                f'layer= must be specified for multilayer graphs with aspects {self._aspects}. '
                'Pass a dict {aspect: value}, a tuple of values, or a bare string (single-aspect).'
            )

        if isinstance(layer_spec, str):
            if len(self._aspects) != 1:
                raise ValueError(
                    f'String layer spec {layer_spec!r} is only valid for single-aspect graphs. '
                    f'Got aspects {self._aspects}. Pass a dict or tuple.'
                )
            coord = (layer_spec,)

        elif isinstance(layer_spec, dict):
            missing = [asp for asp in self._aspects if asp not in layer_spec]
            if missing:
                raise ValueError(
                    f'Missing aspects {missing} in layer spec. Required: {list(self._aspects)}'
                )
            coord = tuple(layer_spec[asp] for asp in self._aspects)

        elif isinstance(layer_spec, tuple):
            coord = layer_spec

        else:
            raise TypeError(
                f'layer_spec must be None, str, dict, or tuple; got {type(layer_spec).__name__!r}'
            )

        # Validate all values against declared layer sets
        if len(coord) != len(self._aspects):
            raise ValueError(
                f'Layer coord length {len(coord)} != number of aspects {len(self._aspects)}'
            )
        for asp, val in zip(self._aspects, coord, strict=False):
            if val not in self._layers[asp]:
                raise ValueError(
                    f'Layer value {val!r} not declared for aspect {asp!r}. '
                    f'Valid: {sorted(self._layers[asp])}'
                )
        return coord

    def _resolve_entity_key(self, vid_or_key) -> tuple:
        """Resolve a vertex identifier to an internal (vid, layer_coord) key.

        Flat graphs (_aspects == ("_",)):
            "alice"               -> ("alice", ("_",))
            ("alice", ("_",))     -> ("alice", ("_",))   [validated]

        Multilayer graphs:
            "alice"               -> returns the first existing supra-node for "alice";
                                     if none exists, creates an annotation-only basal key.
                                     Use (vid, layer_coord) tuples for unambiguous resolution.
            ("alice", ("t1",))    -> ("alice", ("t1",))  [validated]
        """
        if isinstance(vid_or_key, str):
            is_flat = self._aspects == ('_',)
            if is_flat:
                return (vid_or_key, ('_',))
            # Multilayer: find any existing supra-node for this bare vid (stable: lowest row_idx)
            best = None
            for ekey in self._vid_to_ekeys.get(vid_or_key, ()):
                rec = self._entities.get(ekey)
                if rec is None:
                    continue
                if best is None or rec.row_idx < self._entities[best].row_idx:
                    best = ekey
            if best is not None:
                return best
            # No supra-node found: placeholder key — won't have a matrix row until vertex is added
            return (vid_or_key, self._placeholder_layer_coord())
        if isinstance(vid_or_key, tuple) and len(vid_or_key) == 2:
            vid, layer_coord = vid_or_key
            if not isinstance(layer_coord, tuple):
                raise TypeError(
                    f'Layer coordinate must be a tuple, got {type(layer_coord).__name__!r}'
                )
            coord = self._make_layer_coord(layer_coord)  # validates values
            return (vid, coord)
        raise TypeError(
            f'vertex_id must be str or (str, tuple[str,...]), got {type(vid_or_key).__name__!r}'
        )

    def _index_entity_key(self, ekey) -> None:
        if isinstance(ekey, tuple) and len(ekey) == 2 and isinstance(ekey[0], str):
            bucket = self._vid_to_ekeys.setdefault(ekey[0], [])
            if ekey not in bucket:
                bucket.append(ekey)

    def _unindex_entity_key(self, ekey) -> None:
        if isinstance(ekey, tuple) and len(ekey) == 2 and isinstance(ekey[0], str):
            bucket = self._vid_to_ekeys.get(ekey[0])
            if not bucket:
                return
            try:
                bucket.remove(ekey)
            except ValueError:
                return
            if not bucket:
                self._vid_to_ekeys.pop(ekey[0], None)

    def _register_entity_record(self, ekey, rec: EntityRecord) -> None:
        old = self._entities.get(ekey)
        if old is not None and old.row_idx in self._row_to_entity:
            self._row_to_entity.pop(old.row_idx, None)
        self._entities[ekey] = rec
        self._row_to_entity[rec.row_idx] = ekey
        self._index_entity_key(ekey)

    def _remove_entity_record(self, ekey):
        rec = self._entities.pop(ekey)
        self._row_to_entity.pop(rec.row_idx, None)
        self._unindex_entity_key(ekey)
        return rec

    def _rebuild_entity_indexes(self) -> None:
        self._row_to_entity = {}
        self._vid_to_ekeys = {}
        for ekey, rec in self._entities.items():
            self._row_to_entity[rec.row_idx] = ekey
            self._index_entity_key(ekey)

    # Aspect / layer registry queries

    def list_aspects(self) -> tuple:
        """Return the declared aspect names (immutable, in declaration order).

        Returns ("_",) for flat graphs — callers can check ``is_multilayer`` instead.
        """
        return self._aspects

    @property
    def is_multilayer(self) -> bool:
        """True iff the graph has named aspects (is not a flat graph)."""
        return self._aspects != ('_',)

    def list_layers(self, aspect: str | None = None):
        """Return declared layer values for one or all aspects.

        Parameters
        ----------
        aspect : str | None
            Aspect name. If None, returns a dict mapping every aspect to its sorted layer list.

        Returns
        -------
        list[str] | dict[str, list[str]]
        """
        if aspect is None:
            return {asp: sorted(vals) for asp, vals in self._layers.items()}
        if aspect not in self._aspects:
            raise ValueError(f'Unknown aspect {aspect!r}. Valid: {self._aspects}')
        return sorted(self._layers[aspect])

    # Build graph

    def add_vertex(self, vertex_id, slice=None, layer=None, **attributes):
        """Add or update a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.
        slice : str, optional
            Slice receiving the vertex membership. Defaults to the active slice.
        layer : str | dict | tuple | None, optional
            Multilayer placement. Accepted forms are:

            - ``None`` for the flat graph default, or for placeholder placement
              in a layered graph
            - ``str`` for the single-aspect shorthand
            - ``dict`` mapping aspect name to value
            - canonical coordinate tuple
        **attributes
            Vertex attributes to upsert into the vertex attribute table.

        Returns
        -------
        str
            The inserted vertex identifier.

        Notes
        -----
        In multilayer graphs, omitting ``layer`` places the vertex at the
        placeholder coordinate ``("_", ..., "_")`` and emits a warning.
        """
        if slice is None:
            slice = self._current_slice

        # Resolve to internal (vid, layer_coord) key
        # layer= can be None (flat/default), str (single-aspect), dict, or tuple
        coord = self._resolve_vertex_insert_coord(
            layer,
            vertex_ids=vertex_id,
            context='add_vertex',
        )
        key = (vertex_id, coord)
        vid = vertex_id

        _ent = self._entities

        # Register entity if new
        if key not in _ent:
            idx = len(_ent)
            self._register_entity_record(key, EntityRecord(row_idx=idx, kind='vertex'))
            self._grow_rows_to(len(_ent))

        # Maintain _V (pure vertex id cache) and _VM (supra-node cache) for _Layers.py compat
        if _ent[key].kind == 'vertex':
            self._V.add(vid)
            self._VM.add(key)

        # Add to slice (slice tracks bare vid for backward compat with _Slices.py)
        slices = self._slices
        if slice not in slices:
            slices[slice] = {'vertices': set(), 'edges': set(), 'attributes': {}}
        slices[slice]['vertices'].add(vid)

        self._ensure_vertex_table()
        self._ensure_vertex_row(vid)

        if attributes:
            self.vertex_attributes = self._upsert_row(self.vertex_attributes, vid, attributes)

        return vid

    def _ensure_edge_entity_placeholder(self, edge_id, slice=None, **attributes):
        """Ensure a connectable edge-entity placeholder exists without structural incidence."""
        self._register_edge_as_entity(edge_id)

        if edge_id not in self._edges:
            self._edges[edge_id] = EdgeRecord(
                src=None,
                tgt=None,
                weight=1.0,
                directed=False,
                etype='vertex_edge',
                col_idx=-1,
                ml_kind=None,
                ml_layers=None,
                direction_policy=None,
            )

        slice = slice or self._current_slice
        if slice is not None:
            if slice not in self._slices:
                self._slices[slice] = {'vertices': set(), 'edges': set(), 'attributes': {}}
            self._slices[slice]['edges'].add(edge_id)
        if attributes:
            self.set_edge_attrs(edge_id, **attributes)
        return edge_id

    def _register_edge_as_entity(self, edge_id):
        """Make an existing edge connectable as an endpoint."""
        ekey = self._resolve_entity_key(edge_id)
        if ekey in self._entities:
            return
        idx = len(self._entities)
        self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind='edge_entity'))
        self._grow_rows_to(len(self._entities))

    # ── Edge input helpers ────────────────────────────────────────────────────

    def _parse_edge_inputs(self, src, tgt, weight):
        """Normalize src/tgt to (src_nodes, tgt_nodes, col_entries_or_None, etype).

        col_entries is a dict[node, float] with literal matrix values for stoich
        forms, or None for binary/hyper forms (direction applied separately).
        etype is 'binary' | 'hyper' | 'stoich'.
        """
        # Dict forms: literal incidence entries
        if isinstance(src, dict):
            if tgt is None:
                # single dict: negative values → source side, positive → target side
                src_nodes = frozenset(k for k, v in src.items() if v <= 0)
                tgt_nodes = frozenset(k for k, v in src.items() if v > 0)
                return src_nodes, tgt_nodes, dict(src), 'stoich'
            if isinstance(tgt, dict):
                return frozenset(src), frozenset(tgt), {**src, **tgt}, 'stoich'
            raise TypeError(f'If src is dict, tgt must be dict or None, got {type(tgt).__name__!r}')

        # Supra-node binary edge: (vid, layer_coord) tuples
        if (
            isinstance(src, tuple)
            and len(src) == 2
            and isinstance(src[1], tuple)
            and tgt is not None
            and isinstance(tgt, tuple)
            and len(tgt) == 2
            and isinstance(tgt[1], tuple)
        ):
            return frozenset({src}), frozenset({tgt}), None, 'binary'

        # Plain string binary edge
        if isinstance(src, str):
            if tgt is None:
                raise ValueError('Binary edge requires tgt when src is a string.')
            if not isinstance(tgt, str):
                raise TypeError(f'tgt must be str for binary edge, got {type(tgt).__name__!r}')
            return frozenset({src}), frozenset({tgt}), None, 'binary'

        # List/set forms
        if isinstance(src, (list, set, frozenset)):
            src_seq = list(src)
            if tgt is None:
                return frozenset(src_seq), frozenset(), None, 'hyper'
            if isinstance(tgt, (list, set, frozenset)):
                return frozenset(src_seq), frozenset(tgt), None, 'hyper'
            raise TypeError(
                f'If src is list/set, tgt must be list/set or None, got {type(tgt).__name__!r}'
            )

        raise TypeError(
            f'src must be str, tuple (supra-node), list, set, or dict; got {type(src).__name__!r}'
        )

    @staticmethod
    def _infer_ml_kind(src_key, tgt_key):
        """Infer multilayer edge kind from two supra-node endpoint keys."""
        vid_s, lay_s = src_key
        vid_t, lay_t = tgt_key
        if vid_s == vid_t:
            return 'coupling'
        if lay_s == lay_t:
            return 'intra'
        return 'inter'

    def _find_parallel_edges(self, endpoint_set, etype):
        """Return edge_ids with the same endpoint set (any direction)."""
        if etype == 'binary':
            nodes = list(endpoint_set)
            a, b = (nodes[0], nodes[0]) if len(nodes) == 1 else (nodes[0], nodes[1])
            result = list(self._adj.get((a, b), []))
            if a != b:
                result.extend(self._adj.get((b, a), []))
            return result
        # Hyperedge / stoich: scan _edges for matching member frozenset
        all_members = endpoint_set
        result = []
        for eid, rec in self._edges.items():
            if rec.etype != 'hyper' or rec.col_idx < 0:
                continue
            members = set()
            if isinstance(rec.src, frozenset):
                members.update(rec.src)
            elif rec.src is not None:
                members.add(rec.src)
            if isinstance(rec.tgt, frozenset):
                members.update(rec.tgt)
            elif rec.tgt is not None:
                members.add(rec.tgt)
            if frozenset(members) == all_members:
                result.append(eid)
        return result

    def _zero_edge_column(self, rec, col_idx):
        """Zero out all matrix entries for an existing edge column."""
        M = self._matrix
        _fast = getattr(M, '_set_intXint', None)

        def _z(node):
            try:
                r = self._entity_row(node)
                if _fast:
                    _fast(r, col_idx, 0)
                else:
                    M[r, col_idx] = 0
            except (KeyError, ValueError, TypeError):
                pass

        if rec.etype == 'hyper':
            for side in (rec.src, rec.tgt):
                if isinstance(side, frozenset):
                    for n in side:
                        _z(n)
                elif side is not None:
                    _z(side)
        else:
            if rec.src is not None:
                _z(rec.src)
            if rec.tgt is not None and rec.tgt != rec.src:
                _z(rec.tgt)

    # ── Unified edge builder ──────────────────────────────────────────────────

    def add_edge(
        self,
        src=None,
        tgt=None,
        *,
        weight=1.0,
        edge_id=None,
        directed=None,
        parallel='update',
        slice=None,
        as_entity=False,
        propagate='none',
        flexible=None,
        **attrs,
    ):
        """Add or update an edge.

        Parameters
        ----------
        src : str | tuple | list | dict | None
            Edge source specification. Supported forms are:

            - ``str`` or ``(vertex_id, layer_coord)`` for a binary edge source
            - ``list`` for hyperedge members or heads
            - ``dict`` for explicit incidence coefficients
            - ``None`` only when creating an edge-entity placeholder via
              ``as_entity=True`` and ``edge_id=...``
        tgt : str | tuple | list | dict | None
            Edge target specification. ``None`` indicates an undirected
            hyperedge when ``src`` is a list or dict.
        weight : float, optional
            Default coefficient magnitude for binary edges and list-based
            hyperedges.
        edge_id : str | None
            Explicit edge identifier. If omitted, a fresh identifier is generated.
        directed : bool | None
            Per-edge directedness override.
        parallel : {"update", "error", "parallel"}, optional
            Policy for duplicate endpoint sets when ``edge_id`` is not supplied.
        slice : str | None
            Slice receiving the edge membership. Defaults to the active slice.
        as_entity : bool
            If ``True``, also register the edge as an entity with its own row.
            When combined with ``src is None`` and ``tgt is None``, this creates
            a connectable edge-entity placeholder with no structural incidence.
        propagate : {"none", "shared", "all"}, optional
            Slice propagation strategy for the created edge.
        flexible : dict | None
            Flexible-direction policy configuration.
        **attrs
            Edge attributes to upsert into the edge attribute table.

        Returns
        -------
        str
            Identifier of the created or updated edge.

        Notes
        -----
        This is the canonical edge constructor. It covers binary edges,
        supra-node edges, undirected and directed hyperedges, stoichiometric
        hyperedges, and edge-entities.

        Raises
        ------
        ValueError
            If the argument combination is invalid or structurally ambiguous.
        """
        if parallel not in {'update', 'error', 'parallel'}:
            raise ValueError(f"parallel must be 'update'|'error'|'parallel', got {parallel!r}")
        if propagate not in {'none', 'shared', 'all'}:
            raise ValueError(f"propagate must be 'none'|'shared'|'all', got {propagate!r}")
        if not isinstance(weight, (int, float)):
            raise TypeError(f'weight must be numeric, got {type(weight).__name__!r}')
        if flexible is not None and (
            not isinstance(flexible, dict) or 'var' not in flexible or 'threshold' not in flexible
        ):
            raise ValueError(
                "flexible must be a dict with keys {'var','threshold'[,'scope','above','tie']}"
            )

        slice = slice if slice is not None else self._current_slice

        if src is None and tgt is None:
            if as_entity:
                if edge_id is None:
                    raise ValueError(
                        'edge_id is required when creating an edge-entity without endpoints.'
                    )
                return self._ensure_edge_entity_placeholder(edge_id, slice=slice, **attrs)
            raise ValueError('add_edge requires structural endpoints unless as_entity=True.')

        # ── 1. Parse inputs ────────────────────────────────────────────────
        src_nodes, tgt_nodes, col_entries_literal, etype = self._parse_edge_inputs(src, tgt, weight)

        if self.is_multilayer:
            non_supra = [
                node
                for node in (src_nodes | tgt_nodes)
                if not (isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple))
            ]
            if non_supra:
                sample = ', '.join(repr(node) for node in non_supra[:3])
                suffix = '' if len(non_supra) <= 3 else ', ...'
                raise ValueError(
                    'Multilayer structural edges require explicit supra-node endpoints '
                    f'(vertex_id, layer_coord); got bare endpoints: {sample}{suffix}'
                )

        # ── 2. Resolve direction ───────────────────────────────────────────
        if directed is not None:
            is_dir = bool(directed)
        elif etype == 'hyper':
            # For hyperedges, direction is topological: has explicit tail → directed
            is_dir = bool(tgt_nodes)
        elif self.directed is not None:
            is_dir = bool(self.directed)
        else:
            is_dir = True

        # ── 3. Build column entries ────────────────────────────────────────
        if col_entries_literal is not None:
            # stoich: literal values already in col_entries_literal
            col_entries = col_entries_literal
        else:
            # binary / hyper: apply ±weight based on direction
            col_entries = {}
            for n in src_nodes:
                col_entries[n] = float(weight)
            for n in tgt_nodes:
                col_entries[n] = -float(weight) if is_dir else float(weight)

        endpoint_set = frozenset(col_entries)

        # ── 4. Resolve parallel ────────────────────────────────────────────
        # If the caller provided an explicit edge_id that already exists → always update it.
        # If the caller provided an explicit edge_id that doesn't exist → always create it.
        # The parallel policy only applies when edge_id is None (auto-generated).
        explicit_id = edge_id is not None
        if explicit_id and edge_id in self._edges:
            # targeting an existing named edge: update in-place, skip endpoint dedup
            pass
        elif explicit_id:
            # new named edge: create regardless of parallel edges between same endpoints
            if parallel == 'error':
                existing = self._find_parallel_edges(endpoint_set, etype)
                if existing:
                    raise ValueError(
                        f'Edge already exists between {endpoint_set}. '
                        "Use parallel='parallel' to allow parallel edges."
                    )
        else:
            # auto-generated id: apply parallel policy
            existing = self._find_parallel_edges(endpoint_set, etype)
            if existing:
                if parallel == 'error':
                    raise ValueError(
                        f'Edge already exists between {endpoint_set}. '
                        "Use parallel='parallel' to allow parallel edges."
                    )
                if parallel == 'update':
                    edge_id = existing[-1]
            if edge_id is None:
                edge_id = self._get_next_edge_id()

        # ── 5. Ensure endpoints exist ──────────────────────────────────────
        _ent = self._entities
        _edg = self._edges
        for node in endpoint_set:
            ekey = self._resolve_entity_key(node)
            if ekey not in _ent:
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                    self.add_vertex(node[0], layer=node[1], slice=slice)
                else:
                    self.add_vertex(node, slice=slice)

        self._grow_rows_to(len(_ent))

        # ── 6. Column allocation / zeroing ─────────────────────────────────
        is_new = edge_id not in _edg or _edg[edge_id].col_idx < 0
        if is_new:
            col_idx = len(self._col_to_edge)
            self._col_to_edge[col_idx] = edge_id
            self._grow_cols_to(col_idx + 1)
        else:
            col_idx = _edg[edge_id].col_idx
            self._zero_edge_column(_edg[edge_id], col_idx)

        # ── 7. Write matrix column ─────────────────────────────────────────
        M = self._matrix
        _fast = getattr(M, '_set_intXint', None)
        _dt = M.dtype.type
        for node, coeff in col_entries.items():
            r = self._entity_row(node)
            if _fast:
                _fast(r, col_idx, _dt(coeff))
            else:
                M[r, col_idx] = coeff

        # ── 8. Compute src_store / tgt_store for EdgeRecord ────────────────
        if etype == 'binary':
            src_store = next(iter(src_nodes))
            tgt_store = next(iter(tgt_nodes)) if tgt_nodes else None
            rec_etype = 'vertex_edge' if as_entity else 'binary'
        else:
            src_store = frozenset(src_nodes) if src_nodes else None
            tgt_store = frozenset(tgt_nodes) if tgt_nodes else None
            rec_etype = 'hyper'

        # ── 9. Infer ml_kind for supra-node edges ──────────────────────────
        ml_kind = None
        ml_layers = None
        if (
            etype == 'binary'
            and isinstance(src, tuple)
            and len(src) == 2
            and isinstance(src[1], tuple)
            and isinstance(tgt, tuple)
            and len(tgt) == 2
            and isinstance(tgt[1], tuple)
        ):
            ml_kind = self._infer_ml_kind(src, tgt)
            ml_layers = (src[1], tgt[1])

        # ── 10. Store / update EdgeRecord ──────────────────────────────────
        if is_new:
            _edg[edge_id] = EdgeRecord(
                src=src_store,
                tgt=tgt_store,
                weight=float(weight),
                directed=is_dir,
                etype=rec_etype,
                col_idx=col_idx,
                ml_kind=ml_kind,
                ml_layers=ml_layers,
                direction_policy=flexible,
            )
            self._adj.setdefault((src_store, tgt_store), []).append(edge_id)
            if src_store is not None:
                self._src_to_edges.setdefault(src_store, []).append(edge_id)
            if tgt_store is not None:
                self._tgt_to_edges.setdefault(tgt_store, []).append(edge_id)
        else:
            rec = _edg[edge_id]
            old_src, old_tgt = rec.src, rec.tgt
            if (old_src, old_tgt) != (src_store, tgt_store):
                # update adjacency indices
                for _old, _new, _idx in (
                    (old_src, src_store, self._src_to_edges),
                    (old_tgt, tgt_store, self._tgt_to_edges),
                ):
                    if _old != _new:
                        lst = _idx.get(_old)
                        if lst:
                            try:
                                lst.remove(edge_id)
                            except ValueError:
                                pass
                            if not lst:
                                del _idx[_old]
                        if _new is not None:
                            _idx.setdefault(_new, []).append(edge_id)
                lst = self._adj.get((old_src, old_tgt))
                if lst:
                    try:
                        lst.remove(edge_id)
                    except ValueError:
                        pass
                    if not lst:
                        del self._adj[(old_src, old_tgt)]
                self._adj.setdefault((src_store, tgt_store), []).append(edge_id)
            rec.src = src_store
            rec.tgt = tgt_store
            rec.weight = float(weight)
            rec.directed = is_dir
            rec.etype = rec_etype
            rec.ml_kind = ml_kind
            rec.ml_layers = ml_layers
            if flexible is not None:
                rec.direction_policy = flexible

        # ── 11. as_entity ──────────────────────────────────────────────────
        if as_entity:
            self._register_edge_as_entity(edge_id)

        # ── 12. Slice ──────────────────────────────────────────────────────
        if slice is not None:
            slices = self._slices
            if slice not in slices:
                slices[slice] = {'vertices': set(), 'edges': set(), 'attributes': {}}
            slices[slice]['edges'].add(edge_id)
            # Slice tracks bare vids
            for n in endpoint_set:
                slices[slice]['vertices'].add(n[0] if isinstance(n, tuple) else n)

        # ── 13. Propagate ──────────────────────────────────────────────────
        if propagate == 'shared':
            self._propagate_to_shared_slices(edge_id, src_store, tgt_store)
        elif propagate == 'all':
            self._propagate_to_all_slices(edge_id, src_store, tgt_store)

        # ── 14. Flexible direction ─────────────────────────────────────────
        if flexible is not None:
            _edg[edge_id].directed = True
            self._apply_flexible_direction(edge_id)

        # ── 15. Attributes ─────────────────────────────────────────────────
        if attrs:
            self.set_edge_attrs(edge_id, **attrs)

        return edge_id

    def set_edge_coeffs(self, edge_id: str, coeffs: dict[str, float]) -> None:
        """Overwrite incidence coefficients for an existing edge.

        Parameters
        ----------
        edge_id : str
            Edge identifier.
        coeffs : dict[str, float]
            Mapping from entity identifier to numeric coefficient.

        Notes
        -----
        This method is currently edge-type preserving in intent: callers should
        only provide coefficient patterns consistent with the existing edge
        topology.
        """
        col = self._edges[edge_id].col_idx
        for vid, coeff in coeffs.items():
            self._matrix[self._entity_row(vid), col] = float(coeff)

    def add_edge_to_slice(self, lid, eid):
        """Attach an existing edge to a slice (no weight changes).

        Parameters
        ----------
        lid : str
            Slice ID.
        eid : str
            Edge ID.

        Raises
        ------
        KeyError
            If the slice does not exist.
        """
        if lid not in self._slices:
            raise KeyError(f'slice {lid} does not exist')

        if eid not in self._edges:
            if eid not in self._entities:
                self._register_edge_as_entity(eid)
            self._edges[eid] = EdgeRecord(
                src=None,
                tgt=None,
                weight=1.0,
                directed=False,
                etype='vertex_edge',
                col_idx=-1,
                ml_kind=None,
                ml_layers=None,
                direction_policy=None,
            )

        self._slices[lid]['edges'].add(eid)

    def _propagate_to_shared_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to all slices that already contain **both** endpoints.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data['vertices'] and target in slice_data['vertices']:
                slice_data['edges'].add(edge_id)

    def _propagate_to_all_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to any slice containing **either** endpoint and
        insert the missing endpoint into that slice.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for slice_id, slice_data in self._slices.items():
            if source in slice_data['vertices'] or target in slice_data['vertices']:
                slice_data['edges'].add(edge_id)
                # Only add missing endpoint if both vertices should be in slice
                if source in slice_data['vertices']:
                    slice_data['vertices'].add(target)
                if target in slice_data['vertices']:
                    slice_data['vertices'].add(source)

    def _normalize_vertices_arg(self, vertices):
        """Normalize a single vertex or an iterable of vertices into a set.

        This internal utility function standardizes input for methods like
        `incident_edges()` by converting the argument into a set of vertex
        identifiers.

        Parameters
        ----------
        vertices : str | Iterable[str] | None
            - A single vertex ID (string).
            - An iterable of vertex IDs (e.g., list, tuple, set).
            - `None` is allowed and will return an empty set.

        Returns
        -------
        set[str]
            A set of vertex identifiers. If `vertices` is `None`, returns an
            empty set. If a single vertex is provided, returns a one-element set.

        Notes
        -----
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
        """Force the whole graph to be undirected in-place.

        Parameters
        ----------
        drop_flexible : bool, optional
            If True, clear flexible-direction policies.
        update_default : bool, optional
            If True, set the graph default to undirected for future edges.

        Notes
        -----
        Binary edges are rewritten to `(+w, +w)` at `(source, target)`.
        Hyperedges are rewritten to `+w` on all members.
        """

        M = self._matrix

        # 1) Binary / vertex-edge edges
        for eid, rec in list(self._edges.items()):
            if rec.etype == 'hyper':
                continue  # handled below
            if rec.src is None or rec.tgt is None:
                continue

            col = rec.col_idx
            if col < 0:
                continue

            w = float(rec.weight)

            try:
                si = self._entity_row(rec.src) if rec.src is not None else None
            except (KeyError, ValueError, TypeError):
                si = None
            try:
                ti = self._entity_row(rec.tgt) if rec.tgt is not None else None
            except (KeyError, ValueError, TypeError):
                ti = None

            if si is not None:
                M[si, col] = w
            if ti is not None and ti != si:
                M[ti, col] = w

            rec.directed = False
            try:
                self.set_edge_attrs(eid, edge_type=EdgeType.UNDIRECTED)
            except Exception:
                pass

        # 2) Hyperedges
        for eid, rec in list(self._edges.items()):
            if rec.etype != 'hyper':
                continue

            col = rec.col_idx
            if col < 0:
                continue

            w = float(rec.weight)

            # rec.src is frozenset(head or members), rec.tgt is frozenset(tail) or None
            if rec.tgt is not None:
                # directed hyperedge: src=head, tgt=tail
                members = rec.src | rec.tgt
                for u in members:
                    ent = self._entities.get(u)
                    if ent is not None:
                        try:
                            M[ent.row_idx, col] = 0
                        except KeyError:
                            pass
            else:
                members = rec.src  # undirected: src=all members

            for u in members:
                ent = self._entities.get(u)
                if ent is not None:
                    M[ent.row_idx, col] = w

            # Rewrite as undirected: src=all members, tgt=None
            rec.src = frozenset(members)
            rec.tgt = None
            rec.directed = False

        # 3) Optional: drop flexible-direction policies
        if drop_flexible:
            for rec in self._edges.values():
                rec.direction_policy = None

        # 4) Optional: set global default to undirected for future edges
        if update_default:
            self.directed = False

        return self

    # Remove / mutate down

    def remove_edge(self, edge_id):
        """Remove an edge (binary or hyperedge) from the graph.

        Parameters
        ----------
        edge_id : str
            Edge identifier.

        Raises
        ------
        KeyError
            If the edge is not found.

        Notes
        -----
        Physically removes the incidence column (no CSR round-trip) and cleans
        edge attributes and slice memberships.
        """
        if edge_id not in self._edges:
            raise KeyError(f'Edge {edge_id} not found')

        rec = self._edges[edge_id]
        col_idx = rec.col_idx

        # column removal without CSR (single pass over nonzeros)
        M_old = self._matrix
        rows, cols = M_old.shape
        new_cols = cols - 1
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if c == col_idx:
                continue
            elif c > col_idx:
                M_new[r, c - 1] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new
        self._csr_cache = None

        # Shift col indices for all edges after the removed column
        del self._col_to_edge[col_idx]
        num_edges = len(self._col_to_edge) + 1  # before deletion
        for old_c in range(col_idx + 1, num_edges):
            eid = self._col_to_edge.pop(old_c)
            self._col_to_edge[old_c - 1] = eid
            self._edges[eid].col_idx = old_c - 1

        # Adjacency cleanup
        s, t = rec.src, rec.tgt
        if s is not None and t is not None and isinstance(s, str) and isinstance(t, str):
            key = (s, t)
            lst = self._adj.get(key)
            if lst:
                try:
                    lst.remove(edge_id)
                except ValueError:
                    pass
                if not lst:
                    del self._adj[key]
            for v, index in ((s, self._src_to_edges), (t, self._tgt_to_edges)):
                _lst = index.get(v)
                if _lst:
                    try:
                        _lst.remove(edge_id)
                    except ValueError:
                        pass
                    if not _lst:
                        del index[v]

        # Primary record deletion
        del self._edges[edge_id]

        # Remove from edge attributes DataFrame
        ea = self.edge_attributes
        if ea is not None and hasattr(ea, 'columns'):
            is_empty = (getattr(ea, 'height', None) == 0) or (
                hasattr(ea, '__len__') and len(ea) == 0
            )
            if (not is_empty) and ('edge_id' in list(ea.columns)):
                self.edge_attributes = _df_filter_not_equal(ea, 'edge_id', edge_id)

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data['edges'].discard(edge_id)

        # Remove from edge-slice attributes
        esa = self.edge_slice_attributes
        if esa is not None and hasattr(esa, 'columns'):
            is_empty = (getattr(esa, 'height', None) == 0) or (
                hasattr(esa, '__len__') and len(esa) == 0
            )
            if (not is_empty) and ('edge_id' in list(esa.columns)):
                self.edge_slice_attributes = _df_filter_not_equal(esa, 'edge_id', edge_id)

        # Legacy slice-edge weight dicts
        for d in self.slice_edge_weights.values():
            d.pop(edge_id, None)

    def remove_vertex(self, vertex_id):
        """Remove a vertex and all incident edges (binary + hyperedges).

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Raises
        ------
        KeyError
            If the vertex is not found.

        Notes
        -----
        Rebuilds entity indexing and shrinks the incidence matrix accordingly.
        """
        ekey = self._resolve_entity_key(vertex_id)
        if ekey not in self._entities:
            raise KeyError(f'vertex {vertex_id!r} not found')

        entity_idx = self._entities[ekey].row_idx

        # Collect all incident edges in one pass over _edges
        edges_to_remove = set()
        for eid, rec in list(self._edges.items()):
            if rec.etype == 'hyper':
                # src is frozenset(head or members), tgt is frozenset(tail) or None
                if vertex_id in rec.src or (rec.tgt is not None and vertex_id in rec.tgt):
                    edges_to_remove.add(eid)
            else:
                if rec.src == vertex_id or rec.tgt == vertex_id:
                    edges_to_remove.add(eid)

        for eid in edges_to_remove:
            self.remove_edge(eid)

        # Row removal without CSR: rebuild DOK with rows-1 and shift indices
        M_old = self._matrix
        rows, cols = M_old.shape
        new_rows = rows - 1
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        for (r, c), v in M_old.items():
            if r == entity_idx:
                continue
            elif r > entity_idx:
                M_new[r - 1, c] = v
            else:
                M_new[r, c] = v
        self._matrix = M_new
        self._csr_cache = None

        # Update entity mappings
        self._remove_entity_record(ekey)
        self._V.discard(vertex_id if isinstance(vertex_id, str) else vertex_id)
        self._VM.discard(ekey)

        # Shift row indices for all entities after the removed row
        num_entities = len(self._entities) + 1  # before deletion
        for old_r in range(entity_idx + 1, num_entities):
            ent_id = self._row_to_entity.pop(old_r)
            self._row_to_entity[old_r - 1] = ent_id
            self._entities[ent_id].row_idx = old_r - 1

        # Remove from vertex attributes DataFrame
        va = self.vertex_attributes
        if va is not None and hasattr(va, 'columns'):
            is_empty = (getattr(va, 'height', None) == 0) or (
                hasattr(va, '__len__') and len(va) == 0
            )
            if (not is_empty) and ('vertex_id' in list(va.columns)):
                self.vertex_attributes = _df_filter_not_equal(va, 'vertex_id', vertex_id)

        # Remove from per-slice membership
        for slice_data in self._slices.values():
            slice_data['vertices'].discard(vertex_id)

    def remove_slice(self, slice_id):
        """Remove a non-default slice and its per-slice attributes.

        Parameters
        ----------
        slice_id : str
            Slice identifier.

        Raises
        ------
        ValueError
            If attempting to remove the internal default slice.
        KeyError
            If the slice does not exist.

        Notes
        -----
        Does not delete vertices or edges globally; only membership and metadata.
        """
        if slice_id == self._default_slice:
            raise ValueError('Cannot remove default slice')
        if slice_id not in self._slices:
            raise KeyError(f'slice {slice_id} not found')

        # Purge per-slice attributes
        ela = getattr(self, 'edge_slice_attributes', None)
        if ela is not None and hasattr(ela, 'columns'):
            cols = list(ela.columns)
            is_empty = (getattr(ela, 'height', None) == 0) or (
                hasattr(ela, '__len__') and len(ela) == 0
            )
            if (not is_empty) and ('slice_id' in cols):
                self.edge_slice_attributes = _df_filter_not_equal(ela, 'slice_id', slice_id)

        # Drop legacy dict slice if present
        if isinstance(getattr(self, 'slice_edge_weights', None), dict):
            self.slice_edge_weights.pop(slice_id, None)

        # Remove the slice and reset current if needed
        del self._slices[slice_id]
        if self._current_slice == slice_id:
            self._current_slice = self._default_slice

    def remove_orphans(self):
        """Remove all vertices with no incident edges from the AnnNet graph."""
        csr = self._get_csr()
        orphans = []
        for idx in range(len(self._entities)):
            ent = self._row_to_entity[idx]
            if self._entities[ent].kind == 'vertex':
                if csr.indptr[idx + 1] - csr.indptr[idx] == 0:
                    orphans.append(ent)
        if orphans:
            self._remove_vertices_bulk(orphans)
        return len(orphans)

    # Basic queries & metrics

    def get_vertex(self, index: int) -> str:
        """Return the vertex ID corresponding to a given internal index.

        Parameters
        ----------
        index : int
            Internal vertex index.

        Returns
        -------
        str
            The vertex ID.
        """
        entry = self._row_to_entity[index]
        return entry[0] if isinstance(entry, tuple) else entry

    def get_edge(self, index: int):
        """Return edge endpoints in a canonical form.

        Parameters
        ----------
        index : int | str
            Internal edge index or edge ID.

        Returns
        -------
        tuple[frozenset, frozenset]
            `(S, T)` where `S` and `T` are frozensets of vertex IDs.
        """
        if isinstance(index, str):
            eid = index
            if eid not in self._edges:
                raise KeyError(f'Unknown edge id: {eid}') from None
        else:
            eid = self._col_to_edge[index]

        rec = self._edges[eid]

        if rec.etype == 'hyper':
            # src = frozenset(head or members), tgt = frozenset(tail) or None
            if rec.tgt is not None:
                return (frozenset(rec.src), frozenset(rec.tgt))
            else:
                M = frozenset(rec.src)
                return (M, M)
        else:
            u, v = rec.src, rec.tgt
            d = (
                rec.directed
                if rec.directed is not None
                else (True if self.directed is None else self.directed)
            )
            if d:
                return (frozenset([u]), frozenset([v]))
            else:
                M = frozenset([u, v])
                return (M, M)

    def _incident_edge_indices(self, vertex_id) -> list[int]:
        """Return matrix column indices of all edges incident to a vertex."""
        incident = []
        ent = self._entities.get(vertex_id)
        if ent is not None:
            try:
                incident.extend(self._get_csr().getrow(ent.row_idx).indices.tolist())
                return incident
            except Exception:
                pass
        for j in range(len(self._col_to_edge)):
            eid = self._col_to_edge[j]
            rec = self._edges[eid]
            if rec.etype == 'hyper':
                if vertex_id in rec.src or (rec.tgt is not None and vertex_id in rec.tgt):
                    incident.append(j)
            else:
                if rec.src == vertex_id or rec.tgt == vertex_id:
                    incident.append(j)
        return incident

    def _is_directed_edge(self, edge_id):
        """Check if an edge is directed (per-edge flag overrides graph default).

        Parameters
        ----------
        edge_id : str

        Returns
        -------
        bool

        """
        rec = self._edges.get(edge_id)
        if rec is None:
            return bool(self.directed)
        d = rec.directed
        return bool(d if d is not None else self.directed)

    def has_edge(
        self,
        source: str | None = None,
        target: str | None = None,
        edge_id: str | None = None,
    ) -> bool | tuple[bool, list[str]]:
        """Check edge existence using one of three modes.

        Parameters
        ----------
        source : str, optional
            Source entity ID.
        target : str, optional
            Target entity ID.
        edge_id : str, optional
            Edge identifier.

        Returns
        -------
        bool | tuple[bool, list[str]]
            One of:
            - bool, if only `edge_id` is provided.
            - (bool, [edge_ids...]) if `source` and `target` are provided.

        Raises
        ------
        ValueError
            If the argument combination is invalid.
        """

        # ---- Mode 1: edge_id only ----
        if edge_id is not None and source is None and target is None:
            return edge_id in self._edges

        # ---- Mode 2: source + target only ----
        if edge_id is None and source is not None and target is not None:
            eids = list(self._adj.get((source, target), []))
            return (len(eids) > 0, eids)

        # ---- Mode 3: edge_id + source + target ----
        if edge_id is not None and source is not None and target is not None:
            rec = self._edges.get(edge_id)
            if rec is None:
                return False
            return rec.src == source and rec.tgt == target

        # ---- Anything else is ambiguous / invalid ----
        raise ValueError(
            'Invalid argument combination: use either '
            '(edge_id), (source,target), or (source,target,edge_id).'
        )

    def has_vertex(self, vertex_id: str) -> bool:
        """Test for the existence of a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        bool
            True if the vertex exists.
        """
        ent = self._entities.get(vertex_id)
        return ent is not None and ent.kind == 'vertex'

    def get_edge_ids(self, source, target):
        """List all edge IDs between two endpoints.

        Parameters
        ----------
        source : str
            Source entity ID.
        target : str
            Target entity ID.

        Returns
        -------
        list[str]
            Edge IDs (may be empty).
        """
        return list(self._adj.get((source, target), []))

    def _get_csr(self):
        """Return a cached CSR view of _matrix. Rebuilt when _csr_cache is None."""
        if self._csr_cache is None:
            self._csr_cache = self._matrix.tocsr()
        return self._csr_cache

    def degree(self, entity_id):
        """Degree of a vertex or edge-entity (number of incident non-zero entries).

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        int
            Degree of the entity.
        """
        ekey = self._resolve_entity_key(entity_id)
        ent = self._entities.get(ekey)
        if ent is None:
            return 0
        csr = self._get_csr()
        return int(csr.indptr[ent.row_idx + 1] - csr.indptr[ent.row_idx])

    def vertices(self):
        """Get all vertex IDs (excluding edge-entities).

        Returns
        -------
        list[str]
            Vertex IDs.
        """
        return [
            eid[0] if isinstance(eid, tuple) else eid
            for eid, rec in self._entities.items()
            if rec.kind == 'vertex'
        ]

    def edges(self):
        """Get all edge IDs.

        Returns
        -------
        list[str]
            Edge IDs.
        """
        return [eid for eid, rec in self._edges.items() if rec.col_idx >= 0]

    def edge_list(self):
        """Materialize (source, target, edge_id, weight) for binary/vertex-edge edges.

        Returns
        -------
        list[tuple[str, str, str, float]]
            Tuples of `(source, target, edge_id, weight)`.
        """
        edges = []
        for edge_id, rec in self._edges.items():
            if rec.etype == 'hyper' or rec.src is None or rec.tgt is None:
                continue
            edges.append((rec.src, rec.tgt, edge_id, rec.weight))
        return edges

    def get_edges_by_direction(self, directed: bool):
        """List edge identifiers matching a directedness flag.

        Parameters
        ----------
        directed : bool
            Desired directedness.

        Returns
        -------
        list[str]
            Edge identifiers whose effective directedness matches ``directed``.
        """
        default_dir = True if self.directed is None else self.directed
        return [
            eid
            for eid, rec in self._edges.items()
            if rec.col_idx >= 0
            and bool(rec.directed if rec.directed is not None else default_dir) is bool(directed)
        ]

    def global_count(self, kind: str) -> int:
        """Count unique members across all slices.

        Parameters
        ----------
        kind : {"vertices", "edges", "entities"}
            Membership domain to count.

        Returns
        -------
        int
            Number of unique members observed across all slices.
        """
        if kind not in {'vertices', 'edges', 'entities'}:
            raise ValueError("kind must be one of {'vertices', 'edges', 'entities'}")
        members = set()
        for slice_data in self._slices.values():
            if kind in {'vertices', 'entities'}:
                members.update(slice_data['vertices'])
            if kind in {'edges', 'entities'}:
                members.update(slice_data['edges'])
        return len(members)

    # ── Backward-compat thin wrappers ─────────────────────────────────────────

    def number_of_vertices(self) -> int:
        """Number of vertices. Prefer the ``nv`` property."""
        return self.nv

    def number_of_edges(self) -> int:
        """Number of edges. Prefer the ``ne`` property."""
        return self.ne

    def global_vertex_count(self) -> int:
        """Unique vertices across all slices. Prefer ``global_count('vertices')``."""
        return self.global_count('vertices')

    def global_entity_count(self) -> int:
        """Unique entities across all slices. Prefer ``global_count('entities')``."""
        return self.global_count('entities')

    def global_edge_count(self) -> int:
        """Unique edges across all slices. Prefer ``global_count('edges')``."""
        return self.global_count('edges')

    def in_edges(self, vertices):
        """Incoming edges. Prefer ``incident_edges(direction='in')``."""
        return self.incident_edges(vertices, direction='in')

    def out_edges(self, vertices):
        """Outgoing edges. Prefer ``incident_edges(direction='out')``."""
        return self.incident_edges(vertices, direction='out')

    def get_directed_edges(self) -> list[str]:
        """Directed edge IDs. Prefer ``get_edges_by_direction(True)``."""
        return self.get_edges_by_direction(True)

    def get_undirected_edges(self) -> list[str]:
        """Undirected edge IDs. Prefer ``get_edges_by_direction(False)``."""
        return self.get_edges_by_direction(False)

    # ── Traversal ────────────────────────────────────────────────────────────

    def incident_edges(self, vertices, direction: str = 'both'):
        """Iterate over edges incident to one or more vertices.

        Parameters
        ----------
        vertices : str | Iterable[str]
            One vertex identifier or an iterable of identifiers.
        direction : {"in", "out", "both"}, optional
            Directional filter applied to binary edges. Undirected edges are
            yielded for both directions.

        Yields
        ------
        tuple[int, tuple]
            Pairs of ``(column_index, edge_tuple)`` as returned by
            :meth:`get_edge`.
        """
        if direction not in {'in', 'out', 'both'}:
            raise ValueError("direction must be 'in', 'out', or 'both'")
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return
        seen = set()
        for v in V:
            if direction in {'in', 'both'}:
                for eid in self._tgt_to_edges.get(v, []):
                    if eid not in seen:
                        seen.add(eid)
                        rec = self._edges.get(eid)
                        if rec is not None and rec.col_idx >= 0:
                            yield rec.col_idx, self.get_edge(rec.col_idx)
            if direction in {'out', 'both'}:
                for eid in self._src_to_edges.get(v, []):
                    if eid not in seen:
                        seen.add(eid)
                        rec = self._edges.get(eid)
                        if rec is not None and rec.col_idx >= 0:
                            yield rec.col_idx, self.get_edge(rec.col_idx)
            if direction == 'in':
                secondary = self._src_to_edges.get(v, [])
            elif direction == 'out':
                secondary = self._tgt_to_edges.get(v, [])
            else:
                secondary = []
            for eid in secondary:
                if eid not in seen and not self._is_directed_edge(eid):
                    seen.add(eid)
                    rec = self._edges.get(eid)
                    if rec is not None and rec.col_idx >= 0:
                        yield rec.col_idx, self.get_edge(rec.col_idx)

    @property
    def nv(self):
        """Total number of vertices in the graph."""
        return sum(1 for r in self._entities.values() if r.kind == 'vertex')

    @property
    def ne(self):
        """Total number of edges in the graph."""
        return len(self._col_to_edge)

    @property
    def num_vertices(self):
        """Alias for `nv`."""
        return self.nv

    @property
    def num_edges(self):
        """Alias for `ne`."""
        return self.ne

    @property
    def shape(self):
        """AnnNet shape as a tuple: (num_vertices, num_edges)."""
        return (self.nv, self.ne)

    def get_or_create_vertex_by_attrs(self, slice=None, **attrs) -> str:
        """Return vertex ID for the given composite-key attributes.

        Parameters
        ----------
        slice : str, optional
            Slice to place a newly created vertex into.
        **attrs
            Attributes used to build the composite key.

        Returns
        -------
        str
            Vertex ID matching the composite key.

        Raises
        ------
        RuntimeError
            If no composite key fields are configured.
        ValueError
            If required key fields are missing.

        Notes
        -----
        Requires `set_vertex_key(...)` to have been called.
        """
        if not self._vertex_key_fields:
            raise RuntimeError(
                'Call set_vertex_key(...) before using get_or_create_vertex_by_attrs'
            )

        key = self._build_key_from_attrs(attrs)
        if key is None:
            missing = [f for f in self._vertex_key_fields if f not in attrs or attrs[f] is None]
            raise ValueError(f'Missing composite key fields: {missing}')

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

    def _gen_vertex_id_from_key(self, key) -> str:
        """Generate a stable vertex id from a composite key tuple.

        The id is deterministic for a given key and namespaced away from
        user-provided ids. If a generated id already exists for a different
        key, append a numeric suffix until it becomes unique.
        """
        from urllib.parse import quote

        base = 'cid:' + '|'.join(quote(str(part), safe='') for part in key)
        vid = base
        i = 1
        while self.has_vertex(vid):
            current = self._current_key_of_vertex(vid)
            if current == key:
                return vid
            vid = f'{base}::{i}'
            i += 1
        return vid

    def vertex_key_tuple(self, vertex_id) -> tuple | None:
        """Return the composite-key tuple for a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        tuple | None
            Composite key tuple, or None if incomplete or not configured.
        """
        return self._current_key_of_vertex(vertex_id)

    @property
    def V(self):
        """All vertices as a tuple.

        Returns
        -------
        tuple
            Vertex IDs.
        """
        return tuple(self.vertices())

    @property
    def E(self):
        """All edges as a tuple.

        Returns
        -------
        tuple
            Edge identifiers.
        """
        return tuple(self.edges())

    # Backend accessors
    ## NetworkX backend accessor

    @property
    def nx(self):
        """NetworkX backend accessor.

        Returns
        -------
        _NXBackendAccessor
            Accessor exposing NetworkX algorithms.
        """
        if not hasattr(self, '_nx_accessor'):
            self._nx_accessor = _NXBackendAccessor(self)
        return self._nx_accessor

    ## igraph backend accessor

    @property
    def ig(self):
        """Igraph backend accessor.

        Returns
        -------
        _IGBackendAccessor
            Accessor exposing igraph algorithms.
        """
        if not hasattr(self, '_ig_accessor'):
            self._ig_accessor = _IGBackendAccessor(self)
        return self._ig_accessor

    ## graph-tool backend accessor

    @property
    def gt(self):
        """graph-tool backend accessor.

        Returns
        -------
        _GTBackendAccessor
            Accessor exposing graph-tool algorithms.
        """
        if not hasattr(self, '_gt_accessor'):
            self._gt_accessor = _GTBackendAccessor(self)
        return self._gt_accessor

    # AnnNet API

    def X(self):
        """Sparse incidence matrix.

        Returns
        -------
        scipy.sparse.dok_matrix
        """
        return self._matrix

    @property
    def obs(self):
        """Vertex attribute table (observations).

        Returns
        -------
        DataFrame-like
        """
        return self.vertex_attributes

    @property
    def var(self):
        """Edge attribute table (variables).

        Returns
        -------
        DataFrame-like
        """
        return self.edge_attributes

    @property
    def uns(self):
        """Unstructured metadata.

        Returns
        -------
        dict
        """
        return self.graph_attributes

    @property
    def slices(self):
        """Slice operations manager.

        Returns
        -------
        SliceManager
        """
        if not hasattr(self, '_slice_manager'):
            self._slice_manager = SliceManager(self)
        return self._slice_manager

    @property
    def layers(self):
        """Layer operations manager.

        Returns
        -------
        LayerManager
        """
        if not hasattr(self, '_layer_manager'):
            self._layer_manager = LayerManager(self)
        return self._layer_manager

    @property
    def idx(self):
        """Index lookups (entity_id<->row, edge_id<->col).

        Returns
        -------
        IndexManager
        """
        if not hasattr(self, '_index_manager'):
            self._index_manager = IndexManager(self)
        return self._index_manager

    @property
    def cache(self):
        """Cache management (CSR/CSC materialization).

        Returns
        -------
        CacheManager
        """
        if not hasattr(self, '_cache_manager'):
            self._cache_manager = CacheManager(self)
        return self._cache_manager

    # I/O
    def write(self, path, **kwargs):
        """Write the graph to the native ``.annnet`` format.

        Parameters
        ----------
        path : str | pathlib.Path
            Output file path.
        **kwargs
            Passed to `annnet.io.annnet_format.write`.
        """
        from ..io.annnet_format import write

        write(self, path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        """Read a graph from the native ``.annnet`` format.

        Parameters
        ----------
        path : str | pathlib.Path
            Input file path.
        **kwargs
            Passed to `annnet.io.annnet_format.read`.

        Returns
        -------
        AnnNet
        """
        from ..io.annnet_format import read

        return read(path, **kwargs)

    # View API
    def view(self, vertices=None, edges=None, slices=None, predicate=None):
        """Create a lazy graph view.

        Parameters
        ----------
        vertices : Iterable[str], optional
            Vertex IDs to include.
        edges : Iterable[str], optional
            Edge IDs to include.
        slices : Iterable[str], optional
            Slice IDs to include.
        predicate : callable, optional
            Predicate used for additional filtering.

        Returns
        -------
        GraphView
        """
        return GraphView(self, vertices, edges, slices, predicate)

    # Audit
    def snapshot(self, label=None):
        """Capture a lightweight snapshot of the current graph state.

        Parameters
        ----------
        label : str, optional
            Human-readable label for the snapshot. Auto-generated if None.

        Returns
        -------
        dict
            Snapshot metadata record.
        """
        if label is None:
            label = f'snapshot_{len(self._snapshots)}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        snapshot = {
            'label': label,
            'version': self._version,
            'timestamp': datetime.now(UTC).isoformat(),
            'counts': {
                'vertices': self.nv,
                'edges': self.ne,
                'slices': len(self._slices),
            },
            # Store minimal state for comparison (uses existing AnnNet attributes)
            'vertex_ids': set(
                eid[0] if isinstance(eid, tuple) else eid
                for eid, r in self._entities.items()
                if r.kind == 'vertex'
            ),
            'edge_ids': set(self._col_to_edge.values()),
            'slice_ids': set(self._slices.keys()),
        }

        self._snapshots.append(snapshot)
        return snapshot

    def diff(self, a, b=None):
        """Compare two snapshots or compare snapshot with current state.

        Parameters
        ----------
        a : str | dict | AnnNet
            First snapshot (label, snapshot dict, or AnnNet instance).
        b : str | dict | AnnNet | None
            Second snapshot. If None, compare with current state.

        Returns
        -------
        GraphDiff
            Difference object with added/removed entities.
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
                if snap['label'] == ref:
                    return snap
            raise ValueError(f"Snapshot '{ref}' not found")
        elif isinstance(ref, AnnNet):
            # Create snapshot from another graph (uses AnnNet attributes)
            return {
                'label': 'external',
                'version': ref._version,
                'vertex_ids': set(eid for eid, r in ref._entities.items() if r.kind == 'vertex'),
                'edge_ids': set(ref._col_to_edge.values()),
                'slice_ids': set(ref._slices.keys()),
            }
        else:
            raise TypeError(f'Invalid snapshot reference: {type(ref)}')

    def _current_snapshot(self):
        """Create snapshot of current state (uses AnnNet attributes)."""
        return {
            'label': 'current',
            'version': self._version,
            'vertex_ids': set(
                eid[0] if isinstance(eid, tuple) else eid
                for eid, r in self._entities.items()
                if r.kind == 'vertex'
            ),
            'edge_ids': set(self._col_to_edge.values()),
            'slice_ids': set(self._slices.keys()),
        }

    def list_snapshots(self):
        """List all snapshots.

        Returns
        -------
        list[dict]
            Snapshot metadata.
        """
        return [
            {
                'label': snap['label'],
                'timestamp': snap['timestamp'],
                'version': snap['version'],
                'counts': snap['counts'],
            }
            for snap in self._snapshots
        ]

    # -------------------------------------------------------------------------
    # aspects / elem_layers — thin read/write properties backed by _aspects / _layers
    # -------------------------------------------------------------------------

    @property
    def aspects(self) -> list[str]:
        """Aspect names for this multilayer graph (empty list for flat graphs)."""
        if self._aspects == ('_',):
            return []
        return list(self._aspects)

    @aspects.setter
    def aspects(self, val: list[str]):
        if not val:
            self._aspects = ('_',)
            if '_' not in self._layers:
                self._layers = {'_': {'_'}}
        else:
            self._aspects = tuple(val)
            for a in self._aspects:
                if a not in self._layers:
                    self._layers[a] = set()

    @property
    def elem_layers(self) -> dict[str, list[str]]:
        """Elementary layer labels per aspect (empty dict for flat graphs)."""
        if self._aspects == ('_',):
            return {}
        return {k: sorted(v) for k, v in self._layers.items() if k != '_'}

    @elem_layers.setter
    def elem_layers(self, val: dict[str, list[str]]):
        if not val:
            self._layers = {'_': {'_'}}
        else:
            self._layers = {k: set(v) for k, v in val.items()}

    # -------------------------------------------------------------------------
    # edge_layers — backed by EdgeRecord.ml_layers
    # -------------------------------------------------------------------------

    @property
    def edge_layers(self) -> dict:
        """edge_id -> layer_tuple mapping (backed by EdgeRecord.ml_layers)."""
        proxy = getattr(self, '_compat_edge_layers', None)
        if proxy is None:
            proxy = EdgeLayersCompat(self)
            self._compat_edge_layers = proxy
        return proxy

    @edge_layers.setter
    def edge_layers(self, mapping):
        for edge_id, layers in dict(mapping).items():
            if edge_id in self._edges:
                self._edges[edge_id].ml_layers = layers

    # -------------------------------------------------------------------------
    # Compatibility properties (read-only views derived from SSOT structures)
    # These expose the old flat-dict interface that adapters and tests still use.
    # They are transitional: once all adapters are migrated, these will be removed.
    # -------------------------------------------------------------------------

    @property
    def entity_to_idx(self) -> dict:
        """entity_id -> row_idx compatibility mapping."""
        proxy = getattr(self, '_compat_entity_to_idx', None)
        if proxy is None:
            proxy = EntityToIdxCompat(self)
            self._compat_entity_to_idx = proxy
        return proxy

    @entity_to_idx.setter
    def entity_to_idx(self, mapping):
        self._entities.clear()
        self._row_to_entity.clear()
        self._vid_to_ekeys.clear()
        for entity_id, row_idx in dict(mapping).items():
            self._register_entity_record(
                entity_id, EntityRecord(row_idx=int(row_idx), kind='vertex')
            )

    @property
    def idx_to_entity(self) -> dict:
        """row_idx -> entity_id compatibility mapping."""
        proxy = getattr(self, '_compat_idx_to_entity', None)
        if proxy is None:
            proxy = IdxToEntityCompat(self)
            self._compat_idx_to_entity = proxy
        return proxy

    @idx_to_entity.setter
    def idx_to_entity(self, mapping):
        self._row_to_entity = {int(k): v for k, v in dict(mapping).items()}
        for row_idx, entity_id in self._row_to_entity.items():
            rec = self._entities.get(entity_id)
            kind = rec.kind if rec is not None else 'vertex'
            self._entities[entity_id] = EntityRecord(row_idx=row_idx, kind=kind)
        self._rebuild_entity_indexes()

    @property
    def entity_types(self) -> dict:
        """entity_id -> kind compatibility mapping."""
        proxy = getattr(self, '_compat_entity_types', None)
        if proxy is None:
            proxy = EntityTypesCompat(self)
            self._compat_entity_types = proxy
        return proxy

    @entity_types.setter
    def entity_types(self, mapping):
        for entity_id, kind in dict(mapping).items():
            self.entity_types[entity_id] = kind

    @property
    def edge_to_idx(self) -> dict:
        """edge_id -> col_idx compatibility mapping."""
        proxy = getattr(self, '_compat_edge_to_idx', None)
        if proxy is None:
            proxy = EdgeToIdxCompat(self)
            self._compat_edge_to_idx = proxy
        return proxy

    @edge_to_idx.setter
    def edge_to_idx(self, mapping):
        self._col_to_edge.clear()
        for edge_id, col_idx in dict(mapping).items():
            self.edge_to_idx[edge_id] = col_idx

    @property
    def idx_to_edge(self) -> dict:
        """col_idx -> edge_id compatibility mapping."""
        proxy = getattr(self, '_compat_idx_to_edge', None)
        if proxy is None:
            proxy = IdxToEdgeCompat(self)
            self._compat_idx_to_edge = proxy
        return proxy

    @idx_to_edge.setter
    def idx_to_edge(self, mapping):
        self._col_to_edge.clear()
        for col_idx, edge_id in dict(mapping).items():
            self.idx_to_edge[int(col_idx)] = edge_id

    @property
    def edge_weights(self) -> dict:
        """edge_id -> weight compatibility mapping."""
        proxy = getattr(self, '_compat_edge_weights', None)
        if proxy is None:
            proxy = EdgeWeightsCompat(self)
            self._compat_edge_weights = proxy
        return proxy

    @edge_weights.setter
    def edge_weights(self, mapping):
        for edge_id, weight in dict(mapping).items():
            self.edge_weights[edge_id] = weight

    @property
    def edge_directed(self) -> dict:
        """edge_id -> directed compatibility mapping."""
        proxy = getattr(self, '_compat_edge_directed', None)
        if proxy is None:
            proxy = EdgeDirectedCompat(self)
            self._compat_edge_directed = proxy
        return proxy

    @edge_directed.setter
    def edge_directed(self, mapping):
        for edge_id, directed in dict(mapping).items():
            self.edge_directed[edge_id] = directed

    @property
    def edge_definitions(self) -> dict:
        """edge_id -> (src, tgt, etype) compatibility mapping."""
        proxy = getattr(self, '_compat_edge_definitions', None)
        if proxy is None:
            proxy = EdgeDefinitionsCompat(self)
            self._compat_edge_definitions = proxy
        return proxy

    @edge_definitions.setter
    def edge_definitions(self, mapping):
        for edge_id, defn in dict(mapping).items():
            self.edge_definitions[edge_id] = defn

    @property
    def hyperedge_definitions(self) -> dict:
        """edge_id -> hyper metadata compatibility mapping."""
        proxy = getattr(self, '_compat_hyperedge_definitions', None)
        if proxy is None:
            proxy = HyperedgeDefinitionsCompat(self)
            self._compat_hyperedge_definitions = proxy
        return proxy

    @hyperedge_definitions.setter
    def hyperedge_definitions(self, mapping):
        items = dict(mapping).items() if isinstance(mapping, dict) else []
        for edge_id, defn in items:
            self.hyperedge_definitions[edge_id] = defn

    @property
    def _num_entities(self) -> int:
        """Number of entities (vertices + edge-entities)."""
        return len(self._entities)

    @_num_entities.setter
    def _num_entities(self, value) -> None:
        return None

    @property
    def _num_edges(self) -> int:
        """Number of edges with a matrix column."""
        return len(self._col_to_edge)

    @_num_edges.setter
    def _num_edges(self, value) -> None:
        return None

    @property
    def edge_direction_policy(self) -> dict:
        """edge_id -> direction policy compatibility mapping."""
        proxy = getattr(self, '_compat_edge_direction_policy', None)
        if proxy is None:
            proxy = EdgeDirectionPolicyCompat(self)
            self._compat_edge_direction_policy = proxy
        return proxy

    @edge_direction_policy.setter
    def edge_direction_policy(self, mapping):
        for edge_id, policy in dict(mapping).items():
            self.edge_direction_policy[edge_id] = policy

    @property
    def edge_kind(self) -> dict:
        """Compatibility mapping combining structural hyperedges and ML edge kinds."""
        proxy = getattr(self, '_compat_edge_kind', None)
        if proxy is None:
            proxy = EdgeKindCompat(self)
            self._compat_edge_kind = proxy
        return proxy

    @edge_kind.setter
    def edge_kind(self, mapping):
        for edge_id, kind in dict(mapping).items():
            self.edge_kind[edge_id] = kind
