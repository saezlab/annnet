from __future__ import annotations

import json
import time
from typing import Any
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, MutableMapping

import numpy as np
import scipy.sparse as sp

from ._Ops import Operations, OperationsAccessor
from ._Views import GraphView, ViewsClass, ViewsAccessor
from ._Layers import LayerAccessor
from ._Matrix import CacheManager, IndexManager, IndexMapping
from ._Slices import SliceManager
from ._History import History, HistoryAccessor
from ._records import (
    _EDGE_RESERVED,
    EdgeView,
    EdgeRecord,
    SliceRecord,
    EntityRecord,
    _slice_RESERVED,
    _vertex_RESERVED,
    _df_filter_not_equal,
    _external_entity_kind,
    _internal_entity_kind,
)
from ._Annotation import AttributesClass, AttributesAccessor
from .backend_accessors import _GTBackendAccessor, _IGBackendAccessor, _NXBackendAccessor
from .._dataframe_backend import (
    _is_polars_df,
    empty_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_drop_rows,
    dataframe_upsert_rows,
    polars_upsert_vertices,
    select_dataframe_backend,
)
from ..algorithms.traversal import Traversal

# ===================================


def _sanitize(v):
    if isinstance(v, (list, tuple, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v


class _EdgeRecordFieldMap(MutableMapping):
    """Mutable mapping view over one field on ``AnnNet._edges`` records."""

    def __init__(self, graph, field_name, *, include, getter=None, setter=None):
        self._graph = graph
        self._field_name = field_name
        self._include = include
        self._getter = getter
        self._setter = setter

    def __getitem__(self, key):
        rec = self._graph._edges[key]
        value = getattr(rec, self._field_name)
        if not self._include(rec, value):
            raise KeyError(key)
        return self._getter(rec, value) if self._getter else value

    def __setitem__(self, key, value):
        if key not in self._graph._edges:
            raise KeyError(key)
        rec = self._graph._edges[key]
        if self._setter:
            self._setter(rec, value)
        else:
            setattr(rec, self._field_name, value)

    def __delitem__(self, key):
        if key not in self._graph._edges:
            raise KeyError(key)
        setattr(self._graph._edges[key], self._field_name, None)

    def __iter__(self):
        for eid, rec in self._graph._edges.items():
            value = getattr(rec, self._field_name)
            if self._include(rec, value):
                yield eid

    def __len__(self):
        return sum(1 for _ in self.__iter__())


_BINARY_BATCH_RESERVED_KEYS = frozenset(
    {
        'source',
        'target',
        'src',
        'tgt',
        'edge_id',
        'slice',
        'weight',
        'edge_directed',
        'directed',
        'edge_type',
        'propagate',
        'flexible',
        'attributes',
        'attrs',
        'slice_weight',
    }
)

_HYPER_BATCH_RESERVED_KEYS = frozenset(
    {
        'members',
        'head',
        'tail',
        'edge_id',
        'slice',
        'weight',
        'edge_directed',
        'directed',
        'attributes',
        'attrs',
    }
)


class AnnNetMeta(type):
    """Metaclass exposing the compact public AnnNet API to introspection."""

    def __dir__(cls):
        api = getattr(cls, '_PUBLIC_API', ())
        return sorted(set(api))


class _BlockedLegacyAttribute:
    """Descriptor that hides removed flat API names without global attr overhead."""

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        raise AttributeError(
            f"AnnNet no longer exposes '{self.name}' directly; use the appropriate namespace or canonical API instead."
        )


class AnnNet(
    Operations,
    History,
    ViewsClass,
    IndexMapping,
    AttributesClass,
    Traversal,
    metaclass=AnnNetMeta,
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
        prefers the first installed supported backend.
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
    add_vertices
    add_edges
    view
    """

    _PUBLIC_API = (
        'add_vertices',
        'add_edges',
        'remove_vertices',
        'remove_edges',
        'has_vertex',
        'has_edge',
        'vertices',
        'edges',
        'degree',
        'incident_edges',
        'num_vertices',
        'num_edges',
        'nv',
        'ne',
        'number_of_vertices',
        'number_of_edges',
        'shape',
        'V',
        'E',
        'obs',
        'var',
        'uns',
        'layers',
        'slices',
        'attrs',
        'views',
        'history',
        'ops',
        'idx',
        'cache',
        'nx',
        'ig',
        'gt',
        'read',
        'write',
        'view',
        'global_count',
        'get_vertex',
        'get_edge',
        'edge_list',
        'make_undirected',
        'X',
        'is_multilayer',
    )

    _BLOCKED_LEGACY_API = frozenset(
        {
            'add_vertex',
            'add_edge',
            'add_slice',
            'remove_slice',
            'set_active_slice',
            'get_active_slice',
            'get_slices_dict',
            'list_slices',
            'has_slice',
            'slice_count',
            'get_slice_info',
            'get_slice_vertices',
            'get_slice_edges',
            'slice_union',
            'slice_intersection',
            'slice_difference',
            'create_slice_from_operation',
            'create_aggregated_slice',
            'slice_statistics',
            'vertex_presence_across_slices',
            'edge_presence_across_slices',
            'hyperedge_presence_across_slices',
            'conserved_edges',
            'slice_specific_edges',
            'temporal_dynamics',
            'set_graph_attribute',
            'get_graph_attribute',
            'get_graph_attributes',
            'set_vertex_attrs',
            'set_vertex_attrs_bulk',
            'get_vertex_attrs',
            'get_attr_vertex',
            'get_attr_vertices',
            'set_edge_attrs',
            'set_edge_attrs_bulk',
            'get_edge_attrs',
            'get_attr_edge',
            'get_attr_edges',
            'get_attr_from_edges',
            'get_edges_by_attr',
            'set_slice_attrs',
            'get_slice_attr',
            'set_edge_slice_attrs',
            'set_edge_slice_attrs_bulk',
            'get_edge_slice_attr',
            'set_slice_edge_weight',
            'get_effective_edge_weight',
            'audit_attributes',
            'edges_view',
            'vertices_view',
            'slices_view',
            'aspects_view',
            'layers_view',
            'enable_history',
            'clear_history',
            'export_history',
            'mark',
            'snapshot',
            'list_snapshots',
            'diff',
            'subgraph',
            'edge_subgraph',
            'extract_subgraph',
            'copy',
            'reverse',
            'memory_usage',
            'vertex_incidence_matrix',
            'get_vertex_incidence_matrix_as_lists',
            'set_aspects',
            'set_elementary_layers',
            'add_elementary_layer',
            'flatten_layers',
            'has_presence',
            'iter_layers',
            'iter_vertex_layers',
            'layer_id_to_tuple',
            'layer_tuple_to_id',
            'supra_adjacency',
            'supra_incidence',
            'build_intra_block',
            'build_inter_block',
            'build_coupling_block',
            'layer_vertex_set',
            'layer_edge_set',
            'layer_union',
            'layer_intersection',
            'layer_difference',
            'set_aspect_attrs',
            'get_aspect_attrs',
            'set_layer_attrs',
            'get_layer_attrs',
            'set_vertex_layer_attrs',
            'get_vertex_layer_attrs',
            'set_elementary_layer_attrs',
            'get_elementary_layer_attrs',
            'list_aspects',
            'list_layers',
        }
    )

    # Construction
    def __init__(
        self,
        directed: bool | None = None,
        v: int = 0,
        e: int = 0,
        annotations: dict[str, Any] | None = None,
        annotations_backend: str = 'polars',
        aspects: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
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
        annotations_backend : {"polars", "pandas"}, optional
            Backend used when empty annotation tables need to be created.
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

        # --- Adjacency indices (src/tgt → edge lists, maintained incrementally) ---
        self._src_to_edges: dict = {}  # src -> [edge_id, ...]
        self._tgt_to_edges: dict = {}  # tgt -> [edge_id, ...]
        self._pair_to_edges: dict = {}  # (src, tgt) -> [edge_id, ...] for binary edges
        self._edge_indexes_built: bool = True

        # --- Composite vertex key support ---
        self._vertex_key_fields = None  # tuple[str, ...] | None
        self._vertex_key_index: dict = {}  # key_tuple -> vertex_id

        # --- Sparse incidence matrix ---
        v = int(v) if v and v > 0 else 0
        e = int(e) if e and e > 0 else 0
        self._matrix = sp.dok_matrix((v, e), dtype=np.float32)
        self._csr_cache = None

        # --- Attribute storage ---
        self._annotations_backend = select_dataframe_backend(annotations_backend)
        self._init_annotation_tables(annotations)
        self.graph_attributes: dict = {}
        self.graph_attributes.update(kwargs)

        # --- Edge ID counter ---
        self._next_edge_id = 0

        # --- Slice state ---
        self._slices: dict = {}  # slice_id -> SliceRecord
        self._current_slice: str | None = None
        self._default_slice: str = 'default'
        self.slice_edge_weights: defaultdict = defaultdict(dict)  # slice_id -> {edge_id: weight}
        self._slices[self._default_slice] = SliceRecord()
        self._current_slice = self._default_slice

        # --- History ---
        self._history_enabled = True
        self._history: list = []
        self._version = 0
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()
        self.history = HistoryAccessor(self)
        self._snapshots: list = []

        # --- Multilayer state ---
        self.vertex_aligned: bool = False
        # Multilayer edge metadata — stored in EdgeRecord.ml_kind / ml_layers (canonical).
        # edge_kind / edge_layers properties are thin compat proxies over EdgeRecord.

        # Build the cartesian-product layer cache from the aspects/layers given
        # at construction. set_aspects/set_elementary_layers do this on mutation;
        # the constructor must do it too, otherwise iter_layers/layer_edge_set/
        # subgraph_from_layer_tuple all see an empty layer space.
        self._rebuild_all_layers_cache()

    def _grow_rows_to(self, target: int):
        """Grow incidence rows geometrically to accommodate ``target`` rows."""
        rows, cols = self._matrix.shape
        if target > rows:
            new_rows = max(target, rows + max(8, rows >> 1))
            self._matrix.resize((new_rows, cols))

    def _grow_cols_to(self, target: int):
        """Grow incidence columns geometrically to accommodate ``target`` cols."""
        rows, cols = self._matrix.shape
        if target > cols:
            new_cols = max(target, cols + max(8, cols >> 1))
            self._matrix.resize((rows, new_cols))

    def _invalidate_sparse_caches(self, formats=None):
        """Invalidate all derived sparse cache views behind one internal hook."""
        if formats is None:
            formats = ('csr', 'csc', 'adjacency')
        else:
            formats = tuple(formats)

        if 'csr' in formats:
            self._csr_cache = None

        cache_manager = getattr(self, '_cache_manager', None)
        if cache_manager is not None:
            cache_manager.invalidate(list(formats))

    def _rebuild_slice_edge_weights_cache(self):
        cache = defaultdict(dict)
        df = self.edge_slice_attributes
        if df is None:
            self.slice_edge_weights = cache
            return

        cols = set(dataframe_columns(df))
        if not {'slice_id', 'edge_id', 'weight'} <= cols:
            self.slice_edge_weights = cache
            return

        rows = dataframe_to_rows(df)
        for row in rows:
            lid = row.get('slice_id')
            eid = row.get('edge_id')
            w = row.get('weight')
            if lid is None or eid is None or w is None:
                continue
            if isinstance(w, float) and np.isnan(w):
                continue
            cache[lid][eid] = float(w)

        self.slice_edge_weights = cache

    def _sync_slice_edge_weights_for_rows(self, slice_id, rows):
        """Incrementally refresh the compatibility slice-weight cache for written rows."""
        if not isinstance(self.slice_edge_weights, defaultdict):
            self.slice_edge_weights = defaultdict(dict, self.slice_edge_weights)

        bucket = self.slice_edge_weights[slice_id]
        touched = set()
        for row in rows:
            eid = row.get('edge_id')
            if eid is not None:
                touched.add(eid)

        for eid in touched:
            bucket.pop(eid, None)

        for row in rows:
            eid = row.get('edge_id')
            weight = row.get('weight')
            if eid is None or weight is None:
                continue
            try:
                import math as _math

                if isinstance(weight, float) and _math.isnan(weight):
                    continue
            except TypeError:
                pass
            bucket[eid] = float(weight)

        if not bucket:
            self.slice_edge_weights.pop(slice_id, None)

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

    def __dir__(self):
        return sorted(set(self._PUBLIC_API))

    def __repr__(self) -> str:
        """Anndata-style multi-line summary."""
        lines = [
            f'AnnNet object with n_vertices × n_edges = {self.nv} × {self.ne}',
            f'    directed: {self.directed}',
        ]

        slice_ids = list(self._slices.keys())
        if slice_ids:
            lines.append(f'    slices: {slice_ids}')

        if self._aspects and self._aspects != ('_',):
            lines.append(f'    aspects: {list(self._aspects)}')

        def _user_cols(df, id_field: str) -> list[str]:
            try:
                cols = [c for c in dataframe_columns(df) if c != id_field]
            except (AttributeError, TypeError):
                return []
            return cols

        obs_cols = _user_cols(self.vertex_attributes, 'vertex_id')
        if obs_cols:
            lines.append(f'    obs: {obs_cols!r}')

        var_cols = _user_cols(self.edge_attributes, 'edge_id')
        if var_cols:
            lines.append(f'    var: {var_cols!r}')

        uns_keys = list(self.graph_attributes.keys()) if self.graph_attributes else []
        if uns_keys:
            lines.append(f'    uns: {uns_keys!r}')

        return '\n'.join(lines)

    def __len__(self) -> int:
        """Number of vertices (NetworkX convention)."""
        return self.nv

    def __iter__(self) -> Iterator[str]:
        """Iterate over vertex IDs (NetworkX convention)."""
        return iter(self.vertices())

    def __contains__(self, item) -> bool:
        """Membership test on vertex IDs. ``edge in G`` is not supported."""
        try:
            ekey = self._resolve_entity_key(item)
        except (KeyError, TypeError, ValueError):
            return False
        rec = self._entities.get(ekey)
        return rec is not None and rec.kind == 'vertex'

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
            "alice"               -> resolves only if exactly one supra-node exists for "alice";
                                     if none exists, returns the placeholder-layer key for deferred creation;
                                     if multiple exist, raises ValueError and requires an explicit
                                     ``(vertex_id, layer_coord)`` tuple.
            ("alice", ("t1",))    -> ("alice", ("t1",))  [validated]
        """
        if isinstance(vid_or_key, str):
            is_flat = self._aspects == ('_',)
            if is_flat:
                return (vid_or_key, ('_',))
            matches = []
            for ekey in self._vid_to_ekeys.get(vid_or_key, ()):
                rec = self._entities.get(ekey)
                if rec is None:
                    continue
                matches.append((rec.row_idx, ekey))
            if len(matches) == 1:
                return matches[0][1]
            if len(matches) > 1:
                choices = [ekey for _, ekey in sorted(matches)]
                raise ValueError(
                    f'Ambiguous bare vertex_id {vid_or_key!r} in multilayer graph; '
                    f'use an explicit (vertex_id, layer_coord) tuple. Choices: {choices!r}'
                )
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

    def _rebuild_edge_indexes(self) -> None:
        """Rebuild adjacency-derived edge indexes from canonical edge records."""
        self._src_to_edges = {}
        self._tgt_to_edges = {}
        self._pair_to_edges = {}
        for eid, rec in self._edges.items():
            if rec.etype == 'hyper' or rec.src is None or rec.tgt is None:
                continue
            self._src_to_edges.setdefault(rec.src, []).append(eid)
            self._tgt_to_edges.setdefault(rec.tgt, []).append(eid)
            self._pair_to_edges.setdefault((rec.src, rec.tgt), []).append(eid)
        self._edge_indexes_built = True

    def _ensure_edge_indexes(self) -> None:
        """Materialize adjacency-derived edge indexes on demand."""
        if not getattr(self, '_edge_indexes_built', True):
            self._rebuild_edge_indexes()

    def _edge_ids_for_pair(self, source, target) -> list[str]:
        """Return edge ids for a binary endpoint pair from canonical src-edge buckets."""
        self._ensure_edge_indexes()
        eids = []
        for eid in self._src_to_edges.get(source, ()):
            rec = self._edges.get(eid)
            if rec is not None and rec.etype != 'hyper' and rec.tgt == target:
                eids.append(eid)
        if eids:
            self._pair_to_edges[(source, target)] = list(eids)
        else:
            self._pair_to_edges.pop((source, target), None)
        return eids

    def _index_edge_pair(self, edge_id, src, tgt) -> None:
        if src is None or tgt is None:
            return
        bucket = self._pair_to_edges.setdefault((src, tgt), [])
        if edge_id not in bucket:
            bucket.append(edge_id)

    def _unindex_edge_pair(self, edge_id, src, tgt) -> None:
        if src is None or tgt is None:
            return
        bucket = self._pair_to_edges.get((src, tgt))
        if not bucket:
            return
        try:
            bucket.remove(edge_id)
        except ValueError:
            return
        if not bucket:
            self._pair_to_edges.pop((src, tgt), None)

    # Aspect / layer registry queries

    @property
    def is_multilayer(self) -> bool:
        """Whether the graph has declared multilayer aspects.

        Returns
        -------
        bool
            ``True`` when the graph has user-declared aspects. Flat graphs use
            the internal sentinel aspect ``"_"`` and return ``False``.
        """
        return self._aspects != ('_',)

    @property
    def _V(self) -> set:
        """Set of all vertex IDs (pure strings, derived from _entities)."""
        return {ekey[0] for ekey, rec in self._entities.items() if rec.kind == 'vertex'}

    @property
    def _VM(self) -> set:
        """Set of (vertex_id, layer_coord) supra-node pairs (derived from _entities)."""
        return {ekey for ekey, rec in self._entities.items() if rec.kind == 'vertex'}

    @_VM.setter
    def _VM(self, value) -> None:
        """Compatibility sink for legacy restore paths.

        The canonical source of truth is ``self._entities``. Some IO/adapters
        still assign ``G._VM = vm_set`` after calling ``_restore_supra_nodes``.
        Accept the assignment without replacing the derived property.
        """
        if value:
            try:
                self._restore_supra_nodes(value)
            except Exception:  # noqa: BLE001
                pass

    # Build graph

    def add_vertices(
        self,
        vertices: str | dict[str, Any] | tuple[str, dict[str, Any]] | Iterable[Any],
        slice: str | None = None,
        layer: str | tuple[str, ...] | dict[str, str] | None = None,
        **attributes: Any,
    ) -> str | list[str]:
        """Add one vertex or many vertices.

        This is the canonical public entry point for vertex creation. Use it
        for both single-vertex and batch insertion.

        Parameters
        ----------
        vertices : str | dict | tuple | Iterable
            Vertex specification or iterable of specifications.

            Accepted single-vertex forms are:

            - ``"A"``
            - ``{"vertex_id": "A", "kind": "gene"}``
            - ``{"id": "A", ...}``
            - ``{"name": "A", ...}``
            - ``("A", {"kind": "gene"})``

            Accepted batch forms are iterables of the same specifications.
        slice : str, optional
            Slice receiving the inserted vertices. If omitted, the active slice
            is used.
        layer : str | tuple | dict, optional
            Layer coordinate for inserted vertices in multilayer graphs. A
            string is valid only for single-aspect graphs; a tuple must already
            be in aspect order; a dict maps aspect name to layer value.
        **attributes
            Attributes applied to a single vertex. These are merged with
            attributes in ``vertices`` when ``vertices`` is a single vertex.

        Returns
        -------
        str | list[str]
            The inserted vertex ID for a single vertex, or a list of vertex IDs
            for batch insertion.

        Raises
        ------
        ValueError
            If a dictionary vertex specification does not contain
            ``"vertex_id"``, ``"id"``, or ``"name"``.

        Notes
        -----
        Vertex attributes are stored in :attr:`obs` and can be edited through
        :attr:`attrs`. In multilayer graphs, omitting ``layer`` places the
        vertex on the placeholder layer coordinate.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_vertices('A', kind='gene')
        'A'
        >>> G.add_vertices(
        ...     [
        ...         {'vertex_id': 'B', 'kind': 'protein'},
        ...         ('C', {'kind': 'metabolite'}),
        ...     ]
        ... )
        ['B', 'C']
        """
        is_single = False
        if isinstance(vertices, (str, bytes, dict)):
            is_single = True
        elif isinstance(vertices, tuple) and vertices:
            is_single = len(vertices) == 1 or (
                len(vertices) == 2
                and isinstance(vertices[0], str)
                and isinstance(vertices[1], dict)
            )

        if is_single:
            if isinstance(vertices, dict):
                if vertices.get('vertex_id') is not None:
                    vertex_id = vertices['vertex_id']
                    attrs = {k: v for k, v in vertices.items() if k != 'vertex_id'}
                elif vertices.get('id') is not None:
                    vertex_id = vertices['id']
                    attrs = {k: v for k, v in vertices.items() if k != 'id'}
                elif vertices.get('name') is not None:
                    vertex_id = vertices['name']
                    attrs = {k: v for k, v in vertices.items() if k != 'name'}
                else:
                    raise ValueError('vertex dict must contain one of: vertex_id, id, name')
            elif isinstance(vertices, tuple):
                vertex_id = vertices[0]
                attrs = vertices[1] if len(vertices) > 1 and isinstance(vertices[1], dict) else {}
            else:
                vertex_id = vertices
                attrs = {}
            attrs.update(attributes)
            return self._add_vertex_impl(vertex_id, slice=slice, layer=layer, **attrs)

        items = list(vertices)
        self._add_vertices_batch(items, layer=layer, slice=slice, default_attrs=attributes or None)
        out = []
        for it in items:
            if isinstance(it, dict):
                if it.get('vertex_id') is not None:
                    out.append(it['vertex_id'])
                elif it.get('id') is not None:
                    out.append(it['id'])
                elif it.get('name') is not None:
                    out.append(it['name'])
            elif isinstance(it, (tuple, list)) and it:
                out.append(it[0])
            else:
                out.append(it)
        return out

    def add_vertex(
        self,
        vertex_id: str,
        slice: str | None = None,
        layer: str | tuple[str, ...] | dict[str, str] | None = None,
        **attributes: Any,
    ) -> str:
        """Compatibility wrapper for the canonical ``add_vertices`` API."""
        return self._add_vertex_impl(vertex_id, slice=slice, layer=layer, **attributes)

    def add_vertices_bulk(self, vertices, *, layer=None, slice=None):
        """Hidden compatibility shim for legacy internal bulk insertion."""
        items = list(vertices)
        self._add_vertices_batch(items, layer=layer, slice=slice)
        out = []
        for it in items:
            if isinstance(it, dict):
                if it.get('vertex_id') is not None:
                    out.append(it['vertex_id'])
                elif it.get('id') is not None:
                    out.append(it['id'])
                elif it.get('name') is not None:
                    out.append(it['name'])
            elif isinstance(it, (tuple, list)) and it:
                out.append(it[0])
            else:
                out.append(it)
        return out

    def _add_vertex_impl(self, vertex_id, slice=None, layer=None, **attributes):
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

        # Add to slice (slice tracks bare vid for backward compat with _Slices.py)
        slices = self._slices
        if slice not in slices:
            slices[slice] = SliceRecord()
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
                etype='edge_placeholder',
                col_idx=-1,
                ml_kind=None,
                ml_layers=None,
                direction_policy=None,
            )

        slice = slice or self._current_slice
        if slice is not None:
            self.slices._ensure_slice(slice)['edges'].add(edge_id)
        if attributes:
            self.attrs.set_edge_attrs(edge_id, **attributes)
        self._ensure_edge_row(edge_id)
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
        self._ensure_edge_indexes()
        if etype == 'binary':
            nodes = list(endpoint_set)
            a, b = (nodes[0], nodes[0]) if len(nodes) == 1 else (nodes[0], nodes[1])
            result = [eid for eid in self._src_to_edges.get(a, []) if self._edges[eid].tgt == b]
            if a != b:
                result.extend(
                    eid for eid in self._src_to_edges.get(b, []) if self._edges[eid].tgt == a
                )
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

    def add_edges(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> str | list[str]:
        """Add one edge, many edges, or hyperedges.

        This is the canonical public entry point for all edge creation. It
        handles binary edges, directed and undirected edges, edge-entities, and
        hyperedges through the shape of the input specification.

        Parameters
        ----------
        *args
            Edge specification. Common forms are:

            - ``G.add_edges("A", "B")``
            - ``G.add_edges("A", "B", weight=2.0, edge_id="e1")``
            - ``G.add_edges({"source": "A", "target": "B"})``
            - ``G.add_edges([{"source": "A", "target": "B"}, ...])``
            - ``G.add_edges([{"members": ["A", "B", "C"]}, ...])``
            - ``G.add_edges([{"tail": ["A"], "head": ["B", "C"]}, ...])``
            - ``G.add_edges([{"edge_id": "EE1", ...}, ...], as_entity=True)``
        **kwargs
            Options for single-edge or batch insertion.

        Other Parameters
        ----------------
        source, src : str
            Source endpoint for a binary edge.
        target, tgt : str
            Target endpoint for a binary edge.
        weight : float, default 1.0
            Incidence weight for the edge.
        edge_id : str, optional
            Explicit edge identifier. If omitted, an ``edge_N`` ID is assigned.
        directed : bool, optional
            Directedness for a single edge.
        edge_directed : bool, optional
            Directedness for edge specs in batch input.
        slice : str, optional
            Slice receiving the inserted edges. If omitted, the active slice is
            used.
        as_entity : bool, default False
            If ``True``, each created edge is also registered as an entity so it
            can be used as the endpoint of later edges. In batch mode, items
            that carry no ``source``/``target`` are treated as null-endpoint
            edge-entity placeholders and require this flag to be set.
        parallel : {"update", "error", "parallel"}, default "update"
            Policy for single-edge insertion when ``edge_id`` is not supplied
            and the same endpoints already have an edge. ``"update"`` reuses
            the existing edge; ``"parallel"`` creates an additional edge;
            ``"error"`` raises ``ValueError``. Ignored in batch mode.
        propagate : {"none", "shared", "all"}, default "none"
            Slice propagation policy. ``"shared"`` adds the edge to every slice
            containing both endpoints; ``"all"`` adds it to every slice
            containing either endpoint.
        flexible : dict, optional
            Data-driven direction policy. Requires keys ``"var"`` and
            ``"threshold"``. Single-edge path only.
        default_weight : float, default 1.0
            Batch default for edge specs without an explicit weight.
        default_edge_directed : bool, optional
            Batch default directedness.
        default_propagate : {"none", "shared", "all"}, default "none"
            Batch default propagation policy.
        default_edge_type : str, default "regular"
            Batch default edge type stored in the edge record.
        default_slice_weight : float, optional
            Batch per-slice weight override.

        Returns
        -------
        str | list[str]
            Edge ID for single-edge insertion, or a list of edge IDs for batch
            insertion.

        Raises
        ------
        TypeError
            If unsupported keyword arguments are supplied for batch insertion.
        ValueError
            If the edge specification is structurally invalid.

        Notes
        -----
        Hyperedges are detected from dictionaries containing ``"members"`` for
        undirected hyperedges or ``"head"``/``"tail"`` for directed hyperedges.
        Binary edges use ``"source"``/``"target"`` or ``"src"``/``"tgt"``.

        For a full guide covering all input forms, dispatch logic, parallel
        policy, propagation, flexible direction, and batch formats, see the
        [Adding edges](../explanations/add-edges.md) explanation page.

        Examples
        --------
        >>> G = AnnNet(directed=True)
        >>> G.add_vertices(['A', 'B', 'C'])
        ['A', 'B', 'C']
        >>> G.add_edges('A', 'B', edge_id='e1', weight=0.5)
        'e1'
        >>> G.add_edges(
        ...     [
        ...         {'source': 'B', 'target': 'C'},
        ...         {'members': ['A', 'B', 'C'], 'edge_id': 'h1'},
        ...     ]
        ... )
        ['edge_0', 'h1']
        """
        if 'edges' in kwargs and not args:
            batch = kwargs.pop('edges')
            args = (batch,)

        batch_candidate = None
        if len(args) == 1 and not kwargs.get('tgt') and 'src' not in kwargs:
            candidate = args[0]
            if not isinstance(candidate, (str, bytes, dict)):
                if isinstance(candidate, list):
                    if candidate and isinstance(candidate[0], (dict, tuple, list)):
                        batch_candidate = candidate
                elif isinstance(candidate, tuple):
                    if candidate and isinstance(candidate[0], (dict, tuple, list)):
                        batch_candidate = list(candidate)
                else:
                    try:
                        materialized = list(candidate)
                    except TypeError:
                        materialized = None
                    if materialized and isinstance(materialized[0], (dict, tuple, list)):
                        batch_candidate = materialized

        if batch_candidate is not None:
            default_slice = kwargs.pop('slice', None)
            as_entity = kwargs.pop('as_entity', False)
            default_weight = kwargs.pop('default_weight', 1.0)
            default_edge_type = kwargs.pop('default_edge_type', 'regular')
            default_propagate = kwargs.pop('default_propagate', 'none')
            default_slice_weight = kwargs.pop('default_slice_weight', None)
            default_edge_directed = kwargs.pop('default_edge_directed', None)
            if kwargs:
                unexpected = ', '.join(sorted(kwargs))
                raise TypeError(f'Unexpected keyword arguments for batch add_edges: {unexpected}')

            def _is_hyper_item(item):
                if not isinstance(item, dict):
                    return False
                # Internal/IO compat: legacy members/head/tail keys still mark a hyperedge.
                if any(k in item for k in ('members', 'head', 'tail')):
                    return True
                # User-facing rule: a list-shaped src or tgt means hyperedge.
                src_val = item.get('src', item.get('source'))
                tgt_val = item.get('tgt', item.get('target'))
                if isinstance(src_val, (list, tuple, set, frozenset)) and not isinstance(
                    src_val, str
                ):
                    return True
                if isinstance(tgt_val, (list, tuple, set, frozenset)) and not isinstance(
                    tgt_val, str
                ):
                    return True
                return False

            kinds = set()
            for item in batch_candidate:
                kinds.add('hyper' if _is_hyper_item(item) else 'binary')
                if len(kinds) > 1:
                    break

            if kinds == {'hyper'}:
                return self._add_hyperedges_batch(
                    batch_candidate,
                    slice=default_slice,
                    default_weight=default_weight,
                    default_edge_directed=default_edge_directed,
                )

            if kinds <= {'binary'}:
                return self._add_edges_batch(
                    batch_candidate,
                    slice=default_slice,
                    as_entity=as_entity,
                    default_weight=default_weight,
                    default_edge_type=default_edge_type,
                    default_propagate=default_propagate,
                    default_slice_weight=default_slice_weight,
                    default_edge_directed=default_edge_directed,
                )

            out = []
            for item in batch_candidate:
                if _is_hyper_item(item):
                    out.extend(
                        self._add_hyperedges_batch(
                            [item],
                            slice=default_slice,
                            default_weight=default_weight,
                            default_edge_directed=default_edge_directed,
                        )
                    )
                else:
                    out.extend(
                        self._add_edges_batch(
                            [item],
                            slice=default_slice,
                            as_entity=as_entity,
                            default_weight=default_weight,
                            default_edge_type=default_edge_type,
                            default_propagate=default_propagate,
                            default_slice_weight=default_slice_weight,
                            default_edge_directed=default_edge_directed,
                        )
                    )
            return out

        return self._add_edge_impl(*args, **kwargs)

    def add_edge(
        self,
        src: str | list[str] | dict[str, float] | tuple[str, tuple[str, ...]] | None = None,
        tgt: str | list[str] | dict[str, float] | tuple[str, tuple[str, ...]] | None = None,
        *,
        weight: float = 1.0,
        edge_id: str | None = None,
        directed: bool | None = None,
        parallel: str = 'update',
        slice: str | None = None,
        as_entity: bool = False,
        propagate: str = 'none',
        flexible: dict[str, Any] | None = None,
        **attrs: Any,
    ) -> str:
        """Compatibility wrapper for the canonical ``add_edges`` API."""
        return self._add_edge_impl(
            src,
            tgt,
            weight=weight,
            edge_id=edge_id,
            directed=directed,
            parallel=parallel,
            slice=slice,
            as_entity=as_entity,
            propagate=propagate,
            flexible=flexible,
            **attrs,
        )

    def add_edges_bulk(
        self,
        edges,
        *,
        slice=None,
        as_entity=False,
        default_weight=1.0,
        default_edge_type='regular',
        default_propagate='none',
        default_slice_weight=None,
        default_edge_directed=None,
    ):
        """Hidden compatibility shim for legacy internal bulk edge insertion."""
        return self._add_edges_batch(
            list(edges),
            slice=slice,
            as_entity=as_entity,
            default_weight=default_weight,
            default_edge_type=default_edge_type,
            default_propagate=default_propagate,
            default_slice_weight=default_slice_weight,
            default_edge_directed=default_edge_directed,
        )

    def _add_edge_impl(
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

        if self.is_multilayer and col_entries_literal is None:
            # Promote bare vertex strings to supra-node keys, mirroring
            # ``add_vertices``: warn + place on the placeholder layer when
            # the bare id has no existing supra-node. Ambiguous bare ids
            # (multiple supra-nodes) raise ValueError via
            # ``_resolve_entity_key``.

            def _promote(node_set):
                promoted = set()
                bare_to_placeholder = []
                for node in node_set:
                    if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                        promoted.add(node)
                        continue
                    ekey = self._resolve_entity_key(node)
                    promoted.add(ekey)
                    if ekey not in self._entities:
                        bare_to_placeholder.append(node)
                return promoted, bare_to_placeholder

            src_nodes, bare_src = _promote(src_nodes)
            tgt_nodes, bare_tgt = _promote(tgt_nodes)
            bare_total = bare_src + bare_tgt
            if bare_total:
                self._ensure_placeholder_layers_declared()
                self._warn_placeholder_vertex_assignment(bare_total, context='add_edges')

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
        self._ensure_edge_indexes()
        _ent = self._entities
        _edg = self._edges
        for node in endpoint_set:
            ekey = self._resolve_entity_key(node)
            if ekey not in _ent:
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                    self._add_vertex_impl(node[0], layer=node[1], slice=slice)
                else:
                    self._add_vertex_impl(node, slice=slice)

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
            if src_store is not None:
                self._src_to_edges.setdefault(src_store, []).append(edge_id)
            if tgt_store is not None:
                self._tgt_to_edges.setdefault(tgt_store, []).append(edge_id)
            if etype == 'binary':
                self._index_edge_pair(edge_id, src_store, tgt_store)
        else:
            rec = _edg[edge_id]
            old_src, old_tgt = rec.src, rec.tgt
            if (old_src, old_tgt) != (src_store, tgt_store):
                self._unindex_edge_pair(edge_id, old_src, old_tgt)
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
                if etype == 'binary':
                    self._index_edge_pair(edge_id, src_store, tgt_store)
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
                slices[slice] = SliceRecord()
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
            self.attrs.set_edge_attrs(edge_id, **attrs)

        # Ensure the edge has a row in ``var`` even when no user attrs were
        # supplied — keeps ``var.shape[0] == ne`` (anndata-style symmetry).
        self._ensure_edge_row(edge_id)

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

    def _propagate_to_shared_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to all slices that already contain **both** endpoints.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for _slice_id, slice_data in self._slices.items():
            if source in slice_data['vertices'] and target in slice_data['vertices']:
                slice_data['edges'].add(edge_id)

    def _propagate_to_all_slices(self, edge_id, source, target):
        """INTERNAL: Add an edge to any slice containing **either** endpoint.

        Inserts the missing endpoint into that slice when only one endpoint is
        already a member.

        Parameters
        ----------
        edge_id : str
        source : str
        target : str

        """
        for _slice_id, slice_data in self._slices.items():
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
        """Convert all existing edges to undirected form in place.

        Parameters
        ----------
        drop_flexible : bool, optional
            If ``True``, clear flexible-direction policies after rewriting
            edge incidence signs.
        update_default : bool, optional
            If ``True``, set ``G.directed = False`` so future edges are
            undirected unless explicitly overridden.

        Returns
        -------
        AnnNet
            The modified graph, returned for chaining.

        Notes
        -----
        Directed binary edges are rewritten from signed incidence
        ``(+w, -w)`` to unsigned incidence ``(+w, +w)``. Directed hyperedges are
        converted to undirected hyperedges over the union of their head and
        tail members.

        Examples
        --------
        >>> G = AnnNet(directed=True)
        >>> G.add_vertices(['A', 'B'])
        ['A', 'B']
        >>> G.add_edges('A', 'B')
        'edge_0'
        >>> G.make_undirected()
        AnnNet(...)
        """

        M = self._matrix

        # 1) Binary / vertex-edge edges
        for _eid, rec in list(self._edges.items()):
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

        # 2) Hyperedges
        for _eid, rec in list(self._edges.items()):
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

        See Also
        --------
        remove_edges : Remove one or more edges through the compact public API.
        """
        self._ensure_edge_indexes()
        if edge_id not in self._edges:
            raise KeyError(f'Edge {edge_id} not found')

        rec = self._edges[edge_id]
        col_idx = rec.col_idx
        if rec.etype != 'hyper':
            self._unindex_edge_pair(edge_id, rec.src, rec.tgt)

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
        self._invalidate_sparse_caches()

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

        self._rebuild_slice_edge_weights_cache()

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

        See Also
        --------
        remove_vertices : Remove one or more vertices through the compact
            public API.
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
        self._invalidate_sparse_caches()

        # Update entity mappings
        self._remove_entity_record(ekey)

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
        """Return the vertex identifier stored at an internal row index.

        Parameters
        ----------
        index : int
            Incidence-matrix row index.

        Returns
        -------
        str
            Vertex ID at ``index``.

        Raises
        ------
        KeyError
            If ``index`` is not a valid entity row.

        Notes
        -----
        This is an index-level lookup. Most user code should use
        :meth:`vertices` unless it specifically needs row-index mapping.
        """
        try:
            entry = self._row_to_entity[index]
        except KeyError:
            raise KeyError(
                f'No vertex at row index {index}; valid range is '
                f'[0, {self.nv}). Use G.vertices() to list vertex IDs.'
            ) from None
        return entry[0] if isinstance(entry, tuple) else entry

    def get_edge(self, index: int | str) -> EdgeView:
        """Return an :class:`EdgeView` for the requested edge.

        Parameters
        ----------
        index : int | str
            Incidence-matrix column index or edge ID.

        Returns
        -------
        EdgeView
            A tuple-shaped record. ``(source, target)`` tuple unpacking still
            works for backward compatibility; ``edge_id``, ``kind``,
            ``members``, ``weight`` and ``directed`` are also exposed as
            attributes.

        Raises
        ------
        KeyError
            If an edge ID is unknown.
        """
        if isinstance(index, str):
            eid = index
            if eid not in self._edges:
                raise KeyError(f'Unknown edge id: {eid}') from None
        else:
            eid = self._col_to_edge[index]

        rec = self._edges[eid]
        return self._edge_tuple_from_record(rec, eid=eid)

    def _edge_tuple_from_record(self, rec, *, eid: str | None = None):
        if rec.etype == 'hyper':
            if rec.tgt is not None:
                src_fs = frozenset(rec.src)
                tgt_fs = frozenset(rec.tgt)
                members = src_fs | tgt_fs
                kind = 'hyper_directed'
                source, target, directed = src_fs, tgt_fs, True
            else:
                M = frozenset(rec.src)
                members = M
                kind = 'hyper_undirected'
                source, target, directed = M, M, False
        elif rec.etype in ('vertex_edge', 'edge_placeholder'):
            u, v = rec.src, rec.tgt
            members = frozenset(x for x in (u, v) if x is not None)
            kind = rec.etype
            source = frozenset([u]) if u is not None else frozenset()
            target = frozenset([v]) if v is not None else frozenset()
            directed = bool(rec.directed) if rec.directed is not None else False
        else:
            u, v = rec.src, rec.tgt
            d = (
                rec.directed
                if rec.directed is not None
                else (True if self.directed is None else self.directed)
            )
            if d:
                source, target = frozenset([u]), frozenset([v])
                directed = True
            else:
                M = frozenset([u, v])
                source, target = M, M
                directed = False
            members = frozenset([u, v])
            kind = 'binary'

        if eid is None:
            eid = ''
        weight = float(rec.weight) if rec.weight is not None else 1.0
        return EdgeView(
            source,
            target,
            edge_id=eid,
            kind=kind,
            members=members,
            weight=weight,
            directed=directed,
        )

    def _incident_edge_indices(self, vertex_id) -> list[int]:
        """Return matrix column indices of all edges incident to a vertex."""
        incident = []
        ent = self._entities.get(vertex_id)
        if ent is not None:
            try:
                incident.extend(self._get_csr().getrow(ent.row_idx).indices.tolist())
                return incident
            except (IndexError, ValueError):
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
        """Check whether an edge exists.

        Parameters
        ----------
        source : str, optional
            Source endpoint.
        target : str, optional
            Target endpoint.
        edge_id : str, optional
            Edge identifier.

        Returns
        -------
        bool | tuple[bool, list[str]]
            If only ``edge_id`` is provided, returns a boolean. If ``source``
            and ``target`` are provided, returns ``(exists, edge_ids)``. If all
            three arguments are provided, returns whether that exact edge ID
            connects the given endpoints.

        Raises
        ------
        ValueError
            If the argument combination is invalid.

        Examples
        --------
        >>> G.has_edge(edge_id='e1')
        True
        >>> G.has_edge('A', 'B')
        (True, ['e1'])
        """

        # ---- Mode 1: edge_id only ----
        if edge_id is not None and source is None and target is None:
            return edge_id in self._edges

        # ---- Mode 2: source + target only ----
        if edge_id is None and source is not None and target is not None:
            eids = self._edge_ids_for_pair(source, target)
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
        """Check whether a vertex exists.

        Parameters
        ----------
        vertex_id : str | tuple
            Bare vertex ID, or explicit ``(vertex_id, layer_coord)`` tuple for
            multilayer graphs.

        Returns
        -------
        bool
            ``True`` if the graph contains a vertex entity matching
            ``vertex_id``.

        Notes
        -----
        In multilayer graphs, a bare vertex ID returns ``True`` if that vertex
        is present on at least one layer coordinate.
        """
        if isinstance(vertex_id, str):
            if self._aspects == ('_',):
                ent = self._entities.get((vertex_id, ('_',)))
                return ent is not None and ent.kind == 'vertex'
            for ekey in self._vid_to_ekeys.get(vertex_id, ()):
                ent = self._entities.get(ekey)
                if ent is not None and ent.kind == 'vertex':
                    return True
            return False

        ekey = self._resolve_entity_key(vertex_id)
        ent = self._entities.get(ekey)
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
        return self._edge_ids_for_pair(source, target)

    def _get_csr(self):
        """Return a cached CSR view of _matrix. Rebuilt when _csr_cache is None."""
        csr = self.cache.csr
        self._csr_cache = csr
        return csr

    def degree(self, entity_id):
        """Return the incidence degree of a vertex or edge-entity.

        Parameters
        ----------
        entity_id : str | tuple
            Vertex ID, edge-entity ID, or explicit multilayer entity key.

        Returns
        -------
        int
            Number of non-zero incidence entries in the entity row. Missing
            entities have degree ``0``.
        """
        ekey = self._resolve_entity_key(entity_id)
        ent = self._entities.get(ekey)
        if ent is None:
            return 0
        csr = self._get_csr()
        return int(csr.indptr[ent.row_idx + 1] - csr.indptr[ent.row_idx])

    def vertices(self) -> list[str]:
        """Return all vertex IDs.

        Returns
        -------
        list[str]
            Vertex identifiers, excluding edge-entities.

        Notes
        -----
        In multilayer graphs, the returned IDs are bare vertex IDs. Use
        ``G.idx`` or layer-specific methods when layer coordinates are needed.
        """
        return [
            eid[0] if isinstance(eid, tuple) else eid
            for eid, rec in self._entities.items()
            if rec.kind == 'vertex'
        ]

    def edges(self) -> list[str]:
        """Return all structural edge IDs.

        Returns
        -------
        list[str]
            Edge identifiers for edges with an incidence-matrix column.
        """
        return [eid for eid, rec in self._edges.items() if rec.col_idx >= 0]

    def edge_list(self) -> list[tuple[str, str, str, float]]:
        """Materialize binary edges as endpoint tuples.

        Returns
        -------
        list[tuple[str, str, str, float]]
            Tuples of ``(source, target, edge_id, weight)`` for binary and
            vertex-edge records. Hyperedges and endpoint-less placeholders are
            omitted. The ``weight`` reflects the active slice's per-edge
            override when one is set; otherwise the edge's stored weight.
        """
        edges = []
        get_eff = self.attrs.get_effective_edge_weight
        for edge_id, rec in self._edges.items():
            if rec.etype == 'hyper' or rec.src is None or rec.tgt is None:
                continue
            edges.append((rec.src, rec.tgt, edge_id, get_eff(edge_id)))
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
        """Count unique members present across slices.

        Parameters
        ----------
        kind : {"vertices", "edges", "entities"}
            Membership domain. ``"vertices"`` counts slice vertex members,
            ``"edges"`` counts slice edge members, and ``"entities"`` counts
            the union of both domains.

        Returns
        -------
        int
            Number of unique members observed in slice membership.

        Raises
        ------
        ValueError
            If ``kind`` is not one of ``"vertices"``, ``"edges"``, or
            ``"entities"``.

        Notes
        -----
        This is a slice-membership count, not a storage count. For graph
        storage counts, use :attr:`num_vertices` and :attr:`num_edges`.
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
        """Return the number of vertices.

        Returns
        -------
        int
            Number of stored vertex entities.

        See Also
        --------
        nv : Property alias for the same count.
        num_vertices : Descriptive property alias for the same count.
        """
        return self.nv

    def number_of_edges(self) -> int:
        """Return the number of edges.

        Returns
        -------
        int
            Number of structural edges with incidence columns.

        See Also
        --------
        ne : Property alias for the same count.
        num_edges : Descriptive property alias for the same count.
        """
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

    def incident_edges(
        self,
        vertices: str | Iterable[str],
        direction: str = 'both',
    ) -> list[tuple[int, EdgeView]]:
        """Return edges incident to one or more vertices.

        Parameters
        ----------
        vertices : str | Iterable[str]
            One vertex identifier or an iterable of identifiers.
        direction : {"in", "out", "both"}, optional
            Directional filter applied to binary edges. Undirected edges are
            included for both ``"in"`` and ``"out"``.

        Returns
        -------
        list[tuple[int, EdgeView]]
            Pairs of ``(column_index, edge_view)`` as returned by
            :meth:`get_edge`, materialized for consistency with the sibling
            ``vertices`` / ``edges`` / ``edge_list`` APIs which all return
            lists.

        Raises
        ------
        ValueError
            If ``direction`` is not ``"in"``, ``"out"``, or ``"both"``.

        Examples
        --------
        >>> G.incident_edges('A', direction='out')
        [(0, EdgeView(edge_id='e0', kind='binary', ...))]
        """
        if direction not in {'in', 'out', 'both'}:
            raise ValueError("direction must be 'in', 'out', or 'both'")
        self._ensure_edge_indexes()
        V = self._normalize_vertices_arg(vertices)
        if not V:
            return []
        seen = set()
        result = []
        for v in V:
            if direction in {'in', 'both'}:
                for eid in self._tgt_to_edges.get(v, []):
                    if eid not in seen:
                        seen.add(eid)
                        rec = self._edges.get(eid)
                        if rec is not None and rec.col_idx >= 0:
                            result.append((rec.col_idx, self._edge_tuple_from_record(rec, eid=eid)))
            if direction in {'out', 'both'}:
                for eid in self._src_to_edges.get(v, []):
                    if eid not in seen:
                        seen.add(eid)
                        rec = self._edges.get(eid)
                        if rec is not None and rec.col_idx >= 0:
                            result.append((rec.col_idx, self._edge_tuple_from_record(rec, eid=eid)))
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
                        result.append((rec.col_idx, self._edge_tuple_from_record(rec, eid=eid)))
        return result

    @property
    def nv(self) -> int:
        """Number of stored vertices.

        Returns
        -------
        int
            Count of entities whose internal kind is ``"vertex"``.
        """
        return sum(1 for r in self._entities.values() if r.kind == 'vertex')

    @property
    def ne(self) -> int:
        """Number of structural edges.

        Returns
        -------
        int
            Count of incidence-matrix edge columns.
        """
        return len(self._col_to_edge)

    @property
    def num_vertices(self) -> int:
        """Number of stored vertices.

        Returns
        -------
        int
            Same value as :attr:`nv`.
        """
        return self.nv

    @property
    def num_edges(self) -> int:
        """Number of structural edges.

        Returns
        -------
        int
            Same value as :attr:`ne`.
        """
        return self.ne

    @property
    def shape(self) -> tuple[int, int]:
        """Graph shape as ``(num_vertices, num_edges)``.

        Returns
        -------
        tuple[int, int]
            Vertex and edge counts.
        """
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
        self._add_vertex_impl(vid, slice=slice, **attrs)

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
        """All vertices as an immutable tuple.

        Returns
        -------
        tuple[str, ...]
            Vertex IDs in graph iteration order.
        """
        return tuple(self.vertices())

    @property
    def E(self):
        """All edges as an immutable tuple.

        Returns
        -------
        tuple[str, ...]
            Edge identifiers in graph iteration order.
        """
        return tuple(self.edges())

    # Lazy proxies
    ## Lazy NetworkX proxy

    @property
    def nx(self) -> _NXBackendAccessor:
        """NetworkX interoperability namespace.

        Returns
        -------
        _NXBackendAccessor
            Lazy proxy that converts to NetworkX only when an algorithm or
            backend graph is requested.

        Examples
        --------
        >>> G.nx.backend()
        >>> G.nx.shortest_path(G, 'A', 'B')
        """
        if not hasattr(self, '_nx_proxy'):
            self._nx_proxy = _NXBackendAccessor(self)
        return self._nx_proxy

    ## Lazy iGraph proxy

    @property
    def ig(self) -> _IGBackendAccessor:
        """Igraph interoperability namespace.

        Returns
        -------
        _IGBackendAccessor
            Lazy proxy that converts to igraph only when requested.
        """
        if not hasattr(self, '_ig_proxy'):
            self._ig_proxy = _IGBackendAccessor(self)
        return self._ig_proxy

    ## Lazy AnnNet-tool proxy

    @property
    def gt(self) -> _GTBackendAccessor:
        """graph-tool interoperability namespace.

        Returns
        -------
        _GTBackendAccessor
            Lazy proxy that converts to graph-tool only when requested.
        """
        if not hasattr(self, '_gt_proxy'):
            self._gt_proxy = _GTBackendAccessor(self)
        return self._gt_proxy

    # AnnNet API

    def X(self):
        """Return the sparse incidence matrix.

        Returns
        -------
        scipy.sparse.dok_matrix
            Internal incidence matrix. Rows are entities; columns are
            structural edges.

        Notes
        -----
        The returned matrix is the live internal DOK matrix. Treat it as
        read-only unless you are implementing core internals.
        """
        return self._matrix

    @property
    def obs(self) -> Any:
        """Notebook-friendly vertex attribute table.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        This is the quickest way to inspect vertex annotations in a notebook.
        It returns the underlying vertex-attribute table unchanged.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_vertices([{'vertex_id': 'A', 'kind': 'gene'}])
        >>> G.obs
        >>> G.views.vertices()

        Use :attr:`obs` when you want the raw vertex table directly. Use
        :attr:`views` when you want an explicit namespace for materialized
        tables.
        """
        return self.vertex_attributes

    @property
    def var(self) -> Any:
        """Notebook-friendly edge attribute table.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        This is the direct edge-annotation table. It is often the most useful
        object to display in a notebook after edge insertion or IO.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_vertices(['A', 'B'])
        >>> G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1'}])
        >>> G.var
        >>> G.views.edges()
        """
        return self.edge_attributes

    @property
    def uns(self) -> dict[str, Any]:
        """Graph-level unstructured metadata.

        Returns
        -------
        dict
            Mutable dictionary of graph-level attributes.
        """
        return self.graph_attributes

    @property
    def slices(self) -> SliceManager:
        """Slice operations namespace.

        Returns
        -------
        SliceManager
            Manager exposing slice creation, membership, set operations, and
            slice-level analysis.

        Examples
        --------
        >>> G.slices.add('baseline')
        >>> G.slices.active = 'baseline'
        >>> G.slices.list()
        """
        if not hasattr(self, '_slice_manager'):
            self._slice_manager = SliceManager(self)
        return self._slice_manager

    @property
    def attrs(self) -> AttributesAccessor:
        """Attribute operations namespace.

        Returns
        -------
        AttributesAccessor
            Manager for graph-, vertex-, edge-, slice-, and edge-slice
            annotations.

        Notes
        -----
        Use this namespace for graph-, vertex-, edge-, and slice-level
        annotations.

        Examples
        --------
        >>> G.attrs.set_vertex_attrs('A', symbol='TP53')
        >>> G.attrs.get_vertex_attrs('A')
        >>> G.attrs.set_edge_slice_attrs('baseline', 'e1', weight=0.5)
        """
        try:
            return self._attrs_accessor
        except AttributeError:
            self._attrs_accessor = AttributesAccessor(self)
            return self._attrs_accessor

    @property
    def views(self) -> ViewsAccessor:
        """Materialized table namespace.

        Returns
        -------
        ViewsAccessor
            Manager for dataframe-style materialized views.

        Notes
        -----
        This is the preferred namespace for notebook inspection and export of
        graph tables.

        Examples
        --------
        >>> G.views.vertices()
        >>> G.views.edges()
        >>> G.views.slices()
        >>> G.views.layers()
        """
        try:
            return self._views_accessor
        except AttributeError:
            self._views_accessor = ViewsAccessor(self)
            return self._views_accessor

    @property
    def ops(self) -> OperationsAccessor:
        """Structural operations namespace.

        Returns
        -------
        OperationsAccessor
            Manager for subgraphs, copies, reversals, incidence extraction, and
            memory inspection.

        Examples
        --------
        >>> H = G.ops.subgraph(['A', 'B', 'C'])
        >>> M = G.ops.vertex_incidence_matrix(sparse=True)
        >>> usage = G.ops.memory_usage()
        """
        try:
            return self._ops_accessor
        except AttributeError:
            self._ops_accessor = OperationsAccessor(self)
            return self._ops_accessor

    @property
    def layers(self) -> LayerAccessor:
        """Layer operations namespace.

        Returns
        -------
        LayerAccessor
            Manager for multilayer aspects, layer coordinates, supra matrices,
            and layer set operations.

        Notes
        -----
        All multilayer configuration and layer-aware analysis lives here.

        Examples
        --------
        >>> G.layers.set_aspects(['condition'], {'condition': ['ctrl', 'stim']})
        >>> G.layers.list_layers()
        >>> G.views.layers()
        """
        try:
            return self._layer_accessor
        except AttributeError:
            self._layer_accessor = LayerAccessor(self)
            return self._layer_accessor

    @property
    def idx(self):
        """Index lookup namespace.

        Returns
        -------
        IndexManager
            Manager for entity-to-row and edge-to-column index lookups.
        """
        if not hasattr(self, '_index_manager'):
            self._index_manager = IndexManager(self)
        return self._index_manager

    @property
    def cache(self):
        """Sparse matrix cache namespace.

        Returns
        -------
        CacheManager
            Manager for derived sparse matrix formats such as CSR and CSC.
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
            Passed to `annnet.io.io_annnet.write`.

        Returns
        -------
        None

        Examples
        --------
        >>> G.write('graph.annnet')
        """
        from ..io.io_annnet import write

        write(self, path, **kwargs)

    @classmethod
    def read(cls, path, **kwargs):
        """Read a graph from the native ``.annnet`` format.

        Parameters
        ----------
        path : str | pathlib.Path
            Input file path.
        **kwargs
            Passed to `annnet.io.io_annnet.read`.

        Returns
        -------
        AnnNet
            Deserialized graph.

        Examples
        --------
        >>> G = AnnNet.read('graph.annnet')
        """
        from ..io.io_annnet import read

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
            View object backed by this graph.

        Notes
        -----
        Views are lightweight filters over an existing graph. Use
        :attr:`views` for materialized dataframe views.
        """
        return GraphView(self, vertices, edges, slices, predicate)

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
                'vertex_ids': {eid for eid, r in ref._entities.items() if r.kind == 'vertex'},
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
            'vertex_ids': {
                eid[0] if isinstance(eid, tuple) else eid
                for eid, r in self._entities.items()
                if r.kind == 'vertex'
            },
            'edge_ids': set(self._col_to_edge.values()),
            'slice_ids': set(self._slices.keys()),
        }

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
            self._layers = {'_': {'_'}}
        else:
            self._aspects = tuple(val)
            self._layers = {a: set(self._layers.get(a, set())) for a in self._aspects}
            for a in self._aspects:
                self._layers.setdefault(a, set())
        self._rebuild_all_layers_cache()

    @property
    def elem_layers(self) -> dict[str, list[str]]:
        """Elementary layer labels per aspect (empty dict for flat graphs)."""
        if self._aspects == ('_',):
            return {}
        return {k: sorted(x for x in v if x != '_') for k, v in self._layers.items() if k != '_'}

    @elem_layers.setter
    def elem_layers(self, val: dict[str, list[str]]):
        if not val:
            self._layers = {'_': {'_'}}
        else:
            self._layers = {k: set(v) for k, v in val.items()}
        self._rebuild_all_layers_cache()

    # -------------------------------------------------------------------------
    # Computed read properties — derived from _entities / _edges / _col_to_edge
    # Mutation must go through the record fields directly:
    #   self._edges[eid].ml_layers = val
    #   self._edges[eid].weight = float(val)
    #   self._edges[eid].direction_policy = val
    # -------------------------------------------------------------------------

    @property
    def edge_layers(self) -> dict:
        """edge_id -> ml_layers for all edges that have a layer assignment."""
        return _EdgeRecordFieldMap(
            self,
            'ml_layers',
            include=lambda _rec, value: value is not None,
        )

    @edge_layers.setter
    def edge_layers(self, mapping):
        for eid, layers in dict(mapping).items():
            if eid in self._edges:
                self._edges[eid].ml_layers = layers

    @property
    def edge_kind(self) -> dict:
        """edge_id -> kind (hyper edges use 'hyper'; others use ml_kind)."""
        return _EdgeRecordFieldMap(
            self,
            'ml_kind',
            include=lambda rec, value: rec.etype == 'hyper' or value is not None,
            getter=lambda rec, value: 'hyper' if rec.etype == 'hyper' else value,
            setter=lambda rec, value: (
                setattr(rec, 'etype', 'hyper')
                if value == 'hyper'
                else setattr(rec, 'ml_kind', value)
            ),
        )

    @edge_kind.setter
    def edge_kind(self, mapping):
        for eid, kind in dict(mapping).items():
            if eid not in self._edges:
                continue
            rec = self._edges[eid]
            if kind == 'hyper':
                rec.etype = 'hyper'
            else:
                rec.ml_kind = kind

    @property
    def _aspect_attrs(self) -> dict:
        return self.layers._aspect_attrs

    @_aspect_attrs.setter
    def _aspect_attrs(self, value) -> None:
        self.layers._aspect_attrs = dict(value or {})

    @property
    def _layer_attrs(self) -> dict:
        return self.layers._layer_attrs

    @_layer_attrs.setter
    def _layer_attrs(self, value) -> None:
        self.layers._layer_attrs = dict(value or {})

    @property
    def _state_attrs(self) -> dict:
        return self.layers._state_attrs

    @_state_attrs.setter
    def _state_attrs(self, value) -> None:
        self.layers._state_attrs = dict(value or {})

    @property
    def entity_to_idx(self) -> dict:
        """vertex_id -> row_idx (bare string key, first supra-node wins)."""
        return {ekey[0]: rec.row_idx for ekey, rec in self._entities.items()}

    @entity_to_idx.setter
    def entity_to_idx(self, mapping):
        self._entities.clear()
        self._row_to_entity.clear()
        self._vid_to_ekeys.clear()
        for vid, row_idx in dict(mapping).items():
            coord = self._placeholder_layer_coord()
            self._register_entity_record(
                (vid, coord), EntityRecord(row_idx=int(row_idx), kind='vertex')
            )

    @property
    def idx_to_entity(self) -> dict:
        """row_idx -> vertex_id (bare string)."""
        return {idx: ekey[0] for idx, ekey in self._row_to_entity.items()}

    @property
    def entity_types(self) -> dict:
        """vertex_id -> kind string ('vertex' or 'edge')."""
        return {ekey[0]: _external_entity_kind(rec.kind) for ekey, rec in self._entities.items()}

    @entity_types.setter
    def entity_types(self, mapping):
        for vid, kind in dict(mapping).items():
            ekey = self._resolve_entity_key(vid)
            rec = self._entities.get(ekey)
            row_idx = rec.row_idx if rec is not None else len(self._entities)
            self._register_entity_record(
                ekey, EntityRecord(row_idx=row_idx, kind=_internal_entity_kind(kind))
            )

    @property
    def edge_to_idx(self) -> dict:
        """edge_id -> col_idx for all edges with a matrix column."""
        return {eid: rec.col_idx for eid, rec in self._edges.items() if rec.col_idx >= 0}

    @property
    def idx_to_edge(self) -> dict:
        """col_idx -> edge_id."""
        return dict(self._col_to_edge)

    @property
    def edge_weights(self) -> dict:
        """edge_id -> weight for all edges."""
        return {eid: rec.weight for eid, rec in self._edges.items()}

    @property
    def edge_directed(self) -> dict:
        """edge_id -> directed for edges with an explicit directedness flag."""
        return {eid: rec.directed for eid, rec in self._edges.items() if rec.directed is not None}

    @property
    def edge_definitions(self) -> dict:
        """edge_id -> (src, tgt, etype) for binary edges."""
        return {
            eid: (rec.src, rec.tgt, rec.etype)
            for eid, rec in self._edges.items()
            if rec.etype != 'hyper' and rec.src is not None
        }

    @edge_definitions.setter
    def edge_definitions(self, mapping):
        for eid, defn in dict(mapping).items():
            if eid not in self._edges:
                continue
            src, tgt, etype = defn
            rec = self._edges[eid]
            rec.src = src
            rec.tgt = tgt
            rec.etype = etype if etype != 'hyper' else 'binary'

    @property
    def hyperedge_definitions(self) -> dict:
        """edge_id -> hyper metadata dict for hyperedges."""
        out = {}
        for eid, rec in self._edges.items():
            if rec.etype != 'hyper':
                continue
            if rec.tgt is not None:
                out[eid] = {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
            else:
                out[eid] = {'directed': False, 'members': set(rec.src)}
        return out

    @hyperedge_definitions.setter
    def hyperedge_definitions(self, mapping):
        for eid, defn in dict(mapping).items():
            if eid not in self._edges:
                continue
            rec = self._edges[eid]
            rec.etype = 'hyper'
            if isinstance(defn, list):
                rec.src = frozenset(defn)
                rec.tgt = None
                rec.directed = False
            else:
                directed = bool(defn.get('directed', False))
                if directed:
                    rec.src = frozenset(defn.get('head', []))
                    rec.tgt = frozenset(defn.get('tail', []))
                    rec.directed = True
                else:
                    rec.src = frozenset(defn.get('members', []))
                    rec.tgt = None
                    rec.directed = False

    @property
    def edge_direction_policy(self) -> dict:
        """edge_id -> direction_policy for edges that have one set."""
        return {
            eid: rec.direction_policy
            for eid, rec in self._edges.items()
            if rec.direction_policy is not None
        }

    @edge_direction_policy.setter
    def edge_direction_policy(self, mapping):
        for eid, policy in dict(mapping).items():
            if eid in self._edges:
                self._edges[eid].direction_policy = policy

    @property
    def _num_entities(self) -> int:
        return len(self._entities)

    @_num_entities.setter
    def _num_entities(self, value) -> None:
        return None

    @property
    def _num_edges(self) -> int:
        return len(self._col_to_edge)

    @_num_edges.setter
    def _num_edges(self, value) -> None:
        return None

    # ------------------------------------------------------------------
    # Bulk mutation API
    # ------------------------------------------------------------------

    def _add_vertices_batch(self, vertices, layer=None, slice=None, default_attrs=None):
        """Add many vertices in one pass.

        Parameters
        ----------
        vertices : Iterable[str] | Iterable[tuple[str, dict]] | Iterable[dict]
            Each item may be a vertex_id string, a ``(vertex_id, attrs)`` tuple,
            or a dict with a ``vertex_id`` key plus attribute keys.
        layer : str | dict | tuple | None, optional
            Layer spec applied to every vertex in this batch.
        slice : str, optional
            Target slice. Defaults to the active slice.
        default_attrs : dict | None, optional
            Attributes broadcast across every item. Per-item attrs take
            precedence on key collision.
        """
        slice = slice or self._current_slice
        default_attrs = default_attrs or {}

        # --- normalize input ---
        norm = []
        for it in vertices:
            if isinstance(it, dict):
                if it.get('vertex_id'):
                    vid = it['vertex_id']
                    _id_keys = {'vertex_id'}
                elif it.get('id'):
                    vid = it['id']
                    _id_keys = {'vertex_id', 'id'}
                elif it.get('name'):
                    vid = it['name']
                    _id_keys = {'vertex_id', 'id', 'name'}
                else:
                    vid = None
                if vid is None:
                    continue
                attrs = {k: v for k, v in it.items() if k not in _id_keys}
            elif isinstance(it, (tuple, list)) and it:
                vid = it[0]
                attrs = it[1] if len(it) > 1 and isinstance(it[1], dict) else {}
            else:
                vid = it
                attrs = {}
            if default_attrs:
                merged = dict(default_attrs)
                merged.update(attrs)
                attrs = merged
            norm.append((vid, attrs))

        if not norm:
            return

        try:
            import sys as _sys

            norm = [
                (_sys.intern(vid) if isinstance(vid, str) else vid, attrs) for vid, attrs in norm
            ]
            if isinstance(slice, str):
                slice = _sys.intern(slice)
        except TypeError:
            pass

        # --- entity registration ---
        coord = self._resolve_vertex_insert_coord(
            layer, vertex_ids=[vid for vid, _ in norm], context='_add_vertices_batch'
        )
        new_rows = 0
        for vid, _ in norm:
            ekey = (vid, coord)
            if ekey not in self._entities:
                idx = len(self._entities)
                self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind='vertex'))
                new_rows += 1
        if new_rows:
            self._grow_rows_to(len(self._entities))

        # --- slice ---
        self.slices._ensure_slice(slice)['vertices'].update(vid for vid, _ in norm)

        # --- attribute table (Polars fast path) ---
        self._ensure_vertex_table()
        if _is_polars_df(self.vertex_attributes):
            keys = {k for _, attrs in norm for k in attrs}
            df = self.vertex_attributes
            if keys:
                df = self._ensure_attr_columns(df, dict.fromkeys(keys))
            if _is_polars_df(df):
                result = polars_upsert_vertices(df, norm)
                if result is not None:
                    self.vertex_attributes = result
                    return

        # --- generic fallback (pandas / pyarrow, or polars with non-string vids) ---
        # Non-string vids (e.g. multilayer supra-node tuples) are tracked in
        # ``_entities`` only and stay out of the obs table. The polars backend
        # cannot represent tuples in a String-typed vertex_id column anyway.
        df2 = self.vertex_attributes

        rows = [
            {'vertex_id': vid, **{k: _sanitize(v) for k, v in attrs.items()}}
            for vid, attrs in norm
            if isinstance(vid, str)
        ]
        if rows:
            df2 = dataframe_upsert_rows(df2, rows, ('vertex_id',))

        self.vertex_attributes = df2

    def _add_edges_batch(
        self,
        edges,
        *,
        slice=None,
        as_entity=False,
        default_weight=1.0,
        default_edge_type='regular',
        default_propagate='none',
        default_slice_weight=None,
        default_edge_directed=None,
    ):
        """Add many binary edges in one pass.

        Parameters
        ----------
        edges : Iterable
            Each item may be a ``(source, target)`` tuple, ``(source, target,
            weight)`` tuple, or a dict with ``source`` / ``target`` plus
            optional edge fields.
        slice : str, optional
            Default slice for edges missing an explicit slice.
        as_entity : bool, optional
            Register each edge as a connectable entity.
        default_weight : float, optional
            Weight for edges missing an explicit weight.
        default_edge_type : str, optional
            Edge type when not provided.
        default_propagate : {'none', 'shared', 'all'}, optional
            Slice propagation mode.
        default_slice_weight : float, optional
            Per-slice weight override.
        default_edge_directed : bool, optional
            Per-edge directedness override.

        Returns
        -------
        list[str]
            Edge IDs for created/updated edges.
        """
        self._ensure_edge_indexes()
        slice = self._current_slice if slice is None else slice
        pending_attrs = {}

        norm = []
        for idx, it in enumerate(edges):
            if isinstance(it, dict):
                d = dict(it)
                if 'src' in d and 'source' not in d:
                    d['source'] = d.pop('src')
                if 'tgt' in d and 'target' not in d:
                    d['target'] = d.pop('tgt')
                if 'directed' in d and 'edge_directed' not in d:
                    d['edge_directed'] = d.pop('directed')
                has_src = 'source' in d
                has_tgt = 'target' in d
                if has_src ^ has_tgt:
                    missing = 'target' if has_src else 'source'
                    raise ValueError(
                        f'add_edges batch item at index {idx} is missing '
                        f"'{missing}' (or its alias '{'tgt' if missing == 'target' else 'src'}'): "
                        f'{it!r}'
                    )
            elif isinstance(it, (tuple, list)):
                if len(it) == 2:
                    d = {'source': it[0], 'target': it[1], 'weight': default_weight}
                else:
                    d = {'source': it[0], 'target': it[1], 'weight': it[2]}
            else:
                continue
            d.setdefault('weight', default_weight)
            d.setdefault('edge_type', default_edge_type)
            d.setdefault('propagate', default_propagate)
            if 'slice' not in d:
                d['slice'] = slice
            if 'edge_directed' not in d:
                d['edge_directed'] = default_edge_directed
            norm.append(d)

        if not norm:
            return []

        # ── Null-endpoint entity-placeholders ──────────────────────────────────
        _EE_RESERVED = _BINARY_BATCH_RESERVED_KEYS
        entity_items = [d for d in norm if 'source' not in d and 'target' not in d]
        if entity_items:
            if not as_entity:
                raise ValueError(
                    'Batch items without source/target require as_entity=True to be '
                    'treated as edge-entity placeholders.'
                )
            norm = [d for d in norm if 'source' in d or 'target' in d]

        entity_out: list = []
        for d in entity_items:
            e_id = d.get('edge_id') or self._get_next_edge_id()
            sl = d.get('slice', slice)
            extra = {k: v for k, v in d.items() if k not in _EE_RESERVED}
            self._ensure_edge_entity_placeholder(e_id, slice=sl, **extra)
            entity_out.append(e_id)

        if not norm:
            return entity_out

        try:
            import sys as _sys

            for d in norm:
                s, t = d['source'], d['target']
                if isinstance(s, str):
                    d['source'] = _sys.intern(s)
                if isinstance(t, str):
                    d['target'] = _sys.intern(t)
                if isinstance(d.get('slice'), str):
                    d['slice'] = _sys.intern(d['slice'])
                if isinstance(d.get('edge_id'), str):
                    d['edge_id'] = _sys.intern(d['edge_id'])
                try:
                    d['weight'] = float(d['weight'])
                except (TypeError, ValueError):
                    pass
        except Exception:  # noqa: BLE001
            pass

        M = self._matrix
        _m_fast_set = getattr(M, '_set_intXint', None)
        _m_dtype = M.dtype.type

        unique_vids: dict = {}
        for d in norm:
            s, t = d['source'], d['target']
            et = d.get('edge_type', 'regular')
            unique_vids[s] = et
            unique_vids[t] = et

        endpoint_cache: dict = {}
        for vid, et in unique_vids.items():
            if isinstance(vid, tuple) and len(vid) == 2 and isinstance(vid[1], tuple):
                ekey = vid
            else:
                coord = self._resolve_vertex_insert_coord(
                    None, vertex_ids=vid, context='_add_edges_batch'
                )
                ekey = (vid, coord)
            if ekey not in self._entities:
                if (
                    et in {'vertex_edge', 'edge_placeholder'}
                    and isinstance(vid, str)
                    and vid.startswith('edge_')
                ):
                    self._ensure_edge_entity_placeholder(vid)
                else:
                    idx = len(self._entities)
                    self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind='vertex'))
            endpoint_cache[vid] = self._entities[ekey].row_idx

        self._grow_rows_to(len(self._entities))

        _edges_store = self._edges
        new_count = 0
        _need_auto_id = []
        for _i, d in enumerate(norm):
            eid = d.get('edge_id')
            if eid not in _edges_store or _edges_store[eid].col_idx < 0:
                new_count += 1
            if eid is None:
                _need_auto_id.append(_i)

        if new_count:
            self._grow_cols_to(len(self._col_to_edge) + new_count)

        if _need_auto_id:
            _base_id = self._next_edge_id
            self._next_edge_id += len(_need_auto_id)
            _auto_ids = iter(range(_base_id, _base_id + len(_need_auto_id)))
            for _i in _need_auto_id:
                norm[_i]['edge_id'] = f'edge_{next(_auto_ids)}'

        out_ids = []
        _M_writes: dict = {}
        _M_zero_keys: list = []
        _slice_eids: dict = {}
        _slice_vids: dict = {}
        _slice_weights: list = []

        for d in norm:
            s, t = d['source'], d['target']
            w = d['weight']
            prop = d.get('propagate', default_propagate)
            slice_local = d.get('slice', slice)
            slice_w = d.get('slice_weight', default_slice_weight)
            e_dir = d.get('edge_directed', default_edge_directed)
            edge_id = d.get('edge_id')

            if e_dir is not None:
                is_dir = bool(e_dir)
            elif self.directed is not None:
                is_dir = self.directed
            else:
                is_dir = True
            s_idx = endpoint_cache[s]
            t_idx = endpoint_cache[t]
            fw = _m_dtype(w)

            if edge_id in self._edges and self._edges[edge_id].col_idx >= 0:
                rec = self._edges[edge_id]
                col = rec.col_idx
                old_s, old_t = rec.src, rec.tgt
                try:
                    _M_zero_keys.append((self._entity_row(old_s), col))
                except KeyError:
                    pass
                if old_t is not None and old_t != old_s:
                    try:
                        _M_zero_keys.append((self._entity_row(old_t), col))
                    except KeyError:
                        pass
                _M_writes[(s_idx, col)] = fw
                if s != t:
                    _M_writes[(t_idx, col)] = _m_dtype(-w) if is_dir else fw
                rec.src = s
                rec.tgt = t
                rec.weight = w
                rec.directed = is_dir
                if (old_s, old_t) != (s, t):
                    self._unindex_edge_pair(edge_id, old_s, old_t)
                    for _old, _new, _index in (
                        (old_s, s, self._src_to_edges),
                        (old_t, t, self._tgt_to_edges),
                    ):
                        if _old != _new:
                            _lst = _index.get(_old)
                            if _lst:
                                try:
                                    _lst.remove(edge_id)
                                except ValueError:
                                    pass
                                if not _lst:
                                    del _index[_old]
                            _index.setdefault(_new, []).append(edge_id)
                    self._index_edge_pair(edge_id, s, t)
            else:
                col = len(self._col_to_edge)
                self._col_to_edge[col] = edge_id
                if edge_id in self._edges:
                    rec = self._edges[edge_id]
                    rec.src = s
                    rec.tgt = t
                    rec.weight = w
                    rec.directed = is_dir
                    rec.etype = 'binary'
                    rec.col_idx = col
                else:
                    self._edges[edge_id] = EdgeRecord(
                        src=s,
                        tgt=t,
                        weight=w,
                        directed=is_dir,
                        etype='binary',
                        col_idx=col,
                        ml_kind=None,
                        ml_layers=None,
                        direction_policy=None,
                    )
                _M_writes[(s_idx, col)] = fw
                if s != t:
                    _M_writes[(t_idx, col)] = _m_dtype(-w) if is_dir else fw
                self._src_to_edges.setdefault(s, []).append(edge_id)
                self._tgt_to_edges.setdefault(t, []).append(edge_id)
                self._index_edge_pair(edge_id, s, t)

            if slice_local is not None:
                _lst = _slice_eids.get(slice_local)
                if _lst is None:
                    _slice_eids[slice_local] = [edge_id]
                    _slice_vids[slice_local] = [s, t]
                else:
                    _lst.append(edge_id)
                    _slice_vids[slice_local].extend((s, t))
                if slice_w is not None:
                    _slice_weights.append((slice_local, edge_id, float(slice_w)))

            if prop == 'shared':
                self._propagate_to_shared_slices(edge_id, s, t)
            elif prop == 'all':
                self._propagate_to_all_slices(edge_id, s, t)

            sub_attrs = d.get('attributes') or d.get('attrs') or {}
            flat_attrs = {k: v for k, v in d.items() if k not in _BINARY_BATCH_RESERVED_KEYS}
            if sub_attrs or flat_attrs:
                merged_attrs = dict(sub_attrs)
                merged_attrs.update(flat_attrs)
                pending_attrs.setdefault(edge_id, {}).update(merged_attrs)

            out_ids.append(edge_id)

        for key in _M_zero_keys:
            dict.pop(M, key, None)
        if _M_writes:
            dict.update(M, _M_writes)
            self._invalidate_sparse_caches()

        for sid, eids in _slice_eids.items():
            self.slices._ensure_slice(sid)['edges'].update(eids)
        for sid, vids in _slice_vids.items():
            self._slices[sid]['vertices'].update(vids)
        for sid, eid, sw in _slice_weights:
            self.attrs.set_edge_slice_attrs(sid, eid, weight=sw)

        if pending_attrs:
            self.attrs.set_edge_attrs_bulk(pending_attrs)

        if as_entity:
            entities = self._entities
            flat = self._aspects == ('_',)
            flat_coord = ('_',)
            for eid in out_ids:
                ekey = (
                    (eid, flat_coord)
                    if flat and isinstance(eid, str)
                    else self._resolve_entity_key(eid)
                )
                if ekey not in entities:
                    self._register_entity_record(
                        ekey,
                        EntityRecord(row_idx=len(entities), kind='edge_entity'),
                    )
                rec = self._edges[eid]
                if rec.etype == 'binary':
                    rec.etype = 'vertex_edge'
            self._grow_rows_to(len(entities))

        self._ensure_edge_rows_bulk(entity_out + out_ids)

        return entity_out + out_ids

    def _add_hyperedges_batch(
        self,
        hyperedges,
        *,
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Add many hyperedges in one pass.

        Parameters
        ----------
        hyperedges : Iterable[dict]
            Each item declares a hyperedge via ``src`` (or its alias
            ``source``) and optionally ``tgt`` (or ``target``):

            - list-shaped ``src`` with no ``tgt`` → undirected hyperedge
              (``src`` is the member set)
            - list-shaped ``src`` and ``tgt`` → directed hyperedge
              (``src`` is stored in ``rec.src`` and gets the ``+w`` matrix
              entries; ``tgt`` is stored in ``rec.tgt`` and gets ``-w``,
              matching the single-edge ``add_edges`` path).

            Legacy ``members`` / ``head`` / ``tail`` keys are still accepted
            for internal IO compatibility but should not be used in
            user-written code.
        slice : str, optional
            Default slice for hyperedges missing an explicit slice.
        default_weight : float, optional
            Default weight.
        default_edge_directed : bool, optional
            Default directedness override.

        Returns
        -------
        list[str]
            Hyperedge IDs.
        """
        slice = self._current_slice if slice is None else slice

        items = []
        for it in hyperedges:
            if not isinstance(it, dict):
                continue
            d = dict(it)
            if 'directed' in d and 'edge_directed' not in d:
                d['edge_directed'] = d.pop('directed')

            # ── Normalize user-facing src/tgt to internal members/head/tail ──
            # annnet stores rec.src = head (the +w side in the incidence
            # matrix) and rec.tgt = tail (the -w side). To keep the batch
            # path consistent with the single-edge path — where the user's
            # ``src`` ends up in rec.src and gets +w — map user.src → head
            # and user.tgt → tail here.
            if 'src' in d and 'source' not in d:
                d['source'] = d.pop('src')
            if 'tgt' in d and 'target' not in d:
                d['target'] = d.pop('tgt')
            has_legacy = any(k in d for k in ('members', 'head', 'tail'))
            has_new = 'source' in d or 'target' in d
            if has_new and not has_legacy:
                src_val = d.pop('source', None)
                tgt_val = d.pop('target', None)

                def _as_list(v):
                    if v is None:
                        return None
                    if isinstance(v, str):
                        return [v]
                    return list(v)

                src_list = _as_list(src_val)
                tgt_list = _as_list(tgt_val)
                if tgt_list is None:
                    d['members'] = src_list or []
                else:
                    d['head'] = src_list or []
                    d['tail'] = tgt_list

            d.setdefault('weight', default_weight)
            if 'slice' not in d:
                d['slice'] = slice
            if 'edge_directed' not in d:
                d['edge_directed'] = default_edge_directed
            items.append(d)

        if not items:
            return []

        try:
            import sys as _sys

            for d in items:
                if 'members' in d and d['members'] is not None:
                    d['members'] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d['members']
                    ]
                else:
                    d['head'] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get('head', [])
                    ]
                    d['tail'] = [
                        _sys.intern(x) if isinstance(x, str) else x for x in d.get('tail', [])
                    ]
                if isinstance(d.get('slice'), str):
                    d['slice'] = _sys.intern(d['slice'])
                if isinstance(d.get('edge_id'), str):
                    d['edge_id'] = _sys.intern(d['edge_id'])
                try:
                    d['weight'] = float(d['weight'])
                except (TypeError, ValueError):
                    pass
        except Exception:  # noqa: BLE001
            pass

        all_verts: set = set()
        for d in items:
            if 'members' in d and d['members'] is not None:
                all_verts.update(d['members'])
            else:
                all_verts.update(d.get('head', []))
                all_verts.update(d.get('tail', []))

        for u in all_verts:
            coord = self._resolve_vertex_insert_coord(
                None, vertex_ids=u, context='_add_hyperedges_batch'
            )
            ekey = (u, coord)
            if ekey not in self._entities:
                idx = len(self._entities)
                self._register_entity_record(ekey, EntityRecord(row_idx=idx, kind='vertex'))

        self._grow_rows_to(len(self._entities))

        new_count = sum(1 for d in items if d.get('edge_id') not in self._edges)
        if new_count:
            self._grow_cols_to(len(self._col_to_edge) + new_count)

        M = self._matrix
        slices = self._slices
        _m_fast_set = getattr(M, '_set_intXint', None)
        _m_dtype = M.dtype.type

        out_ids = []
        attrs_batch = {}

        for d in items:
            members = d.get('members')
            head = d.get('head')
            tail = d.get('tail')
            slice_local = d.get('slice', slice)
            w = float(d.get('weight', default_weight))
            e_id = d.get('edge_id')
            directed = d.get('edge_directed')
            if directed is None:
                directed = members is None

            if e_id is None:
                e_id = self._get_next_edge_id()

            if e_id in self._edges:
                rec = self._edges[e_id]
                col = rec.col_idx
                if rec.etype == 'hyper':
                    old_verts = rec.src if rec.tgt is None else (rec.src | rec.tgt)
                    for vid in old_verts:
                        try:
                            r = self._entity_row(vid)
                            if _m_fast_set is not None:
                                _m_fast_set(r, col, 0)
                            else:
                                M[r, col] = 0
                        except (KeyError, IndexError):
                            pass
                else:
                    for vid in (rec.src, rec.tgt):
                        if vid is None:
                            continue
                        try:
                            r = self._entity_row(vid)
                            if _m_fast_set is not None:
                                _m_fast_set(r, col, 0)
                            else:
                                M[r, col] = 0
                        except (KeyError, IndexError):
                            pass
            else:
                col = len(self._col_to_edge)
                self._col_to_edge[col] = e_id
                rec = EdgeRecord(
                    src=None,
                    tgt=None,
                    weight=1.0,
                    directed=False,
                    etype='hyper',
                    col_idx=col,
                    ml_kind=None,
                    ml_layers=None,
                    direction_policy=None,
                )
                self._edges[e_id] = rec

            fw = _m_dtype(w)
            if members is not None:
                if _m_fast_set is not None:
                    for u in members:
                        _m_fast_set(self._entity_row(u), col, fw)
                else:
                    for u in members:
                        M[self._entity_row(u), col] = fw
                rec.src = frozenset(members)
                rec.tgt = None
                rec.directed = False
            else:
                neg_fw = _m_dtype(-w)
                if _m_fast_set is not None:
                    for u in head:
                        _m_fast_set(self._entity_row(u), col, fw)
                    for v in tail:
                        _m_fast_set(self._entity_row(v), col, neg_fw)
                else:
                    for u in head:
                        M[self._entity_row(u), col] = fw
                    for v in tail:
                        M[self._entity_row(v), col] = neg_fw
                rec.src = frozenset(head)
                rec.tgt = frozenset(tail)
                rec.directed = True

            rec.weight = w
            rec.etype = 'hyper'

            if slice_local is not None:
                if slice_local not in slices:
                    slices[slice_local] = SliceRecord()
                slices[slice_local]['edges'].add(e_id)
                if members is not None:
                    slices[slice_local]['vertices'].update(members)
                else:
                    slices[slice_local]['vertices'].update(head)
                    slices[slice_local]['vertices'].update(tail)

            sub_attrs = d.get('attributes') or d.get('attrs') or {}
            flat_attrs = {k: v for k, v in d.items() if k not in _HYPER_BATCH_RESERVED_KEYS}
            if sub_attrs or flat_attrs:
                merged = dict(sub_attrs)
                merged.update(flat_attrs)
                attrs_batch[e_id] = merged

            out_ids.append(e_id)

        self._invalidate_sparse_caches()
        self._ensure_edge_rows_bulk(out_ids)
        if attrs_batch:
            self.attrs.set_edge_attrs_bulk(attrs_batch)

        return out_ids

    def add_hyperedges_bulk(
        self,
        hyperedges,
        *,
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
    ):
        """Hidden compatibility shim for legacy internal hyperedge insertion."""
        return self._add_hyperedges_batch(
            list(hyperedges),
            slice=slice,
            default_weight=default_weight,
            default_edge_directed=default_edge_directed,
        )

    def _add_edges_to_slice_batch(self, slice_id, edge_ids):
        """Add many edges to a slice and attach all incident vertices.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        edge_ids : Iterable[str]
            Edge identifiers to add.
        """
        slice = slice_id if slice_id is not None else self._current_slice
        L = self.slices._ensure_slice(slice)

        add_edges = {
            eid for eid in edge_ids if eid in self._edges and self._edges[eid].col_idx >= 0
        }
        if not add_edges:
            return

        L['edges'].update(add_edges)

        verts: set = set()
        for eid in add_edges:
            rec = self._edges[eid]
            if rec.etype == 'hyper':
                verts.update(rec.src)
                if rec.tgt is not None:
                    verts.update(rec.tgt)
            else:
                if rec.src is not None:
                    verts.add(rec.src)
                if rec.tgt is not None:
                    verts.add(rec.tgt)

        L['vertices'].update(verts)

    def add_edges_to_slice_bulk(self, slice_id, edge_ids):
        """Hidden compatibility shim for legacy internal slice edge attachment."""
        return self._add_edges_to_slice_batch(slice_id, edge_ids)

    def set_vertex_key(self, *fields: str):
        """Declare composite key fields and rebuild the uniqueness index.

        Parameters
        ----------
        *fields : str
            Ordered field names forming the composite key.

        Raises
        ------
        ValueError
            If duplicates exist among already-populated vertices.
        """
        if not fields:
            raise ValueError('set_vertex_key requires at least one field')
        self._vertex_key_fields = tuple(str(f) for f in fields)
        self._vertex_key_index.clear()

        df = self.vertex_attributes
        if df is None or dataframe_height(df) == 0:
            return

        missing = [f for f in self._vertex_key_fields if f not in dataframe_columns(df)]
        if missing:
            pass  # rows without those fields are simply skipped

        for row in dataframe_to_rows(df):
            vid = row.get('vertex_id')
            key = tuple(row.get(f) for f in self._vertex_key_fields)
            if any(v is None for v in key):
                continue
            owner = self._vertex_key_index.get(key)
            if owner is not None and owner != vid:
                raise ValueError(f'Composite key conflict for {key}: {owner} vs {vid}')
            self._vertex_key_index[key] = vid

    def remove_edges(
        self,
        edge_ids: str | Iterable[str],
        *,
        errors: str = 'raise',
    ) -> None:
        """Remove one edge or many edges.

        Parameters
        ----------
        edge_ids : str | Iterable[str]
            Edge ID or iterable of edge IDs to remove.
        errors : {"raise", "ignore"}, default "raise"
            ``"raise"`` (NetworkX convention) raises ``KeyError`` listing the
            unknown IDs. ``"ignore"`` silently skips them.

        Returns
        -------
        None

        Examples
        --------
        >>> G.remove_edges('e1')
        >>> G.remove_edges(['e2', 'e3'])
        >>> G.remove_edges('nope', errors='ignore')
        """
        if errors not in {'raise', 'ignore'}:
            raise ValueError(f"errors must be 'raise' or 'ignore', got {errors!r}")
        if isinstance(edge_ids, (str, bytes)):
            edge_ids = [edge_ids]
        else:
            edge_ids = list(edge_ids)

        missing = [eid for eid in edge_ids if eid not in self._edges]
        if missing and errors == 'raise':
            sample = ', '.join(repr(e) for e in missing[:3])
            suffix = '' if len(missing) <= 3 else ', ...'
            raise KeyError(f'Unknown edge id(s): {sample}{suffix}')

        to_drop = [eid for eid in edge_ids if eid in self._edges and self._edges[eid].col_idx >= 0]
        if not to_drop:
            return
        self._remove_edges_bulk(to_drop)

    def remove_vertices(
        self,
        vertex_ids: str | tuple[str, tuple[str, ...]] | Iterable[Any],
        *,
        errors: str = 'raise',
    ) -> None:
        """Remove one vertex or many vertices.

        Parameters
        ----------
        vertex_ids : str | tuple | Iterable[str | tuple]
            Vertex ID, explicit multilayer vertex key, or iterable of IDs/keys.
        errors : {"raise", "ignore"}, default "raise"
            ``"raise"`` (NetworkX convention) raises ``KeyError`` listing the
            unknown IDs. ``"ignore"`` silently skips them.

        Returns
        -------
        None

        Notes
        -----
        Incident edges are removed with each vertex.

        Examples
        --------
        >>> G.remove_vertices('A')
        >>> G.remove_vertices(['B', 'C'])
        >>> G.remove_vertices('nope', errors='ignore')
        """
        if errors not in {'raise', 'ignore'}:
            raise ValueError(f"errors must be 'raise' or 'ignore', got {errors!r}")
        if isinstance(vertex_ids, (str, bytes)):
            vertex_ids = [vertex_ids]
        elif (
            isinstance(vertex_ids, tuple)
            and len(vertex_ids) == 2
            and isinstance(vertex_ids[1], tuple)
        ):
            vertex_ids = [vertex_ids]
        else:
            vertex_ids = list(vertex_ids)

        missing = []
        to_drop = []
        for vid in vertex_ids:
            try:
                ekey = self._resolve_entity_key(vid)
            except (KeyError, ValueError, TypeError):
                missing.append(vid)
                continue
            if ekey in self._entities:
                to_drop.append(vid)
            else:
                missing.append(vid)

        if missing and errors == 'raise':
            sample = ', '.join(repr(v) for v in missing[:3])
            suffix = '' if len(missing) <= 3 else ', ...'
            raise KeyError(f'Unknown vertex id(s): {sample}{suffix}')

        if not to_drop:
            return
        self._remove_vertices_bulk(to_drop)

    def _remove_edges_bulk(self, edge_ids):
        self._ensure_edge_indexes()
        drop = set(edge_ids)
        if not drop:
            return

        keep_pairs = [(col, eid) for col, eid in self._col_to_edge.items() if eid not in drop]
        old_to_new = {old: new for new, (old, _eid) in enumerate(keep_pairs)}
        new_cols = len(keep_pairs)

        M_old = self._matrix
        rows, _cols = M_old.shape
        M_new = sp.dok_matrix((rows, new_cols), dtype=M_old.dtype)
        new_data = {(r, old_to_new[c]): v for (r, c), v in M_old.items() if c in old_to_new}
        if new_data:
            dict.update(M_new, new_data)
        self._matrix = M_new
        self._invalidate_sparse_caches()

        self._col_to_edge.clear()
        for new_idx, (_old_idx, eid) in enumerate(keep_pairs):
            self._col_to_edge[new_idx] = eid
            self._edges[eid].col_idx = new_idx

        for eid in drop:
            rec = self._edges.pop(eid, None)
            if (
                rec is not None
                and rec.etype != 'hyper'
                and rec.src is not None
                and rec.tgt is not None
            ):
                s, t = rec.src, rec.tgt
                self._unindex_edge_pair(eid, s, t)
                for v, index in ((s, self._src_to_edges), (t, self._tgt_to_edges)):
                    _lst = index.get(v)
                    if _lst:
                        try:
                            _lst.remove(eid)
                        except ValueError:
                            pass
                        if not _lst:
                            del index[v]
            rec2 = self._edges.get(eid)
            if rec2 is not None:
                rec2.ml_kind = None
            if eid in self._edges:
                self._edges[eid].ml_layers = None
        for slice_data in self._slices.values():
            slice_data['edges'].difference_update(drop)
        for d in self.slice_edge_weights.values():
            for eid in drop:
                d.pop(eid, None)

        ea = self.edge_attributes
        if ea is not None and 'edge_id' in dataframe_columns(ea):
            self.edge_attributes = dataframe_drop_rows(ea, 'edge_id', drop)
        ela = self.edge_slice_attributes
        if ela is not None and 'edge_id' in dataframe_columns(ela):
            self.edge_slice_attributes = dataframe_drop_rows(ela, 'edge_id', drop)

    def _remove_vertices_bulk(self, vertex_ids):
        drop_keys = set()
        drop_vertex_ids = set()
        for vid in vertex_ids:
            try:
                ekey = self._resolve_entity_key(vid)
            except (KeyError, ValueError, TypeError):
                continue
            if ekey not in self._entities:
                continue
            drop_keys.add(ekey)
            drop_vertex_ids.add(ekey[0] if isinstance(ekey, tuple) and len(ekey) == 2 else ekey)

        if not drop_keys:
            return

        drop_es: set = set()
        for eid, rec in list(self._edges.items()):
            if rec.etype == 'hyper':
                if drop_vertex_ids & set(rec.src):
                    drop_es.add(eid)
                elif rec.tgt is not None and (drop_vertex_ids & set(rec.tgt)):
                    drop_es.add(eid)
            else:
                if rec.src in drop_vertex_ids or rec.tgt in drop_vertex_ids:
                    drop_es.add(eid)

        if drop_es:
            self._remove_edges_bulk(drop_es)

        keep_idx = sorted(
            rec.row_idx for eid, rec in self._entities.items() if eid not in drop_keys
        )
        old_to_new = {old: new for new, old in enumerate(keep_idx)}
        new_rows = len(keep_idx)

        M_old = self._matrix
        _rows, cols = M_old.shape
        M_new = sp.dok_matrix((new_rows, cols), dtype=M_old.dtype)
        new_data = {(old_to_new[r], c): v for (r, c), v in M_old.items() if r in old_to_new}
        if new_data:
            dict.update(M_new, new_data)
        self._matrix = M_new
        self._invalidate_sparse_caches()

        new_entities: dict = {}
        new_row_to_entity: dict = {}
        for new_i, old_i in enumerate(keep_idx):
            ent = self._row_to_entity[old_i]
            old_rec = self._entities[ent]
            new_entities[ent] = EntityRecord(row_idx=new_i, kind=old_rec.kind)
            new_row_to_entity[new_i] = ent
        self._entities = new_entities
        self._row_to_entity = new_row_to_entity
        self._rebuild_entity_indexes()

        va = self.vertex_attributes
        if va is not None and 'vertex_id' in dataframe_columns(va):
            self.vertex_attributes = dataframe_drop_rows(va, 'vertex_id', drop_vertex_ids)

        for slice_data in self._slices.values():
            slice_data['vertices'].difference_update(drop_vertex_ids)

    # ------------------------------------------------------------------
    # Layer internals used by LayerAccessor
    # ------------------------------------------------------------------

    def _restore_supra_nodes(self, *args, **kwargs):
        return self.layers._restore_supra_nodes(*args, **kwargs)

    def _rebuild_all_layers_cache(self, *args, **kwargs):
        return self.layers._rebuild_all_layers_cache(*args, **kwargs)

    def _validate_layer_tuple(self, *args, **kwargs):
        return self.layers._validate_layer_tuple(*args, **kwargs)

    def nl_to_row(self, *args, **kwargs):
        """Convert a (node, layer) key to its matrix row index."""
        return self.layers.nl_to_row(*args, **kwargs)

    def row_to_nl(self, *args, **kwargs):
        """Convert a matrix row index to its (node, layer) key."""
        return self.layers.row_to_nl(*args, **kwargs)

    def _build_supra_index(self, *args, **kwargs):
        return self.layers._build_supra_index(*args, **kwargs)

    def _assert_presence(self, *args, **kwargs):
        return self.layers._assert_presence(*args, **kwargs)


for _legacy_name in AnnNet._BLOCKED_LEGACY_API:
    setattr(AnnNet, _legacy_name, _BlockedLegacyAttribute(_legacy_name))
del _legacy_name
