from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from collections import defaultdict
from collections.abc import Iterable, Iterator, MutableMapping

import numpy as np

from . import _state, _derive, _mutate, _identity, _validate
from ._Ops import Operations, OperationsAccessor
from ._Views import GraphView, ViewsClass, ViewsAccessor
from ._Layers import LayerAccessor
from ._Matrix import CacheManager, IndexManager, IndexMapping
from ._Slices import SliceManager
from ._History import History, HistoryAccessor
from ._records import (
    EdgeView,
    _external_entity_kind,
)
from ._Annotation import AttributesClass, AttributesAccessor
from ..algorithms.traversal import Traversal
from .._support.dataframe_backend import (
    empty_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    select_dataframe_backend,
)

if TYPE_CHECKING:
    from .backend_accessors.gt_accessor import _GTBackendAccessor
    from .backend_accessors.ig_accessor import _IGBackendAccessor
    from .backend_accessors.nx_accessor import _NXBackendAccessor
else:
    _GTBackendAccessor = Any
    _IGBackendAccessor = Any
    _NXBackendAccessor = Any

# ===================================


def _is_multilayer_endpoint(v) -> bool:
    """A ``(vertex_id, layer_coord)`` multilayer binary endpoint (not a member list)."""
    return (
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], tuple)
    )


def _is_hyper_item(item) -> bool:
    """Whether a batch item describes a hyperedge (list-shaped endpoints / legacy keys)."""
    if not isinstance(item, dict):
        return False
    if 'members' in item or 'head' in item or 'tail' in item:
        return True
    # Preserve 'src'/'tgt' precedence over 'source'/'target', but avoid the
    # eagerly-evaluated default of ``get('src', get('source'))`` on every item.
    src_val = item.get('src')
    if src_val is None:
        src_val = item.get('source')
    tgt_val = item.get('tgt')
    if tgt_val is None:
        tgt_val = item.get('target')
    for val in (src_val, tgt_val):
        if (
            isinstance(val, (list, tuple, set, frozenset))
            and not isinstance(val, str)
            and not _is_multilayer_endpoint(val)
        ):
            return True
    return False


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
        if self._setter:
            self._setter(self._graph, key, value)
        else:
            _mutate.set_edge_field(self._graph, key, self._field_name, value)

    def __delitem__(self, key):
        if key not in self._graph._edges:
            raise KeyError(key)
        _mutate.set_edge_field(self._graph, key, self._field_name, None)

    def __iter__(self):
        for eid, rec in self._graph._edges.items():
            value = getattr(rec, self._field_name)
            if self._include(rec, value):
                yield eid

    def __len__(self):
        return sum(1 for _ in self.__iter__())


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
        'num_supra_vertices',
        'nv',
        'ne',
        'nv_supra',
        'number_of_vertices',
        'number_of_edges',
        'shape',
        'supra_shape',
        'supra_vertices',
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
        annotations_backend: str = 'auto',
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
        _state.init_state(self, directed=directed, v=v, e=e, aspects=aspects)

        # Attribute storage
        self._annotations_backend = select_dataframe_backend(annotations_backend)
        self._init_annotation_tables(annotations)
        self.graph_attributes = {}
        self.graph_attributes.update(kwargs)

        # Per-slice edge-weight compatibility cache
        self.slice_edge_weights = defaultdict(dict)

        # History
        self._history_enabled = True
        self._history = []
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()
        self.history = HistoryAccessor(self)
        self._snapshots = []

        # Cartesian-product layer cache (set_aspects refreshes it on mutation).
        self._rebuild_all_layers_cache()

    def _grow_rows_to(self, *args, **kwargs):
        return _derive.grow_rows_to(self, *args, **kwargs)

    def _grow_cols_to(self, *args, **kwargs):
        return _derive.grow_cols_to(self, *args, **kwargs)

    def _invalidate_sparse_caches(self, *args, **kwargs):
        return _derive.invalidate_sparse_caches(self, *args, **kwargs)

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
        # Buffered vertex/edge attribute tables: backing frame + pending id rows
        # (see IndexMapping._ensure_*_row / _flush_*_rows for the rationale).
        self._vertex_attributes = None
        self._edge_attributes = None
        self._pending_vertex_ids: list = []
        self._pending_edge_ids: list = []
        self._vertex_attr_ids = None
        self._edge_attr_ids = None
        self._vertex_attr_df_id = None
        self._edge_attr_df_id = None

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

    @property
    def vertex_attributes(self):
        """Vertex (obs) attribute table; flushes buffered id-only rows on read."""
        if self._pending_vertex_ids:
            self._flush_vertex_rows()
        return self._vertex_attributes

    @vertex_attributes.setter
    def vertex_attributes(self, value):
        """Replace the vertex attribute table and clear buffered row state."""
        self._vertex_attributes = value
        self._pending_vertex_ids = []
        self._vertex_attr_ids = None
        self._vertex_attr_df_id = None

    @property
    def edge_attributes(self):
        """Edge (var) attribute table; flushes buffered id-only rows on read."""
        if self._pending_edge_ids:
            self._flush_edge_rows()
        return self._edge_attributes

    @edge_attributes.setter
    def edge_attributes(self, value):
        """Replace the edge attribute table and clear buffered row state."""
        self._edge_attributes = value
        self._pending_edge_ids = []
        self._edge_attr_ids = None
        self._edge_attr_df_id = None

    def __dir__(self):
        return sorted(set(self._PUBLIC_API))

    def __repr__(self) -> str:
        """Anndata-style multi-line summary."""
        lines = [
            f'AnnNet object with n_vertices × n_edges = {self.num_vertices} × {self.ne}',
            f'    directed: {self.directed}',
        ]

        slice_ids = list(self._slices.keys())
        if slice_ids:
            lines.append(f'    slices: {slice_ids}')

        if self._aspects and self._aspects != ('_',):
            lines.append(f'    aspects: {list(self._aspects)}')
            lines.append(f'    supra_nodes (vertex × layer rows): {self.nv_supra}')

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
        """Number of unique vertices (NetworkX convention)."""
        return self.num_vertices

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

    def _entity_row(self, *args, **kwargs):
        return _identity.entity_row(self, *args, **kwargs)

    def _placeholder_layer_coord(self, *args, **kwargs):
        return _identity.placeholder_layer_coord(self, *args, **kwargs)

    def _ensure_placeholder_layers_declared(self, *args, **kwargs):
        return _identity.ensure_placeholder_layers_declared(self, *args, **kwargs)

    def _warn_placeholder_vertex_assignment(self, *args, **kwargs):
        return _identity.warn_placeholder_vertex_assignment(self, *args, **kwargs)

    def _resolve_vertex_insert_coord(self, *args, **kwargs):
        return _identity.resolve_vertex_insert_coord(self, *args, **kwargs)

    def _make_layer_coord(self, *args, **kwargs):
        return _identity.make_layer_coord(self, *args, **kwargs)

    @staticmethod
    def _is_explicit_entity_key(*args, **kwargs):
        return _identity.is_explicit_entity_key(*args, **kwargs)

    def _resolve_entity_key(self, *args, **kwargs):
        return _identity.resolve_ekey(self, *args, **kwargs)

    def _index_entity_key(self, *args, **kwargs):
        return _derive.index_entity_key(self, *args, **kwargs)

    def _unindex_entity_key(self, *args, **kwargs):
        return _derive.unindex_entity_key(self, *args, **kwargs)

    def _register_entity_record(self, *args, **kwargs):
        return _mutate.register_entity_record(self, *args, **kwargs)

    def _remove_entity_record(self, *args, **kwargs):
        return _mutate.remove_entity_record(self, *args, **kwargs)

    def _rebuild_entity_indexes(self, *args, **kwargs):
        return _derive.rebuild_entity_indexes(self, *args, **kwargs)

    def _rebuild_edge_indexes(self, *args, **kwargs):
        return _derive.rebuild_edge_indexes(self, *args, **kwargs)

    def _ensure_edge_indexes(self, *args, **kwargs):
        return _derive.ensure_edge_indexes(self, *args, **kwargs)

    def _edge_ids_for_pair(self, *args, **kwargs):
        return _derive.edge_ids_for_pair(self, *args, **kwargs)

    def _index_edge_pair(self, *args, **kwargs):
        return _derive.index_edge_pair(self, *args, **kwargs)

    def _unindex_edge_pair(self, *args, **kwargs):
        return _derive.unindex_edge_pair(self, *args, **kwargs)

    @staticmethod
    def _remove_edge_id_from_index(*args, **kwargs):
        return _derive.remove_edge_id_from_index(*args, **kwargs)

    def _endpoint_slice_vertex_ids(self, *args, **kwargs):
        return _identity.endpoint_slice_vertex_ids(self, *args, **kwargs)

    def _slice_contains_endpoint(self, *args, **kwargs):
        return _identity.slice_contains_endpoint(self, *args, **kwargs)

    def _add_endpoint_to_slice_vertices(self, *args, **kwargs):
        return _identity.add_endpoint_to_slice_vertices(self, *args, **kwargs)

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
        return {ekey[0] for ekey, rec in self._entities.items() if rec.kind == 'vertex'}

    @property
    def _VM(self) -> set:
        return {ekey for ekey, rec in self._entities.items() if rec.kind == 'vertex'}

    @_VM.setter
    def _VM(self, value) -> None:
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
            return self._add_vertex(vertex_id, slice=slice, layer=layer, **attrs)

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

    def _add_vertices_bulk(self, vertices, *, layer=None, slice=None):
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

    def _add_vertex(self, *args, **kwargs):
        return _mutate.add_vertex(self, *args, **kwargs)

    def _ensure_edge_entity_placeholder(self, *args, **kwargs):
        return _mutate.ensure_edge_entity_placeholder(self, *args, **kwargs)

    def _register_edge_as_entity(self, *args, **kwargs):
        return _mutate.register_edge_as_entity(self, *args, **kwargs)

    # ── Edge input helpers ────────────────────────────────────────────────────

    def _parse_edge_inputs(self, *args, **kwargs):
        return _mutate.parse_edge_inputs(self, *args, **kwargs)

    @staticmethod
    def _infer_ml_kind(*args, **kwargs):
        return _mutate.infer_ml_kind(*args, **kwargs)

    @staticmethod
    def _infer_hyper_ml(*args, **kwargs):
        return _mutate.infer_hyper_ml(*args, **kwargs)

    def _find_parallel_edges(self, *args, **kwargs):
        return _mutate.find_parallel_edges(self, *args, **kwargs)

    def _zero_edge_column(self, *args, **kwargs):
        return _mutate.zero_edge_column(self, *args, **kwargs)

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
        [Adding edges](../../explanations/add-edges.md) explanation page.

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
            default_layer = kwargs.pop('layer', None)
            if kwargs:
                unexpected = ', '.join(sorted(kwargs))
                raise TypeError(f'Unexpected keyword arguments for batch add_edges: {unexpected}')

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
                    layer=default_layer,
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
                            layer=default_layer,
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

        return self._add_edge(*args, **kwargs)

    def _add_edges_bulk(
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

    def _add_edge(self, *args, **kwargs):
        return _mutate.add_edge(self, *args, **kwargs)

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
        return _mutate.set_edge_coeffs(self, edge_id, coeffs)

    def _propagate_to_shared_slices(self, *args, **kwargs):
        return _mutate.propagate_to_shared_slices(self, *args, **kwargs)

    def _propagate_to_all_slices(self, *args, **kwargs):
        return _mutate.propagate_to_all_slices(self, *args, **kwargs)

    def _normalize_vertices_arg(self, vertices):
        if vertices is None:
            return set()
        if isinstance(vertices, (str, bytes)) or self._is_explicit_entity_key(vertices):
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
        return _mutate.make_undirected(
            self, drop_flexible=drop_flexible, update_default=update_default
        )

    def validate(self, *, strict: bool = True) -> list[str]:
        """Assert internal-consistency invariants; return problems (raises if ``strict``)."""
        return _validate.validate_internal_consistency(self, strict=strict)

    # Remove / mutate down

    def remove_edge(self, *args, **kwargs):
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
        return _mutate.remove_edge(self, *args, **kwargs)

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

        See Also
        --------
        remove_vertices : Remove one or more vertices through the compact
            public API.
        """
        # Single shrink + index shift via the bulk path. Doing it per call
        # used to be O(M+V) per vertex (per-incident-edge remove_edge,
        # then a full matrix row-shift); routing through the bulk path
        # collapses that into a single pass.
        ekey = self._resolve_entity_key(vertex_id)
        if ekey not in self._entities:
            raise KeyError(f'vertex {vertex_id!r} not found')
        self._remove_vertices_bulk([vertex_id])

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
                f'[0, {self.nv_supra}). Use G.vertices() to list vertex IDs.'
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
        """Return unique vertex IDs (one per vertex, deduplicated across layers).

        Returns
        -------
        list[str]
            Distinct vertex identifiers, excluding edge-entities. In a
            multilayer graph each vertex appears exactly once regardless of
            how many elementary layers it inhabits.

        See Also
        --------
        supra_vertices : ``(vertex_id, layer_coord)`` pairs (one per row of
            the supra-incidence matrix).
        """
        seen: set[str] = set()
        out: list[str] = []
        for eid, rec in self._entities.items():
            if rec.kind != 'vertex':
                continue
            vid = eid[0] if isinstance(eid, tuple) else eid
            if vid not in seen:
                seen.add(vid)
                out.append(vid)
        return out

    def supra_vertices(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return all ``(vertex_id, layer_coord)`` supra-nodes.

        Returns
        -------
        list[tuple[str, tuple[str, ...]]]
            One entry per row of the supra-incidence matrix. In flat graphs
            the layer coordinate is the sentinel ``('_',)``.

        See Also
        --------
        vertices : unique vertex IDs (one per vertex regardless of layer).
        """
        return [
            eid
            for eid, rec in self._entities.items()
            if rec.kind == 'vertex' and isinstance(eid, tuple)
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
    def nv_supra(self) -> int:
        """Number of supra-nodes (rows of the supra-incidence matrix).

        Returns
        -------
        int
            Count of entities whose internal kind is ``"vertex"`` — one per
            ``(vertex_id, layer_coord)`` pair. In a flat graph this equals
            :attr:`nv`; in a multilayer graph it equals the sum over vertices
            of the number of layers each vertex inhabits.
        """
        return sum(1 for r in self._entities.values() if r.kind == 'vertex')

    @property
    def nv(self) -> int:
        """Number of unique vertices (deduplicated across layers).

        Returns
        -------
        int
            Distinct vertex IDs, ignoring layer multiplicity. Use
            :attr:`nv_supra` for the supra-incidence row count.
        """
        return len(self._V)

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
        """Number of unique vertices.

        Returns
        -------
        int
            Same value as :attr:`nv`. Use :attr:`num_supra_vertices` for the
            supra-incidence row count.
        """
        return self.nv

    @property
    def num_supra_vertices(self) -> int:
        """Number of supra-nodes (rows of the supra-incidence matrix).

        Returns
        -------
        int
            Same value as :attr:`nv_supra`.
        """
        return self.nv_supra

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
            Unique vertex count and edge count. Use :attr:`supra_shape` for
            ``(nv_supra, ne)``.
        """
        return (self.num_vertices, self.ne)

    @property
    def supra_shape(self) -> tuple[int, int]:
        """Supra-matrix shape as ``(nv_supra, num_edges)``.

        Returns
        -------
        tuple[int, int]
            Supra-incidence row count and edge count.
        """
        return (self.nv_supra, self.ne)

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
        self._add_vertex(vid, slice=slice, **attrs)

        # Index ownership
        self._vertex_key_index[key] = vid
        return vid

    def _gen_vertex_id_from_key(self, key) -> str:
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
            from .backend_accessors.nx_accessor import _NXBackendAccessor

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
            from .backend_accessors.ig_accessor import _IGBackendAccessor

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
            from .backend_accessors.gt_accessor import _GTBackendAccessor

            self._gt_proxy = _GTBackendAccessor(self)
        return self._gt_proxy

    # AnnNet API

    @property
    def _matrix(self):
        """Incidence matrix, rebuilt lazily from records when marked dirty.

        Records (``_edges`` + entity row indices) are the complete source of
        truth, so the matrix is a warm cache: mutation marks it dirty instead of
        patching cells, and the next read materializes a compact CSR once.
        """
        if self._matrix_dirty:
            self._matrix_cache = _derive.rebuild_matrix(self)
            self._matrix_dirty = False
        return self._matrix_cache

    @_matrix.setter
    def _matrix(self, value) -> None:
        self._matrix_cache = value
        self._matrix_shape = tuple(value.shape)
        self._matrix_dirty = False

    def _mark_matrix_dirty(self) -> None:
        """Flag the incidence cache for lazy rebuild from records."""
        self._matrix_dirty = True

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
            Passed to `annnet.io.annnet_format.write`.

        Returns
        -------
        None

        Examples
        --------
        >>> G.write('graph.annnet')
        """
        from .. import write

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
            Deserialized graph.

        Examples
        --------
        >>> G = AnnNet.read('graph.annnet')
        """
        from .. import read

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
        """Set the graph aspect names and rebuild the layer registry."""
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
        """Replace the declared elementary layers and rebuild layer caches."""
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
        """Set multilayer layer assignments for existing edges."""
        for eid, layers in dict(mapping).items():
            _mutate.set_edge_field(self, eid, 'ml_layers', layers)

    @property
    def edge_kind(self) -> dict:
        """edge_id -> kind (hyper edges use 'hyper'; others use ml_kind)."""
        return _EdgeRecordFieldMap(
            self,
            'ml_kind',
            include=lambda rec, value: rec.etype == 'hyper' or value is not None,
            getter=lambda rec, value: 'hyper' if rec.etype == 'hyper' else value,
            setter=_mutate.set_edge_kind,
        )

    @edge_kind.setter
    def edge_kind(self, mapping):
        """Set edge kinds for existing edges."""
        for eid, kind in dict(mapping).items():
            _mutate.set_edge_kind(self, eid, kind)

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
        """Rebuild the entity registry from a ``vertex_id -> row_idx`` mapping."""
        _mutate.set_entity_to_idx(self, mapping)

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
        """Set entity kinds from a ``vertex_id -> kind`` mapping."""
        _mutate.set_entity_types(self, mapping)

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
        """Rewrite binary edge endpoint definitions from a mapping."""
        for eid, defn in dict(mapping).items():
            _mutate.set_edge_definition(self, eid, *defn)

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
        """Rewrite hyperedge memberships from a mapping."""
        for eid, defn in dict(mapping).items():
            _mutate.set_hyperedge_definition(self, eid, defn)

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
        """Attach flexible-direction policies from a mapping."""
        for eid, policy in dict(mapping).items():
            _mutate.set_edge_direction_policy(self, eid, policy)

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

    def _add_vertices_batch(self, *args, **kwargs):
        return _mutate.batch_add_vertices(self, *args, **kwargs)

    def _add_edges_batch(self, *args, **kwargs):
        return _mutate.batch_add_edges(self, *args, **kwargs)

    def _add_hyperedges_batch(self, *args, **kwargs):
        return _mutate.batch_add_hyperedges(self, *args, **kwargs)

    def add_hyperedges_bulk(
        self,
        hyperedges,
        *,
        slice=None,
        default_weight=1.0,
        default_edge_directed=None,
        layer=None,
    ):
        """Hidden compatibility shim for legacy internal hyperedge insertion."""
        return self._add_hyperedges_batch(
            list(hyperedges),
            slice=slice,
            default_weight=default_weight,
            default_edge_directed=default_edge_directed,
            layer=layer,
        )

    def _add_edges_to_slice_batch(self, slice_id, edge_ids):
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
                verts.update(self._endpoint_slice_vertex_ids(rec.src))
                if rec.tgt is not None:
                    verts.update(self._endpoint_slice_vertex_ids(rec.tgt))
            else:
                verts.update(self._endpoint_slice_vertex_ids(rec.src))
                verts.update(self._endpoint_slice_vertex_ids(rec.tgt))

        L['vertices'].update(verts)

    def _add_edges_to_slice_bulk(self, slice_id, edge_ids):
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

    def _remove_edges_bulk(self, *args, **kwargs):
        return _mutate.remove_edges_bulk(self, *args, **kwargs)

    def _remove_vertices_bulk(self, *args, **kwargs):
        return _mutate.remove_vertices_bulk(self, *args, **kwargs)

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
