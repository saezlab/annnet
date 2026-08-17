from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Iterator, MutableMapping

from . import _build, _state, _derive, _mutate, _identity, _validate, _structure, _contextual
from ._Ops import Operations, OperationsAccessor
from ._attrs import AttributeStore
from ._Views import GraphView, ViewsClass, EdgeSequence, NodeSequence, ViewsAccessor
from ._Layers import LayerAccessor
from ._Matrix import CacheManager, IndexManager, IndexMapping, MatrixNamespace
from ._Slices import SliceManager
from ._History import History, HistoryAccessor
from ._records import (
    EdgeView,
    NodeView,
    _external_entity_kind,
)
from ._Annotation import AttributesClass, AttributesAccessor
from ._contextual import ContextualStore
from ._stored_kinds import STORED_EDGE_KIND, STORED_ENTITY_KIND
from ..algorithms.traversal import Traversal
from .._support.dataframe_backend import (
    clone_dataframe,
    empty_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_from_rows,
    rename_dataframe_columns,
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


def _renamed_id(table, id_column: str):
    """Return ``table`` with its id column named the way this graph names it.

    A reader that speaks another vocabulary hands over a table whose key column
    is called ``id``. The column store addresses a row by the id it carries, so
    the column has to be found before the row can be read.
    """
    columns = dataframe_columns(table)
    if id_column in columns or 'id' not in columns:
        return table
    return rename_dataframe_columns(table, {'id': id_column})


def _is_multilayer_endpoint(v) -> bool:
    """A ``(node_id, layer_coord)`` multilayer binary endpoint (not a member list)."""
    return (
        isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str) and isinstance(v[1], tuple)
    )


def _is_hyper_item(item) -> bool:
    """Whether a batch item describes a hyperedge (list-shaped endpoints / legacy keys)."""
    if not isinstance(item, dict):
        return False
    if 'members' in item or 'head' in item or 'tail' in item:
        return True
    # A batch scan calls this once per item, so the item it almost always gets
    # decides the whole cost: it names neither alias and carries two plain
    # strings. Two membership probes settle the aliases without reading a
    # value, and two type tests settle the endpoints before anything
    # list-shaped is considered.
    if 'src' in item or 'tgt' in item:
        # Preserve 'src'/'tgt' precedence over 'source'/'target', but avoid the
        # eagerly-evaluated default of ``get('src', get('source'))``.
        src_val = item.get('src')
        if src_val is None:
            src_val = item.get('source')
        tgt_val = item.get('tgt')
        if tgt_val is None:
            tgt_val = item.get('target')
    else:
        src_val = item.get('source')
        tgt_val = item.get('target')
        if type(src_val) is str and type(tgt_val) is str:
            return False
    for val in (src_val, tgt_val):
        if (
            isinstance(val, (list, tuple, set, frozenset))
            and not isinstance(val, str)
            and not _is_multilayer_endpoint(val)
        ):
            return True
    return False


class _EdgeFieldMap(MutableMapping):
    """Mutable mapping view over one field of every edge of a graph.

    It reads through the facade and writes through the gateway, so it knows
    nothing about which store holds the field.
    """

    def __init__(self, graph, field_name, *, include, getter=None, setter=None):
        self._graph = graph
        self._field_name = field_name
        self._include = include
        self._getter = getter
        self._setter = setter

    def __getitem__(self, key):
        ref = _structure.edge_ref(self._graph, key)
        value = getattr(ref, self._field_name)
        if not self._include(ref, value):
            raise KeyError(key)
        return self._getter(ref, value) if self._getter else value

    def __setitem__(self, key, value):
        if not _structure.has_edge(self._graph, key):
            raise KeyError(key)
        if self._setter:
            self._setter(self._graph, key, value)
        else:
            _mutate.set_edge_field(self._graph, key, self._field_name, value)

    def __delitem__(self, key):
        if not _structure.has_edge(self._graph, key):
            raise KeyError(key)
        _mutate.set_edge_field(self._graph, key, self._field_name, None)

    def __iter__(self):
        for ref in _structure.iter_edges(self._graph, include_placeholders=True):
            value = getattr(ref, self._field_name)
            if self._include(ref, value):
                yield ref.id

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


def _is_endpoint_pair(item) -> bool:
    """Return True when a value names two endpoints of one edge.

    A two-element tuple is either a pair of endpoints or one ``(id, layer)``
    entity key. A layer coordinate is itself a tuple, and an endpoint id is not,
    so the second element says which of the two this is.
    """
    return (
        isinstance(item, tuple) and len(item) == 2 and all(isinstance(part, str) for part in item)
    )


def _element_operand(other):
    """Classify the right operand of ``+=`` or ``-=``.

    Returns the pair ``(kind, items)``, where ``kind`` is ``'nodes'`` or
    ``'edges'``, or ``(None, None)`` when the operand names neither. A single
    element is a collection of one, so both arrive here as a list.
    """
    if isinstance(other, str):
        return 'nodes', [other]
    if isinstance(other, tuple):
        return ('edges', [other]) if _is_endpoint_pair(other) else (None, None)
    if isinstance(other, dict):
        return 'edges', [other]
    if isinstance(other, (list, set, frozenset)):
        items = list(other)
        if not items:
            return 'nodes', items
        first = items[0]
        if isinstance(first, str):
            return 'nodes', items
        if isinstance(first, (tuple, dict)):
            return 'edges', items
    return None, None


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
    entity and edge registries. A row represents an entity, typically a node
    or an edge-entity, and a column represents an edge. The class supports:

    - binary directed and undirected edges
    - hyperedges, including directed head/tail hyperedges
    - edge-entities that can themselves participate as endpoints
    - slice membership and per-slice edge weights
    - optional multilayer coordinates on nodes and edges
    - dataframe-backed attribute storage

    Parameters
    ----------
    directed : bool | None, optional
        Default directedness for newly created binary edges. If ``None``,
        methods fall back to directed semantics unless a per-edge flag is set.
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
    add_node
    add_edge
    add_nodes
    add_edges
    view
    """

    _PUBLIC_API = (
        'add_nodes',
        'add_edges',
        'remove_nodes',
        'remove_edges',
        'has_node',
        'has_edge',
        'nodes',
        'edges',
        'degree',
        'incident_edges',
        'ncount',
        'ecount',
        'nv',
        'ne',
        'nv_supra',
        'shape',
        'supra_shape',
        'supra_nodes',
        'N',
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
        'get_node',
        'get_edge',
        'neighbors',
        'edge_list',
        'make_undirected',
        'is_multilayer',
        'A',
        'B',
        'H',
        'S',
        'L',
        'matrices',
    )

    _BLOCKED_LEGACY_API = frozenset(
        {
            'add_node',
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
            'get_slice_nodes',
            'get_slice_edges',
            'slice_union',
            'slice_intersection',
            'slice_difference',
            'create_slice_from_operation',
            'create_aggregated_slice',
            'slice_statistics',
            'node_presence_across_slices',
            'edge_presence_across_slices',
            'hyperedge_presence_across_slices',
            'conserved_edges',
            'slice_specific_edges',
            'temporal_dynamics',
            'set_graph_attribute',
            'get_graph_attribute',
            'get_graph_attributes',
            'set_node_attrs',
            'set_node_attrs_bulk',
            'get_node_attrs',
            'get_attr_node',
            'get_attr_nodes',
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
            'nodes_view',
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
            'node_incidence_matrix',
            'get_node_incidence_matrix_as_lists',
            'set_aspects',
            'set_elementary_layers',
            'add_elementary_layer',
            'flatten_layers',
            'has_presence',
            'iter_layers',
            'iter_node_layers',
            'layer_id_to_tuple',
            'layer_tuple_to_id',
            'supra_adjacency',
            'supra_incidence',
            'build_intra_block',
            'build_inter_block',
            'build_coupling_block',
            'layer_node_set',
            'layer_edge_set',
            'layer_union',
            'layer_intersection',
            'layer_difference',
            'set_aspect_attrs',
            'list_aspects',
            'list_layers',
        }
    )

    # Construction
    def __init__(
        self,
        directed: bool | None = None,
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
        _state.init_state(self, directed=directed, aspects=aspects)

        # Attribute storage
        self._annotations_backend = select_dataframe_backend(annotations_backend)
        self._init_annotation_tables(annotations)
        self.graph_attributes = {}
        self.graph_attributes.update(kwargs)

        # Per-slice edge-weight compatibility cache

        # History
        self._history_enabled = True
        self._history = []
        self._history_clock0 = time.perf_counter_ns()
        self._install_history_hooks()
        self._snapshots = []

        # Cartesian-product layer cache (set_aspects refreshes it on mutation).
        self._rebuild_all_layers_cache()

    def _invalidate_sparse_caches(self, *args, **kwargs):
        return _derive.invalidate_sparse_caches(self, *args, **kwargs)


    @property
    def slice_edge_weights(self):
        """Per-slice edge weight overrides, as ``{slice_id: {edge_id: weight}}``.

        Derived from the edge-by-slice attributes on each read. It used to be a
        cache kept beside them and synchronised by two routines, which is one
        copy of the same fact too many: the two could disagree, and a removal had
        to remember to prune both.
        """
        if self._pending_edge_slice_drops:
            self._flush_edge_slice_rows()
        out: dict[str, dict[str, float]] = {}
        for (slice_id, edge_id), attrs in self._contextual.edge_slice_attrs.items():
            weight = attrs.get('weight')
            if weight is None or (isinstance(weight, float) and weight != weight):
                continue
            out.setdefault(slice_id, {})[edge_id] = float(weight)
        return out

    @slice_edge_weights.setter
    def slice_edge_weights(self, value):
        """Install overrides by writing them where the weight actually lives."""
        for slice_id, weights in (value or {}).items():
            for edge_id, weight in (weights or {}).items():
                self._contextual.set('edge_slice_attrs', (slice_id, edge_id), {'weight': float(weight)})

    def _init_annotation_tables(self, annotations):
        # The generic node and edge attributes live in the slot-indexed column
        # store. The graph holds no table of its own for them: ``_node_table``
        # and ``_edge_table`` are built from those columns when a reader asks.
        self._attr_store = AttributeStore(
            self._store, node_id_column='node_id', edge_id_column='edge_id'
        )
        # Every contextual level — the ones keyed by a pair — in one store. See
        # ``annnet.core._contextual`` for why none of them is a dataframe.
        self._contextual = ContextualStore()
        # A layer table a caller assigned that carries no ``layer_id``.
        self._layer_table_passthrough = None
        # Materialized contextual tables, against the store's version.
        self._contextual_tables: dict = {}
        self._edge_slice_attributes = None
        self._pending_edge_slice_drops: set = set()

        # 1) If user provided tables, keep them (we’ll wrap with Narwhals in ops)
        if annotations is not None:
            self._node_table = annotations.get('_node_table')
            self._edge_table = annotations.get('_edge_table')
            self.slice_attributes = annotations.get('slice_attributes')
            self.edge_slice_attributes = annotations.get('edge_slice_attributes')
            self.layer_attributes = annotations.get('layer_attributes')
            return

        # 2) Otherwise, create empty tables with the centrally selected backend.
        backend = self._annotations_backend
        self.slice_attributes = empty_dataframe({'slice_id': 'text'}, backend=backend)
        self.edge_slice_attributes = empty_dataframe(
            {'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
            backend=backend,
        )
        self.layer_attributes = empty_dataframe({'layer_id': 'text'}, backend=backend)

    @property
    def directed(self):
        """The direction an edge takes when it declares none.

        The canonical store answers with it too, so the two cannot drift apart.
        An edge that inherits the default would otherwise report the direction
        the graph had when it was built.
        """
        return self._directed

    @directed.setter
    def directed(self, value) -> None:
        self._directed = value
        # A graph sets its default before it builds its store, and the store
        # takes the same value when it is built.
        store = getattr(self, '_store', None)
        if store is not None:
            store.directed = value

    @property
    def _node_table(self):
        """The node table, built from the attribute columns when it is read.

        The graph stores columns and not a frame, so this is a table for the
        caller. One build serves every read until the next write.
        """
        return self._attr_store.obs(backend=self._annotations_backend)

    @_node_table.setter
    def _node_table(self, value):
        """Give the attribute columns what the rows of a table say.

        A caller that holds a whole table states the node attributes of the
        graph with it. A row for a node the graph does not hold names nothing,
        so it is left out.
        """
        self._attr_store.load_node_rows(
            () if value is None else dataframe_to_rows(_renamed_id(value, 'node_id'))
        )

    @property
    def _edge_table(self):
        """The edge table, built from the attribute columns when it is read."""
        return self._attr_store.var(backend=self._annotations_backend)

    @_edge_table.setter
    def _edge_table(self, value):
        """Give the attribute columns what the rows of a table say."""
        self._attr_store.load_edge_rows(
            () if value is None else dataframe_to_rows(_renamed_id(value, 'edge_id'))
        )

    @property
    def edge_slice_attributes(self):
        """The edge-by-slice attributes, as a table.

        Built on each read from the contextual store, in this graph's annotation
        backend. It is a rendering of canonical state, not the state itself, so
        editing the returned table changes nothing — write through
        ``G.attrs.set_edge_slice_attrs``.
        """
        if self._pending_edge_slice_drops:
            self._flush_edge_slice_rows()
        return self._contextual_table(
            'edge_slice_attrs',
            ('slice_id', 'edge_id'),
            {'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
        )

    @edge_slice_attributes.setter
    def edge_slice_attributes(self, value):
        """Install the edge-by-slice attributes from a table."""
        self._install_contextual_table('edge_slice_attrs', ('slice_id', 'edge_id'), value)
        self._pending_edge_slice_drops = set()

    @property
    def slice_attributes(self):
        """The per-slice attributes, as a table.

        Built on each read; see :attr:`edge_slice_attributes`.
        """
        return self._contextual_table('slice_attrs', 'slice_id', {'slice_id': 'text'})

    @slice_attributes.setter
    def slice_attributes(self, value):
        self._install_contextual_table('slice_attrs', 'slice_id', value)

    @property
    def layer_attributes(self):
        """The layer attribute table.

        Two things share this name. ``G.layers.set_elementary_attrs`` writes rows
        keyed by ``layer_id``, and those are canonical in the contextual store and
        rendered here. A caller may also assign a table of their own that carries
        no ``layer_id`` — the elementary-layer API cannot read it, but adapters
        round-trip it, so it is kept verbatim and handed back unchanged.
        """
        if self._layer_table_passthrough is not None:
            return self._layer_table_passthrough
        return self._contextual_table('elementary_attrs', 'layer_id', {'layer_id': 'text'})

    @layer_attributes.setter
    def layer_attributes(self, value):
        if value is not None and 'layer_id' not in dataframe_columns(value):
            # Not addressable by the elementary-layer API. Keep it as given.
            self._layer_table_passthrough = value
            self._contextual.elementary_attrs.clear()
            return
        self._layer_table_passthrough = None
        self._install_contextual_table('elementary_attrs', 'layer_id', value)

    def contextual_table(self, level_name, *, backend=None):
        """Render one contextual level as a table, in the backend you name.

        The store holds dicts, so the backend is a property of this call and not
        of the graph. ``G.contextual_table('slice_attrs', backend='pandas')``
        answers in pandas whatever the graph was constructed with.

        Parameters
        ----------
        level_name : str
            One of :data:`annnet.core._contextual.LEVELS`.
        backend : {"polars", "pandas", "pyarrow"}, optional
            Defaults to the graph's annotation backend.

        Returns
        -------
        DataFrame-like
        """
        keys, schema = _CONTEXTUAL_TABLE_SHAPE[level_name]
        return self._contextual_table(level_name, keys, schema, backend=backend)

    def _contextual_table(self, level_name, key_columns, schema, *, backend=None):
        """Render one contextual level as a table, cached against the level.

        A read after a read costs nothing; a write drops the entry. This is the
        same discipline the matrices use, and it is what keeps the dict-canonical
        model from paying for a rebuild on every access.
        """
        level = getattr(self._contextual, level_name)
        backend = backend or self._annotations_backend
        token = (backend, self._contextual.version)
        cached = self._contextual_tables.get(level_name)
        if cached is not None and cached[0] == token:
            return cached[1]
        rows = _contextual.rows_of(level, key_columns)
        table = (
            empty_dataframe(schema, backend=backend)
            if not rows
            else dataframe_from_rows(rows, backend=backend)
        )
        self._contextual_tables[level_name] = (token, table)
        return table


    def _install_contextual_table(self, level_name, key_columns, value) -> None:
        """Fill one contextual level from a table, a mapping, or nothing."""
        level = getattr(self._contextual, level_name)
        if value is None:
            level.clear()
            return
        if isinstance(value, dict):
            level.clear()
            level.update({key: dict(attrs) for key, attrs in value.items()})
            return
        _contextual.install_rows(level, dataframe_to_rows(value), key_columns)

    def __dir__(self):
        return sorted(set(self._PUBLIC_API))

    def __repr__(self) -> str:
        """Anndata-style multi-line summary."""
        lines = [
            f'AnnNet object with n_nodes × n_edges = {self.nv} × {self.ne}',
            f'    directed: {self.directed}',
        ]

        slice_ids = list(self._slices.keys())
        if slice_ids:
            lines.append(f'    slices: {slice_ids}')

        if self._aspects and self._aspects != ('_',):
            lines.append(f'    aspects: {list(self._aspects)}')
            lines.append(f'    supra_nodes (node × layer rows): {self.nv_supra}')

        def _user_cols(df, id_field: str) -> list[str]:
            try:
                cols = [c for c in dataframe_columns(df) if c != id_field]
            except (AttributeError, TypeError):
                return []
            return cols

        obs_cols = _user_cols(self._node_table, 'node_id')
        if obs_cols:
            lines.append(f'    obs: {obs_cols!r}')

        var_cols = _user_cols(self._edge_table, 'edge_id')
        if var_cols:
            lines.append(f'    var: {var_cols!r}')

        uns_keys = list(self.graph_attributes.keys()) if self.graph_attributes else []
        if uns_keys:
            lines.append(f'    uns: {uns_keys!r}')

        return '\n'.join(lines)

    def __len__(self) -> int:
        """Number of nodes, the same count as :meth:`ncount`."""
        return self.ncount()

    def __iter__(self) -> Iterator[str]:
        """Iterate over node IDs (NetworkX convention)."""
        return iter(self.nodes())

    def __contains__(self, item) -> bool:
        """Membership test. A string is a node and a pair is an edge.

        A pair of endpoints asks whether an edge joins them. A ``(id, layer)``
        key asks about one node on one layer, which is why the second element
        decides which of the two a pair is.
        """
        if _is_endpoint_pair(item):
            # Asked with two endpoints, ``has_edge`` answers with the edges too.
            answer = self.has_edge(item[0], item[1])
            return bool(answer[0] if isinstance(answer, tuple) else answer)
        try:
            ekey = self._resolve_entity_key(item)
        except (KeyError, TypeError, ValueError):
            return False
        if not _structure.has_entity(self, ekey):
            return False
        return _structure.entity_ref(self, ekey).kind == _structure.NODE

    def __bool__(self) -> bool:
        """True when the graph holds any node."""
        return self.ncount() > 0

    # ── Operators ─────────────────────────────────────────────────────────────
    #
    # The type of the right operand decides what an operator does, and nothing
    # else. A string is one node, a tuple is one edge, a list holds many of
    # either, and a graph runs set algebra over the two element sets.

    def __iadd__(self, other):
        """Add nodes or edges. ``G += "A"`` and ``G += ("A", "B")``."""
        kind, items = _element_operand(other)
        if kind is None:
            return NotImplemented
        if kind == 'nodes':
            self.add_nodes(items)
        else:
            self.add_edges(items)
        return self

    def __isub__(self, other):
        """Remove nodes or edges. ``G -= "A"`` and ``G -= ("A", "B")``."""
        kind, items = _element_operand(other)
        if kind is None:
            return NotImplemented
        if kind == 'nodes':
            self.remove_nodes(items)
        else:
            self.remove_edges([self._edge_id_of_pair(item) for item in items])
        return self

    def _edge_id_of_pair(self, item):
        """Return the id of the edge a ``(source, target)`` pair names."""
        if isinstance(item, str):
            return item
        found, edge_ids = self.has_edge(item[0], item[1])
        if not found:
            raise KeyError(f'no edge joins {item[0]!r} and {item[1]!r}')
        return edge_ids[0]

    def __or__(self, other):
        """Union of two graphs. The left operand wins where the two disagree."""
        if not isinstance(other, AnnNet):
            return NotImplemented
        return self.ops.union(other)

    def __ior__(self, other):
        """Merge another graph into this one."""
        if not isinstance(other, AnnNet):
            return NotImplemented
        return self.ops.merge(other)

    def __and__(self, other):
        """Intersection of two graphs."""
        if not isinstance(other, AnnNet):
            return NotImplemented
        return self.ops.intersection(other)

    def __sub__(self, other):
        """Difference of two graphs. Use ``-=`` to remove one element."""
        if not isinstance(other, AnnNet):
            return NotImplemented
        return self.ops.difference(other)

    def __xor__(self, other):
        """Symmetric difference of two graphs."""
        if not isinstance(other, AnnNet):
            return NotImplemented
        return self.ops.symmetric_difference(other)

    def _placeholder_layer_coord(self, *args, **kwargs):
        return _identity.placeholder_layer_coord(self, *args, **kwargs)

    def _ensure_placeholder_layers_declared(self, *args, **kwargs):
        return _identity.ensure_placeholder_layers_declared(self, *args, **kwargs)

    def _resolve_node_insert_coord(self, *args, **kwargs):
        return _identity.resolve_node_insert_coord(self, *args, **kwargs)

    def _make_layer_coord(self, *args, **kwargs):
        return _identity.make_layer_coord(self, *args, **kwargs)

    @staticmethod
    def _is_explicit_entity_key(*args, **kwargs):
        return _identity.is_explicit_entity_key(*args, **kwargs)

    def _resolve_entity_key(self, *args, **kwargs):
        return _identity.resolve_ekey(self, *args, **kwargs)

    def _register_entity(self, *args, **kwargs):
        return _mutate.register_entity(self, *args, **kwargs)

    # The doors a reader uses. A reader rebuilds a whole graph from a file or from
    # another library, so it writes structure rather than querying it, and the
    # structural query facade has no answer for that. Each of these installs or
    # changes canonical state and keeps every store of the graph in step.

    def _install_structure(self, *args, **kwargs):
        return _build.install_structure(self, *args, **kwargs)

    def _set_entity_kinds(self, *args, **kwargs):
        return _mutate.set_entity_kinds(self, *args, **kwargs)

    def _remap_entity_keys(self, *args, **kwargs):
        return _mutate.remap_entity_keys(self, *args, **kwargs)

    def _set_edge_field(self, *args, **kwargs):
        return _mutate.set_edge_field(self, *args, **kwargs)

    def _set_hyperedge_members(self, *args, **kwargs):
        return _mutate.set_hyperedge_members(self, *args, **kwargs)

    def _replace_edge_coeffs(self, *args, **kwargs):
        return _mutate.replace_edge_coeffs(self, *args, **kwargs)

    def _endpoint_slice_node_ids(self, *args, **kwargs):
        return _identity.endpoint_slice_node_ids(self, *args, **kwargs)

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
        return {key[0] for key in _structure.node_keys(self)}

    def _VM_ordered(self) -> list:
        """Return the key of every node, in row order.

        A writer needs the order as well as the membership, because a file that
        records the rows of a graph has to record them in the order the graph
        holds them.
        """
        return _structure.node_keys(self)

    @property
    def _VM(self) -> set:
        return set(_structure.node_keys(self))

    @_VM.setter
    def _VM(self, value) -> None:
        if value:
            try:
                self._restore_supra_nodes(value)
            except Exception:  # noqa: BLE001
                pass

    # Build graph

    def add_nodes(
        self,
        nodes: str | dict[str, Any] | tuple[str, dict[str, Any]] | Iterable[Any],
        slice: str | None = None,
        layer: str | tuple[str, ...] | dict[str, str] | None = None,
        **attributes: Any,
    ) -> str | list[str]:
        """Add one node or many nodes.

        This is the canonical public entry point for node creation. Use it
        for both single-node and batch insertion.

        Parameters
        ----------
        nodes : str | dict | tuple | Iterable
            Node specification or iterable of specifications.

            Accepted single-node forms are:

            - ``"A"``
            - ``{"node_id": "A", "kind": "source"}``
            - ``{"id": "A", ...}``
            - ``{"name": "A", ...}``
            - ``("A", {"kind": "source"})``

            Accepted batch forms are iterables of the same specifications.
        slice : str, optional
            Slice receiving the inserted nodes. If omitted, the active slice
            is used.
        layer : str | tuple | dict, optional
            Layer coordinate for inserted nodes in multilayer graphs. A
            string is valid only for single-aspect graphs; a tuple must already
            be in aspect order; a dict maps aspect name to layer value.
        **attributes
            Attributes applied to a single node. These are merged with
            attributes in ``nodes`` when ``nodes`` is a single node.

        Returns
        -------
        str | list[str]
            The inserted node ID for a single node, or a list of node IDs
            for batch insertion.

        Raises
        ------
        ValueError
            If a dictionary node specification does not contain
            ``"node_id"``, ``"id"``, or ``"name"``.

        Notes
        -----
        Node attributes are stored in :attr:`obs` and can be edited through
        :attr:`attrs`. In multilayer graphs, omitting ``layer`` places the
        node on the placeholder layer coordinate.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_nodes('A', kind='source')
        'A'
        >>> G.add_nodes(
        ...     [
        ...         {'node_id': 'B', 'kind': 'relay'},
        ...         ('C', {'kind': 'sink'}),
        ...     ]
        ... )
        ['B', 'C']
        """
        is_single = False
        if isinstance(nodes, (str, bytes, dict)):
            is_single = True
        elif isinstance(nodes, tuple) and nodes:
            is_single = len(nodes) == 1 or (
                len(nodes) == 2 and isinstance(nodes[0], str) and isinstance(nodes[1], dict)
            )

        if is_single:
            if isinstance(nodes, dict):
                if nodes.get('node_id') is not None:
                    node_id = nodes['node_id']
                    attrs = {k: v for k, v in nodes.items() if k != 'node_id'}
                elif nodes.get('id') is not None:
                    node_id = nodes['id']
                    attrs = {k: v for k, v in nodes.items() if k != 'id'}
                elif nodes.get('name') is not None:
                    node_id = nodes['name']
                    attrs = {k: v for k, v in nodes.items() if k != 'name'}
                else:
                    raise ValueError('node dict must contain one of: node_id, id, name')
            elif isinstance(nodes, tuple):
                node_id = nodes[0]
                attrs = nodes[1] if len(nodes) > 1 and isinstance(nodes[1], dict) else {}
            else:
                node_id = nodes
                attrs = {}
            attrs.update(attributes)
            return self._add_node(node_id, slice=slice, layer=layer, **attrs)

        items = list(nodes)
        self._add_nodes_batch(items, layer=layer, slice=slice, default_attrs=attributes or None)
        out = []
        for it in items:
            if isinstance(it, dict):
                if it.get('node_id') is not None:
                    out.append(it['node_id'])
                elif it.get('id') is not None:
                    out.append(it['id'])
                elif it.get('name') is not None:
                    out.append(it['name'])
            elif isinstance(it, (tuple, list)) and it:
                out.append(it[0])
            else:
                out.append(it)
        return out

    def _add_nodes_bulk(self, nodes, *, layer=None, slice=None):
        items = list(nodes)
        self._add_nodes_batch(items, layer=layer, slice=slice)
        out = []
        for it in items:
            if isinstance(it, dict):
                if it.get('node_id') is not None:
                    out.append(it['node_id'])
                elif it.get('id') is not None:
                    out.append(it['id'])
                elif it.get('name') is not None:
                    out.append(it['name'])
            elif isinstance(it, (tuple, list)) and it:
                out.append(it[0])
            else:
                out.append(it)
        return out

    def _add_node(self, *args, **kwargs):
        return _mutate.add_node(self, *args, **kwargs)

    def _ensure_edge_entity_placeholder(self, *args, **kwargs):
        return _mutate.ensure_edge_entity_placeholder(self, *args, **kwargs)

    # ── Edge input helpers ────────────────────────────────────────────────────

    @staticmethod
    def _infer_ml_kind(*args, **kwargs):
        return _mutate.infer_ml_kind(*args, **kwargs)

    @staticmethod
    def _infer_hyper_ml(*args, **kwargs):
        return _mutate.infer_hyper_ml(*args, **kwargs)

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
        >>> G.add_nodes(['A', 'B', 'C'])
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
            if isinstance(candidate, dict):
                # One edge is a batch of one. The two paths would otherwise
                # answer differently for the same edge, stated the same way.
                batch_candidate = [candidate]
            elif not isinstance(candidate, (str, bytes)):
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

            # Two flags rather than a set of two names: the scan walks the whole
            # batch whenever it is uniform, which is the case worth paying for,
            # and a set costs a string hash and a length read per item.
            has_hyper = has_binary = False
            for item in batch_candidate:
                if _is_hyper_item(item):
                    has_hyper = True
                else:
                    has_binary = True
                if has_hyper and has_binary:
                    break

            if has_hyper and not has_binary:
                return self._add_hyperedges_batch(
                    batch_candidate,
                    slice=default_slice,
                    default_weight=default_weight,
                    default_edge_directed=default_edge_directed,
                    layer=default_layer,
                )

            if not has_hyper:
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

    def _normalize_nodes_arg(self, nodes):
        if nodes is None:
            return set()
        if isinstance(nodes, (str, bytes)) or self._is_explicit_entity_key(nodes):
            return {nodes}
        try:
            return set(nodes)
        except TypeError:
            return {nodes}

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
        >>> G.add_nodes(['A', 'B'])
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
        """Check the internal consistency of the graph and report every problem.

        The check covers the agreement between identity and address, the member
        list of every edge, the link between the two sides of an edge-entity,
        the agreement between the store and the materialized matrix, the level
        of the node table and the edge table, and slice membership.

        The check picks its rules from the store the graph holds, so the same
        call serves the record store and the slot-addressed store.

        Parameters
        ----------
        strict : bool, default True
            Raise ``AssertionError`` when the graph has any problem.

        Returns
        -------
        list[str]
            One message per problem. An empty list means a consistent graph.

        Examples
        --------
        >>> G = AnnNet(directed=True)
        >>> G.add_edges('A', 'B')
        'edge_0'
        >>> G.validate()
        []
        """
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

    def remove_node(self, node_id):
        """Remove a node and all incident edges (binary + hyperedges).

        Parameters
        ----------
        node_id : str
            Node identifier.

        Raises
        ------
        KeyError
            If the node is not found.

        See Also
        --------
        remove_nodes : Remove one or more nodes through the compact
            public API.
        """
        # Single shrink + index shift via the bulk path. Doing it per call
        # used to be O(M+V) per node (per-incident-edge remove_edge,
        # then a full matrix row-shift); routing through the bulk path
        # collapses that into a single pass.
        ekey = self._resolve_entity_key(node_id)
        if not _structure.has_entity(self, ekey):
            raise KeyError(f'node {node_id!r} not found')
        self._remove_nodes_bulk([node_id])

    def remove_orphans(self):
        """Remove all nodes with no incident edges from the AnnNet graph."""
        csr = self._get_csr()
        orphans = []
        for idx, ref in enumerate(_structure.iter_entities(self)):
            if ref.kind == _structure.NODE:
                if csr.indptr[idx + 1] - csr.indptr[idx] == 0:
                    orphans.append(ref.key)
        if orphans:
            self._remove_nodes_bulk(orphans)
        return len(orphans)

    # Basic queries & metrics

    def get_node(self, node_id: str) -> NodeView:
        """Return a :class:`NodeView` for one node.

        Parameters
        ----------
        node_id : str
            Node identifier. A lookup takes an id and nothing else. A caller
            holding a row of the incidence matrix asks
            ``G.idx.row_to_entity(row)`` for the identity on it, and a caller
            who wants the n-th node of a sequence writes ``G.N[n]``.

        Returns
        -------
        NodeView
            A string-shaped record equal to the id. ``kind``, ``layers`` and
            ``attrs`` are exposed as attributes.

        Raises
        ------
        TypeError
            If the argument is not an id.
        KeyError
            If the id is unknown.
        """
        if not isinstance(node_id, str):
            raise TypeError(
                f'get_node takes a node id, not {type(node_id).__name__}. For the '
                f'node on a matrix row, use G.idx.row_to_entity(row).'
            )
        keys = self._store.entity_keys_of_id(node_id)
        if not keys:
            raise KeyError(f'Unknown node id: {node_id}')
        ref = _structure.entity_ref(self, keys[0])
        return NodeView(
            node_id,
            kind=_external_entity_kind(STORED_ENTITY_KIND[ref.kind]),
            layers=tuple(layer for _id, layer in keys),
            attrs=self._attr_store.node_attrs(node_id),
        )

    def get_edge(self, edge_id: str) -> EdgeView:
        """Return an :class:`EdgeView` for one edge.

        Parameters
        ----------
        edge_id : str
            Edge identifier. A lookup takes an id and nothing else. A caller
            holding a column of the incidence matrix asks
            ``G.idx.col_to_edge(column)`` for the id on it.

        Returns
        -------
        EdgeView
            A tuple-shaped record. ``(source, target)`` tuple unpacking still
            works; ``edge_id``, ``kind``, ``members``, ``weight`` and
            ``directed`` are also exposed as attributes.

        Raises
        ------
        TypeError
            If the argument is not an id.
        KeyError
            If the id is unknown.
        """
        if not isinstance(edge_id, str):
            raise TypeError(
                f'get_edge takes an edge id, not {type(edge_id).__name__}. For the '
                f'edge on a matrix column, use G.idx.col_to_edge(column).'
            )
        if not _structure.has_edge(self, edge_id):
            raise KeyError(f'Unknown edge id: {edge_id}') from None

        return self._edge_tuple(edge_id)

    def _edge_tuple(self, eid: str) -> EdgeView:
        """Return the public view of one edge.

        An undirected edge shows the same members on both sides, because neither
        side means a direction.
        """
        ref = _structure.edge_ref(self, eid)
        sides = _structure.edge_sides(self, eid)
        members = sides.source | sides.target

        if ref.kind == _structure.HYPER:
            kind = 'hyper_directed' if ref.directed else 'hyper_undirected'
        else:
            kind = STORED_EDGE_KIND[ref.kind]

        if ref.directed or ref.kind in (_structure.NODE_EDGE, _structure.PLACEHOLDER):
            source, target = sides.source, sides.target
        else:
            source = target = members

        return EdgeView(
            source,
            target,
            edge_id=eid,
            kind=kind,
            members=members,
            weight=ref.weight,
            directed=ref.directed,
        )

    def _is_directed_edge(self, edge_id):
        if not _structure.has_edge(self, edge_id):
            return bool(self.directed)
        d = _structure.edge_ref(self, edge_id).declared_directed
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
            return _structure.has_edge(self, edge_id)

        # ---- Mode 2: source + target only ----
        if edge_id is None and source is not None and target is not None:
            eids = _structure.edges_between(self, source, target)
            return (len(eids) > 0, eids)

        # ---- Mode 3: edge_id + source + target ----
        if edge_id is not None and source is not None and target is not None:
            if not _structure.has_edge(self, edge_id):
                return False
            sides = _structure.edge_sides(self, edge_id)
            return sides.source == frozenset({source}) and sides.target == frozenset({target})

        # ---- Anything else is ambiguous / invalid ----
        raise ValueError(
            'Invalid argument combination: use either '
            '(edge_id), (source,target), or (source,target,edge_id).'
        )

    def has_node(self, node_id: str) -> bool:
        """Check whether a node exists.

        Parameters
        ----------
        node_id : str | tuple
            Bare node ID, or explicit ``(node_id, layer_coord)`` tuple for
            multilayer graphs.

        Returns
        -------
        bool
            ``True`` if the graph contains a node entity matching
            ``node_id``.

        Notes
        -----
        In multilayer graphs, a bare node ID returns ``True`` if that node
        is present on at least one layer coordinate.
        """
        if isinstance(node_id, str):
            return _structure.has_entity_id(self, node_id, kind=_structure.NODE)

        ekey = self._resolve_entity_key(node_id)
        if not _structure.has_entity(self, ekey):
            return False
        return _structure.entity_ref(self, ekey).kind == _structure.NODE

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
        return _structure.edges_between(self, source, target)

    def _get_csr(self):
        csr = self.cache.csr
        self._csr_cache = csr
        return csr

    def degree(self, entity_id):
        """Return the incidence degree of a node or edge-entity.

        Parameters
        ----------
        entity_id : str | tuple
            Node ID, edge-entity ID, or explicit multilayer entity key.

        Returns
        -------
        int
            Number of non-zero incidence entries in the entity row. Missing
            entities have degree ``0``.
        """
        ekey = self._resolve_entity_key(entity_id)
        try:
            row = _structure.entity_row(self, ekey)
        except KeyError:
            return 0
        csr = self._get_csr()
        return int(csr.indptr[row + 1] - csr.indptr[row])

    def nodes(self) -> list[str]:
        """Return unique node IDs (one per node, deduplicated across layers).

        Returns
        -------
        list[str]
            Distinct node identifiers, excluding edge-entities. In a
            multilayer graph each node appears exactly once regardless of
            how many elementary layers it inhabits.

        See Also
        --------
        supra_nodes : ``(node_id, layer_coord)`` pairs (one per row of
            the supra-incidence matrix).
        """
        return _structure.node_ids(self)

    def supra_nodes(self) -> list[tuple[str, tuple[str, ...]]]:
        """Return all ``(node_id, layer_coord)`` supra-nodes.

        Returns
        -------
        list[tuple[str, tuple[str, ...]]]
            One entry per row of the supra-incidence matrix. In flat graphs
            the layer coordinate is the sentinel ``('_',)``.

        See Also
        --------
        nodes : unique node IDs (one per node regardless of layer).
        """
        return _structure.node_keys(self)

    def edges(self) -> list[str]:
        """Return all structural edge IDs.

        Returns
        -------
        list[str]
            Edge identifiers for edges with an incidence-matrix column.
        """
        return _structure.edge_ids(self)

    def edge_list(self) -> list[tuple[str, str, str, float]]:
        """Materialize binary edges as endpoint tuples.

        Returns
        -------
        list[tuple[str, str, str, float]]
            Tuples of ``(source, target, edge_id, weight)`` for binary and
            node-edge records. Hyperedges and endpoint-less placeholders are
            omitted. The ``weight`` reflects the active slice's per-edge
            override when one is set; otherwise the edge's stored weight.
        """
        edges = []
        get_eff = self.attrs.get_effective_edge_weight
        for ref in _structure.iter_edges(self):
            if ref.kind == _structure.HYPER:
                continue
            sides = _structure.edge_sides(self, ref.id)
            if not sides.source or not sides.target:
                continue
            edges.append(
                (
                    next(iter(sides.source)),
                    next(iter(sides.target)),
                    ref.id,
                    get_eff(ref.id),
                )
            )
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
            ref.id
            for ref in _structure.iter_edges(self)
            if bool(ref.declared_directed if ref.declared_directed is not None else default_dir)
            is bool(directed)
        ]

    def global_count(self, kind: str) -> int:
        """Count unique members present across slices.

        Parameters
        ----------
        kind : {"nodes", "edges", "entities"}
            Membership domain. ``"nodes"`` counts slice node members,
            ``"edges"`` counts slice edge members, and ``"entities"`` counts
            the union of both domains.

        Returns
        -------
        int
            Number of unique members observed in slice membership.

        Raises
        ------
        ValueError
            If ``kind`` is not one of ``"nodes"``, ``"edges"``, or
            ``"entities"``.

        Notes
        -----
        This is a slice-membership count, not a storage count. For graph
        storage counts, use :meth:`ncount` and :meth:`ecount`.
        """
        if kind not in {'nodes', 'edges', 'entities'}:
            raise ValueError("kind must be one of {'nodes', 'edges', 'entities'}")
        members = set()
        for slice_data in self._slices.values():
            if kind in {'nodes', 'entities'}:
                members.update(slice_data['nodes'])
            if kind in {'edges', 'entities'}:
                members.update(slice_data['edges'])
        return len(members)

    # ── Backward-compat thin wrappers ─────────────────────────────────────────

    def in_edges(self, nodes):
        """Incoming edges. Prefer ``incident_edges(direction='in')``."""
        return self.incident_edges(nodes, direction='in')

    def out_edges(self, nodes):
        """Outgoing edges. Prefer ``incident_edges(direction='out')``."""
        return self.incident_edges(nodes, direction='out')

    # ── Traversal ────────────────────────────────────────────────────────────

    def incident_edges(
        self,
        nodes: str | Iterable[str],
        direction: str = 'both',
    ) -> list[tuple[int, EdgeView]]:
        """Return edges incident to one or more nodes.

        Parameters
        ----------
        nodes : str | Iterable[str]
            One node identifier or an iterable of identifiers.
        direction : {"in", "out", "both"}, optional
            Directional filter applied to binary edges. Undirected edges are
            included for both ``"in"`` and ``"out"``.

        Returns
        -------
        list[tuple[int, EdgeView]]
            Pairs of ``(column_index, edge_view)`` as returned by
            :meth:`get_edge`, materialized for consistency with the sibling
            ``nodes`` / ``edges`` / ``edge_list`` APIs which all return
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
        V = self._normalize_nodes_arg(nodes)
        if not V:
            return []
        seen = set()
        result = []
        for v in V:
            for eid in _structure.entity_edges(self, v, direction):
                if eid in seen:
                    continue
                seen.add(eid)
                column = _structure.edge_column(self, eid)
                if column >= 0:
                    result.append((column, self._edge_tuple(eid)))
        return result

    @property
    def nv_supra(self) -> int:
        """Number of supra-nodes (rows of the supra-incidence matrix).

        Returns
        -------
        int
            Count of entities whose internal kind is ``"node"`` — one per
            ``(node_id, layer_coord)`` pair. In a flat graph this equals
            :attr:`nv`; in a multilayer graph it equals the sum over nodes
            of the number of layers each node inhabits.
        """
        return _structure.node_count(self)

    @property
    def nv(self) -> int:
        """Number of unique nodes (deduplicated across layers).

        Returns
        -------
        int
            Distinct node IDs, ignoring layer multiplicity. Use
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
        return _structure.edge_count(self)

    def ncount(self, *, supra: bool = False) -> int:
        """Number of nodes.

        Parameters
        ----------
        supra : bool, optional
            Count supra-nodes instead of nodes. A supra-node is one node on one
            layer coordinate, so a flat graph gives the same answer either way.

        Returns
        -------
        int
            Node count, or supra-node count when ``supra`` is set.
        """
        return self.nv_supra if supra else self.nv

    def ecount(self) -> int:
        """Number of edges.

        Returns
        -------
        int
            Count of structural edges.
        """
        return self.ne

    @property
    def shape(self) -> tuple[int, int]:
        """Graph shape as ``(nv, ne)``.

        Returns
        -------
        tuple[int, int]
            Node count and edge count. Use :attr:`supra_shape` for
            ``(nv_supra, ne)``.
        """
        return (self.nv, self.ne)

    @property
    def supra_shape(self) -> tuple[int, int]:
        """Supra-matrix shape as ``(nv_supra, ne)``.

        Returns
        -------
        tuple[int, int]
            Supra-incidence row count and edge count.
        """
        return (self.nv_supra, self.ne)

    def get_or_create_node_by_attrs(self, slice=None, **attrs) -> str:
        """Return node ID for the given composite-key attributes.

        Parameters
        ----------
        slice : str, optional
            Slice to place a newly created node into.
        **attrs
            Attributes used to build the composite key.

        Returns
        -------
        str
            Node ID matching the composite key.

        Raises
        ------
        RuntimeError
            If no composite key fields are configured.
        ValueError
            If required key fields are missing.

        Notes
        -----
        Requires `set_node_key(...)` to have been called.
        """
        if not self._node_key_fields:
            raise RuntimeError('Call set_node_key(...) before using get_or_create_node_by_attrs')

        key = self._build_key_from_attrs(attrs)
        if key is None:
            missing = [f for f in self._node_key_fields if f not in attrs or attrs[f] is None]
            raise ValueError(f'Missing composite key fields: {missing}')

        # Existing?
        owner = self._node_key_index.get(key)
        if owner is not None:
            return owner

        # Create new node
        vid = self._gen_node_id_from_key(key)
        # No need to pre-check entity_to_idx here; ids are namespaced by 'cid:' prefix
        self._add_node(vid, slice=slice, **attrs)

        # Index ownership
        self._node_key_index[key] = vid
        return vid

    def _gen_node_id_from_key(self, key) -> str:
        from urllib.parse import quote

        base = 'cid:' + '|'.join(quote(str(part), safe='') for part in key)
        vid = base
        i = 1
        while self.has_node(vid):
            current = self._current_key_of_node(vid)
            if current == key:
                return vid
            vid = f'{base}::{i}'
            i += 1
        return vid

    @property
    def N(self):
        """The node sequence.

        Returns
        -------
        NodeSequence
            The nodes in graph order. Iterating it yields ids, a string key
            reads or writes one attribute column, an integer key gives the node
            at that position, and ``select`` and ``find`` filter it.

        Examples
        --------
        >>> list(G.N)
        >>> G.N['kind']
        >>> G.N['kind'] = values
        >>> G.N.find(id='A')
        """
        return NodeSequence(self)

    @property
    def E(self):
        """The edge sequence.

        Returns
        -------
        EdgeSequence
            The edges in graph order, with the same keys as :attr:`N`. The
            direction, the weight, and the kind of an edge read like a column,
            so a filter over them needs no attribute.

        Examples
        --------
        >>> list(G.E)
        >>> G.E['weight']
        >>> G.E.select(directed=True)
        """
        return EdgeSequence(self)

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
        """The signed incidence matrix of every structural edge.

        This is the same matrix as :attr:`S`, and it comes from the same cache.
        The store holds every entity, every edge and the coefficient of every
        member, so the matrix is derived from it and nothing keeps the two in
        step: a read after a write extends the cached matrix by the columns the
        write appended, and rebuilds it when the write was anything else.
        """
        return self.matrices.signed_matrix()

    def _mark_structure_changed(self) -> None:
        """Drop the caches that a structural change invalidates.

        The matrix is not one of them. It is keyed to the clock of the store,
        which every write to the store advances, so it needs no hook here. What
        does is the supra (node-layer) index, which is keyed to nothing, and
        the structural clock that the version-keyed caches read.

        The removal paths in ``_mutate`` bump the clock through
        ``_derive.bump_structure`` and never come through here.
        """
        self._supra_index_cache = None
        _derive.bump_structure(self)

    # The named matrices. Each is one purpose-built projection of the member
    # lists the store holds, so no one matrix has to carry a convention another
    # one needs. Read ``G.matrices`` for the same matrices with the maps between
    # an identity and the position it holds, and for the parameterized forms.

    @property
    def B(self):
        """Return the incidence matrix of the binary edges.

        Returns
        -------
        scipy.sparse.sparray
            Rows are entities and columns are binary edges, both in the order
            the store holds them.
        """
        return self.matrices.binary_matrix()

    @property
    def H(self):
        """Return the incidence matrix of the hyperedges.

        Returns
        -------
        scipy.sparse.sparray
            Unsigned, so an entry reports that a hyperedge holds an entity
            rather than which side it holds it on.
        """
        return self.matrices.hypergraph_matrix()

    @property
    def S(self):
        """Return the coefficient incidence matrix of every edge.

        Returns
        -------
        scipy.sparse.sparray
            Signed, so the two entries of a self-loop cancel and a
            stoichiometric column carries the coefficients the user set.
        """
        return self.matrices.signed_matrix()

    @property
    def A(self):
        """Return the adjacency matrix.

        Returns
        -------
        scipy.sparse.sparray
            A self-loop lands on the diagonal. A boundary edge joins nothing and
            a hyperedge names more than two entities, so neither is here.
        """
        return self.matrices.adjacency_matrix()

    @property
    def L(self):
        """Return the graph Laplacian, the degree matrix minus the adjacency.

        Returns
        -------
        scipy.sparse.sparray
            Every row sums to zero.
        """
        return self.matrices.laplacian_matrix()

    @property
    def matrices(self):
        """Return the parameterized-matrix namespace of this graph.

        Returns
        -------
        MatrixNamespace
        """
        if not hasattr(self, '_matrix_namespace'):
            self._matrix_namespace = MatrixNamespace(self)
        return self._matrix_namespace

    # ``x @ G`` for a numpy array on the left. Without this, the array handles
    # the operator itself and raises before the graph is asked, because ``@`` is
    # a ufunc and an array claims every ufunc it appears in.
    __array_ufunc__ = None

    def __matmul__(self, other):
        """Apply the adjacency matrix to ``other``, so ``G @ x`` is ``G.A @ x``."""
        return self.A @ other

    def __rmatmul__(self, other):
        """Apply ``other`` to the adjacency matrix, so ``x @ G`` is ``x @ G.A``."""
        return other @ self.A

    @property
    def obs(self) -> Any:
        """The node attribute table, materialized on each read.

        Returns
        -------
        DataFrame-like
            One row per node, with the id column first.

        Notes
        -----
        This is a table built for the caller and not the storage of the graph,
        so writing to it changes nothing. Write through :attr:`N` for a whole
        column, or through :attr:`attrs` for one value.

        A whole table is the expensive way to read one column. ``G.N["kind"]``
        is the cheap one.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_nodes([{'node_id': 'A', 'kind': 'source'}])
        >>> G.obs
        """
        return clone_dataframe(self._node_table)

    @property
    def var(self) -> Any:
        """The edge attribute table, materialized on each read.

        Returns
        -------
        DataFrame-like
            One row per edge, with the id column first.

        Notes
        -----
        This is a table built for the caller and not the storage of the graph,
        so writing to it changes nothing. Write through :attr:`E` for a whole
        column, or through :attr:`attrs` for one value.

        Examples
        --------
        >>> G = AnnNet()
        >>> G.add_nodes(['A', 'B'])
        >>> G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1'}])
        >>> G.var
        """
        return clone_dataframe(self._edge_table)

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
            Manager for graph-, node-, edge-, slice-, and edge-slice
            annotations.

        Notes
        -----
        Use this namespace for graph-, node-, edge-, and slice-level
        annotations.

        Examples
        --------
        >>> G.attrs.set_node_attrs('A', symbol='TP53')
        >>> G.attrs.get_node_attrs('A')
        >>> G.attrs.set_edge_slice_attrs('baseline', 'e1', weight=0.5)
        """
        try:
            return self._attrs_accessor
        except AttributeError:
            self._attrs_accessor = AttributesAccessor(self)
            return self._attrs_accessor

    @property
    def history(self) -> HistoryAccessor:
        """Mutation history and snapshot namespace.

        Returns
        -------
        HistoryAccessor
            Callable namespace: ``G.history()`` reads the log, and its methods
            enable it, clear it, export it, mark it, and snapshot the graph.

        Examples
        --------
        >>> G.history()
        >>> G.history.snapshot('before')
        """
        try:
            return self._history_accessor
        except AttributeError:
            self._history_accessor = HistoryAccessor(self)
            return self._history_accessor

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
        >>> G.views.nodes()
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
        >>> M = G.ops.node_incidence_matrix(sparse=True)
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
    def write(self, path, *, matrix: bool = False, **kwargs):
        """Write the graph to the native ``.annnet`` format.

        Parameters
        ----------
        path : str | pathlib.Path
            Output file path.
        matrix : bool, default False
            Also persist the incidence matrix. The records are the source of truth
            and fully reconstruct it — including explicit coefficients, which are
            written as records data either way — so this is a size/load-time trade,
            never a correctness one. Left off, ``read`` defers the rebuild until the
            matrix is first touched, which is usually cheaper than loading it. Turn
            it on for graphs large enough that the rebuild dominates.
        **kwargs
            Passed to `annnet.io.annnet_format.write`.

        Returns
        -------
        None

        Examples
        --------
        >>> G.write('graph.annnet')  # matrix rebuilt on demand
        >>> G.write('graph.annnet', matrix=True)  # cache it alongside
        """
        from .. import write

        write(self, path, matrix=matrix, **kwargs)

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
    def view(self, nodes=None, edges=None, slices=None, predicate=None):
        """Create a lazy graph view.

        Parameters
        ----------
        nodes : Iterable[str], optional
            Node IDs to include.
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
        return GraphView(self, nodes, edges, slices, predicate)

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
                'node_ids': set(_structure.node_keys(ref)),
                'edge_ids': set(_structure.edge_ids(ref)),
                'slice_ids': set(ref._slices.keys()),
            }
        else:
            raise TypeError(f'Invalid snapshot reference: {type(ref)}')

    def _current_snapshot(self):
        return {
            'label': 'current',
            'version': self._version,
            'node_ids': {key[0] for key in _structure.node_keys(self)},
            'edge_ids': set(_structure.edge_ids(self)),
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
    # Computed read properties. Each answers from the facade, so a map of one
    # edge field costs no knowledge of which store holds it. Mutation goes
    # through the gateway, which is what keeps every store of the graph in step.
    # -------------------------------------------------------------------------

    @property
    def edge_layers(self) -> MutableMapping:
        """edge_id -> ml_layers for all edges that have a layer assignment."""
        return _EdgeFieldMap(
            self,
            'ml_layers',
            include=lambda _ref, value: value is not None,
        )

    @edge_layers.setter
    def edge_layers(self, mapping):
        """Set multilayer layer assignments for existing edges."""
        for eid, layers in dict(mapping).items():
            _mutate.set_edge_field(self, eid, 'ml_layers', layers)

    @property
    def edge_kind(self) -> MutableMapping:
        """edge_id -> kind (hyper edges use 'hyper'; others use ml_kind)."""
        return _EdgeFieldMap(
            self,
            'ml_kind',
            include=lambda ref, value: ref.kind == _structure.HYPER or value is not None,
            getter=lambda ref, value: 'hyper' if ref.kind == _structure.HYPER else value,
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

    def _set_entity_kinds_by_id(self, mapping):
        """Set entity kinds from a ``node_id -> kind`` mapping (a reader door)."""
        _mutate.set_entity_types(self, mapping)

    @property
    def edge_weights(self) -> dict:
        """edge_id -> weight for all edges."""
        return {
            ref.id: ref.declared_weight
            for ref in _structure.iter_edges(self, include_placeholders=True)
        }

    @property
    def edge_directed(self) -> dict:
        """edge_id -> directed for edges with an explicit directedness flag."""
        return {
            ref.id: ref.declared_directed
            for ref in _structure.iter_edges(self, include_placeholders=True)
            if ref.declared_directed is not None
        }

    @property
    def edge_definitions(self) -> dict:
        """edge_id -> (src, tgt, etype) for binary edges."""
        out = {}
        for ref in _structure.iter_edges(self):
            if ref.kind == _structure.HYPER:
                continue
            sides = _structure.edge_sides(self, ref.id)
            if not sides.source:
                continue
            out[ref.id] = (
                next(iter(sides.source)),
                next(iter(sides.target)) if sides.target else None,
                STORED_EDGE_KIND[ref.kind],
            )
        return out

    @edge_definitions.setter
    def edge_definitions(self, mapping):
        """Rewrite binary edge endpoint definitions from a mapping."""
        for eid, defn in dict(mapping).items():
            _mutate.set_edge_definition(self, eid, *defn)

    @property
    def hyperedge_definitions(self) -> dict:
        """edge_id -> hyper metadata dict for hyperedges."""
        out = {}
        for ref in _structure.iter_edges(self):
            if ref.kind != _structure.HYPER:
                continue
            sides = _structure.edge_sides(self, ref.id)
            if ref.directed:
                out[ref.id] = {
                    'directed': True,
                    'head': set(sides.source),
                    'tail': set(sides.target),
                }
            else:
                out[ref.id] = {'directed': False, 'members': set(sides.source)}
        return out

    @hyperedge_definitions.setter
    def hyperedge_definitions(self, mapping):
        """Rewrite hyperedge memberships from a mapping."""
        for eid, defn in dict(mapping).items():
            _mutate.set_hyperedge_definition(self, eid, defn)

    @property
    def edge_direction_policy(self) -> dict:
        """edge_id -> direction_policy for edges that have one set."""
        return _structure.edge_policies(self)

    @edge_direction_policy.setter
    def edge_direction_policy(self, mapping):
        """Attach flexible-direction policies from a mapping."""
        for eid, policy in dict(mapping).items():
            _mutate.set_edge_direction_policy(self, eid, policy)

    @property
    def _num_entities(self) -> int:
        return _structure.entity_count(self)

    @_num_entities.setter
    def _num_entities(self, value) -> None:
        return None

    @property
    def _num_edges(self) -> int:
        return _structure.edge_count(self)

    @_num_edges.setter
    def _num_edges(self, value) -> None:
        return None

    # ------------------------------------------------------------------
    # Bulk mutation API
    # ------------------------------------------------------------------

    def _add_nodes_batch(self, *args, **kwargs):
        return _mutate.batch_add_nodes(self, *args, **kwargs)

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
            eid
            for eid in edge_ids
            if _structure.has_edge(self, eid) and _structure.carries_structure(self, eid)
        }
        if not add_edges:
            return

        L['edges'].update(add_edges)

        verts: set = set()
        for eid in add_edges:
            sides = _structure.edge_sides(self, eid)
            for member in sides.source | sides.target:
                verts.update(self._endpoint_slice_node_ids(member))

        L['nodes'].update(verts)

    def _add_edges_to_slice_bulk(self, slice_id, edge_ids):
        return self._add_edges_to_slice_batch(slice_id, edge_ids)

    def set_node_key(self, *fields: str):
        """Declare composite key fields and rebuild the uniqueness index.

        Parameters
        ----------
        *fields : str
            Ordered field names forming the composite key.

        Raises
        ------
        ValueError
            If duplicates exist among already-populated nodes.
        """
        if not fields:
            raise ValueError('set_node_key requires at least one field')
        self._node_key_fields = tuple(str(f) for f in fields)
        self._node_key_index.clear()

        df = self._node_table
        if df is None or dataframe_height(df) == 0:
            return

        missing = [f for f in self._node_key_fields if f not in dataframe_columns(df)]
        if missing:
            pass  # rows without those fields are simply skipped

        for row in dataframe_to_rows(df):
            vid = row.get('node_id')
            key = tuple(row.get(f) for f in self._node_key_fields)
            if any(v is None for v in key):
                continue
            owner = self._node_key_index.get(key)
            if owner is not None and owner != vid:
                raise ValueError(f'Composite key conflict for {key}: {owner} vs {vid}')
            self._node_key_index[key] = vid

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

        missing = [eid for eid in edge_ids if not _structure.has_edge(self, eid)]
        if missing and errors == 'raise':
            sample = ', '.join(repr(e) for e in missing[:3])
            suffix = '' if len(missing) <= 3 else ', ...'
            raise KeyError(f'Unknown edge id(s): {sample}{suffix}')

        to_drop = [
            eid
            for eid in edge_ids
            if _structure.has_edge(self, eid) and _structure.carries_structure(self, eid)
        ]
        if not to_drop:
            return
        self._remove_edges_bulk(to_drop)

    def remove_nodes(
        self,
        node_ids: str | tuple[str, tuple[str, ...]] | Iterable[Any],
        *,
        errors: str = 'raise',
    ) -> None:
        """Remove one node or many nodes.

        Parameters
        ----------
        node_ids : str | tuple | Iterable[str | tuple]
            Node ID, explicit multilayer node key, or iterable of IDs/keys.
        errors : {"raise", "ignore"}, default "raise"
            ``"raise"`` (NetworkX convention) raises ``KeyError`` listing the
            unknown IDs. ``"ignore"`` silently skips them.

        Returns
        -------
        None

        Notes
        -----
        Incident edges are removed with each node.

        Examples
        --------
        >>> G.remove_nodes('A')
        >>> G.remove_nodes(['B', 'C'])
        >>> G.remove_nodes('nope', errors='ignore')
        """
        if errors not in {'raise', 'ignore'}:
            raise ValueError(f"errors must be 'raise' or 'ignore', got {errors!r}")
        if isinstance(node_ids, (str, bytes)):
            node_ids = [node_ids]
        elif isinstance(node_ids, tuple) and len(node_ids) == 2 and isinstance(node_ids[1], tuple):
            node_ids = [node_ids]
        else:
            node_ids = list(node_ids)

        missing = []
        to_drop = []
        for vid in node_ids:
            try:
                ekey = self._resolve_entity_key(vid)
            except (KeyError, ValueError, TypeError):
                missing.append(vid)
                continue
            if _structure.has_entity(self, ekey):
                to_drop.append(vid)
            else:
                missing.append(vid)

        if missing and errors == 'raise':
            sample = ', '.join(repr(v) for v in missing[:3])
            suffix = '' if len(missing) <= 3 else ', ...'
            raise KeyError(f'Unknown node id(s): {sample}{suffix}')

        if not to_drop:
            return
        self._remove_nodes_bulk(to_drop)

    def _remove_edges_bulk(self, *args, **kwargs):
        return _mutate.remove_edges_bulk(self, *args, **kwargs)

    def _remove_nodes_bulk(self, *args, **kwargs):
        return _mutate.remove_nodes_bulk(self, *args, **kwargs)

    def _remove_orphan_node_layers(self, *args, **kwargs):
        return _mutate.remove_orphan_node_layers(self, *args, **kwargs)

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


# The column shape of each contextual level, so ``contextual_table`` can render
# any of them by name without the caller repeating the key columns.
_CONTEXTUAL_TABLE_SHAPE = {
    'slice_attrs': ('slice_id', {'slice_id': 'text'}),
    'edge_slice_attrs': (
        ('slice_id', 'edge_id'),
        {'slice_id': 'text', 'edge_id': 'text', 'weight': 'float'},
    ),
    'elementary_attrs': ('layer_id', {'layer_id': 'text'}),
    'aspect_attrs': ('aspect', {'aspect': 'text'}),
    'layer_attrs': ('layer', {'layer': 'text'}),
    'node_layer_attrs': (('node_id', 'layer'), {'node_id': 'text', 'layer': 'text'}),
}
