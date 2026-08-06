"""Structural state initialization and the field inventory."""

from __future__ import annotations

from typing import Any

from ._records import (
    _EDGE_RESERVED,
    SliceRecord,
    _node_RESERVED,
    _slice_RESERVED,
)

# Canonical source of truth (everything else is reconstructable from these).
SOT_FIELDS = (
    'directed',
    '_aspects',
    '_layers',
    '_store',
    '_slices',
)

# Purely derived / cached state — rebuilt by ``_derive`` from the SoT, never
# hand-patched. The incidence matrix is one of these: it lives in the matrix
# cache of the store, which is why no field of the graph holds it.
DERIVED_FIELDS = ('_csr_cache',)


def init_state(g, *, directed=None, aspects=None) -> None:
    """Initialize the structural fields of an empty graph on ``g``."""
    g.directed = directed

    g._node_RESERVED = set(_node_RESERVED)
    g._EDGE_RESERVED = set(_EDGE_RESERVED)
    g._slice_RESERVED = set(_slice_RESERVED)

    # Aspect / layer registry (aspect count is immutable after init).
    if aspects is None:
        g._aspects = ('_',)
        g._layers = {'_': {'_'}}
    else:
        if not aspects:
            raise ValueError('aspects dict must not be empty')
        for asp, vals in aspects.items():
            if not vals:
                raise ValueError(f'Aspect {asp!r} must have at least one layer value')
        g._aspects = tuple(aspects.keys())
        g._layers = {k: set(val) for k, val in aspects.items()}

    # Composite node key support.
    g._node_key_fields = None
    g._node_key_index = {}

    # The incidence matrix is derived state, so init builds nothing and holds
    # no field for it. It is materialized on the first read and kept in the
    # matrix cache of the store, against the clock of that store.
    g._csr_cache = None

    g._next_edge_id = 0

    # Slice state.
    g._slices = {}
    g._default_slice = 'default'
    g._slices['default'] = SliceRecord()
    g._current_slice = 'default'

    # History/audit clock: bumped only by ``_History._log_event`` / ``_log_mutation``.
    # User-visible via ``_current_snapshot()``; drives snapshot & diff numbering. It does
    # NOT track structural mutations (removes and ``set_aspects`` leave it unchanged), so
    # it must never be used to key a derived cache.
    g._version = 0

    # Structural clock: bumped by every add / remove / rekey that can change the
    # incidence structure (see ``_derive.bump_structure``). This is the key derived
    # caches must validate against.
    g._structure_version = 0

    g._supra_index_cache = None

    g.node_aligned = False

    # The canonical store. It is built last, because it takes the aspects the
    # graph has just declared and answers in the identity form they imply.
    from ._store import CoreState

    g._store = CoreState(directed=directed, aspects=g._aspects)


class GraphState:
    """The fields of a graph, declared once for the mixins that read them.

    ``AnnNet`` is assembled from mixins, and :func:`init_state` and the
    constructor fill its fields. A mixin therefore reads a field it never
    declares, which leaves a reader — and a type checker — to work out from the
    whole class what any one method needs.

    Every field below is an annotation and not an assignment, so this class
    holds nothing at runtime and every mixin that inherits it is unchanged. What
    it gives is one place that says what a graph is made of.
    """

    # The canonical state (see SOT_FIELDS).
    directed: Any
    _store: Any
    _slices: dict
    _aspects: tuple
    _layers: dict

    # The attribute columns, and the two contextual tables that are frames.
    _attr_store: Any
    _annotations_backend: Any
    graph_attributes: dict
    slice_attributes: Any
    edge_slice_attributes: Any
    layer_attributes: Any
    slice_edge_weights: Any

    # Slices.
    _default_slice: str
    _current_slice: str | None

    # The composite node key, when a caller declares one.
    _node_key_fields: Any
    _node_key_index: dict

    # Reserved names, per level.
    _node_RESERVED: set
    _EDGE_RESERVED: set
    _slice_RESERVED: set

    # History and its clock.
    _history: list
    _history_enabled: bool
    _history_clock0: int
    _snapshots: list
    _version: int

    # Derived state and the clocks it is checked against.
    _structure_version: int
    _csr_cache: Any
    _supra_index_cache: Any
    _next_edge_id: int
    node_aligned: bool

    # The namespaces and the public calls a mixin reaches for. They live on
    # ``AnnNet`` or on another mixin, so naming them here is what lets one
    # mixin call another without the whole class in view.
    attrs: Any
    ops: Any
    slices: Any
    layers: Any
    views: Any
    history: Any
    _history_accessor: Any
    idx: Any
    cache: Any
    add_nodes: Any
    add_edges: Any
    remove_nodes: Any
    remove_edges: Any
    nodes: Any
    edges: Any
    has_node: Any
    has_edge: Any
    obs: Any
    var: Any
    N: Any
    E: Any
    _node_table: Any
    _edge_table: Any
    _matrix: Any
    _add_nodes_bulk: Any
    _add_edges_bulk: Any
    _install_history_hooks: Any
    _mark_structure_changed: Any
    _invalidate_sparse_caches: Any
    _resolve_entity_key: Any
    _upsert_row: Any
    _upsert_rows_bulk: Any
