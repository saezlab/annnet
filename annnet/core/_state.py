"""Structural state initialization and the field inventory."""

from __future__ import annotations

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
