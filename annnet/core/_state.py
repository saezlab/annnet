"""Structural state initialization and the field inventory."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._records import (
    _EDGE_RESERVED,
    SliceRecord,
    _slice_RESERVED,
    _vertex_RESERVED,
)

# Canonical source of truth (everything else is reconstructable from these).
SOT_FIELDS = (
    'directed',
    '_aspects',
    '_layers',
    '_entities',
    '_edges',
    '_slices',
)

# Warm-cached canonical projection. Since EdgeRecord.coeffs now carries stoichiometric /
# explicit coefficients, the incidence matrix is FULLY reconstructable from the records
# (records are the complete source of truth). The matrix is nonetheless kept warm by the
# mutation gateway (incrementally patched, not re-derived per use) for linear-algebra speed,
# and IO may assign it directly. So: derived in principle, co-maintained in practice.
CO_MAINTAINED_FIELDS = ('_matrix',)

# Purely derived / cached state — rebuilt by ``_derive`` from the SoT, never hand-patched.
DERIVED_FIELDS = (
    '_csr_cache',
    '_row_to_entity',
    '_vid_to_ekeys',
    '_col_to_edge',
    '_src_to_edges',
    '_tgt_to_edges',
    '_pair_to_edges',
)


def init_state(g, *, directed=None, v=0, e=0, aspects=None) -> None:
    """Initialize the structural fields of an empty graph on ``g``."""
    g.directed = directed

    g._vertex_RESERVED = set(_vertex_RESERVED)
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

    # Entity store + derived entity indexes.
    g._entities = {}  # (vid, layer_coord) -> EntityRecord
    g._row_to_entity = {}  # row_idx -> ekey
    g._vid_to_ekeys = {}  # vid -> [ekey, ...]

    # Edge store + derived edge/adjacency indexes.
    g._edges = {}  # edge_id -> EdgeRecord
    g._col_to_edge = {}  # col_idx -> edge_id
    g._src_to_edges = {}  # src -> [edge_id, ...]
    g._tgt_to_edges = {}  # tgt -> [edge_id, ...]
    g._pair_to_edges = {}  # (src, tgt) -> [edge_id, ...]  (binary only)
    g._edge_indexes_built = True

    # Composite vertex key support.
    g._vertex_key_fields = None
    g._vertex_key_index = {}

    # Sparse incidence matrix (lazily derived from records). ``_matrix_shape``
    # tracks the logical (rows, cols) capacity; ``_matrix_cache`` holds the last
    # materialized matrix and ``_matrix_dirty`` signals a pending rebuild.
    v = int(v) if v and v > 0 else 0
    e = int(e) if e and e > 0 else 0
    g._matrix_shape = (v, e)
    g._matrix_cache = sp.csr_array((v, e), dtype=np.float32)
    g._matrix_dirty = False
    g._csr_cache = None

    g._next_edge_id = 0

    # Slice state.
    g._slices = {}
    g._default_slice = 'default'
    g._slices['default'] = SliceRecord()
    g._current_slice = 'default'

    # Version clock (dirty signal for derived sparse caches).
    g._version = 0
    g._supra_index_cache = None

    g.vertex_aligned = False
