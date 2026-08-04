"""Derived-state machinery."""

from __future__ import annotations

from . import _matrices
from ._structure import store_of

# ---------------------------------------------------------------------------
# Incidence matrix capacity + cache invalidation
# ---------------------------------------------------------------------------


def bump_structure(g) -> int:
    """Advance the structural clock, invalidating every version-keyed derived cache.

    Distinct from ``g._version``, which is a history/audit counter: it is bumped only
    by the ``_History`` hooks, is user-visible through snapshots, and deliberately does
    not move on removes. Caches that key on it go stale after a removal, so they key on
    ``_structure_version`` instead and this is the single point that advances it.
    """
    g._structure_version = getattr(g, '_structure_version', 0) + 1
    return g._structure_version


def mark_matrix_stale(g) -> None:
    """Say that the shape of the incidence matrix has changed.

    The matrix is derived from the store, and the store already holds every
    entity and every edge that a rebuild reads, so nothing about the extent has
    to be recorded. What is left is the cache: the next read rebuilds it, and
    every version-keyed derived cache goes with it.
    """
    g._matrix_dirty = True
    bump_structure(g)


def rebuild_matrix(g):
    """Materialize the incidence matrix (CSR) from the canonical store.

    The member lists of the store already are an incidence matrix in compressed
    form, so building one is a gather over its arrays rather than a pass over
    every member of every edge in Python.
    """
    return _matrices.structural_incidence(store_of(g))


def invalidate_sparse_caches(g, formats=None) -> None:
    """Invalidate all derived sparse cache views behind one internal hook."""
    formats = ('csr', 'csc', 'adjacency') if formats is None else tuple(formats)
    if 'csr' in formats:
        g._csr_cache = None
    cache_manager = getattr(g, '_cache_manager', None)
    if cache_manager is not None:
        cache_manager.invalidate(list(formats))
    g._supra_index_cache = None
    bump_structure(g)
