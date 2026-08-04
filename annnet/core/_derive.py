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


# ---------------------------------------------------------------------------
# Entity indexes:  _row_to_entity, _vid_to_ekeys
# ---------------------------------------------------------------------------


def index_entity_key(g, ekey) -> None:
    # Flat graphs (single placeholder layer) never need the bare-vid -> [ekey]
    # index: every vid has exactly one ekey ``(vid, ('_',))``, which resolve_ekey
    # returns directly. Skipping it saves the largest non-record index in memory.
    # A later set_aspects() rebuilds the index for the multilayer graph.
    """Register an entity key in the bare-vertex lookup index when needed."""
    if g._aspects == ('_',):
        return
    if isinstance(ekey, tuple) and len(ekey) == 2 and isinstance(ekey[0], str):
        bucket = g._vid_to_ekeys.setdefault(ekey[0], [])
        if ekey not in bucket:
            bucket.append(ekey)


def unindex_entity_key(g, ekey) -> None:
    """Remove an entity key from the bare-vertex lookup index."""
    if isinstance(ekey, tuple) and len(ekey) == 2 and isinstance(ekey[0], str):
        bucket = g._vid_to_ekeys.get(ekey[0])
        if not bucket:
            return
        try:
            bucket.remove(ekey)
        except ValueError:
            return
        if not bucket:
            g._vid_to_ekeys.pop(ekey[0], None)


def rebuild_entity_indexes(g) -> None:
    """Rebuild ``_row_to_entity`` and ``_vid_to_ekeys`` from canonical entity records."""
    g._row_to_entity = {}
    g._vid_to_ekeys = {}
    for ekey, rec in g._entities.items():
        g._row_to_entity[rec.row_idx] = ekey
        index_entity_key(g, ekey)


# ---------------------------------------------------------------------------
# Adjacency indexes:  _src_to_edges, _tgt_to_edges, _pair_to_edges (binary edges)
# ---------------------------------------------------------------------------


def rebuild_col_index(g) -> None:
    """Rebuild ``_col_to_edge`` (column index -> edge id) from canonical edge records."""
    g._col_to_edge = {rec.col_idx: eid for eid, rec in g._edges.items() if rec.col_idx >= 0}


def rebuild_edge_indexes(g) -> None:
    """Rebuild adjacency-derived edge indexes from canonical edge records."""
    g._src_to_edges = {}
    g._tgt_to_edges = {}
    g._pair_to_edges = {}
    for eid, rec in g._edges.items():
        if rec.etype == 'hyper' or rec.src is None or rec.tgt is None:
            continue
        g._src_to_edges.setdefault(rec.src, []).append(eid)
        g._tgt_to_edges.setdefault(rec.tgt, []).append(eid)
        g._pair_to_edges.setdefault((rec.src, rec.tgt), []).append(eid)
    g._edge_indexes_built = True


def clear_edge_indexes(g) -> None:
    """Drop the adjacency-derived edge indexes so the first query rebuilds them."""
    g._src_to_edges = {}
    g._tgt_to_edges = {}
    g._pair_to_edges = {}
    g._edge_indexes_built = False


def ensure_edge_indexes(g) -> None:
    """Materialize adjacency-derived edge indexes on demand."""
    if not getattr(g, '_edge_indexes_built', True):
        rebuild_edge_indexes(g)


def edge_ids_for_pair(g, source, target) -> list[str]:
    """Return edge ids for a binary endpoint pair from canonical src-edge buckets."""
    ensure_edge_indexes(g)
    eids = []
    for eid in g._src_to_edges.get(source, ()):
        rec = g._edges.get(eid)
        if rec is not None and rec.etype != 'hyper' and rec.tgt == target:
            eids.append(eid)
    if eids:
        g._pair_to_edges[(source, target)] = list(eids)
    else:
        g._pair_to_edges.pop((source, target), None)
    return eids


def index_edge_pair(g, edge_id, src, tgt) -> None:
    """Register an edge ID in the binary endpoint-pair index."""
    if src is None or tgt is None:
        return
    bucket = g._pair_to_edges.setdefault((src, tgt), [])
    if edge_id not in bucket:
        bucket.append(edge_id)


def unindex_edge_pair(g, edge_id, src, tgt) -> None:
    """Remove an edge ID from the binary endpoint-pair index."""
    if src is None or tgt is None:
        return
    bucket = g._pair_to_edges.get((src, tgt))
    if not bucket:
        return
    try:
        bucket.remove(edge_id)
    except ValueError:
        return
    if not bucket:
        g._pair_to_edges.pop((src, tgt), None)


def remove_edge_id_from_index(index: dict, key, edge_id: str) -> None:
    """Remove one edge id from an adjacency bucket if present."""
    if key is None:
        return
    bucket = index.get(key)
    if not bucket:
        return
    try:
        bucket.remove(edge_id)
    except ValueError:
        return
    if not bucket:
        index.pop(key, None)
