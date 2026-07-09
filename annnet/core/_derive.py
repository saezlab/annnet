"""Derived-state machinery."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Incidence matrix capacity + cache invalidation
# ---------------------------------------------------------------------------


def grow_rows_to(g, target: int) -> None:
    """Grow the logical incidence row capacity to accommodate ``target`` rows."""
    rows, cols = g._matrix_shape
    if target > rows:
        g._matrix_shape = (target, cols)
        g._matrix_dirty = True


def grow_cols_to(g, target: int) -> None:
    """Grow the logical incidence column capacity to accommodate ``target`` cols."""
    rows, cols = g._matrix_shape
    if target > cols:
        g._matrix_shape = (rows, target)
        g._matrix_dirty = True


def set_matrix_shape(g, shape) -> None:
    """Set the logical incidence shape and mark the matrix cache dirty."""
    g._matrix_shape = (int(shape[0]), int(shape[1]))
    g._matrix_dirty = True


def rebuild_matrix(g):
    """Materialize the incidence matrix (CSR) from canonical edge records.

    Records are the complete source of truth: an edge carries its explicit
    ``coeffs`` column when set (stoichiometry / resolved flexible direction),
    otherwise the column is derived from ``src``/``tgt``/``weight``/``directed``
    (src endpoints get ``+w``; tgt endpoints get ``-w`` when directed, else ``+w``;
    a tgt sharing a src's row overwrites it, matching the canonical constructor).
    """
    shape = g._matrix_shape
    entity_row = g._entity_row
    rows: list = []
    cols: list = []
    data: list = []
    for rec in g._edges.values():
        col = rec.col_idx
        if col < 0:
            continue
        coeffs = rec.coeffs
        if coeffs is not None:
            col_entries = coeffs
        else:
            w = rec.weight if rec.weight is not None else 1.0
            tv = -w if rec.directed else w
            src, tgt = rec.src, rec.tgt
            col_entries = {}
            if isinstance(src, frozenset):
                for n in src:
                    col_entries[n] = w
            elif src is not None:
                col_entries[src] = w
            if isinstance(tgt, frozenset):
                for n in tgt:
                    col_entries[n] = tv
            elif tgt is not None:
                col_entries[tgt] = tv
        for node, v in col_entries.items():
            if v == 0:
                continue
            try:
                r = entity_row(node)
            except (KeyError, ValueError, TypeError):
                continue
            rows.append(r)
            cols.append(col)
            data.append(v)
    if data:
        coo = sp.coo_matrix(
            (
                np.asarray(data, dtype=np.float32),
                (np.asarray(rows, dtype=np.intp), np.asarray(cols, dtype=np.intp)),
            ),
            shape=shape,
        )
        return coo.tocsr()
    return sp.csr_matrix(shape, dtype=np.float32)


def invalidate_sparse_caches(g, formats=None) -> None:
    """Invalidate all derived sparse cache views behind one internal hook."""
    formats = ('csr', 'csc', 'adjacency') if formats is None else tuple(formats)
    if 'csr' in formats:
        g._csr_cache = None
    cache_manager = getattr(g, '_cache_manager', None)
    if cache_manager is not None:
        cache_manager.invalidate(list(formats))


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
