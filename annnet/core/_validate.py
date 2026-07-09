"""Internal-consistency validation.

``validate_internal_consistency(g)`` asserts the invariants that the mutation gateway and
derive layer are supposed to maintain: row/col index bijections, derived indexes matching
the canonical records, the bare-vid slice contract, and matrix bounds. It is a debugging /
test aid — never part of a hot path.
"""

from __future__ import annotations

from . import _derive as D, _identity as I


def validate_internal_consistency(g, *, strict: bool = True) -> list[str]:
    """Check graph invariants; return a list of problems (and raise if ``strict``).

    Asserts: entity row bijection with ``_row_to_entity``; ``_vid_to_ekeys`` matches the
    entity keys; edge column bijection with ``_col_to_edge``; adjacency indexes match the
    records (when built); slice ``vertices`` are bare strings; matrix shape covers all
    rows/cols. Returns ``[]`` for a consistent graph.
    """
    problems: list[str] = []

    # --- entities <-> rows -----------------------------------------------------
    seen_rows: dict[int, tuple] = {}
    vid_to_ekeys: dict[str, list] = {}
    for ekey, rec in g._entities.items():
        r = rec.row_idx
        if r in seen_rows:
            problems.append(f'duplicate row_idx {r}: {ekey!r} and {seen_rows[r]!r}')
        seen_rows[r] = ekey
        if g._row_to_entity.get(r) != ekey:
            problems.append(f'_row_to_entity[{r}] != {ekey!r} (got {g._row_to_entity.get(r)!r})')
        if isinstance(ekey, tuple) and len(ekey) == 2 and isinstance(ekey[0], str):
            vid_to_ekeys.setdefault(ekey[0], []).append(ekey)
    # row indices should be a contiguous 0..n-1 block
    if seen_rows and set(seen_rows) != set(range(len(g._entities))):
        problems.append(f'entity row indices not contiguous 0..{len(g._entities) - 1}')
    # _row_to_entity must not carry stale rows
    for r, ekey in g._row_to_entity.items():
        if ekey not in g._entities or g._entities[ekey].row_idx != r:
            problems.append(f'_row_to_entity has stale entry [{r}]={ekey!r}')
    # _vid_to_ekeys must match — but it is intentionally left empty on flat
    # graphs (resolve_ekey returns the placeholder ekey directly there).
    if g._aspects != ('_',):
        for vid, ekeys in vid_to_ekeys.items():
            if set(g._vid_to_ekeys.get(vid, ())) != set(ekeys):
                problems.append(f'_vid_to_ekeys[{vid!r}] mismatch')

    # --- edges <-> columns -----------------------------------------------------
    seen_cols: dict[int, str] = {}
    for eid, rec in g._edges.items():
        c = rec.col_idx
        if c < 0:
            continue
        if c in seen_cols:
            problems.append(f'duplicate col_idx {c}: {eid!r} and {seen_cols[c]!r}')
        seen_cols[c] = eid
        if g._col_to_edge.get(c) != eid:
            problems.append(f'_col_to_edge[{c}] != {eid!r} (got {g._col_to_edge.get(c)!r})')
    if seen_cols and set(seen_cols) != set(range(len(seen_cols))):
        problems.append(f'edge column indices not contiguous 0..{len(seen_cols) - 1}')
    for c, eid in g._col_to_edge.items():
        rec = g._edges.get(eid)
        if rec is None or rec.col_idx != c:
            problems.append(f'_col_to_edge has stale entry [{c}]={eid!r}')

    # --- adjacency indexes carry no stale / mis-keyed entries ------------------
    # (the incremental and rebuilt forms can differ benignly for hyperedges, which are
    # never queried via these maps; what must hold is that every entry points at a real
    # edge whose endpoint actually equals the key.)
    if getattr(g, '_edge_indexes_built', True):
        for label, index, field in (
            ('_src_to_edges', g._src_to_edges, 'src'),
            ('_tgt_to_edges', g._tgt_to_edges, 'tgt'),
        ):
            for key, eids in index.items():
                for eid in eids:
                    rec = g._edges.get(eid)
                    if rec is None or getattr(rec, field) != key:
                        problems.append(f'{label}[{key!r}] has stale/mis-keyed edge {eid!r}')
        for key, eids in g._pair_to_edges.items():
            for eid in eids:
                rec = g._edges.get(eid)
                if rec is None or (rec.src, rec.tgt) != key:
                    problems.append(f'_pair_to_edges[{key!r}] has stale/mis-keyed edge {eid!r}')

    # --- slice contract: vertices are bare ids --------------------------------
    for sid, srec in g._slices.items():
        for v in srec['vertices']:
            if not isinstance(v, str):
                problems.append(f'slice {sid!r} has non-string vertex {v!r} (must be bare vid)')
                break
        for eid in srec['edges']:
            if eid not in g._edges:
                problems.append(f'slice {sid!r} references unknown edge {eid!r}')
                break

    # --- numerical: coeff-bearing columns match their records ------------------
    # (possible now that records carry stoich/explicit coefficients; the matrix is a
    # warm cache of what the records imply.)
    for eid, rec in g._edges.items():
        if rec.coeffs is None or rec.col_idx < 0:
            continue
        for node, val in rec.coeffs.items():
            try:
                r = I.entity_row(g, node)
            except (KeyError, ValueError, TypeError):
                problems.append(f'edge {eid!r} coeff node {node!r} not resolvable')
                continue
            actual = float(g._matrix[r, rec.col_idx])
            if abs(actual - float(val)) > 1e-4 * max(1.0, abs(float(val))):
                problems.append(f'edge {eid!r} coeff[{node!r}]={val} != matrix {actual}')

    # --- matrix bounds ---------------------------------------------------------
    nrows, ncols = g._matrix.shape
    if len(g._entities) > nrows:
        problems.append(f'matrix rows {nrows} < #entities {len(g._entities)}')
    if seen_cols and max(seen_cols) >= ncols:
        problems.append(f'matrix cols {ncols} <= max col_idx {max(seen_cols)}')

    if strict and problems:
        raise AssertionError('internal consistency violated:\n  ' + '\n  '.join(problems))
    return problems


def rebuild_and_compare(g) -> list[str]:
    """Diagnostic: rebuild all derived indexes from records and report what changed."""
    before = (
        dict(g._row_to_entity),
        dict(g._col_to_edge),
        dict(g._src_to_edges),
        dict(g._tgt_to_edges),
    )
    D.rebuild_entity_indexes(g)
    D.rebuild_col_index(g)
    D.rebuild_edge_indexes(g)
    after = (g._row_to_entity, g._col_to_edge, g._src_to_edges, g._tgt_to_edges)
    names = ('_row_to_entity', '_col_to_edge', '_src_to_edges', '_tgt_to_edges')
    return [n for n, b, a in zip(names, before, after, strict=True) if b != a]
