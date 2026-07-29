"""Internal-consistency validation.

``validate_internal_consistency(g)`` asserts the invariants that the mutation
gateway and the derive layer are supposed to maintain. It is a debugging and
test aid. It never runs on a hot path.

The core holds two store models during the refactor. The record store keeps one
record per entity and one record per edge, and addresses a row and a column by
position. The slot store keeps slot-addressed member lists. The two models need
different checks, so this module keeps one check list per model and picks the
list from the store the graph holds.

Every check takes the graph and a list, and appends one message per problem it
finds. A check never raises. ``validate_internal_consistency`` raises once at
the end when the caller asks for strict mode.
"""

from __future__ import annotations

from . import _derive as D, _identity as I

RECORD_STORE = 'records'
SLOT_STORE = 'slots'


def detect_store_kind(g) -> str:
    """Return which store model backs a graph.

    The slot store lives at ``g._store``. A graph without that attribute still
    runs on the record store.
    """
    return SLOT_STORE if getattr(g, '_store', None) is not None else RECORD_STORE


# ---------------------------------------------------------------------------
# Record-store checks
# ---------------------------------------------------------------------------


def _check_entity_rows(g, problems) -> None:
    """Entities and rows form a bijection, and the row block is contiguous."""
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
    if seen_rows and set(seen_rows) != set(range(len(g._entities))):
        problems.append(f'entity row indices not contiguous 0..{len(g._entities) - 1}')
    for r, ekey in g._row_to_entity.items():
        if ekey not in g._entities or g._entities[ekey].row_idx != r:
            problems.append(f'_row_to_entity has stale entry [{r}]={ekey!r}')
    # ``_vid_to_ekeys`` is intentionally left empty on a flat graph, because
    # ``resolve_ekey`` returns the placeholder key directly there.
    if g._aspects != ('_',):
        for vid, ekeys in vid_to_ekeys.items():
            if set(g._vid_to_ekeys.get(vid, ())) != set(ekeys):
                problems.append(f'_vid_to_ekeys[{vid!r}] mismatch')


def _check_edge_columns(g, problems) -> None:
    """Edges and columns form a bijection, and the column block is contiguous."""
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


def _check_adjacency_indexes(g, problems) -> None:
    """Every adjacency entry points at a real edge whose endpoint equals the key.

    The incremental form and the rebuilt form differ benignly for a hyperedge,
    which is never queried through these maps.
    """
    if not getattr(g, '_edge_indexes_built', True):
        return
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


def _check_slice_membership(g, problems) -> None:
    """A slice holds bare node ids and known edge ids."""
    for sid, srec in g._slices.items():
        for v in srec['vertices']:
            if not isinstance(v, str):
                problems.append(f'slice {sid!r} has non-string vertex {v!r} (must be bare vid)')
                break
        for eid in srec['edges']:
            if eid not in g._edges:
                problems.append(f'slice {sid!r} references unknown edge {eid!r}')
                break


def _check_coefficients(g, problems) -> None:
    """A coefficient-bearing column matches the coefficients its record holds."""
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


def _check_matrix_bounds(g, problems) -> None:
    """The matrix is large enough for every row and every column in use."""
    nrows, ncols = g._matrix.shape
    if len(g._entities) > nrows:
        problems.append(f'matrix rows {nrows} < #entities {len(g._entities)}')
    cols = [rec.col_idx for rec in g._edges.values() if rec.col_idx >= 0]
    if cols and max(cols) >= ncols:
        problems.append(f'matrix cols {ncols} <= max col_idx {max(cols)}')


RECORD_CHECKS = (
    _check_entity_rows,
    _check_edge_columns,
    _check_adjacency_indexes,
    _check_slice_membership,
    _check_coefficients,
    _check_matrix_bounds,
)

# The slot store arrives with the new core. Its checks register here.
SLOT_CHECKS: tuple = ()

_CHECKS_BY_STORE = {
    RECORD_STORE: RECORD_CHECKS,
    SLOT_STORE: SLOT_CHECKS,
}


def checks_for(store_kind: str) -> tuple:
    """Return the check list for one store model."""
    try:
        return _CHECKS_BY_STORE[store_kind]
    except KeyError:
        raise ValueError(
            f'Unknown store kind {store_kind!r}. Known: {sorted(_CHECKS_BY_STORE)}'
        ) from None


def validate_internal_consistency(g, *, strict: bool = True) -> list[str]:
    """Check the invariants of a graph and return one message per problem.

    The function picks the check list from the store the graph holds. It returns
    an empty list for a consistent graph. It raises ``AssertionError`` on a
    problem when ``strict`` is set.
    """
    store_kind = detect_store_kind(g)
    checks = checks_for(store_kind)
    if not checks:
        raise NotImplementedError(
            f'No invariant checks are registered for the {store_kind!r} store.'
        )

    problems: list[str] = []
    for check in checks:
        check(g, problems)

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
