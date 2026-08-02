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

from . import _derive as D, _identity as I, _structure as S

RECORD_STORE = 'records'
SLOT_STORE = 'slots'

# A materialized cell and the coefficient it comes from may differ by the rounding
# of the float32 matrix storage.
_TOLERANCE = 1e-4


def _materialized_matrix(g, problems):
    """Return the incidence matrix, or None when it cannot be built.

    A check must never raise, because the caller wants the whole problem list
    and not the first exception. A store that cannot even produce its matrix is
    itself a problem, so it joins the list.
    """
    try:
        return g._matrix
    except Exception as error:  # noqa: BLE001 - any failure here is a reportable problem
        message = f'the incidence matrix cannot be materialized: {error}'
        if message not in problems:
            problems.append(message)
        return None


def detect_store_kind(g) -> str:
    """Return which store model backs a graph.

    A slot store answers to ``member_start``, either as the graph itself or through
    ``g._store``. A graph without one runs on the record store.
    """
    if hasattr(g, 'member_start'):
        return SLOT_STORE
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


def _check_member_liveness(g, problems) -> None:
    """Every member of every edge is a live entity.

    A member that names no entity leaves a column that no row can carry, so the
    matrix silently drops it.
    """
    for eid in g._edges:
        for member_key in S.edge_members(g, eid):
            if member_key not in g._entities:
                problems.append(f'edge {eid!r} has member {member_key[0]!r} that is not an entity')


def _check_edge_entity_linkage(g, problems) -> None:
    """An edge-entity holds an edge and an entity under one id."""
    for ekey, rec in g._entities.items():
        if rec.kind == 'edge_entity' and ekey[0] not in g._edges:
            problems.append(f'entity {ekey[0]!r} is marked as an edge but no edge carries that id')
    for eid, rec in g._edges.items():
        if rec.etype not in ('vertex_edge', 'edge_placeholder'):
            continue
        matches = [ekey for ekey in g._entities if ekey[0] == eid]
        if not matches:
            problems.append(f'edge {eid!r} is an edge-entity but no entity carries that id')
            continue
        for ekey in matches:
            if g._entities[ekey].kind != 'edge_entity':
                problems.append(
                    f'edge-entity {eid!r} has entity kind {g._entities[ekey].kind!r}, '
                    "expected 'edge_entity'"
                )


def _check_matrix_agrees_with_members(g, problems) -> None:
    """The materialized matrix holds exactly the member lists the store implies.

    This is the rule that a directly assigned matrix breaks. It also covers the
    sign rule, because the member list of an edge with no explicit coefficient
    carries the signs its kind and directedness imply.
    """
    raw = _materialized_matrix(g, problems)
    if raw is None:
        return
    matrix = raw.tocsc()
    n_rows, n_cols = matrix.shape
    seen_cells: set[tuple[int, int]] = set()

    for eid, rec in g._edges.items():
        col = rec.col_idx
        if col < 0:
            continue
        if col >= n_cols:
            problems.append(f'edge {eid!r} has column {col} outside the matrix of {n_cols} columns')
            continue
        expected: dict[int, float] = {}
        for member_key, coefficient in S.edge_members(g, eid).items():
            record = g._entities.get(member_key)
            if record is None or coefficient == 0:
                continue
            expected[record.row_idx] = coefficient
        block = matrix[:, [col]].tocoo()
        found = {
            int(r): float(v) for r, v in zip(block.row, block.data, strict=False) if float(v) != 0.0
        }
        seen_cells.update((row, col) for row in found)
        for row in set(expected) | set(found):
            want = expected.get(row, 0.0)
            have = found.get(row, 0.0)
            if abs(want - have) > _TOLERANCE * max(1.0, abs(want)):
                ekey = g._row_to_entity.get(row)
                name = repr(ekey) if ekey is not None else f'row {row}, which holds no entity'
                problems.append(
                    f'edge {eid!r} cell for {name}: store says {want}, matrix says {have}'
                )

    live_rows = {rec.row_idx for rec in g._entities.values()}
    block = raw.tocoo()
    for row, col, value in zip(block.row, block.col, block.data, strict=False):
        if float(value) == 0.0 or (int(row), int(col)) in seen_cells:
            continue
        if int(row) not in live_rows:
            problems.append(f'matrix holds {float(value)} at row {int(row)}, which holds no entity')
        elif int(col) not in g._col_to_edge:
            problems.append(
                f'matrix holds {float(value)} at column {int(col)}, which holds no edge'
            )
    if n_rows < len(live_rows):
        problems.append(f'matrix has {n_rows} rows but the store holds {len(live_rows)} entities')


def _check_table_levels(g, problems) -> None:
    """The node table stays node-level and the edge table stays edge-level."""
    entity_ids = {ekey[0] for ekey in g._entities}
    for label, table, key_column, known in (
        ('obs', g.vertex_attributes, 'vertex_id', entity_ids),
        ('var', g.edge_attributes, 'edge_id', set(g._edges)),
    ):
        if table is None:
            continue
        try:
            rows = table.to_dicts() if hasattr(table, 'to_dicts') else list(table.rows(named=True))
        except (AttributeError, TypeError):  # pragma: no cover - unknown backend
            continue
        for row in rows:
            key = row.get(key_column)
            if key is not None and key not in known:
                problems.append(f'{label} holds a row for {key!r}, which the store does not hold')


def _check_slice_membership(g, problems) -> None:
    """A slice holds bare node ids and known edge ids, and both must be live."""
    node_ids = {ekey[0] for ekey in g._entities}
    for sid, srec in g._slices.items():
        for v in srec['vertices']:
            if not isinstance(v, str):
                problems.append(f'slice {sid!r} has non-string vertex {v!r} (must be bare vid)')
                break
            if v not in node_ids:
                problems.append(f'slice {sid!r} references unknown vertex {v!r}')
                break
        for eid in srec['edges']:
            if eid not in g._edges:
                problems.append(f'slice {sid!r} references unknown edge {eid!r}')
                break


def _check_coefficients(g, problems) -> None:
    """A coefficient-bearing column matches the coefficients its record holds."""
    matrix = _materialized_matrix(g, problems)
    if matrix is None:
        return
    n_rows, n_cols = matrix.shape
    for eid, rec in g._edges.items():
        if rec.coeffs is None or rec.col_idx < 0 or rec.col_idx >= n_cols:
            continue
        for node, val in rec.coeffs.items():
            try:
                r = I.entity_row(g, node)
            except (KeyError, ValueError, TypeError):
                problems.append(f'edge {eid!r} coeff node {node!r} not resolvable')
                continue
            if r >= n_rows:
                continue
            actual = float(matrix[r, rec.col_idx])
            if abs(actual - float(val)) > 1e-4 * max(1.0, abs(float(val))):
                problems.append(f'edge {eid!r} coeff[{node!r}]={val} != matrix {actual}')


def _check_matrix_bounds(g, problems) -> None:
    """The matrix is large enough for every row and every column in use."""
    matrix = _materialized_matrix(g, problems)
    if matrix is None:
        return
    nrows, ncols = matrix.shape
    if len(g._entities) > nrows:
        problems.append(f'matrix rows {nrows} < #entities {len(g._entities)}')
    cols = [rec.col_idx for rec in g._edges.values() if rec.col_idx >= 0]
    if cols and max(cols) >= ncols:
        problems.append(f'matrix cols {ncols} <= max col_idx {max(cols)}')


# ---------------------------------------------------------------------------
# Slot-store checks
# ---------------------------------------------------------------------------


def _slot_of(g):
    """Return the slot store of a graph, or the graph itself when it is one."""
    return getattr(g, '_store', None) or g


def _check_slot_bijections(store, problems) -> None:
    """Identity and slot agree in both directions, for entities and for edges."""
    for label, forward, backward in (
        ('entity', store._entity_slot, store._entity_key),
        ('edge', store._edge_slot, store._edge_id),
    ):
        for identity, slot in forward.items():
            if not 0 <= slot < len(backward):
                problems.append(f'{label} {identity!r} claims slot {slot}, which does not exist')
            elif backward[slot] != identity:
                problems.append(
                    f'{label} {identity!r} claims slot {slot}, which holds {backward[slot]!r}'
                )
        for slot, identity in enumerate(backward):
            if identity is None:
                continue
            if forward.get(identity) != slot:
                problems.append(
                    f'slot {slot} holds {label} {identity!r}, which claims another slot'
                )


def _check_freelists(store, problems) -> None:
    """A free slot holds no identity, and it appears on its freelist exactly once."""
    for label, free, backward in (
        ('entity', store.entity_free, store._entity_key),
        ('edge', store.edge_free, store._edge_id),
    ):
        if len(set(free)) != len(free):
            problems.append(f'the {label} freelist holds a slot more than once')
        for slot in free:
            if not 0 <= slot < len(backward):
                problems.append(f'the {label} freelist holds slot {slot}, which does not exist')
            elif backward[slot] is not None:
                problems.append(
                    f'{label} slot {slot} is on the freelist but still holds {backward[slot]!r}'
                )


def _check_member_segments(store, problems) -> None:
    """Every member segment lies inside the pools, and no two live segments overlap."""
    used = store._member_used
    seen: dict[int, int] = {}
    for slot, edge_id in store.live_edges():
        start = int(store.member_start[slot])
        length = int(store.member_len[slot])
        if length < 0 or start < 0 or start + length > used:
            problems.append(
                f'edge {edge_id!r} has member segment [{start}, {start + length}) '
                f'outside the pool of {used}'
            )
            continue
        for offset in range(start, start + length):
            other = seen.get(offset)
            if other is not None:
                problems.append(f'edge {edge_id!r} shares member entry {offset} with slot {other}')
            seen[offset] = slot
    for slot in store.edge_free:
        if int(store.member_len[slot]) != 0:
            problems.append(f'free edge slot {slot} still holds member entries')


def _check_slot_member_liveness(store, problems) -> None:
    """Every member entry names a live entity slot."""
    for slot, edge_id in store.live_edges():
        for entity_slot in store.members(slot).entities:
            if store.entity_key(int(entity_slot)) is None:
                problems.append(
                    f'edge {edge_id!r} has a member on slot {int(entity_slot)}, which holds no entity'
                )


def _check_member_counts(store, problems) -> None:
    """The member entry count of an edge matches its kind.

    This is what keeps a self-loop apart from a boundary edge. A binary edge holds
    two entries, one per role. A boundary edge holds one. A self-loop holds two on
    one entity slot.
    """
    from . import _store as S_

    for slot, edge_id in store.live_edges():
        kind = int(store.edge_kind[slot])
        count = store.member_count(slot)
        if kind == S_.PLACEHOLDER:
            if count != 0:
                problems.append(f'placeholder edge {edge_id!r} holds {count} member entries')
            continue
        if kind in (S_.BINARY, S_.NODE_EDGE):
            if count == 2:
                continue
            if count == 1 and bool(store.edge_explicit[slot]):
                # A one-sided binary edge is a boundary edge, and a boundary edge
                # states its own coefficient.
                continue
            problems.append(
                f'binary edge {edge_id!r} holds {count} member entries. A binary edge holds '
                'one entry per role, so two, and a self-loop holds two on one entity slot. '
                'One entry means a lost role, unless the edge declares its own coefficient '
                'and is therefore a boundary edge.'
            )


def _check_edge_entity_slots(store, problems) -> None:
    """An edge-entity holds an edge slot and an entity slot under one id."""
    from . import _store as S_

    for slot, edge_id in store.live_edges():
        if int(store.edge_kind[slot]) not in (S_.NODE_EDGE, S_.PLACEHOLDER):
            continue
        matches = [key for _slot, key in store.live_entities() if key[0] == edge_id]
        if not matches:
            problems.append(f'edge {edge_id!r} is an edge-entity but no entity carries that id')
            continue
        for key in matches:
            entity_slot = store.entity_slot(key)
            if int(store.entity_kind[entity_slot]) != S_.EDGE_ENTITY:
                problems.append(f'edge-entity {edge_id!r} has an entity that is not marked as one')


def _check_incidence_index(store, problems) -> None:
    """The entity-to-edge index matches the member lists it is derived from.

    The index carries the sides an entity takes as well as the edges it is in,
    and a traversal reads the sides from here and never from the member list. So
    the sides are checked too, and nothing else would catch a wrong one.
    """
    from . import _store as ST

    expected: dict[int, dict[int, tuple]] = {slot: {} for slot, _key in store.live_entities()}
    for edge_slot, _edge_id in store.live_edges():
        members = store.members(edge_slot)
        entities = [int(entity_slot) for entity_slot in members.entities]
        for entity_slot, role in zip(entities, members.roles, strict=False):
            sides = expected.get(entity_slot)
            if sides is not None:
                side = ST.ON_TARGET if int(role) == ST.TARGET else ST.ON_SOURCE
                held = sides.get(edge_slot)
                sides[edge_slot] = (side if held is None else held[0] | side, None)
        if len(entities) == 2:
            first, second = entities
            for mine, peer in ((first, second), (second, first)):
                sides = expected.get(mine)
                if sides is not None and edge_slot in sides:
                    sides[edge_slot] = (sides[edge_slot][0], peer)
    for entity_slot, sides in expected.items():
        held = store._entity_edges.get(entity_slot, {})
        if held != sides:
            problems.append(
                f'the edge index of entity slot {entity_slot} says {sorted(held.items())}, '
                f'the member lists say {sorted(sides.items())}'
            )


def _check_matrix_matches_member_lists(store, problems) -> None:
    """A materialized matrix holds exactly what the member lists imply."""
    from . import _matrices as MX

    try:
        view = MX.incidence(store)
    except Exception as error:  # noqa: BLE001 - a store that cannot materialize is a problem
        problems.append(f'the incidence matrix cannot be materialized: {error}')
        return
    matrix = view.matrix.tocsc()
    for column, edge_id in enumerate(view.edge_of_column):
        slot = store.edge_slot(edge_id)
        members = store.members(slot)
        expected: dict[int, float] = {}
        for entity_slot, coefficient in zip(members.entities, members.coefficients, strict=False):
            row = view.row_of_entity[int(entity_slot)]
            expected[row] = expected.get(row, 0.0) + float(coefficient)
        expected = {row: value for row, value in expected.items() if value != 0.0}
        block = matrix[:, [column]].tocoo()
        found = {
            int(r): float(v) for r, v in zip(block.row, block.data, strict=False) if float(v) != 0.0
        }
        for row in set(expected) | set(found):
            want, have = expected.get(row, 0.0), found.get(row, 0.0)
            if abs(want - have) > _TOLERANCE * max(1.0, abs(want)):
                problems.append(
                    f'edge {edge_id!r} cell at row {row}: store says {want}, matrix says {have}'
                )


def _check_clock(store, problems) -> None:
    """The clock and the append log agree with each other."""
    if store.structure_version < 0:
        problems.append('the structural clock is negative')
    if store.append_log_from_version > store.structure_version:
        problems.append('the append log starts after the current clock value')
    if len(store.append_log) > store.structure_version - store.append_log_from_version:
        problems.append('the append log holds more entries than the clock accounts for')


SLOT_CHECKS_IMPL = (
    _check_slot_bijections,
    _check_freelists,
    _check_member_segments,
    _check_slot_member_liveness,
    _check_member_counts,
    _check_edge_entity_slots,
    _check_incidence_index,
    _check_matrix_matches_member_lists,
    _check_clock,
)


def _slot_check(check):
    """Adapt a store-level check so the dispatcher can call it with a graph."""

    def run(g, problems):
        check(_slot_of(g), problems)

    run.__name__ = check.__name__
    run.__doc__ = check.__doc__
    return run


RECORD_CHECKS = (
    _check_entity_rows,
    _check_edge_columns,
    _check_adjacency_indexes,
    _check_member_liveness,
    _check_edge_entity_linkage,
    _check_matrix_agrees_with_members,
    _check_table_levels,
    _check_slice_membership,
    _check_coefficients,
    _check_matrix_bounds,
)

SLOT_CHECKS: tuple = tuple(_slot_check(check) for check in SLOT_CHECKS_IMPL)

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
