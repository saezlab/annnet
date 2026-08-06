"""Internal-consistency validation.

``validate_internal_consistency(g)`` asserts the invariants that the mutation
gateway and the derive layer are supposed to maintain. It is a debugging and
test aid. It never runs on a hot path.

The canonical store holds the topology, and the graph around it holds the
slices, the attribute tables and any materialized matrix. So this module keeps
two check lists: the rules of the store alone, and the rules that tie the graph
to it. A caller that hands a bare store gets the first list.

Every check takes the graph or the store and a list, and appends one message per
problem it finds. A check never raises. ``validate_internal_consistency`` raises
once at the end when the caller asks for strict mode.
"""

from __future__ import annotations

from . import _structure as S

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


def _is_store(g) -> bool:
    """Return True when the argument is a canonical store rather than a graph."""
    return hasattr(g, 'member_start')


# ---------------------------------------------------------------------------
# Graph-level checks
# ---------------------------------------------------------------------------
# The canonical store holds the topology alone. Slices, the attribute tables and
# a materialized matrix belong to the graph around it, so the rules that tie
# those to the topology are checked here, through the query facade.
#
# One rule that used to sit here is gone rather than passing: the node table
# stays node-level and the edge table stays edge-level. The graph derives both
# tables from columns indexed by slot, so a row that names an element the store
# does not hold cannot be written, and there is nothing left to check.


def _check_slice_membership(g, problems) -> None:
    """A slice holds bare node ids and known edge ids, and both must be live."""
    node_ids = {ref.id for ref in S.iter_entities(g)}
    edge_ids = {ref.id for ref in S.iter_edges(g, include_placeholders=True)}
    for sid, srec in g._slices.items():
        for v in srec['vertices']:
            if not isinstance(v, str):
                problems.append(f'slice {sid!r} has non-string vertex {v!r} (must be bare vid)')
                break
            if v not in node_ids:
                problems.append(f'slice {sid!r} references unknown vertex {v!r}')
                break
        for eid in srec['edges']:
            if eid not in edge_ids:
                problems.append(f'slice {sid!r} references unknown edge {eid!r}')
                break


def _check_materialized_matrix(g, problems) -> None:
    """The matrix the graph answers with is the one its member lists imply.

    The graph derives its matrix rather than holding one it was handed, so this
    compares two derivations of the same thing: the one the matrix builder makes
    in a single gather over the store arrays, and the one the query facade reads
    member list by member list. A cached matrix the store has outgrown, and a
    builder that places an entry wrongly, both show up here.
    """
    raw = _materialized_matrix(g, problems)
    if raw is None:
        return
    store = _slot_of(g)
    matrix = raw.tocsc()
    n_rows, n_cols = matrix.shape
    seen_cells: set[tuple[int, int]] = set()
    rows_of_entity = {key: row for row, key in enumerate(S.entity_keys(store))}

    for column, edge_id in enumerate(S.edge_ids(store)):
        if column >= n_cols:
            problems.append(
                f'edge {edge_id!r} has column {column} outside the matrix of {n_cols} columns'
            )
            continue
        expected: dict[int, float] = {}
        for member_key, coefficient in S.edge_members(store, edge_id).items():
            row = rows_of_entity.get(member_key)
            if row is None or coefficient == 0:
                continue
            expected[row] = coefficient
        block = matrix[:, [column]].tocoo()
        found = {
            int(r): float(v) for r, v in zip(block.row, block.data, strict=False) if float(v) != 0.0
        }
        seen_cells.update((row, column) for row in found)
        for row in set(expected) | set(found):
            want = expected.get(row, 0.0)
            have = found.get(row, 0.0)
            if abs(want - have) > _TOLERANCE * max(1.0, abs(want)):
                problems.append(
                    f'edge {edge_id!r} cell at row {row}: store says {want}, matrix says {have}'
                )

    live_rows = set(rows_of_entity.values())
    live_columns = set(range(len(S.edge_ids(store))))
    block = raw.tocoo()
    for row, column, value in zip(block.row, block.col, block.data, strict=False):
        if float(value) == 0.0 or (int(row), int(column)) in seen_cells:
            continue
        if int(row) not in live_rows:
            problems.append(f'matrix holds {float(value)} at row {int(row)}, which holds no entity')
        elif int(column) not in live_columns:
            problems.append(
                f'matrix holds {float(value)} at column {int(column)}, which holds no edge'
            )
    if n_rows < len(live_rows):
        problems.append(f'matrix has {n_rows} rows but the store holds {len(live_rows)} entities')


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


def _check_bare_id_index(store, problems) -> None:
    """The bare-id index names exactly the slots the entities hold.

    A multilayer graph resolves a bare id through this index, so a slot missing
    from it makes the id resolve to nothing and a stale slot makes it resolve to
    an entity the store no longer holds. Neither raises on its own.
    """
    if store.aspects == ('_',):
        if store._id_slots:
            problems.append('a flat store keeps a bare-id index, which it never reads')
        return
    expected: dict = {}
    for slot, key in store.live_entities():
        expected.setdefault(key[0], []).append(slot)
    for entity_id, slots in store._id_slots.items():
        held = expected.get(entity_id)
        if held is None:
            problems.append(f'the bare-id index names {entity_id!r}, which no entity carries')
        elif sorted(slots) != held:
            problems.append(
                f'the bare-id index gives {entity_id!r} slots {sorted(slots)}, '
                f'but its entities hold {held}'
            )
    for entity_id in expected.keys() - store._id_slots.keys():
        problems.append(f'the bare-id index is missing {entity_id!r}')


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

    edge_ids = set(store.live_edge_ids())
    for slot, key in store.live_entities():
        if int(store.entity_kind[slot]) == S_.EDGE_ENTITY and key[0] not in edge_ids:
            problems.append(f'entity {key[0]!r} is marked as an edge but no edge carries that id')


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
            row = view.row_of_entity[store.entity_key(int(entity_slot))]
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
    _check_bare_id_index,
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


SLOT_CHECKS: tuple = tuple(_slot_check(check) for check in SLOT_CHECKS_IMPL)

# The rules that tie the graph around the store to the topology in it. A caller
# may hand a bare store instead of a graph, which holds none of what these read,
# so they run only for a graph.
GRAPH_CHECKS = (
    _check_slice_membership,
    _check_materialized_matrix,
)


def validate_internal_consistency(g, *, strict: bool = True) -> list[str]:
    """Check the invariants of a graph and return one message per problem.

    The argument is a graph or the canonical store of one. A store is checked
    against the rules it holds by itself, and a graph against those and the rules
    that tie its slices, its tables and its matrix to the store. The function
    returns an empty list for a consistent graph, and raises ``AssertionError``
    on a problem when ``strict`` is set.
    """
    checks = SLOT_CHECKS if _is_store(g) else SLOT_CHECKS + GRAPH_CHECKS

    problems: list[str] = []
    for check in checks:
        check(g, problems)

    if strict and problems:
        raise AssertionError('internal consistency violated:\n  ' + '\n  '.join(problems))
    return problems
