"""The two-core equivalence harness.

The core refactor replaces the store behind the graph object. The replacement is
correct when the new store answers every question the old store answers, with
the same result. This harness asks those questions of two graphs and reports
every difference.

The harness compares content addressed by identity. It never compares a row
number or a column number, because a position is internal to one materialized
matrix and carries no meaning across two stores. A matrix comparison therefore
looks up the row of a node id and the column of an edge id in each graph, and
compares the cell.

Use ``compare`` to get the list of differences. Use ``assert_equivalent`` in a
test.
"""

from __future__ import annotations

import math


def _sorted_repr(values):
    """Return a stable sorted list of the string forms of ``values``."""
    return sorted(repr(v) for v in values)


def _close(left, right, *, tol=1e-6) -> bool:
    """Return True when two numbers agree within a small tolerance."""
    if left is None or right is None:
        return left is right
    return math.isclose(float(left), float(right), rel_tol=tol, abs_tol=tol)


# ---------------------------------------------------------------------------
# Structural projections
# ---------------------------------------------------------------------------


def _node_ids(graph) -> list[str]:
    return list(graph.vertices())


def _supra_node_keys(graph) -> list[tuple]:
    return list(graph.supra_vertices())


def _edge_ids(graph) -> list[str]:
    return list(graph.edges())


def _edge_view_signature(view) -> tuple:
    """Reduce an edge view to a comparable, position-free signature."""
    return (
        view.edge_id,
        view.kind,
        frozenset(view.source),
        frozenset(view.target),
        frozenset(view.members),
        round(float(view.weight), 6),
        bool(view.directed),
    )


def _incident_signature(graph, node_id, direction) -> set:
    """Return the edge signatures incident to one node, without the columns."""
    return {_edge_view_signature(view) for _col, view in graph.incident_edges(node_id, direction)}


def _matrix_cells(graph) -> dict[tuple, float]:
    """Return the incidence matrix as a mapping from id pairs to values.

    The key is ``(node_key, edge_id)``. The node key is the supra-node key, so
    the projection stays correct for a multilayer graph, where one node id
    covers more than one row.
    """
    matrix = graph.X().tocsr()
    row_of_key = {}
    for key in _supra_node_keys(graph):
        try:
            row_of_key[key] = graph.idx.entity_to_row(key)
        except (KeyError, ValueError, TypeError):
            continue
    cells: dict[tuple, float] = {}
    for edge_id in _edge_ids(graph):
        try:
            col = graph.idx.edge_to_col(edge_id)
        except KeyError:
            continue
        for key, row in row_of_key.items():
            if row >= matrix.shape[0] or col >= matrix.shape[1]:
                continue
            value = float(matrix[row, col])
            if value != 0.0:
                cells[(key, edge_id)] = value
    return cells


def _table_rows(table, key_column) -> dict:
    """Return a dataframe as a mapping from its key column to a row dictionary."""
    if table is None:
        return {}
    rows = table.to_dicts() if hasattr(table, 'to_dicts') else list(table.rows(named=True))
    out = {}
    for row in rows:
        key = row.get(key_column)
        out[key] = {k: v for k, v in row.items() if k != key_column and v is not None}
    return out


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _compare_sets(problems, label, left, right):
    left_set, right_set = set(left), set(right)
    if left_set != right_set:
        only_left = _sorted_repr(left_set - right_set)
        only_right = _sorted_repr(right_set - left_set)
        problems.append(f'{label} differ: only left {only_left}, only right {only_right}')


def compare(left, right) -> list[str]:
    """Compare two graphs and return one message per difference.

    An empty list means the two graphs answer every structural question the
    same way.
    """
    problems: list[str] = []

    _compare_sets(problems, 'nodes', _node_ids(left), _node_ids(right))
    _compare_sets(problems, 'supra-nodes', _supra_node_keys(left), _supra_node_keys(right))
    _compare_sets(problems, 'edges', _edge_ids(left), _edge_ids(right))

    # edge_list carries the endpoints and the effective weight of every binary edge.
    left_list = {(s, t, e): w for s, t, e, w in left.edge_list()}
    right_list = {(s, t, e): w for s, t, e, w in right.edge_list()}
    _compare_sets(problems, 'edge_list entries', left_list, right_list)
    for key in sorted(set(left_list) & set(right_list), key=repr):
        if not _close(left_list[key], right_list[key]):
            problems.append(
                f'edge_list weight for {key!r} differs: {left_list[key]} != {right_list[key]}'
            )

    # get_edge must return the same view for every shared edge id.
    for edge_id in sorted(set(_edge_ids(left)) & set(_edge_ids(right))):
        left_view = _edge_view_signature(left.get_edge(edge_id))
        right_view = _edge_view_signature(right.get_edge(edge_id))
        if left_view != right_view:
            problems.append(f'get_edge({edge_id!r}) differs: {left_view} != {right_view}')

    # incident_edges must return the same edges in every direction.
    for node_id in sorted(set(_node_ids(left)) & set(_node_ids(right))):
        for direction in ('in', 'out', 'both'):
            left_inc = _incident_signature(left, node_id, direction)
            right_inc = _incident_signature(right, node_id, direction)
            if left_inc != right_inc:
                problems.append(
                    f'incident_edges({node_id!r}, {direction!r}) differs: '
                    f'only left {_sorted_repr(left_inc - right_inc)}, '
                    f'only right {_sorted_repr(right_inc - left_inc)}'
                )

    # The matrix comparison is addressed by id, never by position.
    left_cells = _matrix_cells(left)
    right_cells = _matrix_cells(right)
    _compare_sets(problems, 'matrix nonzeros', left_cells, right_cells)
    for key in sorted(set(left_cells) & set(right_cells), key=repr):
        if not _close(left_cells[key], right_cells[key]):
            problems.append(f'matrix cell {key!r} differs: {left_cells[key]} != {right_cells[key]}')

    # The node table and the edge table.
    for label, table_name, key_column in (
        ('obs', 'obs', 'vertex_id'),
        ('var', 'var', 'edge_id'),
    ):
        left_rows = _table_rows(getattr(left, table_name), key_column)
        right_rows = _table_rows(getattr(right, table_name), key_column)
        _compare_sets(problems, f'{label} keys', left_rows, right_rows)
        for key in sorted(set(left_rows) & set(right_rows), key=repr):
            if left_rows[key] != right_rows[key]:
                problems.append(
                    f'{label} row {key!r} differs: {left_rows[key]} != {right_rows[key]}'
                )

    return problems


def assert_equivalent(left, right) -> None:
    """Fail the test when the two graphs answer any structural question differently."""
    problems = compare(left, right)
    assert not problems, 'cores are not equivalent:\n  ' + '\n  '.join(problems)
