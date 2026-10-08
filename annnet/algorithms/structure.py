"""Predicates about the shape of a graph, with no meaning attached.

A method that reads a graph often needs one fact about its shape before the
numbers mean anything: whether the directed edges close a cycle, or whether a
node ever sits on both ends of them. Those facts are topology, so they are
answered here, in the biology-free layer, and a domain package asks rather than
re-deriving them. Everything reads through the structural query facade.
"""

from __future__ import annotations

from ..core import _structure as S, as_endpoint


def directed_pairs(graph):
    """Yield ``(source_id, target_id)`` for every directed binary edge.

    An undirected edge has two sides in storage order and no direction, so it
    is left out. A hyperedge has no single source and no single target, so it is
    left out too, as are edges incident to edge entities. Ids are bare node
    ids, whatever layer the node sits in.
    """
    for ref in S.iter_edges(graph):
        if not ref.directed or ref.kind == S.HYPER:
            continue
        sides = S.edge_sides(graph, ref.id)
        if len(sides.source) != 1 or len(sides.target) != 1:
            continue
        source_key = next(iter(sides.source))
        target_key = next(iter(sides.target))
        if any(S.entity_ref(graph, key).kind != S.NODE for key in (source_key, target_key)):
            continue
        source = as_endpoint(source_key).node_id
        target = as_endpoint(target_key).node_id
        yield source, target


def sources(graph) -> set[str]:
    """The nodes a directed binary edge starts from."""
    return {source for source, _target in directed_pairs(graph)}


def targets(graph) -> set[str]:
    """The nodes a directed binary edge ends at."""
    return {target for _source, target in directed_pairs(graph)}


def directed_cycle(graph) -> list[str] | None:
    """One directed cycle, as the nodes along it, or ``None`` when there is none.

    Depth-first over the directed binary edges, stopping at the first cycle
    found. The list starts and ends on the same node, so a self-loop comes back
    as ``[node, node]``. Which cycle is found when there are several is not
    promised.
    """
    successors: dict[str, list[str]] = {}
    for source, target in directed_pairs(graph):
        successors.setdefault(source, []).append(target)

    # Iterative DFS avoids recursion limits and repeated path copies.
    finished = set()
    for root in successors:
        if root in finished:
            continue
        path = [root]
        positions = {root: 0}
        stack = [iter(successors.get(root, ()))]
        while stack:
            nxt = next(stack[-1], None)
            if nxt is None:
                node = path.pop()
                finished.add(node)
                del positions[node]
                stack.pop()
                continue
            if nxt in finished:
                continue
            if nxt in positions:
                return [*path[positions[nxt] :], nxt]
            positions[nxt] = len(path)
            path.append(nxt)
            stack.append(iter(successors.get(nxt, ())))
    return None
