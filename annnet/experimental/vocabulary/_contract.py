"""Whether a graph's numbers mean what a method will assume.

:meth:`AnnNet.validate` asks whether the object is *internally consistent* — the
slot bijections hold, the member segments do not overlap, the matrix agrees with
the store. A failure there means the graph is broken.

This asks a different question: do the attributes **mean** what a method reading
them will assume? A failure here means the graph is fine and the data in it will
be misread. Sign written into ``weight``; ``0.0`` standing for "not reported"; a
confidence and a coefficient in one column. None of that is a structural fault,
and every one of it produces a wrong answer that looks right.

The two are separate entry points on purpose. Folded together, a broken store and
a mislabelled attribute would arrive as two strings in one list, and a caller
could not tell which one means *your object is corrupt* from which means *your
data does not say what you think*.
"""

from __future__ import annotations

from collections.abc import Iterator

from ._names import (
    SIGN,
    WEIGHT,
    CONFIDENCE,
    SIGN_DOMAIN,
    EDGE_RESERVED,
    NODE_RESERVED,
    CONFIDENCE_RANGE,
    ContractViolation,
)

#: One message per problem; an empty report means the graph honours the contract.
Report = list[str]

#: The contract a graph is read against when none is named.
DEFAULT_CONTRACT = 'sysbio'

#: What a method may require of a graph's shape.
CAPABILITIES = (
    'signed',
    'directed',
    'dyadic',
    'weighted',
    'stoichiometric',
    'acyclic',
    'bipartite',
)


def _edge_attrs(graph) -> dict:
    return graph._attr_store.edge_attr_rows()


def _node_attrs(graph) -> dict:
    return graph._attr_store.node_attr_rows()


# ---------------------------------------------------------------------------
# Rules. Each reads a graph and yields one message per problem, through the
# public surface only — so a rule keeps working when the store underneath moves.
# ---------------------------------------------------------------------------


def _sign_domain(graph) -> Iterator[str]:
    """``sign`` holds ``-1`` or ``+1``, and nothing else."""
    for edge_id, attrs in _edge_attrs(graph).items():
        value = attrs.get(SIGN)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            yield f'edge {edge_id!r} has sign {value!r}, which is not a number'
            continue
        if number == 0.0:
            yield (
                f'edge {edge_id!r} has sign 0, which the contract reserves for '
                f'nothing: unknown is null, not zero'
            )
        elif number not in SIGN_DOMAIN:
            yield (
                f'edge {edge_id!r} has sign {number!r}, and the domain is '
                f'{{-1, +1}}. A continuous value belongs in {CONFIDENCE!r}.'
            )


def _confidence_range(graph) -> Iterator[str]:
    """``confidence`` is a number in ``[0, 1]``."""
    low, high = CONFIDENCE_RANGE
    for edge_id, attrs in _edge_attrs(graph).items():
        value = attrs.get(CONFIDENCE)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            yield f'edge {edge_id!r} has confidence {value!r}, which is not a number'
            continue
        if not (low <= number <= high):
            yield f'edge {edge_id!r} has confidence {number!r}, and the range is [{low}, {high}]'


def _weight_is_structural(graph) -> Iterator[str]:
    """``weight`` is the incidence coefficient, so it does not hold a sign.

    The tell is a graph whose every weight is ``-1``, ``0`` or ``+1`` while no
    edge carries a ``sign``: that is a sign column wearing the structural name,
    and every matrix built from it is arithmetic over the wrong quantity.
    """
    rows = _edge_attrs(graph)
    if any(attrs.get(SIGN) is not None for attrs in rows.values()):
        return
    # Read the structural weight, not an attribute row. `weight` *is* the
    # incidence coefficient, so it lives on the edge itself — looking for it in
    # the attribute table is the same confusion this rule exists to catch.
    weights = []
    for edge_id in graph.edges():
        found = graph.get_edge(edge_id).weight
        if found is None:
            continue
        try:
            weights.append(float(found))
        except (TypeError, ValueError):
            continue
    if len(weights) < 2 or not any(value < 0 for value in weights):
        return
    if set(weights) <= {-1.0, 0.0, 1.0}:
        yield (
            f'every {WEIGHT!r} in this graph is -1, 0 or +1 and no edge carries a '
            f'{SIGN!r}. That reads as a sign column under the structural name; '
            f'{WEIGHT!r} is the incidence coefficient and arithmetic over the '
            f'matrix is arithmetic over it.'
        )


def _no_zero_for_unknown(graph) -> Iterator[str]:
    """Zero is a number. Unknown is null."""
    for edge_id, attrs in _edge_attrs(graph).items():
        if attrs.get(CONFIDENCE) == 0.0 and attrs.get(SIGN) is None:
            yield (
                f'edge {edge_id!r} has confidence 0.0 and no sign, which usually '
                f'means "not reported". Leave it null: zero is a value and it '
                f'will be averaged.'
            )


def _reserved_names_on_the_right_axis(graph) -> Iterator[str]:
    """An edge name on the node axis, or the other way round, is a mistake."""
    for node_id, attrs in _node_attrs(graph).items():
        for name in attrs:
            if name in EDGE_RESERVED and name not in NODE_RESERVED:
                yield (
                    f'node {node_id!r} carries {name!r}, which this vocabulary '
                    f'reserves for the edge axis'
                )


#: The rules each named contract is made of.
_CONTRACTS: dict[str, tuple] = {
    'sysbio': (
        _sign_domain,
        _confidence_range,
        _weight_is_structural,
        _no_zero_for_unknown,
        _reserved_names_on_the_right_axis,
    ),
}


# ---------------------------------------------------------------------------
# Capabilities — questions about shape rather than about meaning
# ---------------------------------------------------------------------------


def _capability_problems(graph, name: str) -> Report:
    """One capability, checked."""
    from ...core import _structure

    if name == 'signed':
        rows = _edge_attrs(graph)
        missing = [e for e in graph.edges() if rows.get(e, {}).get(SIGN) is None]
        return _shortfall(missing, len(list(graph.edges())), f'carry {SIGN!r}')
    if name == 'directed':
        undirected = [e for e in graph.edges() if not graph.get_edge(e).directed]
        return _shortfall(undirected, len(list(graph.edges())), 'run one way')
    if name == 'dyadic':
        if _structure.is_flat(graph):
            return []
        found = [ref.id for ref in _structure.iter_edges(graph) if ref.kind == _structure.HYPER]
        return _shortfall(found, len(list(graph.edges())), 'join exactly two entities', have=False)
    if name == 'weighted':
        missing = [e for e in graph.edges() if graph.get_edge(e).weight is None]
        return _shortfall(missing, len(list(graph.edges())), f'carry {WEIGHT!r}')
    if name == 'stoichiometric':
        if _structure.hyperedges_with_coefficients(graph):
            return []
        return ['no hyperedge in this graph weights its members separately']
    if name == 'bipartite':
        return _bipartite_problems(graph)
    if name == 'acyclic':
        cycle = _find_cycle(graph)
        return [] if cycle is None else [f'this graph has a directed cycle: {cycle!r}']
    raise ValueError(f'unknown capability {name!r}; known: {list(CAPABILITIES)}')


def _shortfall(offending, total: int, what: str, *, have: bool = True) -> Report:
    """One message naming how many elements fall short, and a few of them."""
    if not offending:
        return []
    shown = sorted(str(item) for item in offending)[:3]
    verb = 'do not' if have else 'do'
    return [f'{len(offending)} of {total} edges {verb} {what} (e.g. {shown!r})']


def _bipartite_problems(graph) -> Report:
    """No entity is both a source and a target."""
    sources, targets = set(), set()
    for edge_id in graph.edges():
        view = graph.get_edge(edge_id)
        if view.source_id is not None:
            sources.add(view.source_id)
        if view.target_id is not None:
            targets.add(view.target_id)
    both = sources & targets
    if not both:
        return []
    return [
        f'{len(both)} entit(ies) are both a source and a target '
        f'(e.g. {sorted(both)[:3]!r}), so this graph is not bipartite'
    ]


def _find_cycle(graph) -> list | None:
    """One directed cycle, or ``None``. Depth-first, and stops at the first."""
    successors: dict[str, list[str]] = {}
    for edge_id in graph.edges():
        view = graph.get_edge(edge_id)
        if not view.directed or view.source_id is None or view.target_id is None:
            continue
        successors.setdefault(view.source_id, []).append(view.target_id)

    WHITE, GREY, BLACK = 0, 1, 2
    colour: dict[str, int] = {}

    def walk(node: str, trail: list[str]) -> list | None:
        colour[node] = GREY
        for nxt in successors.get(node, ()):
            state = colour.get(nxt, WHITE)
            if state == GREY:
                return [*trail[trail.index(nxt) :], nxt] if nxt in trail else [nxt, node, nxt]
            if state == WHITE:
                found = walk(nxt, [*trail, nxt])
                if found is not None:
                    return found
        colour[node] = BLACK
        return None

    for node in list(successors):
        if colour.get(node, WHITE) == WHITE:
            found = walk(node, [node])
            if found is not None:
                return found
    return None


# ---------------------------------------------------------------------------
# The entry points
# ---------------------------------------------------------------------------


def contracts() -> list[str]:
    """The named contracts this vocabulary knows."""
    return sorted(_CONTRACTS)


def capabilities() -> list[str]:
    """The capabilities a method may require."""
    return list(CAPABILITIES)


def rules(contract: str = DEFAULT_CONTRACT) -> list[str]:
    """The names of the rules one contract is made of."""
    return [rule.__name__.lstrip('_') for rule in _rules(contract)]


def _rules(contract: str) -> tuple:
    try:
        return _CONTRACTS[contract]
    except KeyError:
        raise KeyError(f'unknown contract {contract!r}; known: {contracts()!r}') from None


def check(graph, contract: str = DEFAULT_CONTRACT, *, method=None, strict: bool = False) -> Report:
    """Report where a graph's data does not mean what it is read as.

    Parameters
    ----------
    graph : AnnNet
    contract : str, default "sysbio"
        Which set of rules to read against.
    method : MethodSpec, optional
        Also check what this method requires — its attributes and its shape.
    strict : bool, default False
        Raise :class:`ContractViolation` instead of returning the report.

    Returns
    -------
    list[str]
        One message per problem. Empty means the graph honours the contract.

    Raises
    ------
    KeyError
        If the contract is unknown.
    ContractViolation
        Under ``strict``, when the report is not empty.

    Examples
    --------
    >>> exp.vocabulary.check(G)  # doctest: +SKIP
    []
    >>> exp.vocabulary.check(G, method=SPEC, strict=True)  # doctest: +SKIP
    """
    found: Report = []
    for rule in _rules(contract):
        found.extend(rule(graph))
    if method is not None:
        found.extend(method_problems(graph, method))
    if strict and found:
        raise ContractViolation('; '.join(found))
    return found


def requires(graph, *names: str, strict: bool = True) -> Report:
    """Report where a graph falls short of the named capabilities.

    Parameters
    ----------
    graph : AnnNet
    *names
        Capability names from :data:`CAPABILITIES`.
    strict : bool, default True
        Raise :class:`ContractViolation` instead of returning the report. The
        default is the strict one here, because a caller writing ``requires`` is
        stating a precondition rather than asking a question.

    Returns
    -------
    list[str]

    Raises
    ------
    ValueError
        If a capability name is unknown.
    ContractViolation
        Under ``strict``, when the report is not empty.
    """
    found: Report = []
    for name in names:
        found.extend(_capability_problems(graph, name))
    if strict and found:
        raise ContractViolation('; '.join(found))
    return found


def method_problems(graph, spec) -> Report:
    """Everything one method's requirements are not met by."""
    found: Report = []
    rows = _edge_attrs(graph)
    total = len(list(graph.edges()))
    for name in spec.requires_edge:
        missing = [e for e in graph.edges() if rows.get(e, {}).get(name) is None]
        if missing:
            shown = sorted(str(item) for item in missing)[:3]
            found.append(
                f'{spec.name!r} needs all {total} edges to carry {name!r}; '
                f'{len(missing)} do not (e.g. {shown!r})'
            )
    node_rows = _node_attrs(graph)
    for name in spec.requires_node:
        missing = [n for n in graph.nodes() if node_rows.get(n, {}).get(name) is None]
        if missing:
            shown = sorted(str(item) for item in missing)[:3]
            found.append(
                f'{spec.name!r} needs every node to carry {name!r}; '
                f'{len(missing)} do not (e.g. {shown!r})'
            )
    for capability in spec.capabilities():
        for message in _capability_problems(graph, capability):
            found.append(f'{spec.name!r} needs {capability}: {message}')
    return found
