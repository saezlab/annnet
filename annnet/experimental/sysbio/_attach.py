"""Join a measurement matrix onto a graph, as a join with a stated policy.

Not to be confused with :mod:`annnet.experimental.scverse`, which serialises a
*graph* as an AnnData. This is the opposite direction: an AnnData holding
measurements — ``obs`` conditions, ``var`` measured entities — joined onto a
graph that already exists.

**Attach is a join, not a merge.** The entities a network names and the entities
an assay measures do not correspond one to one. One measured thing may reach
several nodes; one node may be reached by several measured things; plenty of both
match nothing. Every one of those is a decision, so the caller states it and what
did not match comes back as data.

**Values are attached, not copied.** The array stays where it was and the graph
holds two index maps onto it. Twenty million cells attach in milliseconds and
read back in under a second, which is what makes the AnnData the place the
measurements live.

**Placement is separate, and small.** Reading an attached array never consults
presence, so nothing here needs a node-layer to exist. What does need one is a
*result* written back onto the graph later — so ``place`` defaults to the pairs
the join actually matched, which is the size of the network rather than the size
of the assay.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ...core import AnnNet
from ._support import column as _column, coordinate as _coordinate
from ..vocabulary import names_of, as_mapper
from ..scverse._shared import require_dependency
from ..._support.dataframe_backend import dataframe_from_columns

#: How several measured things reaching one node become one number. ``never`` and
#: ``error`` are refusals and are handled before these are reached.
REDUCERS: dict[str, Any] = {
    'min': lambda block: np.nanmin(block, axis=1),
    'max': lambda block: np.nanmax(block, axis=1),
    'mean': lambda block: np.nanmean(block, axis=1),
    'sum': lambda block: np.nansum(block, axis=1),
    'geometric': lambda block: np.exp(np.nanmean(np.log(block), axis=1)),
    'first': lambda block: block[:, 0],
}

#: What a ``kind``-keyed policy may say about several entities reaching one node.
KIND_POLICIES = ('never', 'error', *REDUCERS)

#: What a bare-string policy may say — the other direction of multiplicity.
ENTITY_POLICIES = ('allow', 'error', 'first')

#: What to do about a composite only some of whose members were measured.
PARTIAL_POLICIES = ('reduce', 'skip', 'error')

#: Which node-layers a join creates.
PLACEMENTS = ('matched', 'none', 'all')


class AttachReport:
    """What a join matched, and what it did not.

    The shortfall is the interesting part, so it is the return value rather than
    a log line — and it comes back as frames, because a list answers *how many*
    and a frame answers *which, and why*.

    Attributes
    ----------
    layers : list[tuple]
        The layer coordinates the conditions became.
    matched : dict[str, list[str]]
        Measured entity -> the node ids it was joined to.
    multiplicity : dict[str, int]
        Measured entities that reached more than one node, and how many.
    placed : int
        Node-layers created.
    """

    __slots__ = ('layers', 'matched', 'multiplicity', 'placed', '_entities', '_unreached')

    def __init__(
        self,
        layers=None,
        matched=None,
        multiplicity=None,
        placed=0,
        entities=(),
        unreached=(),
    ) -> None:
        self.layers = list(layers or ())
        self.matched = dict(matched or {})
        self.multiplicity = dict(multiplicity or {})
        self.placed = placed
        self._entities = list(entities)
        self._unreached = list(unreached)

    # -- the frames -------------------------------------------------------

    @property
    def mapping(self) -> Any:
        """One row per (measured entity, node it reached), with a status.

        Columns ``input``, ``resolved``, ``status``, where status is ``"ok"``,
        ``"ambiguous"`` (one row per node reached) or ``"unmapped"``. Long rather
        than wide: a list column is slow and hard to read.
        """
        inputs: list = []
        resolved: list = []
        status: list = []
        for entity in self._entities:
            found = self.matched.get(entity) or []
            if not found:
                inputs.append(entity)
                resolved.append(None)
                status.append('unmapped')
                continue
            label = 'ambiguous' if len(found) > 1 else 'ok'
            for node_id in found:
                inputs.append(entity)
                resolved.append(node_id)
                status.append(label)
        return dataframe_from_columns(
            {'input': inputs, 'resolved': resolved, 'status': status},
            schema={'input': 'text', 'resolved': 'text', 'status': 'text'},
        )

    @property
    def unmapped(self) -> Any:
        """The measured entities that reached no node, as a frame."""
        inputs = self.unmapped_ids
        return dataframe_from_columns(
            {
                'input': inputs,
                'resolved': [None] * len(inputs),
                'status': ['unmapped'] * len(inputs),
            },
            schema={'input': 'text', 'resolved': 'text', 'status': 'text'},
        )

    @property
    def unmapped_ids(self) -> list[str]:
        """The same, as a list.

        Kept beside the frame because ``report.unmapped == ['Q']`` against a
        frame is silently ``False`` — a failure mode that lives in test code and
        says nothing when it fires.
        """
        return [entity for entity in self._entities if not self.matched.get(entity)]

    @property
    def unmeasured(self) -> Any:
        """The nodes in scope no measured entity reached, as a frame."""
        return dataframe_from_columns(
            {'node_id': list(self._unreached)}, schema={'node_id': 'text'}
        )

    @property
    def unmeasured_ids(self) -> list[str]:
        """The same, as a list."""
        return list(self._unreached)

    @property
    def coverage(self) -> float:
        """The share of measured entities that reached a node."""
        total = len(self._entities)
        return len(self.matched) / total if total else 0.0

    def __repr__(self) -> str:
        unmapped = len(self._entities) - len(self.matched)
        return (
            f'AttachReport(layers={len(self.layers)}, matched={len(self.matched)}, '
            f'unmapped={unmapped}, unmeasured={len(self._unreached)}, '
            f'coverage={self.coverage:.1%})'
        )


def _check_multiplicity(multiplicity) -> None:
    """Refuse an unusable policy before any work is done."""
    if isinstance(multiplicity, str):
        if multiplicity not in ENTITY_POLICIES:
            raise ValueError(
                f'multiplicity must be one of {ENTITY_POLICIES}, or a dict keyed on '
                f'node kind, got {multiplicity!r}'
            )
        return
    if not isinstance(multiplicity, dict):
        raise ValueError(
            f'multiplicity must be a string or a dict keyed on node kind, got '
            f'{type(multiplicity).__name__}'
        )
    unknown = sorted({value for value in multiplicity.values() if value not in KIND_POLICIES})
    if unknown:
        raise ValueError(
            f'unknown multiplicity policy(ies) {unknown!r}; a kind may take one of '
            f'{sorted(KIND_POLICIES)!r}'
        )


def _policy_for(kind, multiplicity) -> str:
    """What this node's kind says to do about several entities reaching it."""
    if isinstance(multiplicity, str):
        return 'error'
    return multiplicity.get(str(kind), multiplicity.get('default', 'error'))


def _match(graph, entities, on, policy, mapper, partial):
    """Return ``(matched, unmapped, multiplicity, composites)`` for the join.

    ``matched`` is entity -> the nodes it reached; ``composites`` is node ->
    ``(reducer, member names)`` for the nodes needing one number out of several.
    """
    wanted = set(entities)
    kinds: dict[str, Any] = {}
    holders: dict[str, list[str]] = {}
    parts: dict[str, tuple] = {}

    if on == 'node_id':
        rows = {node_id: {} for node_id in graph.nodes()}

        def key_of(node_id, attrs):
            return node_id
    else:
        rows = graph._attr_store.node_attr_rows()

        def key_of(node_id, attrs):
            return attrs.get(on)

    for node_id, attrs in rows.items():
        key = key_of(node_id, attrs)
        if key is None:
            continue
        kinds[node_id] = attrs.get('kind')
        # What this name stands for is the resource's business, not the join's.
        members = list(names_of(mapper, key))
        present = [name for name in members if name in wanted]
        if not present:
            # Index the whole name too, so an entity named exactly like the
            # unsplit value still finds it.
            holders.setdefault(str(key), []).append(node_id)
            continue
        for name in present:
            holders.setdefault(name, []).append(node_id)
        parts[node_id] = (tuple(members), tuple(present))

    matched: dict[str, list[str]] = {}
    unmapped: list[str] = []
    multiplicities: dict[str, int] = {}
    refused: set[str] = set()
    composites: dict[str, tuple] = {}

    for node_id, (members, present) in parts.items():
        rule = _policy_for(kinds.get(node_id), policy)
        if rule == 'never':
            refused.add(node_id)
            continue
        if len(members) == 1:
            continue
        if rule == 'error':
            # A bare string answered the *other* direction, so this one is still
            # unstated — and saying only "not stated" leaves a caller who did
            # pass a policy wondering which one the message means.
            aside = (
                f' A bare multiplicity={policy!r} states how one measured entity '
                f'reaching several nodes behaves; this is the other direction and '
                f'needs a dict.'
                if isinstance(policy, str)
                else ''
            )
            raise ValueError(
                f'node {node_id!r} of kind {kinds.get(node_id)!r} is made of '
                f'{len(members)} measured entities ({sorted(members)[:5]!r}), and how to '
                f'combine them is not stated. Pass multiplicity={{{kinds.get(node_id)!r}: '
                f'"min"}} — the scarcest member — or name another of '
                f'{sorted(REDUCERS)!r}.{aside}'
            )
        if len(present) < len(members):
            if partial == 'skip':
                refused.add(node_id)
                continue
            if partial == 'error':
                raise ValueError(
                    f'node {node_id!r} is made of {len(members)} entities and only '
                    f'{len(present)} were measured. Pass partial="reduce" to combine '
                    f'what there is, or partial="skip" to leave the node unmeasured.'
                )
        composites[node_id] = (rule, tuple(sorted(present)))

    for entity in entities:
        found = [node_id for node_id in holders.get(entity, ()) if node_id not in refused]
        if not found:
            unmapped.append(entity)
            continue
        if len(found) > 1:
            multiplicities[entity] = len(found)
            if isinstance(policy, str):
                if policy == 'error':
                    raise ValueError(
                        f'{entity!r} reaches {len(found)} nodes ({sorted(found)[:5]!r}). '
                        f'Pass multiplicity="allow" to join all of them, or "first" to '
                        f'take one.'
                    )
                if policy == 'first':
                    found = [sorted(found)[0]]
        matched[entity] = sorted(found)
    return matched, unmapped, multiplicities, composites


def _attach_composites(graph, arrays, coordinates, composites, positions, mask) -> None:
    """One computed column per node that several measured things make up.

    A second backing, so the caller's array stays uncopied and only the combined
    columns are materialised. Attached last, so it wins for its own nodes.
    """
    order = sorted(composites)
    computed: dict[str, Any] = {}
    for name, array in arrays.items():
        block = np.empty((array.shape[0], len(order)), dtype=float)
        for index, node_id in enumerate(order):
            rule, members = composites[node_id]
            taken = [positions[part] for part in members if part in positions]
            values = np.asarray(array, dtype=float)[:, taken]
            if mask is not None:
                values = np.where(np.asarray(mask)[:, taken], values, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                block[:, index] = REDUCERS[rule](values)
        computed[name] = block

    graph.layers.attach_values(
        computed,
        layers=coordinates,
        nodes=order,
        rows={coordinate: row for row, coordinate in enumerate(coordinates)},
        columns={node_id: index for index, node_id in enumerate(order)},
    )


def attach(
    graph: AnnNet,
    adata: Any,
    *,
    aspect: str,
    layer_key: str | None = None,
    on: str = 'node_id',
    var_key: str | None = None,
    values: dict[str, str | None] | None = None,
    gate: str | None = None,
    within: dict | None = None,
    mapper: Any = None,
    unmapped: str = 'report',
    multiplicity: str | dict = 'allow',
    partial: str = 'reduce',
    place: str = 'matched',
    scope: Any = None,
) -> AttachReport:
    """Join an AnnData's measurements onto a graph's node-layers.

    Parameters
    ----------
    graph : AnnNet
        The graph to attach to. It must declare ``aspect``.
    adata : anndata.AnnData
        ``obs`` are the conditions, ``var`` the measured entities.
    aspect : str
        Which aspect of the graph the conditions are values of.
    layer_key : str, optional
        The ``obs`` column holding the condition name. Default: ``obs_names``.
    on : str, default "node_id"
        What to match a measured entity against: ``"node_id"``, or a node
        attribute, which lets one measured thing reach several nodes.
    var_key : str, optional
        The ``var`` column holding the entity name. Default: ``var_names``.
    values : dict[str, str | None], optional
        ``{attribute name: adata layer name}``; ``None`` means ``adata.X``.
        Default: ``{"value": None}``.
    gate : str, optional
        A boolean ``adata`` layer. A cell holds a value only where it is true, so
        a condition that was not measured stays absent rather than becoming zero.
    within : dict, optional
        Fix the other aspects, as ``{aspect: value}``. Required when the graph
        declares more than one.
    mapper : Mapper | Mapping, optional
        How a node's join key resolves, and what a composite name is made of.
        Default: names are taken as they are. This is where a resource's own
        spelling rules live — see
        :mod:`annnet.experimental.vocabulary._identifiers`.
    unmapped : {"report", "error", "ignore"}, default "report"
        What to do about a measured entity that reached no node.
    multiplicity : str | dict, default "allow"
        Multiplicity runs in two directions, named apart here by type.

        A **string** is one measured entity reaching several nodes. Every node
        takes the same value, so nothing is combined: ``"allow"`` joins all,
        ``"error"`` refuses, ``"first"`` takes one.

        A **dict** is the other direction — one *node* made of several measured
        things — keyed on the node's ``kind``, as
        ``{"complex": "min", "default": "error"}``. Values are the reducers in
        :data:`REDUCERS`, plus ``"never"`` (this kind takes no measurement) and
        ``"error"``, the default for a kind nobody named.
    partial : {"reduce", "skip", "error"}, default "reduce"
        What to do about a composite only *some* of whose members were measured.
        ``"reduce"`` combines what there is — so a node is as scarce as its
        scarcest **measured** member, which is a different claim from the one the
        policy names, and is why this is stated rather than assumed.
    place : {"matched", "none", "all"}, default "matched"
        Which node-layers to create. Reading an attached array never consults
        presence, so nothing here needs them — but a *result* written back later
        does. ``"matched"`` places the pairs the join reached, gated by ``gate``.
        ``"none"`` places nothing. ``"all"`` places the whole cross product,
        which is the size of the assay and rarely what is wanted.
    scope : Iterable[str], optional
        Which nodes ``unmeasured`` is counted against. Default: the nodes the
        join could have reached. Without this a two-slice graph reports every
        node of the other slice as unmeasured, which is true and useless.

    Returns
    -------
    AttachReport

    Raises
    ------
    KeyError
        If ``aspect`` is not declared, or a named column is missing.
    ValueError
        If a policy value is unknown, a condition is not a value of ``aspect``,
        or ``within`` does not fix every other aspect.

    Examples
    --------
    >>> report = exp.sysbio.attach(  # doctest: +SKIP
    ...     G,
    ...     adata,
    ...     aspect='condition',
    ...     on='symbol',
    ...     multiplicity={'complex': 'min', 'default': 'error'},
    ... )
    >>> report.coverage  # doctest: +SKIP
    0.94
    """
    require_dependency('anndata', 'annnet[scverse] or pip install anndata')
    if unmapped not in ('report', 'error', 'ignore'):
        raise ValueError(f"unmapped must be one of ('report', 'error', 'ignore'), got {unmapped!r}")
    if partial not in PARTIAL_POLICIES:
        raise ValueError(f'partial must be one of {PARTIAL_POLICIES}, got {partial!r}')
    if place not in PLACEMENTS:
        raise ValueError(f'place must be one of {PLACEMENTS}, got {place!r}')
    _check_multiplicity(multiplicity)

    aspects = tuple(graph.aspects or ())
    if aspect not in aspects:
        raise KeyError(f'unknown aspect {aspect!r}; this graph declares {list(aspects)!r}')

    conditions = _column(adata.obs, layer_key, adata.obs_names, 'obs')
    entities = _column(adata.var, var_key, adata.var_names, 'var')
    coordinates = [_coordinate(aspects, aspect, value, within) for value in conditions]

    declared = set(graph.layers.aspect(aspect).values)
    unknown = sorted({c for c in conditions if c not in declared})
    if unknown:
        raise ValueError(
            f'condition(s) {unknown!r} are not values of aspect {aspect!r}, which holds '
            f'{sorted(declared)!r}. Declare them before attaching.'
        )

    matched, unmapped_entities, multiplicities, composites = _match(
        graph, entities, on, multiplicity, as_mapper(mapper), partial
    )
    if unmapped == 'error' and unmapped_entities:
        raise ValueError(
            f'{len(unmapped_entities)} measured entit(ies) reached no node, the first few '
            f'being {unmapped_entities[:5]!r}. Pass unmapped="report" to get them back as '
            f'data, or unmapped="ignore" to accept the loss.'
        )

    positions = {entity: index for index, entity in enumerate(entities)}
    columns: dict[str, int] = {}
    for position, entity in enumerate(entities):
        for node_id in matched.get(entity, ()):
            columns[node_id] = position
    # A node made of several entities gets no column; its value is computed
    # below into a second backing.
    for node_id in composites:
        columns.pop(node_id, None)

    mask = np.asarray(adata.layers[gate]) if gate is not None else None
    arrays = {}
    for name, source in (values or {'value': None}).items():
        arrays[name] = np.asarray(adata.layers[source] if source is not None else adata.X)

    if arrays and columns:
        graph.layers.attach_values(
            arrays,
            layers=coordinates,
            nodes=sorted(columns),
            rows={coordinate: row for row, coordinate in enumerate(coordinates)},
            columns=columns,
            mask=mask,
        )
    if arrays and composites:
        _attach_composites(graph, arrays, coordinates, composites, positions, mask)

    reached = {node_id for ids in matched.values() for node_id in ids}
    placed = _place(graph, coordinates, matched, entities, mask, place, reached)

    in_scope = set(scope) if scope is not None else set(graph.nodes())
    return AttachReport(
        layers=coordinates,
        matched=matched,
        multiplicity=multiplicities,
        placed=placed,
        entities=entities if unmapped != 'ignore' else list(matched),
        unreached=sorted(in_scope - reached),
    )


def _place(graph, coordinates, matched, entities, mask, place, reached) -> int:
    """Create the node-layers the policy asks for, honouring the gate."""
    if place == 'none' or not coordinates:
        return 0
    if place == 'all':
        return graph.layers.place(sorted(graph.nodes()), coordinates)
    if not reached:
        return 0
    if mask is None:
        return graph.layers.place(sorted(reached), coordinates)

    # With a gate, a node-layer exists only where one of the entities reaching
    # that node was measured in that condition.
    order = sorted(reached)
    index = {node_id: position for position, node_id in enumerate(order)}
    gate = np.zeros((len(coordinates), len(order)), dtype=bool)
    for position, entity in enumerate(entities):
        for node_id in matched.get(entity, ()):
            column = index.get(node_id)
            if column is not None:
                gate[:, column] |= np.asarray(mask)[:, position].astype(bool)
    return graph.layers.place(order, coordinates, mask=gate)
