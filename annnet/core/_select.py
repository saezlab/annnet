"""Typed selections: sequences, attribute-row selections and their algebra.

This module owns one thing: how a selection is described, resolved and
combined. A selection is a small immutable *plan*. Its leaves are evaluated
inside the scope they were made in — ``G.N.select(...)`` inside the graph,
``V.N.select(...)`` inside the view ``V`` — and an inner node combines two
plans with ``&``, ``|`` or ``-``. Because a leaf remembers its own scope, two
selections made in two different views combine correctly: each side is
evaluated where it was asked, and the combination is ordered by the root
graph's axis.

Three families share the machinery:

- :class:`NodeSequence` and :class:`EdgeSequence` — one axis of a graph or a
  view, readable as a sequence (positions, ranges, columns, records) and
  filterable into a live selection;
- :class:`RowSelection` — the rows of one attribute address, keyed as the
  address keys them, with explicit :meth:`RowSelection.project` onto an axis;
- :class:`SliceSelection` — slice ids named by a projection.

Predicate parsing and evaluation live in :mod:`annnet.core._predicate`;
membership and boundary resolution of a whole view in
:mod:`annnet.core._resolve`; the read-only view object in
:mod:`annnet.core._Views`. Nothing here imports those two.

The scope protocol: an owner of an axis (a graph or a view) answers
``_selection_context()`` with ``(root_graph, resolved)`` where ``resolved`` is
``None`` for the graph itself and the view's resolved membership otherwise. An
attribute reader (``G.attrs`` or ``V.attrs``) answers ``_domain``, ``_row``,
``_query_row``, ``_orders``, ``_key``, ``_selection_owner`` and ``_root_reader``.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from collections.abc import Mapping

import numpy as np

from . import _structure
from ._attrs import read_only
from ._records import edge_record, node_record
from ._predicate import describe, is_missing, column_mask, parse_conditions
from ._stored_kinds import STORED_EDGE_KIND

_MISSING = object()


class ReadOnlyViewError(TypeError):
    """Raised by every write route of a view, its selections and its attributes."""


def refuse_write(what: str = 'a view') -> ReadOnlyViewError:
    """The one message every refused write of a view carries."""
    return ReadOnlyViewError(
        f'{what} is read-only; call .materialize() for an independent graph you can edit, '
        f'or write through the parent graph.'
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class Leaf(NamedTuple):
    """One evaluable step of a plan, bound to the scope it was made in.

    ``kind`` is one of ``'all'`` (every element of the owner's axis), ``'ids'``
    (a fixed selector, intersected with what exists), ``'where'`` (conditions,
    live), ``'callable'`` (an id predicate, live and never cached) or
    ``'project'`` (the projection of attribute rows onto an axis, live).
    ``owner`` is the graph, view or attribute reader the leaf is evaluated in.
    """

    kind: str
    payload: Any
    owner: Any


class Combine(NamedTuple):
    """Two plans joined by ``'and'``, ``'or'`` or ``'sub'``."""

    op: str
    left: Any
    right: Any


def owner_is_live(owner) -> bool:
    """Whether reading through ``owner`` resolves again on every read.

    A view built on a callable filter does: the callable can depend on state the
    graph does not own, so no clock of the graph says when its answer changed.
    Whatever is selected inside such a view inherits that.
    """
    probe = getattr(owner, '_is_live', None)
    return bool(probe()) if callable(probe) else False


def plan_is_live(plan) -> bool:
    """Whether a plan must be evaluated again on every read, and never cached.

    True when any leaf holds an id predicate, is evaluated inside a live view, or
    projects rows read through one. A plan made of conditions on a graph or a
    plain view depends on the graph alone, and the clocks of the graph say when
    it changed.
    """
    if isinstance(plan, Leaf):
        if plan.kind == 'callable' or owner_is_live(plan.owner):
            return True
        if plan.kind == 'project':
            rows, _target = plan.payload
            return plan_is_live(rows._plan)
        return False
    return plan_is_live(plan.left) or plan_is_live(plan.right)


def _holds_callable(plan) -> bool:
    """Whether a plan holds an id predicate, which is not called just to check a selection."""
    if isinstance(plan, Leaf):
        return plan.kind == 'callable'
    return _holds_callable(plan.left) or _holds_callable(plan.right)


def describe_plan(plan) -> str:
    """One readable line for a repr."""
    if isinstance(plan, Leaf):
        kind, payload = plan.kind, plan.payload
        if kind == 'all':
            return 'all'
        if kind == 'ids':
            ids = sorted(payload, key=repr)
            return f'{len(ids)} id(s)' if len(ids) > 6 else f'ids {ids!r}'
        if kind == 'where':
            return describe(payload) or 'all'
        if kind == 'callable':
            return f'<{getattr(payload, "__name__", "callable")}>'
        if kind == 'project':
            rows, target = payload
            return f'{rows.address}→{target}'
        return kind
    symbol = {'and': '&', 'or': '|', 'sub': '-'}[plan.op]
    return f'({describe_plan(plan.left)}) {symbol} ({describe_plan(plan.right)})'


def call_predicate(fn, element):
    """Call an id predicate, attaching the offending id to whatever it raises."""
    try:
        return bool(fn(element))
    except Exception as exc:
        raise type(exc)(f'{exc} (while evaluating the predicate on {element!r})') from exc


def as_mask(value):
    """Return ``value`` as a boolean numpy array when it is a mask, else None."""
    if isinstance(value, np.ndarray):
        return value if value.dtype == bool else None
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, (bool, np.bool_)) for v in value)
    ):
        return np.asarray(value, dtype=bool)
    return None


def _context(owner):
    """``(root_graph, resolved_or_None)`` of an axis owner."""
    return owner._selection_context()


def _root_of(owner):
    return _context(owner)[0]


# ---------------------------------------------------------------------------
# The sequences
# ---------------------------------------------------------------------------


def _cached_axis_ids(graph, axis: str, build) -> tuple:
    """The ids of one whole axis, walked once per structural version.

    Every leaf of a selection asks for the axis it evaluates over, so a
    composed selection asked the graph to walk its slots once per leaf. The
    walk depends on the structure alone, and the structure has a clock.
    """
    store = graph._store
    stamp = (id(store), store.structure_version)  # a copied graph has a new store
    cache = graph.__dict__.get('_axis_ids_cache')
    if cache is None:
        cache = graph.__dict__['_axis_ids_cache'] = {}
    held = cache.get(axis)
    if held is not None and held[0] == stamp:
        return held[1]
    ids = tuple(build(graph))
    cache[axis] = (stamp, ids)
    return ids


class ElementSequence:
    """One axis of a graph or of a view, read as a sequence and filtered as a selection.

    Three kinds of key: an integer is a position, a slice is a range of
    positions, a string is an attribute name and reads the column. A boolean
    mask over the whole axis selects positions and binds to the ids at those
    positions now.

    A sequence built by ``select`` carries a plan and is live: ``.ids``
    re-evaluates it against the graph as it is now, in the graph's order.
    Explicit ids and masks are fixed selectors; the ids they name are kept as
    long as they exist and are never retargeted.
    """

    axis = 'element'
    id_key = 'id'
    id_column = 'id'
    intrinsic_names: tuple[str, ...] = ('id',)

    __slots__ = ('_owner', '_plan', '_cache')

    def __init__(self, owner, plan=None):
        self._owner = owner
        self._plan = plan
        self._cache = None

    # -- identity -----------------------------------------------------------

    @property
    def graph(self):
        """The root graph this selection reads."""
        return _root_of(self._owner)

    @property
    def _graph(self):
        return _root_of(self._owner)

    @property
    def _scope(self):
        return _context(self._owner)[1]

    def _read_only(self) -> bool:
        return self._scope is not None

    def _subsequence(self, plan, owner=None):
        return type(self)(self._owner if owner is None else owner, plan)

    def _selection_context(self):
        """A sequence may itself own a narrower sequence (``s.select(...)``)."""
        return _context(self._owner)

    def _is_live(self) -> bool:
        """Whether the ids of this sequence must be worked out again on every read."""
        if self._plan is None:
            return owner_is_live(self._owner)
        return plan_is_live(self._plan)

    # -- the ids ------------------------------------------------------------

    def _axis_ids(self, owner) -> tuple:
        """Every id of the axis of ``owner``, in the graph's order."""
        raise NotImplementedError

    def _count_all(self, owner) -> int:
        raise NotImplementedError

    def _all_ids(self) -> tuple:
        return self._axis_ids(self._owner)

    @property
    def ids(self) -> tuple:
        """The ids of this sequence, in the order the graph holds them."""
        plan = self._plan
        if plan is None:
            return self._all_ids()
        if plan_is_live(plan):
            return tuple(self._resolve(plan))
        clock = self._graph._state_clock()
        held = self._cache
        if held is not None and held[0] == clock:
            return held[1]
        ids = tuple(self._resolve(plan))
        self._cache = (clock, ids)
        return ids

    def _resolve(self, plan) -> list:
        if isinstance(plan, Leaf):
            return self._resolve_leaf(plan)
        left = self._resolve(plan.left)
        right = set(self._resolve(plan.right))
        if plan.op == 'and':
            return [item for item in left if item in right]
        if plan.op == 'sub':
            return [item for item in left if item not in right]
        chosen = set(left) | right
        # A union is ordered by the axis of the owner of the combined
        # selection: the root graph when the two sides came from different
        # scopes, so nothing is dropped and the order is the graph's.
        return [item for item in self._axis_ids(self._owner) if item in chosen]

    def _resolve_leaf(self, leaf: Leaf) -> list:
        kind, payload, owner = leaf
        axis = self._axis_ids(owner)
        if kind == 'all':
            return list(axis)
        if kind == 'ids':
            return [item for item in axis if item in payload]
        if kind == 'where':
            return type(self)(owner)._matches(payload)
        if kind == 'callable':
            return [item for item in axis if call_predicate(payload, item)]
        if kind == 'project':
            rows, target = payload
            wanted = set(rows._reader.project_keys(rows, target))
            return [item for item in axis if item in wanted]
        raise ValueError(f'unknown plan leaf {kind!r}')

    def __len__(self) -> int:
        if self._plan is None:
            return self._count_all(self._owner)
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, item) -> bool:
        if self._plan is None:
            return self._holds(item)
        return item in self.ids

    def _holds(self, item) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        plan = 'all' if self._plan is None else describe_plan(self._plan)
        return f'<{type(self).__name__} of {len(self)}: {plan}>'

    def __eq__(self, other):
        if isinstance(other, ElementSequence):
            return self.ids == other.ids and self._graph is other._graph
        return NotImplemented

    __hash__ = None  # type: ignore[assignment]

    # -- the keys ------------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.column(key)
        if isinstance(key, slice):
            return self._subsequence(Leaf('ids', frozenset(self.ids[key]), self._owner))
        if isinstance(key, (int, np.integer)) and not isinstance(key, (bool, np.bool_)):
            return self.ids[key]
        mask = as_mask(key)
        if mask is not None:
            return self._masked(mask)
        raise TypeError(
            f'a sequence key is an attribute name, a position, a range of positions or a '
            f'boolean mask, not {type(key).__name__}'
        )

    def _masked(self, mask: np.ndarray):
        ids = self.ids
        if mask.shape != (len(ids),):
            raise ValueError(
                f'a mask of {len(ids)} booleans is needed for this sequence, got {mask.shape[0]}'
            )
        chosen = frozenset(item for item, keep in zip(ids, mask, strict=True) if keep)
        return self._subsequence(Leaf('ids', chosen, self._owner))

    def __setitem__(self, key, values):
        if not isinstance(key, str):
            raise TypeError('only an attribute column can be assigned to a sequence')
        self.set_column(key, values)

    # -- the columns ------------------------------------------------------------

    def _attribute_map(self, name: str) -> dict | None:
        raise NotImplementedError

    def _attribute_vector(self, name: str):
        raise NotImplementedError

    def _intrinsic(self, name: str, ids):
        if name in (self.id_key, self.id_column):
            return list(ids)
        return _MISSING

    def _intrinsic_vector(self, name: str):
        return None

    def _fields(self) -> set:
        """Every field a condition may name on this axis."""
        raise NotImplementedError

    def column(self, name: str, default=None):
        """Return one attribute of every element of this sequence, as a vector.

        A read of the whole axis of a graph is a slice of the array the store
        holds. A selection, a view, and a caller who names a default for the
        elements that carry none, read element by element. **Every path gives
        back a read-only array**; a caller who means to change values copies.
        """
        whole = self._plan is None and self._scope is None
        if whole and default is None:
            vector = (
                self._intrinsic_vector(name)
                if name in self.intrinsic_names
                else self._attribute_vector(name)
            )
            if vector is not None:
                return read_only(vector)
        ids = self.ids
        found = self._intrinsic(name, ids)
        if found is not _MISSING:
            if isinstance(found, np.ndarray):
                return read_only(found)
            return read_only(np.array(found, dtype=object if not found else None))
        values = self._attribute_map(name)
        if values is None:
            raise KeyError(f'no attribute named {name!r} on this sequence')
        if default is None:
            return read_only(np.array([values.get(element) for element in ids], dtype=object))
        return read_only(
            np.array(
                [
                    default if is_missing(values.get(element)) else values[element]
                    for element in ids
                ],
                dtype=object,
            )
        )

    def set_column(self, name: str, values) -> None:
        """Set one attribute of every element of this sequence, all or nothing.

        The target ids are resolved once, the whole update is checked, and only
        then written: a value that changes a field the selection reads cannot
        retarget the remaining writes.
        """
        if self._read_only():
            raise refuse_write('a view')
        ids = self.ids
        if isinstance(values, (str, bytes)) or not hasattr(values, '__len__'):
            values = [values] * len(ids)
        values = list(values)
        if len(values) != len(ids):
            raise ValueError(f'a column of {len(ids)} values is needed, {len(values)} were given')
        self._write_column(name, dict(zip(ids, values, strict=True)))

    def _write_column(self, name: str, values: dict) -> None:
        raise NotImplementedError

    # -- selecting ----------------------------------------------------------------

    def _matches(self, conditions) -> list:
        """Evaluate conditions over the whole axis of this sequence's owner."""
        ids = self._all_ids()
        if not conditions:
            return list(ids)
        whole = self if self._plan is None else self._subsequence(None)
        keep = np.ones(len(ids), dtype=bool)
        for condition in conditions:
            column = whole.column(condition.field)
            keep &= column_mask(condition, column)
            if not keep.any():
                break
        return [element for element, kept in zip(ids, keep, strict=True) if kept]

    def select(self, conditions=None, /, **keywords):
        """Return the selection of this sequence that matches every condition.

        Parameters
        ----------
        conditions : Mapping | Iterable[id] | id | mask | callable, optional
            A mapping of conditions (with ``(field, operator)`` keys where a
            keyword would be ambiguous); or explicit ids — one, or many — which
            are checked to exist and kept as a fixed selector; or a boolean
            mask over this sequence; or an id predicate, the advanced escape
            hatch, whose exceptions propagate.
        **keywords
            ``field=value`` or ``field__operator=value``; see
            :mod:`annnet.core._predicate`. Keywords combine with AND.

        Returns
        -------
        NodeSequence | EdgeSequence
            Live, ordered, and combinable with ``&``, ``|`` and ``-``.
        """
        base = self._plan
        if conditions is None or isinstance(conditions, Mapping):
            parsed = parse_conditions(conditions, fields=self._fields(), **keywords)
            if not parsed:
                return self._subsequence(base)
            leaf = Leaf('where', parsed, self._owner)
        elif keywords:
            raise TypeError('give either conditions or explicit ids, not both')
        elif callable(conditions):
            leaf = Leaf('callable', conditions, self._owner)
        else:
            mask = as_mask(conditions)
            if mask is not None:
                return self._masked(mask)
            leaf = Leaf('ids', self._checked_ids(conditions), self._owner)
        found = self._subsequence(leaf if base is None else Combine('and', base, leaf))
        if not _holds_callable(found._plan):
            # Resolve once now: an unknown value or an incomparable pair is
            # reported at the call, and the resolution is cached for the read.
            _ = found.ids
        return found

    def _checked_ids(self, items) -> frozenset:
        if isinstance(items, (str, bytes)):
            items = [items]
        elif isinstance(items, ElementSequence):
            raise TypeError('combine selections with &, | and - rather than passing one to select')
        found = []
        for item in items:
            if not self._exists(item):
                raise KeyError(f'unknown {self.axis} id {item!r}')
            found.append(item)
        return frozenset(found)

    def _exists(self, item) -> bool:
        raise NotImplementedError

    def find(self, **conditions):
        """Return the one element that matches every condition.

        A filter that matches nothing, and a filter that matches more than one
        element, are both errors. A caller that wants either of those wants
        :meth:`select`.
        """
        if not conditions:
            raise TypeError('find needs at least one condition')
        matched = self.select(**conditions).ids
        if not matched:
            raise KeyError(f'nothing matches {conditions!r}')
        if len(matched) > 1:
            raise ValueError(f'{len(matched)} elements match {conditions!r}, expected one')
        return matched[0]

    # -- algebra ----------------------------------------------------------------------

    def _as_plan(self):
        return Leaf('all', None, self._owner) if self._plan is None else self._plan

    def _combine(self, other, how: str):
        if not isinstance(other, ElementSequence):
            return NotImplemented
        if type(other) is not type(self):
            raise TypeError(
                f'cannot combine a {self.axis} selection with a{"n" if other.axis[0] in "aeiou" else ""} '
                f'{other.axis} selection'
            )
        if other._graph is not self._graph:
            raise ValueError('cannot combine selections of two different graphs')
        # The combined selection lives in the scope both operands share, or in
        # the root graph when they come from two different scopes. Each leaf
        # still resolves inside the scope it was made in.
        owner = self._owner if other._owner is self._owner else self._graph
        return self._subsequence(Combine(how, self._as_plan(), other._as_plan()), owner=owner)

    def __and__(self, other):
        return self._combine(other, 'and')

    def __or__(self, other):
        return self._combine(other, 'or')

    def __sub__(self, other):
        return self._combine(other, 'sub')


class NodeSequence(ElementSequence):
    """The nodes of a graph or a view, in the order the graph holds them."""

    axis = 'node'
    id_key = 'id'
    id_column = 'node_id'
    intrinsic_names = ('id', 'node_id')

    __slots__ = ()

    def _axis_ids(self, owner) -> tuple:
        graph, resolved = _context(owner)
        if resolved is not None:
            return resolved.node_ids
        return _cached_axis_ids(graph, 'node', _structure.node_ids)

    def _count_all(self, owner) -> int:
        graph, resolved = _context(owner)
        if resolved is not None:
            return len(resolved.node_ids)
        return _structure.distinct_node_count(graph)

    def _holds(self, item) -> bool:
        graph, resolved = _context(self._owner)
        if resolved is not None:
            return item in resolved.node_set
        return isinstance(item, str) and graph.has_node(item)

    def _exists(self, item) -> bool:
        return isinstance(item, str) and self._graph.has_node(item)

    def _fields(self) -> set:
        return set(self.intrinsic_names) | set(self._graph._attr_store.node_column_names())

    def _attribute_map(self, name: str) -> dict | None:
        return self._graph._attr_store.node_attr_map(name)

    def _attribute_vector(self, name: str):
        return self._graph._attr_store.node_vector(name)

    def _write_column(self, name: str, values: dict) -> None:
        if name in self.intrinsic_names:
            raise KeyError(
                f'{name!r} is the id of a node, not an attribute of one. '
                'Renaming a node is a structural change, not a column write.'
            )
        self._graph.attrs.update(
            'nodes', {element: {name: value} for element, value in values.items()}
        )

    def at(self, node_id: str):
        """Return the record of one node by its id.

        A string-shaped record equal to the id, carrying ``kind``, ``layers``
        and a detached ``attrs`` dict. In a view, ``layers`` are the placements
        the view holds.
        """
        if not isinstance(node_id, str):
            raise TypeError(
                f'at takes a node id, not {type(node_id).__name__}. For the node on a '
                f'matrix row, use G.idx.row_to_entity(row).'
            )
        if self._plan is not None and node_id not in self.ids:
            raise KeyError(f'{node_id!r} is not in this selection')
        graph, resolved = _context(self._owner)
        if resolved is not None:
            if node_id not in resolved.node_set:
                raise KeyError(f'{node_id!r} is not in this view')
            layers = tuple(layer for _id, layer in resolved.node_keys if _id == node_id)
            return node_record(graph, node_id, layers=layers)
        return node_record(graph, node_id)


# The two intrinsic edge fields a caller may write. ``kind`` and ``ml_kind``
# follow from the shape and placement of the edge.
_EDGE_STRUCTURAL_WRITES = frozenset({'weight', 'directed'})
_EDGE_INTRINSIC_COLUMNS = ('directed', 'weight', 'kind', 'ml_kind')

STORED_EDGE_KIND_NAMES = tuple(
    STORED_EDGE_KIND[_structure._SLOT_EDGE_KIND[code]]
    for code in sorted(_structure._SLOT_EDGE_KIND)
)


class EdgeSequence(ElementSequence):
    """The edges of a graph or a view, in the order the graph holds them.

    Four structural fields read like columns because a filter over them is as
    common as one over an attribute: ``kind`` (``binary``, ``hyper`` or
    ``node_edge``), ``directed``, ``weight`` and ``ml_kind``.
    """

    axis = 'edge'
    id_key = 'id'
    id_column = 'edge_id'
    intrinsic_names = ('id', 'edge_id', *_EDGE_INTRINSIC_COLUMNS)

    __slots__ = ()

    def _axis_ids(self, owner) -> tuple:
        graph, resolved = _context(owner)
        if resolved is not None:
            return resolved.edge_ids
        return _cached_axis_ids(graph, 'edge', _structure.edge_ids)

    def _count_all(self, owner) -> int:
        graph, resolved = _context(owner)
        if resolved is not None:
            return len(resolved.edge_ids)
        return _structure.edge_count(graph)

    def _holds(self, item) -> bool:
        graph, resolved = _context(self._owner)
        if resolved is not None:
            return item in resolved.edge_set
        return (
            isinstance(item, str)
            and _structure.has_edge(graph, item)
            and _structure.carries_structure(graph, item)
        )

    def _exists(self, item) -> bool:
        return isinstance(item, str) and _structure.has_edge(self._graph, item)

    def _fields(self) -> set:
        return set(self.intrinsic_names) | set(self._graph._attr_store.edge_column_names())

    def _intrinsic(self, name: str, ids):
        if name in (self.id_key, self.id_column):
            return list(ids)
        if name in _EDGE_INTRINSIC_COLUMNS:
            graph = self._graph
            refs = [_structure.edge_ref(graph, element) for element in ids]
            if name == 'kind':
                return [STORED_EDGE_KIND[ref.kind] for ref in refs]
            return [getattr(ref, name) for ref in refs]
        return _MISSING

    def _intrinsic_vector(self, name: str):
        """Return the whole intrinsic column from the edge arrays, or None."""
        if name not in _EDGE_INTRINSIC_COLUMNS:
            return None
        store = self._graph._store
        if not store.edge_axis_contiguous:
            return None
        count = store.edge_count
        if name == 'weight':
            return store.edge_weight[:count]
        if name == 'directed':
            return store.edge_directed_column()[:count]
        if name == 'kind':
            return store.edge_kind_column(STORED_EDGE_KIND_NAMES)[:count]
        return np.array([store.edge_ml_kind_of(slot) for slot in range(count)], dtype=object)

    def _attribute_map(self, name: str) -> dict | None:
        return self._graph._attr_store.edge_attr_map(name)

    def _attribute_vector(self, name: str):
        return self._graph._attr_store.edge_vector(name)

    def _write_column(self, name: str, values: dict) -> None:
        graph = self._graph
        if name in _EDGE_STRUCTURAL_WRITES:
            from . import _mutate

            for element, value in values.items():
                _mutate.set_edge_field(graph, element, name, value)
            graph._mark_structure_changed()
            return
        if name in self.intrinsic_names:
            raise KeyError(
                f'{name!r} follows from the shape of an edge, so it cannot be written. '
                'Set the members of the edge instead.'
            )
        graph.attrs.update('edges', {element: {name: value} for element, value in values.items()})

    def at(self, edge_id: str):
        """Return the record of one edge by its id.

        A tuple-shaped record: ``(source, target)`` unpacks; ``edge_id``,
        ``kind``, ``members``, ``weight`` and ``directed`` are attributes.
        """
        if not isinstance(edge_id, str):
            raise TypeError(
                f'at takes an edge id, not {type(edge_id).__name__}. For the edge on a '
                f'matrix column, use G.idx.col_to_edge(column).'
            )
        graph, resolved = _context(self._owner)
        if not _structure.has_edge(graph, edge_id):
            raise KeyError(f'Unknown edge id: {edge_id}')
        if self._plan is not None and edge_id not in self.ids:
            raise KeyError(f'{edge_id!r} is not in this selection')
        if resolved is not None and edge_id not in resolved.edge_set:
            raise KeyError(f'{edge_id!r} is not in this view')
        return edge_record(graph, edge_id)

    def effective_weight(self, edge_id: str, slice: str | None = None) -> float:
        """Return the weight of one edge in one slice: the override, or the stored weight.

        Parameters
        ----------
        edge_id : str
        slice : str, optional
            Defaults to the graph's active slice.
        """
        graph = self._graph
        if slice is None:
            slice = graph._current_slice
        held = graph._contextual.edge_slice_attrs.get((slice, edge_id))
        if held:
            weight = held.get('weight')
            if weight is not None and not (isinstance(weight, float) and weight != weight):
                return float(weight)
        if not _structure.has_edge(graph, edge_id):
            return 1.0
        return float(_structure.edge_ref(graph, edge_id).weight)


# ---------------------------------------------------------------------------
# Attribute-row selections
# ---------------------------------------------------------------------------


class RowSelection:
    """The rows of one attribute address that satisfy a query, keyed as the address keys them.

    Built by ``G.attrs.select`` / ``V.attrs.select``. It keeps the full keys —
    a node-layer row keeps its ``(node_id, layer)`` — until :meth:`project` is
    called. It is live: ``.keys`` re-evaluates the query against the graph as
    it is now, each leaf inside the scope it was asked in. Two selections of
    the same address on the same graph combine with ``&``, ``|`` and ``-``;
    the result keeps the address's own order.
    """

    __slots__ = ('_reader', '_address', '_plan')

    def __init__(self, reader, address, plan):
        self._reader = reader
        self._address = address
        self._plan = plan

    @property
    def address(self) -> str:
        return self._address

    @property
    def graph(self):
        return self._reader._G

    @property
    def keys(self) -> tuple:
        """The keys of the matching rows, in the address's own order."""
        return tuple(self._resolve(self._plan))

    @property
    def ids(self) -> tuple:
        """Alias of :attr:`keys`, so every selection answers ``.ids``."""
        return self.keys

    def _resolve(self, plan) -> list:
        if isinstance(plan, Leaf):
            reader = plan.owner
            if plan.kind == 'where':
                return reader._rows_where(self._address, plan.payload)
            if plan.kind == 'all':
                return list(reader._domain(self._address))
            if plan.kind == 'ids':
                # Fixed identities, taken in the address's own order and only
                # while the graph still holds them. Nothing here reads a
                # storage slot, so a key that is removed and added again is the
                # same key, and no other key is ever picked up in its place.
                fixed = plan.payload
                return [key for key in reader._domain(self._address) if key in fixed]
            raise ValueError(f'unknown row plan leaf {plan.kind!r}')
        left = self._resolve(plan.left)
        right = set(self._resolve(plan.right))
        if plan.op == 'and':
            return [key for key in left if key in right]
        if plan.op == 'sub':
            return [key for key in left if key not in right]
        chosen = set(left) | right
        return [key for key in self._reader._domain(self._address) if key in chosen]

    def __iter__(self):
        return iter(self.keys)

    def __len__(self) -> int:
        return len(self.keys)

    def __contains__(self, key) -> bool:
        try:
            key = self._reader._key(self._address, key)
        except (KeyError, ValueError, TypeError):
            return False
        return key in set(self.keys)

    def rows(self) -> dict:
        """The matching rows as detached dictionaries, keyed by their keys."""
        return {key: self._reader._row(self._address, key) for key in self.keys}

    def _combine(self, other, how: str):
        if not isinstance(other, RowSelection):
            return NotImplemented
        if other.graph is not self.graph:
            raise ValueError('cannot combine selections of two different graphs')
        if other._address != self._address:
            raise TypeError(
                f'cannot combine {self._address} rows with {other._address} rows; '
                f'project both to a shared axis first'
            )
        # The combination is ordered by the reader both sides share, or by the
        # root graph's reader when they came from two different scopes.
        reader = self._reader if other._reader is self._reader else self._reader._root_reader()
        return RowSelection(reader, self._address, Combine(how, self._plan, other._plan))

    def __and__(self, other):
        return self._combine(other, 'and')

    def __or__(self, other):
        return self._combine(other, 'or')

    def __sub__(self, other):
        return self._combine(other, 'sub')

    def project(self, target: str):
        """Project the selected rows onto one axis of their keys.

        ``node_layers`` rows project to ``nodes``; ``edge_slices`` rows project
        to ``edges`` or ``slices``. The projection is existential — an element
        is selected when at least one matching row names it — and it keeps the
        target axis's order with duplicates removed. It does not say "in every
        slice"; that is a reduction over the frame, not a selection.

        Returns
        -------
        NodeSequence | EdgeSequence | SliceSelection
            A live selection of the target axis, in the scope of this selection.
        """
        allowed = {
            'node_layers': ('nodes',),
            'edge_slices': ('edges', 'slices'),
            'nodes': ('nodes',),
            'edges': ('edges',),
            'slices': ('slices',),
        }.get(self._address, ())
        if target not in allowed:
            raise ValueError(
                f'{self._address!r} rows project to {list(allowed) or "nothing"}, not {target!r}'
            )
        owner = self._reader._selection_owner()
        if target == 'slices':
            return SliceSelection(self._reader, self)
        sequence = NodeSequence if target == 'nodes' else EdgeSequence
        return sequence(owner, Leaf('project', (self, target), owner))

    def __repr__(self):
        return f'RowSelection({self._address!r}, {len(self)} row(s): {describe_plan(self._plan)})'


class SliceSelection:
    """A live selection of slice ids, the result of projecting rows onto slices."""

    __slots__ = ('_reader', '_rows')

    def __init__(self, reader, rows: RowSelection):
        self._reader = reader
        self._rows = rows

    @property
    def ids(self) -> tuple:
        found = {key[0] for key in self._rows.keys}
        return tuple(sid for sid in self._reader._domain('slices') if sid in found)

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __contains__(self, item):
        return item in self.ids

    def __repr__(self):
        return f'SliceSelection({list(self.ids)!r})'


def project_keys(reader, rows: RowSelection, target: str) -> list:
    """The ids of ``target`` named by ``rows``, deduplicated in the reader's axis order."""
    keys = rows.keys
    address = rows.address
    if address == 'node_layers':
        found = {key[0] for key in keys}
    elif address == 'edge_slices':
        found = {key[0] if target == 'slices' else key[1] for key in keys}
    else:
        found = set(keys)
    return [item for item in reader._domain(target) if item in found]


__all__ = [
    'Combine',
    'EdgeSequence',
    'ElementSequence',
    'Leaf',
    'NodeSequence',
    'ReadOnlyViewError',
    'RowSelection',
    'SliceSelection',
    'as_mask',
    'call_predicate',
    'describe_plan',
    'owner_is_live',
    'plan_is_live',
    'project_keys',
    'refuse_write',
]
