"""The slice registry, the public records, and the reserved attribute names.

A slice is a membership record, :class:`EdgeView` and :class:`NodeView` are the
records ``G.E.at`` and ``G.N.at`` hand back, :func:`edge_record` and
:func:`node_record` build them from the structural facade, and the reserved
sets say which attribute names the structural columns own.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple

import narwhals as nw

from .._support.dataframe_backend import dataframe_filter_ne, dataframe_from_rows


def _get_numeric_supertype(left, right):
    left_cls = left.base_type() if hasattr(left, 'base_type') else left
    right_cls = right.base_type() if hasattr(right, 'base_type') else right

    if left_cls.is_float() or right_cls.is_float():
        if left_cls == nw.Float64 or right_cls == nw.Float64:
            return nw.Float64
        return nw.Float32

    type_order = {
        nw.Int8: 1,
        nw.Int16: 2,
        nw.Int32: 3,
        nw.Int64: 4,
        nw.Int128: 5,
        nw.UInt8: 1,
        nw.UInt16: 2,
        nw.UInt32: 3,
        nw.UInt64: 4,
        nw.UInt128: 5,
    }
    left_unsigned = left_cls.is_unsigned_integer()
    right_unsigned = right_cls.is_unsigned_integer()
    if left_unsigned != right_unsigned:
        return nw.Float64
    return left_cls if type_order.get(left_cls, 0) >= type_order.get(right_cls, 0) else right_cls


def build_dataframe_from_rows(rows):
    """Build a dataframe from a sequence of row dictionaries."""
    return dataframe_from_rows(rows)


def _df_filter_not_equal(df, col: str, value):
    return dataframe_filter_ne(df, col, value)


class EdgeType(Enum):
    DIRECTED = 'DIRECTED'
    UNDIRECTED = 'UNDIRECTED'


class Clock:
    """One mutable counter, shared by everything that has to tick it.

    A membership set and the registry that holds it both tick the same clock,
    so a reader that cached a slice-dependent answer records one number and
    compares one number.
    """

    __slots__ = ('value',)

    def __init__(self) -> None:
        self.value = 0

    def tick(self) -> None:
        self.value += 1


class ClockedSet(set):
    """A set that ticks a :class:`Clock` on every mutation.

    Slice membership is written from many places — the slice manager, the
    mutation gateway, the loaders — and each of them writes the set directly.
    Ticking here, rather than at every call site, is what makes a slice clock
    complete: a write that reaches the set reaches the clock.
    """

    __slots__ = ('_clock',)

    def __init__(self, items=(), clock=None):
        super().__init__(items)
        self._clock = clock

    def _tick(self) -> None:
        clock = self._clock
        if clock is not None:
            clock.value += 1

    def add(self, item):
        super().add(item)
        self._tick()

    def discard(self, item):
        super().discard(item)
        self._tick()

    def remove(self, item):
        super().remove(item)
        self._tick()

    def pop(self):
        item = super().pop()
        self._tick()
        return item

    def clear(self):
        super().clear()
        self._tick()

    def update(self, *others):
        super().update(*others)
        self._tick()

    def intersection_update(self, *others):
        super().intersection_update(*others)
        self._tick()

    def difference_update(self, *others):
        super().difference_update(*others)
        self._tick()

    def symmetric_difference_update(self, other):
        super().symmetric_difference_update(other)
        self._tick()

    def __ior__(self, other):  # type: ignore[misc]
        super().__ior__(other)
        self._tick()
        return self

    def __iand__(self, other):  # type: ignore[misc]
        super().__iand__(other)
        self._tick()
        return self

    def __isub__(self, other):  # type: ignore[misc]
        super().__isub__(other)
        self._tick()
        return self

    def __ixor__(self, other):  # type: ignore[misc]
        super().__ixor__(other)
        self._tick()
        return self

    def copy(self):
        return set(self)

    def __reduce__(self):
        return (set, (set(self),))


class SliceRecord:
    """Slice membership: the node ids, the edge ids and the attributes.

    ``record['nodes']`` and ``record.nodes`` are the same set. Once the record
    sits in a graph's registry the two sets tick the graph's slice clock, and
    assigning a new set to either field wraps it so that it does too.
    """

    __slots__ = ('_nodes', '_edges', 'attributes', '_clock')

    def __init__(self, nodes=(), edges=(), attributes=None, *, clock=None):
        self._clock = clock
        self._nodes = ClockedSet(nodes, clock)
        self._edges = ClockedSet(edges, clock)
        self.attributes = {} if attributes is None else dict(attributes)

    def _bind(self, clock) -> None:
        """Make this record tick ``clock`` from now on."""
        self._clock = clock
        self._nodes._clock = clock
        self._edges._clock = clock
        if clock is not None:
            clock.value += 1

    @property
    def nodes(self) -> ClockedSet:
        return self._nodes

    @nodes.setter
    def nodes(self, value) -> None:
        self._nodes = ClockedSet(value, self._clock)
        if self._clock is not None:
            self._clock.value += 1

    @property
    def edges(self) -> ClockedSet:
        return self._edges

    @edges.setter
    def edges(self, value) -> None:
        self._edges = ClockedSet(value, self._clock)
        if self._clock is not None:
            self._clock.value += 1

    def __getitem__(self, key):
        if key in ('nodes', 'edges', 'attributes'):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key, value):
        if key in ('nodes', 'edges', 'attributes'):
            setattr(self, key, value)
            return
        raise KeyError(key)

    def get(self, key, default=None):
        """Return a slice field by name with an optional default."""
        return getattr(self, key, default)

    def __eq__(self, other):
        if isinstance(other, SliceRecord):
            return (
                set(self._nodes) == set(other._nodes)
                and set(self._edges) == set(other._edges)
                and self.attributes == other.attributes
            )
        return NotImplemented

    def __repr__(self) -> str:
        return (
            f'SliceRecord(nodes={set(self._nodes)!r}, edges={set(self._edges)!r}, '
            f'attributes={self.attributes!r})'
        )


class SliceRegistry(dict):
    """The ``slice_id -> SliceRecord`` mapping of one graph, with its clock.

    Every record installed here is bound to the registry's clock, so a
    membership write through any record ticks it, and adding or dropping a
    slice ticks it too. A reader that depends on slice membership records
    ``clock.value`` and compares it later.
    """

    __slots__ = ('clock',)

    def __init__(self, clock=None, mapping=None):
        super().__init__()
        self.clock = Clock() if clock is None else clock
        if mapping:
            for key, record in mapping.items():
                self[key] = record

    @staticmethod
    def _as_record(value) -> SliceRecord:
        if isinstance(value, SliceRecord):
            return value
        if isinstance(value, dict):
            return SliceRecord(
                value.get('nodes', ()), value.get('edges', ()), value.get('attributes', {})
            )
        raise TypeError(f'a slice registry holds SliceRecord values, not {type(value).__name__}')

    def __setitem__(self, key, value):
        record = self._as_record(value)
        record._bind(self.clock)
        super().__setitem__(key, record)

    def __delitem__(self, key):
        super().__delitem__(key)
        self.clock.tick()

    def pop(self, key, *default):
        found = super().pop(key, *default)
        self.clock.tick()
        return found

    def popitem(self):
        found = super().popitem()
        self.clock.tick()
        return found

    def clear(self):
        super().clear()
        self.clock.tick()

    def setdefault(self, key, default=None):
        if key in self:
            return self[key]
        self[key] = SliceRecord() if default is None else default
        return self[key]

    def update(self, *args, **kwargs):
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    def __reduce__(self):
        return (SliceRegistry, (None, dict(self)))


class Endpoint(NamedTuple):
    """One side of one edge: the node, and the layer it sits in.

    The store spells an endpoint two ways — a bare id in a flat graph, an
    ``(id, layer)`` pair in a layered one — so reading one meant asking which it
    was::

        node = next(iter(sides.source))
        node_id = node[0] if isinstance(node, tuple) else node

    That check is a defect rather than an idiom. A graph holding both layered and
    unlayered edges makes it wrong, and nothing reports it. An endpoint read
    through :func:`as_endpoint` has one shape everywhere, and ``layer`` is
    ``None`` when there is not one.

    The positional shape is the store's, so ``endpoint[0]`` is the id and
    ``endpoint[1]`` is the layer. ``str(endpoint)`` is the id, which is what a
    label, a dataframe cell and a join all want.

    Examples
    --------
    >>> as_endpoint(('akt', ('stim',)))
    Endpoint(node_id='akt', layer=('stim',))
    >>> str(as_endpoint('akt'))
    'akt'
    """

    node_id: str
    layer: tuple | None = None

    def __str__(self) -> str:
        return self.node_id

    @property
    def key(self) -> Any:
        """The endpoint as the store spells it: a bare id, or an ``(id, layer)`` pair."""
        return self.node_id if self.layer is None else (self.node_id, self.layer)


def as_endpoint(value) -> Endpoint:
    """Return one stored endpoint as an :class:`Endpoint`.

    Parameters
    ----------
    value : str | tuple[str, tuple[str, ...]] | Endpoint
        An endpoint in any shape the store holds.

    Returns
    -------
    Endpoint
    """
    if isinstance(value, Endpoint):
        return value
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], tuple)
    ):
        return Endpoint(value[0], value[1])
    return Endpoint(str(value), None)


def as_endpoints(side) -> frozenset:
    """Return one side of an edge as a frozenset of :class:`Endpoint`.

    Parameters
    ----------
    side : Iterable
        One side of an edge, as :func:`annnet.core._structure.edge_sides` holds it.

    Returns
    -------
    frozenset[Endpoint]
    """
    return frozenset(as_endpoint(item) for item in side)


def _one_endpoint(side) -> Endpoint | None:
    """The one endpoint of a one-member side, or ``None`` when it is not one."""
    if side is None or len(side) != 1:
        return None
    return as_endpoint(next(iter(side)))


class EdgeView(tuple):
    """Tuple-shaped edge record returned by ``G.E.at``.

    ``source``, ``target`` and ``members`` hold endpoints as the store spells
    them. :func:`as_endpoints` normalises a side; :attr:`source_id`,
    :attr:`target_id` and :attr:`layer` answer the three questions a caller
    almost always has instead.
    """

    edge_id: str
    kind: Any
    source: Any
    target: Any
    members: Any
    weight: float
    directed: bool

    @property
    def source_id(self) -> str | None:
        """The id of the one source, or ``None`` when the side is not one node."""
        found = _one_endpoint(self.source)
        return None if found is None else found.node_id

    @property
    def target_id(self) -> str | None:
        """The id of the one target, or ``None`` when the side is not one node."""
        found = _one_endpoint(self.target)
        return None if found is None else found.node_id

    @property
    def layer(self) -> tuple | None:
        """The layer this edge sits in, or ``None`` when it crosses two or has none."""
        source = _one_endpoint(self.source)
        target = _one_endpoint(self.target)
        if source is None or target is None or source.layer != target.layer:
            return None
        return source.layer

    def __new__(cls, source, target, *, edge_id, kind, members, weight, directed):
        self = super().__new__(cls, (source, target))
        self.edge_id = edge_id
        self.kind = kind
        self.source = source
        self.target = target
        self.members = members
        self.weight = weight
        self.directed = directed
        return self

    def __repr__(self) -> str:
        return (
            f'EdgeView(edge_id={self.edge_id!r}, kind={self.kind!r}, '
            f'source={self.source!r}, target={self.target!r}, '
            f'members={self.members!r}, weight={self.weight!r}, '
            f'directed={self.directed!r})'
        )


class NodeView(str):
    """String-shaped node record returned by ``G.N.at``.

    A node is its id, so this is the id, and everything the graph holds about
    it hangs off that. An edge is a pair, which is why :class:`EdgeView` is a
    tuple and this is a string.
    """

    node_id: str
    kind: Any
    layers: tuple
    attrs: dict

    def __new__(cls, node_id, *, kind, layers, attrs):
        self = super().__new__(cls, node_id)
        self.node_id = node_id
        self.kind = kind
        self.layers = layers
        self.attrs = attrs
        return self

    def __repr__(self) -> str:
        return (
            f'NodeView(node_id={self.node_id!r}, kind={self.kind!r}, '
            f'layers={self.layers!r}, attrs={self.attrs!r})'
        )


def _external_entity_kind(kind: str) -> str:
    return 'edge' if kind == 'edge_entity' else kind


def edge_record(graph, edge_id: str) -> EdgeView:
    """Build the public record of one edge.

    ``kind`` is the structural kind (``binary``, ``hyper``, ``node_edge``) and
    ``directed`` is separate, on every surface. An undirected edge shows the
    same members on both sides, because neither side means a direction.
    """
    from . import _structure
    from ._stored_kinds import STORED_EDGE_KIND

    ref = _structure.edge_ref(graph, edge_id)
    sides = _structure.edge_sides(graph, edge_id)
    members = sides.source | sides.target
    if ref.directed or ref.kind in (_structure.NODE_EDGE, _structure.PLACEHOLDER):
        source, target = sides.source, sides.target
    else:
        source = target = members
    return EdgeView(
        source,
        target,
        edge_id=edge_id,
        kind=STORED_EDGE_KIND[ref.kind],
        members=members,
        weight=ref.weight,
        directed=ref.directed,
    )


def node_record(graph, node_id: str, *, layers=None) -> NodeView:
    """Build the public record of one node.

    ``layers`` defaults to every placement the graph holds for the id; a view
    passes the placements it holds. ``attrs`` is a detached copy.
    """
    from . import _structure
    from ._stored_kinds import STORED_ENTITY_KIND

    if layers is None:
        keys = graph._store.entity_keys_of_id(node_id)
        if not keys:
            raise KeyError(f'Unknown node id: {node_id}')
        layers = tuple(layer for _id, layer in keys)
    elif not layers:
        raise KeyError(f'Unknown node id: {node_id}')
    ref = _structure.entity_ref(graph, (node_id, layers[0]))
    return NodeView(
        node_id,
        kind=_external_entity_kind(STORED_ENTITY_KIND[ref.kind]),
        layers=tuple(layers),
        attrs=graph._attr_store.node_attrs(node_id),
    )


def _internal_entity_kind(kind: str) -> str:
    return 'edge_entity' if kind == 'edge' else kind


_node_RESERVED = {'node_id'}
_EDGE_RESERVED = {
    'edge_id',
    'source',
    'target',
    'weight',
    'edge_type',
    'directed',
    'slice',
    'slice_weight',
    'kind',
    'members',
    'head',
    'tail',
    'flexible',
}
_slice_RESERVED = {'slice_id'}
