"""The slice registry, the public edge view, and the reserved attribute names.

What is left here after the canonical store took over: a slice is still a
membership record, ``EdgeView`` is the tuple shape ``AnnNet.get_edge`` returns,
and the reserved sets say which attribute names the structural columns own.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, NamedTuple
from dataclasses import field, dataclass

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


@dataclass(slots=True)
class SliceRecord:
    """Typed slice membership record with dict-style compatibility."""

    nodes: set = field(default_factory=set)
    edges: set = field(default_factory=set)
    attributes: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        """Return a slice field by name with an optional default."""
        return getattr(self, key, default)


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
    """Tuple-shaped edge record returned by :meth:`AnnNet.get_edge`.

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
    """String-shaped node record returned by :meth:`AnnNet.get_node`.

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
