"""The slice registry, the public edge view, and the reserved attribute names.

What is left here after the canonical store took over: a slice is still a
membership record, ``EdgeView`` is the tuple shape ``AnnNet.get_edge`` returns,
and the reserved sets say which attribute names the structural columns own.
"""

from __future__ import annotations

from enum import Enum
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

    vertices: set = field(default_factory=set)
    edges: set = field(default_factory=set)
    attributes: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        """Return a slice field by name with an optional default."""
        return getattr(self, key, default)


class EdgeView(tuple):
    """Tuple-shaped edge record returned by :meth:`AnnNet.get_edge`."""

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


class VertexView(str):
    """String-shaped vertex record returned by :meth:`AnnNet.get_vertex`.

    A vertex is its id, so this is the id, and everything the graph holds about
    it hangs off that. An edge is a pair, which is why :class:`EdgeView` is a
    tuple and this is a string.
    """

    def __new__(cls, vertex_id, *, kind, layers, attrs):
        self = super().__new__(cls, vertex_id)
        self.vertex_id = vertex_id
        self.kind = kind
        self.layers = layers
        self.attrs = attrs
        return self

    def __repr__(self) -> str:
        return (
            f'VertexView(vertex_id={self.vertex_id!r}, kind={self.kind!r}, '
            f'layers={self.layers!r}, attrs={self.attrs!r})'
        )


def _external_entity_kind(kind: str) -> str:
    return 'edge' if kind == 'edge_entity' else kind


def _internal_entity_kind(kind: str) -> str:
    return 'edge_entity' if kind == 'edge' else kind


_vertex_RESERVED = {'vertex_id'}
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
