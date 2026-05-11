from __future__ import annotations

from enum import Enum
from dataclasses import field, dataclass

import narwhals as nw

from .._dataframe_backend import dataframe_filter_ne, dataframe_from_rows


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
    return dataframe_from_rows(rows)


def _df_filter_not_equal(df, col: str, value):
    return dataframe_filter_ne(df, col, value)


class EdgeType(Enum):
    DIRECTED = 'DIRECTED'
    UNDIRECTED = 'UNDIRECTED'


@dataclass(slots=True)
class EntityRecord:
    """One record per entity (vertex or edge-entity) with an incidence-matrix row."""

    row_idx: int  # row index in the incidence matrix
    kind: str  # "vertex" | "edge_entity"


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
        return getattr(self, key, default)


@dataclass(slots=True)
class EdgeRecord:
    """One record per edge (binary, hyperedge, structural edge-entity, or placeholder).

    Topology encoding
    -----------------
    binary edge          : src=str,       tgt=str,       directed=bool|None
    undirected hyperedge : src=frozenset, tgt=None,      directed=False
    directed hyperedge   : src=frozenset, tgt=frozenset, directed=True
    structural edge-entity : src=str,       tgt=str,       etype="vertex_edge"
    edge placeholder       : src=None,      tgt=None,      etype="edge_placeholder"
    """

    src: object  # str (binary) | frozenset (hyper) | None
    tgt: object  # str (binary) | frozenset (directed hyper) | None
    weight: float
    directed: object  # bool | None — None inherits graph default
    etype: str  # "binary" | "hyper" | "vertex_edge" | "edge_placeholder"
    col_idx: int  # column index in the incidence matrix (-1 = no column)
    ml_kind: object  # str | None — "intra" | "inter" | "coupling"
    ml_layers: object  # tuple | None — multilayer layer assignment
    direction_policy: object  # dict | None


class EdgeView(tuple):
    """Edge view returned by :meth:`AnnNet.get_edge`.

    Iteration / unpacking yields ``(source, target)`` for backward compat
    with code that already does ``S, T = G.get_edge(j)``. Additional fields
    are exposed as attributes:

    - ``edge_id`` (str)
    - ``kind`` ("binary" | "hyper_undirected" | "hyper_directed" |
      "vertex_edge" | "edge_placeholder")
    - ``source`` / ``target`` (same as tuple positions 0 / 1)
    - ``members`` (frozenset of every entity incident to the edge)
    - ``weight`` (float)
    - ``directed`` (bool)
    """

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
