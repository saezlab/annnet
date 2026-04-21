from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from collections.abc import MutableMapping

import narwhals as nw

from .._dataframe_backend import dataframe_from_rows

try:
    import polars as pl
except Exception:
    pl = None


def _get_numeric_supertype(left, right):
    """Get the supertype for two numeric dtypes (Narwhals).

    Returns the wider type that can hold values from both input types.
    For mixed int/float, returns float. For different sizes, returns larger size.
    For mixed signed/unsigned, promotes to Float64 for safety.
    """

    left_cls = left.base_type() if hasattr(left, 'base_type') else left
    right_cls = right.base_type() if hasattr(right, 'base_type') else right

    # If either is float, result is float (wider float wins)
    if left_cls.is_float() or right_cls.is_float():
        if left_cls == nw.Float64 or right_cls == nw.Float64:
            return nw.Float64
        return nw.Float32

    # Both are integers
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
    """Build a DataFrame from a list of row records using available backends.

    Parameters
    ----------
    rows : list[dict] | list[list] | list[tuple]
        Row records to load into a DataFrame.

    Returns
    -------
    DataFrame-like
        DataFrame using AnnNet's selected backend order: Polars, pandas, then
        PyArrow.

    Raises
    ------
    RuntimeError
        If no supported dataframe backend is available.

    Notes
    -----
    Uses the centralized dataframe backend selector.
    """
    return dataframe_from_rows(rows)


def _df_filter_not_equal(df, col: str, value):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.filter(pl.col(col) != value)
    import narwhals as nw

    ndf = nw.from_native(df)
    return nw.to_native(ndf.filter(nw.col(col) != value))


class EdgeType(Enum):
    DIRECTED = 'DIRECTED'
    UNDIRECTED = 'UNDIRECTED'


@dataclass(slots=True)
class EntityRecord:
    """Single source of truth for a vertex or edge-entity.

    Every entity with an incidence-matrix row owns one record.
    Vertices are plain nodes; edge-entities are edges that can also
    be endpoints of other edges.
    """

    row_idx: int  # row index in the incidence matrix
    kind: str  # "vertex" | "edge_entity"


@dataclass(slots=True)
class EdgeRecord:
    """Single source of truth for all edge data.

    Topology encoding
    -----------------
    binary edge          : src=str,       tgt=str,       directed=bool|None
    undirected hyperedge : src=frozenset, tgt=None,      directed=False
    directed hyperedge   : src=frozenset, tgt=frozenset, directed=True
    vertex-edge entity   : src=str|None,  tgt=str|None,  etype="vertex_edge"

    For hyperedges src holds the head-set (directed) or all-members set
    (undirected); tgt holds the tail-set or None.
    """

    src: object  # str (binary) | frozenset (hyper) | None
    tgt: object  # str (binary) | frozenset (directed hyper) | None
    weight: float
    directed: object  # bool | None — None means inherit graph default
    etype: str  # "binary" | "hyper" | "vertex_edge"
    col_idx: int  # column index in the incidence matrix (-1 if no column)
    ml_kind: object  # str | None — "intra" | "inter" | "coupling"
    ml_layers: object  # tuple | None — multilayer layer assignment
    direction_policy: object  # dict | None — flexible direction config


def _external_entity_kind(kind: str) -> str:
    return 'edge' if kind == 'edge_entity' else kind


def _internal_entity_kind(kind: str) -> str:
    return 'edge_entity' if kind == 'edge' else kind


class _CompatMapping(MutableMapping):
    """Write-through MutableMapping view used by SSOT compatibility shims."""

    def __init__(self, graph):
        self._G = graph

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(list(self.keys()))


class EntityToIdxCompat(_CompatMapping):
    def keys(self):
        # Expose bare vid strings (first element of each tuple key) for backward compat
        return [k[0] for k in self._G._entities.keys()]

    def __getitem__(self, key):
        ekey = self._G._resolve_entity_key(key)
        return self._G._entities[ekey].row_idx

    def __setitem__(self, key, value):
        ekey = self._G._resolve_entity_key(key)
        row_idx = int(value)
        rec = self._G._entities.get(ekey)
        kind = rec.kind if rec is not None else 'vertex'
        self._G._register_entity_record(ekey, EntityRecord(row_idx=row_idx, kind=kind))

    def __delitem__(self, key):
        ekey = self._G._resolve_entity_key(key)
        self._G._remove_entity_record(ekey)


class IdxToEntityCompat(_CompatMapping):
    def keys(self):
        return self._G._row_to_entity.keys()

    def __getitem__(self, key):
        # Return bare vid for backward compat
        entry = self._G._row_to_entity[key]
        return entry[0] if isinstance(entry, tuple) else entry

    def __setitem__(self, key, value):
        row_idx = int(key)
        ekey = self._G._resolve_entity_key(value)
        old_ekey = self._G._row_to_entity.get(row_idx)
        if old_ekey is not None and old_ekey in self._G._entities:
            old_rec = self._G._entities[old_ekey]
            self._G._register_entity_record(
                old_ekey, EntityRecord(row_idx=old_rec.row_idx, kind=old_rec.kind)
            )
        rec = self._G._entities.get(ekey)
        kind = rec.kind if rec is not None else 'vertex'
        self._G._register_entity_record(ekey, EntityRecord(row_idx=row_idx, kind=kind))

    def __delitem__(self, key):
        ekey = self._G._row_to_entity.pop(key)
        rec = self._G._entities.get(ekey)
        if rec is not None and rec.row_idx == int(key):
            self._G._remove_entity_record(ekey)


class EntityTypesCompat(_CompatMapping):
    def keys(self):
        return [k[0] for k in self._G._entities.keys()]

    def __getitem__(self, key):
        ekey = self._G._resolve_entity_key(key)
        return _external_entity_kind(self._G._entities[ekey].kind)

    def __setitem__(self, key, value):
        ekey = self._G._resolve_entity_key(key)
        rec = self._G._entities.get(ekey)
        row_idx = rec.row_idx if rec is not None else len(self._G._entities)
        self._G._register_entity_record(
            ekey, EntityRecord(row_idx=row_idx, kind=_internal_entity_kind(value))
        )

    def __delitem__(self, key):
        ekey = self._G._resolve_entity_key(key)
        self._G._remove_entity_record(ekey)


def _ensure_edge_record(graph, edge_id: str) -> EdgeRecord:
    rec = graph._edges.get(edge_id)
    if rec is None:
        rec = EdgeRecord(
            src=None,
            tgt=None,
            weight=1.0,
            directed=None,
            etype='binary',
            col_idx=-1,
            ml_kind=None,
            ml_layers=None,
            direction_policy=None,
        )
        graph._edges[edge_id] = rec
    return rec


class EdgeToIdxCompat(_CompatMapping):
    def keys(self):
        return [eid for eid, rec in self._G._edges.items() if rec.col_idx >= 0]

    def __getitem__(self, key):
        col_idx = self._G._edges[key].col_idx
        if col_idx < 0:
            raise KeyError(key)
        return col_idx

    def __setitem__(self, key, value):
        rec = _ensure_edge_record(self._G, key)
        if rec.col_idx >= 0:
            self._G._col_to_edge.pop(rec.col_idx, None)
        rec.col_idx = int(value)
        self._G._col_to_edge[rec.col_idx] = key

    def __delitem__(self, key):
        rec = self._G._edges[key]
        if rec.col_idx < 0:
            raise KeyError(key)
        self._G._col_to_edge.pop(rec.col_idx, None)
        rec.col_idx = -1


class IdxToEdgeCompat(_CompatMapping):
    def keys(self):
        return self._G._col_to_edge.keys()

    def __getitem__(self, key):
        return self._G._col_to_edge[key]

    def __setitem__(self, key, value):
        col_idx = int(key)
        edge_id = value
        rec = _ensure_edge_record(self._G, edge_id)
        if rec.col_idx >= 0:
            self._G._col_to_edge.pop(rec.col_idx, None)
        rec.col_idx = col_idx
        self._G._col_to_edge[col_idx] = edge_id

    def __delitem__(self, key):
        edge_id = self._G._col_to_edge.pop(int(key))
        rec = self._G._edges.get(edge_id)
        if rec is not None:
            rec.col_idx = -1


class EdgeWeightsCompat(_CompatMapping):
    def keys(self):
        return self._G._edges.keys()

    def __getitem__(self, key):
        return self._G._edges[key].weight

    def __setitem__(self, key, value):
        _ensure_edge_record(self._G, key).weight = float(value)

    def __delitem__(self, key):
        _ensure_edge_record(self._G, key).weight = 1.0


class EdgeDirectedCompat(_CompatMapping):
    def keys(self):
        return [eid for eid, rec in self._G._edges.items() if rec.directed is not None]

    def __getitem__(self, key):
        rec = self._G._edges[key]
        if rec.directed is None:
            raise KeyError(key)
        return rec.directed

    def __setitem__(self, key, value):
        _ensure_edge_record(self._G, key).directed = bool(value) if value is not None else None

    def __delitem__(self, key):
        _ensure_edge_record(self._G, key).directed = None


class EdgeDefinitionsCompat(_CompatMapping):
    def keys(self):
        return [
            eid
            for eid, rec in self._G._edges.items()
            if rec.etype != 'hyper' and rec.src is not None
        ]

    def __getitem__(self, key):
        rec = self._G._edges[key]
        if rec.etype == 'hyper' or rec.src is None:
            raise KeyError(key)
        return (rec.src, rec.tgt, rec.etype)

    def __setitem__(self, key, value):
        src, tgt, etype = value
        rec = _ensure_edge_record(self._G, key)
        rec.src = src
        rec.tgt = tgt
        rec.etype = etype if etype != 'hyper' else 'binary'

    def __delitem__(self, key):
        rec = self._G._edges[key]
        rec.src = None
        rec.tgt = None


class HyperedgeDefinitionsCompat(_CompatMapping):
    def keys(self):
        return [eid for eid, rec in self._G._edges.items() if rec.etype == 'hyper']

    def __getitem__(self, key):
        rec = self._G._edges[key]
        if rec.etype != 'hyper':
            raise KeyError(key)
        if rec.tgt is not None:
            return {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
        return {'directed': False, 'members': set(rec.src)}

    def __setitem__(self, key, value):
        rec = _ensure_edge_record(self._G, key)
        rec.etype = 'hyper'
        if isinstance(value, list):
            rec.src = frozenset(value)
            rec.tgt = None
            rec.directed = False
            return
        directed = bool(value.get('directed', False))
        if directed:
            rec.src = frozenset(value.get('head', []))
            rec.tgt = frozenset(value.get('tail', []))
            rec.directed = True
        else:
            rec.src = frozenset(value.get('members', []))
            rec.tgt = None
            rec.directed = False

    def __delitem__(self, key):
        rec = self._G._edges[key]
        rec.etype = 'binary'
        rec.src = None
        rec.tgt = None


class EdgeDirectionPolicyCompat(_CompatMapping):
    def keys(self):
        return [eid for eid, rec in self._G._edges.items() if rec.direction_policy is not None]

    def __getitem__(self, key):
        val = self._G._edges[key].direction_policy
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        _ensure_edge_record(self._G, key).direction_policy = value

    def __delitem__(self, key):
        _ensure_edge_record(self._G, key).direction_policy = None


class EdgeKindCompat(_CompatMapping):
    def keys(self):
        return {
            eid
            for eid, rec in self._G._edges.items()
            if rec.etype == 'hyper' or rec.ml_kind is not None
        }

    def __getitem__(self, key):
        rec = self._G._edges.get(key)
        if rec is not None and rec.etype == 'hyper':
            return 'hyper'
        if rec is None or rec.ml_kind is None:
            raise KeyError(key)
        return rec.ml_kind

    def __setitem__(self, key, value):
        rec = _ensure_edge_record(self._G, key)
        if value == 'hyper':
            rec.etype = 'hyper'
            return
        rec.ml_kind = value

    def __delitem__(self, key):
        rec = self._G._edges.get(key)
        if rec is None:
            raise KeyError(key)
        if rec.etype == 'hyper':
            rec.etype = 'binary'
        else:
            rec.ml_kind = None


class EdgeLayersCompat(_CompatMapping):
    """edge_id -> ml_layers (backed by EdgeRecord.ml_layers)."""

    def keys(self):
        return [eid for eid, rec in self._G._edges.items() if rec.ml_layers is not None]

    def __getitem__(self, key):
        val = self._G._edges[key].ml_layers
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        _ensure_edge_record(self._G, key).ml_layers = value

    def __delitem__(self, key):
        _ensure_edge_record(self._G, key).ml_layers = None


_vertex_RESERVED = {'vertex_id'}  # nothing structural for vertices
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
