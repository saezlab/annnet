"""Generic internal helpers for graph/row record conversion."""

from __future__ import annotations

from enum import Enum
from typing import Any

import narwhals as nw

from .dataframe_backend import dataframe_to_rows, dataframe_from_rows


def _is_directed_eid(graph, eid):
    """Best-effort directedness probe; default True."""
    try:
        rec = getattr(graph, '_edges', {}).get(eid)
        if rec is not None and rec.directed is not None:
            return bool(rec.directed)
    except (AttributeError, TypeError):
        pass
    try:
        value = graph.attrs.get_attr_edge(eid, 'directed')
        return bool(value) if value is not None else True
    except (AttributeError, KeyError, TypeError, ValueError):
        return True


def _iter_vertex_ids(graph):
    """Yield vertex ids in stable graph/entity order."""
    entities = getattr(graph, '_entities', None)
    if isinstance(entities, dict):
        seen = set()
        for ekey, rec in sorted(entities.items(), key=lambda item: item[1].row_idx):
            if getattr(rec, 'kind', None) != 'vertex':
                continue
            vid = ekey[0]
            if vid in seen:
                continue
            seen.add(vid)
            yield vid
        return

    try:
        for vid in graph.vertices():
            yield vid
        return
    except AttributeError as exc:
        raise AttributeError('Graph does not expose an adapter-readable vertex store') from exc


def _serialize_value(val: Any) -> Any:
    if isinstance(val, Enum):
        return val.name
    if hasattr(val, 'items'):
        return dict(val)
    return val


def _attrs_to_dict(attrs_dict: dict) -> dict:
    out = {}
    for key, val in attrs_dict.items():
        if isinstance(val, Enum):
            out[key] = val.name
        elif hasattr(val, 'items'):
            out[key] = {
                inner_key: (inner_val.name if isinstance(inner_val, Enum) else inner_val)
                for inner_key, inner_val in dict(val).items()
            }
        else:
            out[key] = val
    return out


def _rows_like(table):
    if table is None:
        return []
    try:
        return dataframe_to_rows(table)
    except (AttributeError, TypeError, ValueError):
        pass
    if hasattr(table, 'fetchall') and hasattr(table, 'columns'):
        try:
            cols = list(table.columns)
            return [dict(zip(cols, row, strict=False)) for row in table.fetchall()]
        except (AttributeError, TypeError):
            pass
    if isinstance(table, dict):
        keys = list(table.keys())
        if keys and isinstance(table[keys[0]], list):
            n_rows = len(table[keys[0]])
            return [{key: table[key][idx] for key in keys} for idx in range(n_rows)]
    if isinstance(table, list) and table and isinstance(table[0], dict):
        return list(table)
    return []


def _iter_edge_records(graph):
    """Yield ``(eid, rec)`` for edge-like entities in stable graph order."""
    col_to_edge = getattr(graph, '_col_to_edge', None)
    edges = getattr(graph, '_edges', None)

    if isinstance(col_to_edge, dict) and edges is not None:
        for eidx in range(graph.ne):
            eid = col_to_edge[eidx]
            rec = edges[eid]
            yield eid, rec
        return

    if edges is not None:
        for eid, rec in edges.items():
            yield eid, rec
        return

    raise AttributeError('Graph does not expose an adapter-readable edge record store')


def _rows_to_df(rows: list[dict]):
    """Build a dataframe from list-of-dicts, preserving first-seen column order."""
    if not rows:
        return dataframe_from_rows(rows)
    order = []
    for row in rows:
        for key in row.keys():
            if key not in order:
                order.append(key)
    df = dataframe_from_rows(rows)
    try:
        return nw.from_native(df, eager_only=True).select(order).to_native()
    except (AttributeError, TypeError, ValueError):
        return df
