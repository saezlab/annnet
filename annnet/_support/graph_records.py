"""Generic internal helpers for graph/row record conversion."""

from __future__ import annotations

from enum import Enum
from typing import Any

import narwhals as nw

from .dataframe_backend import dataframe_to_rows, dataframe_from_rows


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
