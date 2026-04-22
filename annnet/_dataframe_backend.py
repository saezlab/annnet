"""Central dataframe backend helpers.

AnnNet accepts Narwhals-compatible dataframe inputs, but newly-created internal
tables need a concrete eager backend. Backend selection is centralized here so
callers do not each implement their own fallback chain.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from ._optional_components import (
    DATAFRAME_BACKENDS,
    component_names,
    select_component,
    available_optional_components,
)

DATAFRAME_BACKEND_PRIORITY = component_names(DATAFRAME_BACKENDS)
_DEFAULT_DATAFRAME_BACKEND = 'auto'
_TEXT = 'text'
_FLOAT = 'float'
_BOOL = 'bool'
_LIST_TEXT = 'list_text'


def available_dataframe_backends() -> dict[str, bool]:
    """Return installed dataframe backends in AnnNet preference order."""
    return available_optional_components(DATAFRAME_BACKENDS)


def select_dataframe_backend(preferred: str | None = 'auto') -> str:
    """Resolve a dataframe backend name.

    ``"auto"`` selects the first installed backend in this order: Polars,
    pandas, then PyArrow.
    """
    preferred = _DEFAULT_DATAFRAME_BACKEND if preferred is None else preferred
    return select_component(
        DATAFRAME_BACKENDS,
        preferred,
        kind='dataframe',
        install_message='Install polars, pandas, or pyarrow',
    )


def get_default_dataframe_backend() -> str:
    """Return the configured default dataframe backend."""
    return _DEFAULT_DATAFRAME_BACKEND


def set_default_dataframe_backend(backend: str | None = 'auto') -> str:
    """Set the default dataframe backend for new AnnNet annotation tables."""
    global _DEFAULT_DATAFRAME_BACKEND

    requested = 'auto' if backend is None else str(backend).lower()
    if requested != 'auto':
        select_dataframe_backend(requested)
    elif not any(available_dataframe_backends().values()):
        select_dataframe_backend('auto')
    _DEFAULT_DATAFRAME_BACKEND = requested
    return _DEFAULT_DATAFRAME_BACKEND


def dataframe_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = 'auto',
):
    """Build an eager dataframe/table using the selected backend."""
    resolved = select_dataframe_backend(backend)
    return _native_from_rows(rows or [], schema=schema, backend=resolved)


def dataframe_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = 'auto',
):
    """Build an eager dataframe/table from column-oriented data."""
    resolved = select_dataframe_backend(backend)
    return _native_from_columns(columns or {}, schema=schema, backend=resolved)


def empty_dataframe(schema: dict[str, str], *, backend: str | None = 'auto'):
    """Build an empty dataframe/table with a generic schema."""
    return dataframe_from_rows([], schema=schema, backend=backend)


def dataframe_to_rows(df) -> list[dict[str, Any]]:
    """Return rows from a Narwhals-compatible eager dataframe/table."""
    if df is None:
        return []
    return [dict(row) for row in _to_nw(df).rows(named=True)]


def dataframe_height(df) -> int:
    """Return the row count for a dataframe-like object."""
    if df is None:
        return 0
    return len(_to_nw(df).rows())


def dataframe_memory_usage(df) -> int:
    """Best-effort memory usage for a dataframe-like object."""
    if df is None:
        return 0
    try:
        return int(_to_nw(df).estimated_size())
    except Exception:  # noqa: BLE001
        return 0


def dataframe_columns(df) -> list[str]:
    """Return column names for a dataframe-like object."""
    if df is None:
        return []
    return _schema_names(_schema_from_df(df))


def dataframe_backend(df, *, default: str | None = 'auto') -> str:
    """Return the concrete backend for a dataframe-like object."""
    if df is None:
        return select_dataframe_backend(default)

    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return 'polars'
    except Exception:  # noqa: BLE001
        pass

    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return 'pandas'
    except Exception:  # noqa: BLE001
        pass

    if hasattr(df, 'schema') and hasattr(df, 'num_rows') and hasattr(df, 'to_pylist'):
        return 'pyarrow'

    return select_dataframe_backend(default)


def clone_dataframe(df):
    """Return a shallow-safe copy/clone of a dataframe-like object."""
    if df is None:
        return None
    return _from_nw(_to_nw(df).clone())


def dataframe_filter_eq(df, column: str, value):
    """Filter rows where ``column == value``."""
    if df is None or column not in dataframe_columns(df):
        return _empty_like(df)
    return _from_nw(_to_nw(df).filter(nw.col(column) == value))


def dataframe_filter_ne(df, column: str, value):
    """Filter rows where ``column != value``."""
    if df is None or column not in dataframe_columns(df):
        return clone_dataframe(df)
    return _from_nw(_to_nw(df).filter(nw.col(column) != value))


def dataframe_filter_in(df, column: str, values):
    """Filter rows where ``column`` is in ``values``."""
    vals = list(values or [])
    if df is None or column not in dataframe_columns(df):
        return _empty_like(df)
    if not vals:
        return _empty_like(df)
    return _from_nw(_to_nw(df).filter(nw.col(column).is_in(vals)))


def dataframe_filter_not_in(df, column: str, values):
    """Filter rows where ``column`` is not in ``values``."""
    vals = list(values or [])
    if df is None or column not in dataframe_columns(df):
        return clone_dataframe(df)
    if not vals:
        return clone_dataframe(df)
    return _from_nw(_to_nw(df).filter(~nw.col(column).is_in(vals)))


def dataframe_drop_rows(df, column: str, values):
    """Return ``df`` without rows whose ``column`` is in ``values``."""
    return dataframe_filter_not_in(df, column, values)


def dataframe_append_rows(df, rows: list[dict[str, Any]], *, backend: str | None = None):
    """Append rows to a dataframe-like object, preserving the existing backend."""
    rows = [dict(row) for row in (rows or [])]
    if not rows:
        return clone_dataframe(df)

    resolved_backend = dataframe_backend(df, default=backend or 'auto')
    add_ndf = _build_nw_from_rows(rows, schema=None, backend=resolved_backend)

    if df is None:
        return _from_nw(add_ndf)

    base_ndf = _to_nw(df)
    out = nw.concat([base_ndf, add_ndf], how='diagonal')
    return _from_nw(out)


def dataframe_upsert_rows(
    df,
    rows: list[dict[str, Any]],
    key_columns: str | list[str] | tuple[str, ...],
    *,
    backend: str | None = None,
):
    """Replace rows with matching key values, then append the new rows."""
    rows = [dict(row) for row in (rows or [])]
    if not rows:
        return clone_dataframe(df)

    keys = (key_columns,) if isinstance(key_columns, str) else tuple(key_columns)
    incoming_keys = {tuple(row.get(key) for key in keys) for row in rows}
    kept = [
        row
        for row in dataframe_to_rows(df)
        if tuple(row.get(key) for key in keys) not in incoming_keys
    ]

    base = _native_from_rows(
        kept,
        schema=_schema_from_df(df),
        backend=dataframe_backend(df, default=backend or 'auto'),
    )
    return dataframe_append_rows(
        base,
        rows,
        backend=backend or dataframe_backend(df, default='auto'),
    )


def dataframe_write_csv(df, path) -> None:
    """Write a dataframe-like object to CSV."""
    _to_nw(df).write_csv(path)


def dataframe_write_parquet(df, path) -> None:
    """Write a dataframe-like object to Parquet."""
    _to_nw(df).write_parquet(path)


def rename_dataframe_columns(df, mapping: dict[str, str]):
    """Rename dataframe columns across supported backends."""
    if df is None or not mapping:
        return df
    return _from_nw(_to_nw(df).rename(mapping))


def _rows_filter(df, predicate):
    """Fallback row-based filter preserving schema and backend."""
    rows = [row for row in dataframe_to_rows(df) if predicate(row)]
    return _native_from_rows(
        rows,
        schema=_schema_from_df(df),
        backend=dataframe_backend(df),
    )


def _empty_like(df):
    if df is None:
        return dataframe_from_rows([], schema={}, backend='auto')
    return _native_from_rows(
        [],
        schema=_schema_from_df(df),
        backend=dataframe_backend(df),
    )


def _text_schema(columns: list[str]) -> dict[str, str]:
    return dict.fromkeys(columns, _TEXT)


def _to_nw(df):
    return nw.from_native(df, eager_only=True)


def _from_nw(df):
    return df.to_native()


def _schema_from_df(df) -> nw.Schema | None:
    if df is None:
        return None
    return _to_nw(df).collect_schema()


def _schema_names(schema: nw.Schema | None) -> list[str]:
    if schema is None:
        return []
    return list(schema.names())


def _build_nw_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    rows = [dict(row) for row in (rows or []) if isinstance(row, dict)]

    if rows:
        return nw.from_dicts(rows, backend=backend)

    nw_schema = _normalize_schema(schema)
    empty_cols = {name: [] for name in _schema_names(nw_schema)}
    if nw_schema is not None:
        if backend == 'pandas':
            return nw.from_dict(empty_cols, backend=backend)
        return nw.from_dict(empty_cols, schema=nw_schema, backend=backend)
    return nw.from_dict({}, backend=backend)


def _build_nw_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    cols = {name: list(values) for name, values in (columns or {}).items()}
    if any(cols.values()):
        return nw.from_dict(cols, backend=backend)

    nw_schema = _normalize_schema(schema)
    if nw_schema is not None:
        # Ensure empty schema columns are present.
        for name in nw_schema.names():
            cols.setdefault(name, [])
        if backend == 'pandas' and not any(cols.values()):
            return nw.from_dict(cols, backend=backend)
        return nw.from_dict(cols, schema=nw_schema, backend=backend)

    return nw.from_dict(cols, backend=backend)


def _native_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    return _from_nw(_build_nw_from_rows(rows, schema=schema, backend=backend))


def _native_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    return _from_nw(_build_nw_from_columns(columns, schema=schema, backend=backend))


def _normalize_schema(schema: nw.Schema | dict[str, str] | None) -> nw.Schema | None:
    if schema is None:
        return None
    if isinstance(schema, nw.Schema):
        return schema
    return _narwhals_schema(schema)


def _narwhals_schema(schema: dict[str, str]) -> nw.Schema:
    return nw.Schema({name: _narwhals_dtype(kind) for name, kind in schema.items()})


def _narwhals_dtype(kind: str):
    if kind == _FLOAT:
        return nw.Float64()
    if kind == _BOOL:
        return nw.Boolean()
    if kind == _LIST_TEXT:
        return nw.List(nw.String())
    return nw.String()
