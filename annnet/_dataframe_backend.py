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
    rows = list(rows or [])

    if resolved == 'polars':
        import polars as pl

        if rows:
            return pl.DataFrame(rows)
        return pl.DataFrame(schema=_polars_schema(schema or {}))

    if resolved == 'pandas':
        import pandas as pd

        if rows:
            return pd.DataFrame.from_records(rows)
        return pd.DataFrame(
            {name: pd.Series(dtype=_pandas_dtype(kind)) for name, kind in (schema or {}).items()}
        )

    import pyarrow as pa

    if rows:
        return pa.Table.from_pylist(rows)
    return pa.Table.from_arrays(
        [pa.array([], type=_pyarrow_type(kind)) for kind in (schema or {}).values()],
        names=list((schema or {}).keys()),
    )


def dataframe_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = 'auto',
):
    """Build an eager dataframe/table from column-oriented data."""
    resolved = select_dataframe_backend(backend)
    columns = {name: list(values) for name, values in (columns or {}).items()}

    if resolved == 'polars':
        import polars as pl

        if schema:
            return pl.DataFrame(columns, schema=_polars_schema(schema))
        return pl.DataFrame(columns)

    if resolved == 'pandas':
        import pandas as pd

        if columns:
            return pd.DataFrame(columns)
        return pd.DataFrame(
            {name: pd.Series(dtype=_pandas_dtype(kind)) for name, kind in (schema or {}).items()}
        )

    import pyarrow as pa

    if columns:
        return pa.Table.from_pydict(columns)
    return pa.Table.from_arrays(
        [pa.array([], type=_pyarrow_type(kind)) for kind in (schema or {}).values()],
        names=list((schema or {}).keys()),
    )


def empty_dataframe(schema: dict[str, str], *, backend: str | None = 'auto'):
    """Build an empty dataframe/table with a generic schema."""
    return dataframe_from_rows([], schema=schema, backend=backend)


def dataframe_to_rows(df) -> list[dict[str, Any]]:
    """Return rows from a Narwhals-compatible eager dataframe/table."""
    if df is None:
        return []
    if hasattr(df, 'to_dicts'):
        return [dict(row) for row in df.to_dicts()]
    if hasattr(df, 'to_dict'):
        try:
            return [dict(row) for row in df.to_dict(orient='records')]
        except TypeError:
            pass
    if hasattr(df, 'to_pylist'):
        return [dict(row) for row in df.to_pylist()]

    ndf = nw.from_native(df, eager_only=True)
    return [dict(zip(ndf.columns, row, strict=False)) for row in ndf.rows()]


def dataframe_height(df) -> int:
    """Return the row count for a dataframe-like object."""
    if df is None:
        return 0
    if hasattr(df, 'height'):
        return int(df.height)
    if hasattr(df, 'num_rows'):
        return int(df.num_rows)
    if hasattr(df, 'shape'):
        return int(df.shape[0])
    return len(dataframe_to_rows(df))


def dataframe_columns(df) -> list[str]:
    """Return column names for a dataframe-like object."""
    if df is None:
        return []
    if hasattr(df, 'columns'):
        return list(df.columns)
    if hasattr(df, 'column_names'):
        return list(df.column_names)
    return list(dataframe_to_rows(df)[0]) if dataframe_height(df) else []


def rename_dataframe_columns(df, mapping: dict[str, str]):
    """Rename dataframe columns across supported backends."""
    if not mapping:
        return df
    if hasattr(df, 'rename'):
        try:
            return df.rename(columns=mapping)
        except TypeError:
            pass
        try:
            return df.rename(mapping)
        except TypeError:
            pass
    if hasattr(df, 'rename_columns'):
        names = [mapping.get(name, name) for name in dataframe_columns(df)]
        return df.rename_columns(names)
    rows = []
    for row in dataframe_to_rows(df):
        rows.append({mapping.get(key, key): value for key, value in row.items()})
    return dataframe_from_rows(rows)


def _polars_schema(schema: dict[str, str]):

    return {name: _polars_dtype(kind) for name, kind in schema.items()}


def _polars_dtype(kind: str):
    import polars as pl

    if kind == _FLOAT:
        return pl.Float64
    if kind == _BOOL:
        return pl.Boolean
    if kind == _LIST_TEXT:
        return pl.List(pl.Utf8)
    return pl.Utf8


def _pandas_dtype(kind: str) -> str:
    if kind == _FLOAT:
        return 'float64'
    if kind == _BOOL:
        return 'bool'
    if kind == _LIST_TEXT:
        return 'object'
    return 'string'


def _pyarrow_type(kind: str):
    import pyarrow as pa

    if kind == _FLOAT:
        return pa.float64()
    if kind == _BOOL:
        return pa.bool_()
    if kind == _LIST_TEXT:
        return pa.list_(pa.string())
    return pa.string()
