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


def dataframe_memory_usage(df) -> int:
    """Best-effort memory usage for a dataframe-like object."""
    if df is None:
        return 0
    if hasattr(df, 'estimated_size'):
        try:
            return int(df.estimated_size())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(df, 'memory_usage'):
        try:
            usage = df.memory_usage(deep=True)
            return int(usage.sum() if hasattr(usage, 'sum') else usage)
        except Exception:  # noqa: BLE001
            pass
    if hasattr(df, 'nbytes'):
        try:
            return int(df.nbytes)
        except Exception:  # noqa: BLE001
            pass
    return 0


def dataframe_columns(df) -> list[str]:
    """Return column names for a dataframe-like object."""
    if df is None:
        return []
    if hasattr(df, 'columns'):
        return list(df.columns)
    if hasattr(df, 'column_names'):
        return list(df.column_names)
    return list(dataframe_to_rows(df)[0]) if dataframe_height(df) else []


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
    if hasattr(df, 'clone'):
        return df.clone()
    if hasattr(df, 'copy'):
        try:
            return df.copy(deep=True)
        except TypeError:
            return df.copy()
    return dataframe_from_rows(dataframe_to_rows(df), backend=dataframe_backend(df))


def dataframe_filter_eq(df, column: str, value):
    """Filter rows where ``column == value``."""
    if df is None or column not in dataframe_columns(df):
        return _empty_like(df)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.filter(pl.col(column) == value)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df[df[column] == value].copy()
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow.compute as pc

        if hasattr(df, 'filter') and hasattr(df, 'column_names'):
            return df.filter(pc.equal(df[column], value))
    except Exception:  # noqa: BLE001
        pass
    return _rows_filter(df, lambda row: row.get(column) == value)


def dataframe_filter_ne(df, column: str, value):
    """Filter rows where ``column != value``."""
    if df is None or column not in dataframe_columns(df):
        return clone_dataframe(df)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.filter(pl.col(column) != value)
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df[df[column] != value].copy()
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow.compute as pc

        if hasattr(df, 'filter') and hasattr(df, 'column_names'):
            return df.filter(pc.invert(pc.equal(df[column], value)))
    except Exception:  # noqa: BLE001
        pass
    return _rows_filter(df, lambda row: row.get(column) != value)


def dataframe_filter_in(df, column: str, values):
    """Filter rows where ``column`` is in ``values``."""
    vals = list(values or [])
    if df is None or column not in dataframe_columns(df):
        return _empty_like(df)
    if not vals:
        return _empty_like(df)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.filter(pl.col(column).is_in(vals))
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df[df[column].isin(vals)].copy()
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        if hasattr(df, 'filter') and hasattr(df, 'column_names'):
            return df.filter(pc.is_in(df[column], value_set=pa.array(vals)))
    except Exception:  # noqa: BLE001
        pass
    valset = set(vals)
    return _rows_filter(df, lambda row: row.get(column) in valset)


def dataframe_filter_not_in(df, column: str, values):
    """Filter rows where ``column`` is not in ``values``."""
    vals = list(values or [])
    if df is None or column not in dataframe_columns(df):
        return clone_dataframe(df)
    if not vals:
        return clone_dataframe(df)
    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return df.filter(~pl.col(column).is_in(vals))
    except Exception:  # noqa: BLE001
        pass
    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return df[~df[column].isin(vals)].copy()
    except Exception:  # noqa: BLE001
        pass
    try:
        import pyarrow as pa
        import pyarrow.compute as pc

        if hasattr(df, 'filter') and hasattr(df, 'column_names'):
            return df.filter(pc.invert(pc.is_in(df[column], value_set=pa.array(vals))))
    except Exception:  # noqa: BLE001
        pass
    valset = set(vals)
    return _rows_filter(df, lambda row: row.get(column) not in valset)


def dataframe_drop_rows(df, column: str, values):
    """Return ``df`` without rows whose ``column`` is in ``values``."""
    return dataframe_filter_not_in(df, column, values)


def dataframe_append_rows(df, rows: list[dict[str, Any]], *, backend: str | None = None):
    """Append rows to a dataframe-like object, preserving the existing backend."""
    rows = [dict(row) for row in (rows or [])]
    resolved = backend or dataframe_backend(df)
    if not rows:
        return clone_dataframe(df)

    existing_cols = dataframe_columns(df)
    all_cols = list(existing_cols)
    for row in rows:
        for col in row:
            if col not in all_cols:
                all_cols.append(col)

    if df is not None:
        try:
            import polars as pl

            if isinstance(df, pl.DataFrame):
                add_df = pl.DataFrame([{col: row.get(col) for col in all_cols} for row in rows])
                if existing_cols == all_cols:
                    return pl.concat([df, add_df], how='vertical_relaxed')
                return pl.concat([df, add_df], how='diagonal_relaxed')
        except Exception:  # noqa: BLE001
            pass

        try:
            import pandas as pd

            if isinstance(df, pd.DataFrame):
                add_df = pd.DataFrame.from_records(rows)
                return pd.concat([df, add_df], ignore_index=True)
        except Exception:  # noqa: BLE001
            pass

        try:
            import pyarrow as pa

            if hasattr(df, 'schema') and hasattr(df, 'num_rows') and hasattr(df, 'to_pylist'):
                add_df = pa.Table.from_pylist(
                    [{col: row.get(col) for col in all_cols} for row in rows]
                )
                if existing_cols != all_cols:
                    df = pa.Table.from_pylist(
                        [{col: row.get(col) for col in all_cols} for row in dataframe_to_rows(df)]
                    )
                return pa.concat_tables([df, add_df], promote_options='default')
        except Exception:  # noqa: BLE001
            pass

    normalized = []
    for row in [*dataframe_to_rows(df), *rows]:
        normalized.append({col: row.get(col) for col in all_cols})
    return dataframe_from_rows(normalized, schema=_text_schema(all_cols), backend=resolved)


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
    return dataframe_append_rows(
        dataframe_from_rows(
            kept,
            schema=_text_schema(dataframe_columns(df)),
            backend=backend or dataframe_backend(df),
        ),
        rows,
        backend=backend or dataframe_backend(df),
    )


def dataframe_write_csv(df, path) -> None:
    """Write a dataframe-like object to CSV."""
    if hasattr(df, 'write_csv'):
        df.write_csv(path)
        return
    if hasattr(df, 'to_csv'):
        df.to_csv(path, index=False)
        return
    if hasattr(df, 'schema') and hasattr(df, 'num_rows'):
        import pyarrow.csv as pa_csv

        pa_csv.write_csv(df, path)
        return
    dataframe_from_rows(dataframe_to_rows(df)).write_csv(path)


def dataframe_write_parquet(df, path) -> None:
    """Write a dataframe-like object to Parquet."""
    if hasattr(df, 'write_parquet'):
        df.write_parquet(path)
        return
    if hasattr(df, 'to_parquet'):
        df.to_parquet(path, index=False)
        return
    if hasattr(df, 'schema') and hasattr(df, 'num_rows'):
        import pyarrow.parquet as pq

        pq.write_table(df, path)
        return
    table = dataframe_from_rows(dataframe_to_rows(df), backend='pyarrow')
    import pyarrow.parquet as pq

    pq.write_table(table, path)


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


def _rows_filter(df, predicate):
    rows = [row for row in dataframe_to_rows(df) if predicate(row)]
    return dataframe_from_rows(
        rows,
        schema=_text_schema(dataframe_columns(df)),
        backend=dataframe_backend(df),
    )


def _empty_like(df):
    return dataframe_from_rows(
        [],
        schema=_text_schema(dataframe_columns(df)),
        backend=dataframe_backend(df),
    )


def _text_schema(columns: list[str]) -> dict[str, str]:
    return dict.fromkeys(columns, _TEXT)


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
