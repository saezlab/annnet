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
_INT = 'int'
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


def _dataframe_width(df) -> int:
    """Return the column count for a dataframe-like object."""
    return len(dataframe_columns(df))


def _dataframe_is_empty(df) -> bool:
    """Return whether a dataframe-like object has no rows."""
    return dataframe_height(df) == 0


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


def _dataframe_column_values(df, column: str) -> list[Any]:
    """Return one dataframe column as a Python list."""
    if df is None or column not in dataframe_columns(df):
        return []
    return [row.get(column) for row in dataframe_to_rows(df)]


def _dataframe_select_to_numpy(df, columns: list[str]):
    """Return selected columns as a NumPy array via Narwhals."""
    return _to_nw(df).select(columns).to_numpy()


def _dataframe_column_is_numeric(df, column: str) -> bool:
    """Best-effort numeric column probe across supported eager backends."""
    if df is None or column not in dataframe_columns(df):
        return False

    try:
        dtype = _to_nw(df).collect_schema()[column]
        is_numeric = getattr(dtype, 'is_numeric', None)
        if callable(is_numeric) and is_numeric():
            return True
    except Exception:  # noqa: BLE001
        pass

    values = _dataframe_column_values(df, column)
    seen = False
    for value in values:
        if value is None or value == '':
            continue
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        seen = True
    return seen


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
    base_rows = dataframe_to_rows(df)
    merged_schema = _merge_schema_specs(
        _schema_spec_from_schema(_schema_from_df(df)),
        _schema_spec_from_rows(base_rows + rows),
    )
    return _native_from_rows(
        base_rows + rows,
        schema=merged_schema,
        backend=resolved_backend,
    )


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

    resolved_backend = dataframe_backend(df, default=backend or 'auto')
    merged_schema = _merge_schema_specs(
        _schema_spec_from_schema(_schema_from_df(df)),
        _schema_spec_from_rows(kept + rows),
    )
    return _native_from_rows(
        kept + rows,
        schema=merged_schema,
        backend=resolved_backend,
    )


def _dataframe_read_delimited(
    source,
    *,
    separator: str = ',',
    backend: str | None = 'auto',
    infer_schema_length: int | None = None,
    encoding: str | None = None,
    null_values: list[str] | None = None,
    low_memory: bool | None = None,
):
    """Read a delimited file/buffer into a Narwhals-compatible dataframe."""

    resolved = select_dataframe_backend(backend)

    # Read using backend-native reader
    if resolved == 'polars':
        import polars as pl

        options = {'separator': separator}
        if infer_schema_length is not None:
            options['infer_schema_length'] = infer_schema_length
        if encoding is not None:
            options['encoding'] = encoding
        if null_values is not None:
            options['null_values'] = null_values
        if low_memory is not None:
            options['low_memory'] = low_memory
        native = pl.read_csv(source, **options)

    elif resolved == 'pandas':
        import pandas as pd

        options = {'sep': separator}
        if encoding is not None:
            options['encoding'] = encoding
        if null_values is not None:
            options['na_values'] = null_values
        native = pd.read_csv(source, **options)

    else:
        import pyarrow.csv as pacsv

        read_options = pacsv.ReadOptions(encoding=encoding or 'utf8')
        convert_options = (
            pacsv.ConvertOptions(null_values=null_values)
            if null_values is not None
            else pacsv.ConvertOptions()
        )
        native = pacsv.read_csv(
            source,
            read_options=read_options,
            parse_options=pacsv.ParseOptions(delimiter=separator),
            convert_options=convert_options,
        )

    # Normalize via Narwhals roundtrip (optional but consistent)
    return _from_nw(_to_nw(native))


def _dataframe_read_tsv(source, *, backend: str | None = 'auto'):
    return _dataframe_read_delimited(source, separator='\t', backend=backend)


def _dataframe_read_excel(source, *, sheet_name=None):
    """Read an Excel sheet into a Narwhals-compatible dataframe.

    Excel support is intentionally a pandas-backed boundary because Narwhals does
    not define an Excel reader.
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            'Excel support requires `pandas` at runtime. '
            'Install it or convert the file to CSV manually.'
        ) from e

    native = pd.read_excel(source, sheet_name=sheet_name)
    if isinstance(native, dict):
        if sheet_name is None:
            _, native = next(iter(native.items()))
        else:
            native = native[sheet_name]
    return _from_nw(_to_nw(native))


def _dataframe_write_csv(df, path) -> None:
    """Write a dataframe-like object to CSV."""
    _to_nw(df).write_csv(path)


def _dataframe_write_parquet(df, path) -> None:
    """Write a dataframe-like object to Parquet."""
    _to_nw(df).write_parquet(path)


def _dataframe_read_parquet(path, *, backend: str | None = None):
    """Read a Parquet file into the configured dataframe backend."""
    resolved = select_dataframe_backend(backend)
    if resolved == 'polars':
        import polars as pl

        return pl.read_parquet(path)
    if resolved == 'pandas':
        import pandas as pd

        return pd.read_parquet(path, engine='pyarrow')

    import pyarrow.parquet as pq

    return pq.read_table(path)


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
    nw_schema = _normalize_schema(schema)

    if rows:
        all_names = set()
        for row in rows:
            all_names.update(row.keys())
        if nw_schema is not None:
            all_names.update(nw_schema.names())
        cols = {name: [row.get(name) for row in rows] for name in sorted(all_names)}
        return _build_nw_from_columns(cols, schema=nw_schema, backend=backend)

    empty_cols = {name: [] for name in _schema_names(nw_schema)}
    if nw_schema is not None:
        return _cast_nw_to_schema(nw.from_dict(empty_cols, backend=backend), nw_schema)
    return nw.from_dict({}, backend=backend)


def _build_nw_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    cols = {name: list(values) for name, values in (columns or {}).items()}
    nw_schema = _normalize_schema(schema)
    if any(cols.values()):
        if nw_schema is not None:
            for name in nw_schema.names():
                cols.setdefault(name, [None] * len(next(iter(cols.values()), [])))
            return _cast_nw_to_schema(nw.from_dict(cols, backend=backend), nw_schema)
        return nw.from_dict(cols, backend=backend)

    if nw_schema is not None:
        for name in nw_schema.names():
            cols.setdefault(name, [])
        return _cast_nw_to_schema(nw.from_dict(cols, backend=backend), nw_schema)

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
    if kind == _INT:
        return nw.Int64()
    if kind == _FLOAT:
        return nw.Float64()
    if kind == _BOOL:
        return nw.Boolean()
    if kind == _LIST_TEXT:
        return nw.List(nw.String())
    return nw.String()


def _cast_nw_to_schema(df, schema: nw.Schema):
    out = df
    current = out.collect_schema()
    for name in schema.names():
        target = schema[name]
        if name not in current:
            try:
                out = out.with_columns(nw.lit(None).cast(target).alias(name))
            except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
                out = out.with_columns(nw.lit(None).alias(name))
            continue

        cur = current[name]
        if cur == target:
            continue
        if cur == nw.Unknown() or target != nw.Unknown():
            try:
                out = out.with_columns(nw.col(name).cast(target))
            except (AttributeError, NotImplementedError, RuntimeError, TypeError, ValueError):
                pass
        current = out.collect_schema()
    return out


def _schema_spec_from_schema(schema: nw.Schema | None) -> dict[str, str | None]:
    if schema is None:
        return {}
    return {name: _kind_from_dtype(schema[name]) for name in schema.names()}


def _schema_spec_from_rows(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    spec: dict[str, str | None] = {}
    for row in rows:
        for name, value in row.items():
            spec[name] = _merge_kind(spec.get(name), _kind_for_value(value))
    return spec


def _kind_from_dtype(dtype) -> str | None:
    if dtype == nw.Unknown():
        return None
    if dtype == nw.Boolean():
        return _BOOL
    if dtype in {
        nw.Int8(),
        nw.Int16(),
        nw.Int32(),
        nw.Int64(),
        nw.UInt8(),
        nw.UInt16(),
        nw.UInt32(),
        nw.UInt64(),
    }:
        return _INT
    if dtype in {nw.Float32(), nw.Float64()}:
        return _FLOAT
    if dtype == nw.List(nw.String()):
        return _LIST_TEXT
    return _TEXT


def _kind_for_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return _BOOL
    if isinstance(value, int) and not isinstance(value, bool):
        return _INT
    if isinstance(value, float):
        return _FLOAT
    if isinstance(value, (list, tuple)):
        return _LIST_TEXT
    return _TEXT


def _merge_kind(left: str | None, right: str | None) -> str | None:
    if left is None:
        return right
    if right is None:
        return left
    if left == right:
        return left
    if _TEXT in {left, right}:
        return _TEXT
    if _LIST_TEXT in {left, right}:
        return _TEXT
    if _FLOAT in {left, right}:
        return _FLOAT
    if _INT in {left, right}:
        return _INT
    return _TEXT


def _merge_schema_specs(
    left: dict[str, str | None],
    right: dict[str, str | None],
) -> dict[str, str]:
    merged: dict[str, str] = {}
    for name in sorted(set(left) | set(right)):
        kind = _merge_kind(left.get(name), right.get(name))
        merged[name] = _TEXT if kind is None else kind
    return merged
