"""Central dataframe backend helpers.

AnnNet accepts Narwhals-compatible dataframe inputs, but newly-created internal
tables need a concrete eager backend. Backend selection is centralized here so
callers do not each implement their own fallback chain.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from .optional_components import (
    DATAFRAME_BACKENDS,
    select_component,
    available_optional_components,
)

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
    select_dataframe_backend(requested)

    _DEFAULT_DATAFRAME_BACKEND = requested
    return requested


def dataframe_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = 'auto',
):
    """Build an eager dataframe/table using the selected backend."""
    return _from_nw(
        _build_nw_from_rows(
            rows or [],
            schema=schema,
            backend=select_dataframe_backend(backend),
        )
    )


def dataframe_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = 'auto',
):
    """Build an eager dataframe/table from column-oriented data."""
    return _from_nw(
        _build_nw_from_columns(
            columns or {},
            schema=schema,
            backend=select_dataframe_backend(backend),
        )
    )


def empty_dataframe(schema: dict[str, str], *, backend: str | None = 'auto'):
    """Build an empty dataframe/table with a generic schema."""
    return dataframe_from_columns({}, schema=schema, backend=backend)


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


def dataframe_width(df) -> int:
    """Return the column count for a dataframe-like object."""
    return len(dataframe_columns(df))


def dataframe_memory_usage(df) -> int:
    """Best-effort memory usage for a dataframe-like object."""
    if df is None:
        return 0
    try:
        return int(_to_nw(df).estimated_size())
    except (AttributeError, TypeError, ValueError):
        return 0


def dataframe_columns(df) -> list[str]:
    """Return column names for a dataframe-like object."""
    return [] if df is None else list(_to_nw(df).collect_schema().names())


def dataframe_column_values(df, column: str) -> list[Any]:
    """Return one dataframe column as a Python list."""
    if df is None or column not in dataframe_columns(df):
        return []
    return [row.get(column) for row in dataframe_to_rows(df)]


def dataframe_select_to_numpy(df, columns: list[str]):
    """Return selected columns as a NumPy array via Narwhals."""
    return _to_nw(df).select(columns).to_numpy()


def dataframe_column_is_numeric(df, column: str) -> bool:
    """Best-effort numeric column probe across supported eager backends."""
    if not _has_column(df, column):
        return False

    try:
        is_numeric = getattr(_to_nw(df).collect_schema()[column], 'is_numeric', None)
        if callable(is_numeric):
            return bool(is_numeric())
    except (AttributeError, KeyError, TypeError, ValueError):
        pass

    values = [v for v in dataframe_column_values(df, column) if v not in (None, '')]
    if not values:
        return False

    try:
        for value in values:
            float(value)
    except (TypeError, ValueError):
        return False

    return True


def dataframe_backend(df, *, default: str | None = 'auto') -> str:
    """Return the concrete backend for a dataframe-like object."""
    if df is None:
        return select_dataframe_backend(default)

    try:
        import polars as pl

        if isinstance(df, pl.DataFrame):
            return 'polars'
    except ImportError:
        pass

    try:
        import pandas as pd

        if isinstance(df, pd.DataFrame):
            return 'pandas'
    except ImportError:
        pass

    if hasattr(df, 'schema') and hasattr(df, 'num_rows') and hasattr(df, 'to_pylist'):
        return 'pyarrow'

    return select_dataframe_backend(default)


def clone_dataframe(df):
    """Return a shallow-safe copy/clone of a dataframe-like object."""
    if df is None:
        return None
    return _from_nw(_to_nw(df).clone())


def _fallback_dataframe(df, *, empty: bool):
    return _empty_like(df) if empty else clone_dataframe(df)


def _has_column(df, column: str) -> bool:
    return df is not None and column in dataframe_columns(df)


def _filter_column(df, column: str, expr, *, empty_if_missing: bool):
    if not _has_column(df, column):
        return _fallback_dataframe(df, empty=empty_if_missing)
    return _from_nw(_to_nw(df).filter(expr(nw.col(column))))


def dataframe_filter_eq(df, column: str, value):
    """Filter rows where ``column == value``."""
    return _filter_column(df, column, lambda col: col == value, empty_if_missing=True)


def dataframe_filter_ne(df, column: str, value):
    """Filter rows where ``column != value``."""
    return _filter_column(df, column, lambda col: col != value, empty_if_missing=False)


def _filter_in(df, column: str, values, *, negate: bool):
    vals = list(values or [])
    if not _has_column(df, column) or not vals:
        return _fallback_dataframe(df, empty=not negate)

    expr = nw.col(column).is_in(vals)
    return _from_nw(_to_nw(df).filter(~expr if negate else expr))


def dataframe_filter_in(df, column: str, values):
    """Filter rows where ``column`` is in ``values``."""
    return _filter_in(df, column, values, negate=False)


def dataframe_filter_not_in(df, column: str, values):
    """Filter rows where ``column`` is not in ``values``."""
    return _filter_in(df, column, values, negate=True)


def dataframe_drop_rows(df, column: str, values):
    """Return ``df`` without rows whose ``column`` is in ``values``."""
    return dataframe_filter_not_in(df, column, values)


def dataframe_append_rows(df, rows: list[dict[str, Any]], *, backend: str | None = None):
    """Append rows to a dataframe-like object, preserving the existing backend."""
    rows = [dict(row) for row in (rows or [])]
    if not rows:
        return clone_dataframe(df)

    base_rows = dataframe_to_rows(df)
    return _rebuild_dataframe(df, base_rows + rows, backend=backend)


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
    return _rebuild_dataframe(df, kept + rows, backend=backend)


def dataframe_read_delimited(
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

    def _drop_none(**kwargs):
        return {key: value for key, value in kwargs.items() if value is not None}

    resolved = select_dataframe_backend(backend)

    if resolved == 'polars':
        import polars as pl

        native = pl.read_csv(
            source,
            **_drop_none(
                separator=separator,
                infer_schema_length=infer_schema_length,
                encoding=encoding,
                null_values=null_values,
                low_memory=low_memory,
            ),
        )

    elif resolved == 'pandas':
        import pandas as pd

        native = pd.read_csv(
            source,
            **_drop_none(
                sep=separator,
                encoding=encoding,
                na_values=null_values,
            ),
        )

    else:
        import pyarrow.csv as pacsv

        native = pacsv.read_csv(
            source,
            read_options=pacsv.ReadOptions(encoding=encoding or 'utf8'),
            parse_options=pacsv.ParseOptions(delimiter=separator),
            convert_options=pacsv.ConvertOptions(null_values=null_values)
            if null_values is not None
            else pacsv.ConvertOptions(),
        )

    return _from_nw(_to_nw(native))


def dataframe_read_tsv(source, *, backend: str | None = 'auto'):
    return dataframe_read_delimited(source, separator='\t', backend=backend)


def dataframe_read_excel(source, *, sheet_name=None):
    """Read an Excel sheet into a dataframe via pandas."""
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


def dataframe_write_csv(df, path) -> None:
    """Write a dataframe-like object to CSV."""
    _to_nw(df).write_csv(path)


def dataframe_write_parquet(df, path) -> None:
    """Write a dataframe-like object to Parquet."""
    _to_nw(df).write_parquet(path)


def dataframe_read_parquet(path, *, backend: str | None = None):
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


def _empty_like(df):
    schema = {} if df is None else _schema_spec(df)
    backend = 'auto' if df is None else dataframe_backend(df)
    return dataframe_from_columns({}, schema=schema, backend=backend)


def _rebuild_dataframe(df, rows: list[dict[str, Any]], *, backend: str | None = None):
    return dataframe_from_rows(
        rows,
        schema=_schema_spec(df, rows),
        backend=dataframe_backend(df, default=backend or 'auto'),
    )


def _to_nw(df):
    return nw.from_native(df, eager_only=True)


def _from_nw(df):
    return df.to_native()


def _build_nw_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    rows = [dict(row) for row in (rows or []) if isinstance(row, dict)]
    nw_schema = _normalize_schema(schema)
    names = list(nw_schema.names()) if nw_schema is not None else []
    for row in rows:
        for name in row:
            if name not in names:
                names.append(name)
    cols = {name: [row.get(name) for row in rows] for name in names}
    return _build_nw_from_columns(cols, schema=nw_schema, backend=backend)


def _build_nw_from_columns(
    columns: dict[str, list[Any]],
    *,
    schema: nw.Schema | dict[str, str] | None,
    backend: str,
):
    cols = {name: list(values) for name, values in (columns or {}).items()}
    nw_schema = _normalize_schema(schema)
    if nw_schema is not None:
        row_count = len(next(iter(cols.values()), []))
        for name in nw_schema.names():
            cols.setdefault(name, [None] * row_count)
    df = nw.from_dict(cols, backend=backend)
    return _cast_nw_to_schema(df, nw_schema) if nw_schema is not None else df


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


def _schema_spec(df=None, rows: list[dict[str, Any]] | None = None) -> dict[str, str]:
    spec: dict[str, str | None] = {}
    if df is not None:
        schema = _to_nw(df).collect_schema()
        spec.update({name: _kind_from_dtype(schema[name]) for name in schema.names()})
    for row in rows or []:
        for name, value in row.items():
            spec[name] = _merge_kind(spec.get(name), _kind_for_value(value))
    return {name: (_TEXT if kind is None else kind) for name, kind in spec.items()}


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
    if isinstance(value, int):
        return _INT
    if isinstance(value, float):
        return _FLOAT
    if isinstance(value, (list, tuple)):
        return _LIST_TEXT
    return _TEXT


def _merge_kind(left: str | None, right: str | None) -> str | None:
    kinds = {left, right} - {None}
    if not kinds:
        return None
    if len(kinds) == 1:
        return next(iter(kinds))
    if kinds & {_TEXT, _LIST_TEXT}:
        return _TEXT
    if _FLOAT in kinds:
        return _FLOAT
    if _INT in kinds:
        return _INT
    return _TEXT


def is_polars_dataframe(df) -> bool:
    """Return whether ``df`` is a live Polars dataframe."""
    try:
        nwd = _to_nw(df)
        return nwd.implementation.is_polars()
    except (TypeError, AttributeError):
        return False


def _pl_numeric_supertype(lc, rc):
    """Return the wider of two Polars numeric dtypes."""
    import polars as pl

    float_priority = {pl.Float32: 1, pl.Float64: 2}
    int_priority = {
        pl.Int8: 1,
        pl.Int16: 2,
        pl.Int32: 3,
        pl.Int64: 4,
        pl.UInt8: 1,
        pl.UInt16: 2,
        pl.UInt32: 3,
        pl.UInt64: 4,
    }
    lct, rct = type(lc), type(rc)
    if lct in float_priority or rct in float_priority:
        return pl.Float64 if pl.Float64 in (lct, rct) else pl.Float32
    return lc if int_priority.get(lct, 0) >= int_priority.get(rct, 0) else rc


def _pl_align_schemas(left, right):
    """Align two Polars dataframes to the same columns and compatible dtypes."""
    import polars as pl

    for c in left.columns:
        if c not in right.columns:
            right = right.with_columns(pl.lit(None).alias(c))
    for c in right.columns:
        if c not in left.columns:
            left = left.with_columns(pl.lit(None).alias(c))
    for c in left.columns:
        lc, rc = left.schema[c], right.schema[c]
        if lc == pl.Null and rc != pl.Null:
            left = left.with_columns(pl.col(c).cast(rc))
        elif rc == pl.Null and lc != pl.Null:
            right = right.with_columns(pl.col(c).cast(lc).alias(c))
        elif lc != rc:
            if lc.is_numeric() and rc.is_numeric():
                st = _pl_numeric_supertype(lc, rc)
                left = left.with_columns(pl.col(c).cast(st))
                right = right.with_columns(pl.col(c).cast(st).alias(c))
            else:
                left = left.with_columns(pl.col(c).cast(pl.Utf8))
                right = right.with_columns(pl.col(c).cast(pl.Utf8).alias(c))
    return left, right.select(left.columns)


def polars_upsert_vertices(df, norm: list[tuple]):
    """Polars-native upsert of ``(vertex_id, attrs)`` pairs."""
    import polars as pl

    if not isinstance(df, pl.DataFrame):
        df = _from_nw(_to_nw(df))

    keys: set[str] = {k for _, attrs in norm for k in attrs}
    norm_vids = [vid for vid, _ in norm]

    if not all(isinstance(vid, str) for vid in norm_vids):
        return None

    if not keys:
        incoming = pl.DataFrame({'vertex_id': norm_vids})
        if df.height == 0:
            return incoming.select(df.columns) if df.columns else incoming
        to_insert = incoming.join(df.select('vertex_id'), on='vertex_id', how='anti')
        if to_insert.height:
            df, to_insert = _pl_align_schemas(df, to_insert)
            return pl.concat([df, to_insert], how='vertical', rechunk=False)
        return df

    norm_attrs = [attrs for _, attrs in norm]
    cols: dict = {'vertex_id': norm_vids}
    for k in keys:
        cols[k] = [a.get(k) for a in norm_attrs]
    incoming = pl.DataFrame(cols, nan_to_null=True, strict=False)

    nrows = len(df)
    id_df = df.select('vertex_id') if ('vertex_id' in df.columns and nrows > 0) else None

    if id_df is None:
        to_insert, to_update = incoming, None
    else:
        to_insert = incoming.join(id_df, on='vertex_id', how='anti')
        to_update = incoming.join(id_df, on='vertex_id', how='semi')

    if to_insert is not None and len(to_insert) > 0:
        df, to_insert = _pl_align_schemas(df, to_insert)
        df = pl.concat([df, to_insert], how='vertical', rechunk=False)

    if to_update is not None and len(to_update) > 0:
        suffix = '__new'
        df = df.drop([c for c in df.columns if c.endswith(suffix)])
        to_update = to_update.drop([c for c in to_update.columns if c.endswith(suffix)])
        df, to_update = _pl_align_schemas(df, to_update)
        df2 = df.join(to_update, on='vertex_id', how='left', suffix=suffix)
        exprs, drops = [], []
        for k in keys:
            nk = k + suffix
            if k in df2.columns and nk in df2.columns:
                exprs.append(pl.coalesce([pl.col(nk), pl.col(k)]).alias(k))
                drops.append(nk)
        if exprs:
            df2 = df2.with_columns(exprs)
        if drops:
            df2 = df2.drop(drops)
        df = df2

    return df
