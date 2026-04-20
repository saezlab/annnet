"""Central dataframe backend helpers.

AnnNet accepts Narwhals-compatible dataframe inputs, but newly-created internal
tables need a concrete eager backend. Backend selection is centralized here so
callers do not each implement their own fallback chain.
"""

from __future__ import annotations

from importlib import util
from typing import Any

import narwhals as nw

DATAFRAME_BACKEND_PRIORITY = ("polars", "pandas", "pyarrow")
_TEXT = "text"
_FLOAT = "float"
_BOOL = "bool"
_LIST_TEXT = "list_text"


def available_dataframe_backends() -> dict[str, bool]:
    """Return installed dataframe backends in AnnNet preference order."""
    modules = {"polars": "polars", "pandas": "pandas", "pyarrow": "pyarrow"}
    return {name: util.find_spec(module) is not None for name, module in modules.items()}


def select_dataframe_backend(preferred: str | None = "auto") -> str:
    """Resolve a dataframe backend name.

    ``"auto"`` selects the first installed backend in this order: Polars,
    pandas, then PyArrow.
    """
    requested = "auto" if preferred is None else str(preferred).lower()
    available = available_dataframe_backends()

    if requested == "auto":
        for backend in DATAFRAME_BACKEND_PRIORITY:
            if available[backend]:
                return backend
        raise RuntimeError("No dataframe backend available. Install polars, pandas, or pyarrow.")

    if requested not in DATAFRAME_BACKEND_PRIORITY:
        allowed = ", ".join(("auto", *DATAFRAME_BACKEND_PRIORITY))
        raise ValueError(f"Unknown dataframe backend {preferred!r}; expected one of: {allowed}.")

    if not available[requested]:
        raise RuntimeError(
            f"Dataframe backend {requested!r} is not installed. "
            "Install polars, pandas, or pyarrow, or use annotations_backend='auto'."
        )
    return requested


def dataframe_from_rows(
    rows: list[dict[str, Any]] | list[Any],
    *,
    schema: dict[str, str] | None = None,
    backend: str | None = "auto",
):
    """Build an eager dataframe/table using the selected backend."""
    resolved = select_dataframe_backend(backend)
    rows = list(rows or [])

    if resolved == "polars":
        import polars as pl

        if rows:
            return pl.DataFrame(rows)
        return pl.DataFrame(schema=_polars_schema(schema or {}))

    if resolved == "pandas":
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


def empty_dataframe(schema: dict[str, str], *, backend: str | None = "auto"):
    """Build an empty dataframe/table with a generic schema."""
    return dataframe_from_rows([], schema=schema, backend=backend)


def dataframe_to_rows(df) -> list[dict[str, Any]]:
    """Return rows from a Narwhals-compatible eager dataframe/table."""
    if df is None:
        return []
    if hasattr(df, "to_dicts"):
        return [dict(row) for row in df.to_dicts()]
    if hasattr(df, "to_dict"):
        try:
            return [dict(row) for row in df.to_dict(orient="records")]
        except TypeError:
            pass
    if hasattr(df, "to_pylist"):
        return [dict(row) for row in df.to_pylist()]

    ndf = nw.from_native(df, eager_only=True)
    return [dict(zip(ndf.columns, row)) for row in ndf.rows()]


def dataframe_height(df) -> int:
    """Return the row count for a dataframe-like object."""
    if df is None:
        return 0
    if hasattr(df, "height"):
        return int(df.height)
    if hasattr(df, "num_rows"):
        return int(df.num_rows)
    if hasattr(df, "shape"):
        return int(df.shape[0])
    return len(dataframe_to_rows(df))


def _polars_schema(schema: dict[str, str]):
    import polars as pl

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
        return "float64"
    if kind == _BOOL:
        return "bool"
    if kind == _LIST_TEXT:
        return "object"
    return "string"


def _pyarrow_type(kind: str):
    import pyarrow as pa

    if kind == _FLOAT:
        return pa.float64()
    if kind == _BOOL:
        return pa.bool_()
    if kind == _LIST_TEXT:
        return pa.list_(pa.string())
    return pa.string()
