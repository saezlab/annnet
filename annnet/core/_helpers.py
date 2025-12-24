from enum import Enum

import narwhals as nw

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

    def _dtype_cls(dt):
        return dt.base_type() if hasattr(dt, "base_type") else dt

    left_cls = _dtype_cls(left)
    right_cls = _dtype_cls(right)

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
    try:
        import polars as pl

        return pl.DataFrame(rows)
    except Exception:
        try:
            import pandas as pd

            return pd.DataFrame.from_records(rows)
        except Exception:
            raise RuntimeError(
                "No dataframe backend available. Install polars (recommended) or pandas."
            )


def _df_filter_not_equal(df, col: str, value):
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.filter(pl.col(col) != value)
    import narwhals as nw

    ndf = nw.from_native(df)
    return nw.to_native(ndf.filter(nw.col(col) != value))


class EdgeType(Enum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"


_vertex_RESERVED = {"vertex_id"}  # nothing structural for vertices
_EDGE_RESERVED = {
    "edge_id",
    "source",
    "target",
    "weight",
    "edge_type",
    "directed",
    "slice",
    "slice_weight",
    "kind",
    "members",
    "head",
    "tail",
}
_slice_RESERVED = {"slice_id"}
