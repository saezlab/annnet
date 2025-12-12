from enum import Enum

import polars as pl


def _get_numeric_supertype(left, right):
    """Get the supertype for two numeric dtypes.

    Returns the wider type that can hold values from both input types.
    For mixed int/float, returns float. For different sizes, returns larger size.
    """
    # Type hierarchy (lower to higher precedence)
    type_order = {
        pl.Int8: 1,
        pl.Int16: 2,
        pl.Int32: 3,
        pl.Int64: 4,
        pl.Int128: 5,
        pl.UInt8: 1,
        pl.UInt16: 2,
        pl.UInt32: 3,
        pl.UInt64: 4,
        pl.UInt128: 5,
        pl.Float32: 10,
        pl.Float64: 11,
    }

    # If either is float, result is float
    if left in (pl.Float32, pl.Float64) or right in (pl.Float32, pl.Float64):
        # Return the wider float type
        if left == pl.Float64 or right == pl.Float64:
            return pl.Float64
        return pl.Float32

    # Both are integers - return the wider one
    left_order = type_order.get(left, 0)
    right_order = type_order.get(right, 0)

    # If mixing signed and unsigned, promote to next larger signed type
    left_unsigned = left in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.UInt128)
    right_unsigned = right in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.UInt128)

    if left_unsigned != right_unsigned:
        # Mixed signed/unsigned - promote to Float64 for safety
        return pl.Float64

    # Same signedness - return the wider type
    return left if left_order >= right_order else right


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
