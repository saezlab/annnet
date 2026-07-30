"""The slot-indexed attribute columns.

One attribute is one typed array. The array is indexed by slot, so a value keeps
its place when another element goes away. A free slot holds a null.

Fixed-width numeric and boolean columns use numpy. String, categorical, and list
columns use pyarrow. This storage does not depend on the dataframe backend of
the user. The backend matters only when the node table and the edge table
materialize.

The node table and the edge table are derived. They gather the live slots of
each column in the current order and hand the result to narwhals.
"""

from __future__ import annotations
