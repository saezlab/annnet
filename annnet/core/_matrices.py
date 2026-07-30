"""The derived matrices.

Every matrix here is built from the member lists of the canonical store. A
matrix is derived state. It is safe to drop, and it is never authoritative.

The package builds several purpose-built matrices instead of one matrix that
mixes every edge kind. A binary incidence matrix holds the binary edges. A
hypergraph incidence matrix holds the hyperedges. A signed coefficient incidence
matrix holds every edge with its coefficients. An adjacency matrix and a
Laplacian follow from the binary edges. A rule that one matrix needs, for example
how to place a boundary edge, lives in that matrix and does not reach the
canonical store or another matrix.

Each matrix records the clock value it was built at. It rebuilds when the clock
of the store has moved.
"""

from __future__ import annotations
