# In-memory architecture

This page is for contributors changing the implementation. For the user-facing
model, start with [The graph model](../guide/graph-model.ipynb).

## Canonical state

`annnet.core._store.CoreState` holds entity and edge identities, scalar edge
fields, and pooled participant arrays. Each participant entry names an entity
slot, role, and coefficient. A self-loop retains separate source and target
roles even though they refer to the same entity. Matrix construction may sum
those entries; the canonical structure retains both.

An entity key contains a node or edge-entity ID and a layer coordinate. Slots
are internal addresses: deletion can free a slot for reuse, while public
annotations and exchange data refer to identities. Maintained lookups connect
bare node IDs to placements and entity slots to incident edges.

Generic node and edge attributes live in `_attrs.AttributeStore` as slot-indexed
columns. `_contextual.ContextualStore` holds six keyed dictionaries for slice,
edge-slice, node-layer, aspect, layer, and elementary-layer attributes.
Dataframes are materialized at the public table boundary. Changing the table
backend should not change the graph's canonical state.

## Read and write boundaries

| Module | Responsibility |
| --- | --- |
| `_structure.py` | Structural queries used by readers and adapters. |
| `_mutate.py` | Element-level structural writes and maintained indices. |
| `_build.py` | Installing complete graph state for copies and readers. |
| `_attribute_api.py` | Public `G.attrs` reads, validation, updates, and selection. |
| `_transaction.py` | Atomic attribute transactions. |
| `_resolve.py`, `_select.py`, `_Views.py` | Selection resolution and read-only graph views. |
| `_Layers.py`, `_Slices.py` | Layer coordinates and named memberships. |
| `_matrices.py`, `_Matrix.py` | Sparse matrix construction, caching, and identity maps. |

Slices hold memberships in the shared structure. Layer placements are part of
entity identity; aspect declarations and layer metadata describe their
coordinate system. Neither should be inferred from incidental row positions.

## Derived data and invalidation

Each named matrix selects the edge kinds and numerical convention it needs.
`G.B` covers binary and node-edge relations, `G.H` reports hyperedge membership,
and `G.S` contains coefficients of every structural edge. Adjacency and the
Laplacian have their own builders. Consult `tests/test_named_matrices.py` and
`tests/test_selfloop_boundary.py` before changing these rules.

The matrix cache checks the structural store's version. Incidence caches can
extend after eligible frontier appends; adjacency and Laplacian caches rebuild.
Avoid claiming that every matrix read after any append has constant cost.

Attribute selections and graph views depend on more than topology. Their state
clock includes generic and contextual attribute versions, slice membership,
and aspect declarations. Callable filters can also depend on external state
and must be evaluated accordingly.

The history counter is a separate mechanism. Its hooks currently do not log
all removals or manager-level mutations. Backend accessors still cache against
that counter, so their projections can become stale after some edits. Matrix
and selection invalidation should not be described as guaranteeing backend
cache freshness. The user-facing workaround is a fresh adapter conversion or
an explicit accessor `clear()`.

## Testing changes

Use existing behavior tests to cover the state you change: identities and slot
lifecycle, named matrices, selection/view liveness, attribute transactions,
and adapter round trips. Check actual endpoints, coefficients, IDs, and
contextual values; matching node and edge counts alone does not establish an
equivalent graph.

The [performance report](performance.md) records measured costs and workloads.
Keep benchmark results separate from semantic guarantees and qualify timing
claims by their workload and environment.
