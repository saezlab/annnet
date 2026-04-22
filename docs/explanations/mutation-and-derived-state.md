# Mutation and derived state

The internal model of AnnNet is clean because mutation follows a disciplined
rule:

- canonical structural stores are updated first
- derived indices are synchronized immediately when cheap
- expensive materializations are rebuilt lazily

This page explains how that rule shows up in practice.

## Scalar mutation follows the structural path

The scalar mutation API in `graph.py` works against the canonical stores.

Typical `add_vertex(...)` flow:

1. normalize the layer coordinate
2. resolve placeholder behavior if needed
3. register or reuse the entity record
4. ensure matrix row capacity
5. update slice membership
6. upsert attributes

Typical `add_edge(...)` flow:

1. parse the endpoint specification
2. infer the structural edge type
3. validate multilayer explicitness
4. ensure endpoint entities exist
5. allocate or reuse the matrix column
6. write the incidence coefficients
7. write the `EdgeRecord`
8. update adjacency indices
9. update slice membership and attributes

That is the key sequencing principle: matrix and records move together.

## Bulk mutation does not define a second model

`_BulkOps.py` is easy to misread if you assume it is an alternate storage path.
It is not.

The bulk methods:

- `add_vertices_bulk(...)`
- `add_edges_bulk(...)`
- `add_hyperedges_bulk(...)`
- batched slice membership operations

exist to amortize repeated overhead:

- matrix growth
- dataframe upserts
- identifier generation
- endpoint registration

The semantics are supposed to match the scalar API. The difference is only in
execution shape.

## Immediate derived updates versus lazy rebuilds

AnnNet maintains some derived structures eagerly and others lazily.

### Updated eagerly

These are cheap enough, or central enough, to keep in sync during mutation:

- `_row_to_entity`
- `_col_to_edge`
- `_adj`
- `_src_to_edges`
- `_tgt_to_edges`
- `_vid_to_ekeys`

Without them, even ordinary graph operations would degrade badly.

### Rebuilt lazily

These are materializations rather than canonical state:

- CSR and CSC matrices
- adjacency matrices
- backend graphs from `G.nx`/`G.ig`/`G.gt` lazy accessors
- certain layer-specific matrix views

They are either cached behind version checks or rebuilt on demand.

This is why cache invalidation is part of the architecture rather than a
separate optimization story.

## Why `_version` exists

Mutations advance a version counter. Derived materializations use that counter
to decide whether their cached state is still valid.

This is a pragmatic choice.

It avoids trying to make every materialization update incrementally under every
possible mutation path. Instead, AnnNet keeps the structural model coherent and
lets derived caches rebuild when needed.

## Slices are membership overlays

A slice is not a second graph. It is a named context over the same structure.

Mutation therefore has to answer two different questions:

1. what changed structurally in the graph?
2. in which slices should that structure be considered present?

This is why edge insertion includes propagation logic and why slices carry both
vertex and edge memberships.

The structural edge is global. Slice membership is contextual.

## Layering is a coordinate system, not a duplicated topology

Multilayer mutation works because the row identity already includes the layer
coordinate.

That leads to a clean distinction:

- adding presence changes which supra-node rows exist
- adding edges changes which columns exist and which rows they touch

The package does keep some derived multilayer indices such as `_VM`,
`_nl_to_row`, and `_row_to_nl`, but those are execution helpers over the same
entity registry.

## Placeholder mutation is deliberate

Placeholder coordinates are not a loose convention. They are a disciplined
fallback.

When vertices are inserted without explicit layer placement in a layered graph,
AnnNet:

- assigns the placeholder coordinate
- warns explicitly
- keeps the row invariant intact

When aspects are declared over an existing flat graph, the same logic is used:

- previous flat vertices are lifted into the placeholder coordinate
- the graph becomes multilayer without inventing a second row semantics

This is a subtle but important design choice. It prevents the matrix from
containing a mix of "real supra-node rows" and "global unresolved rows".

## Views, subgraphs, and copies are deliberately different

AnnNet has several ways to work with a subset of the graph. They should not be
confused.

### Views

`G.view(...)` creates a lazy filtered lens over the same graph.

- no structural copy
- no new source of truth
- filtering happens at access time

### Subgraphs

`subgraph(...)`, `edge_subgraph(...)`, and related operations materialize a new
graph object.

- structure is copied
- relevant attributes are copied
- slice memberships may be restricted or rebuilt

### Copy

`copy()` preserves the graph shape more faithfully than a subgraph operation.
It is the right tool when you want another graph object with the same topology
and metadata, not a filtered projection.

## History records mutations, not abstract intent

History hooks wrap mutating methods and record:

- operation name
- version
- timestamp
- monotonic clock
- captured arguments
- result

This matters for interpretation.

The history system logs what the API call did, not a reconstructed semantic
"meaning" of the mutation after the fact. That keeps the log simple, explicit,
and serializable.
