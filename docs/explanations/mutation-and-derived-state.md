# Mutation and derived state

The internal model of AnnNet is clean because mutation follows a disciplined
rule:

- one module writes the canonical store
- indices that must stay local are maintained during the write
- everything else is derived, and rebuilds against a clock

This page explains how that rule shows up in practice.

## One gateway writes

`annnet.core._mutate` is the mutation gateway, and it is the only module that
writes the canonical store. `annnet.core._build` is the second writer, and it
exists for the paths that install a whole graph at once: copy, subgraph,
flatten, and every input-output load.

Nothing else writes. The derive layer and the read facade only read.

Typical `add_nodes(...)` flow:

1. normalize the layer coordinate
2. resolve placeholder behavior if needed
3. take an entity slot from the freelist, or grow the arrays
4. update slice membership
5. write the attribute cells of that slot

Typical `add_edges(...)` flow:

1. parse the endpoint specification
2. infer the structural edge kind
3. validate multilayer explicitness
4. make sure the endpoint entities exist
5. take an edge slot and append its member entries to the pools
6. write the per-edge scalars
7. update the entity-to-edge index
8. update slice membership and attributes

Matrix rows and columns appear in none of those steps. A matrix is built from
the member lists when a caller asks for one.

## Bulk mutation does not define a second model

Bulk mutation is easy to misread as an alternate storage path. It is not.

`add_nodes(...)` and `add_edges(...)` take one element or a collection through
the same call, and batched slice-membership operations do the same. They exist to
amortize repeated overhead — array growth, identifier generation, endpoint
registration — not to reach a different store. A single element and a
one-element collection produce the same graph.

## Maintained versus derived

### Maintained during the write

The gateway keeps two lookups in step, because a rebuild of either would make a
local change cost the size of the graph:

- `_id_slots`, one bare node id to the entity slots it stands for
- `_entity_edges`, one entity slot to the edges that touch it

Both are rebuildable from the member lists and neither is authoritative.

### Derived

These are materializations rather than canonical state:

- every named matrix: `G.A`, `G.B`, `G.H`, `G.S` and `G.L`
- the node table and the edge table, `G.obs` and `G.var`
- backend graphs from the `G.nx`, `G.ig` and `G.gt` lazy accessors
- layer-specific matrix views

They are cached behind a clock check or rebuilt on demand. Cache invalidation is
part of the architecture, not a separate optimization layer.

## The two clocks

`structure_version` rises on every write that can change the incidence
structure. It is the clock a derived cache validates against, and the store owns
it.

`_version` is the history clock. `_History` advances it when it logs an event,
it drives snapshot and diff numbering, and it is user-visible. It does not track
structural mutation — a remove and `set_aspects` both leave it unchanged — so it
must never key a derived cache.

Keeping the two apart is what lets the package rebuild a derived structure only
when the structure actually moved.

## An append is not a rebuild

The store keeps an append log. It holds the edge slots that arrived at the
frontier since the last write that was not such an append, and the clock value
that log starts from.

A matrix cache reads that log. When every write since the cached build was a
frontier append, the cache extends the cached matrix with the new columns instead
of dropping it. A loop of N appends with one matrix read after each one therefore
costs time proportional to N, not to N squared.

## Slices are membership overlays

A slice is not a second graph. It is a named context over the same structure.

Mutation therefore answers two different questions:

1. what changed structurally in the graph?
2. in which slices should that structure be considered present?

This is why edge insertion includes propagation logic and why a slice carries
both node and edge membership.

The structural edge is global. Slice membership is contextual.

## Layering is a coordinate system, not a duplicated topology

Multilayer mutation works because the entity key already includes the layer
coordinate.

That leads to a clean distinction:

- adding presence changes which supra-node slots exist
- adding edges changes which member segments exist and which slots they name

The package builds a supra index on demand, behind `G.layers.nl_to_row` and
`G.layers.row_to_nl`, but that is an execution helper over the same entity
slots.

## A row cannot name an element the graph does not hold

A slot indexes a column, and a slot lives only as long as its element. So a
table cannot carry a row for an element the graph never held. The rule is
structural rather than checked. The internal validator used to test it, and that
check now no longer exists rather than passing.

## Placeholder mutation is deliberate

Placeholder coordinates are a disciplined fallback, not a loose convention.

When nodes are inserted without explicit layer placement in a layered graph,
AnnNet:

- assigns the placeholder coordinate
- warns explicitly
- keeps the entity-key invariant intact

When aspects are declared over an existing flat graph, the same logic applies:
previous flat nodes are lifted into the placeholder coordinate, and the graph
becomes multilayer without inventing a second key semantics.

This prevents a mix of real supra-node entities and globally unresolved ones.

## Views, subgraphs and copies are deliberately different

AnnNet has several ways to work with a subset of a graph, and they should not be
confused.

### Views

`G.view(...)` creates a lazy filtered lens over the same graph.

- no structural copy
- no new source of truth
- filtering happens at access time

### Subgraphs

`G.ops.subgraph(...)`, `G.ops.edge_subgraph(...)` and related operations
materialize a new graph object.

- structure is copied
- relevant attributes are copied
- slice memberships may be restricted or rebuilt

### Copy

`G.ops.copy()` preserves the graph shape more faithfully than a subgraph
operation. It is the right tool when you want another graph object with the same
topology and metadata, not a filtered projection.

## History records mutations, not abstract intent

History hooks wrap mutating methods and record:

- operation name
- version
- timestamp
- monotonic clock
- captured arguments
- result

The history system logs what the API call did, not a reconstructed semantic
meaning of the mutation after the fact. That keeps the log simple, explicit and
serializable.
