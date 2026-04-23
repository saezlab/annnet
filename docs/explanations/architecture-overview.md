# Package architecture

AnnNet is not organized as a thin graph class plus a pile of utility modules.
It is organized around one central object with a deliberately layered internal
model:

1. canonical structural state
2. contextual overlays
3. derived materializations
4. compatibility and interoperability boundaries

That split is the main mental model behind the package.

## The center: one object, one structural truth

`AnnNet` is the coordination point for the whole package. It owns:

- the sparse incidence matrix
- the canonical entity and edge registries
- slice membership
- multilayer state
- annotation tables
- history and snapshots
- caches
- backend adapters and lazy interoperability accessors

The important point is that these concerns are not independent subsystems
floating next to each other. They all describe different views of the same
graph state.

The structural single source of truth is described in detail in
[Internal representation](internal-representation.md).

## The role of `annnet.core`

The `annnet.core` package is where the in-memory model lives.

- `graph.py`
  The `AnnNet` class itself. This is where the canonical stores are created
  and where scalar graph mutation is defined.
- `_records.py`
  Shared low-level structures such as `EntityRecord`, `EdgeRecord`, and
  `SliceRecord`, plus record-oriented helpers used across the core.
- `_Annotation.py`
  Attribute storage and upsert logic for graph-, vertex-, edge-, slice-, and
  edge-slice-level metadata.
- `_Layers.py`
  Multi-aspect and multilayer semantics: aspect declarations, elementary
  layers, supra-node presence, supra-matrices, and layer-derived operators.
- `_Slices.py`
  Named graph contexts over the same underlying structure.
- `_Views.py`
  Lazy filtered views that read from the same graph instead of materializing
  copies.
- `_Matrix.py`
  Translation between external graph identifiers and incidence coordinates,
  plus matrix/cache helpers such as CSR/CSC and adjacency materialization.
- `_Ops.py`
  Materialized copy/subgraph operations and topology-oriented graph transforms.
- `_History.py`
  Mutation logging, exported history, snapshots, and diffs.

## Structural state versus overlays

The most useful distinction in the architecture is this:

- structural state says what the graph is
- overlays say in which context that structure is being considered

Structural state includes the incidence matrix, entities, edges, and the
direct adjacency indices derived from them.

Overlays include:

- slices
- multilayer coordinates and aspect registries
- annotation tables
- history

These overlays are not fake or secondary. They are first-class parts of the
object. But they are not independent graph stores. They enrich one structural
graph rather than replacing it.

## Derived materializations

Several pieces of state are intentionally derived rather than canonical:

- CSR and CSC matrix forms
- adjacency matrices
- graph views
- subgraphs and reversed graphs
- backend graphs for NetworkX, igraph, and graph-tool

This matters for two reasons.

First, the package avoids fragmenting topology across several competing stores.
Second, it explains why cache invalidation and view logic are part of the core
architecture rather than afterthoughts.

The operational side of this is covered in
[Mutation and derived state](mutation-and-derived-state.md).

## Public namespaces follow the architecture

The current manager-first public API mirrors the internal split:

- `G.layers` for multilayer state
- `G.slices` for slice state
- `G.idx` for incidence-coordinate translation
- `G.cache` for derived matrix materializations
- `G.ops` for materialized graph operations

This is not just naming preference. It is an architectural statement about
which concerns are canonical, which are overlays, and which are derived.

## Compatibility is now a boundary, not an implementation model

The codebase still exposes compatibility-oriented properties such as
`entity_to_idx`, `edge_definitions`, `edge_weights`, and similar legacy names.
Those should be understood as a public compatibility boundary, not as the
internal storage model.

Internally, the package is organized around the structured SSOT described in
[Internal representation](internal-representation.md).

That distinction matters because compatibility views do not replace the
canonical model.

## Outside `annnet.core`

The rest of the package has a simpler split:

- `annnet.algorithms`
  Algorithms that operate against AnnNet's internal model.
- `annnet.adapters`
  Runtime conversion into external graph backends.
- `annnet.io`
  Persistence and exchange formats.
- `annnet.utils`
  Validation, plotting, typing, and smaller support utilities.

Those packages sit around the core object. They do not redefine the graph
model.

Adapters and IO modules use the package's centralized dataframe helpers when
reading annotation tables or creating new tables. This keeps format adapters,
backend adapters, and graph annotations aligned with the configured annotation
backend.
