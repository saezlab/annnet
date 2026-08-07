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

- the canonical store, which holds every entity and every edge
- the attribute columns
- slice membership
- multilayer state
- annotation tables
- history and snapshots
- caches
- backend adapters and lazy interoperability accessors

These concerns are coordinated views of the same graph state, not independent
subsystems.

The structural single source of truth is the canonical in-memory graph model.

## The role of `annnet.core`

The `annnet.core` package is where the in-memory model lives.

- `graph.py`
  The `AnnNet` class itself. This is where the namespaces are assembled and
  where the public mutation surface is defined.
- `_state.py`
  What a graph is made of: the field inventory, declared once, and the
  initialization that fills it.
- `_store.py`
  The slot-addressed canonical store. Topology lives here and nowhere else.
- `_mutate.py`
  The mutation gateway, and the only module that writes the canonical store
  element by element.
- `_build.py`
  The single construction path for a whole graph: copy, subgraph, flatten and
  every input-output load.
- `_structure.py`
  The read-only structural query facade, and the one boundary between the store
  and the rest of the package.
- `_attrs.py`
  The slot-indexed attribute columns, the contextual stores, and the node and
  edge tables derived from them.
- `_matrices.py`
  The named matrices, built from the member lists, and the cache that keeps one
  against the clock of the store.
- `_records.py`
  What is left after the store took over: the slice record, the `EdgeView` shape
  `get_edge` returns, and the reserved attribute names.
- `_Annotation.py`
  The public `G.attrs` surface over the attribute columns. It stores nothing of
  its own.
- `_Layers.py`
  Multi-aspect and multilayer semantics: aspect declarations, elementary
  layers, supra-node presence, supra-matrices, and layer-derived operators.
- `_Slices.py`
  Named graph contexts over the same underlying structure.
- `_Views.py`
  Lazy filtered views that read from the same graph instead of materializing
  copies.
- `_Matrix.py`
  Translation between graph identities and the coordinates of a materialized
  matrix, behind `G.idx`.
- `_Ops.py`
  Materialized copy/subgraph operations and topology-oriented graph transforms.
- `_History.py`
  Mutation logging, exported history, snapshots, and diffs.

## Structural state versus overlays

The most useful distinction in the architecture is this:

- structural state says what the graph is
- overlays say in which context that structure is being considered

Structural state is the canonical store: the entities, the edges, and the
member lists that say which entity takes which role in which edge.

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

- every named matrix: `G.A`, `G.B`, `G.H`, `G.S` and `G.L`
- the node table and the edge table, `G.obs` and `G.var`
- graph views
- subgraphs and reversed graphs
- backend graphs for NetworkX, igraph, and graph-tool

This has two consequences.

First, the package avoids fragmenting topology across several competing stores.
Second, it explains why cache invalidation and view logic are part of the core
architecture rather than afterthoughts.

Mutation writes the canonical store. A derived structure rebuilds from it when
the clock of the store has moved past the one that structure recorded.

## Public namespaces follow the architecture

The current manager-first public API mirrors the internal split:

- `G.layers` for multilayer state
- `G.slices` for slice state
- `G.idx` for incidence-coordinate translation
- `G.cache` for derived matrix materializations
- `G.ops` for materialized graph operations

This is not just naming preference. It is an architectural statement about
which concerns are canonical, which are overlays, and which are derived.

## The public surface names no position

A position belongs to one materialized matrix, so no public name hands one back.
`G.get_node` and `G.get_edge` take an id. `G.N[n]` is the n-th node of the node
sequence. `G.idx` translates a coordinate a caller already holds, in both
directions, and `G.views.entity_kinds()` reads the kind of each entity.

The maps from an id to a position that earlier releases exposed are gone. They
described one materialization as though it were the model.

## Outside `annnet.core`

The rest of the package has a simpler split:

- `annnet.algorithms`
  Algorithms that operate against AnnNet's internal model.
- `annnet.adapters`
  Runtime conversion into external graph backends.
- `annnet.io`
  Persistence and exchange formats.
- `annnet.utils`
  A small public utility namespace, currently centered on plotting helpers.

- `annnet._support`
  Private cross-cutting support modules such as metadata, optional component
  detection, dataframe backend selection, plotting backend selection, and lazy
  export helpers.

Those packages sit around the core object. They do not redefine the graph
model.

Adapters and IO modules use the package's centralized dataframe helpers when
reading annotation tables or creating new tables. This keeps format adapters,
backend adapters, and graph annotations aligned with the configured annotation
backend.
