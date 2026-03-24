# Package architecture

annnet combines one graph object with several cooperating subsystems. The goal
is to keep expressive graph structure, annotations, views, and conversion logic
in one coherent model.

## The central object

`AnnNet` is the main container. It holds:

- graph structure
- annotation tables
- slice state
- multilayer state
- caches and history
- access points to adapters and optional backends

This is why annnet feels like more than a plain graph class with attribute
maps: the object is the coordination point for several related concerns.

## Core package responsibilities

The `annnet.core` package defines the in-memory graph model.

- `graph`: the `AnnNet` object and core graph-facing API.
- `_Index`: mappings between stable external IDs and internal matrix indices.
- `_Annotation`: tabular attribute storage for vertices, edges, layers, and slices.
- `_Slices`: named subgraph contexts and slice-specific memberships or overrides.
- `_Layers`: multilayer and multi-aspect state.
- `_Views`: read-only filtered views over the same underlying graph.
- `_History`: mutation tracking, snapshots, and diffs.
- `_Cache`: cached matrix forms and derived execution helpers.
- `_BulkOps`: high-volume structural and annotation updates.

## How the pieces fit together

At the center is the sparse incidence representation. Around it, annnet keeps
explicit ID maps, table-backed annotations, and optional state for slices,
layers, and history.

A typical flow looks like this:

1. topology is stored in incidence form;
2. IDs are mapped to matrix indices through the indexing layer;
3. annotations are stored in aligned tables;
4. slices and layers add contextual structure without duplicating the graph;
5. caches materialize algorithm-friendly representations when needed;
6. adapters and IO modules expose the graph to external tools or files.

## Package-level module layout

Outside `annnet.core`, the rest of the package has a clearer operational split:

- `annnet.algorithms`: algorithms that work directly with annnet structures.
- `annnet.adapters`: in-memory conversion to external graph backends.
- `annnet.io`: persistence and exchange formats.
- `annnet.utils`: typing, validation, and plotting helpers.

## Relationship to the conceptual pages

This page explains where responsibilities live in the package. For the graph
formalism itself, read [Incidence representation](math-incidence.md). For the
layer model, read [Multilayer and multi-aspect graphs](math-multilayer.md). For
multiple contexts over one graph, read [Slices and views](managers-and-views.md).
