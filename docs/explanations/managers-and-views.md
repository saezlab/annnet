# Slices and views

Slices and views let one `AnnNet` object support multiple contexts without
forcing full graph copies.

## Manager-first API

The primary public API is organized around managers:

- `G.layers` for multilayer setup, layer metadata, presence, and supra-layer views
- `G.slices` for named slice contexts
- `G.idx` for entity/edge to incidence-coordinate translation
- `G.cache` for cached or alternative topology representations

Direct graph methods still exist in a few places for compatibility, but the
preferred style in new code is manager-first.

## What a slice is

A slice is a named graph context attached to the same underlying object. A slice
can store:

- vertex memberships
- edge memberships
- slice-level metadata
- edge-level overrides such as slice-specific weights

This makes slices useful for cases such as:

- experimental conditions
- perturbation scenarios
- train and test partitions
- condition-specific subgraphs
- alternative weighting schemes over the same structure

## Why slices exist

Without slices, these contexts usually become separate graph copies. That leads
to duplicated structure, inconsistent updates, and extra bookkeeping. annnet
instead keeps one graph and lets slices describe which parts are active or how
specific attributes change in a given context.

## Active slice and propagation

The graph keeps one active slice, and many operations use that context by
default. When mutating the graph, propagation rules determine whether a change:

- stays local to the current slice,
- is shared with related slices, or
- affects the global structure.

This makes slice handling explicit instead of implicit.

## What a view is

A view is a filtered, read-only lens over the graph. It is useful when you want
to inspect or compute on a selected part of the graph without materializing a
new object.

- `G.view(...)` creates a filtered view.
- `G.subgraph(...)` materializes a vertex-induced graph.
- `G.edge_subgraph(...)` materializes an edge-induced graph.
- `G.extract_subgraph(...)` combines both filters.

The distinction matters: a view is temporary and lightweight, while a subgraph
is a separate graph object.

## Relation to layers

Slices are named contexts. Layers are structured coordinates in a multilayer
space. If you need explicit multi-aspect graph semantics, use layers. If you
need several analysis states over one graph, use slices.

## Placeholder layer coordinates

When aspects are declared before every vertex has an explicit coordinate,
AnnNet uses the placeholder layer tuple `('_', ..., '_')`.

- Declaring aspects on an existing flat graph lifts existing vertices to the
  placeholder tuple and emits a warning.
- Adding vertices to a layered graph without `layer=` places them at the
  placeholder tuple and emits a warning.
- Structural edge construction in layered graphs still requires explicit
  supra-node endpoints such as `(vertex_id, layer_tuple)`.
