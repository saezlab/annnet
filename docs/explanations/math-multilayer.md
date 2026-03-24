# Multilayer and multi-aspect graphs

annnet treats multilayer structure as part of the graph model rather than as a
post-processing trick over a flat graph.

## Aspects and layer tuples

A multilayer graph is built from one or more aspects. An aspect is one axis of
variation, such as:

- time
- modality
- condition
- cell state
- spatial compartment

Each aspect has a set of elementary layers. A concrete layer in the graph is a
layer tuple formed by taking one elementary layer from each aspect.

If the aspects are `time = {t0, t1}` and `modality = {rna, protein}`, then the
possible layer tuples are:

- `(t0, rna)`
- `(t0, protein)`
- `(t1, rna)`
- `(t1, protein)`

This is what “multi-aspect” means in annnet: a layer is not just one label, but
potentially a tuple over several dimensions.

## What is assigned to layers

annnet tracks several kinds of multilayer state:

- which aspects exist
- which elementary layers belong to each aspect
- which vertices are present in which layer tuples
- whether edges are intra-layer, inter-layer, or coupling edges
- layer-level and vertex-layer annotations

That lets the package describe the multilayer graph explicitly instead of
encoding layer membership indirectly in names or attributes.

## Supra representations

For computation, the multilayer graph can be unfolded into supra structures over
`(vertex, layer)` pairs.

- Supra incidence uses the same sign logic as the monolayer incidence model.
- Supra adjacency expresses relations between vertex-layer pairs.
- Derived operators such as Laplacians or transition operators can then be built
  in the usual matrix form.

The important point is that these are derived views of the multilayer state, not
separate primary data structures that replace the graph.

## Why this matters

A plain collection of annotated edges cannot easily distinguish:

- a vertex missing from a layer
- a vertex present but isolated in a layer
- an intra-layer edge
- an inter-layer edge
- a coupling edge connecting the same vertex across layers

annnet models those distinctions directly.

## Relation to slices

Layers and slices solve different problems.

- Layers describe a structured space of graph state across one or more aspects.
- Slices describe named graph contexts or subgraphs.

A timepoint can be modeled as a layer if it is part of the graph's intrinsic
multilayer structure. The same timepoint can also appear in a slice if you need
a named analysis context. The two ideas are related, but not interchangeable.
