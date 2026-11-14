# Layers

`anet.layers` declares the aspects of a multilayer graph, places nodes on
layers, selects layers, couples them, and holds the values of nodes on each
layer. See the [layers guide](../../guide/layers.ipynb).

::: annnet.core._Layers.LayerAccessor
    options:
      filters: public
      show_root_heading: true

## Aspects

An aspect's values, and whether they come one before another. See
[Aspects, order, and windows](../../guide/layers.ipynb).

::: annnet.core._aspects.Aspect
    options:
      show_root_heading: true

::: annnet.core._aspects.OrderedLabels
    options:
      show_root_heading: true

::: annnet.core._aspects.BOUNDARIES
    options:
      show_root_heading: true

::: annnet.core._aspects.require_boundary
    options:
      show_root_heading: true

::: annnet.core._aspects.as_aspect
    options:
      show_root_heading: true

## Layer selection

`anet.layers.where(...)` names layers by predicate; `anet.layers.select(...)`
returns the nodes and edges they hold as a `LayerSet`.

::: annnet.core._Layers.LayerSet
    options:
      show_root_heading: true

::: annnet.core._selection.LayerSelection
    options:
      show_root_heading: true

::: annnet.core._selection.parse_predicate
    options:
      show_root_heading: true

::: annnet.core._selection.satisfies
    options:
      show_root_heading: true

## Node-layer values

The two backings a value may live in, the resolver over them, and the array a
method is handed. See
[Node-layer values and scale](../../guide/annotations-and-views.ipynb).

::: annnet.core._values.ValueMatrix
    options:
      show_root_heading: true

::: annnet.core._values.MatrixValues
    options:
      show_root_heading: true

::: annnet.core._values.ContextualValues
    options:
      show_root_heading: true

::: annnet.core._values.ValueResolver
    options:
      show_root_heading: true

::: annnet.core._values.ValueBacking
    options:
      show_root_heading: true

## Multilayer measures

Measures and dynamics over the supra-graph are functions of the graph in
[`annnet.algorithms.multilayer`](../algorithms/multilayer.md).
