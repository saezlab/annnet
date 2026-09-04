# Layers

Multilayer helpers from `annnet.core._Layers`.

Use `G.layers` and the `AnnNet` layer methods for layer workflows. Direct
imports from underscore modules follow the [internal API policy](../api-boundary.md).

::: annnet.core._Layers.LayerAccessor
    options:
      filters: public
      show_root_heading: true

## Aspects

An aspect's values, and whether they come one before another. See
[Aspects, order, and windows](../../explanations/aspects-and-windows.md).

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
[Node-layer values and scale](../../explanations/values-and-scale.md).

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
