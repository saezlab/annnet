# Attributes

Attribute table helpers from `annnet.core._Annotation`.

These methods are mixed into `AnnNet`. Direct imports from underscore modules
follow the [internal API policy](../api-boundary.md).

`AttributesAccessor` is what `G.attrs` gives back: the eight attribute tables and
the setters that write them. For how those tables relate to the frames under
`G.views`, and to the older `obs`/`var`/`slice_attributes` spellings, see
[Reading the graph](../../explanations/reading-the-graph.md) and
[Internal representation](../../explanations/internal-representation.md).

::: annnet.core._Annotation.AttributesClass
    options:
      filters: public
      show_root_heading: true

::: annnet.core._Annotation.AttributesAccessor
    options:
      filters: public
      show_root_heading: true
