# Attributes

The attribute API from `annnet.core._attribute_api` and the row selections from
`annnet.core._select`. Direct imports from underscore modules follow the
[internal API policy](../api-boundary.md); the public names are exported as
`annnet.Attrs`, `annnet.RowSelection` and `annnet.Schema`.

`Attrs` is what `G.attrs` gives back: the eight attribute addresses — `nodes`,
`edges`, `slices`, `aspects`, `layers`, `edge_slices`, `node_layers`,
`elementary_layers` — read and written the same way. A read (`nodes`, `table`,
`row`, `rows`) returns a detached copy; a write is a named call (`update`,
`replace`, `delete`); `select` is a live query and `from_frame` a fixed
selection made from the rows of a table you filtered yourself. The contract is stated in
[the API contract](../api-contract.md), section 3; the reading workflow in
[Reading the graph](../../guide/annotations-and-views.ipynb); the storage in
[Internal representation](../../guide/annotations-and-views.ipynb).

::: annnet.core._attribute_api.Attrs
    options:
      filters: public
      show_root_heading: true

::: annnet.core._attribute_api.Schema
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.RowSelection
    options:
      filters: public
      show_root_heading: true

## Predicates

The one predicate language every `select` parses.

::: annnet.core._predicate
    options:
      members:
        - OPERATORS
        - parse_conditions
      show_root_heading: true
