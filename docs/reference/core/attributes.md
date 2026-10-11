# Attributes

`anet.attrs` holds all attributes of a graph, at eight addresses: `nodes`,
`edges`, `slices`, `aspects`, `layers`, `edge_slices`, `node_layers` and
`elementary_layers`. You read and write each address the same way:

- a read (`nodes`, `table`, `row`, `rows`) gives a copy;
- a write is a named call (`update`, `replace`, `delete`);
- `select` is a live query, and `from_frame` makes a fixed selection from the
  rows of a table that you filtered yourself.

`anet.attrs` is an `Attrs` object. `select` and `from_frame` give a
`RowSelection`, and `schema` gives a `Schema`. The full rules are in
[The public surface](../api-contract.md), section 3. The
[annotations guide](../../guide/annotations-and-views.ipynb) shows the use.

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
