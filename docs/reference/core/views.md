# Views and selections

`anet.view(...)` gives a `GraphView`: a live, read-only part of the graph.
`anet.N` and `anet.E` are the node and edge sequences (`NodeSequence`,
`EdgeSequence`), and their `select` method makes a selection. A write through a
view raises `ReadOnlyViewError`.

The rules that decide which nodes and edges a view keeps are in
[The public surface](../api-contract.md), section 5.

::: annnet.core._Views.GraphView
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.NodeSequence
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.EdgeSequence
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.ReadOnlyViewError
    options:
      show_root_heading: true
