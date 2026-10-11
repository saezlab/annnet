# Views and selections

`anet.view(...)` gives a `GraphView`: a live, read-only part of the graph.
`anet.N` and `anet.E` are the node and edge sequences (`NodeSequence`,
`EdgeSequence`), and their `select` method makes a selection. A write through a
view raises `ReadOnlyViewError`.

```python
view = anet.view(nodes=None, edges=None, layers=None, slices=None, *, predicate=None, boundary='closed')
```

`nodes` takes IDs, placements, a `NodeSequence`, a `RowSelection` of `nodes` or
`node_layers`, a mask or a function. `edges` takes IDs, an `EdgeSequence`, a
`RowSelection` of `edges`, a mask or a function. `layers` takes coordinates or
`anet.layers.where(...)`, and `slices` takes slice names or a slice selection.

## Which nodes and edges a view keeps

| `boundary` | result |
|---|---|
| `'closed'` (default) | An edge stays only if all its endpoints are selected. A hyperedge stays whole, or not at all. A selected node without edges stays. A filter on edges only keeps those edges and their endpoints. |
| `'open'` | The edges that touch a selected node stay, with all their endpoints: one step out. The edge and slice filters still apply. `view.summary()['expanded']` lists the added endpoints. |

- **Edges that end on an edge:** such an edge needs the edge that it ends on.
  A filter on edges brings that edge with it.
- **A view of a view** only removes more. Its filters add to the filters of
  the first view.
- A slice is a set of members. A view of a slice does not add the edges between
  its nodes.

## Live and read-only

- A view is live: it follows all changes of the graph. If a filter is a
  function, the view calls it again at each read.
- `view.N`, `view.E` and `view.attrs` work as on the graph, but only for the
  nodes and edges in the view.
- A write through a view raises `ReadOnlyViewError`, a subclass of
  `TypeError`. The message tells you to use `materialize()`.
- `view.materialize()` gives a new, independent `AnnNet` with the selected
  nodes and edges, their slices, layers and attributes.

## Selections

`anet.N.select(...)` and `anet.E.select(...)` give a selection of the same
type. Selections combine with `&`, `|` and `-`, and the result is in the
order of the graph. A selection from conditions is live, like a view. A
selection from IDs or a mask is fixed. Selections from two graphs raise
`ValueError` when you combine them, and a node selection with an edge selection
raises `TypeError`.

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
