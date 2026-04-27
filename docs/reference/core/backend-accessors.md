# Backend Accessors

`AnnNet` has graph-owned backend accessors:

- `G.nx` for NetworkX
- `G.ig` for igraph
- `G.gt` for graph-tool

These are public access points on `AnnNet`. They are different from the
module-level conversion functions in `annnet.adapters`:

- `annnet.adapters.to_nx(...)` / `annnet.adapters.from_nx(...)`
- `annnet.adapters.to_igraph(...)` / `annnet.adapters.from_igraph(...)`
- `annnet.adapters.to_graphtool(...)` / `annnet.adapters.from_graphtool(...)`

Use `annnet.adapters` when you want an explicit in-memory conversion to or from
another Python graph object. Use `G.nx`, `G.ig`, or `G.gt` when `G` remains the
source object and AnnNet should convert, cache, and dispatch a backend
algorithm call for you.

## Common Accessor Methods

All graph-owned backend accessors provide:

- `backend(...)`: return the concrete projected backend graph object.
- `clear()`: clear cached backend projections for that accessor.

NetworkX and igraph accessors also provide:

- `peek_vertices(k=10)`: return a small sample of backend vertex identifiers.

## Algorithm Dispatch

The accessors resolve backend functions dynamically. For example:

```python
centrality = G.nx.degree_centrality(G)
distances = G.ig.distances(source="a", target="c", weights="weight")
```

In these calls AnnNet projects the graph to the backend, replaces the `G`
argument with the projected backend graph where needed, calls the backend
function, and maps vertex identifiers back where supported.

## NetworkX Options

`G.nx.backend(...)` accepts options such as:

- `directed`
- `hyperedge_mode`
- `slice`
- `slices`
- `needed_attrs`
- `simple`
- `edge_aggs`

Dynamic `G.nx.<algorithm>(...)` calls accept the corresponding underscore
options, such as `_nx_directed`, `_nx_hyperedge`, `_nx_slice`, `_nx_slices`,
`_nx_label_field`, `_nx_simple`, and `_nx_edge_aggs`.

## igraph Options

`G.ig.backend(...)` accepts options such as:

- `directed`
- `hyperedge_mode`
- `slice`
- `slices`
- `needed_attrs`
- `simple`
- `edge_aggs`

Dynamic `G.ig.<algorithm>(...)` calls accept the corresponding underscore
options, such as `_ig_directed`, `_ig_hyperedge`, `_ig_slice`, `_ig_slices`,
`_ig_label_field`, `_ig_simple`, and `_ig_edge_aggs`.

## graph-tool Options

`G.gt.backend()` returns the projected graph-tool graph object. Dynamic
`G.gt.<namespace>.<algorithm>(...)` calls dispatch to graph-tool namespaces
where supported.

The implementation classes for these accessors live in
`annnet.core.backend_accessors` and are underscore-prefixed. Direct imports from
those implementation modules follow the [internal API policy](../api-boundary.md).
