# igraph adapter

Convert between an `AnnNet` object and an [igraph](https://python.igraph.org/)
graph. igraph holds nodes, binary edges and their attributes. `to_igraph` also
gives back a manifest with what igraph cannot hold, such as hyperedges, slices
and layers. Give the manifest to `from_igraph` to rebuild the full graph. To
run an igraph algorithm without converting yourself, use
[`anet.ig`](../core/backend-accessors.md).

::: annnet.adapters.igraph_adapter.from_igraph
    options:
      show_root_heading: true

::: annnet.adapters.igraph_adapter.to_igraph
    options:
      show_root_heading: true
