# NetworkX adapter

Convert between an `AnnNet` object and a [NetworkX](https://networkx.org/)
graph. NetworkX holds nodes, binary edges and their attributes. `to_nx` also
gives back a manifest with what NetworkX cannot hold, such as hyperedges,
slices and layers. Give the manifest to `from_nx` to rebuild the full graph.
To run a NetworkX algorithm without converting yourself, use
[`anet.nx`](../core/backend-accessors.md).

::: annnet.adapters.networkx_adapter.from_nx
    options:
      show_root_heading: true

::: annnet.adapters.networkx_adapter.to_nx
    options:
      show_root_heading: true

::: annnet.adapters.networkx_adapter.from_nx_without_manifest
    options:
      show_root_heading: true
