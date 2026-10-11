# graph-tool adapter

Convert between an `AnnNet` object and a [graph-tool](https://graph-tool.skewed.de/)
graph. graph-tool holds nodes and binary edges, with one direction for the
whole graph and a `weight` edge property. `to_graphtool` also gives back a
manifest with everything else: hyperedges, the direction of each edge, layers,
slices and the attribute tables. Give the manifest to `from_graphtool` to
rebuild the full graph. graph-tool is not on PyPI: install it with conda or
your system package manager. To run a graph-tool algorithm without converting
yourself, use [`anet.gt`](../core/backend-accessors.md).

::: annnet.adapters.graphtool_adapter.from_graphtool
    options:
      show_root_heading: true

::: annnet.adapters.graphtool_adapter.to_graphtool
    options:
      show_root_heading: true
