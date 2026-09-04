# Graph

Primary graph objects from `annnet.core.graph`.

The main graph API centers on `AnnNet`/`Graph`, bulk node and edge
construction with `add_nodes` and `add_edges`, graph-owned accessors
(`slices`, `layers`, `attrs`, `views`, `ops`, `idx`, `cache`), annotation
tables (`obs`, `var`, `uns`), and backend accessors (`nx`, `ig`, `gt`).

The eight attribute tables are read under `attrs`, each named for what addresses
it: `nodes`, `edges`, `slices`, `aspects`, `layers`, `edge_slices`,
`node_layers` and `elementary_layers`. `obs` and `var` are the same tables as
`attrs.nodes` and `attrs.edges` under the anndata spelling. See
[Reading the graph](../../explanations/reading-the-graph.md) for how these
differ from the frames `views` builds.

## AnnNet

::: annnet.core.graph.AnnNet
    options:
      filters: public
      inherited_members: false
      members:
        - add_nodes
        - add_edges
        - remove_nodes
        - remove_edges
        - has_node
        - has_edge
        - nodes
        - edges
        - degree
        - incident_edges
        - num_nodes
        - num_edges
        - nv
        - ne
        - number_of_nodes
        - number_of_edges
        - shape
        - V
        - E
        - obs
        - var
        - uns
        - attrs
        - views
        - history
        - ops
        - layers
        - slices
        - idx
        - cache
        - nx
        - ig
        - gt
        - read
        - write
        - view
        - global_count
        - get_node
        - get_edge
        - edge_list
        - make_undirected
        - X
        - is_multilayer
      show_root_heading: true
      show_bases: false

## EdgeType

::: annnet.core._records.EdgeType
    options:
      show_root_heading: true

## Endpoint

One side of one edge: the node, and the layer it sits in. Read a stored endpoint
through `as_endpoint` and it has the same shape whether or not the graph is
layered — see [Reading the graph](../../explanations/reading-the-graph.md).

::: annnet.core._records.Endpoint
    options:
      show_root_heading: true

::: annnet.core._records.as_endpoint
    options:
      show_root_heading: true

::: annnet.core._records.as_endpoints
    options:
      show_root_heading: true
