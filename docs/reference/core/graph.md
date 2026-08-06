# Graph

Primary graph objects from `annnet.core.graph`.

The main graph API centers on `AnnNet`/`Graph`, bulk node and edge
construction with `add_nodes` and `add_edges`, graph-owned accessors
(`slices`, `layers`, `attrs`, `views`, `ops`, `idx`, `cache`), annotation
tables (`obs`, `var`, `uns`), and backend accessors (`nx`, `ig`, `gt`).

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
