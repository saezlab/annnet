# Graph

Primary graph objects from `annnet.core.graph`.

The main graph API centers on `AnnNet`/`Graph`, vertex and edge construction
with `add_vertex` and `add_edge`, graph-owned managers (`slices`, `layers`,
`idx`, `cache`), annotation tables (`obs`, `var`, `uns`), and backend accessors
(`nx`, `ig`, `gt`).

## AnnNet

::: annnet.core.graph.AnnNet
    options:
      filters: public
      inherited_members: false
      members:
        - add_vertices
        - add_edges
        - remove_vertices
        - remove_edges
        - has_vertex
        - has_edge
        - vertices
        - edges
        - degree
        - incident_edges
        - num_vertices
        - num_edges
        - nv
        - ne
        - number_of_vertices
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
        - get_vertex
        - get_edge
        - edge_list
        - make_undirected
        - X
        - is_multilayer
      show_root_heading: true
      show_bases: false

## EdgeType

::: annnet.core.graph.EdgeType
    options:
      show_root_heading: true
