# Graph

`AnnNet` is the graph object. `annnet.Graph` is the same class.

An `AnnNet` object holds the structure (nodes, edges, hyperedges), the
attribute tables, and the slices and layers. The members below are grouped by
what they do:

- **Build and edit:** `add_nodes`, `add_edges`, `remove_nodes`, `remove_edges`,
  `make_undirected`.
- **Nodes and edges:** `N`, `E`, `shape`, `has_node`, `has_edge`, `degree`,
  `neighbors`, `in_neighbors`, `out_neighbors`, `predecessors`, `successors`,
  `incident_edges`, `edge_list`, `entity_kinds`.
- **Layers:** `at`, `exists`, `is_multilayer`, `nv_supra`, `supra_nodes`,
  `supra_shape`.
- **Attributes and metadata:** `attrs`, `uns`, `summary`, `provenance`.
- **Matrices:** `A`, `B`, `H`, `S`, `L`, `matrices`, `idx`, `cache`.
- **Namespaces:** `slices`, `layers`, `ops`, `history`, `spaces`, and the
  [backend accessors](backend-accessors.md) `nx`, `ig`, `gt`.
- **Views and files:** `view`, `read`, `write`.

The eight attribute tables are under `attrs`: `nodes`, `edges`, `slices`,
`aspects`, `layers`, `edge_slices`, `node_layers` and `elementary_layers`. See
[Attributes](attributes.md) and the
[annotations guide](../../guide/annotations-and-views.ipynb).

## AnnNet

::: annnet.core.graph.AnnNet
    options:
      filters: public
      inherited_members: true
      members:
        - add_nodes
        - add_edges
        - remove_nodes
        - remove_edges
        - make_undirected
        - N
        - E
        - shape
        - has_node
        - has_edge
        - degree
        - neighbors
        - in_neighbors
        - out_neighbors
        - predecessors
        - successors
        - incident_edges
        - edge_list
        - entity_kinds
        - global_count
        - at
        - exists
        - is_multilayer
        - nv_supra
        - supra_nodes
        - supra_shape
        - attrs
        - uns
        - summary
        - provenance
        - A
        - B
        - H
        - S
        - L
        - matrices
        - idx
        - cache
        - slices
        - layers
        - ops
        - history
        - spaces
        - nx
        - ig
        - gt
        - view
        - read
        - write
      show_root_heading: true
      show_bases: false

## Endpoint

One side of one edge: the node, and the layer it sits in. `as_endpoint` gives
a stored endpoint the same shape, whether or not the graph has layers. See the
[annotations guide](../../guide/annotations-and-views.ipynb).

::: annnet.core._records.Endpoint
    options:
      show_root_heading: true

::: annnet.core._records.as_endpoint
    options:
      show_root_heading: true

::: annnet.core._records.as_endpoints
    options:
      show_root_heading: true

## Provenance

The sources that a graph was built from. The readers of annnet add a record for
each source they read: its name, format, location, version, the time and a
checksum. Read the records with `anet.provenance()`.

::: annnet.core._provenance.Provenance
    options:
      show_root_heading: true
