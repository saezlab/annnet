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

## Matrices

All matrices are SciPy sparse arrays. Rows are nodes (placements, on a
multilayer graph) and edge entities. The properties use the default options.
`anet.matrices.<name>(...)` takes options, and gives the matrix together with
the maps between IDs and positions (see [Indexing](indexing.md)).

| property | `anet.matrices` | rows × columns | edges in it | values |
|---|---|---|---|---|
| `anet.B` | `binary()` | entities × edges | binary edges | `+weight` on the source, `-weight` on the target |
| `anet.S` | `signed()`, `incidence()` | entities × edges | all edges | the coefficients; the two entries of a self-loop cancel |
| `anet.H` | `hypergraph()` | entities × edges | hyperedges | 1 if the edge holds the entity, on any side |
| `anet.A` | `adjacency()` | entities × entities | binary edges | the edge weight, in the row of the source; a self-loop is on the diagonal |
| `anet.L` | `laplacian()` | entities × entities | binary edges | the degree matrix minus `A`; each row sums to zero |

A hyperedge is not in `A` or `L`: give `kinds=` to `adjacency()` to choose how
to include it. `incidence(kinds=None, signed=True)` gives the incidence for the
edge kinds that you select, with or without the coefficients.

On a multilayer graph, `anet.layers` has the matrices over all placements:
`supra_adjacency()`, its parts `build_intra_block()`, `build_inter_block()` and
`build_coupling_block()`, `supra_laplacian()` and `transition_matrix()`. See
[Layers](layers.md) and the [layers guide](../../guide/layers.ipynb#supra-adjacency-and-its-blocks).

The [graph model guide](../../guide/graph-model.ipynb#structure-and-numerical-representations)
explains the matrices with an example.

## Edge records

`anet.E.at(edge_id)` gives an `EdgeView`. Its fields:

- `kind`: the structure, `binary`, `hyper` or `node_edge`;
- `directed`: the direction, separate from the kind;
- `ml_kind`: the place in a multilayer graph, `intra`, `inter` or `coupling`.
  For a graph without layers, it is `intra`.

The same fields have the same names in the `anet.E[...]` columns, in the
derived edge table and in `anet.summary()`.

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
