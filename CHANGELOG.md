# Changelog

The package is before its first stable release, so a removed name carries no
deprecation and no alias. Each removal below names what replaces it.

## Unreleased

### Removed

- **`annnet.from_omnipath` and `annnet.io.from_omnipath`.** Access to one
  knowledge base belongs in the client for that knowledge base, which is what
  returns AnnNet objects. The replacement is `omnipath_client.to_annnet`, which
  builds a graph from any OmniPath table, and `omnipath_client.annotate_nodes`,
  which gives every node of that graph what OmniPath knows about it.
  `omnipath_client.relations(as_graph=True)` fetches and builds in one call.
  The package now declares no HTTP client and downloads nothing.
- **Every map from an id to a position**: `entity_to_idx`, `idx_to_entity`,
  `edge_to_idx`, `idx_to_edge` and `entity_types`. A position belongs to one
  materialized matrix. `G.idx` translates a coordinate a caller already holds,
  and `G.views.entity_kinds()` reads the kind of each entity.
- **Every position in a lookup.** `get_edge` takes an id and raises on a column.
  `get_node` takes an id too, and gives back a `NodeView`. The n-th node of a
  sequence is `G.N[n]`.
- **`G.X()`**, which was a second name for `G.S`, the signed coefficient
  incidence. The named matrices are `G.A`, `G.B`, `G.H`, `G.S` and `G.L`.
- **The count aliases**: `num_vertices`, `num_edges`, `num_supra_vertices`,
  `number_of_vertices`, `number_of_edges` and the three `global_*_count`
  wrappers. Use `ncount()` and `ecount()`, with the supra-node count an option
  of the first. `nv`, `ne` and `nv_supra` stay as the property spelling.
- **`G.vertex_attributes` and `G.edge_attributes`**, which were the storage of
  the graph under a public name. `G.obs` and `G.var` build a table for the
  caller, and writing into one changes nothing the graph holds.

### Changed

- The generic attributes of a node and of an edge live in slot-indexed columns.
  One write lands in one cell and builds no table, at any size, and reading one
  attribute of every element is a slice of the array the store holds.
- Set algebra between two graphs: `|`, `&`, `-`, `^`, `|=`. It applies to the
  node set and the edge set together, and an edge survives only when every node
  it names does.
- Each contextual attribute level has one entry point, named for the level:
  `G.slices.attrs`, `G.attrs.edge_slice`, `G.layers.attrs`,
  `G.layers.node_attrs`, `G.layers.aspect_attrs` and
  `G.layers.elementary_attrs`.
- The PyTorch Geometric writer moved from `annnet.adapters.pyg_adapter` to
  `annnet.io.pyg`, with no alias at the old path.

### Renamed

- **The package says "node", everywhere and only.** `vertex` is gone from every
  method, parameter, attribute, column name and document: `add_vertices` is
  `add_nodes`, `remove_vertices` is `remove_nodes`, `vertices()` is `nodes()`,
  `has_vertex` is `has_node`, `supra_vertices` is `supra_nodes`, and `vertex_id`
  is `node_id` in every table the package hands back. Two words for one concept
  was the largest of the faults this release fixes, not a reason to keep it.
  `nv`, `ne` and `nv_supra` never carried the word and do not move.
- The native format writes the new words. Its reader takes both, so an archive
  written before this release still loads: four member names, two columns and
  the entity kind each map the old spelling forward.

### Added

- `G.N` and `G.E`, the node sequence and the edge sequence. A string key is an
  attribute column, an integer key is a position in that sequence, and `select`
  and `find` filter it.
- `G.get_node(node_id)`, which gives a `NodeView`: the id, the kind of the node,
  the layers it lives in, and its attributes.
