# Changelog

The package is before its first stable release, so a removed name carries no
deprecation and no alias. Each removal below names what replaces it.

A removal also lands in every package that bridges to AnnNet, and none of them
is in this test suite. `DEPENDENTS.md` says who those packages are and what to
do about it, and `tests/test_dependents.py` fails the build when a name one of
them calls goes away.

## Unreleased

### Changed, and a caller can see it

- **A column read gives back a read-only array.** `G.N["score"]` and
  `G.E["weight"]` now hand back a window onto the storage rather than a copy of
  it, which is what makes the read cost what slicing an array costs. A write
  through that window would reach the graph with no validation, no clock bump and
  no history entry, so it is refused:

  ```python
  column = G.N['score']
  column.sum()  # works, as before
  column * 2  # works, as before — the result is a new array
  column[0] = 1.0  # ValueError: assignment destination is read-only
  ```

  To change values, copy first — `G.N["score"].copy()` is your own array — or
  write through the entry points that already existed, `G.N["score"] = values`
  and `G.attrs.set_node_attrs`. The rule holds on every read path, so a caller
  never has to ask which one answered.

- **A column is good until the next write to the graph.** After a write, a column
  you are still holding is stale, and what it shows then is not something the
  package promises. `.copy()` is the documented way to hold values across a
  change. Code that reads and uses a column in one expression — which is nearly
  all code — never reaches that boundary.

- **The native format carries a direction policy.** A graph whose edges declare a
  flexible-direction policy used to lose it on a round trip through `.annnet`,
  although cx2 kept it. It now survives. A file written before this change reads
  as before.

### Removed

- **`GraphView.X`**, which was the incidence matrix under the name the graph
  itself dropped. A view spells its matrices the way the graph does, so it is
  `view.B`.

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

- **The eight attribute tables, under one namespace and one convention.** They
  carried three spellings — `G.obs` and `G.var` for the two generic axes,
  `G.slice_attributes` and two siblings for three of the contextual levels, and
  `G.contextual_table(level)` for all six. Same concept, three ways to reach it,
  and the read side in a different namespace from the setter that writes it.
  They are `G.attrs.<address>` now, beside those setters:

  ```python
  G.attrs.nodes  # G.obs
  G.attrs.edges  # G.var
  G.attrs.slices
  G.attrs.aspects
  G.attrs.layers  # one label per aspect, the whole coordinate
  G.attrs.edge_slices
  G.attrs.node_layers
  G.attrs.elementary_layers  # one label inside one aspect
  ```

  Every older spelling still answers, and `obs` and `var` keep the anndata
  parallel, so nothing has to move.

- **`G.attrs.backend`, which every table follows**, and
  `G.attrs.table(name, backend=...)` for the workflow that genuinely mixes two.
  The backend picks the container and never the content.

- `G.N` and `G.E`, the node sequence and the edge sequence. A string key is an
  attribute column, an integer key is a position in that sequence, and `select`
  and `find` filter it.
- `G.get_node(node_id)`, which gives a `NodeView`: the id, the kind of the node,
  the layers it lives in, and its attributes.

### Fixed

- **A whole table assigned to the graph is visible to the next read.** Assigning
  `G.slice_attributes`, `G.edge_slice_attributes` or `G.layer_attributes` wrote
  the store but left the materialized table where it was, so the next read
  answered with the values the assignment had **replaced** — without the rows it
  added, and with nothing to say so. Reading a table before assigning one was
  enough to hit it, which is what a round trip through an adapter does.

- **`G.attrs.table(name, backend=...)` keeps the columns of a table with no
  rows.** It went through rows, and rows carry no schema, so an empty table came
  back with no columns at all — including the column it is addressed by.

- **Asking for the backend a table already has costs nothing.** The name passed
  in was compared against the table without being resolved first, so `"auto"`
  never matched a concrete backend and rebuilt the whole table.

- **A layer column is typed the same whether or not the table holds a row.** A
  layer coordinate is a tuple, so the column holding it is a list of strings.
  `G.attrs.layers` and `G.attrs.node_layers` declared it text, so an empty table
  and a filled one disagreed about the type of the column they are keyed by.

- **A write to one contextual level no longer rebuilds the tables of the other
  five.** They shared one clock, so annotating a slice aged the node-layer table
  as well. Each level keeps its own now.

