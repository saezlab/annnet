# Changelog

The package is before its first stable release, so a removed name carries no
deprecation and no alias. Each removal below names what replaces it.

A removal also lands in every package that bridges to AnnNet, and none of them
is in this test suite. `DEPENDENTS.md` says who those packages are and what to
do about it, and `tests/test_dependents.py` fails the build when a name one of
them calls goes away.

## 0.4.0 (unreleased)

### One way to reach every attribute, and views that resolve correctly

The surface after this change is stated in `docs/reference/api-contract.md`,
the migration of every removed name in `docs/explanations/api-migration.md`,
and the machine-readable inventory in `docs/reference/public-surface.json`.
Nothing removed here carries an alias: each removed name raises
`AttributeError` naming its replacement.

#### Removed

- **`G.obs` and `G.var`.** The node and edge tables are `G.attrs.nodes` and
  `G.attrs.edges`, the two of eight addresses that read and write the same way.
- **`G.views`** (`nodes`, `edges`, `hyperedges`, `slices`, `aspects`, `layers`,
  `layers_view`, `entity_kinds`). A derived table is
  `G.attrs.table(address, derived=True, ...)` with the filters and joins as
  explicit arguments (`slice=`, `in_slice=`, `layer=`, `include_hyper=`,
  `include_binary=`, `layout='incidences'`); `G.entity_kinds()` moved to the
  graph.
- **`G.nodes()`, `G.edges()`, `G.nv`, `G.ne`, `G.ncount()`, `G.ecount()`.**
  The axes are `G.N` and `G.E`: `list(G.N)`, `G.N.ids`, `len(G.N)`, `len(G.E)`;
  the placement count is `G.nv_supra`.
- **`G.get_node`, `G.get_edge`, `G.get_edge_ids`, `G.get_edges_by_direction`,
  `G.in_edges`, `G.out_edges`, `G.remove_node`, `G.remove_edge`,
  `G.contextual_table`.** Use `G.N.at`, `G.E.at`, `G.has_edge(s, t)[1]`,
  `G.E.select(directed=...)`, `G.incident_edges(nodes, direction=...)`,
  `G.remove_nodes`, `G.remove_edges`, `G.attrs.table(address)`.
- **The 24 verb-prefixed attribute methods of `G.attrs`** (`get_attr_node`,
  `get_attr_nodes`, `get_node_attrs`, `set_node_attrs`, `set_node_attrs_bulk`,
  `get_attr_edge`, `get_attr_edges`, `get_attr_from_edges`, `get_edge_attrs`,
  `get_edges_by_attr`, `set_edge_attrs`, `set_edge_attrs_bulk`,
  `get_slice_attr`, `set_slice_attrs`, `edge_slice`, `get_edge_slice_attr`,
  `set_edge_slice_attrs`, `set_edge_slice_attrs_bulk`, `set_slice_edge_weight`,
  `get_effective_edge_weight`, `get_graph_attribute`, `get_graph_attributes`,
  `set_graph_attribute`, `audit_attributes`). One row is
  `G.attrs.row(address, key)`, many are `G.attrs.rows(address, keys)`, a merge is
  `G.attrs.update(address, {key: {...}})`, a filter is
  `G.attrs.select(address, ...)`, the resolved weight is
  `G.E.effective_weight(edge_id, slice=None)`, the graph metadata is `G.uns`,
  and the audit is `G.attrs.audit()`.
- **Indexing `G.attrs`, live rows and table assignment.** `G.attrs[address]`,
  `G.attrs[address, key]`, assigning to or deleting either, and
  `G.attrs.nodes = table` (likewise for the other addresses) raise `TypeError`
  naming the call that replaces them: `G.attrs.<address>` or `table` to read a
  table, `row` to read a row, `update` to change fields, `replace` to swap a
  table, `delete` to remove attributes. A row is a detached `dict`; editing a
  value read from `G.attrs` never changes the graph.
- **The attribute methods of `G.layers` and `G.slices`** (`layers.attrs`,
  `layers.set_attrs`, `layers.node_attrs`, `layers.set_node_attrs`,
  `layers.set_node_attrs_bulk`, `layers.aspect_attrs`, `layers.set_aspect_attrs`,
  `layers.elementary_attrs`, `layers.set_elementary_attrs`, `slices.attrs`).
  They are the `layers`, `node_layers`, `aspects`, `elementary_layers` and
  `slices` addresses of `G.attrs`, read and written like the other three.
- **`G.cache.get_adjacency()`, `get_csr()`, `get_csc()`.** The cached matrices
  are `G.cache.adjacency`, `G.cache.csr` and `G.cache.csc`.
- **On a view: `obs`, `var`, `nodes_df`, `edges_df`, `node_count`,
  `edge_count`, `node_ids`, `edge_ids`, `subview`.** A view carries the
  graph's reading vocabulary under the graph's names: `V.attrs.table(...)`,
  `len(V.N)`, `len(V.E)`, `V.N.ids`, `V.E.ids`, `V.view(...)`.
- **`annnet.core.ProvenanceAccessor`** is exported as `Provenance`.

#### Changed, and a caller can see it

- **Every attribute address reads and writes the same way.** Reads return
  copies: `G.attrs.nodes`, `G.attrs.table('nodes', ...)`,
  `G.attrs.row('nodes', key)`, `G.attrs.rows('nodes')` and
  `G.attrs.schema('nodes')`. Writes are named: `G.attrs.update('nodes', rows)`
  merges the fields it names, `G.attrs.replace('nodes', table)` replaces the
  whole address (a key the table omits loses its attributes and keeps its
  existence), and `G.attrs.delete('nodes', keys=..., names=...)` removes
  attributes and never the elements that carry them: keys alone clear those
  rows, names alone drop those fields everywhere, both drop those fields from
  those rows, neither raises `ValueError`, and an empty collection is a no-op.
  The same expressions serve the other seven addresses. Elementary layers are
  keyed `(aspect, label)`; the legacy `aspect_label` id is accepted at the IO
  boundary only and refused when it could name two rows. In `update`, a null
  value removes the field, at every address and in every backend. A write is all
  or nothing: a failed one leaves the attribute values, the fields and their
  types and order, the topology a flexible-direction policy rewrote, the
  node-key index and the caches as they were.
- **Live queries and filtered tables are different things.** `select` is a
  query over the graph and answers again after the graph changes.
  `G.attrs.from_frame(address, frame)` turns the rows of a table you filtered
  with its dataframe library into a fixed selection of the graph's keys: it
  reads only the key columns, keeps the address's order without repeats, holds
  identities rather than storage slots, and an empty frame is an empty selection.
  There is no dataframe-expression form of `select` and no second keyword
  syntax for fields.
- **One predicate language.** `G.N.select`, `G.E.select`, `G.attrs.select` and
  `G.layers.where` parse the same `field__operator=value` conditions
  (`eq`, `ne`, `in`, `not_in`, `lt`, `lte`, `gt`, `gte`, `isnull`), exclude
  nulls except under `isnull`, and compare an ordered aspect by its declared
  order. Selections are typed and live, keep their plan, and combine with `&`,
  `|` and `-` in the root graph's order; a leaf made in a view is evaluated in
  that view. Row selections on a composite address project explicitly
  (`.project('nodes')`, `.project('edges')`, `.project('slices')`).
- **Views resolve correctly and stay live.** One resolver over placement keys
  and edge ids implements the closed and open boundaries, keeps a hyperedge
  whole or drops it, keeps selected isolated nodes, handles self-loops,
  half-edges and edge-entity endpoints (the backing edge is retained or the
  dependent edge dropped, never a dangling reference), and lets a nested view
  only restrict. A view re-resolves when any clock of the graph moves. Every
  write through a view raises `ReadOnlyViewError` naming `.materialize()`.
  `V.materialize()` is the same resolution copied, with the attributes at all
  eight addresses.
- **`G.E.at(edge_id).kind` is the structural kind** — `binary`, `hyper`,
  `node_edge` — and `directed` is the separate direction, in the record, the
  `G.E` columns, the derived table and the summary alike.
- **The general path costs what it uses.** Constructing an empty graph builds
  no table (56.4 µs, from 655 µs before); `len(G.N)` is a
  maintained counter (0.4 µs at ten nodes and at a million); `G.degree` reads
  the incident-edge index (1.6 µs, independent of the size of the graph). A
  flat graph records no per-edge multilayer role: `intra` is answered from the
  aspects, and the record is materialized when aspects are declared. Numbers
  and method: `docs/explanations/performance.md`.
- **`G.attrs.table(..., limit=n)` builds `n` rows and no more**, and a view of a
  node selection resolves through the incident-edge index rather than a scan of
  every edge.

#### Added

- `G.summary()` and `V.summary()`: counts, aspects, slices, attribute fields and
  edge tallies, without a frame or a matrix.
- `G.attrs.row`, `rows`, `update`, `replace`, `delete`, `select`, `from_frame`,
  `schema` and `audit`; `G.attrs.table(..., derived=True, layout='incidences')`.
- `G.N.at`, `G.E.at`, `G.E.effective_weight`, `G.N.axis`, `G.N.graph`; the
  `NodeSequence`, `EdgeSequence`, `RowSelection`, `Schema`, `Attrs`,
  `GraphView` and `Provenance` exports.
- `G.entity_kinds()`, `G.at`, `G.exists`, `G.provenance`, `G.spaces` on the
  graph; `V.N`, `V.E`, `V.attrs`, `V.layers`, `V.slices`, `V.uns`, `V.view`,
  `V.summary`, `V.degree`, `V.entity_kinds`, `V.supra_nodes` and the traversal
  reads on a view.
- An edge entity on a multilayer graph can be named as an endpoint: registering
  it declares the placeholder coordinate it sits on.

#### Fixed

- **A directed edge keeps its direction in every format.** The JSON and NDJSON
  writers and the manifests of the NetworkX and igraph adapters (and so GraphML
  and GEXF) sorted the two endpoints of a binary edge, so an edge from `B` to `A`
  came back as an edge from `A` to `B`. Undirected edges are still written in
  sorted order.
- **Edge entities survive every format.** No format but the native one used to
  hold an edge entity: it came back as a node, and the edge that joined it to a
  node came back as an ordinary binary edge. The native format, JSON, NDJSON,
  Parquet, CX2, the dataframe frames and the NetworkX, igraph and graph-tool
  manifests now record the entities, the edges that join them to nodes, and the
  attributes stored under them; SIF, CSV, Excel, GraphML and GEXF keep them in
  their sidecar. `from_nx`, `from_graphml` and `from_gexf` no longer fail on a
  graph that has an edge entity. A record written before edge entities were
  recorded reads with the entity as a node and reports the attributes that had
  no edge to go to as an `AnnNetLossWarning`. `docs/explanations/format-semantics.md`
  states what each format keeps.
- **CSV and Excel keep edge ids and slices.** The reader ignored the `edge_id`
  column and renumbered the edges, and a blank spreadsheet cell became a slice
  named `nan`. The sidecar wrote a `hyperedges` payload that nothing restored;
  hyperedges whose members are plain node ids come back, and others are
  reported.
- **igraph reads multi-aspect graphs** (it declared no aspects before adding the
  hyperedges) and no longer stores unit endpoint coefficients as attributes.
  **graph-tool** restores hyperedge attributes and the edges that end on an edge
  entity, and no longer fails on a graph that has one.
- **A callable filter reaches everything derived from it.** A selection made in
  a view built on a callable, the union, difference or intersection of two of
  them, the projection of its attribute rows, and a view built on any of these
  are evaluated again on every read, so external state that changed without a
  change of the graph is followed.
- **A repeated key in an iterable of node-layer updates raises before anything
  is written**, however the placement is spelled (a full key, a bare id with
  `layer=`, a label or a tuple for the layer); it used to keep the last row.
- **A failed write restores the schema.** A column the batch introduced, a type
  it widened and a column a `delete` dropped come back as they were, for row
  batches and table replacements. Writing a null to a field the store never held
  no longer creates an empty column.
- The benchmark suite could not run `igraph.delete_nodes` (the method is
  `delete_vertices`) and passed anyway, because its tests filtered out failed
  operations; they now fail on any operation that errors, and only a missing
  optional dependency is skipped.

#### Known limitations

- A manifest is the lossless channel of the NetworkX, igraph and graph-tool
  adapters. Read a library graph without its manifest and an edge entity is a
  node, because the graph itself says nothing else.
- SIF, CSV and Excel hold two-endpoint edges; a hyperedge on a multilayer graph
  (whose members are placements) is not restored from their sidecar.
- `omnipath-client` and `corneto` need the changes in `dependents.toml` before a
  user of either can upgrade to this version. Both migrations are verified
  against it but neither is merged in its repository; see `DEPENDENTS.md`.

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
  wrappers. (The attribute rework above then removed `ncount()`, `ecount()`, `nv` and `ne` as
  well; the counts are `len(G.N)`, `len(G.E)` and `G.nv_supra`.)
- **`G.vertex_attributes` and `G.edge_attributes`**, which were the storage of
  the graph under a public name. The tables are built for the caller (as
  `G.attrs.nodes` and `G.attrs.edges` since the attribute rework above), and writing into one
  changes nothing the graph holds.

### Changed

- The generic attributes of a node and of an edge live in slot-indexed columns.
  One write lands in one cell and builds no table, at any size, and reading one
  attribute of every element is a slice of the array the store holds.
- Set algebra between two graphs: `|`, `&`, `-`, `^`, `|=`. It applies to the
  node set and the edge set together, and an edge survives only when every node
  it names does.
- Each contextual attribute level has one entry point, named for the level.
  (The attribute rework above then made every level an address of `G.attrs`, read with `G.attrs.row(address, key)`.)
- The PyTorch Geometric writer moved from `annnet.adapters.pyg_adapter` to
  `annnet.io.pyg`, with no alias at the old path.

### Renamed

- **The package says "node", everywhere and only.** `vertex` is gone from every
  method, parameter, attribute, column name and document: `add_vertices` is
  `add_nodes`, `remove_vertices` is `remove_nodes`, `vertices()` is `nodes()`,
  `has_vertex` is `has_node`, `supra_vertices` is `supra_nodes`, and `vertex_id`
  is `node_id` in the public tables.
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

  The attribute rework above then removed the older spellings, `obs` and `var`
  among them.

- **`G.attrs.backend`, which every table follows**, and
  `G.attrs.table(name, backend=...)` for the workflow that genuinely mixes two.
  The backend picks the container and never the content.

- `G.N` and `G.E`, the node sequence and the edge sequence. A string key is an
  attribute column, an integer key is a position in that sequence, and `select`
  and `find` filter it.
- A node record, `NodeView`: the id, the kind of the node, the layers it lives
  in, and its attributes (reached as `G.N.at(node_id)`).

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

