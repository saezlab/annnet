# Internal representation

This page describes the in-memory model of `AnnNet`, as implemented in
`annnet.core._store`, `annnet.core._attrs`, `annnet.core._matrices` and
`annnet.core._structure`.

The key fact is simple:

- the graph has one structural source of truth, the canonical store
- everything else is an overlay, a derived structure, or a read facade

If you keep that distinction straight, the internal model is clean.

## The canonical store

`annnet.core._store.CoreState` holds the structure of one graph. The graph
object keeps it in `_store`, and nothing else holds topology.

A caller addresses an element two ways:

- by **identity**, which is an entity key or an edge id
- by **slot**, which is an integer the store assigns on insert

A slot is stable. The store never renumbers one, so a remove frees the slot of
that element and leaves every other slot where it was. The store reuses a freed
slot before it grows the arrays.

An entity key is a tuple:

```python
(node_id, layer_coord)
```

where `layer_coord` is itself a tuple of aspect values.

Examples:

- flat graph node: `("TP53", ("_",))`
- two-aspect supra-node: `("TP53", ("treated", "t1"))`

Rows therefore belong to entities, not to plain nodes. An entity is a node or an
edge-entity, which is the `entity_kind` array.

## Topology lives in the member lists

The store keeps no edge record. What an edge holds is a segment of three pooled
arrays:

```python
member_ent   # which entity slot this entry names
member_role  # SOURCE, TARGET or MEMBER
member_coef  # the coefficient of this entry
```

`member_start` and `member_len` say which segment belongs to which edge slot.
The segments together are an incidence matrix in compressed sparse column form,
addressed by slot. One pool holds every edge kind. A binary edge, a hyperedge
and a node-edge therefore differ in how many entries they own, and in nothing
else.

**One entry per role.** A member entry records one role of one entity in one
edge, not one entity. An entity that takes two roles in one edge appears twice
in that edge. That is what keeps a self-loop distinct from a boundary edge. A
self-loop holds two entries on one entity slot, and a boundary edge holds one.
Without it, a directed self-loop would collapse to a single entry and look
exactly like a one-sided edge.

The per-edge scalars sit in arrays beside the pools: `edge_kind`,
`edge_directed`, `edge_weight` and `edge_explicit`. Rare per-edge state stays in
dictionaries keyed by slot: the multilayer kind, the layer tuple and the
direction policy. That state costs nothing on an edge that does not carry it.

## Indices the store maintains

The store maintains two lookups rather than rebuilding them, because a rebuild
of either would make a local change cost the size of the graph:

- `_id_slots` maps one bare node id to the slots it stands for. A flat graph
  needs it for nothing, because an id names exactly one entity there. A
  multilayer graph asks it on every resolution of a bare id.
- `_entity_edges` maps an entity slot to the edges that touch it. It also holds
  the sides that entity takes, and the peer on the other side when the edge has
  exactly two entries.

Both are rebuildable from the member lists and neither is authoritative.

## The store holds no matrix

The store imports no matrix library and holds no matrix object. A matrix is
derived state. `annnet.core._matrices` builds one: it gathers the member lists
and remaps slots to rows.

The package builds several purpose-built matrices rather than one that mixes
every edge kind. Each named matrix decides the two awkward shapes for itself:

| Matrix    | Self-loop                              | Boundary edge                   |
|-----------|----------------------------------------|---------------------------------|
| incidence | two entries, which sum to zero         | the single entry stays          |
| adjacency | one diagonal entry                     | left out, so no false loop      |
| Laplacian | follows the adjacency                  | follows the adjacency           |

`MatrixCache` keeps a built matrix against the clock of the store. A write that
only appends edges at the frontier extends the cached matrix instead of dropping
it. A read after an append therefore costs the new columns and not a rebuild.
Without that, a loop of N appends with a read after each one would be
quadratic.

## Attributes are slot-indexed columns

`annnet.core._attrs.AttributeStore` holds one generic attribute as one typed
array, indexed by slot. One write lands in one cell, at any graph size, and a
value keeps its place when another element goes away. A free slot holds a null.

A value keeps the type of the write that set it. The store converts nothing, so
an integer in a column that later takes a string stays an integer.

`G.obs` and `G.var` **derive** their content. They gather the live slots of
every column and hand the result to narwhals, so one materialization serves
every dataframe backend. A write into the table a materialization handed back
changes nothing the graph holds.

A column of a dataframe has one type, so a materialized table widens where the
columns do not. A column that holds an integer and a string materializes as an
object column rather than a float one. A count therefore does not read back as
`3.0`.

**A generic node attribute belongs to a node and not to a node-layer.** A
multilayer graph holds one entity per layer the node lives in, so one bare id
covers several slots. A write by bare id lands in the cell of each of them, and
the derived table shows the id once.

A contextual attribute belongs to a pair rather than to one element — one edge in
one slice, or one node in one layer. Almost no pair carries a value, so a dense
column per pair would waste nearly every cell. Those stores stay keyed by the
pair, and each level has one public entry point: `G.slices.attrs`,
`G.attrs.edge_slice`, `G.layers.attrs`, `G.layers.node_attrs`,
`G.layers.aspect_attrs` and `G.layers.elementary_attrs`.

## Slices are overlays, not duplicate graphs

Slice state lives in `_slices`, which maps a slice identifier to a
`SliceRecord`:

- node membership
- edge membership
- slice attributes

This is not another topology store. Slices do not redefine the graph
structurally. They describe which parts of the same graph are active in a named
context.

`edge_slice_attributes` and `slice_edge_weights` hold the per-slice edge
weights, separately from the rest.

## Multilayer state is also an overlay

The multilayer state sits in four places:

- `_aspects` and `_layers`, the aspect and elementary-layer registry
- the layer coordinate inside each entity key
- the multilayer kind and layer tuple of an edge, in the store
- the supra index, built on demand behind `G.layers.nl_to_row` and
  `G.layers.row_to_nl`

Again, this does not replace the structural graph. It enriches it. The supra-node
model resolves back to the same entity slots and member lists.

## The read facade

`annnet.core._structure` is the one boundary between the canonical store and the
rest of the package. Input-output code, adapters and bridges read topology
through it. They never read a private store attribute of a graph.

The facade answers questions about structure only. It reports which entities and
edges exist, which entities an edge holds, and which edges touch an entity. It
does not report attributes, and it never writes.

Every address in the facade is an identity. A row number and a column number
belong to one materialized matrix, so neither appears in an answer. `G.idx`
translates a coordinate a caller already holds, in both directions, against the
matrix the graph would build now.
