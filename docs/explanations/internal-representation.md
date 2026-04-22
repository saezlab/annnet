# Internal representation

This page describes the actual in-memory model of `AnnNet` as implemented in
`annnet.core.graph` and `annnet.core._helpers`.

The key fact is simple:

- the graph has one structural source of truth
- everything else is either an overlay, a derived index, or a compatibility
  boundary

If you keep that distinction straight, the internal model is clean.

## The four canonical structural stores

AnnNet's structural topology is organized around four mappings:

```python
_entities: dict[tuple, EntityRecord]
_edges: dict[str, EdgeRecord]
_row_to_entity: dict[int, tuple]
_col_to_edge: dict[int, str]
```

They play distinct roles.

### `_entities`

`_entities` maps an internal entity key to an `EntityRecord`.

The key is a tuple:

```python
(vertex_id, layer_coord)
```

where `layer_coord` is itself a tuple of aspect values.

Examples:

- flat graph vertex: `("TP53", ("_",))`
- two-aspect supra-node: `("TP53", ("treated", "t1"))`

The value is an `EntityRecord`, currently:

```python
EntityRecord(row_idx: int, kind: str)
```

`kind` distinguishes at least:

- `"vertex"`
- `"edge_entity"`

This means that rows belong to entities, not just to plain vertices.

### `_edges`

`_edges` maps an edge identifier to an `EdgeRecord`.

`EdgeRecord` is the canonical store for edge semantics:

```python
EdgeRecord(
    src,
    tgt,
    weight,
    directed,
    etype,
    col_idx,
    ml_kind,
    ml_layers,
    direction_policy,
)
```

Important fields:

- `etype`
  Structural edge type, currently distinguishing binary, hyper, and
  vertex-edge cases.
- `src` and `tgt`
  The current internal field names for structural source and target endpoint
  sets. Public-facing docs should use "source" and "target" terminology even
  while these record fields remain abbreviated internally.
- `col_idx`
  The incidence column index, or `-1` for an edge-entity placeholder with no
  structural column yet.
- `ml_kind` and `ml_layers`
  Multilayer metadata. These are not a second edge store; they annotate the
  same edge record.

### `_row_to_entity`

`_row_to_entity` is the reverse lookup for the row space.

It answers:

- given a matrix row, which entity key owns it?

This is not redundant bookkeeping for convenience. It is what allows the
incidence matrix to stay usable as soon as you need to map matrix outputs back
to graph objects.

### `_col_to_edge`

`_col_to_edge` is the reverse lookup for the column space.

It answers:

- given a matrix column, which edge does it represent?

Like `_row_to_entity`, this is part of the structural model, not just an API
extra.

## What the sparse matrix means

`AnnNet` stores a sparse incidence matrix in DOK form as its editable primary
matrix:

```python
_matrix: scipy.sparse.dok_matrix
```

The canonical interpretation is:

- rows correspond to entities
- columns correspond to edges

The matrix is not the only source of truth. It is one half of the structural
truth together with the registries above.

Why this matters:

- a matrix entry alone does not tell you whether a column is binary or hyper
- a matrix entry alone does not tell you whether a row is a vertex or an
  edge-entity
- a matrix entry alone does not carry multilayer metadata

That information lives in the records. The matrix holds incidence; the records
hold semantics.

## Why the row key is `(vertex_id, layer_coord)`

The clean invariant is:

- one row represents one vertex-layer state

That is why `_entities` is keyed by `(vertex_id, layer_coord)` rather than by
plain vertex id.

For flat graphs, the layer coordinate is the basal placeholder:

```python
("_",)
```

For layered graphs, the coordinate is a tuple over declared aspects.

This avoids mixing two row meanings inside one matrix. There is no separate
"global vertex row" once the graph is in the layered model. A vertex without
explicit multilayer placement is represented at the placeholder coordinate,
not as a structurally different kind of row.

## Placeholder coordinates are real coordinates

The placeholder tuple

```python
("_", ..., "_")
```

is not just an implementation accident. It is a valid fallback coordinate.

It appears in three important situations:

1. flat graphs before aspects are declared
2. layered graphs when old flat vertices are lifted into the layered model
3. layered graphs when the user adds vertices without an explicit `layer=`

This keeps the row invariant intact:

- every row still represents one `(vertex, layer_coord)`

The placeholder exists only as long as some graph state still references it.
When nothing refers to it anymore, it can be dropped from the active layer
registry.

## Derived adjacency indices

Several indices are maintained incrementally from `_edges`:

- `_adj`
- `_src_to_edges`
- `_tgt_to_edges`

These are derived execution indices, not competing edge stores.

Their role is to make common local operations efficient:

- neighborhood lookups
- endpoint-based edge existence checks
- directional incident-edge scans

They matter for performance, but the semantics still come from `EdgeRecord`.

## Attribute tables are not structural state

AnnNet keeps attributes in dataframe-like tables, not inside the structural
records:

- `vertex_attributes`
- `edge_attributes`
- `slice_attributes`
- `edge_slice_attributes`
- `layer_attributes`

This is an intentional separation.

Structural state answers questions like:

- which row does this entity own?
- which endpoints does this edge connect?
- which column does this edge occupy?

Attribute tables answer questions like:

- what annotation value is attached to this vertex or edge?
- what label or score does this vertex carry?
- what metadata is attached to this edge?
- which slice-specific edge weight should be used?
- what slice-specific override applies here?
- which dataframe backend owns newly-created annotation rows?

Core code treats those tables through shared dataframe helpers. The graph object
can accept Narwhals-compatible eager dataframes, while newly-created annotation
tables follow the configured annotation backend selection. This keeps core
annotation reads, history exports, views, adapters, and IO modules on the same
backend policy instead of letting each call site choose Polars, pandas, or
PyArrow independently.

That separation is one of the core design choices in AnnNet.

## Slices are overlays, not duplicate graphs

Slice state lives in `_slices`, which maps slice identifiers to membership and
metadata:

- vertex membership
- edge membership
- slice attributes

This is not another topology store. Slices do not redefine the graph
structurally. They describe which parts of the same graph are active in a named
context.

Per-slice edge weights are handled separately through `edge_slice_attributes`
and `slice_edge_weights`.

## Multilayer state is also an overlay

Multilayer state is carried by:

- `_aspects`
- `_layers`
- `_all_layers`
- `_state_attrs`
- edge multilayer metadata in `EdgeRecord`
- derived layer indices such as `_VM`, `_nl_to_row`, `_row_to_nl`

Again, the important point is that this does not replace the structural graph.
It enriches it.

The supra-node model still ultimately resolves back to the canonical entity
rows and edge columns.

## Compatibility views

AnnNet exposes dict-like compatibility views such as:

- `entity_to_idx`
- `idx_to_entity`
- `entity_types`
- `edge_to_idx`
- `idx_to_edge`
- `edge_weights`
- `edge_definitions`
- `hyperedge_definitions`
- `edge_kind`

These views are useful for compatibility and inspection, but they are separate
from the canonical structural stores described above.
