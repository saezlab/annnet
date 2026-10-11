# The public surface

<!--
tests/test_public_surface.py reads this page both ways: every name it states
as `anet.<name>` must be on the object, and every public name the object
carries must be stated here. The machine-readable inventory of the same
surface is docs/reference/public-surface.json. Regenerate it with
`python tools/public_surface.py` after a deliberate change of the surface.
-->

This page lists what the package exports, and what an `AnnNet` object, its
namespaces, a view and a selection answer to. It also states the behavior
that you can rely on.

In the examples, `anet` is an `AnnNet`, `V` is a `GraphView`, and `address`
is one of the eight attribute addresses. A name that this page does not state
is internal: a name that begins with an underscore, and a module under
`annnet.core` other than the names that `annnet.core` exports. Internal names
can change without notice. See [Public and internal APIs](api-boundary.md).

## 1. Exports

`annnet` exports the graph and its types, the IO functions, the adapters, the
backend selectors and the subpackages. The exact list is `annnet.__all__`, and
the inventory holds it. The types a caller meets through the graph surface:

| name | what it is |
|---|---|
| `AnnNet` (`Graph` is the same class) | the graph |
| `GraphView` | the live, read-only view `anet.view(...)` returns |
| `Attrs` | the attribute namespace `anet.attrs` |
| `NodeSequence`, `EdgeSequence` (from `annnet.core`) | `anet.N`, `anet.E`, and every selection made from them |
| `RowSelection` | what `anet.attrs.select(address, ...)` and `anet.attrs.from_frame(address, frame)` return |
| `Schema` | what `anet.attrs.schema(address)` returns |
| `NodeView`, `EdgeView`, `EdgeType` | the records `anet.N.at(id)` and `anet.E.at(id)` return |
| `Aspect`, `OrderedLabels`, `as_aspect`, `LayerSelection` | aspect declarations and a layer window |
| `Endpoint`, `as_endpoint`, `as_endpoints` | endpoint identities |
| `Provenance`, `Set`, `Space`, `ValueMatrix`, `ValueResolver`, `MatrixValues`, `ContextualValues` | provenance, spaces and attached numerical backings |
| `BOUNDARIES` | `('closed', 'open')`, the two boundary rules of a view |

`annnet.core.__all__` is exactly: `AnnNet`, `Aspect`, `Attrs`, `BOUNDARIES`,
`ContextualValues`, `EdgeSequence`, `EdgeType`, `EdgeView`, `Endpoint`,
`Graph`, `GraphView`, `LayerSelection`, `MatrixValues`, `NodeSequence`,
`NodeView`, `OrderedLabels`, `Provenance`, `RowSelection`, `Schema`, `Set`,
`Space`, `ValueMatrix`, `ValueResolver`, `as_aspect`, `as_endpoint`,
`as_endpoints`.

## 2. The graph object

### 2.1 Construction

```python
anet = AnnNet(directed=None, aspects=None)
```

`directed` is the default of a binary edge that declares none; `aspects` is
`{aspect: [label, ...]}` for a multilayer graph. Nothing else is built: an
empty graph holds no table, index or per-element record it was not asked for.
A flat graph gains aspects through `anet.layers.set_aspects(...)` and hyperedges
through `anet.add_edges(...)` with no change of surface.

### 2.2 Add and remove elements

```python
anet.add_nodes(nodes, slice=None, layer=None, **attributes)   # one or many
anet.add_edges(edges_or_source, target=None, *, weight=1.0, edge_id=None, directed=None,
            parallel='update', slice=None, as_entity=False, propagate='none',
            flexible=None, **attributes)
anet.remove_nodes(node_ids, *, errors='raise')
anet.remove_edges(edge_ids, *, errors='raise')
```

A string is a node, a `(node_id, layer_coord)` tuple is a placement, a list
holds many. An edge is `(source, target)`, a mapping with `source`/`target` or
`head`/`tail` or `members`, or a list of those; a mapping with `members` is a
hyperedge; a `head`/`tail` mapping is a directed hyperedge. On a multilayer
graph a bare id names a node only while it sits on one layer; otherwise the
placement tuple is required and a bare id raises `ValueError`. `remove_nodes`
of a bare id removes every placement of that id.

### 2.3 Axes, counts and identity

```python
anet.N  # NodeSequence: the nodes, in row order
anet.E  # EdgeSequence: the structural edges, in column order
len(anet.N), len(anet.E)  # constant time, whatever the size of the graph
anet.shape  # (len(anet.N), len(anet.E))
anet.supra_shape  # (placements, edges)
anet.nv_supra  # the number of placements
anet.supra_nodes()  # every (node_id, layer_coord) placement, in row order
anet.at(node_id, **aspects)  # the placement key one node sits on; KeyError if it does not
anet.exists(node_id, **aspects)  # the same question, answered with a bool
anet.has_node(node_id)
anet.has_edge(source=None, target=None, edge_id=None)
anet.entity_kinds()  # {entity_id: 'node' | 'edge'}
```

`len(anet.N)` is the number of distinct node ids; `anet.nv_supra` counts placements.
Both are maintained counters and never enumerate. Iterating `anet.N` yields ids;
`anet.N[i]` is the id at one position; `anet.N['field']` reads a column as a
read-only array and `anet.N['field'] = values` writes one; `anet.N.at(id)` is one
record. The same holds of `anet.E`.

### 2.4 Attributes and metadata

```python
anet.attrs  # the Attrs namespace (section 3)
anet.uns  # graph-level metadata, a dict
anet.summary()  # counts, aspects, slices, fields and edge tallies, built without a frame or a matrix
```

### 2.5 Matrices

```python
anet.B  # incidence matrix over every entity row (placements and edge entities)
anet.S  # signed incidence matrix
anet.A  # adjacency matrix
anet.H  # hypergraph incidence matrix
anet.L  # Laplacian
anet.matrices  # the named-matrix namespace (build, cache, drop)
anet.idx  # position <-> identity translation for a materialized matrix
```

### 2.6 Lookups and traversal

```python
anet.degree(entity)             # distinct incident structural edges; costs the degree, not the graph
anet.neighbors(entity)          # every neighbour; the directed reads:
anet.out_neighbors(node_id), anet.in_neighbors(node_id), anet.successors(node_id), anet.predecessors(node_id)
anet.incident_edges(nodes, direction='both')   # [(column, EdgeView), ...]
anet.edge_list()                # every binary edge as (source, target, edge_id, weight)
anet.global_count(kind)         # one count of one kind of slice member, by name
anet.is_multilayer              # whether the graph declares aspects beyond the flat one
anet.make_undirected(*, drop_flexible=True, update_default=True)
```

A self-loop and a hyperedge each count once in the degree; a zero-weight edge
counts; a placeholder edge never does; an unknown entity has degree 0; an
ambiguous bare id on a multilayer graph raises `ValueError`.

### 2.7 Namespaces

```python
anet.layers  # aspects, layer windows (where), placements, values, layer algebra
anet.slices  # slice registry, membership, compare, edge_frame
anet.ops  # copy, subgraph, extract, union/intersection/difference, reverse, incidence
anet.history  # snapshots and diffs
anet.provenance  # provenance records
anet.spaces  # the spaces a node belongs to
anet.cache  # matrix caches
anet.nx, anet.ig, anet.gt  # proxies onto networkx / igraph / graph-tool over this graph
```

The public names of each owned namespace are in the inventory. `anet.nx`,
`anet.ig` and `anet.gt` expose the surface of the library they wrap, which is that
library's and changes with its version.

### 2.8 Views and IO

```python
anet.view(nodes=None, edges=None, layers=None, slices=None, *, predicate=None, boundary='closed')
anet.write(path, *, matrix=False, **kwargs)
AnnNet.read(path, **kwargs)
```

### 2.9 The complete graph surface

Every public name an `AnnNet` object answers to, and nothing else:

`anet.A`, `anet.B`, `anet.E`, `anet.H`, `anet.L`, `anet.N`, `anet.S`, `anet.add_edges`, `anet.add_nodes`,
`anet.at`, `anet.attrs`, `anet.cache`, `anet.degree`, `anet.edge_list`, `anet.entity_kinds`,
`anet.exists`, `anet.global_count`, `anet.gt`, `anet.has_edge`, `anet.has_node`,
`anet.history`, `anet.idx`, `anet.ig`, `anet.in_neighbors`, `anet.incident_edges`,
`anet.is_multilayer`, `anet.layers`, `anet.make_undirected`, `anet.matrices`,
`anet.neighbors`, `anet.nv_supra`, `anet.nx`, `anet.ops`, `anet.out_neighbors`,
`anet.predecessors`, `anet.provenance`, `anet.read`, `anet.remove_edges`,
`anet.remove_nodes`, `anet.shape`, `anet.slices`, `anet.spaces`, `anet.successors`,
`anet.summary`, `anet.supra_nodes`, `anet.supra_shape`, `anet.uns`, `anet.view`, `anet.write`.

## 3. Attributes: eight addresses, one shape

An address is what a set of attributes is keyed by. The eight, with the key
each takes:

| address | key | key columns |
|---|---|---|
| `nodes` | `node_id` | `node_id` |
| `edges` | `edge_id` | `edge_id` |
| `slices` | `slice_id` | `slice_id` |
| `aspects` | `aspect` | `aspect` |
| `layers` | layer coordinate tuple | `layer` |
| `edge_slices` | `(slice_id, edge_id)` | `slice_id`, `edge_id` |
| `node_layers` | `(node_id, layer_coord)` | `node_id`, `layer` |
| `elementary_layers` | `(aspect, label)` | `aspect`, `elementary_layer` |

Every address answers the same expressions, differing only in the name of the
address. A read returns a copy, and a write is a named call:

```python
# Read: detached copies. Editing one changes nothing in the graph.
anet.attrs.nodes                      # the stored table, a dataframe; one property per address
anet.attrs.table('nodes', derived=False, backend=None, columns=None, limit=None, layout=None, **query)
anet.attrs.row('nodes', key)          # one row, a dict; a cell is row(...).get(name, default)
anet.attrs.rows('nodes', keys=None)   # {key: dict}
anet.attrs.schema('nodes', derived=False)      # keys and fields without a frame
anet.attrs.audit()                    # rows whose structural key no longer exists
anet.attrs.backend                    # the dataframe backend tables are rendered in

# Write: each call is checked whole before anything changes.
anet.attrs.update('nodes', {key: {...}, ...}, *, layer=None, key=None)   # merge the fields named
anet.attrs.replace('nodes', table)                  # replace the whole address with a table
anet.attrs.delete('nodes', keys=None, names=None)   # remove attributes, never elements

# Select.
anet.attrs.select('nodes', score__gte=0.5)     # a live RowSelection (section 4)
anet.attrs.from_frame('nodes', frame)          # a fixed RowSelection from a frame's key columns

address in anet.attrs; iter(anet.attrs)  # the eight addresses
```

`anet.attrs` is not indexed and its table properties are read-only. The forms
`anet.attrs[address]`, `anet.attrs[address, key]`, assigning to or deleting either,
and `anet.attrs.nodes = table` raise `TypeError` naming the call to use instead
(`row`, `update`, `replace`, `delete`).

Semantics:

- **A null in an update removes the field.** `update` with `None` or NaN for a
  field removes it from the row; a row never shows a null field; a table cell
  that is null means the same. `delete` is the explicit way to say it. One
  rule for every address and every backend.
- **`delete(address, keys=None, names=None)`** removes attributes and never the
  node, edge, slice or layer that carries them:

    | `keys` | `names` | effect |
    |---|---|---|
    | given | omitted | clear every attribute of those rows |
    | omitted | given | drop those fields from every row of the address |
    | given | given | drop those fields from those rows |
    | omitted | omitted | `ValueError`; never "everything" |

    An explicit empty `keys` or `names` selects nothing and is a no-op. A name
    the address holds no attribute for raises `KeyError`, a reserved name
    `ValueError`. A field that no row carries after the call is gone from the
    schema and the table columns. The return value is the number of rows that
    lost an attribute.
- **`replace(address, table)`** needs the address's key columns; a key the
  table omits loses its attributes and keeps its existence. Elements are never
  added or removed by a table.
- **A write is all or nothing.** `update`, `replace` and `delete` are checked
  whole before any write. If one fails partway — an unknown key, a reserved
  field, a duplicate key, a composite-key collision, a value a
  flexible-direction policy cannot evaluate — the attribute values, **the
  fields and their types and order**, the topology any policy rewrote, the
  node-key index and every derived cache are as they were, and the exception
  propagates. A scalar write never copies the graph.
- **Reserved fields** — the key columns and, for edges, the structural columns
  (`source`, `target`, `head`, `tail`, `members`, `weight`, `directed`, `kind`,
  `slice`, `slice_weight`, `flexible`, `edge_type`) — are refused with
  `ValueError`. A node attribute named `kind` is a user attribute and is never
  overwritten by the entity kind.
- **Keys** are validated: an unknown key raises `KeyError`; a key of the wrong
  shape raises `TypeError`; a key that appears twice in one batch, however it
  is spelled, raises `ValueError` before anything is written. `node_layers`
  takes `(node_id, layer_coord)`; a bare node id needs `layer=` in `update()`.
  Elementary layers are keyed `(aspect, label)`; the legacy `aspect_label`
  string is accepted only at the IO boundary and rejected as ambiguous when it
  could name two rows.
- `table(...)` with `derived=True` renders everything the graph can say about
  the subject: for `edges`, `kind` (`binary`, `hyper`, `node_edge`),
  `directed`, `weight`, `ml_kind`, `source`, `target`, `src_layer`,
  `dst_layer`, `head`, `tail`, `members`; with `layout='incidences'`, one row
  per endpoint with `edge_id`, `position`, `entity_id`, `entity_kind`,
  `layer`, `layer_id`, `role`, `coefficient`, `kind` and `directed`, so an
  endpoint joins to its node, layer and edge rows without parsing anything.
  `limit=` builds that many rows and no more; it is a preview, never a
  selection. Query arguments (`slice=`, `in_slice=`, `layer=`,
  `include_hyper=`, `include_binary=`, `include_directed=`, `include_weight=`,
  `resolved_weight=`) apply to the derived edges table only.
- `schema(address)` reports `keys`, `fields` (name, type), `names`, `count`
  and `index` without building a frame.

## 4. Predicates and selections

### 4.1 One predicate language

Every `select` — `anet.N.select`, `anet.E.select`, `anet.attrs.select`,
`anet.layers.where` — parses the same conditions: `field=value` or
`field__operator=value`, with operators `eq`, `ne`, `in`, `not_in`, `lt`,
`lte`, `gt`, `gte`, `isnull`. A mapping with `(field, operator)` tuple keys
names a field that itself contains `__`; a literal field named `a__b` wins
over the split. A null value satisfies nothing but `isnull=True`. An ordered
aspect compares by its declared order; an ordering operator on a categorical
aspect raises `ValueError`; an unknown field raises `KeyError`; an
incomparable pair raises `TypeError`, at the call.

### 4.2 Typed selections

```python
sel = anet.N.select(score__gte=0.5)  # NodeSequence, live
sel.ids  # tuple of ids, in the graph's order
len(sel)
id in sel
list(sel)
sel & other
sel | other
sel - other  # typed algebra; result is ordered by the root graph
sel.select(...)  # narrows further
anet.N.select(['a', 'b'])  # explicit ids: checked to exist, fixed
anet.N.select(mask)  # a boolean mask over this sequence, bound now
anet.N.select(callable)  # an id predicate; its exceptions propagate
anet.N.find(**conditions)  # the one match, or KeyError
anet.N.at(node_id)  # NodeView; anet.E.at(edge_id) -> EdgeView
anet.N.column(name, default=None)
anet.N.set_column(name, values)
anet.N.axis, anet.N.id_key, anet.N.id_column, anet.N.intrinsic_names, anet.N.graph
anet.E.effective_weight(edge_id, slice=None)
```

A selection keeps its plan, so a view built on it re-evaluates when the graph
changes; explicit ids and masks are fixed at the call and never retarget after
a slot is reused. A callable is evaluated again on every read, and so is
everything derived from a view that holds one: a selection made in it, a union
or difference, the projection of its attribute rows and a view built on any of
these, even when the state the callable reads changed without any change of the
graph. Each leaf is evaluated in the scope it was made in — a
selection from `V.N` inside `V` — and two selections from different scopes
combine by the root graph's order. Combining selections of different graphs
raises `ValueError`; a node selection with an edge selection raises
`TypeError`.

### 4.3 Row selections and projections

`anet.attrs.select(address, ...)` returns a `RowSelection` with `address`,
`graph`, `keys` (the address's keys, in the address's order), `ids` (the same
for single-key addresses), `rows()`, `len`, iteration, membership and the same
`&`, `|`, `-`. A selection on a composite address projects explicitly:
`select('node_layers', ...).project('nodes')` and
`select('edge_slices', ...).project('edges')` or `.project('slices')`. Any
other projection raises `TypeError`; nothing collapses a placement to a bare
id silently.

### 4.4 A live query and a table you filtered are different things

`select` (on `anet.N`, `anet.E`, `anet.attrs`, `anet.layers`) is a query over the graph.
It keeps its conditions and answers again after the graph changes. A table read
with `anet.attrs.table` or `anet.attrs.<address>` is a snapshot of the moment it was
read, and filtering it is the dataframe library's job:

```python
table = anet.attrs.table('nodes')  # any supported backend
kept = table.filter(pl.col('score') > 0.5)  # the caller's own dataframe API
chosen = anet.attrs.from_frame('nodes', kept)  # a fixed RowSelection
```

`from_frame(address, frame)` reads only the address's key columns, validates
them (a missing key column or a null key raises `ValueError`, an unknown key
`KeyError`), and returns a `RowSelection` of exactly those keys, in the
address's order and without repeats. It is fixed: it holds the identities the
frame named, not the condition that produced the frame, so a node added later
that satisfies the same filter is not picked up, and a key that is removed and
added again is the same key. An empty frame is an empty selection. It composes
with live selections through `&`, `|` and `-` and projects like any other row
selection. AnnNet accepts no dataframe expression in `select` and has no second
keyword syntax for fields.

## 5. Views: live, read-only, composable

```python
V = anet.view(nodes=None, edges=None, layers=None, slices=None, *, predicate=None, boundary='closed')
```

`nodes` takes ids, placement keys, a `NodeSequence`, a `RowSelection` of
`nodes` or `node_layers`, a mask or a callable; `edges` takes ids, an
`EdgeSequence`, a `RowSelection` of `edges`, a mask or a callable; `layers`
takes coordinates or `anet.layers.where(...)`; `slices` takes ids or a slice
selection. An unknown id raises `KeyError`. A selection of another graph
raises `ValueError`.

Resolution, over stable placement keys and edge ids, in the graph's order:

- **closed** (the default): an edge is kept only when every endpoint is
  selected; a hyperedge is kept whole or dropped; a self-loop has one
  endpoint; a half-edge is judged by its nonempty side; a selected isolated
  node stays. An edge-only filter keeps its edges and their endpoints, not
  every unrelated node. A slice is a membership set, not a request to induce
  edges.
- **open**: the edges touching a selected placement are kept with their full
  endpoints — one hop — with the edge and slice restrictions still applied
  and never beyond the parent view. The added endpoints are reported by
  `V.summary()['expanded']`.
- **edge entities**: an edge that ends on an edge entity needs the backing
  edge; an edge-only filter brings it along, a closed node filter keeps the
  dependent edge only when the backing edge is kept, an open filter brings the
  backing edge when the filters allow it and drops the dependent edge
  otherwise. Nothing substitutes a node or leaves a dangling reference.
- **nested**: `V.view(...)` only restricts; its filters intersect with `V`'s
  and keep every predicate.

The surface of a view:

`V.A`, `V.B`, `V.E`, `V.H`, `V.L`, `V.N`, `V.S`, `V.aspects`, `V.at`,
`V.attrs`, `V.boundary`, `V.degree`, `V.directed`, `V.edge_list`,
`V.entity_kinds`, `V.exists`, `V.graph`, `V.has_edge`, `V.has_node`, `V.idx`,
`V.in_neighbors`, `V.incident_edges`, `V.is_multilayer`, `V.layers`,
`V.materialize`, `V.matrices`, `V.neighbors`, `V.nv_supra`,
`V.out_neighbors`, `V.parent`, `V.predecessors`, `V.shape`, `V.slices`,
`V.successors`, `V.summary`, `V.supra_nodes`, `V.supra_shape`, `V.uns`,
`V.view`.

- `V.N`, `V.E`, `V.attrs` are the graph's sequences and attribute namespace
  scoped to the resolved membership: the same expressions, the same eight
  addresses, keys kept only while their elements remain in the view.
- `V.B` rows are the selected placements and the edge entities of the
  selected edges, in row order; `V.shape` follows the graph's collapsed-node
  convention and `V.supra_shape` the placement convention. Each named matrix
  follows its own documented projection.
- `V.layers` and `V.slices` carry the read operations of the graph's
  namespaces (`where`, `window`, `aspect`, `list_layers`, `has_presence`,
  `layer_node_set`, `layer_edge_set`, `values`, `matrix`, `node_frame`;
  `list`, `exists`, `count`, `active`, `nodes`, `edges`, `info`, `union`,
  `intersect`, `difference`, `compare`, `edge_frame`), restricted to the view.
- `V.uns` is a read-only mapping over the graph's metadata.
- A view is live: it re-resolves when any clock of the graph moves (structure,
  either attribute axis, slice membership, aspects, the six contextual
  levels), and on every read when a filter is a callable.
- **Every write is refused** with `ReadOnlyViewError` (a `TypeError`) naming
  `.materialize()`: through `V.N[...] = `, `V.attrs.update`, `V.attrs.replace`,
  `V.attrs.delete`, the `V.attrs.backend` setter, `V.uns`, the table
  properties, and every selection made from `V`. `V.attrs.row`, `rows`,
  `table`, `select` and `from_frame` read, scoped to the view, and a row is a
  detached dict. A mutating graph method (`add_nodes`, `remove_edges`, ...) is absent
  and `AttributeError` names `.materialize()`. A read the view does not
  support raises rather than answering from the parent.
- `V.materialize(copy_attributes=True)` returns an independent `AnnNet` that
  describes exactly the selection: the kept edges with their full endpoints,
  coefficients, direction, weight, kind, layer assignment and policy; the
  aspect declarations whole; the slices in scope with memberships cut to the
  selection (the default slice always exists); the attributes at all eight
  addresses cut to the selected keys (elementary-layer attributes travel with
  the declarations); `uns` deep-copied where it can be, with
  `uns['selection']` recording what was selected; attached numerical backings
  by reference. No placeholder placement is inserted.

## 6. Records and edge fields

`anet.E.at(edge_id)` is an `EdgeView`: `kind` is the structural kind — `binary`,
`hyper` or `node_edge` — and `directed` is the separate direction. `ml_kind`
is the multilayer role (`intra`, `inter`, `coupling`), answered as `intra` for
every structural edge of a flat graph without a per-edge record. The same
three fields appear under the same names in `anet.E[...]` columns, the derived
edges table and the summary, so no reader has to translate between spellings.

## 7. Exceptions

| situation | exception |
|---|---|
| unknown node, edge, slice, aspect, layer or attribute key | `KeyError` |
| a key or argument of the wrong shape or type; an incomparable predicate pair; a node selection given as edges; a removed subscription or table assignment of `anet.attrs` | `TypeError` |
| a reserved field; a repeated key in one batch; a null or missing key column; `delete` given neither keys nor names; an ordering operator on a categorical aspect; an ambiguous bare id; a selection of another graph; a bad `boundary`, `limit`, `layout` or operator | `ValueError` |
| any write through a view | `ReadOnlyViewError` (subclass of `TypeError`) |
| a removed name | `AttributeError` naming the replacement (section 8) |

## 8. Removed names and their replacements

Every removed name raises `AttributeError` with its replacement in the
message.

| removed | replacement |
|---|---|
| `anet.obs`, `anet.var` | `anet.attrs.nodes`, `anet.attrs.edges` |
| `anet.views.*` | `anet.attrs.table(address, derived=True, ...)` |
| `anet.nodes()`, `anet.edges()` | `list(anet.N)`, `list(anet.E)` |
| `anet.nv`, `anet.ne`, `anet.ncount()`, `anet.ecount()` | `len(anet.N)`, `len(anet.E)`, `anet.nv_supra` |
| `anet.get_node`, `anet.get_edge` | `anet.N.at`, `anet.E.at` |
| `anet.get_edge_ids`, `anet.get_edges_by_direction` | `anet.has_edge(s, t)[1]`, `anet.E.select(directed=...)` |
| `anet.in_edges`, `anet.out_edges` | `anet.incident_edges(nodes, direction=...)` |
| `anet.remove_node`, `anet.remove_edge` | `anet.remove_nodes`, `anet.remove_edges` |
| the 24 `get_*`/`set_*` attribute methods | `anet.attrs.row`, `anet.attrs.update`, `anet.attrs.rows`, `anet.attrs.select`, `anet.E.effective_weight`, `anet.uns` |
| `anet.attrs[address]`, `anet.attrs[address, key]`, their assignment and `del`, `anet.attrs.<address> = table`; the live `Row` | `anet.attrs.<address>` / `table`, `row`, `update`, `replace`, `delete` |
| `anet.layers.*attrs*`, `anet.slices.attrs` | the `layers`, `aspects`, `elementary_layers`, `node_layers` and `slices` addresses of `anet.attrs` |
| `anet.cache.get_adjacency/get_csr/get_csc` | `anet.cache.adjacency/csr/csc` |
| `V.obs`, `V.var`, `V.nodes_df`, `V.edges_df`, `V.node_count`, `V.edge_count`, `V.node_ids`, `V.edge_ids`, `V.subview` | `V.attrs.table(...)`, `len(V.N)`, `len(V.E)`, `V.N.ids`, `V.E.ids`, `V.view(...)` |
| `annnet.core.ProvenanceAccessor` | `annnet.core.Provenance` |
