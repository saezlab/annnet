# Reading the graph

A graph holds three kinds of thing a question can be about: the structure, the
attributes, and where each element sits. Most questions cross all three — *which
edges carry a sign, inside this layer, in the slice I called `prior`* — and the
answer to a crossing question is a table, or a selection you can hand to a view.

So the shape of this page is one sentence: **the frame is the default answer,
and the selection is the default argument.** If reading something out of an
AnnNet takes a loop, a set conversion or a tuple parse, that is a gap in the API
and not a thing you were supposed to write. The surface this page describes is
stated in [the contract](../reference/api-contract.md).

## Start with the summary

`G.summary()` answers the first questions of an exploration — how big is this,
which aspects and slices does it declare, what attributes are there to filter
on, what kinds of edges does it hold — from counters and registries alone. It
builds no frame and no matrix, so it is the right first call on a graph of any
size:

```python
G.summary()
```

```
AnnNet summary: 5 node(s), 10 edge(s), 10 placement(s), 1 edge entity
  directed default: True
  aspects (2): time (ordered): [0h, 6h, 24h]; condition: [ctrl, stim, stim_x]
  layers occurring: 5
  slices (3): default (5n/10e), prior (3n/4e), fit (3n/4e)
  attrs: nodes: ['group', 'score', 'weird__name']; edges: ['confidence', 'source_db']; ...
  edges (computed): binary=6, hyper=3, node_edge=1; directed=8, undirected=2
```

`G.attrs.schema(address)` is the same idea for one address: its key columns
and its fields, with `derived=True` adding the structural columns a derived
table would carry.

## An endpoint is a node id

`G.attrs.table('edges', derived=True)` gives one row per edge, and its `source`
and `target` columns hold **node ids**:

```python
frame = G.attrs.table('edges', derived=True)
frame.select(['edge_id', 'source', 'target', 'src_layer', 'dst_layer'])
```

```
┌────────────┬────────┬────────┬───────────┬───────────┐
│ edge_id    ┆ source ┆ target ┆ src_layer ┆ dst_layer │
╞════════════╪════════╪════════╪═══════════╪═══════════╡
│ intra_ctrl ┆ A      ┆ B      ┆ ctrl      ┆ ctrl      │
│ coupling   ┆ A      ┆ A      ┆ ctrl      ┆ stim      │
└────────────┴────────┴────────┴───────────┴───────────┘
```

The layer each endpoint sits in is its own column. That is what makes the table
joinable: `source` joins against the node table, `src_layer` groups, and a
crossing edge is visible as a row whose two layer columns differ. `kind` is the
structural kind — `binary`, `hyper`, `node_edge` — and `directed` is the
direction, the same two fields `G.E.at(edge_id)` and `G.E['kind']` carry.

A hyperedge has no single `source`; its shape is in `head`, `tail` and
`members`. When every endpoint of every edge is the question, ask for the
incidence layout — one row per endpoint, with the participant's id, kind,
layer, role and coefficient, so nothing is parsed out of a string:

```python
G.attrs.table('edges', derived=True, layout='incidences')
```

```
┌─────────┬──────────┬───────────┬─────────────┬───────────┬──────────┬────────┬─────────────┐
│ edge_id ┆ position ┆ entity_id ┆ entity_kind ┆ layer     ┆ layer_id ┆ role   ┆ coefficient │
╞═════════╪══════════╪═══════════╪═════════════╪═══════════╪══════════╪════════╪═════════════╡
│ h_dir   ┆ 0        ┆ a         ┆ node        ┆ [0h, stim]┆ 0h×stim  ┆ source ┆ 1.0         │
│ h_dir   ┆ 1        ┆ b         ┆ node        ┆ [0h, stim]┆ 0h×stim  ┆ target ┆ -1.0        │
└─────────┴──────────┴───────────┴─────────────┴───────────┴──────────┴────────┴─────────────┘
```

### The structured form, when you want it

An endpoint is a bare id in a flat graph and an `(id, layer)` pair in a layered
one. Read one through
[`as_endpoint`][annnet.core._records.as_endpoint] and it has one shape
everywhere:

```python
from annnet import as_endpoint, as_endpoints

as_endpoint(('akt', ('stim',)))  # Endpoint(node_id='akt', layer=('stim',))
as_endpoint('akt')  # Endpoint(node_id='akt', layer=None)
as_endpoints(edge.source)  # frozenset[Endpoint]
```

For the common case there is no unpacking at all:

```python
edge = G.E.at('intra_ctrl')
edge.source_id  # 'A'
edge.target_id  # 'B'
edge.layer  # ('ctrl',)  — None when the edge crosses two layers
```

## Filtering a table: `slice=` joins, `in_slice=` filters

These two take the same argument and do different things, and the difference is
worth reading twice:

```python
G.attrs.table('edges', derived=True, slice='prior')  # every row, with prior's attributes joined on
G.attrs.table('edges', derived=True, in_slice='prior')  # only prior's rows
```

`slice=` is a **join**: every edge in the graph still gets a row, and the ones
that are in `prior` gain `slice_*` columns. `in_slice=` is a **filter**: the rows
that are not in `prior` are gone.

The other filters are unsurprising, and they compose:

```python
G.attrs.table('edges', derived=True, layer=('ctrl',))  # the edges of one layer
G.attrs.table('edges', derived=True, include_hyper=False)  # binary rows only
G.attrs.table('edges', derived=True, include_binary=False)  # hyper rows only
G.attrs.table('edges', derived=True, layer=('ctrl',), in_slice='prior', include_hyper=False)
```

`layer=` names exactly the set
[`layers.layer_edge_set`][annnet.core._Layers.LayerAccessor.layer_edge_set]
names, so the frame and the id set never disagree.

`limit=` is a preview: `G.attrs.table('edges', derived=True, limit=20)`
builds twenty rows and no more, so looking at a large graph costs twenty rows.
It is never a selection; nothing about the graph changes.

## Selecting, and composing selections

A question about attributes is a selection, and a selection is typed, live and
combinable. The same predicate language answers on every axis and at every
attribute address: `field=value` or `field__operator=value`, with `eq`, `ne`,
`in`, `not_in`, `lt`, `lte`, `gt`, `gte` and `isnull`. A null satisfies nothing
but `isnull=True`, and an ordered aspect compares by its declared order.

```python
strong = G.N.select(score__gte=0.5)
group_x = G.N.select(group='x')
flagged = G.N.select(flag=True)
hits = (strong | group_x) - flagged  # typed algebra, ordered by the graph
hits.ids  # ('a', 'iso')
'a' in hits, len(hits)
```

No set conversion, no id intersection by hand, and no loss of identity: a
selection made from `G.N` stays a node selection, and one made inside a view
stays scoped to that view. Two selections of different graphs refuse to combine.

The attribute addresses select the same way, with the address's own keys:

```python
G.attrs.select('node_layers', expr__gte=0.9).keys  # (('a', ('0h', 'stim')),)
G.attrs.select('node_layers', expr__gte=0.9).project('nodes')  # a node selection
G.attrs.select('edge_slices', activity__gt=0).project('edges')  # the edges
G.attrs.select('edge_slices', activity__gt=0).project('slices')  # the slices
```

A composite key projects **explicitly**. A placement is not silently collapsed
to a bare id, and an edge-slice row is not silently read as an edge.

Layers select through `G.layers.where`, with the same operators over the
aspects, and an ordered aspect understanding `__lte`:

```python
early_stim = G.layers.where(time__lte='6h', condition='stim')
```

## Views: a live, read-only window

A selection becomes a graph to read through `G.view(...)`:

```python
V = G.view(nodes=hits)  # closed boundary: edges inside the selection
V = G.view(nodes=hits, boundary='open')  # plus the edges touching it, one hop
V = G.view(layers=early_stim, slices=['prior', 'fit'])
V = G.view(edges=G.E.select(kind='hyper', confidence__gte=0.9))
```

A view resolves over stable placement keys and edge ids, in the graph's order,
with one rule set:

- **closed** keeps an edge only when every endpoint is selected; a hyperedge is
  kept whole or dropped; a self-loop has one endpoint; a selected isolated node
  stays; an edge-only filter keeps its edges and their endpoints, not every
  unrelated node; a slice is a membership set, not a request to induce edges.
- **open** keeps the edges touching a selected placement with their full
  endpoints — one hop — and reports what it added in
  `V.summary()['expanded']`.
- an edge that ends on an edge entity needs its backing edge: it is brought
  along when the filters allow it and the dependent edge is dropped otherwise,
  never a dangling reference.
- `V.view(...)` only restricts.

The view carries the graph's reading vocabulary under the graph's names —
`V.N`, `V.E`, `V.attrs` at all eight addresses, `V.summary()`, `V.B`, `V.S`,
`V.degree`, `V.layers.where`, `V.slices.compare` — scoped to what it resolved.
It is **live**: edit the parent and the view follows. It is **read-only**:
every write through it raises `ReadOnlyViewError` naming `.materialize()`.

```python
H = V.materialize()  # an independent AnnNet describing exactly the selection
```

`H` carries the kept edges whole, the aspects, the slices in scope with their
memberships cut to the selection, the attributes at all eight addresses cut to
the selected keys, and `H.uns['selection']` says what was selected. Editing `H`
changes nothing in `G`.

## Naming a node-layer instead of spelling it

A layer coordinate is a tuple in the graph's aspect order. Writing one by hand
means holding a fact about the graph at the call site, and it goes wrong silently
the first time an aspect is added. Name the aspects instead:

```python
G.at('akt', condition='stim')  # ('akt', ('stim',))
G.exists('akt', condition='stim')  # True
```

`at` returns the key every layered call takes — `add_edges`,
`G.attrs.row('node_layers', key)`, `slices.add_nodes` — and raises when the node is
not there, because a key you cannot use is not an answer. `exists` is the same
question asked without raising.

Both refuse a malformed *question* even when they would answer `False` to the
node: an aspect you did not declare, or one you declared and did not name, raises
rather than quietly resolving to something else.

## One namespace hands back the attributes

`G.attrs` is the one place attributes are read and written, at eight addresses
that answer the same expressions. Reading gives a copy you own: the stored
table (`G.attrs.nodes`), one row as a dict (`G.attrs.row('nodes', 'A')`), many
(`G.attrs.rows('nodes')`), a derived table (`G.attrs.table(...,
derived=True)`) and a schema. Editing what you read changes nothing in the
graph. Writing is a named call: `update` merges fields, `replace` swaps a whole
table, `delete` removes attributes (never the elements that carry them). The
stored table holds what was written; the derived table adds what the graph
knows — endpoints, layers, direction, kind, memberships — with the joins and
filters as explicit arguments. There is no second namespace to reconcile with.

Choosing rows comes in two kinds. `G.attrs.select('nodes', score__gt=0.5)` is a
query over the graph and stays a query: it answers again after the graph
changes. Filtering a table you have already read is your dataframe library's
job, and `G.attrs.from_frame('nodes', kept)` turns the rows that are left into a
fixed selection of the graph's keys, which does not follow the graph.

A window is a different thing and does not hand back a frame at all.
`G.layers.where(...).nodes` is a **set of ids**, the way `G.N` is a sequence of
ids. A bare plural is identities. The same plural under `attrs` is the table.

## Where to go next

- [Slices and views](managers-and-views.md) — what a slice is, and how it differs
  from a layer.
- [Multilayer and multi-aspect graphs](math-multilayer.md) — the coordinate
  system the layer columns above are written in.
- [Migrating to the current surface](api-migration.md) — every removed
  name and its replacement.

## Reading many slices at once

A slice is a named subset. Per-slice attributes are how a result lands on an
object without overwriting anything — a fit writes `activity` on the edges it
selected, inside its own slice, and the prior is untouched.

Reading that back is a cube — edge by slice by attribute — and
`slices.edge_frame` is the cube:

```python
G.slices.edge_frame(attrs=['activity'])
```

```
┌──────────┬──────┬──────┬──────┐
│ slice_id ┆ e1   ┆ e2   ┆ e3   │
╞══════════╪══════╪══════╪══════╡
│ prior    ┆ 1.0  ┆ null ┆ null │
│ fit      ┆ null ┆ -1.0 ┆ 1.0  │
└──────────┴──────┴──────┴──────┘
```

That table is the answer to *which interactions carried signal in which
condition*, and it is a shape a `source`/`target`/`weight` frame structurally
cannot hold. `format='long'` gives one row per cell instead, and `pairs=`
cherry-picks columns without paying for their cross product. The same cube is
one address of `G.attrs` — `G.attrs.table('edge_slices')` — when the question
is a row per membership rather than a matrix.

### Diffing two slices

The set operations — `union`, `intersect`, `difference` — answer *how many*.
`compare` answers *which, and on which side*:

```python
G.slices.compare('prior', 'fit', axis='edges')
```

```
┌─────────┬────────┐
│ edge_id ┆ status │
╞═════════╪════════╡
│ e1      ┆ a_only │
│ e2      ┆ both   │
│ e3      ┆ b_only │
└─────────┴────────┘
```

`axis='nodes'` is the other axis. The order matters — `a_only` names the first
argument — so `compare(a, b)` and `compare(b, a)` are different tables. The
intersection is a selection a view takes as it is:
`G.view(slices=G.slices.intersect(['prior', 'fit']))`.

### Building a slice

Creating a slice and filling it was three calls; it is one:

```python
G.slices.add('prior', edges=prior_ids, role='input')
```

A slice built by naming *nodes* holds no edges, so every read of it sees an
edgeless graph. `induce_edges` is the missing half, and which edges it means is a
choice rather than a default worth guessing:

```python
G.slices.induce_edges('picked')  # both: the induced subgraph
G.slices.induce_edges('picked', mode='any')  # any: reaches outside the slice
```

`hyper='skip'` leaves hyperedges out, for a reader that cannot hold one.
