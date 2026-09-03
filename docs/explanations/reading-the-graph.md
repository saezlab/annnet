# Reading the graph

A graph holds three kinds of thing a question can be about: the structure, the
attributes, and where each element sits. Most questions cross all three — *which
edges carry a sign, inside this layer, in the slice I called `prior`* — and the
answer to a crossing question is a table.

So the shape of this page is one sentence: **the frame is the default answer.**
If reading something out of an AnnNet takes a loop, the loop is a gap in the API
and not a thing you were supposed to write.

## An endpoint is a node id

`G.views.edges()` gives one row per edge, and its `source` and `target` columns
hold **node ids**:

```python
frame = G.views.edges()
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
crossing edge is visible as a row whose two layer columns differ.

!!! warning "This column changed"

    Before this release, `source` held the *repr of an internal tuple*, as a
    string: `"('A', ('ctrl',))"`. Nothing downstream could consume it without
    `ast.literal_eval`, and it said nothing about that. If you have code doing
    that parse, delete it — `source` is the id and `src_layer` is the layer.

### The structured form, when you want it

An endpoint is a bare id in a flat graph and an `(id, layer)` pair in a layered
one, which used to mean reading one looked like this:

```python
node = next(iter(sides.source))
node_id = node[0] if isinstance(node, tuple) else node
```

That check is a defect, not an idiom: a graph holding both layered and unlayered
edges makes it wrong, and nothing reports it. Read an endpoint through
[`as_endpoint`][annnet.core._records.as_endpoint] instead, and it has one shape everywhere:

```python
from annnet import as_endpoint, as_endpoints

as_endpoint(('akt', ('stim',)))  # Endpoint(node_id='akt', layer=('stim',))
as_endpoint('akt')  # Endpoint(node_id='akt', layer=None)
as_endpoints(edge.source)  # frozenset[Endpoint]
```

`Endpoint` keeps the store's positional shape — `endpoint[0]` is the id,
`endpoint[1]` is the layer — and `str(endpoint)` is the id, which is what a
label, a dataframe cell and a join all want.

For the common case there is no unpacking at all:

```python
edge = G.get_edge('intra_ctrl')
edge.source_id  # 'A'
edge.target_id  # 'B'
edge.layer  # ('ctrl',)  — None when the edge crosses two layers
```

## Filtering: `slice=` joins, `in_slice=` filters

These two take the same argument and do different things, and the difference is
worth reading twice:

```python
G.views.edges(slice='prior')  # every row, with prior's attributes joined on
G.views.edges(in_slice='prior')  # only prior's rows
```

`slice=` is a **join**: every edge in the graph still gets a row, and the ones
that are in `prior` gain `slice_*` columns. `in_slice=` is a **filter**: the rows
that are not in `prior` are gone.

A call that means "the edges of this slice" wants the second. Reaching for the
first and then wondering why the row count did not change is the mistake this
sentence exists to prevent.

The other two filters are unsurprising:

```python
G.views.edges(layer=('ctrl',))  # the edges of one layer
G.views.edges(include_hyper=False)  # binary rows only
G.views.hyperedges()  # hyper rows only
```

`layer=` names exactly the set
[`layers.layer_edge_set`][annnet.core._Layers.LayerAccessor.layer_edge_set]
names, so the frame and the id set never disagree. They compose:

```python
G.views.edges(layer=('ctrl',), in_slice='prior', include_hyper=False)
```

`hyperedges()` is worth its own call rather than a filter you write, because
`head`, `tail` and `members` are the columns that carry a hyperedge's shape and
they are null on every binary row.

## Naming a node-layer instead of spelling it

A layer coordinate is a tuple in the graph's aspect order. Writing one by hand
means holding a fact about the graph at the call site, and it goes wrong silently
the first time an aspect is added — `('stim',)` was right and is now the wrong
length, and a tuple of the wrong length is a coordinate nobody is on.

Name the aspects instead:

```python
G.at('akt', condition='stim')  # ('akt', ('stim',))
G.exists('akt', condition='stim')  # True
```

`at` returns the key every layered call takes — `add_edges`,
`layers.node_attrs`, `slices.add_nodes` — and raises when the node is not there,
because a key you cannot use is not an answer. `exists` is the same question
asked without raising.

Both refuse a malformed *question* even when they would answer `False` to the
node: an aspect you did not declare, or one you declared and did not name, raises
rather than quietly resolving to something else.

```python
G.exists('akt')  # KeyError: ... needs a value for ['condition']
G.exists('akt', mechansim='x')  # KeyError: unknown aspect ['mechansim']
```

## Where to go next

- [Slices and views](managers-and-views.md) — what a slice is, and how it differs
  from a layer.
- [Multilayer and multi-aspect graphs](math-multilayer.md) — the coordinate
  system the layer columns above are written in.
