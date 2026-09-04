# Edge tables, ids, and what a format cannot hold

Two boundaries meet in this package's IO: the one where a table becomes a graph,
and the one where a graph becomes a file that holds less than it does. Both are
places where something can be lost quietly, so both name what they do.

## An id says what the edge is

An edge list is the most common way a network is handed around. Turning one into
a graph used to look like this:

```python
prior_edge_ids = []
for index, (source, effect, target) in enumerate(rows):
    edge_id = f'prior_{index:02d}'
    G.add_edges(source, target, edge_id=edge_id, slice='prior')
    G.attrs.set_edge_attrs(edge_id, interaction=effect)
    prior_edge_ids.append(edge_id)
```

Two of those lines are defects rather than verbosity:

- **The id comes from the row's position.** Re-sort the input and every edge is
  renamed, so two runs over the same data disagree about what anything is called.
- **The side list exists** only because nothing on the graph could answer *which
  edges came from this table*. Slices can, and could.

One call, and neither problem:

```python
G = an.from_edge_frame(rows, sign='effect', slice='prior')
an.add_edges_from_frame(G, more_rows, sign='effect', slice='signalling')
```

An id is **derived from what the edge is** — its endpoints, where it sits, and
everything the row says about it, hashed in sorted key order. Re-sort the table
and the ids are identical; change a recorded value and the id changes with it.
`G.slices.edges('prior')` is what the side list was for.

## When an id is already taken

A batch that half-lands is worse than one that does not: the caller sees an
exception and a graph that is neither what it was nor what it asked for. So the
policy is decided before anything is written.

```python
an.add_edges_from_frame(G, rows, on_conflict='error')  # the default
an.add_edges_from_frame(G, rows, on_conflict='skip')  # keep what the graph has
an.add_edges_from_frame(G, rows, on_conflict='replace')  # keep what the batch brings
an.add_edges_from_frame(G, rows, on_conflict='rename')  # keep both
```

`error` raises `EdgeIdConflict`, names the ids, and **leaves the graph
untouched**. `rename` rewrites the returned id list in place, so the row-order
mapping the call hands back stays true whatever the policy did.

## What a projection costs

Most formats hold pairs. A hyperedge is not a pair, so every exporter has to
decide, and the words for that decision now live in one place:

| | |
|---|---|
| `skip` | drop the hyperedge |
| `reify` (alias `star`) | add a node standing for the relation, join every member to it |
| `expand` (alias `clique`) | join every pair of members |

**None of the three is lossless**, which is why a caller picks rather than a
default deciding. `skip` loses the relation. `expand` loses *which* members were
one relation — a four-member hyperedge becomes six pairs that no longer say they
belong together. `reify` keeps that, at the cost of a node that is not an entity.

```python
an.to_nx(G, hyperedges='star')
an.to_graphml(G, path, hyperedges='clique', coefficients='drop')
```

One loss is loud enough to refuse by default: a hyperedge may weight each member
separately, and a pair between two members has nowhere to record that. `expand`
raises `CoefficientsWouldBeLost` rather than dropping them, and
`coefficients='drop'` is how you say you accept it.

`is_flat(G)` answers whether the question arises at all, and
`_structure.hyperedges_with_coefficients(G)` names the edges that make it sharp.

## What a write cannot hold, and what it now does hold

A node-layer value that arrived as a matrix lives in an attached array rather
than in the contextual store. The manifest read only the store — so **every value
an attach had joined was written nowhere**, and the file read back null in its
place without saying anything.

```python
G.write(path)  # materialise: the cells survive
G.write(path, attached='drop')  # leave them out, deliberately
G.write(path, attached='error')  # refuse, naming what was refused
```

There is no mode that loses them quietly. Materialising costs size — a dense
array becomes one stored row per non-null pair the graph holds a node-layer for —
and holding arrays as arrays is a storage-format question that is still open.

!!! note "Only placed node-layers are written"

    Materialising walks the node-layers the graph holds. A value on a pair
    nothing placed has no row to be written on, which is consistent with
    [what placement is for](values-and-scale.md): attach and read at any scale,
    place the network's own entities.

## Where to go next

- [Node-layer values and scale](values-and-scale.md) — why a value may be in an
  array in the first place.
- [Storage and IO](io-annnet.md) — the native format's layout.
