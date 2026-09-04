# Node-layer values, and what scale does to them

A node-layer attribute is a value keyed by a pair — one node, one layer. There
are two very different ways that pair comes to hold a number, and the difference
is not a matter of degree.

**A person typed it.** Nine conditions, six timepoints, a few dozen nodes that
matter, and almost every pair carrying nothing. A dict keyed by the pair is
exactly right: what is absent costs nothing.

**It arrived as a table.** Twenty thousand entities across a thousand samples,
and every pair carrying a number. That is 2 × 10⁷ cells. In a dict that is
minutes to write and gigabytes to hold. As a dense `float32` array it is eighty
megabytes and a microsecond a read.

The difference is not the implementation. It is the shape. So AnnNet holds
either, and the reader never learns which.

## Attaching a table

```python
G.layers.attach_values(
    {'expression': matrix},          # conditions x nodes
    layers=[(c,) for c in conditions],
    nodes=node_ids,
)
```

Attaching costs the two index maps and nothing else. **The array is not copied,
not converted, and not read until a cell is asked for** — write into it
afterwards and the graph sees the new value.

Two options are worth knowing:

- `columns=` lets several nodes share one column, which is what a join of many
  nodes onto one measured entity needs — and it needs it without copying the
  column, which is the whole reason the array is attached rather than unpacked.
- `mask=` is a boolean array gating which cells hold a value at all, so a
  condition that was never measured stays absent rather than becoming a zero.

`G.layers.detach_values(backing)` drops one. The dict store is never dropped.

## Reading: one cell, or a rectangle

```python
G.layers.values()                 # the resolver
G.layers.matrix('expression')     # a rectangle
G.layers.node_frame(attrs=[...])  # a table
```

The resolver asks each backing in turn and **a later one wins** for a cell it can
answer. Attaching a table therefore *shadows* whatever the dict store held for
the same pair rather than blending with it — two sources for one cell is a
conflict, and blending would hide it.

`matrix` is what to hand a method:

```python
block = G.layers.matrix('expression', nodes=wanted)
block.values     # ndarray, layers x nodes
block.nodes      # the node of each column
block.layers     # the layer of each row
```

A frame of Python objects has to be unpacked before any arithmetic. This is the
arithmetic's own shape, plus the two labels needed to put an answer back on the
right rows.

### Why it is fast, and when it is not

Where the values live in one attached array, `matrix` gathers them in a single
pass in C. Where they live in the dict store, or span both, it falls back to
reading cell by cell — and **the two give the same numbers**, which is pinned by
test.

Measured on this machine:

| cells | `matrix` | cell by cell | |
|---|---|---|---|
| 102,400 | 6.8 ms (0.07 µs/cell) | 186 ms (1.82 µs/cell) | **27×** |
| 400,000 | 10.0 ms (0.03 µs/cell) | 679 ms (1.70 µs/cell) | **68×** |

The advantage grows with size, which is the signature of removing a per-cell
cost rather than making one cheaper.

The fallback is conservative: if *any* backing holding the name has no rectangle
to give — the dict store never does — the whole read goes cell by cell, even
where the fast answer would have been right. Telling those cases apart costs more
than taking the slow path does, and the slow path is never wrong.

## Identity is a separate question, and it does not shrink

Values moved into an array. **Presence did not.** Whether node `n` exists on
layer `c` is a fact the structure store holds per pair, and that is what
`layers.place` writes:

```python
G.layers.place(node_ids, [(c,) for c in conditions], mask=measured)
```

`place` registers the whole rectangle in one call, going straight at the store
rather than through the general node-adding path — which normalises each item,
resolves a coordinate and merges default attributes, none of which a rectangle of
bare ids needs.

| node-layers | before | after |
|---|---|---|
| 400,000 | 1.78 s | **0.45 s** |
| 1,600,000 | 13.69 s | **1.88 s** |

!!! warning "Identity still costs memory, and this is the open problem"

    Placing 1.6 M node-layers holds about **470 MB**, and 20 M holds several
    gigabytes. That is not the values — those are 80 MB as an array. It is the
    per-pair identity: a key in a dict, a slot in a list, and a container per
    entity.

    So the read path scales and the *placement* path does not, and no amount of
    making `place` faster changes that: it is one dict entry per pair by
    construction. Making it otherwise means a node-layer rectangle the store can
    hold without enumerating — which is a change to the store rather than to this
    seam, and it has not been made.

    Until it is: attach and read at any scale, and place deliberately. `mask=`
    is the tool — place what was measured, not the cross product.

## Where to go next

- [Reading the graph](reading-the-graph.md) — the frame as the default answer.
- [Aspects, order, and windows](aspects-and-windows.md) — the coordinates these
  values are keyed by.
