# Aspects, order, and windows

A layer coordinate is one label per aspect. `('12h', 'mapk')` says *the twelve-hour
timepoint of the MAPK mechanism*, and which position means which aspect is the
graph's business, declared once.

Two things follow from that, and this page is about both: an aspect may or may
not have an **order**, and selecting layers should be a **query** rather than
arithmetic over tuples.

## Ordinal and categorical

Some aspects are categorical — a mechanism, a compartment, a name. No value comes
before another, and any order you see is the order somebody typed.

Some are ordinal — a timepoint, a dose, a stage. `'1h'` comes before `'12h'`, and
that is a fact about the data rather than about the typing.

Declare which:

```python
from annnet import Aspect

G.layers.set_aspects(
    ['time', 'mechanism'],
    {
        'time': Aspect(['0h', '1h', '12h', '24h'], ordered=True),
        'mechanism': ['mapk', 'pi3k'],  # a bare list is categorical
    },
)
```

A bare list still works and means `ordered=False`, so nothing declared the old
way changes. To say it after the fact:

```python
G.layers.set_ordered('time')
```

An ordered aspect answers the questions that need to know what comes before what:

```python
time = G.layers.aspect('time')
time.index('12h')  # 2
time.before('12h')  # ['0h', '1h']
time.after('12h', inclusive=True)  # ['12h', '24h']
time.consecutive_pairs()  # [('0h','1h'), ('1h','12h'), ('12h','24h')]
time.normalized_position('12h')  # 0.666…
```

A categorical one **refuses** them:

```python
G.layers.aspect('mechanism').index('mapk')
# ValueError: index needs an ordered aspect, and this one is categorical.
```

That refusal is the point. The declaration order is always *available*; answering
with it would be the declaration order pretending to be a meaning, and an ordinal
question asked of a categorical aspect is a question about data that does not
exist.

!!! note "Declaration order is kept"

    An aspect's values used to live in a `set`, which has no order — so a graph
    declared with `['basal', 'stim', 'late']` read its layers back as
    `['basal', 'late', 'stim']`. For a categorical aspect that is cosmetic. For
    an ordinal one it was the wrong order, silently. `elem_layers`,
    `list_layers` and the layer product now all read back in declaration order.

    The `'_'` placeholder is synthetic rather than declared, so it has no
    declared position; `list_layers(include_placeholder=True)` puts it first,
    where it cannot be mistaken for one of your values.

## Windows

Selecting layers used to be a comprehension over the layer product:

```python
window = [aa for aa in G.layers._all_layers if aa[0] in TIMES[:3]]
```

Three facts about the graph live in that line — which position `time` is in, what
its values are, and what order they come in. All three are the graph's, and all
three go wrong silently the first time an aspect is added.

Name the aspect instead:

```python
G.layers.where(time__lte='12h')
# LayerSelection(6 layer(s): [('0h','mapk'), ('0h','pi3k'), ('1h','mapk'), …])
```

The predicates are `aspect=value` for equality, or `aspect__operator=value`:

| operator | means |
|---|---|
| `eq` (the default) | equal to |
| `ne` | not equal to |
| `in`, `not_in` | in a collection, or not |
| `lt`, `lte`, `gt`, `gte` | before, at-or-before, after, at-or-after |

The last four ask *where a value sits*, so they need an ordered aspect and refuse
a categorical one. Several predicates combine with **and**:

```python
G.layers.where(time__lte='12h', mechanism='mapk')
```

The window is resolved off the aspect declaration, so it costs the number of
layers rather than the size of the graph.

### What sits on a window

```python
window = G.layers.where(time='0h', mechanism='mapk')

window.nodes  # {'A', 'B', 'C'} — the ids on those layers
window.node_layers  # {('A', ('0h','mapk')), …} — the keys, when one node is on several
window.edges  # the edges whose every endpoint is inside
window.crossing  # the edges with an endpoint inside and one outside
window.boundary  # the nodes inside that a crossing edge touches
```

`edges` is **closed**: an edge with one endpoint outside the window is not in the
window. That is the safe default — a window that quietly included half-edges
would give an analysis neighbours it did not select. What closure leaves out is
exactly `crossing`, and the two never overlap.

`boundary` is where the window was cut. A node there has a neighbour the window
does not hold, so an analysis over `edges` alone treats it as though it had none —
which is worth knowing before you read the result.

Widening the window turns crossing edges into inside ones:

```python
narrow = G.layers.where(time='0h', mechanism='mapk')
wide = G.layers.where(time='0h')

'across_mechanism' in narrow.crossing  # True
'across_mechanism' in wide.edges  # True
```

Each of the five is one pass over the axis it asks about, and `edges` and
`crossing` share theirs.

## The boundary, on the layer algebra

`layer_union`, `layer_intersection`, `layer_difference`,
`create_slice_from_layer`, `subgraph_from_layer_tuple` and
`subgraph_from_layer_union` all take `boundary=`, and it means what closure means
above:

```python
G.layers.layer_union([('a',), ('b',)], include_coupling=True)
# closed, the default: an edge is kept only if every layer it touches is in the union

G.layers.layer_union([('a',), ('b',)], include_coupling=True, boundary='open')
# open: an edge that merely touches the union is kept, even if its other end is outside
```

The distinction only bites when you asked for a crossing edge. With the default
`include_inter=False` and `include_coupling=False` the two boundaries agree,
because an intra-layer edge never leaves its layer.

!!! note "This was the behaviour and had no name"

    Asked for the union of `a` and `b` *with coupling edges*, the algebra used to
    return every coupling edge **touching** either — including one running from
    `b` out to `c`, which is not in the window at all. A selection that reaches
    outside the window it names is a leak, and an unnamed leak is one nobody can
    ask for or refuse. `boundary="open"` is that behaviour, named; `"closed"` is
    the new default.

`layer_edge_set` deliberately takes **no** `boundary=`. It asks about one layer,
where *touching* is the whole question; whether a selection may keep an edge that
leaves it is a question about the selection, not about a layer.

### One walk, not one per layer

The algebra used to call `layer_edge_set` once per layer, and each call
re-derived the whole edge table — so a window of *n* layers cost `n × |E|`.
Nothing about the answer needed that: one walk can drop each edge into every
layer it touches. Measured on 4,800 edges across 12 layers, `layer_union` is
**5.4× faster** than the per-layer loop it replaced, and the advantage grows with
the layer count because that is the factor being removed.

## Where to go next

- [Reading the graph](reading-the-graph.md) — the frame as the default answer.
- [Multilayer and multi-aspect graphs](math-multilayer.md) — the coordinate
  system these windows are written in.
