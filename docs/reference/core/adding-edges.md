# Adding edges

`anet.add_edges` adds one edge, or a batch of edges. This page lists the input
forms and options it accepts. To make a graph from a table of edges, use
[`from_edge_frame`](../io/dataframes.md). The
[graph model guide](../../guide/graph-model.ipynb) explains edges, hyperedges
and the incidence matrix.

A call with one edge returns the edge ID as a string. A batch returns a list of
IDs. A node that does not exist yet is created.

## One edge

### Binary edge

```python
anet.add_edges('A', 'B')
anet.add_edges('A', 'B', weight=0.5, edge_id='e1', directed=True)
```

### Edge between layers

```python
anet.add_edges(('A', ('healthy',)), ('B', ('treated',)))
```

Each endpoint is a `(node_id, layer)` pair. On a graph with aspects, always
give the pairs. annnet sets the multilayer kind of the edge:

| endpoints | kind |
|---|---|
| same node ID, different layers | `coupling` |
| same layer, different node IDs | `intra` |
| different node IDs and layers | `inter` |

### Undirected hyperedge

```python
anet.add_edges(['A', 'B', 'C'])
```

### Directed hyperedge

```python
anet.add_edges(['A', 'B'], ['C', 'D'])  # sources, then targets
```

### Edge with coefficients

```python
anet.add_edges({'glucose': -1.0, 'atp': -1.0}, {'glucose-6p': 1.0, 'adp': 1.0})
```

The first dictionary is the source side, the second the target side. annnet
writes the coefficients into the incidence matrix as they are. Only a call with
one edge accepts this form.

### Edge entity without endpoints

```python
anet.add_edges(None, None, as_entity=True, edge_id='virtual_e1', role='enzyme')
```

This makes an edge that has no endpoints yet. Another edge can use it as an
endpoint. `edge_id` is necessary.

## A batch of edges

Give one list or generator as the only positional argument.

### Binary edges

```python
# 2-tuples
anet.add_edges([('A', 'B'), ('C', 'D')])

# 3-tuples: the third item is the weight
anet.add_edges([('A', 'B', 0.5), ('C', 'D', 2.0)])

# dictionaries, with the keys source/target or src/tgt
anet.add_edges(
    [
        {'source': 'A', 'target': 'B', 'weight': 0.5, 'edge_id': 'e1'},
        {'src': 'C', 'tgt': 'D'},
    ]
)
```

### Hyperedges

A list in `src` without `tgt` is an undirected hyperedge. A list in `src` and
a list in `tgt` is a directed hyperedge. `source` and `target` are the same as
`src` and `tgt`.

```python
anet.add_edges(
    [
        {'src': ['A', 'B', 'C']},  # undirected
        {'src': ['B', 'D'], 'edge_id': 'h2'},  # undirected
        {'src': ['A', 'B'], 'tgt': ['C']},  # directed
    ]
)
```

A batch can mix binary edges and hyperedges.

### Edge entities without endpoints

Give dictionaries without `source` and `target`, and set `as_entity=True`.
`edge_id` is optional.

```python
anet.add_edges(
    [
        {'edge_id': 'EE1', 'role': 'enzyme', 'pathway': 'glycolysis'},
        {'edge_id': 'EE2', 'role': 'enzyme', 'pathway': 'tca'},
    ],
    as_entity=True,
    slice='Healthy',
)
```

In a batch item, the keys other than `edge_id`, `slice`, `weight`,
`edge_directed`, `edge_type` and `propagate` are edge attributes.

## Options

### One edge

| option | default | description |
|---|---|---|
| `weight` | `1.0` | The edge weight. In the incidence matrix: `+weight` on the source side and `-weight` on the target side of a directed edge, `+weight` on all members of an undirected edge. Not used for an edge with coefficients. |
| `edge_id` | automatic | The edge ID. If the ID exists, annnet updates that edge. |
| `directed` | `None` | The direction of this edge. `None` uses the default of the graph, and then `True`. |
| `parallel` | `'update'` | What to do when `edge_id` is `None` and the endpoints already have an edge. See below. |
| `slice` | active slice | The slice that gets the edge. annnet creates the slice if it does not exist. |
| `as_entity` | `False` | Make the edge an entity that other edges can use as an endpoint. |
| `propagate` | `'none'` | Which other slices get the edge. See below. |
| `flexible` | `None` | Set the direction from an attribute value. See below. |
| other keywords | | Edge attributes. |

### Batch

| option | default | description |
|---|---|---|
| `default_weight` | `1.0` | The weight of an item that does not give one. |
| `default_edge_directed` | `None` | The direction of an item that does not give one. |
| `default_propagate` | `'none'` | The `propagate` value of an item that does not give one. |
| `default_slice_weight` | `None` | The slice weight of an item that does not give one. |
| `slice` | active slice | The slice of an item that does not give one. |
| `layer` | `None` | The layer of the items. |
| `as_entity` | `False` | Make all edges entities. |

### `parallel`

Only for one edge, and only when `edge_id` is `None`.

| value | result |
|---|---|
| `'update'` (default) | If the endpoints already have an edge, annnet updates that edge and returns its ID. |
| `'parallel'` | annnet adds a new edge. |
| `'error'` | If the endpoints already have an edge, annnet raises `ValueError`. |

With an `edge_id` that exists, annnet updates that edge. With a new `edge_id`,
annnet adds a new edge. If `parallel='error'` and the endpoints already have a
different edge, annnet raises `ValueError`.

### `propagate`

| value | result |
|---|---|
| `'none'` (default) | Only the slice in `slice` gets the edge. |
| `'shared'` | Each slice that holds **both** endpoints also gets the edge. |
| `'all'` | Each slice that holds **one or both** endpoints also gets the edge. |

### `flexible`

The value of an edge attribute sets the direction of the edge.

```python
anet.add_edges(
    'A',
    'B',
    score=-0.4,
    flexible={
        'var': 'score',  # the attribute to read
        'threshold': 0.0,  # compare the value to this number
        'scope': 'edge',  # read it from the edge
        'above': 's->t',  # direction when the value is above the threshold
        'tie': 'undirected',  # when the value is equal to the threshold
    },
)
```

`above` is `'s->t'` (from source to target) or `'t->s'`. Below the threshold,
the direction is the opposite. `tie` is `'keep'`, `'undirected'`, `'s->t'` or
`'t->s'`. With `'scope': 'node'`, annnet compares the attribute of the source
node to that of the target node.

## Limits

| limit | detail |
|---|---|
| Coefficients in a batch | Only a call with one edge takes the form with coefficients. |
| `parallel` in a batch | A batch adds each item as a new edge, also when the endpoints already have an edge. A batch does not accept `parallel=`. |
| `flexible` in a batch | A batch does not accept `flexible=`. |
| Items without endpoints | In a batch, they need `as_entity=True`, else annnet raises `ValueError`. |
| One edge without endpoints | It needs `edge_id`, else annnet raises `ValueError`. In a batch, annnet makes the ID. |

## See also

- [Graph model guide](../../guide/graph-model.ipynb): edges, hyperedges and
  the incidence matrix
- [Layers guide](../../guide/layers.ipynb): edges between layers
- [Slices guide](../../guide/slices.ipynb): slices and their edges
- [`AnnNet.add_edges`][annnet.core.graph.AnnNet.add_edges]
