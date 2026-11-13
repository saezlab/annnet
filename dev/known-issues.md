# Known issues

Defects found while the documentation was revised in October 2026. Each one has
a short script that shows it. The scripts were run against `main` at `d19c195`.

The package now reads a placement the same way in one call and in a batch, so
the batch `add_edges` defect leaves this page.
`tests/test_add_nodes_placement.py` holds both forms of `add_nodes` and of
`add_edges` to that one rule.

## A hyperedge batch reads a bare id by its own rule

A bare node id names the node while it sits on one layer (contract, section
2.2). `add_edges` and the binary batch refuse an id that sits on two, and they
name the choices. The hyperedge batch drops the placeholder placement before
it counts. So an id on the placeholder and on one real layer resolves to the
real one.

```python
from annnet import AnnNet

anet = AnnNet(directed=True)
anet.layers.set_aspects(['condition'], {'condition': ['control', 'treated']})
anet.add_nodes(['A', 'B'])
anet.add_nodes(['A', 'B'], layer=('control',))
anet.N.at('A').layers
# (('_',), ('control',))

anet.add_edges([{'members': ['A', 'B'], 'edge_id': 'h'}])
anet.E.at('h').members
# frozenset({('A', ('control',)), ('B', ('control',))})

anet.add_edges('A', 'B', edge_id='e')
# ValueError: Ambiguous bare node_id 'A' in multilayer graph; ...
```

**Expected:** one rule for all three paths. Either the placeholder counts as a
layer, and the hyperedge batch raises as well. Or it does not, and the other
two paths resolve the way the hyperedge batch does. The code is
`_resolve_member_key` in `annnet/core/_mutate.py`.

## `flexible` changes the coefficients but not the direction

With a `flexible` policy, `add_edges` sets the signs of the incidence column
from the attribute values. The source and target of the edge, and the
neighbour queries, do not change.

```python
from annnet import AnnNet

anet = AnnNet(directed=True)
anet.add_nodes(['a', 'b'])
anet.attrs.update('nodes', {'a': {'x': 5.0}, 'b': {'x': 1.0}})
anet.add_edges(
    'a',
    'b',
    edge_id='f',
    flexible={'var': 'x', 'threshold': 0, 'scope': 'node', 'above': 't->s'},
)

anet.matrices.signed()  # column of 'f': a = -1, b = +1, so b is the source
anet.E.at('f').source  # frozenset({'a'})
anet.successors('a')  # ['b']
```

**Expected:** `E.at('f')` gives `b` as the source and `a` as the target, and
`successors('b')` contains `a`. The same applies when a later attribute update
changes the direction.

## `flexible` in a batch item is dropped

`add_edges` refuses `flexible=` as a keyword for a batch, with `TypeError`. A
`flexible` key inside a batch item is accepted and has no effect.

```python
anet = AnnNet(directed=True)
anet.add_nodes(['a', 'b'])
anet.add_edges(
    [{'source': 'a', 'target': 'b', 'edge_id': 'g', 'flexible': {'var': 'x', 'threshold': 0}}]
)

anet.edge_direction_policy  # {}
```

**Expected:** the policy is attached, or the item is refused with `TypeError`,
as the keyword is.

## A slice does not limit the backend graph

`to_nx(slice=...)`, `anet.nx.backend(slice=...)` and the `_nx_slice=` option
of accessor calls convert all edges of the graph. The `slice` argument only
changes which hyperedges are reified.

```python
anet = AnnNet(directed=True)
anet.add_edges('A', 'B', edge_id='e1')
anet.add_edges('B', 'C', edge_id='e2')
anet.add_edges('C', 'D', edge_id='e3')
anet.slices.add('s', edges=['e1'])

anet.nx.backend(slice='s').number_of_edges()  # 3
sorted(anet.nx.pagerank(anet, _nx_slice='s'))  # ['A', 'B', 'C', 'D']
```

**Expected:** 1 edge, and PageRank over `A` and `B` only. Until then, use
`anet.view(slices='s').materialize().nx`. The igraph accessor has the same
`slice` parameter and needs the same check.

## Docstrings that tell the history of the code

Some docstrings describe what the code used to do, or why it changed, instead
of what it does now. The documentation renders them. Examples:

- `Endpoint` (`annnet/core/_records.py`): "reading one meant asking which it
  was", with code that is no longer necessary.
- The module docstring of `annnet/io/edge_frame.py`: the loop that the module
  replaced, and its defects.
- `_coupling_edge_spec` in `annnet/core/_Layers.py`: "The id used to be ...".
- `annnet/core/_contextual.py` and `annnet/io/_shared/importing.py`: "used to
  be stored", "used to be".

**Expected:** a docstring says what the object does and how to call it. A
design reason, when one is necessary, goes in a comment or in
[`internals.md`](internals.md).

## Not reproduced

One round trip through SIF lost a hyperedge, on a graph built with
`from_edge_frame`. Round trips of graphs with undirected and directed
hyperedges, with and without `from_edge_frame`, all kept every edge. The
original graph is not known.
