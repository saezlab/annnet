# Backend accessors

An `AnnNet` object gives access to the algorithms of three graph libraries:

| accessor | library | what `anet.<accessor>.<name>` calls |
|---|---|---|
| `anet.nx` | [NetworkX](https://networkx.org/) | the function `networkx.<name>`, for example from the [NetworkX algorithms](https://networkx.org/documentation/stable/reference/algorithms/index.html) |
| `anet.ig` | [igraph](https://python.igraph.org/) | the method `igraph.Graph.<name>` of the converted graph (see the [`Graph` API](https://python.igraph.org/en/stable/api/igraph.Graph.html) and [graph analysis](https://python.igraph.org/en/stable/analysis.html)), or else the function `igraph.<name>` |
| `anet.gt` | [graph-tool](https://graph-tool.skewed.de/) | the function `graph_tool.<namespace>.<name>`, called as `anet.gt.<namespace>.<name>` |

The accessor converts `anet` to a graph of the library, calls the function,
and gives back the result of the library. The conversion is cached.

```python
# networkx.betweenness_centrality(...)
scores = anet.nx.betweenness_centrality(anet)

# igraph.Graph.pagerank() on the converted graph
pagerank = anet.ig.pagerank()

# graph_tool.centrality.pagerank(...)
pagerank = anet.gt.centrality.pagerank(anet)
```

Pass `anet` where the library function expects its graph. The accessor puts the
converted graph in its place. Where the library takes a node, you can give the
annnet node ID.

The graph-tool namespaces are those of the
[graph-tool documentation](https://graph-tool.skewed.de/static/docs/stable/):
[`topology`](https://graph-tool.skewed.de/static/docs/stable/topology.html),
[`centrality`](https://graph-tool.skewed.de/static/docs/stable/centrality.html),
[`clustering`](https://graph-tool.skewed.de/static/docs/stable/clustering.html),
[`flow`](https://graph-tool.skewed.de/static/docs/stable/flow.html),
[`inference`](https://graph-tool.skewed.de/static/docs/stable/inference.html),
[`generation`](https://graph-tool.skewed.de/static/docs/stable/generation.html),
[`search`](https://graph-tool.skewed.de/static/docs/stable/search_module.html)
and [`util`](https://graph-tool.skewed.de/static/docs/stable/util.html). A name
that only one namespace has also works without the namespace:
`anet.gt.pagerank(anet)`.

## Accessors and adapters

The adapter functions in `annnet.adapters` also convert, but they give you the
graph of the other library:

- `annnet.adapters.to_nx(...)` and `annnet.adapters.from_nx(...)`
- `annnet.adapters.to_igraph(...)` and `annnet.adapters.from_igraph(...)`
- `annnet.adapters.to_graphtool(...)` and `annnet.adapters.from_graphtool(...)`

Use an adapter when you want to work with the other graph object yourself. Use
`anet.nx`, `anet.ig` or `anet.gt` when `anet` stays your data and you only
want the result of an algorithm.

!!! warning "Clear the cache after a change"

    Not every change of `anet` clears the cached conversion. After you edit the
    graph, call `clear()` on the accessor before you run another algorithm, or
    convert again with an adapter. The
    [computation guide](../../guide/computation-and-storage.ipynb) shows both
    forms.

## Methods of every accessor

- `backend(...)`: the converted graph of the library.
- `clear()`: remove the cached conversions of this accessor.

`anet.nx` and `anet.ig` also have:

- `peek_nodes(k=10)`: a few node identifiers of the converted graph.

## Conversion options

`anet.nx.backend(...)` and `anet.ig.backend(...)` take these options:

- `directed`
- `hyperedge_mode`
- `slice`
- `slices`
- `needed_attrs`
- `simple`
- `edge_aggs`

An algorithm call takes the same options with a prefix, so that they do not
mix with the arguments of the library function: `_nx_directed`,
`_nx_hyperedge`, `_nx_slice`, `_nx_slices`, `_nx_label_field`, `_nx_simple`
and `_nx_edge_aggs` for NetworkX, and the same names with `_ig_` for igraph.

```python
# Betweenness on the graph read as undirected
scores = anet.nx.betweenness_centrality(anet, _nx_directed=False)
```

`anet.gt.backend()` takes no options and gives the converted graph-tool graph.
