# Quickstart

This quickstart shows how to create a graph, work with slices and annotations, add hyperedges, and run an algorithm through a backend proxy.

Prerequisite: install annnet via the [Installation guide](installation.md) (add extras like `networkx` if you want backend proxies).

For exact APIs used below, see [AnnNet](reference/core/graph.md), [Slices](reference/core/slices.md), [NetworkX adapter](reference/adapters/networkx.md), and [Native .annnet format](reference/io/annnet-format.md).

## Build a small graph

```python
import annnet as an

G = an.AnnNet(directed=True)
G.add_slice("toy")
G.slices.active = "toy"

for v in ["A", "B", "C", "D"]:
    G.add_vertex(v, label=v, kind="gene")

# Binary edges (directed + undirected)
G.add_edge("A", "B", weight=2.0, edge_directed=True, relation="activates")
G.add_edge("B", "C", weight=1.0, edge_directed=False, relation="binds")

# Hyperedge (directed head→tail)
G.add_hyperedge(head=["A", "B"], tail=["C", "D"], weight=1.0)
```

## Run an algorithm (NetworkX)

```python
# Requires networkx installed
deg = G.nx.degree_centrality(G)
```

You can fetch a concrete NetworkX graph with options:

```python
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",  # or "expand"
    slice="toy",
    simple=True,            # collapse multiedges
)
```

See also [Interoperability](explanations/interoperability.md) and the [NetworkX adapter reference](reference/adapters/networkx.md).

## Convert and save

```python
import annnet as an

# File formats
an.to_graphml(G, "graph.gml", directed=True, hyperedge_mode="reify")

# Lossless storage
an.annnet.write(G, "my_graph.annnet", overwrite=True)
R = an.annnet.read("my_graph.annnet")
```

See the [GraphML and GEXF reference](reference/io/graphml-gexf.md) and the [Native .annnet format reference](reference/io/annnet-format.md).

## Next step

- Continue with [Tutorials and use cases](tutorials/index.md).
- Read [Explanations](explanations/index.md) when you want the conceptual model behind the data structures.
- Use [Reference](reference/index.md) when you need exact APIs.
