# Quickstart

This quickstart shows how to create a graph, work with slices and annotations, add hyperedges, and run an algorithm through a backend proxy.

## Install

```bash
pip install annnet
# optional backends and IO
pip install "annnet[networkx,igraph]" "annnet[io]"
```

## Build a small graph

```python
import annnet as an

G = an.Graph(directed=True)
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
    simple=True,             # collapse multiedges
)
```

## Convert and save

```python
import annnet as an

# File formats
an.to_graphml(G, "graph.graphml", directed=True, hyperedge_mode="reify")

# Lossless storage
an.annnet.write(G, "my_graph.annnet", overwrite=True)
R = an.annnet.read("my_graph.annnet")
```

Next steps:
- See Explanations → Interoperability for round‑trip manifests.
- See Explanations → annnet zero‑loss serialization for storage details.
