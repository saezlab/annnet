# annnet — Annotated Network Data Structures for Science

annnet (Annotated Network) is a unified, high‑expressivity graph platform that brings anndata‑style, annotated containers to networks, multilayer structures, and hypergraphs.

Use it for systems/network biology, omics integration, and any domain needing flexible graph semantics with stable, lossless storage and excellent interoperability.

## Why annnet?
- Maximum graph expressiveness: hyperedges; directed/undirected per‑edge; parallel edges; self‑loops; vertex–edge and edge–edge relations; graph/edge‑level semantics.
- Multilayer networks: layers, aspects, inter‑layer edges.
- Powerful slicing: create subnetworks, clusters, arbitrary “slices”; switch active slice.
- Annotated tables: Polars‑backed tables for vertices, edges, layers, slices, metadata.
- Interoperability: NetworkX/igraph/graph‑tool via lazy proxies; file IO for GraphML, GEXF, SIF, SBML, CX2, Parquet GraphDir, JSON/Excel/DataFrames.
- Disk‑backed, lossless `.annnet` storage using Zarr + Parquet + JSON.

## Quickstart

```python
import annnet as an

G = an.Graph(directed=True)
G.add_slice("toy")
G.slices.active = "toy"
for v in ["A", "B", "C", "D"]:
    G.add_vertex(v, label=v, kind="gene")

G.add_edge("A", "B", weight=2.0, edge_directed=True, relation="activates")
G.add_edge("B", "C", weight=1.0, edge_directed=False, relation="binds")
G.add_hyperedge(head=["A", "B"], tail=["C", "D"], weight=1.0)

# NetworkX algorithm (if installed)
deg = G.nx.degree_centrality(G)
```

See Get Started → Quickstart for a walkthrough.

## Install

```bash
pip install annnet

# Optional extras
pip install "annnet[networkx,igraph]"   # backends
pip install "annnet[io]"                # JSON/Parquet/Zarr, Excel, Narwhals
pip install "annnet[all]"               # common extras (graph‑tool not on PyPI)
```

Graph‑tool is supported if installed from your OS/package manager.

## Learn more
- Project: goals and scope
- Design philosophy: principles and trade‑offs
- Explanations: zero‑loss storage and interoperability
- Reference: API docs

This is the main documentation page.
