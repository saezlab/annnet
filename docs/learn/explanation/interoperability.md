# Interoperability and adapters

annnet distinguishes two kinds of adapters:

- Runtime backend adapters (interoperability): power the lazy proxies `G.nx`, `G.ig`, `G.gt` to call algorithms and create in‑memory graphs for NetworkX, igraph, and graph‑tool.
- Format/data adapters (I/O): read/write files and data tables (GraphML, GEXF, SIF, SBML, CX2, Parquet GraphDir, JSON/NDJSON, Excel/CSV, DataFrames via Narwhals).

In short: proxies = nx/ig/gt at runtime; file formats = adapters under the hood.

## Using runtime proxies

```python
# NetworkX
bc = G.nx.betweenness_centrality(G)

# Get a concrete backend graph with options
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",  # or "expand"
    slice="toy",
    simple=True,             # collapse multiedges
)

# igraph (if installed)
pagerank = G.ig.pagerank(G)
```

Behavioral notes:
- Hyperedges: may be dropped or expanded depending on `hyperedge_mode` and backend.
- Slices: backends operate on a single view; multiple slices can be flattened unless specified.
- Attributes: proxies keep only needed attributes by default for performance.

## Manifests for round‑trip conversions

When converting to external graphs, annnet returns a manifest capturing details needed to reconstruct the original structure as faithfully as possible on import.

```python
import annnet as an

nxG, manifest = an.to_nx(G, directed=True, hyperedge_mode="skip")
# ... use nxG ...
G2 = an.from_nx(nxG, manifest)
```

Manifests are especially useful when:
- Hyperedges are reified or skipped.
- Multiple slices are flattened for a backend.
- Multigraphs are collapsed to simple graphs for algorithm convenience.
