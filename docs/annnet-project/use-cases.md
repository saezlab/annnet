# Use cases

This page highlights practical scenarios captured in the example notebooks, especially `notebooks/SBUC.ipynb`.

## 1) Analysis with backend proxies

- Run NetworkX algorithms directly: `G.nx.degree_centrality(G)`, `G.nx.louvain_communities(G)` (if community is available).
- Use igraph for metrics like `pagerank`, `betweenness`, or community detection via `G.ig` (if `python-igraph` installed).
- Construct a simplified backend graph with options:

```python
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",  # or "expand" to reify
    slices=["eval"],         # or a single slice="eval"
    simple=True,             # collapse multiedges (sum weights)
)
```

The SBUC notebook shows patterns like degree/betweenness centrality, shortest path, PageRank, Louvain, etc., across `G.nx` and `G.ig`.

## 2) Multilayer modeling and visualization

- Define aspects (e.g., time, modality) and construct intra/inter/coupling edges per layer combination.
- Export to Cytoscape via CX2 for visualization:

```python
from annnet.adapters.cx2_adapter import to_cx2
cx2 = to_cx2(G, hyperedges="reify")
# write json to file for Cytoscape import
```

## 3) SBML integration and stoichiometry

- Import biochemical models from SBML (libSBML/COBRA) and represent reactions as directed hyperedges with stoichiometric coefficients embedded in the incidence matrix.
- Convert to backend for algorithmic analysis or to CX2/GraphML for visualization/archival.

## 4) Interoperability and round‑trips

- Convert to NetworkX/igraph with a manifest; edit externally; round‑trip back:

```python
import annnet as an
nxG, manifest = an.to_nx(G, directed=True, hyperedge_mode="skip")
# ... external processing ...
G2 = an.from_nx(nxG, manifest)
```

## 5) Slices for scenarios/contexts

- Use slices to represent scenarios (train/eval), conditions, or experimental contexts.
- Override edge weights per slice and toggle the active slice during analysis.

## Notebook

- Open `notebooks/SBUC.ipynb` to explore these patterns interactively.
- The documentation site can render notebooks placed under `docs/` (via mkdocs‑jupyter). If you want this notebook live on the site, I can copy or adapt it under `docs/`.
