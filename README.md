# annnet — Annotated Network Data Structure for Science
[annnet](https://saezlab.github.io/annnet/) (Annotated Network) is a unified, high‑expressivity graph platform that brings anndata‑style, annotated containers to networks, multilayer structures, and hypergraphs. It targets systems biology, network biology, omics integration, computational social science, and any domain needing fully flexible graph semantics with modern, stable storage and interoperability.

---

## Why annnet?
Nothing else combines all of this at once:
- Maximum graph expressiveness: hyperedges; directed/undirected per‑edge; parallel edges; self‑loops; vertex–edge and edge–edge relations; graph/edge‑level semantics.
- Multilayer networks: support aligned with the Kivelä et al. framework (layers, aspects, inter‑layer edges).
- Powerful slicing: create subnetworks, clusters, and arbitrary “slices”; switch active slice quickly.
- Annotated tables everywhere: Narwhals-compatible tables for vertices, edges, layers, vertex-layer couples, slices, and graph metadata.
- anndata‑style feel: familiar patterns like obs/var/layers‑like concepts, caches, and indices.
- Interoperability without friction: import/export with NetworkX, igraph, GraphML, GEXF, SIF, SBML, CX2 (Cytoscape), Excel/CSV/JSON, Parquet graph directories, and DataFrames (via Narwhals).
- Algorithm interoperability: seamless, lazy calls into NetworkX/igraph/graph‑tool via the graph-owned `G.nx`, `G.ig`, and `G.gt` accessors.
- Disk‑backed, lossless `.annnet` storage:
  - Zarr for sparse matrices
  - Parquet for annotated tables
  - JSON for metadata
  Reopen without loss of structure or metadata.

annnet is under active development. Feedback is welcome via GitHub issues: https://github.com/saezlab/annnet

---

## Features

### General graph modeling
- Simple graphs, directed graphs, multigraphs
- Hyperedges (undirected and directed head→tail)
- Signed/weighted edges and rich node/edge annotations
- Efficient indexing and fast lookups

### Import/Export (file formats/data)
- GraphML, GEXF, SIF, SBML
- CX2 (Cytoscape)
- Parquet graph directory (directory of Parquet files)
- CSV/TSV, JSON/NDJSON
- Excel (via pandas/openpyxl)
- DataFrames via Narwhals (Polars, pandas, and friends)

### Interoperability (runtime backends)
- NetworkX via `G.nx` and `to_nx`/`from_nx`
- igraph via `G.ig` and `to_igraph`/`from_igraph`
- graph‑tool via `G.gt` and `to_graphtool` (if installed)

### Export (file formats/data)
- GraphML, GEXF, SIF
- CX2 (Cytoscape)
- Parquet graph directory (directory of Parquet files)
- JSON/NDJSON
- DataFrames via Narwhals (Polars, pandas, and friends)

### Backend-specific integration
If `networkx` is installed, you can call algorithms directly from the AnnNet graph:

```python
centrality = G.nx.degree_centrality(G)
```

Similarly, `G.ig` (igraph) and `G.gt` (graph-tool) are available if those libraries are present. On `G.nx.<function>(G, ...)`, AnnNet resolves the NetworkX callable, converts the AnnNet graph to a NetworkX graph with the requested projection options, replaces the `G` argument with that backend graph, dispatches the call, and returns the backend result. Backend graphs are cached and refreshed after AnnNet mutations.

Use `G.nx.backend(...)`, `G.ig.backend(...)`, or `G.gt.backend()` when you want the concrete projected backend graph object. Use `G.nx.<function>(G, ...)`, `G.ig.<function>(G, ...)`, or `G.gt.<namespace>.<function>(G, ...)` when you want AnnNet to do the conversion and dispatch for an algorithm call. You can control directionality, hyperedge handling, slice selection, and multigraph collapsing where the backend supports those options.

Examples:

```python
# Keep multiedges, drop hyperedges (default); operate on active slice
bc = G.nx.betweenness_centrality(G)

# Convert a specific slice and collapse multiedges using weights
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",  # or "expand"
    slice="toy",
    simple=True,             # collapse Multi(Di)Graph -> (Di)Graph
)

# igraph interoperability (if python-igraph installed)
pagerank = G.ig.pagerank(G)
```

---

## Installation

Base install:

```bash
pip install annnet
```

Backends and IO/storage extras (choose what you need):

```bash
# Backend libraries
pip install "annnet[networkx,igraph]"

# IO/serialization and storage (JSON/Parquet/Zarr, Excel, Narwhals)
pip install "annnet[io]"

# Or more granular extras
pip install "annnet[parquet]"   # Parquet graph directory support (pyarrow)
pip install "annnet[zarr_io]"   # Zarr + numcodecs for .annnet storage
pip install "annnet[excel]"     # Excel loader (pandas/openpyxl)
pip install "annnet[sbml]"      # SBML import (libxml2/lxml)

# Everything commonly available via pip (graph‑tool is not on PyPI)
pip install "annnet[all]"
```

Note: graph-tool is supported by adapters and `G.gt` interoperability if installed, but it is not available on PyPI; install it via your OS/package manager and annnet will detect it.

### Dev/test setup with Pixi 

Use Pixi to get a fully loaded dev environment (Python 3.12, all extras, graph‑tool from conda‑forge):

```bash
pixi install          # solves the dev env
pixi run test-all     # runs pytest in the Pixi dev env (with graph-tool)
```

Notes:
- `graph-tool` is only available on some platforms (e.g., macOS arm may need Rosetta, linux-64 is supported); the Pixi manifest includes multiple platforms.
- The Pixi env installs annnet editable with extras `all` and `dev`, plus graph-tool from conda-forge.

### Docs with uv

Use the module entrypoint instead of the `mkdocs` console script. In this repo, that path picks up the startup customizations used to avoid Jupyter warning noise during docs builds.

```bash
uv sync --group docs
uv run python -m mkdocs serve
```

For a one-off build:

```bash
uv run python -m mkdocs build --strict
```

---

## Quick Start

```python
import annnet as an

G = an.Graph(directed=True)  # default direction; can be overridden per-edge

# Create slices and set active
G.add_slice("toy")
G.add_slice("train")
G.add_slice("eval")
G.slices.active = "toy"   # same effect as set_active_slice("toy")

# Add vertices (with attributes) in 'toy'
for v in ["A", "B", "C", "D"]:
    G.add_vertex(v, label=v, kind="gene")

# 1) Binary directed
e_dir = G.add_edge("A", "B", weight=2.0, directed=True, relation="activates")

# 2) Binary undirected
e_undir = G.add_edge("B", "C", weight=1.0, directed=False, relation="binds")

# 3) Self-loop
e_loop = G.add_edge("D", "D", weight=0.5, directed=True, relation="self")

# 4) Parallel edge
e_parallel = G.add_edge("A", "B", weight=5.0, parallel="parallel", relation="alternative")

# 5) Vertex–edge (hybrid) edge
G.add_edge(edge_id="edge_e1", as_entity=True, description="signal")
e_vx = G.add_edge("edge_e1", "C", directed=True, as_entity=True, channel="edge->vertex")

# 6) Hyperedge (undirected, 3-way membership)
e_hyper_undir = G.add_edge(["A", "C", "D"], weight=1.0, directed=False, tag="complex")

# 7) Hyperedge (directed head→tail)
e_hyper_dir = G.add_edge(["A", "B"], ["C", "D"], weight=1.0, directed=True, reaction="A+B->C+D")

# 8) Run a NetworkX algorithm (if installed)
deg = G.nx.degree_centrality(G)
```

---

## Interoperability

High‑fidelity conversions aim to preserve IDs, attributes, and directionality. When a conversion is lossy (e.g., hyperedges to NetworkX), functions return a manifest you can pass back to restore structure where possible.

Adapter types (what goes where):
- Runtime backend adapters (interoperability): used by `G.nx`, `G.ig`, and `G.gt` for algorithm calls and by `to_nx`/`from_nx`, `to_igraph`/`from_igraph`, and `to_graphtool`/`from_graphtool` for explicit in-memory conversions.
- Format/data adapters (I/O): used to read/write files and tabular data. Examples: GraphML, GEXF, SIF, SBML, CX2 (Cytoscape), Parquet graph directories, JSON/NDJSON, Excel/CSV, and DataFrames via Narwhals.

In short: `G.nx`/`G.ig`/`G.gt` are runtime algorithm interoperability accessors; file formats use IO adapters.

NetworkX:

```python
import annnet as an

nxG, manifest = an.adapters.to_nx(G, directed=True, hyperedge_mode="skip")

# ... run algorithms or edit nxG ...

G2 = an.adapters.from_nx(nxG, manifest)  # use manifest to reduce loss
```

igraph / graph‑tool (if installed):

```python
igG, ig_manifest = an.adapters.to_igraph(G, directed=True, hyperedge_mode="skip")
G2 = an.adapters.from_igraph(igG, ig_manifest)
```

File formats and dataframes:

```python
import annnet as an

# GraphML / GEXF / SIF
an.io.to_graphml(G, "graph.graphml", directed=True, hyperedge_mode="reify")
G2 = an.io.from_graphml("graph.graphml")

an.io.to_sif(G, "graph.sif", lossless=True)
G3 = an.io.from_sif("graph.sif", manifest_path="graph.sif.manifest.json")

# JSON / NDJSON
an.io.to_json(G, "graph.json")
H = an.io.from_json("graph.json")

# Parquet graph directory
an.io.to_parquet(G, "graph_dir/")
K = an.io.from_parquet("graph_dir/")

# CX2 (Cytoscape)
cx2 = an.io.to_cx2(G, hyperedges="reify")
L = an.io.from_cx2(cx2, hyperedges="manifest")
```

Notes:
- Hyperedges may be dropped or reified depending on `hyperedge_mode` and target format.
- Slices: backend graphs are single‑view; multiple slices may be flattened unless specified.
- Use returned manifests when available to improve round‑trip fidelity.

---

## Storage (.annnet)

annnet can write and read a lossless on‑disk format combining Zarr (sparse arrays) and Parquet (tables), plus JSON sidecars.

```python
import annnet as an

an.io.write(G, "my_graph.annnet", overwrite=True)  # Zarr + Parquet + JSON
R = an.io.read("my_graph.annnet")                   # full fidelity load
```

Layout highlights:
- `manifest.json` with versions, counts, and slice metadata
- `structure/incidence.zarr` stores COO arrays (row/col/data)
- `structure/*parquet` for indices, definitions, weights, kinds
- `tables/*parquet` for vertex/edge/slice attributes
- `layers/`, `slices/`, `audit/`, `uns/` for multilayer, slicing, history, and unstructured data

---

## Internal Design

annnet adapts internal representation for performance and compatibility:
- Sparse incidence matrix for core topology; DOK in‑memory, stored as COO
- Attributes decoupled from structure (Polars tables)
- Lazy conversion to external backends on demand
- Copy‑on‑write graph views and structured change tracking
- Optional caching for converted backend graphs within `G.nx`/`G.ig`/`G.gt`

See the architecture overview in `architecture.md` for deeper details and examples.

---

## Philosophy

- Simple, consistent interface for all graph types
- Interoperability‑first: integrate, don’t replace
- Performance‑aware, not performance‑obsessed
- Extensible and modular, not monolithic

---

## Package Overview

The package is modular to separate core functionality, adapters for external libraries, and IO/storage:

```
annnet/
├── core/         # Graph class, managers, lazy interoperability accessors (nx/ig/gt)
├── adapters/     # Backend adapters: networkx/igraph/graph-tool conversions
├── io/           # Lossless .annnet storage and filesystem/tabular IO
├── algorithms/   # Pure-Python algorithms using core only
└── utils/        # Misc utilities (typing/validation/config)
```

See the [architecture overview](architecture.md) for a deeper design document.

---

## License
annnet is licensed under the BSD‑3 License. See the [LICENSE](LICENSE) file for details.
