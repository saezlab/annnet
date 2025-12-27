# annnet — Annotated Network Data Structures for Science
[annnet](https://saezlab.github.io/annnet/) (Annotated Network) is a unified, high‑expressivity graph platform that brings anndata‑style, annotated containers to networks, multilayer structures, and hypergraphs. It targets systems biology, network biology, omics integration, computational social science, and any domain needing fully flexible graph semantics with modern, stable storage and interoperability.

---

## Why annnet?
Nothing else combines all of this at once:
- Maximum graph expressiveness: hyperedges; directed/undirected per‑edge; parallel edges; self‑loops; vertex–edge and edge–edge relations; graph/edge‑level semantics.
- Multilayer networks: support aligned with the Kivelä et al. framework (layers, aspects, inter‑layer edges).
- Powerful slicing: create subnetworks, clusters, and arbitrary “slices”; switch active slice quickly.
- Annotated tables everywhere: Polars‑backed tables for vertices, edges, layers, vertex‑layer couples, slices, and graph metadata.
- anndata‑style feel: familiar patterns like obs/var/layers‑like concepts, caches, and indices.
- Interoperability without friction: import/export with NetworkX, igraph, GraphML, GEXF, SIF, SBML, CX2 (Cytoscape), Excel/CSV/JSON, Parquet graph directories, and DataFrames (via Narwhals).
- Algorithm proxies: seamless, lazy calls into NetworkX/igraph/graph‑tool via `G.nx`, `G.ig`, and `G.gt` proxies.
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
- Parquet GraphDir (directory of Parquet files)
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
- Parquet GraphDir (directory of Parquet files)
- JSON/NDJSON
- DataFrames via Narwhals (Polars, pandas, and friends)

### Backend‑specific integration (lazy proxies)
If `networkx` is installed, you can call algorithms directly via a proxy:

```python
centrality = G.nx.degree_centrality(G)
```

Similarly, `G.ig` (igraph) and `G.gt` (graph‑tool) proxies are available if those libraries are present. Proxies convert the active slice (or requested slices) to the backend on demand for algorithm calls. You can control directionality, hyperedge handling, and multigraph collapsing when converting.

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

# igraph proxy (if python-igraph installed)
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
pip install "annnet[parquet]"   # Parquet GraphDir (pyarrow)
pip install "annnet[zarr_io]"   # Zarr + numcodecs for .annnet storage
pip install "annnet[excel]"     # Excel loader (pandas/openpyxl)
pip install "annnet[sbml]"      # SBML import (libxml2/lxml)

# Everything commonly available via pip (graph‑tool is not on PyPI)
pip install "annnet[all]"
```

Note: graph‑tool is supported by adapters/proxies if installed, but it is not available on PyPI; install it via your OS/package manager and annnet will detect it.

### Dev/test setup with Pixi 

Use Pixi to get a fully loaded dev environment (Python 3.12, all extras, graph‑tool from conda‑forge):

```bash
pixi install          # solves the dev env
pixi run test-all     # runs pytest in the Pixi dev env (with graph-tool)
```

Notes:
- `graph-tool` is only available on some platforms (e.g., macOS arm may need Rosetta, linux-64 is supported); the Pixi manifest includes multiple platforms.
- The Pixi env installs annnet editable with extras `all`, `tests`, and `dev`, plus graph-tool from conda-forge.

---

## Quick Start

```python
import annnet as an

G = an.Graph(directed=True)  # default direction; can be overridden per-edge

# Add vertices with attributes
G.add_vertices([
    ("A", {"name": "a"}),
    ("B", {"name": "b"}),
    ("C", {"name": "c"}),
    ("D", {"name": "d"}),
])

# Create slices and set active
G.add_slice("toy")
G.add_slice("train")
G.add_slice("eval")
G.slices.active = "toy"   # same effect as set_active_slice("toy")

# Add vertices (with attributes) in 'toy'
for v in ["A", "B", "C", "D"]:
    G.add_vertex(v, label=v, kind="gene")

# 1) Binary directed
e_dir = G.add_edge("A", "B", weight=2.0, edge_directed=True, relation="activates")

# 2) Binary undirected
e_undir = G.add_edge("B", "C", weight=1.0, edge_directed=False, relation="binds")

# 3) Self-loop
e_loop = G.add_edge("D", "D", weight=0.5, edge_directed=True, relation="self")

# 4) Parallel edge
e_parallel = G.add_parallel_edge("A", "B", weight=5.0, relation="alternative")

# 5) Vertex–edge (hybrid) edge
G.add_edge_entity("edge_e1", description="signal")
e_vx = G.add_edge("edge_e1", "C", edge_type="vertex_edge", edge_directed=True, channel="edge->vertex")

# 6) Hyperedge (undirected, 3-way membership)
e_hyper_undir = G.add_hyperedge(members=["A", "C", "D"], weight=1.0, tag="complex")

# 7) Hyperedge (directed head→tail)
e_hyper_dir = G.add_hyperedge(head=["A", "B"], tail=["C", "D"], weight=1.0, reaction="A+B->C+D")

# 8) Run a NetworkX algorithm (if installed)
deg = G.nx.degree_centrality(G)
```

---

## Interoperability

High‑fidelity conversions aim to preserve IDs, attributes, and directionality. When a conversion is lossy (e.g., hyperedges to NetworkX), functions return a manifest you can pass back to restore structure where possible.

Adapter types (what goes where):
- Runtime backend adapters (interoperability): used by lazy proxies `G.nx`, `G.ig`, `G.gt` only. These power algorithm calls and in‑memory conversions to NetworkX, igraph, and graph‑tool.
- Format/data adapters (I/O): used to read/write files and tabular data. Examples: GraphML, GEXF, SIF, SBML, CX2 (Cytoscape), Parquet GraphDir, JSON/NDJSON, Excel/CSV, and DataFrames via Narwhals.

In short: proxies = nx/ig/gt at runtime; file formats = adapters under the hood.

NetworkX:

```python
import annnet as an

nxG, manifest = an.to_nx(G, directed=True, hyperedge_mode="skip")

# ... run algorithms or edit nxG ...

G2 = an.from_nx(nxG, manifest)  # use manifest to reduce loss
```

igraph / graph‑tool (if installed):

```python
from annnet.adapters.igraph_adapter import to_igraph, from_igraph
igG, ig_manifest = to_igraph(G, directed=True, hyperedge_mode="skip")
G2 = from_igraph(igG, ig_manifest)
```

File formats and dataframes:

```python
import annnet as an

# GraphML / GEXF / SIF
an.to_graphml(G, "graph.graphml", directed=True, hyperedge_mode="reify")
G2 = an.from_graphml("graph.graphml")

an.to_sif(G, "graph.sif", lossless=True)
G3 = an.from_sif("graph.sif", manifest_path="graph.sif.manifest.json")

# JSON / NDJSON
an.to_json(G, "graph.json")
H = an.from_json("graph.json")

# Parquet GraphDir
an.write_parquet_graphdir(G, "graph_dir/")
K = an.read_parquet_graphdir("graph_dir/")

# CX2 (Cytoscape)
from annnet.adapters.cx2_adapter import to_cx2, from_cx2
cx2 = to_cx2(G, hyperedges="reify")
L = from_cx2(cx2, hyperedges="manifest")
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

an.annnet.write(G, "my_graph.annnet", overwrite=True)  # Zarr + Parquet + JSON
R = an.annnet.read("my_graph.annnet")                   # full fidelity load
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
- Optional caching for converted backend graphs within proxies

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
├── core/         # Graph class, managers, lazy proxies (nx/ig/gt)
├── adapters/     # Backend adapters: networkx/igraph/graph-tool (runtime proxies)
│                 # Format adapters: GraphML, GEXF, SIF, SBML, CX2, Parquet GraphDir, JSON, DataFrames
├── io/           # Lossless .annnet storage (read/write) + Excel/CSV helpers
├── algorithms/   # Pure-Python algorithms using core only
└── utils/        # Misc utilities (typing/validation/config)
```

See the [architecture overview](architecture.md) for a deeper design document.

---

## License
annnet is licensed under the BSD‑3 License. See the [LICENSE](LICENSE) file for details.
