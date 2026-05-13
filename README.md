# annnet - Annotated Network Data Structure for Science

[![PyPI](https://img.shields.io/pypi/v/annnet?logo=pypi&logoColor=white)](https://pypi.org/project/annnet/)
[![Python](https://img.shields.io/pypi/pyversions/annnet?logo=python)](https://pypi.org/project/annnet/)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/saezlab/annnet/ci-testing-unit.yml?branch=main&label=tests)](https://github.com/saezlab/annnet/actions/workflows/ci-testing-unit.yml)
[![codecov](https://codecov.io/gh/saezlab/annnet/branch/main/graph/badge.svg)](https://codecov.io/gh/saezlab/annnet)
[![Docs](https://img.shields.io/badge/docs_built_with-MkDocs-blue)](https://saezlab.github.io/annnet/)
[![License](https://img.shields.io/github/license/saezlab/annnet)](https://github.com/saezlab/annnet/blob/main/LICENSE)

annnet (Annotated Network) is a unified, high‑expressivity graph platform that brings anndata‑style, annotated containers to networks, multilayer structures, and hypergraphs. It targets systems biology, network biology, omics integration, computational social science, and any domain needing fully flexible graph semantics with modern, stable storage and interoperability.


> 🚧 **annnet is under active development.**
> 
> Feedback and bug reports are very welcome via
> [GitHub issues](https://github.com/saezlab/annnet/issues).


## Why annnet?

![AnnNet](docs/assets/annnet_fig1_layout.png)

annnet aims to combine graph expressiveness, annotation-centric data handling, and practical interoperability in one model:

- Rich graph semantics: directed and undirected edges, parallel edges, self-loops, hyperedges, and edge-as-entity semantic.
- Multilayer and slice-aware modeling: represent layers, aspects, inter-layer links, and named graph slices.
- Annotated tables throughout: keep structured metadata for vertices, edges, slices, layers, and graph-level state, using Narwhals-compatible dataframes.
- Interoperability without friction: import/export with NetworkX, igraph, graph-tool, GraphML, GEXF, SIF, SBML, CX2 (Cytoscape exchange format v2), Excel/CSV/TSV/JSON, Parquet graph directories, and DataFrames.
- Algorithm interoperability: seamless, lazy calls into NetworkX/igraph/graph‑tool via the graph-owned `G.nx`, `G.ig`, and `G.gt` accessors.
- Stable storage: persist graphs in a lossless `.annnet` layout built from Zarr, Parquet, and JSON.


## Features

### Minimal core dependencies

[![NumPy](https://img.shields.io/badge/basic_math_with-NumPy-013243?logo=numpy&logoColor=013243)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/sparse_matrix_via-SciPy-8CAAE6?logo=scipy)](https://scipy.org/)
[![Narwhals](https://img.shields.io/badge/dataframe--agnostic_design_with-Narwhals-blue?logo=narwhals)](https://narwhals-dev.github.io/narwhals/)

The base package keeps the required runtime small. Only `numpy`, `scipy`, and `narwhals` are mandatory, which gives you the core graph model, sparse structure handling, and dataframe interoperability layer without forcing heavyweight optional backends.

### Graph modeling

- Simple graphs, directed graphs, and multigraphs
- Hyperedges, including directed head-to-tail hyperedges
- Signed and weighted relations with rich node and edge annotations
- Efficient indexing, lookup helpers, and slice-based views
- Edge-, vertex-, and graph-level semantics in the same container

### Data and file interoperability

[![Polars](https://img.shields.io/badge/tabular_backend-Polars-0075FF?logo=polars&logoColor=0075FF)](https://pola.rs/)
[![pandas](https://img.shields.io/badge/or-pandas-150458?logo=pandas&logoColor=150458)](https://pandas.pydata.org/)
[![PyArrow](https://img.shields.io/badge/or-PyArrow-5C2D91?logo=apachearrow&logoColor=white)](https://arrow.apache.org/docs/python/)

AnnNet separates tabular in-memory backends from file formats. For annotated tables in memory, it can work through Narwhals with Polars DataFrames, pandas DataFrames, or PyArrow-backed tabular objects, letting you keep the table engine that best fits the rest of your workflow.

The file-format IO modules cover graph exchange and persistence formats such as:

- GraphML, GEXF, SIF, and SBML
- CX2 (Cytoscape exchange)
- Parquet graph directories
- CSV/TSV, JSON/NDJSON, and Excel

### Runtime backend interoperability

[![NetworkX](https://img.shields.io/badge/graph_backend-NetworkX-blue?logo=networkx)](https://networkx.org/)
[![igraph](https://img.shields.io/badge/and/or-igraph-blue)](https://igraph.org/python/)
[![graph-tool](https://img.shields.io/badge/and/or-graph--tool-blue)](https://graph-tool.skewed.de/)

- NetworkX via `G.nx` and `to_nx` / `from_nx`
- igraph via `G.ig` and `to_igraph` / `from_igraph`
- graph-tool via `G.gt` and `to_graphtool` / `from_graphtool`

### Cytoscape Web integration and other network visualizations

[![Cytoscape CX2](https://img.shields.io/badge/interactive_visualization_with-Cytoscape_Web-F7DF1E?logo=cytoscapedotjs)](https://cytoscape.org/)
[![Graphviz](https://img.shields.io/badge/static_plot_with-Graphviz-blue)](https://graphviz.org/)
[![pydot](https://img.shields.io/badge/or-pydot-blue)](https://github.com/pydot/pydot)
[![Matplotlib](https://img.shields.io/badge/or-Matplotlib-blue)](https://matplotlib.org/)

annnet can export CX2 for Cytoscape and also render Cytoscape.js views directly from Python. That makes it practical to inspect graphs in a browser or notebook-oriented workflow without first leaving the AnnNet data model. For static outputs, it also supports Graphviz, pydot, and matplotlib-based rendering for scripts, reports, and export-to-file workflows.

### Disk-backed storage

[![Zarr](https://img.shields.io/badge/matrix_stored_with-Zarr-blue)](https://zarr.readthedocs.io/)
[![Parquet](https://img.shields.io/badge/annotation_tables_strored_with-Parquet-50ABF1?logo=apacheparquet)](https://parquet.apache.org/)

The native `.annnet` format is intended to be lossless and analysis-friendly. Sparse structure is stored in Zarr, annotated tables are stored in Parquet, and metadata is stored in JSON so graphs can be reopened without flattening away structure or annotations.

## Installation

The latest stable release is published on PyPI:

[![PyPI](https://img.shields.io/pypi/v/annnet?logo=pypi&logoColor=white)](https://pypi.org/project/annnet/)
[![Status](https://img.shields.io/pypi/status/annnet)](https://pypi.org/project/annnet/)
[![Package format](https://img.shields.io/pypi/format/annnet)](https://pypi.org/project/annnet/)
[![Implementation](https://img.shields.io/pypi/implementation/annnet)](https://pypi.org/project/annnet/)

Base install:

```bash
pip install annnet
```

Optional extras let you add tabular backends, graph backends, plotting backends, and IO/storage support only when needed:

```bash
# Graph backends
pip install "annnet[networkx,igraph]"

# IO and serialization helpers
pip install "annnet[io]"

# Plotting helpers
pip install "annnet[plot]"

# More granular extras
pip install "annnet[polars]"
pip install "annnet[pandas]"
pip install "annnet[pyarrow]"
pip install "annnet[parquet]"
pip install "annnet[zarr_io]"
pip install "annnet[excel]"
pip install "annnet[sbml]"
pip install "annnet[matplotlib]"
pip install "annnet[pydot]"
pip install "annnet[graphviz]"

# Common pip-installable extras in one bundle
pip install "annnet[all]"
```

`graph-tool` is supported through the adapters and the `G.gt` accessor when installed, but it is not distributed on PyPI. Install it through your OS or conda-based package manager and annnet will detect it.


## Quick Start

```python
import annnet as an

G = an.Graph(directed=True)  # default direction; can be overridden per-edge

# Create slices and set active
G.slices.add_slice("toy")
G.slices.add_slice("train")
G.slices.add_slice("eval")
G.slices.active = "toy"

# Add vertices with attributes
for v in ["A", "B", "C", "D"]:
    G.add_vertices(v, label=v, kind="gene")

# 1) Binary directed edge
e_dir = G.add_edges("A", "B", weight=2.0, directed=True, relation="activates")

# 2) Binary undirected edge
e_undir = G.add_edges("B", "C", weight=1.0, directed=False, relation="binds")

# 3) Self-loop
e_loop = G.add_edges("D", "D", weight=0.5, directed=True, relation="self")

# 4) Parallel edge
e_parallel = G.add_edges("A", "B", weight=5.0, parallel="parallel", relation="alternative")

# 5) Vertex-edge hybrid relation
G.add_edges(edge_id="edge_e1", as_entity=True, description="signal")
e_vx = G.add_edges("edge_e1", "C", directed=True, as_entity=True, channel="edge->vertex")

# 6) Undirected hyperedge (3-way membership)
e_hyper_undir = G.add_edges(["A", "C", "D"], weight=1.0, directed=False, tag="complex")

# 7) Directed hyperedge (head→tail member groups)
e_hyper_dir = G.add_edges(["A", "B"], ["C", "D"], weight=1.0, directed=True, reaction="A+B->C+D")

# 8) Run a NetworkX algorithm if networkx is installed
deg = G.nx.degree_centrality(G)
```


## Backend-Specific Integration

If an optional backend is installed, AnnNet can dispatch directly to that backend while handling conversion for you:

```python
centrality = G.nx.degree_centrality(G)
```

`G.ig` and `G.gt` behave similarly for igraph and graph-tool. On calls such as `G.nx.<function>(G, ...)`, annnet resolves the backend callable, converts the AnnNet graph using the requested projection options, swaps in the converted backend graph, executes the algorithm call, and returns the backend result. Converted backend graphs are cached and refreshed after AnnNet mutations.

Use `G.nx.backend(...)`, `G.ig.backend(...)`, or `G.gt.backend(...)` when you want the concrete projected graph object. Use `G.nx.<function>(G, ...)`, `G.ig.<function>(G, ...)`, or `G.gt.<namespace>.<function>(G, ...)` when you want conversion plus dispatch in one step.

Examples:

```python
# Operate on the active slice
bc = G.nx.betweenness_centrality(G)

# Build a projected NetworkX graph explicitly
nxG = G.nx.backend(
    directed=True,
    hyperedge_mode="skip",  # or "expand"
    slice="toy",
    simple=True,
)

# igraph interoperability
pagerank = G.ig.pagerank(G)
```


## Interoperability

High-fidelity conversions aim to preserve IDs, attributes, and directionality. When a target format is inherently lossy, annnet exposes manifests that can be reused during re-import to recover as much structure as possible.

Two adapter families are involved:

- Runtime backend adapters for in-memory conversion and algorithm dispatch with NetworkX, igraph, and graph-tool
- IO adapters for file formats and tabular exchange such as GraphML, GEXF, SIF, SBML, CX2, Parquet, Excel, JSON, and Narwhals-backed dataframes

NetworkX:

```python
import annnet as an

nxG, manifest = an.adapters.to_nx(G, directed=True, hyperedge_mode="skip")

# ... run algorithms or edit nxG ...

G2 = an.adapters.from_nx(nxG, manifest)
```

igraph:

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

# CX2 (Cytoscape exchange format v2)
cx2 = an.io.to_cx2(G, hyperedges="reify")
L = an.io.from_cx2(cx2, hyperedges="manifest")
```

Notes:

- Hyperedges may be dropped, expanded, or reified depending on the target and conversion mode.
- Backend graphs are single-view projections, so slice handling should be chosen explicitly when it matters.
- Returned manifests improve round-trip fidelity when the external target cannot encode all AnnNet semantics directly.

## Storage

The `.annnet` format is the native persisted representation for annnet graphs. It is designed to preserve structure, attributes, slices, and metadata in a layout that is both lossless and inspectable on disk.

```python
import annnet as an

an.io.write(G, "my_graph.annnet", overwrite=True)
R = an.io.read("my_graph.annnet")
```

Layout highlights:

- `manifest.json` stores versioning, counts, and slice metadata
- `structure/incidence.zarr` stores sparse topology data
- `structure/*.parquet` stores structural indices and edge definitions
- `tables/*.parquet` stores vertex, edge, and slice annotations
- `layers/`, `slices/`, `audit/`, and `uns/` keep multilayer state, slice state, history, and unstructured metadata

## Package Overview

annnet is organized in layers. Most users interact with the high-level graph API, while adapters, IO, and backend-specific helpers stay separated underneath so graph semantics do not get tangled with storage or third-party conversions.

```text
annnet/
├── core/              # Main Graph API, managers, views, and backend accessors
├── adapters/          # In-memory conversion layers for external graph libraries
├── io/                # File readers/writers and native .annnet persistence
├── algorithms/        # Algorithms implemented against AnnNet data-structure
└── utils/             # Plotting and other shared utilities
```

### Internal design

- Sparse incidence matrix for the core topology
- Attributes decoupled from structure through annotated tables
- Lazy conversion to external backends on demand
- Optional caching for converted backend graphs

### Philosophy

- One consistent interface across multiple graph families
- Interoperability first, not ecosystem replacement
- Performance-aware design with room for richer semantics
- Modular internals rather than a monolithic graph object

## Development

### Local dev and test setup

[![uv](https://img.shields.io/badge/package_manager-uv-DE5FE9?logo=uv)](https://github.com/astral-sh/uv)
[![Pixi](https://img.shields.io/badge/secondary_env_manager-Pixi-orange)](https://pixi.sh/)

For local development, `uv` is the lightest path for syncing dependency groups and running day-to-day checks. Pixi is the heavier but more complete route when you need the broader environment, including graph-tool where supported.

```bash
# Dev tools
uv sync --group dev

# Test dependencies
uv sync --group tests
uv run pytest
```

For the full Pixi environment:

```bash
pixi install
pixi run test-all
```

Notes:

- `graph-tool` is only available on selected platforms.
- The Pixi configuration includes Python and graph-tool-aware test tasks for the supported platforms.

### Docs setup

Use the module entrypoint for MkDocs in this repository:

```bash
uv sync --group docs
uv run python -m mkdocs serve
```

## Contributing

Contributions are welcome, especially around public API, interoperability, file adapters, and algorithms. For changes that affect graph semantics or round-trip behavior, include tests that demonstrate the intended behavior and guard against regressions.

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-FAB040?logo=pre-commit)](https://pre-commit.com/)
[![Ruff](https://img.shields.io/badge/linting_&_formatting_with-Ruff-D7FF64?logo=ruff)](https://github.com/astral-sh/ruff)
[![tox](https://img.shields.io/badge/tested%20with-tox-darkgray)](https://tox.wiki/)

Useful contributor tooling includes `pre-commit`, `ruff`, `black`, `pytest`, and `tox`. If you are changing IO or backend conversion code, it is worth testing both happy-path import/export and round-trip fidelity.

### CI and repository activity

[![Lint](https://img.shields.io/github/actions/workflow/status/saezlab/annnet/ci-linting.yml?branch=main&label=lint)](https://github.com/saezlab/annnet/actions/workflows/ci-linting.yml)
[![Security](https://img.shields.io/github/actions/workflow/status/saezlab/annnet/ci-security.yml?branch=main&label=security)](https://github.com/saezlab/annnet/actions/workflows/ci-security.yml)
[![Docs Build](https://img.shields.io/github/actions/workflow/status/saezlab/annnet/ci-docs.yml?branch=main&label=docs%20build)](https://github.com/saezlab/annnet/actions/workflows/ci-docs.yml)
[![Issues](https://img.shields.io/github/issues/saezlab/annnet)](https://github.com/saezlab/annnet/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/saezlab/annnet)](https://github.com/saezlab/annnet/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/saezlab/annnet)](https://github.com/saezlab/annnet/commits/main)

The repository uses GitHub Actions for testing, linting, docs, and security checks. Open issues and pull requests are the best place to discuss design questions or proposed changes before a larger refactor.

## License

annnet is licensed under the [BSD-3](https://opensource.org/license/bsd-3-clause) License.
