# AnnNet Refactoring Readiness Overview

This document is a repository reconnaissance pass for refactoring preparation.
It treats the codebase as the source of truth for current behavior. Documentation
is used only as evidence of intended design, future direction, or mismatch.

## Snapshot Summary

| Topic | Current finding |
|---|---|
| Package purpose | A Python graph container centered on `AnnNet`, using sparse incidence matrices plus dataframe-like annotation tables to represent directed, undirected, mixed, hyperedge, slice, and multilayer graph data. |
| Current maturity | Alpha (`pyproject.toml` classifier), with broad implemented surface but several compatibility shims, stale examples, untracked/stashed files, and some apparently unfinished internal APIs. |
| Architectural style | One large central object (`annnet/core/graph.py`) composed from mixins for bulk ops, annotations, caches, history, indexing, layers, slices, views, and traversal. |
| Main public entry points | `annnet.AnnNet`, `annnet.EdgeType`, top-level lazy IO/conversion functions in `annnet/__init__.py`, `annnet.io.*`, `annnet.adapters.*`, and instance properties such as `G.nx`, `G.ig`, `G.gt`, `G.slices`, `G.layers`, `G.idx`, `G.cache`. |
| Test status | Large test suite under `tests/`, CI runs pytest with coverage across Python 3.11-3.13. Local pytest could not run because `pytest` is not installed in either the shell environment or `.venv`; local `ruff check annnet tests` passed. |
| Docs/code alignment | Mixed. Explanatory docs for the current internal SSOT mostly align with implementation; README/quickstart/older `architecture.md` contain stale or aspirational APIs such as `an.Graph`, `an.annnet`, `add_hyperedge`, and `write_parquet_graphdir`. |

## 1. Executive Summary

### Current implementation

The package currently implements an annotated graph data structure named
`AnnNet` in `annnet/core/graph.py`. It stores topology as a SciPy DOK sparse
incidence matrix (`self._matrix`) plus canonical Python records:

- `_entities: dict[tuple, EntityRecord]`
- `_edges: dict[str, EdgeRecord]`
- `_row_to_entity: dict[int, tuple]`
- `_col_to_edge: dict[int, str]`

The central class also maintains slice membership, multilayer/aspect state,
dataframe-backed attributes, mutation history, view helpers, lazy external
backend accessors, and compatibility mappings for older dict-like APIs.

The implemented package is more than a thin graph wrapper. It is trying to be a
high-expressivity graph container with:

- binary directed and undirected edges
- mixed per-edge directionality
- parallel edges
- hyperedges represented through `add_edge(src=[...], tgt=[...])` and
  `add_hyperedges_bulk(...)`
- edge-entities via `add_edge(..., as_entity=True)`
- slices and per-slice edge attributes
- Kivela-style multilayer state and supra-matrix operations
- JSON, Parquet GraphDir, native `.annnet`, SIF, GraphML/GEXF, CX2, SBML, CSV,
  Excel, DataFrame, NetworkX, igraph, graph-tool, and PyG conversion pathways
- lazy algorithm dispatch through `G.nx`, `G.ig`, and `G.gt`

### Current maturity/status

The package advertises alpha maturity in `pyproject.toml`:
`Development Status :: 3 - Alpha`. That matches the repository state. There is
substantial functionality and test coverage, but the codebase also has signs of
active transition:

- untracked/stashed-looking files in `annnet/core/`, including
  `_Annotation_stashed.py`, `_BulkOps_stashed.py`, and
  `_BulkOps_before_edge_insert_opt.py`
- two Pixi configurations (`pixi.toml` and `[tool.pixi.*]` in `pyproject.toml`)
  with different platform/dependency detail
- docs and README examples that call names not exported or not implemented
- legacy compatibility mappings in `annnet/core/_helpers.py`
- broad optional dependency handling, but some modules import optional
  dependencies at module import time

### Main strengths

- The current internal model has a clear intended single source of truth:
  `EntityRecord`, `EdgeRecord`, sparse incidence matrix, and reverse row/column
  maps in `annnet/core/_helpers.py` and `annnet/core/graph.py`.
- The package has a broad test suite across core graph operations, adapters,
  IO formats, backend accessors, history, plotting, edge cases, and performance
  smoke checks.
- Public top-level lazy loading in `annnet/__init__.py` and `annnet/io/__init__.py`
  reduces hard dependency pressure for optional IO/backends.
- The docs under `docs/explanations/` document the current internal SSOT more
  accurately than the README and older `architecture.md`.
- `ruff check annnet tests --output-format=concise` passes locally in `.venv`.

### Main risks, gaps, and inconsistencies

- `AnnNet` is a very large composition surface. `annnet/core/graph.py` is about
  2,584 lines, `_Layers.py` about 3,851 lines, `_Annotation.py` about 1,616
  lines, `_BulkOps.py` about 1,418 lines, `_Cache.py` about 1,163 lines, and
  `_Views.py` about 1,043 lines. Refactors must preserve cross-mixin invariants.
- Public/internal boundaries are blurry. Tests and adapters frequently use
  private fields like `_edges`, `_entities`, `_slices`, `_state_attrs`, and
  `_matrix` directly.
- Some documented public methods are not implemented: `add_hyperedge`,
  `add_vertices`, `add_parallel_edge`, `add_edge_entity`, `set_hyperedge_coeffs`,
  `an.Graph`, `an.annnet`, `write_parquet_graphdir`, and `read_parquet_graphdir`
  appear in docs/README but not in current exports.
- Some implemented public or semi-public code appears stale: `annnet/adapters/manager.py`
  and `annnet/adapters/_proxy.py` reference a `_state` backend cache model that
  does not match current `AnnNet`.
- Native `.annnet` IO has a placeholder `_write_cache()` that raises
  `NotImplementedError` if cache-writing is triggered.
- Optional dependency boundaries are inconsistent: for example,
  `annnet/io/GraphML_io.py` imports `networkx` at module import time, and
  `annnet/adapters/pyg_adapter.py` imports `torch` and `torch_geometric` at
  module import time.

## 2. Repository Structure

### Top-level layout

- `annnet/`: package source.
- `tests/`: pytest/unittest test suite covering core graph behavior, IO,
  adapters, backend accessors, history, views, edge cases, plotting, public API, and
  performance.
- `docs/`: MkDocs documentation, explanation pages, API reference pages, and
  notebook-based tutorials.
- `notebooks/`: working notebooks and tutorial notebooks; some notebooks are
  also copied under `docs/tutorials/notebooks/`.
- `benchmarks/`: standalone benchmark harness and benchmark scripts.
- `annnet/benchmarks/`: package-included benchmark modules.
- `tools/`: documentation helper scripts (`gen_reference.py`,
  `mkdocs_hooks.py`).
- `.github/`: CI workflows and a shared setup action.
- `README.md`: project-facing overview and examples.
- `architecture.md`: older, long architecture/design document.
- `pyproject.toml`: packaging, dependencies, ruff, pytest, coverage, tox, uv,
  and Pixi config.
- `pixi.toml`, `pixi.lock`: additional Pixi workspace config.
- `environment.yml`, `environment-annnet-anndata.yml`: Conda environment files.

### Important package modules

- `annnet/__init__.py`: curated lazy public API. Exports `AnnNet`, `EdgeType`,
  `Traversal`, metadata helpers, adapters, IO functions, and selected format
  helpers.
- `annnet/_metadata.py`: package metadata and notebook-friendly environment
  capability reporting (`info()`, `get_metadata()`, `get_latest_version()`).
- `annnet/core/graph.py`: central `AnnNet` class, scalar mutation API, edge
  semantics, properties, compatibility views, backend accessor properties, native
  `read`/`write` methods, snapshots, and diffs.
- `annnet/core/_helpers.py`: `EntityRecord`, `EdgeRecord`, `EdgeType`, dataframe
  helper, and legacy dict-like compatibility mappings.
- `annnet/core/_BulkOps.py`: batched vertex, edge, hyperedge, slice, key, and
  removal operations.
- `annnet/core/_Annotation.py`: graph/vertex/edge/slice/edge-slice attributes,
  dataframe upsert logic, audit helpers, flexible direction updates.
- `annnet/core/_Layers.py`: multilayer/aspect/layer state, supra-node presence,
  layer-derived subgraphs/slices, supra matrices/tensors, random-walk and
  spectral helpers.
- `annnet/core/_Slices.py`: slice manager and legacy slice methods.
- `annnet/core/_Views.py`: lazy `GraphView` and table views.
- `annnet/core/_Index.py`: row/column and entity/edge lookup manager.
- `annnet/core/_Cache.py`: derived CSR/CSC/adjacency caches plus subgraph,
  reverse, copy, memory usage, and incidence matrix exports.
- `annnet/core/_History.py`: in-memory mutation log, history export, snapshots,
  and `GraphDiff`.
- `annnet/core/backend_accessors/`: current `G.nx`, `G.ig`, and `G.gt` lazy backend
  accessors.
- `annnet/adapters/`: runtime backend adapters for NetworkX, igraph, graph-tool,
  PyG, plus manifest helpers and an older manager/proxy abstraction.
- `annnet/io/`: file/data IO modules.
- `annnet/algorithms/traversal.py`: core traversal mixin methods.
- `annnet/utils/`: plotting, typing, validation, and placeholder config.

### Notable configuration and CI

- Build backend: Hatchling in `[build-system]`.
- Runtime dependencies in `[project]`: `numpy`, `narwhals`, `scipy`.
- Optional extras: `polars`, `pandas`, `pyarrow`, `networkx`, `python-igraph`,
  `torch`, `torch-geometric`, `matplotlib`, `pydot`, `graphviz`, `openpyxl`,
  `lxml`, `zarr`, `numcodecs`, `cobra`, `toml`.
- Dependency groups: `dev`, `docs`, `tests`.
- Pytest config: `[tool.pytest.ini_options]`, test paths under `tests`, import
  mode `importlib`, strict xfail.
- Coverage config: `[tool.coverage.run]` uses `source = ["annnet"]`.
- Ruff config: `[tool.ruff]`, `[tool.ruff.format]`, `[tool.ruff.lint]`; target
  Python is `py312`, line length 100, annotation linting selected.
- Tox config runs lint, tests, coverage, README check, and docs build across
  several Python versions, though `env_list` still includes Python 3.9 while
  the project requires `>=3.10`.
- CI:
  - `.github/workflows/ci-testing-unit.yml`: pytest with coverage on 3.11,
    3.12, 3.13; installs PyG dependencies; has separate Pixi graph-tool test.
  - `.github/workflows/ci-linting.yml`: `ruff check .` and `ruff format --check .`.
  - `.github/workflows/ci-docs.yml`: MkDocs deploy.
  - `.github/workflows/ci-security.yml`: Bandit scan.
- Docs: `mkdocs.yml` uses Material, mkdocstrings, autorefs, mkdocs-jupyter with
  notebook execution disabled, plus `tools/mkdocs_hooks.py`.

## 3. Package Purpose and Domain Model

### Current implementation

The package is trying to solve the problem of representing richly annotated
network data where a plain adjacency-list graph is insufficient. The core model
uses incidence columns for edges and rows for entities, allowing binary edges,
hyperedges, edge-entities, and multilayer supra-node states to share one
matrix-backed representation.

Main current entities and abstractions:

- `AnnNet`: the graph container and coordination object.
- `EntityRecord`: row-space record for a vertex or edge-entity.
- `EdgeRecord`: column-space/topological record for binary, hyper, and
  vertex-edge cases.
- `EdgeType`: enum with `DIRECTED` and `UNDIRECTED`; exposed but not the main
  internal direction representation.
- entity key: `(vertex_id, layer_coord)`, with flat graphs using `("_",)`.
- edge id: string key mapped to `EdgeRecord` and usually one incidence column.
- slice: named membership overlay in `_slices`.
- layer/aspect: multilayer coordinate state in `_aspects`, `_layers`, `_VM`,
  `_state_attrs`, and related helpers.
- annotation tables: `vertex_attributes`, `edge_attributes`, `slice_attributes`,
  `edge_slice_attributes`, `layer_attributes`.
- view: `GraphView`, a lazy filtered facade over the same graph.
- backend proxy: lazy adapters exposed through `G.nx`, `G.ig`, `G.gt`.

Core workflows implied by code and tests:

- Build a graph with `G = AnnNet(directed=True|False|None)`.
- Add vertices with `add_vertex(...)` or `add_vertices_bulk(...)`.
- Add binary edges, hyperedges, and edge-entities through `add_edge(...)`; add
  many hyperedges with `add_hyperedges_bulk(...)`.
- Attach or query attributes through `set_vertex_attrs`, `set_edge_attrs`,
  `set_slice_attrs`, `set_edge_slice_attrs`, and getter variants.
- Manage slices through `add_slice`, `set_active_slice`, `G.slices`, and slice
  operation helpers.
- Manage multilayer aspects/presence through `set_aspects`,
  `add_elementary_layer`, `add_presence`, `add_intra_edge`, `add_inter_edge`,
  `add_coupling_edge_nl`, and `G.layers`.
- Use `G.view(...)`, `edges_view`, `vertices_view`, `obs`, `var`, and `X` for
  filtered or AnnData-like access.
- Convert to/from external formats through top-level lazy functions or
  `annnet.io.*` / `annnet.adapters.*`.
- Run external algorithms by passing `G` through backend accessors, e.g.
  `G.nx.degree_centrality(G)`.
- Save/load native format with `G.write(path)`, `AnnNet.read(path)`,
  `annnet.write(...)`, or `annnet.read(...)`.

### Documented intention

The README describes the project as an "Annotated Network Data Structure for
Science" combining AnnData-style annotated containers, networks, multilayer
structures, hypergraphs, and interoperability. The explanation docs describe a
central structural SSOT with overlays and derived materializations.

### Inference / hypothesis

The code appears to be in a migration from older dict-style public internals
(`entity_to_idx`, `edge_definitions`, `edge_weights`, etc.) toward structured
records (`EntityRecord`, `EdgeRecord`). The compatibility mappings in
`annnet/core/_helpers.py` strongly indicate this, but the exact migration plan
is unclear.

## 4. Architecture

### High-level architecture

The runtime architecture is centered on one object:

```text
AnnNet
  owns canonical topology
  owns overlays: slices, multilayer, attributes, history
  exposes managers: slices, layers, idx, cache
  materializes views and backend graphs on demand
  delegates IO/conversions to annnet.io and annnet.adapters
```

`AnnNet` inherits from these mixins in `annnet/core/graph.py`:

- `BulkOps`
- `Operations`
- `History`
- `ViewsClass`
- `IndexMapping`
- `LayerClass`
- `SliceClass`
- `AttributesClass`
- `Traversal`

This inheritance stack is the most important architectural fact for refactoring.
Methods from separate files mutate the same private fields and depend on shared
invariants.

### Internal layering and responsibilities

Current layering by implementation:

- Core structural model:
  - `annnet/core/graph.py`
  - `annnet/core/_helpers.py`
- Mutation helpers:
  - scalar methods in `graph.py`
  - bulk methods in `_BulkOps.py`
- Metadata/annotations:
  - `_Annotation.py`
  - dataframe fallback helpers in `_helpers.py` and `io_annnet.py`
- Context overlays:
  - `_Slices.py`
  - `_Layers.py`
- Derived data:
  - `_Cache.py`
  - `_Views.py`
  - backend accessors in `core/backend_accessors/`
- External boundaries:
  - `annnet/adapters/*`
  - `annnet/io/*`
- Public import facade:
  - `annnet/__init__.py`
  - `annnet/io/__init__.py`
  - `annnet/adapters/__init__.py`

### Key classes, functions, and extension points

Key classes:

- `AnnNet` in `annnet/core/graph.py`
- `EntityRecord`, `EdgeRecord`, `EdgeType` in `annnet/core/_helpers.py`
- `GraphView`, `ViewsClass` in `annnet/core/_Views.py`
- `SliceManager`, `SliceClass` in `annnet/core/_Slices.py`
- `LayerManager`, `LayerClass` in `annnet/core/_Layers.py`
- `IndexManager`, `IndexMapping` in `annnet/core/_Index.py`
- `CacheManager`, `Operations` in `annnet/core/_Cache.py`
- `GraphDiff`, `History` in `annnet/core/_History.py`
- `Traversal` in `annnet/algorithms/traversal.py`
- `_NXBackendAccessor`, `_IGBackendAccessor`, `_GTBackendAccessor` in
  `annnet/core/backend_accessors/`
- `NetworkXAdapter`, `IGraphAdapter`, `GraphAdapter`, `BackendProxy` in
  `annnet/adapters/`, though some of these are legacy or incomplete.

Important public functions:

- Top-level facade in `annnet/__init__.py`: `to_nx`, `from_nx`, `to_igraph`,
  `from_igraph`, `to_graphtool`, `from_graphtool`, `to_pyg`, native
  `write`/`read`, and IO helpers.
- IO:
  - `annnet/io/io_annnet.py`: `write`, `read`
  - `json_io.py`: `to_json`, `from_json`, `write_ndjson`
  - `dataframe_io.py`: `to_dataframes`, `from_dataframes`
  - `csv_io.py`: `load_csv_to_graph`, `from_dataframe`,
    `export_edge_list_csv`, `export_hyperedge_csv`
  - `GraphML_io.py`: `to_graphml`, `from_graphml`, `to_gexf`, `from_gexf`
  - `Parquet_io.py`: `to_parquet`, `from_parquet`
  - `SIF_io.py`: `to_sif`, `from_sif`
  - `SBML_io.py`: `from_sbml`
  - `sbml_cobra_io.py`: `from_cobra_model`, `from_sbml`
  - `cx2_io.py`: `to_cx2`, `from_cx2`, `show`
  - `excel.py`: `load_excel_to_graph`
  - `read_omnipath.py`: `read_omnipath`

### Data flow and control flow

Graph construction flow:

1. `AnnNet.__init__` initializes canonical stores, matrix capacity, annotation
   tables, slices, history, multilayer placeholders, and state-attribute maps.
2. `add_vertex` registers `(vertex_id, layer_coord)` in `_entities`, grows
   matrix rows, updates `_V`/`_VM`, updates slice membership, and upserts vertex
   attributes.
3. `add_edge` parses binary/hyper/coefficient inputs, resolves direction,
   ensures endpoints exist, allocates or reuses an edge column, writes incidence
   entries, writes/updates an `EdgeRecord`, updates adjacency indices, optionally
   registers the edge as an entity, updates slice membership, applies flexible
   direction, and upserts edge attributes.
4. Bulk operations in `_BulkOps.py` aim to preserve the same semantics while
   reducing dataframe updates and sparse matrix resizing.

View and backend flow:

1. `G.view(...)` creates `GraphView` over the source object.
2. `GraphView` computes filtered vertex and edge IDs lazily from source graph
   stores and slices.
3. `G.nx`, `G.ig`, and `G.gt` instantiate backend accessors that convert the current
   graph to backend graphs and cache by graph `_version`.
4. Adapter modules use manifests for information external graph libraries cannot
   represent directly, such as hyperedges or multilayer metadata.

Persistence flow:

1. Native `write` in `annnet/io/io_annnet.py` writes a directory or `.annnet`
   archive.
2. The format includes `manifest.json`, `structure/incidence.zarr`, structure
   Parquet tables, attribute tables, layer/slice/audit/uns metadata.
3. Native `read` reconstructs an `AnnNet` and reloads structure, tables, layers,
   slices, audit, and uns.

### Coupling hotspots and architectural entry points

Hotspots:

- `AnnNet` private state is directly accessed across almost every core mixin,
  IO module, adapter, and many tests.
- `_Layers.py` is very large and mixes layer registry operations, supra matrix
  building, tensor views, spectral/random-walk helpers, and coupling generation.
- `_Annotation.py` mixes public attribute APIs, dataframe backend compatibility,
  schema evolution, and flexible edge direction side effects.
- `_Cache.py` mixes cache management, subgraph/copy/reverse operations, memory
  usage, and incidence matrix exports.
- Adapter and IO modules duplicate manifest serialization concepts and direct
  `EdgeRecord` interpretation.
- Compatibility mappings in `_helpers.py` expose dict-like mutable surfaces over
  the new canonical records.

Potential seams:

- Public facade lazy exports in `annnet/__init__.py`, `annnet/io/__init__.py`,
  and `annnet/adapters/__init__.py`.
- Record definitions in `annnet/core/_helpers.py`.
- Adapter manifest helpers in `annnet/adapters/_utils.py`.
- Native IO helper functions in `annnet/io/io_annnet.py`.
- Tests around `test_public_api.py`, `test_graph.py`, `test_graph_views.py`,
  `test_history.py`, and format-specific tests.

## 5. Implemented API Surface

### Current implementation

Top-level public API from `annnet.__all__` includes:

- Classes/objects: `AnnNet`, `EdgeType`, `Traversal`
- Metadata: `__version__`, `__license__`, `get_metadata`, `get_latest_version`,
  `info`, and related names
- Submodules: `adapters`, `io`, `core`, `algorithms`
- Adapter functions: `available_backends`, `to_nx`, `from_nx`, `to_igraph`,
  `from_igraph`, `to_graphtool`, `from_graphtool`, `to_pyg`
- Native storage: `write`, `read`
- IO functions: `to_json`, `from_json`, `write_ndjson`, `to_dataframes`,
  `from_dataframes`, `from_csv`, `from_dataframe`, `edges_to_csv`,
  `hyperedge_to_csv`, `from_excel`, `to_sif`, `from_sif`, `to_graphml`,
  `from_graphml`, `to_gexf`, `from_gexf`, `to_cx2`, `from_cx2`, `show_cx2`,
  `to_parquet`, `from_parquet`, `from_sbml`, `from_cobra_model`,
  `from_sbml_cobra`, `read_omnipath`

`AnnNet` itself exposes many methods and properties. Important implemented
families include:

- construction: `add_vertex`, `add_edge`, `add_vertices_bulk`,
  `add_edges_bulk`, `add_hyperedges_bulk`, `add_edges_to_slice_bulk`
- removal: `remove_edge`, `remove_vertex`, `remove_slice`, `remove_edges`,
  `remove_vertices`, `remove_orphans`
- queries: `vertices`, `edges`, `edge_list`, `has_edge`, `has_vertex`,
  `get_edge_ids`, `degree`, `incident_edges`, `in_edges`, `out_edges`,
  counts and shape properties
- attributes: `set_vertex_attrs`, `set_edge_attrs`, `set_slice_attrs`,
  `set_edge_slice_attrs`, getters, bulk setters, `audit_attributes`
- slices: `add_slice`, `set_active_slice`, `get_active_slice`, `list_slices`,
  `slice_union`, `slice_intersection`, `slice_difference`, slice stats
- multilayer: `set_aspects`, `set_elementary_layers`, `add_presence`,
  `add_intra_edge`, `add_inter_edge`, `add_coupling_edge_nl`,
  `supra_adjacency`, `supra_incidence`, tensor/random-walk/spectral helpers
- views/managers: `view`, `edges_view`, `vertices_view`, `slices_view`,
  `aspects_view`, `layers_view`, and manager properties `slices`, `layers`,
  `idx`, `cache`
- lazy backends: `nx`, `ig`, `gt`
- native storage: instance `write`, classmethod `read`
- history: `history`, `export_history`, `enable_history`, `clear_history`,
  `mark`, `snapshot`, `diff`, `list_snapshots`

### Internal/private

Most fields with leading underscores should be treated as private, but current
code and tests use them heavily:

- `_entities`, `_edges`, `_row_to_entity`, `_col_to_edge`
- `_adj`, `_src_to_edges`, `_tgt_to_edges`
- `_matrix`, `_csr_cache`
- `_slices`, `_current_slice`, `_default_slice`, `slice_edge_weights`
- `_aspects`, `_layers`, `_V`, `_VM`, `_state_attrs`
- `_history`, `_version`, `_snapshots`
- compatibility mappings such as `entity_to_idx`, `edge_definitions`, and
  `edge_weights` expose private state through public properties.

### Unstable or ambiguous boundaries

- `annnet/adapters/manager.py` exposes `get_proxy` and `ensure_materialized`,
  but they appear inconsistent with current `AnnNet` state and current lazy
  proxy implementation.
- `annnet/adapters/_base.py` defines an abstract `GraphAdapter`, but current
  main adapter functions are function-based and have richer signatures than the
  base class.
- `NetworkXAdapter` and `IGraphAdapter` class wrappers exist, but the tested and
  exported surface is primarily function-based.
- `EdgeType` is exported but internal direction is currently represented mostly
  as booleans or `None`.
- The compatibility properties in `graph.py` make old dict-like mutation still
  possible, which can bypass or complicate invariants.

### Mismatch between exposed API and intended/documented API

Contradictions found:

- `README.md` uses `an.Graph(...)`; current top-level API exports `AnnNet`, not
  `Graph`.
- `README.md` uses `G.add_vertices(...)`; current implementation has
  `add_vertex` and `add_vertices_bulk`, not `add_vertices`.
- `README.md` uses `G.add_parallel_edge(...)`; current implementation uses
  `add_edge(..., parallel="parallel")`.
- `README.md` uses `G.add_edge_entity(...)`; current implementation uses
  `add_edge(..., as_entity=True, edge_id=...)` for edge-entity behavior.
- `README.md` and `docs/quickstart.md` use `G.add_hyperedge(...)`; current
  implementation has no scalar `add_hyperedge` method. Hyperedges are created
  with `add_edge(src=[...], tgt=[...])` or `add_hyperedges_bulk(...)`.
- `architecture.md` uses `G.set_hyperedge_coeffs(...)`; current method is
  `set_edge_coeffs(...)`.
- `README.md` uses `an.write_parquet_graphdir` and `an.read_parquet_graphdir`;
  current exports are `to_parquet` and `from_parquet`.
- `README.md` imports `annnet.adapters.cx2_adapter`; current CX2 module is
  `annnet/io/cx2_io.py` and top-level `to_cx2`/`from_cx2`.
- `README.md` and `docs/quickstart.md` use `an.annnet.write/read`; current
  top-level exports are `an.write/read`, and `AnnNet.write/read` methods exist.
- `architecture.md` examples use `read_graphml`, `write_graphml`, and
  `read_json`; current public names are `from_graphml`, `to_graphml`, and
  `from_json`.

## 6. Dependencies

### Runtime dependencies

Declared required runtime dependencies:

- `numpy`: matrix/numeric arrays, IO, PyG conversion, metadata helpers, tests.
- `scipy`: sparse incidence matrix and derived sparse matrix operations.
- `narwhals`: dataframe abstraction in core annotations, views, IO, and helpers.

Declared optional dependencies and observed use:

- `polars`: preferred dataframe backend for attributes, tests, IO, CSV, CX2,
  Parquet helpers, and dataframe exports.
- `pandas`: fallback dataframe backend and Excel support.
- `pyarrow`: Parquet IO support, dataframe tests.
- `networkx`: runtime adapter/proxy, GraphML/GEXF IO via NetworkX.
- `python-igraph`: igraph adapter/proxy.
- `torch`, `torch-geometric`: PyG adapter output (`HeteroData`).
- `graph-tool`: graph-tool adapter/proxy via Conda/Pixi, not PyPI.
- `matplotlib`, `pydot`, `graphviz`: plotting/rendering utilities.
- `openpyxl`: Excel loading through pandas.
- `lxml`: listed for SBML extra, but `annnet/io/SBML_io.py` attempts
  `libsbml`; this extra may not fully represent the actual SBML runtime need.
- `zarr`, `numcodecs`: native `.annnet` storage.
- `cobra`: COBRA/SBML metabolic model conversion.
- `toml`: metadata loading according to `io` extra; current metadata code reads
  `pyproject.toml` via `tomllib`/importlib mechanisms rather than obvious
  direct `toml` use in inspected files.

### Development/test/docs dependencies

- Dev: `distlib`, `pre-commit`, `bump2version`, `twine`, `ruff`, `black`,
  `mypy`.
- Tests: `pytest`, `pytest-cov`, `tox`, `tox-gh`, `coverage`, `codecov-cli`,
  `diff_cover`, `polars`, `pandas`, `networkx`, `python-igraph`, `pyarrow`,
  `zarr`, `numcodecs`, `matplotlib`, `narwhals`.
- Docs: `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`,
  `mkdocs-jupyter`, `mkdocs-autorefs`.
- Benchmarks additionally reference `psutil` in `benchmarks/harness/metrics.py`,
  but `psutil` is not visible in the main dependency groups inspected.

### Potentially unnecessary, overlapping, or tightly coupled dependencies

- `black` is listed, but CI uses Ruff formatting, and docs mention Black. This
  is not necessarily wrong but is overlapping.
- Pixi configuration appears duplicated between `pixi.toml` and `pyproject.toml`.
- GraphML/GEXF IO imports NetworkX at module import time, so `from annnet.io.GraphML_io`
  requires optional NetworkX even though top-level lazy loading avoids it until
  the symbol is accessed.
- PyG adapter imports `torch`/`torch_geometric` at module import time, making
  that module hard optional rather than soft optional.
- `annnet/adapters/_base.py` imports `polars` at module import time even though
  the base class is only a small abstract adapter contract.

## 7. Documentation vs Implementation

### Implemented

The following design statements are supported by implementation:

- `AnnNet` is the main graph object (`annnet/core/graph.py`, `annnet/__init__.py`).
- Topology is incidence-matrix based (`self._matrix` in `AnnNet.__init__`).
- Structural records are `EntityRecord` and `EdgeRecord` (`annnet/core/_helpers.py`).
- Attribute tables are separate from structural records (`_Annotation.py`,
  `graph.py` initialization).
- Slices are named membership overlays (`_slices`, `_Slices.py`).
- Multilayer state exists and is extensive (`_Layers.py`).
- Lazy backend accessors exist as `G.nx`, `G.ig`, `G.gt` (`graph.py` properties,
  `core/backend_accessors/`).
- Native `.annnet` read/write exists (`annnet/io/io_annnet.py`,
  `AnnNet.write`, `AnnNet.read`).
- Explanation docs under `docs/explanations/internal-representation.md` align
  well with current canonical stores.

### Documented/intended

The docs describe a broader user-facing polish layer:

- simpler class alias `Graph`
- scalar hyperedge convenience method `add_hyperedge`
- scalar helper aliases for vertex/parallel/edge-entity operations
- top-level namespace `an.annnet`
- older IO names such as `read_graphml`/`write_graphml`
- an intended division between runtime backend adapters and format/data adapters

### Missing pieces

Missing or unclear relative to docs:

- No `Graph` alias in `annnet.__all__`.
- No scalar `add_hyperedge`, despite docs and history hook list.
- No `add_vertices`, `add_parallel_edge`, `add_edge_entity`, or
  `set_hyperedge_coeffs` on `AnnNet`.
- No `annnet.annnet` submodule exposed by top-level package.
- No `write_parquet_graphdir` or `read_parquet_graphdir` top-level functions.
- No `annnet.adapters.cx2_adapter` module; CX2 lives under `annnet/io/cx2_io.py`.
- `_write_cache` is a placeholder in native IO.

### Divergences and outdated/aspirational sections

- `README.md` is partly aspirational/stale and should not be used as exact API
  documentation today.
- `architecture.md` contains older data model examples with dict fields such as
  `entity_to_idx` and `edge_definitions` as if they were primary stores. Current
  code exposes these as compatibility mappings over structured records.
- `docs/quickstart.md` is closer than README for construction (`an.AnnNet`),
  but still uses `add_hyperedge` and `an.annnet`.
- `docs/explanations/architecture-overview.md` and
  `docs/explanations/internal-representation.md` are the best-aligned design
  docs for current internals.

## 8. Testing and Quality Signals

### Test structure and coverage signals

The test suite is broad:

- Core graph behavior: `tests/test_graph.py`, `tests/test_edge_cases.py`
- Views: `tests/test_graph_views.py`
- Slices/history: `tests/test_history.py`, slice tests embedded in graph tests
- Traversal: `tests/test_traversal.py`
- IO formats: JSON, Parquet, GraphML/GEXF, SIF, CSV, Excel, native `.annnet`,
  dataframe, CX2, SBML, SBML/COBRA
- Backend adapters and accessors: NetworkX, igraph, graph-tool, PyG, backend accessor tests
- Cross-adapter/integration/performance: `tests/test_cross_adapter.py`,
  `tests/test_integration.py`, `tests/test_performance.py`
- Public API: `tests/test_public_api.py`
- Plotting: `tests/test_plotting.py`

Local verification:

- `pytest -q` failed immediately because the shell environment has no `pytest`.
- `./.venv/bin/python -m pytest -q` also failed because `.venv` has no `pytest`.
- `./.venv/bin/ruff check annnet tests --output-format=concise` passed.
- `./.venv/bin/python -c "import annnet; print(annnet.__all__)"` succeeded on
  Python 3.13.12 and confirmed current top-level exports.

### Type hints / static analysis posture

- Type hints are present but inconsistent. Many public methods are untyped or
  partially typed.
- Ruff selects `ANN` annotation rules, but many annotation failures are likely
  avoided by existing configuration/state or ignored paths; local Ruff passed.
- `mypy` is listed in dev dependencies, but no `[tool.mypy]` configuration was
  found in inspected `pyproject.toml`.

### Linting / formatting setup

- Ruff is the active CI linter and formatter.
- `black` is listed in dev dependencies and docs, but Ruff format is used in CI.
- Ruff target version is Python 3.12 while package requires Python >=3.10 and CI
  tests Python 3.11-3.13.

### CI signals

- Unit CI installs tests dependency group and PyG dependencies, then runs
  `pytest --cov=annnet tests/`.
- Graph-tool tests run via Pixi with `graph-tool` from conda-forge.
- Docs deploy on pushes to main/master.
- Security scan runs Bandit with selected skips.

### Areas that appear untested or fragile

Evidence-based fragile areas:

- `annnet/adapters/manager.py` and `annnet/adapters/_proxy.py` appear stale and
  are not obviously exercised by current tests.
- `_write_cache` path in `annnet/io/io_annnet.py` is placeholder logic.
- README/quickstart examples are not tested as executable docs; otherwise stale
  method names would fail.
- Optional dependency import boundaries are not uniformly protected.
- Tests frequently assert private structures, which gives useful regression
  coverage but makes internal refactors harder.
- Local test execution depends on environment setup; the checked-in `.venv`
  lacked pytest.

## 9. Refactoring-Relevant Findings

### Large or overloaded modules

- `annnet/core/_Layers.py`: largest file and highest complexity concentration.
  It covers layer registry, Kivela operations, subgraph/slice creation, supra
  matrices, tensors, random walks, diffusion, spectral calculations, modularity,
  and coupling generation.
- `annnet/core/graph.py`: central object with initialization, scalar mutation,
  topology encoding, compatibility properties, backend accessor properties, storage
  wrappers, snapshots, and public conveniences.
- `annnet/core/_Annotation.py`: public attribute API plus dataframe type/schema
  handling plus flexible direction side effects.
- `annnet/core/_BulkOps.py`: performance-heavy mutation paths duplicating scalar
  semantics.
- `annnet/core/_Cache.py`: cache manager plus graph transformations and copy
  logic.
- `annnet/io/cx2_io.py`, `annnet/adapters/igraph_adapter.py`, and
  `annnet/adapters/networkx_adapter.py` are large format/backend modules with
  manifest logic and graph reconstruction logic embedded.

### Duplication

- Manifest serialization/deserialization ideas recur across `annnet/adapters/_utils.py`,
  NetworkX, igraph, graph-tool, JSON, SIF, Parquet, and CX2 modules.
- Dataframe row conversion helpers exist in multiple places:
  `_helpers.py`, `adapters/_utils.py`, `io/io_annnet.py`, `io/Parquet_io.py`,
  and `io/dataframe_io.py`.
- Direction/edge-weight helpers are repeated in adapter and IO modules.
- Stashed/alternate bulk/annotation files duplicate live implementations and
  should be classified before refactoring.

### Naming inconsistencies

- `AnnNet` vs documented `Graph`.
- `to_parquet`/`from_parquet` vs documented `write_parquet_graphdir`/
  `read_parquet_graphdir`.
- `set_edge_coeffs` vs documented `set_hyperedge_coeffs`.
- Hyperedge creation through `add_edge` vs documented `add_hyperedge`.
- CX2 under `annnet.io.cx2_io` vs README import from `annnet.adapters.cx2_adapter`.
- Uppercase module filenames (`GraphML_io.py`, `Parquet_io.py`, `SBML_io.py`,
  `SIF_io.py`) mixed with lowercase modules.
- `edge_directed` compatibility property vs `directed` field on `EdgeRecord`.

### Dead code or probable dead abstractions

Probable, requiring confirmation:

- `annnet/adapters/manager.py` and `annnet/adapters/_proxy.py` are likely older
  abstractions superseded by `core/backend_accessors/`.
- `GraphAdapter` in `annnet/adapters/_base.py` is not the primary adapter
  interface used by current exported functions.
- `_BulkOps_stashed.py`, `_BulkOps_before_edge_insert_opt.py`, and
  `_Annotation_stashed.py` appear to be work-in-progress or historical files.
  They are untracked in git status and should not be refactored as production
  code until their status is decided.
- `annnet/io/registry.py` is a one-line file and appears unused from inspected
  references.
- Script entry points in `pyproject.toml` point to `annnet:main` and
  `annnet.demo.demo:main`, but no `main` in `annnet/__init__.py` and no
  `annnet/demo/` package were visible in the file inventory.

### Hidden complexity

- The matrix, records, adjacency indices, slice membership, attribute tables,
  history version, and backend accessor caches must all remain coherent after
  mutation.
- Multilayer graphs have two representations at once: canonical entity keys
  and legacy compatibility structures like `_V`, `_VM`, `_nl_to_row`, and
  `_row_to_nl`.
- Bare vertex IDs in multilayer graphs can resolve to the first existing
  supra-node or a placeholder coordinate; this behavior is subtle and easy to
  break.
- Flexible direction policies depend on vertex attributes and can rewrite edge
  direction when attributes change.
- Compatibility mappings are mutable views into canonical stores, so they can
  mutate internals outside the normal scalar/bulk methods.
- Native IO can write archive mode or directory mode, and its `.annnet` suffix
  behavior depends on whether the path exists as a directory.

### Responsibilities blurred

- `AnnNet` is simultaneously model, mutation service, manager factory, IO facade,
  history owner, and backend proxy factory.
- `_Layers.py` mixes domain modeling with matrix algebra/analysis algorithms.
- IO modules often know detailed private internals instead of going through a
  stable serialization boundary.
- Tests validate internal fields directly, making it unclear which private
  fields are actually de facto public.

### Areas likely to resist refactoring

- Any change to `EdgeRecord` or `_edges` shape affects graph methods, adapters,
  IO formats, views, tests, and compatibility mappings.
- Any change to entity key resolution affects flat and multilayer behavior,
  adapters, IO round trips, and layer tests.
- Any change to slice membership semantics affects views, adapters, SIF/JSON/
  Parquet/CX2/native IO, and propagation behavior.
- Any dataframe backend refactor must handle Polars, pandas fallback, Narwhals,
  schema evolution, null handling, and list/dict serialization.

## 10. Recommended Next Steps

### Priority investigations before refactoring

1. Decide which documented stale APIs should be restored as aliases and which
   docs should be corrected. Start with `an.Graph`, `add_hyperedge`,
   `an.annnet`, and Parquet GraphDir naming.
2. Classify untracked/stashed files in `annnet/core/` as keep, delete, or move
   outside package source before broad refactoring.
3. Determine whether `annnet/adapters/manager.py`, `annnet/adapters/_proxy.py`,
   and `GraphAdapter` are supported API, deprecated compatibility, or dead code.
4. Run full CI-equivalent tests in a prepared environment, including optional
   dependency matrices. Local pytest was unavailable during this pass.
5. Map the invariant contract for `AnnNet` mutation: what must be updated after
   every vertex/edge/slice/layer mutation.
6. Audit native `.annnet` read/write with small graphs covering hyperedges,
   slices, edge-entities, multilayer state, history, and attrs.
7. Confirm entry points `an` and `an-demo` are real or remove/fix them.

### Safe AI-assisted refactoring entry points

- Documentation/API alignment:
  - fix README/quickstart examples or add small compatibility aliases if the
    project wants those names public
  - add tests for any restored aliases
- Public facade cleanup:
  - keep `annnet/__init__.py`, `annnet/io/__init__.py`, and
    `annnet/adapters/__init__.py` consistent with implemented functions
- Optional dependency boundaries:
  - move optional imports inside functions for GraphML/GEXF, PyG, adapter base,
    and plotting where practical
- Test harness and environment:
  - ensure a single documented local test command works with `uv` or Pixi
  - reconcile duplicate Pixi config
- Mechanical extraction with low semantic risk:
  - consolidate dataframe row conversion helpers after adding focused tests
  - consolidate manifest serialization helpers where tests already cover round
    trips

### Areas requiring extra caution

- `add_edge`, `_parse_edge_inputs`, edge column allocation, and adjacency index
  maintenance in `graph.py`
- `_resolve_entity_key` and multilayer placeholder semantics
- `_Layers.py` supra matrix/tensor methods and coupling helpers
- `_Annotation.py` dtype/schema upsert logic and flexible direction side effects
- compatibility mappings in `_helpers.py`
- native `.annnet` read/write reconstruction paths
- adapter round-trip behavior for hyperedges, edge-entities, slices, and
  multilayer metadata

## Refactoring Reconnaissance Summary

1. `AnnNet` is the real package center: one large object owns topology,
   overlays, attributes, history, managers, views, and backend accessors.
2. The current structural source of truth is record-based (`EntityRecord`,
   `EdgeRecord`) plus sparse incidence matrix, not the older dict fields shown
   in parts of `architecture.md`.
3. The implemented package is functionally broad but alpha-grade: strong test
   intent, but several stale docs, compatibility shims, and unfinished-looking
   modules remain.
4. The highest-risk refactors are around `add_edge`, multilayer entity keys,
   slices, dataframe attribute upserts, and IO/adapter round trips.
5. The safest first refactors are API/docs alignment, optional import cleanup,
   dead-code classification, and small helper consolidation with tests.
6. The docs under `docs/explanations/` are useful evidence for current intended
   internals; README and older `architecture.md` contain contradictions and
   should not be treated as authoritative.
7. Tests cover many user-visible workflows, but they also lock in private
   internals, which both helps regression detection and raises refactor cost.
8. Adapter/accessor architecture has two generations: current `core/backend_accessors`
   and older `adapters/manager.py`/`_proxy.py`; this needs a deliberate decision.
9. Before large structural changes, establish an invariant checklist and run the
   full optional-dependency test matrix in a complete environment.
10. The repository is ready for incremental refactoring, but not for a broad
    rewrite without first stabilizing public API expectations and dead-code
    boundaries.
