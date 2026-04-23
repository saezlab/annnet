# AnnNet Package Overview

This document describes the current shape of the AnnNet package from an
implementation and maintenance perspective: it records what the package
does today, how the main pieces fit together, and which design choices
explain the current code layout.

For installation and first-use examples, see `README.md` and the documentation
site. For exact public signatures, use the API reference in `docs/reference/`.

## Purpose

AnnNet is an annotated graph container for scientific graph data that needs more
structure than a plain binary graph:

- directed and undirected edges in the same object
- parallel edges and self-loops
- hyperedges, including directed head/tail hyperedges
- edge-entities that can participate as endpoints
- slices for named contexts or scenarios
- multilayer and multi-aspect graph state
- dataframe-backed annotations
- controlled conversion to graph libraries and file formats

The central idea is to keep a high-fidelity AnnNet object as the source of truth
and project it into other representations only when needed.

## Package Layout

The repository is organized around one core graph object and a set of
interoperability, storage, algorithm, and utility modules.

```text
annnet/
  core/                   AnnNet graph class, mixins, managers, views, caches
  core/backend_accessors/ graph-owned G.nx / G.ig / G.gt dispatch accessors
  adapters/               in-memory conversion to/from graph backends
  io/                     file, table, and native .annnet storage formats
  algorithms/             algorithms operating on AnnNet objects
  utils/                  plotting, validation, typing, and small helpers
  _dataframe_backend.py   centralized dataframe creation/conversion helpers
  _plotting_backend.py    plotting backend selection
  _optional_components.py optional dependency registry
  _metadata.py            package metadata and environment summary helpers
```

The public package root (`annnet/__init__.py`) is mostly a lazy export layer. It
keeps import cost low while exposing the main graph class, selected metadata
helpers, backend configuration helpers, IO helpers, and adapter helpers.

## Core Object

`AnnNet` is implemented in `annnet/core/graph.py`. It is a single class composed
from mixins that keep related behavior in separate files:

- `BulkOps`: batched vertex, edge, hyperedge, and slice operations
- `Operations`: copy, reverse, subgraph, and matrix-oriented operations
- `History`: mutation history, snapshots, and diffs
- `ViewsClass`: dataframe-style and graph views
- `IndexMapping`: low-level ID and key helpers
- `LayerClass`: multilayer and multi-aspect behavior
- `SliceClass`: slice behavior
- `AttributesClass`: annotation upserts and lookups
- `Traversal`: graph traversal helpers

This split is mostly organizational. The runtime object is still one `AnnNet`
instance with shared state.

## Structural Model

AnnNet uses a sparse incidence matrix plus registries that map matrix positions
back to graph objects.

Current structural stores:

```text
_matrix         scipy.sparse.dok_matrix
_entities       entity key -> EntityRecord
_edges          edge id -> EdgeRecord
_row_to_entity  matrix row -> entity key
_col_to_edge    matrix column -> edge id
```

Rows represent entities. An entity can be a normal vertex or an edge-entity.
Columns represent edges. Directed incidence uses positive coefficients for
sources or heads and negative coefficients for targets or tails. Undirected
edges and undirected hyperedges use positive coefficients for all incident
entities.

The record objects currently carry structural meaning:

- `EntityRecord` stores row identity and entity kind.
- `EdgeRecord` stores endpoints or member sets, weight, directedness, structural
  edge type, incidence column, and multilayer metadata.

The matrix and records should be understood together: the matrix stores
incidence coefficients, while records store semantic information that is not
recoverable from the matrix alone.

## Derived State

AnnNet keeps several derived structures for lookup speed, layer handling, and
backend conversion:

- `_adj`
- `_src_to_edges`
- `_tgt_to_edges`
- `_V`
- `_VM`
- `_nl_to_row`
- `_row_to_nl`
- `_csr_cache`
- backend caches behind `G.nx`, `G.ig`, and `G.gt`
- compatibility mapping objects derived from the entity and edge registries

These structures are implementation state, not independent graph models. The
mutation paths update cheap derived indices eagerly and rebuild expensive
materializations lazily.

## Slices, Layers, and Views

Slices are named contexts over one graph. They track vertex memberships, edge
memberships, slice attributes, and edge-slice overrides such as context-specific
weights. They let one graph represent multiple conditions or scenarios without
duplicating topology.

Layers model structured multilayer graph state. AnnNet supports aspects,
elementary layers, layer tuples, vertex presence in layer tuples, intra-layer
edges, inter-layer edges, coupling edges, and supra-matrix views.

Views provide lightweight filtered access to part of a graph. Subgraph and copy
operations materialize new graph objects when a standalone object is needed.

## Annotation Tables

AnnNet stores annotations in dataframe-like tables rather than in per-object
dictionaries. Important tables include:

- `vertex_attributes`
- `edge_attributes`
- `slice_attributes`
- `edge_slice_attributes`
- `layer_attributes`

The `obs`, `var`, and `uns` properties provide familiar access to vertex
attributes, edge attributes, and graph-level metadata.

Dataframe creation and conversion is centralized in `annnet/_dataframe_backend.py`.
AnnNet accepts Narwhals-compatible eager dataframe inputs and uses a configured
concrete backend when it needs to create new tables. The supported dataframe
backends are Polars, pandas, and PyArrow.

## IO

The `annnet.io` namespace contains storage and exchange formats. Native AnnNet
storage is exposed as:

```python
annnet.io.write(...)
annnet.io.read(...)
```

The native `.annnet` format stores sparse structure, annotation tables, and
metadata using a combination of array, Parquet, and JSON files.

Other IO modules cover:

- JSON and NDJSON
- DataFrames
- CSV and Excel
- SIF
- GraphML and GEXF
- CX2
- Parquet graph directories
- SBML and SBML/COBRA imports
- OmniPath imports

IO modules are format boundaries. Some formats preserve AnnNet-specific
structure better than others, so conversion options and manifests are important
for round-tripping richer graphs.

## Adapters and Backend Accessors

AnnNet separates explicit in-memory conversion from graph-owned algorithm
dispatch.

Explicit conversion lives in `annnet.adapters`:

```python
annnet.adapters.to_nx(...)
annnet.adapters.from_nx(...)
annnet.adapters.to_igraph(...)
annnet.adapters.from_igraph(...)
annnet.adapters.to_graphtool(...)
annnet.adapters.from_graphtool(...)
annnet.adapters.to_pyg(...)
```

Graph-owned accessors live on each `AnnNet` instance:

```python
G.nx
G.ig
G.gt
```

These accessors lazily project the AnnNet graph to the selected backend, cache
compatible projections, dispatch backend algorithms, and map vertex identifiers
back where supported.

NetworkX, igraph, graph-tool, and PyTorch Geometric support are optional.
graph-tool is not installed from PyPI and is handled through the repository's
Pixi `gt` environment for tests.

## Optional Components

Optional dependency status and backend selection are centralized rather than
duplicated across modules:

- `_optional_components.py` defines named optional component registries.
- `_dataframe_backend.py` selects and builds dataframe backends.
- `_plotting_backend.py` selects plotting backends.
- `_metadata.py` exposes package metadata and environment summaries.

The user-facing entry points are available from `annnet`, for example:

```python
annnet.info()
annnet.available_dataframe_backends()
annnet.set_default_dataframe_backend(...)
annnet.available_plot_backends()
annnet.set_default_plot_backend(...)
```

## Development Tooling

Project metadata and dependency declarations are in `pyproject.toml`.

Key tooling:

- Hatchling is the build backend.
- `uv` is used for local dependency groups and command execution.
- dependency groups are `dev`, `tests`, and `docs`.
- Ruff is configured for linting/import rules and formatting style.
- pytest and pytest-cov are used for tests and coverage.
- tox defines CI-like lint, test, readme, coverage, and docs environments.
- Pixi defines the `gt` environment for graph-tool tests.

Common local commands:

```bash
uv sync --group tests
uv run pytest

uv sync --group docs
uv run python -m mkdocs build --strict

pixi run -e gt test-gt
pixi run -e gt test-all
```

## Documentation

The docs are MkDocs Material pages under `docs/`. The current structure is:

- installation and quickstart
- tutorial/use-case notebooks
- explanation pages for concepts and design
- API reference pages
- contributor/community pages

The API reference documents the public surface and includes selected low-level
core helper pages. Direct imports from underscore-prefixed modules remain
internal implementation details unless explicitly documented otherwise.
