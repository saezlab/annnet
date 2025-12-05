# Architecture overview

This page summarizes the core design from `architecture.md`.

## Core data model
- Incidence matrix (V×E) as the primary structure (sparse DOK in‑memory, stored as COO). Each edge is a column; direction uses sign (+ for head/source, − for tail/target). Undirected edges use + on all incident vertices.
- Unified entity system: vertex rows and edge‑entities (for vertex–edge relations) share the same row space with `entity_to_idx`, `idx_to_entity`, and `entity_types`.
- Edge system: `edge_to_idx`, `idx_to_edge`, `edge_definitions`, per‑edge `weight`, and `directed` flags. Parallel edges are separate columns.

## Direction and hyperedges
- Direction control at three levels: graph default, per‑edge override, mixed mode.
- Hyperedges: undirected (set membership) or directed (head→tail) with explicit `hyperedge_definitions` and `edge_kind`.
- SBML stoichiometry: per‑endpoint coefficients written directly into incidence columns.

## Slices
- Named subgraph memberships with a current active slice for new elements.
- Set algebra (union/intersection/difference) and per‑slice edge attributes (e.g., weight overrides).

## Multilayer networks (Kivelä)
- Aspects and elementary layers define the layer space (cartesian product). Vertex‑layer presence tracked in `V×M` set; edges can be intra‑layer, inter‑layer, or coupling.
- Supra‑adjacency and supra‑Laplacian available; a 4‑index tensor view provides a higher‑order representation.

## Attributes (Polars DataFrames)
- Separate, columnar tables for vertex, edge, slice, and edge‑slice attributes. Structural keys are filtered; only pure user attributes go into tables.
- Upsert semantics with automatic schema growth and type inference; read‑only views for quick inspection.

## Interoperability
- Runtime backend adapters:
  - NetworkX via `G.nx` and `to_nx`/`from_nx`
  - igraph via `G.ig` and `to_igraph`/`from_igraph`
  - graph‑tool via `G.gt` and `to_graphtool` (if installed)
- Conversion options: `directed`, `hyperedge_mode` (skip/expand), slice selection, and simple mode (collapse multiedges). Proxies may slim attributes to those required for algorithms.
- Manifests: returned on conversion to improve round‑trip fidelity on import (hyperedges, slices, multigraph collapsing).

## Storage layout (.annnet)
- Directory structure combines Zarr for arrays and Parquet for tables, plus JSON sidecars. Key paths:
  - `manifest.json` with counts, versions, and slice metadata
  - `structure/incidence.zarr/{row,col,data}` and index maps as Parquet
  - `tables/*` for attributes; `layers/` and `slices/` for multilayer and slice data
  - `audit/` for history; optional `cache/` for CSR/CSC; `uns/` for unstructured results

## History and diffs
- Automatic mutation logging with export to Parquet/JSON/CSV/NDJSON, named snapshots, and diffs between snapshots.

## Spectral methods
- Supra adjacency/Laplacian, transition matrices, eigenvalue routines, diffusion and random walk helpers, and coupling regime sweeps.

For full details, see the top‑level `architecture.md` in the repository.

