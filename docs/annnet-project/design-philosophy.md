# Design philosophy

annnet is designed with these principles:

- Simple, consistent interface for all graph types.
- Interoperability‑first: integrate, don’t replace.
- Performance‑aware, not performance‑obsessed.
- Extensible and modular, not monolithic.

## Architectural choices
- Separate structure from attributes: sparse incidence matrix for topology; attributes in Polars tables.
- Lazy interoperability: convert to NetworkX/igraph/graph‑tool only when needed (via proxies `G.nx`, `G.ig`, `G.gt`).
- Slices and views: copy‑on‑write patterns for subgraphs and state changes.
- Lossless storage: `.annnet` directories combine Zarr (arrays) and Parquet (tables), plus JSON sidecars.

See the architecture overview for a deeper dive into data layout, conversion manifests, hyperedge semantics, and multilayer networks.
