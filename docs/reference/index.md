# API Reference

This section is curated around how you use annnet in practice. Use the left navigation
(or search) to jump directly to a concept or symbol.

## How It Is Organized

- **Annotated Network**: the `AnnNet` core class plus the attribute tables and
  topology/indexing mixins that define the primary API surface.
- **Slices**: subgraph membership and slice-specific attributes.
- **Layers**: multilayer semantics (aspects, layer tuples, and per-layer attributes).
- **Algorithms**: traversal helpers and lazy proxy hooks for optional backends.
- **Storage (.annnet)**: lossless on-disk storage and Parquet GraphDir.
- **Interoperability**: adapters and file-format IO (NetworkX, igraph, graph-tool, PyG,
  SBML, SIF, GraphML, CX2, JSON/CSV/Excel, etc.).

## Navigation Tips

- Each page documents **public members** (names that do not start with `_`).
- Advanced or internal pieces are explicitly called out when they are part of the
  public surface (e.g., lazy proxies).
- If you cannot find a symbol, check the section that matches its domain or use search.
