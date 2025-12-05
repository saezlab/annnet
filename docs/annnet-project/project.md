# Project

annnet provides a high‑expressivity, annotated graph container with:

- Flexible topology: binary edges and hyperedges, directed/undirected on a per‑edge basis, multiedges, self‑loops, vertex–edge and edge–edge relations.
- Multilayer structures: layers, aspects, inter‑layer edges.
- Slicing: define and switch between multiple graph “views”.
- Annotated tables: Polars‑backed DataFrames for vertices, edges, layers, slices, and metadata.
- Interoperability: NetworkX/igraph/graph‑tool (lazy proxies), plus broad file formats.
- Stable storage: lossless on‑disk `.annnet` format (Zarr + Parquet + JSON).

## Package layout

```
annnet/
├── core/         # Graph class, managers, lazy proxies (nx/ig/gt)
├── adapters/     # Backend adapters (nx/ig/gt) and format adapters (GraphML/GEXF/SIF/SBML/CX2/Parquet/JSON/DataFrames)
├── io/           # Lossless .annnet storage (read/write) + Excel/CSV helpers
├── algorithms/   # Pure‑Python algorithms using core only
└── utils/        # Utilities (typing/validation/config)
```

See the architecture overview for design details, data structures, and IO layout.

## Goals
- Interoperability‑first: integrate with existing ecosystems.
- Zero‑loss persistence: save and reload without losing structure.
- Expressive but practical: handle complex structures while staying approachable.
- Familiar ergonomics: anndata‑inspired annotations and workflows.

## Scope
- Supported runtimes: Python 3.10+.
- Optional backends: NetworkX, igraph, graph‑tool (if installed).
- Formats: GraphML, GEXF, SIF, SBML, CX2, Parquet GraphDir, JSON/NDJSON, Excel/CSV, DataFrames (via Narwhals).
