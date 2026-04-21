# Storage and IO

annnet distinguishes between native storage and exchange IO.

## Native storage

The native `.annnet` format is the high-fidelity persistence format for annnet
objects. Its job is to preserve the graph as annnet understands it, including:

- topology
- annotations
- slices
- multilayer state
- metadata and provenance
- history-related data where available

The format mirrors the in-memory model instead of flattening it into a simpler
external graph shape.

## Why the native format is structured this way

The native layout combines different storage technologies because different
parts of the graph have different access patterns.

- Sparse graph structure is stored in Zarr-friendly array form.
- Annotation tables are stored in Parquet-friendly columnar form.
- Metadata and descriptors are stored in JSON.

That split keeps the format both faithful to annnet and practical for the wider
Arrow, Parquet, and Zarr ecosystem.

When the native writer materializes intermediate annotation tables, it uses
annnet's centralized dataframe backend selection instead of choosing Polars or
pandas locally. The storage format is still Parquet-based, but in-memory table
construction follows the configured annotation backend.

## Storage goals

The native format is designed to support:

- faithful round-tripping of annnet state
- typed, columnar attribute storage
- large graphs and chunked access
- cloud-friendly object-store layouts
- forward-compatible schema evolution

## IO beyond the native format

The `annnet.io` package also supports exchange with other representations, such
as:

- GraphML and GEXF
- SIF
- SBML and SBML/COBRA-derived graphs
- CX2
- JSON and NDJSON
- CSV and Excel
- DataFrame-based import and export
- Parquet GraphDir

These formats are useful for interoperability, but they do not all preserve the
full annnet model equally well.

## Storage versus exchange

This distinction is important:

- Use native `.annnet` storage when annnet is the system of record.
- Use exchange formats when you need to interoperate with another tool,
  workflow, or tabular environment.

If the graph contains hyperedges, slices, edge-entities, multilayer structure,
or rich annotation state, the native format is usually the only format that can
preserve the whole object faithfully.

## Where to look next

- For the conceptual meaning of interoperability, read [Interoperability](interoperability.md).
- For exact file and function entry points, use the [IO reference](../reference/io/annnet-format.md).
