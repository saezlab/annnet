# API Reference

Use this section when you need exact signatures, public classes, and module-level
entry points. See [Public and Internal APIs](api-boundary.md) for the stability
policy for documented public APIs and internal underscore-prefixed APIs.

## Reference layout

- `annnet.core`: graph structure, indexing, attributes, views, slices, layers, history, and cache helpers.
- `annnet.algorithms`: algorithmic helpers exposed by the package.
- `annnet.io`: storage and exchange formats.
- `annnet.adapters`: optional backend conversion helpers and backend availability checks.
- `annnet.utils`: plotting helpers.
- Support helpers: package metadata, optional component status, dataframe
  backend selection, and plotting backend selection.

## Recommended namespace use

- Use `annnet.AnnNet` or `annnet.Graph` to construct graphs.
- Use `annnet.io.read` and `annnet.io.write` for native `.annnet` storage.
- Use `annnet.io` for filesystem formats and tabular data.
- Use `annnet.adapters` for in-memory backend conversions.
- Use graph-owned accessors (`G.nx`, `G.ig`, `G.gt`) for backend algorithms.

