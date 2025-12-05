# annnet zero-loss serialization: Zarr + Parquet

## Design goals
1. **Zero topology loss**: Preserve exact incidence matrix, all edge types, hyperedges, parallel edges
2. **Complete metadata**: All attributes, slices, history, provenance
3. **Cross-platform**: Works on Windows/Linux/Mac, Python/R/Julia
4. **Incremental updates**: Can append without full rewrite
5. **Cloud support**: S3/GCS/Azure compatible via Zarr
6. **Fast random access**: Chunked storage for large graphs

---

## File Structure

```
graph.annnet/
├── manifest.json                 # root descriptor (format, counts, compression, etc.)
├── structure/
│   ├── incidence.zarr/           # Zarr v3 group holding COO [coordinate list] arrays
│   │   ├── zarr.json             # Zarr v3 group metadata (includes group attributes)
│   │   ├── row/                  # Zarr array (int32) of entity indices (COO row)
│   │   ├── col/                  # Zarr array (int32) of edge indices   (COO col)
│   │   └── data/                 # Zarr array (float32) of weights      (COO data)
│   │   # group attributes include: {"shape": [n_entities, n_edges]}
│   ├── entity_to_idx.parquet     # entity_id → row index
│   ├── idx_to_entity.parquet     # row index → entity_id
│   ├── entity_types.parquet      # entity_id → "vertex" | "edge"
│   ├── edge_to_idx.parquet       # edge_id → column index
│   ├── idx_to_edge.parquet       # column index → edge_id
│   ├── edge_definitions.parquet  # edge_id → (source, target, edge_type) for simple edges
│   ├── edge_weights.parquet      # edge_id → weight
│   ├── edge_directed.parquet     # edge_id → bool | null
│   ├── edge_kind.parquet         # edge_id → "binary" | "hyper" | "intra" | "inter"
│   └── hyperedge_definitions.parquet
│       # columns: edge_id, directed(bool), members(List[Utf8]) OR head(List[Utf8]), tail(List[Utf8])
│
├── tables/
│   ├── vertex_attributes.parquet      # vertex-level DF [dataframe]
│   ├── edge_attributes.parquet        # edge-level DF
│   ├── slice_attributes.parquet       # slice metadata
│   └── edge_slice_attributes.parquet  # (slice_id, edge_id, weight)
│
├── layers/                           # Kivela Multilayer Structures
│   ├── metadata.json                 # {"aspects": [...], "elem_layers": {...}}
│   ├── vertex_presence.parquet       # vertex_id, layer(List[str])
│   ├── edge_layers.parquet           # edge_id, layer_1(List[str]), layer_2(List[str]|null)
│   ├── elem_layer_attributes.parquet # attributes for elementary layers
│   ├── aspect_attributes.json        # attributes for aspects
│   ├── tuple_layer_attributes.parquet # layer(List[str]), attributes(JSON)
│   └── vertex_layer_attributes.parquet # vertex_id, layer(List[str]), attributes(JSON)
│
├── slices/                           # Subgraph Views
│   ├── registry.parquet              # slice_id, name, metadata…
│   ├── vertex_memberships.parquet    # (slice_id, vertex_id)
│   └── edge_memberships.parquet      # (slice_id, edge_id, weight)
│
├── cache/                            # optional materialized views
│   ├── csr.zarr/                     # CSR [compressed sparse row] cache
│   └── csc.zarr/                     # CSC [compressed sparse column] cache
│
├── audit/
│   ├── history.parquet               # operation log (nested payloads stringified to JSON [JavaScript Object Notation])
│   ├── snapshots/                    # optional labeled snapshots
│   └── provenance.json               # creation time, software versions, etc.
│
└── uns/                              # unstructured metadata & results
    ├── graph_attributes.json
    └── results/
```

---

## Manifest Schema (`manifest.json`)

```json
{
  "format": "annnet",
  "version": "1.1.0",
  "created": "2025-10-23T10:30:00Z",
  "annnet_version": "0.1.0",
  "graph_version": 42,
  "directed": true,
  "counts": {
    "vertices": 1000,
    "edges": 5000,
    "entities": 1050,
    "slices": 3,
    "hyperedges": 50,
    "aspects": 2
  },
  "slices": ["default", "temporal_2023", "temporal_2024"],
  "active_slice": "default",
  "default_slice": "default",
  "schema_version": "1.0",
  "checksum": "sha256:abcdef...",
  "compression": "zstd",
  "encoding": {
    "zarr": "v3",
    "parquet": "2.0"
  }
}
```

### Field mapping to in‑memory graph

- `format`, `version`: file format identifier and schema version.
- `created`: ISO timestamp of export.
- `annnet_version`: library version that wrote the data.
- `graph_version`: internal monotonic version counter (used by caches/proxies).
- `directed`: default directionality for edges when unspecified per‑edge.
- `counts`: sanity counts for fast checks; not authoritative for reading.
  - `vertices` ↔ number of entities with type `vertex` in `entity_types.parquet`.
  - `edges` ↔ number of columns in incidence (and rows in `edge_*` maps).
  - `entities` ↔ total rows in incidence and size of `entity_*` maps.
  - `slices` ↔ number of slices in `slices/registry.parquet`.
  - `hyperedges` ↔ entries in `structure/hyperedge_definitions.parquet`.
  - `aspects` ↔ number of aspects in `layers/metadata.json`.
- `slices`: ordered list of slice IDs; `active_slice` and `default_slice` indicate runtime state.
- `compression`: preferred codec for Parquet writing (e.g., `zstd`).
- `encoding`: explicit versions of Zarr and Parquet encodings used.

### Validation and compatibility

- Readers should not rely solely on `counts` for trust; derive truth from Parquet/Zarr sources.
- Unknown keys MUST be ignored to allow forward‑compatible schema evolution.
- If `schema_version` is older, treat missing sections as optional (e.g., no `layers/` folder → monolayer).
- If `active_slice` is not present in `slices/registry.parquet`, fall back to `default_slice` or `"default"`.
- Zarr root MUST carry `attrs.shape = [n_entities, n_edges]` for the incidence matrix.

## Advantages

1. **Zero loss**: topology + metadata round-trip exactly
2. **Portable**: Parquet/Zarr are first-class in Python/R/Julia
3. **Incremental**: replace just the parts you touched
4. **Cloud-native**: Zarr stores are compatible with S3/GCS/Azure
5. **Interoperable**: PaParquet works with Pandas/DuckDB/Arrow ecosystems
6. **Compressed**: zstd/lz4 where supported
7. **Chunked**: fast random access on large graphs
8. **Schema evolution**: add new tables without breaking old readers

---


