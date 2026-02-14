# Interoperability

This section covers in-memory adapters and file-format IO used to exchange
AnnNet graphs with external tools and formats.

## In-Memory Backends

### NetworkX (optional)

The NetworkX adapter requires the `networkx` extra. Public entry points:

- `to_nx(graph, ...)`
- `from_nx(G, ...)`
- `from_nx_only(G, ...)`
- `to_backend(graph, ...)`
- `save_manifest(manifest, path)`
- `load_manifest(path)`

See `annnet.adapters.networkx_adapter` in source for full details.

### igraph (optional)

::: annnet.adapters.igraph_adapter
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.adapters.igraph_adapter
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### graph-tool (optional)

::: annnet.adapters.graphtool_adapter
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.adapters.graphtool_adapter
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### PyTorch Geometric (optional)

The PyG adapter requires `torch` + `torch_geometric`. Public entry point:

- `to_pyg(graph, ...)`

See `annnet.adapters.pyg_adapter` in source for full details.

### Adapter Manager

::: annnet.adapters.manager
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.adapters.manager
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

## File Formats

### GraphML / GEXF (via NetworkX, optional)

The GraphML/GEXF helpers require NetworkX. Public entry points:

- `to_graphml(graph, path, ...)`
- `from_graphml(path, ...)`
- `to_gexf(graph, path, ...)`
- `from_gexf(path, ...)`

See `annnet.io.GraphML_io` in source for full details.

### SIF

::: annnet.io.SIF_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.SIF_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### SBML

::: annnet.io.SBML_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.SBML_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### SBML (COBRA)

::: annnet.io.sbml_cobra_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.sbml_cobra_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### CX2

::: annnet.io.cx2_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.cx2_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### JSON / NDJSON

::: annnet.io.json_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.json_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### CSV

::: annnet.io.csv_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.csv_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### Excel

::: annnet.io.excel
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.excel
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

### DataFrames (Polars/Pandas/Narwhals)

::: annnet.io.dataframe_io
    options:
      members: true
      show_root_heading: true
      filters:
      - "!^_"

???+ note "Internal helpers"
    ::: annnet.io.dataframe_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"
