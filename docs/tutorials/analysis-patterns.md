# Analysis Patterns

Most AnnNet workflows fall into a small number of practical patterns. The
notebooks below are deliberately small scenarios rather than large case studies.

## Multilayer Analysis

Use this when layers are part of the model: conditions, modalities, timepoints,
or other aspects.

- Define aspects and elementary layers.
- Add intra-layer, inter-layer, or coupling edges.
- Analyze layer subgraphs, layer-derived slices, and supra matrices.

Related pages: [Multilayer and multi-aspect graphs](../explanations/math-multilayer.md),
[Layers](../reference/core/layers.md), and [Graph](../reference/core/graph.md).

Notebook: [Multilayer](notebooks/tutos/06_multilayer.ipynb).

## Hyperedges and Traversal

Use this when binary edges are too limiting and you need complexes, directed
hyperedges, endpoint coefficients, or neighborhood traversal.

- Construct undirected complexes and directed reaction-style hyperedges.
- Inspect incidence values when coefficients matter.
- Traverse successors, predecessors, and local neighborhoods.

Related pages: [Incidence representation](../explanations/math-incidence.md),
[Traversal](../reference/algorithms/traversal.md), and [SBML](../reference/io/sbml.md).

Notebook: [Hyperedges and traversal](notebooks/tutos/05_hyperedges_and_traversal.ipynb).

## Slices and Scenario Management

Use this for multiple contexts in one container: train/eval splits,
perturbation scenarios, confidence subsets, or condition-specific subgraphs.

- Define slices and inspect membership.
- Compare slice union, intersection, and difference.
- Materialize subgraphs for selected contexts.

Related pages: [Slices and views](../explanations/managers-and-views.md),
[Slices](../reference/core/slices.md), and [Operations](../reference/core/bulk-operations.md).

Notebook: [Slices and subgraphs](notebooks/tutos/04_slices_and_subgraphs.ipynb).

## Storage and Interchange

Use this when AnnNet sits between files, tables, and graph-specific formats.

- Load edge tables from dataframe, CSV, or Excel sources.
- Use native `.annnet` or Parquet for rich AnnNet state.
- Use JSON/NDJSON or graph formats for interchange boundaries.

Related pages: [Storage and IO](../explanations/io-annnet.md),
[DataFrames](../reference/io/dataframes.md), and [Native .annnet format](../reference/io/annnet-format.md).

Notebook: [Tables and storage](notebooks/tutos/03_tables_and_storage.ipynb).

## Ecosystem Bridges

Use these when AnnNet should connect graph state to external analysis tools.

- `AnnData`/scverse: graph state as `obs`, `var`, `X`, and `uns`.
- OmniPath-style tables: prior knowledge graphs from interaction data.
- CX2: Cytoscape export with explicit hyperedge projection.
- PyG: heterogeneous tensor data for GNN workflows.
- decoupler/CORNETO-style workflows: activity and causal solution tables as
  AnnNet attributes, slices, and layers.

Related pages: [Interoperability](../explanations/interoperability.md),
[Backend accessors](../reference/core/backend-accessors.md), and
[OmniPath](../reference/io/omnipath.md).

Notebooks:
[AnnData/scverse bridge](notebooks/scenarios/scverse_bridge.ipynb),
[OmniPath table ingestion](notebooks/scenarios/omnipath_table_ingestion.ipynb),
[Cytoscape CX2 export](notebooks/scenarios/cytoscape_cx2_export.ipynb),
[PyG HeteroData export](notebooks/scenarios/pyg_heterodata_export.ipynb), and
[Causal activity bridge](notebooks/scenarios/causal_activity_bridge.ipynb).

## Reproducibility

Use this when graph construction needs to be auditable.

- Clear or preserve history intentionally.
- Mark important construction stages.
- Snapshot and diff graph states.
- Export the mutation log.

Related page: [Tracking changes](../explanations/history-and-diffs.md).

Notebook: [History and reproducibility](notebooks/tutos/07_history_and_reproducibility.ipynb).
