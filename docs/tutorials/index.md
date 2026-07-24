# Notebook Gallery

This section collects runnable AnnNet notebooks with rendered outputs. To
recreate them, use the linked GitHub environment file for the notebook family.

Base HowTos use the
[HowTo environment](https://github.com/saezlab/annnet/blob/main/docs/tutorials/notebooks/environment.yml).
Scenario notebooks each link a narrower environment for the external packages
they demonstrate. Use-case notebooks are coming soon.

## HowTos

Small, self-contained notebooks that show AnnNet APIs directly, including
focused special-topic notebooks.

<div class="grid cards annnet-feature-cards" markdown>

-   __Quickstart__

    ---

    Build a directed graph, inspect it, and round-trip it through `.annnet`.

    [Open notebook](notebooks/tutos/01_quickstart.ipynb)

-   __Attributes and views__

    ---

    Attach metadata to vertices and edges, then inspect dataframe-like views.

    [Open notebook](notebooks/tutos/02_attributes_and_views.ipynb)

-   __Tables and storage__

    ---

    Build from tables and compare CSV, Excel, native, Parquet, JSON, and NDJSON.

    [Open notebook](notebooks/tutos/03_tables_and_storage.ipynb)

-   __Slices and subgraphs__

    ---

    Manage context-specific graph state and materialize smaller graphs.

    [Open notebook](notebooks/tutos/04_slices_and_subgraphs.ipynb)

-   __Hyperedges and traversal__

    ---

    Represent complexes/reactions and traverse local neighborhoods.

    [Open notebook](notebooks/tutos/05_hyperedges_and_traversal.ipynb)

-   __Directed hyperedges and stoichiometry__

    ---

    Work directly with directed hyperedges, incidence coefficients, and
    reaction-style semantics.

    [Open notebook](notebooks/special/sp01_directed_hyperedges.ipynb)

-   __Multilayer__

    ---

    Work with layers, coupling edges, layer-derived slices, and supra matrices.

    [Open notebook](notebooks/tutos/06_multilayer.ipynb)

-   __Multilayer math__

    ---

    Inspect supra-adjacency, Laplacians, diffusion, coupling sweeps, and tensor views.

    [Open notebook](notebooks/special/sp02_multilayer_math.ipynb)

-   __History and reproducibility__

    ---

    Record mutations, create snapshots, diff graph states, and export history.

    [Open notebook](notebooks/tutos/07_history_and_reproducibility.ipynb)

-   __Backend accessors__

    ---

    Inspect optional components and dispatch to installed graph backends.

    [Open notebook](notebooks/tutos/08_backend_accessors.ipynb)

-   __Flexible edge orientation__

    ---

    Compare edge-scope and vertex-scope orientation policies on mixed graphs.

    [Open notebook](notebooks/special/sp03_flexible_edge_orientation.ipynb)

</div>

## Use Cases

Larger applied notebooks that combine AnnNet with external biological data,
optimization, and graph learning workflows.

<div class="grid cards annnet-feature-cards" markdown>

-   __Multi-condition causal signaling__

    ---

    Build patient layers from CPTAC/OmniPath data and write inferred causal
    subnetworks back into AnnNet.

    Coming soon.

-   __HEK293 heterogeneous biology graph__

    ---

    Combine signaling, metabolic, complex, regulatory, organelle, and
    graph-learning workflows in one AnnNet object.

    Coming soon.

</div>

## Scenarios

Short ecosystem bridges. They use tiny deterministic data and keep external
dependencies in scenario-specific environment files.

<div class="grid cards annnet-feature-cards" markdown>

-   __OmniPath table ingestion__

    ---

    Load an OmniPath-style interaction table as prior knowledge without a network call.

    [Open notebook](notebooks/scenarios/omnipath_table_ingestion.ipynb) ·
    [Environment](https://github.com/saezlab/annnet/blob/main/docs/tutorials/notebooks/envs/omnipath_table_ingestion.yml)

-   __Cytoscape CX2 export__

    ---

    Export annotated and hyperedge graphs with explicit CX2 projection modes.

    [Open notebook](notebooks/scenarios/cytoscape_cx2_export.ipynb) ·
    [Environment](https://github.com/saezlab/annnet/blob/main/docs/tutorials/notebooks/envs/cytoscape_cx2_export.yml)

-   __PyG HeteroData export__

    ---

    Prepare typed graph data and numeric features for PyTorch Geometric.

    [Open notebook](notebooks/scenarios/pyg_heterodata_export.ipynb) ·
    [Environment](https://github.com/saezlab/annnet/blob/main/docs/tutorials/notebooks/envs/pyg_heterodata_export.yml)

-   __Causal activity bridge__

    ---

    Store activity scores and causal solution edges from decoupler/CORNETO-style workflows.

    [Open notebook](notebooks/scenarios/causal_activity_bridge.ipynb) ·
    [Environment](https://github.com/saezlab/annnet/blob/main/docs/tutorials/notebooks/envs/causal_activity_bridge.yml)

</div>
