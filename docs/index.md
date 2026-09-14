<div class="annnet-hero">
  <div class="annnet-kicker">Typed graph data for complex networks</div>
  <h1>annnet</h1>
  <p class="annnet-lead">
    annnet (Annotated Network) is a unified, high‑expressivity graph platform that brings anndata‑style, annotated containers to networks, multilayer structures, and hypergraphs. It targets systems biology, network biology, omics integration, computational social science, and any domain needing fully flexible graph semantics with modern, stable storage and interoperability.
  </p>
</div>

## Why annnet

Many real-world networks are structurally heterogeneous and context-dependent. A single dataset may combine
directed and undirected interactions, signed edges, higher-order relations, and multiple contextual or temporal
conditions. Standard graph libraries handle topology and algorithms well, but typically treat attributes as flat,
per-object dictionaries without schema, indexing, or efficient bulk operations. This makes it difficult to manage
annotations, compare conditions, or preserve structure across analysis steps.

annnet addresses this by defining a single container that keeps graph topology, annotation tables, and graph views
aligned. The goal is not to replace existing graph libraries, but to provide a consistent data model that can express
complex networks and still interoperate with established tooling.

![annnet unifies rich graph semantics, annotated tables, and lossless storage.](assets/annnet_fig1_layout.png)

At the core is a sparse incidence-based representation that supports mixed graph types within the same object.
Edges are first-class entities with stable identifiers, and graph type (directed, undirected, hyperedge) is a
property of each edge rather than the container. Around this core, annnet organizes metadata as typed tables and
exposes higher-level constructs such as slices and multilayer structure without duplicating the underlying graph.

<div class="grid cards annnet-feature-cards" markdown>

-   __One object for heterogeneous graphs__

    ---

    Represent simple graphs, digraphs, signed edges, hyperedges, self-loops, parallel edges, and edge-entity relations without switching data models.

-   __Typed annotation tables__

    ---

    Store node, edge, slice, layer, and edge-slice metadata in indexed tabular structures instead of flat per-object dictionaries.

-   __Named slices for conditions and views__

    ---

    Define condition-specific or context-specific graph views without copying the full topology, with optional per-slice edge-weight overrides.

-   __Multilayer network support__

    ---

    Work with aspects, layer tuples, node-layer membership, intra-layer edges, inter-layer edges, and coupling structure as part of the core model.

-   __Interoperability without losing the source of truth__

    ---

    Convert lazily to existing graph libraries when needed, while keeping annnet as the canonical representation.

-   __Native, loss-aware storage__

    ---

    Persist topology, annotations, and metadata together in a format designed for round-trip fidelity and scalable IO.

</div>

## What annnet is built for

annnet is most useful in settings where graph structure is only one part of the data model, and where annotations,
conditions, or multiple representations must be handled explicitly. The design follows patterns that have proven
useful in other domains (for example, matrix-plus-annotation containers in omics, in AnnData), but adapts them to
graphs with heterogeneous topology.

Instead of encoding these requirements through ad hoc conventions or multiple loosely coupled objects, annnet
keeps them within a single, consistent representation that can still be exported or adapted when needed.

<div class="grid cards annnet-feature-cards" markdown>

-   __Systems biology and omics integration__

    ---

    Model regulatory, signaling, metabolic, and cross-modal networks with typed metadata and multiple analysis contexts.

-   __Condition-specific and temporal networks__

    ---

    Keep one shared graph with named slices for perturbations, time points, cohorts, or filtered analytical views.

-   __Multimodal and layered data__

    ---

    Represent networks across modalities, resolutions, or time as explicit multilayer objects rather than ad hoc conventions.

-   __Exchange with existing tooling__

    ---

    Move between annnet and standard graph ecosystems, file formats, and ML pipelines without flattening the original structure too early.

</div>

## Documentation

The documentation is split by purpose: short setup, conceptual explanations,
runnable notebooks, and exact reference pages. Choose one route below and move
to another section only when you need it.

<div class="grid cards annnet-feature-cards" markdown>

-   __Tutorials and notebooks__

    ---

    End-to-end examples showing graph construction, annotation workflows, slices, multilayer models, and interoperability.

    [Open tutorials](tutorials/index.md)

-   __Concepts and design__

    ---

    Explanations of the incidence-matrix core, annotation system, slices, multilayer formalism, adapters, and storage model.

    [Open explanations](explanations/index.md)

-   __API reference__

    ---

    Detailed reference for the object model, bulk APIs, IO, utilities, and public entry points.

    [Open reference](reference/index.md)

-   __Community__

    ---

    Contribution guidance for documentation, package development, and project standards.

    [Open community pages](community/index.md)

</div>

## Recommended order

1. [Install annnet](installation.md).
2. Build one graph with the [Quickstart](quickstart.md).
3. Read the [Explanations](explanations/index.md) page that matches your model.
4. Run a matching example from the [Notebook Gallery](tutorials/index.md).
5. Consult the [API reference](reference/index.md) for exact signatures.

<div class="annnet-hero">
  <div class="annnet-kicker">Get started</div>
  <h2>Choose the next useful page</h2>
  <p class="annnet-lead">
    Use the quickstart for the core loop, explanations for the model, notebooks
    for runnable workflows, and the reference for exact API details.
  </p>
  <div class="annnet-actions">
    <a class="md-button md-button--primary" href="installation/">Installation</a>
    <a class="md-button" href="quickstart/">Quickstart</a>
    <a class="md-button" href="explanations/">Explanations</a>
    <a class="md-button" href="tutorials/">Notebook Gallery</a>
  </div>
</div>
