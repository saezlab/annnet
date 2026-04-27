# Analysis Patterns

Most AnnNet workflows fall into a small number of practical patterns.

## Multilayer Analysis

Use this when layers are part of the scientific model, for example modalities,
states, or timepoints.

- Define aspects and layer tuples.
- Add intra-layer, inter-layer, or coupling edges.
- Analyze through AnnNet's multilayer operators or through a backend accessor.

Related pages: [Multilayer and multi-aspect graphs](../explanations/math-multilayer.md),
[Layers](../reference/core/layers.md), and [Graph](../reference/core/graph.md).

Notebooks: [Multilayer Systems Biology](notebooks/SBUC.ipynb) and
[HEK293 multilayer network](notebooks/UC2.ipynb).

## Hyperedges and Stoichiometry

Use this when binary edges are too limiting and you need directed hyperedges,
weighted memberships, or SBML-style reaction structure.

- Import or construct hyperedges.
- Preserve endpoint-specific coefficients when needed.
- Export through a format that can preserve the structure, or choose an
  explicit projection when moving to simpler graph tools.

Related pages: [Incidence representation](../explanations/math-incidence.md),
[Graph](../reference/core/graph.md), and [SBML](../reference/io/sbml.md).

Notebook: [AnnNet Showcase](notebooks/annnet_showcase.ipynb).

## Slices and Scenario Management

Use this when you need multiple contexts in one container, such as train/eval
splits, perturbation scenarios, or condition-specific subgraphs.

- Define slices.
- Switch the active slice when constructing or inspecting context-specific
  state.
- Compare memberships, weights, or annotations across slices.

Related pages: [Slices and views](../explanations/managers-and-views.md),
[Slices](../reference/core/slices.md), and [Views](../reference/core/views.md).

## Interoperability-First Analysis

Use this when AnnNet is the main container and you want to move into other
tools without losing track of AnnNet-specific structure.

- Use `G.nx`, `G.ig`, or `G.gt` for graph-owned backend algorithm dispatch.
- Use `annnet.adapters.to_*` and `annnet.adapters.from_*` for explicit
  in-memory conversion.
- Use the native `.annnet` format when AnnNet should remain the system of
  record.

Related pages: [Interoperability](../explanations/interoperability.md),
[Backend accessors](../reference/core/backend-accessors.md), and
[Storage and IO](../explanations/io-annnet.md).
