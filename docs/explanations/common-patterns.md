# Typical analysis patterns

Most workflows in annnet fall into a small number of patterns.

## Multilayer analysis

Use this when layers are part of the scientific model, for example modalities, states, or timepoints.

- Typical flow: define aspects, add layer tuples, add intra-layer and inter-layer edges, then analyze through annnet's multilayer operators or an external backend.
- Read first: [Multilayer and multi-aspect graphs](../explanations/math-multilayer.md).
- Useful API pages: [Layers](../reference/core/layers.md), [Graph](../reference/core/graph.md).
- Notebooks: [SBUC](../tutorials/notebooks/SBUC.ipynb), [HEK293 multilayer network](../tutorials/notebooks/UC2.ipynb).

## Hyperedges and stoichiometry

Use this when binary edges are too limiting and you need directed hyperedges, weighted memberships, or SBML-style reaction structure.

- Typical flow: import SBML, inspect hyperedges, preserve stoichiometric weights, then analyze or export.
- Read first: [Incidence representation](../explanations/math-incidence.md).
- Useful API pages: [Graph](../reference/core/graph.md), [SBML](../reference/io/sbml.md).
- Notebook: [AnnNet Showcase](../tutorials/notebooks/annnet_showcase.ipynb).

## Slices and scenario management

Use this when you need multiple contexts in one container, such as train/eval splits, perturbation scenarios, or condition-specific subgraphs.

- Typical flow: define slices, switch the active slice, override per-slice weights or annotations, then compare results.
- Read first: [Slices and views](../explanations/managers-and-views.md).
- Useful API pages: [Slices](../reference/core/slices.md), [Views](../reference/core/views.md).
- Also useful: [Tracking changes](../explanations/history-and-diffs.md).

## Interoperability-first analysis

Use this when annnet is the main container and you want to move into other tools without losing structure.

- Typical flow: convert with `an.to_nx` or `an.to_igraph`, compute or edit, restore with the manifest, then save as `.annnet`.
- Read first: [Interoperability](../explanations/interoperability.md).
- Useful API pages: [NetworkX adapter](../reference/adapters/networkx.md), [igraph adapter](../reference/adapters/igraph.md), [Native .annnet format](../reference/io/annnet-format.md).
- For persistence and exchange formats: [Storage and IO](../explanations/io-annnet.md).

## Notebook picks

- Start with [AnnNet Showcase](../tutorials/notebooks/annnet_showcase.ipynb) if you want the broadest introduction.
- Use [Multilayer Systems Biology](../tutorials/notebooks/SBUC.ipynb) if multilayer biology is your main use case.
- Use [SBUC without Polars](../tutorials/notebooks/SBUC-nopolars.ipynb) if you want the same example in a lighter environment.
- Use [AnnNet Demo](../tutorials/notebooks/Demo.ipynb) if you want a shorter hands-on notebook.

For a broader tour of examples, open the [Notebook gallery](../tutorials/index.md).
