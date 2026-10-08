# Guide

AnnNet keeps graph structure, annotations, and analysis contexts in one object.
These five notebooks explain how those parts work together. Each topic includes
the concepts, relevant mathematics, working examples, and links to the API.

Start with the [quickstart](../quickstart.md), then read the topics you need:

| Topic | What you will learn |
| --- | --- |
| [Graph model and matrices](graph-model.ipynb) | Node and edge identity, hyperedges, incidence coefficients, adjacency, and traversal. |
| [Annotations, selections, and views](annotations-and-views.ipynb) | Where attributes live, table access, filtering, live views, and materialized graphs. |
| [Slices and analysis contexts](slices.ipynb) | Keep a prior network and condition-specific selections and results together. |
| [Layers](layers.ipynb) | Aspects and placements, importing a multilayer graph, measurements, coupling, supra-matrices, and dynamics. |
| [Computation and storage](computation-and-storage.ipynb) | Run backend algorithms, attach results, choose an export, save an analysis, and understand history limits. |

**Slices select existing identities; layers place identities in explicit
coordinates.** Use slices when two analyses select or annotate the same
interactions. Use layers when an entity's placement in a condition, mechanism,
or timepoint is itself part of the graph. The two can be used together.

## Run the notebooks

Install the package with the dependencies used by the guide:

```bash
python -m pip install 'annnet[polars,networkx,storage]' jupyterlab
```

Alternatively, use the [conda environment](environment.yml). Download a notebook
using its source link and put the three [sample CSV files](data/index.md) in a
`data/` directory beside it. From a repository checkout, the notebooks also find
`docs/guide/data` automatically. Run cells in order; each notebook imports its
own graph and can be run independently.

The data are small and synthetic, so these notebooks require no external data
service. For longer analyses with biological data or other software, see
[Examples](../examples/index.md).
