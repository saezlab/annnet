# Quickstart: import and explore a network

Start with an interaction table, inspect what was imported, select a useful
subset, and save the annotated graph. This example uses a small
[synthetic teaching dataset](guide/data/index.md): 11 interactions among
10 nodes, including two separately identified EGFR–GRB2 records.

## Load the table

Install the table and storage dependencies:

```bash
pip install "annnet[polars,storage]"
```

From a repository checkout, run the Python examples from the repository root.
Alternatively, download [interactions.csv](guide/data/interactions.csv) into
your working directory and set `DATA = Path('.')` below.

```python
from pathlib import Path
import annnet as an
import polars as pl

DATA = Path('docs/guide/data')
interactions = pl.read_csv(DATA / 'interactions.csv')
interactions.head()
```

The columns are `edge_id`, `source`, `target`, `effect`, `confidence`, and
`evidence`. Map the identifier and effect columns explicitly:

```python
G = an.from_edge_frame(
    interactions,
    edge_id='edge_id',
    sign='effect',
    directed=True,
    slice='prior',
)
G.attrs.backend = 'polars'
G.summary()
```

You should see 10 nodes, 11 edges, and a `prior` slice containing the imported
network. AnnNet creates the endpoint nodes and preserves both EGFR–GRB2 edges.
The `effect` column becomes the `sign` annotation. Confidence remains an
annotation; structural weights default to 1.

## Inspect and select

```python
G.attrs.table('edges', derived=True).select('edge_id', 'source', 'target', 'sign', 'confidence')
```

Which interactions have confidence at least 0.85?

```python
supported = G.E.select(confidence__gte=0.85)
V = G.view(edges=supported)
print(len(V.N), len(V.E))  # 8 nodes, 8 edges
V.attrs.table('edges', derived=True)
```

`V` is a live, read-only view. It follows changes to the parent graph and the
selection. To keep a separate graph for further editing, use
`H = V.materialize()`.

## Add an annotation

Write through `G.attrs` when you want the graph to change:

```python
G.attrs.update('nodes', {'ERK': {'label': 'ERK response'}})
G.uns['dataset'] = 'AnnNet synthetic tutorial network'
G.attrs.row('nodes', 'ERK')
```

Tables and rows returned by `G.attrs` are detached reads. Editing a returned
dataframe alone does not update AnnNet.

## Save and reopen

```python
G.write('analysis.annnet')
restored = an.read('analysis.annnet')
print(len(restored.N), len(restored.E))  # 10, 11
restored.attrs.row('nodes', 'ERK')
```

The native format retains identities, annotations, and slice membership. The
writer refuses an existing destination unless you pass `overwrite=True`.

Continue with [Annotations, selections, and views](guide/annotations-and-views.ipynb)
for filtering and materialization, or [Slices](guide/slices.ipynb) to attach
contextual results. The [Guide](guide/index.md) also covers the graph model,
layers, and computation and storage.
