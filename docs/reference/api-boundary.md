# API overview

Import annnet as:

```python
import annnet as an
```

All names below are in the top-level `annnet` namespace, unless the name shows
a subpackage. A name that is not on this page or on the page of
its object is internal: for example a name that begins with an underscore, or a
module such as `annnet.core._store`. Internal names can change without notice.

## Graph

| name | description |
|---|---|
| [`AnnNet`](core/graph.md) | The graph object. `Graph` is the same class. |
| [`GraphView`](core/views.md) | A live, read-only view of a graph, from `anet.view(...)`. |
| [`annnet.core.NodeSequence`, `annnet.core.EdgeSequence`](core/views.md) | The nodes and edges, `anet.N` and `anet.E`, and the selections made from them. |
| [`NodeView`, `EdgeView`](core/graph.md) | One node or one edge, from `anet.N.at(id)` and `anet.E.at(id)`. |
| [`Endpoint`, `as_endpoint`, `as_endpoints`](core/graph.md#endpoint) | One side of an edge: the node and its layer. |
| [`Provenance`](core/graph.md#provenance) | The sources that a graph was built from, `anet.provenance`. |

## Attributes

| name | description |
|---|---|
| [`Attrs`](core/attributes.md) | All attributes of a graph, `anet.attrs`. |
| [`RowSelection`](core/attributes.md) | The rows that `anet.attrs.select` and `anet.attrs.from_frame` select. |
| [`Schema`](core/attributes.md) | The keys and fields of one attribute address. |

## Slices and layers

| name | description |
|---|---|
| [`anet.slices`](core/slices.md) | Named subsets of the edges and nodes. |
| [`anet.layers`](core/layers.md) | Aspects, layers and node placements of a multilayer graph. |
| [`Aspect`, `OrderedLabels`, `as_aspect`](core/layers.md#aspects) | One aspect and the order of its labels. |
| [`LayerSelection`](core/layers.md#layer-selection) | The layers that one window names. |
| [`ValueMatrix`, `MatrixValues`, `ValueResolver`](core/layers.md#node-layer-values) | Values of nodes on layers, held as arrays. |

## Other namespaces of the graph

| name | description |
|---|---|
| [`anet.nx`, `anet.ig`, `anet.gt`](core/backend-accessors.md) | Algorithms of NetworkX, igraph and graph-tool on the graph. |
| [`anet.ops`](core/bulk-operations.md) | Copy, subgraph, union, intersection, difference. |
| [`anet.idx`](core/indexing.md) | IDs to matrix positions, and back. |
| [`anet.cache`](core/cache.md) | Cached sparse matrices. |
| [`anet.history`](core/history.md) | A log of changes, and snapshots. |

## Reading and writing

| name | description |
|---|---|
| [`read`, `write`](io/annnet-format.md) | The native `.annnet` format. |
| [`from_edge_frame`, `add_edges_from_frame`, `edge_id_for`](io/dataframes.md) | A graph from a table of edges. |
| [`from_dataframes`, `to_dataframes`](io/dataframes.md) | A graph as a set of tables. |
| [`from_parquet`, `to_parquet`](io/parquet.md) | A directory of Parquet tables. |
| [`from_csv`, `from_dataframe`, `to_csv`, `edges_to_csv`, `hyperedges_to_csv`](io/csv.md) | CSV files. |
| [`from_excel`, `to_excel`](io/excel.md) | Excel workbooks. |
| [`from_json`, `to_json`, `read_ndjson`, `write_ndjson`](io/json.md) | JSON and NDJSON. |
| [`from_graphml`, `to_graphml`, `from_gexf`, `to_gexf`](io/graphml-gexf.md) | GraphML and GEXF files. |
| [`from_sif`, `to_sif`](io/sif.md) | Simple Interaction Format. |
| [`from_sbml`, `to_sbml`, `from_sbml_cobra`, `from_cobra_model`](io/sbml.md) | SBML models. |
| [`from_cx2`, `to_cx2`](io/cx2.md) | Cytoscape CX2. |
| [`from_pyg`, `to_pyg`](io/pyg.md) | PyTorch Geometric. |

[What each format keeps](io/format-comparison.md) compares the formats.

## Other graph libraries

| name | description |
|---|---|
| [`from_nx`, `to_nx`](adapters/networkx.md) | NetworkX graphs. |
| [`from_igraph`, `to_igraph`](adapters/igraph.md) | igraph graphs. |
| [`from_graphtool`, `to_graphtool`](adapters/graphtool.md) | graph-tool graphs. |
| `available_backends` | The graph libraries that are installed. |

## Algorithms

| name | description |
|---|---|
| [`sources`, `targets`, `directed_pairs`, `directed_cycle`](algorithms/traversal.md#structure) | Facts about the directed edges. |

## Plotting

| name | description |
|---|---|
| [`annnet.utils.plot`, `annnet.utils.save`, `annnet.utils.show_cytoscape`](utils/plotting.md) | Draw a graph, write the drawing to a file, or show the graph as an interactive network. |

## Package and settings

| name | description |
|---|---|
| [`info`, `__version__`, `get_latest_version`](support.md#package-metadata) | The package version and its optional components. |
| [`set_default_dataframe_backend` and others](support.md#dataframe-backends) | The dataframe library for new tables. |
| [`set_default_plot_backend` and others](support.md#plotting-backends) | The plotting library. |

## Errors

annnet raises the standard Python exceptions, with a message that names the
problem:

| situation | exception |
|---|---|
| An unknown node, edge, slice, aspect, layer or attribute key. | `KeyError` |
| A key or argument of the wrong shape or type. A predicate that compares values that cannot be compared. A node selection given where edges are expected. | `TypeError` |
| A reserved field. A key that occurs two times in one batch. A missing or null key column. `delete` without `keys` and without `names`. An ordering operator on a categorical aspect. A selection from another graph. A bad `boundary`, `limit`, `layout` or operator. | `ValueError` |
| A write through a view. | `ReadOnlyViewError`, a subclass of `TypeError` |
| A name that annnet removed. | `AttributeError`, with the name to use instead |
