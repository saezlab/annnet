# DataFrames

Build a graph from a table of edges, and exchange a whole graph as a set of
tables.

`from_edge_frame` makes a graph from a table with one row per edge: a source
column, a target column, and optional columns for the edge ID and the edge
attributes. `add_edges_from_frame` adds such a table to a graph that exists.
When the table has no edge IDs, `edge_id_for` makes one from what the edge is:
its endpoints, its place and its attributes. So a sorted copy of the same table
gives the same IDs. The [quickstart](../../quickstart.ipynb) reads a table this
way.

`to_dataframes` gives a dictionary of tables, with the metadata that is not a
table beside them. `from_dataframes` rebuilds the graph. Keep the complete
dictionary for a round trip.

annnet accepts any eager dataframe that [Narwhals](https://narwhals-dev.github.io/narwhals/)
supports. The tables that annnet makes use [Polars](https://pola.rs/),
[pandas](https://pandas.pydata.org/) or [PyArrow](https://arrow.apache.org/docs/python/),
the first one installed in this order. To use one library for one graph, give
`annotations_backend` to `AnnNet`. To change the default for all new graphs,
call `annnet.set_default_dataframe_backend`.

::: annnet.io.edge_frame.from_edge_frame
    options:
      show_root_heading: true

::: annnet.io.edge_frame.add_edges_from_frame
    options:
      show_root_heading: true

::: annnet.io.edge_frame.edge_id_for
    options:
      show_root_heading: true

::: annnet.io.dataframes.from_dataframes
    options:
      show_root_heading: true

::: annnet.io.dataframes.to_dataframes
    options:
      show_root_heading: true
