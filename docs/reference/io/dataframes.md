# DataFrames

Use `annnet.from_edge_frame` to import a table with source, target, and optional
edge-ID and annotation columns. The [quickstart](../../quickstart.md) imports a
CSV this way. `to_dataframes` / `from_dataframes` exchange a graph as a bundle of
tables and metadata; keep the complete bundle for a round trip.

AnnNet accepts Narwhals-compatible eager dataframe inputs. When AnnNet creates
new dataframe outputs, the default backend is selected centrally in preference
order: Polars, pandas, then PyArrow. Pass `annotations_backend` to `AnnNet` when
you need a specific backend for one graph, or use `set_default_dataframe_backend`
to configure the process-wide default for new graphs.

::: annnet.io.dataframes
    options:
      filters: public
      show_root_heading: true

::: annnet.io.edge_frame
    options:
      filters: public
      show_root_heading: true
