# DataFrames

DataFrame conversion helpers from `annnet.io.dataframes`.

AnnNet accepts Narwhals-compatible eager dataframe inputs. When AnnNet creates
new dataframe outputs, the default backend is selected centrally in preference
order: Polars, pandas, then PyArrow. Pass `annotations_backend` to `AnnNet` when
you need a specific backend for one graph, or use `set_default_dataframe_backend`
to configure the process-wide default for new graphs.

::: annnet.io.dataframes
    options:
      filters: public
      show_root_heading: true
