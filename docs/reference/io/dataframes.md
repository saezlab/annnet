# DataFrames

DataFrame conversion helpers from `annnet.io.dataframe_io`.

AnnNet accepts Narwhals-compatible eager dataframe inputs. When AnnNet creates
new dataframe outputs, the default backend is selected centrally in preference
order: Polars, pandas, then PyArrow. Pass `annotations_backend` to `AnnNet` when
you need a specific backend for newly created annotation tables and dataframe
exports.

::: annnet.io.dataframe_io
    options:
      filters: public
      show_root_heading: true
