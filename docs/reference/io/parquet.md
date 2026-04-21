# Parquet

Parquet GraphDir helpers from `annnet.io.Parquet_io`.

The Parquet reader and writer remain format-specific, but intermediate
annotation tables are built through AnnNet's centralized dataframe backend
helpers instead of choosing Polars or pandas locally.

::: annnet.io.Parquet_io
    options:
      filters: public
      show_root_heading: true
