# Parquet

Parquet graph-directory helpers from `annnet.io.parquet`.

The public functions are `to_parquet(...)` and `from_parquet(...)`. The Parquet
reader and writer remain format-specific, but intermediate
annotation tables are built through AnnNet's centralized dataframe backend
helpers instead of choosing Polars or pandas locally.

::: annnet.io.parquet
    options:
      filters: public
      show_root_heading: true
