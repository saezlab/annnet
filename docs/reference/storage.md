# Disk Storage (.annnet)

The `.annnet` format is a lossless on-disk representation built on Zarr +
Parquet + JSON. This page documents the primary read/write API.

## AnnNet Format

::: annnet.io.io_annnet.write
    options:
      show_root_heading: true

::: annnet.io.io_annnet.read
    options:
      show_root_heading: true

???+ note "Internal helpers"
    ::: annnet.io.io_annnet
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"

## Parquet GraphDir (Lossless)

::: annnet.io.Parquet_io.to_parquet
    options:
      show_root_heading: true

::: annnet.io.Parquet_io.from_parquet
    options:
      show_root_heading: true

???+ note "Internal helpers"
    ::: annnet.io.Parquet_io
        options:
          members: true
          show_root_heading: false
          filters:
          - "^_"
