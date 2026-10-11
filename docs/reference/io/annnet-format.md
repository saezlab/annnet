# Native .annnet format

The native format keeps everything an `AnnNet` object holds: the structure,
the eight attribute tables, slices, layers and metadata. A file is a directory
or an archive of [Zarr](https://zarr.dev/) arrays and [Parquet](https://parquet.apache.org/)
tables. The same functions are available as `annnet.read`, `annnet.write` and
`anet.write`.

::: annnet.io.annnet_format.read
    options:
      show_root_heading: true

::: annnet.io.annnet_format.write
    options:
      show_root_heading: true
