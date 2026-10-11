# SIF

Read and write the
[Simple Interaction Format](https://manual.cytoscape.org/en/stable/Supported_Network_File_Formats.html#sif-format)
(SIF): one line per interaction, with a source, an interaction type and a
target. A graph feature that SIF cannot hold, such as a hyperedge, a slice or
an attribute, goes into a sidecar file beside it. See
[What each format keeps](format-comparison.md).

::: annnet.io.sif.from_sif
    options:
      show_root_heading: true

::: annnet.io.sif.to_sif
    options:
      show_root_heading: true
