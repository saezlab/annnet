# CX2

Read and write the
[Cytoscape Exchange format, version 2](https://cytoscape.org/cx/cx2/specification/cytoscape-exchange-format-specification-(version-2)/)
(CX2) of [Cytoscape](https://cytoscape.org/). The structures that CX2 has no
place for, such as hyperedges, slices and layers, go into a manifest inside the
network attributes, so `from_cx2` can rebuild them. To show a graph as an
interactive Cytoscape.js network, use
[`show_cytoscape`](../utils/plotting.md#annnet.utils.plotting.show_cytoscape).

::: annnet.io.cx2.from_cx2
    options:
      show_root_heading: true

::: annnet.io.cx2.to_cx2
    options:
      show_root_heading: true
