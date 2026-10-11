# GraphML and GEXF

Read and write [GraphML](http://graphml.graphdrawing.org/) and
[GEXF](https://gexf.net/) files. Both go through
[NetworkX](https://networkx.org/), so NetworkX must be installed. A graph
feature that the format cannot hold goes into a sidecar file beside it. See
[What each format keeps](format-comparison.md).

::: annnet.io.graphml.from_graphml
    options:
      show_root_heading: true

::: annnet.io.graphml.to_graphml
    options:
      show_root_heading: true

::: annnet.io.graphml.from_gexf
    options:
      show_root_heading: true

::: annnet.io.graphml.to_gexf
    options:
      show_root_heading: true
