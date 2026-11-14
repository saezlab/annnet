# rustworkx adapter

Convert between an `AnnNet` object and a [rustworkx](https://www.rustworkx.org/)
graph. A rustworkx graph addresses nodes and edges by index and carries one
payload per node and per edge: `to_rustworkx` gives each node
`{'node_id': ..., **attributes}` and each edge
`{'edge_id': ..., 'weight': ..., 'directed': ..., **attributes}`, and sets
`rxG.attrs['index_of']` to the map from node id to index. It also gives back a
manifest with what rustworkx cannot hold, such as hyperedges, slices and
layers. Give the manifest to `from_rustworkx` to rebuild the full graph. To run
a rustworkx algorithm without converting yourself, use
[`anet.rx`](../core/backend-accessors.md).

```python
import rustworkx as rx
from annnet.adapters import to_rustworkx, from_rustworkx

rxG, manifest = to_rustworkx(anet)
lengths = rx.dijkstra_shortest_path_lengths(
    rxG, rxG.attrs['index_of']['EGFR'], edge_cost_fn=lambda e: e['weight']
)
back = from_rustworkx(rxG, manifest)
```

::: annnet.adapters.rustworkx_adapter.from_rustworkx
    options:
      show_root_heading: true

::: annnet.adapters.rustworkx_adapter.to_rustworkx
    options:
      show_root_heading: true
