# Bulk Operations

Bulk editing helpers now live on the main graph object and operation namespace,
not in a separate `annnet.core._BulkOps` module.

Use `AnnNet` bulk methods such as `add_vertices_bulk(...)` and
`add_edges_bulk(...)`, plus the `G.ops` namespace for copy/materialization
operations. Direct imports from underscore modules follow the
[internal API policy](../api-boundary.md).

::: annnet.core.graph.AnnNet
    options:
      members:
        - add_vertices_bulk
        - add_edges_bulk
      show_root_heading: true
