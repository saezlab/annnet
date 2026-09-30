# Views and selections

The live, read-only view from `annnet.core._Views` and the typed selections
from `annnet.core._select`. Use `G.view(...)` for view workflows and `G.N` /
`G.E` for selections. Direct imports from underscore modules follow the
[internal API policy](../api-boundary.md); the public names are exported as
`annnet.GraphView`, `annnet.core.NodeSequence` and `annnet.core.EdgeSequence`.

The resolution rules — closed and open boundaries, hyperedge integrity,
edge-entity closure, nested restriction — are stated in
[the API contract](../api-contract.md), section 5.

::: annnet.core._Views.GraphView
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.NodeSequence
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.EdgeSequence
    options:
      filters: public
      show_root_heading: true

::: annnet.core._select.ReadOnlyViewError
    options:
      show_root_heading: true
