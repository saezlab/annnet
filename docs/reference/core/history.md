# History

Change tracking helpers from `annnet.core._History`.

Snapshots compare node, edge, and slice ID sets; they do not capture attribute values
or reconstruct a prior graph. The event log currently omits some mutations,
including edge removal and attribute updates. Save a native `.annnet` artifact
for an analysis checkpoint. See the executed
[history example](../../guide/computation-and-storage.ipynb).

History methods are mixed into `AnnNet`. Direct imports from underscore modules
follow the [internal API policy](../api-boundary.md).

::: annnet.core._History.GraphDiff
    options:
      filters: public
      show_root_heading: true

::: annnet.core._History.History
    options:
      filters: public
      show_root_heading: true

::: annnet.core._History.HistoryAccessor
    options:
      filters: public
      show_root_heading: true
