# History

`anet.history` logs changes to the graph and takes snapshots that you can
compare.

A snapshot compares the sets of node, edge and slice IDs. It does not keep
attribute values, and it cannot restore an earlier graph. The log does not
record all changes: edge removal and attribute updates are not in it. To keep a
state of an analysis, write the graph to a native `.annnet` file. The
[computation guide](../../guide/computation-and-storage.ipynb) shows an example.

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
