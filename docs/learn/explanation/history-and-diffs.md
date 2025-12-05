# History and diffs

annnet records mutations automatically and provides snapshots and diffs for auditability.

## Automatic mutation logging

Every mutating method is wrapped to append an event with timestamp, monotonic clock, operation name, payload, and result.

Example event shape:

```json
{
  "version": 42,
  "ts_utc": "2025-12-05T10:30:00.123456Z",
  "mono_ns": 123456789,
  "op": "add_vertex",
  "vertex_id": "alice",
  "attributes": {"age": 30},
  "result": "alice"
}
```

Controls:

```python
G.enable_history(True)   # toggle logging
G.clear_history()        # clear in‑memory history buffer
G.mark("checkpoint")     # insert a labeled marker
```

Export and views:

```python
df = G.history(as_df=True)          # Polars DataFrame view
n = G.export_history("log.parquet") # or .ndjson / .json / .csv
```

## Snapshots and diffs

Create labeled snapshots, then compute differences between them to see structural and attribute changes.

```python
snap1 = G.snapshot(label="before")
# ... perform mutations ...
snap2 = G.snapshot(label="after")

diff = G.diff("before", "after")
diff.added_vertices     # set[str]
diff.removed_vertices   # set[str]
diff.added_edges        # set[str]
diff.removed_edges      # set[str]
```

Tips:
- Use checkpoints/marks to segment logs for analysis.
- Export history alongside `.annnet` for provenance.
- Pair snapshots with `export_history()` to reconstruct change timelines.

