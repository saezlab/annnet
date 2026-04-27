# Tracking changes

annnet records mutations automatically and provides snapshots and diffs for auditability.

## Automatic mutation logging

Every mutating method is wrapped to append an event with timestamp, monotonic clock, operation name, payload, and result.

Example event:

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
G.history.enable(True)   # toggle logging
G.history.clear()        # clear in‑memory history buffer
G.history.mark("checkpoint")  # insert a labeled marker
```

Export:

```python
df = G.history(as_df=True)           # Polars/pandas DataFrame view
n = G.history.export("log.parquet")  # or .ndjson / .json / .csv
```

Tips:
- Use `G.history.mark(label)` to insert named checkpoints that segment the log for later analysis.
- Export history alongside `.annnet` files for full provenance.
- Filter events by `op` key to isolate specific mutation types.

