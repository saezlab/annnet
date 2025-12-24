import inspect
import time
from datetime import UTC, datetime
from functools import wraps

import narwhals as nw
import numpy as np

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None


class GraphDiff:
    """Represents the difference between two graph states.

    Attributes
    --
    vertices_added : set
        Vertices in b but not in a
    vertices_removed : set
        Vertices in a but not in b
    edges_added : set
        Edges in b but not in a
    edges_removed : set
        Edges in a but not in b
    slices_added : set
        slices in b but not in a
    slices_removed : set
        slices in a but not in b

    """

    def __init__(self, snapshot_a, snapshot_b):
        self.snapshot_a = snapshot_a
        self.snapshot_b = snapshot_b

        # Compute differences
        self.vertices_added = snapshot_b["vertex_ids"] - snapshot_a["vertex_ids"]
        self.vertices_removed = snapshot_a["vertex_ids"] - snapshot_b["vertex_ids"]
        self.edges_added = snapshot_b["edge_ids"] - snapshot_a["edge_ids"]
        self.edges_removed = snapshot_a["edge_ids"] - snapshot_b["edge_ids"]
        self.slices_added = snapshot_b["slice_ids"] - snapshot_a["slice_ids"]
        self.slices_removed = snapshot_a["slice_ids"] - snapshot_b["slice_ids"]

    def summary(self):
        """Human-readable summary of differences."""
        lines = [
            f"Diff: {self.snapshot_a['label']} - {self.snapshot_b['label']}",
            "",
            f"Vertices: {len(self.vertices_added):+d} added, {len(self.vertices_removed)} removed",
            f"Edges: {len(self.edges_added):+d} added, {len(self.edges_removed)} removed",
            f"slices: {len(self.slices_added):+d} added, {len(self.slices_removed)} removed",
        ]
        return "\n".join(lines)

    def is_empty(self):
        """Check if there are no differences."""
        return (
            not self.vertices_added
            and not self.vertices_removed
            and not self.edges_added
            and not self.edges_removed
            and not self.slices_added
            and not self.slices_removed
        )

    def __repr__(self):
        return self.summary()

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "snapshot_a": self.snapshot_a["label"],
            "snapshot_b": self.snapshot_b["label"],
            "vertices_added": list(self.vertices_added),
            "vertices_removed": list(self.vertices_removed),
            "edges_added": list(self.edges_added),
            "edges_removed": list(self.edges_removed),
            "slices_added": list(self.slices_added),
            "slices_removed": list(self.slices_removed),
        }


class History:
    # History and Timeline

    def _utcnow_iso(self) -> str:
        return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def _jsonify(self, x):
        # Make args/return JSON-safe & compact.

        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (set, frozenset)):
            return sorted(self._jsonify(v) for v in x)
        if isinstance(x, (list, tuple)):
            return [self._jsonify(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._jsonify(v) for k, v in x.items()}
        # NumPy scalars
        if isinstance(x, (np.generic,)):
            return x.item()
        # Polars, SciPy, or other heavy objects -> just a tag
        t = type(x).__name__
        return f"<<{t}>>"

    def _log_event(self, op: str, **fields):
        if not self._history_enabled:
            return
        self._version += 1
        evt = {
            "version": self._version,
            "ts_utc": self._utcnow_iso(),  # ISO-8601 with Z
            "mono_ns": time.perf_counter_ns() - self._history_clock0,
            "op": op,
        }
        # sanitize
        for k, v in fields.items():
            evt[k] = self._jsonify(v)
        self._history.append(evt)

    def _log_mutation(self, name=None):
        def deco(fn):
            op = name or fn.__name__
            sig = inspect.signature(fn)

            @wraps(fn)
            def wrapper(*args, **kwargs):
                bound = sig.bind(*args, **kwargs)
                bound.apply_defaults()
                result = fn(*args, **kwargs)
                payload = {}
                # record all call args except 'self'
                for k, v in bound.arguments.items():
                    if k != "self":
                        payload[k] = v
                payload["result"] = result
                self._log_event(op, **payload)
                return result

            return wrapper

        return deco

    def _install_history_hooks(self):
        # Mutating methods to wrap. Add here if you add new mutators.
        to_wrap = [
            "add_vertex",
            "add_edge_entity",
            "add_edge",
            "add_hyperedge",
            "remove_edge",
            "remove_vertex",
            "set_vertex_attrs",
            "set_edge_attrs",
            "set_slice_attrs",
            "set_edge_slice_attrs",
            "register_slice",
            "unregister_slice",
        ]
        for name in to_wrap:
            if hasattr(self, name):
                fn = getattr(self, name)
                # Avoid double-wrapping
                if getattr(fn, "__wrapped__", None) is None:
                    setattr(self, name, self._log_mutation(name)(fn))

    def history(self, as_df: bool = False):
        """Return the append-only mutation history.

        Parameters
        --
        as_df : bool, default False
            If True, return a Polars DF [DataFrame]; otherwise return a list of dicts.

        Returns
        ---
        list[dict] or polars.DataFrame
            Each event includes: 'version', 'ts_utc' (UTC [Coordinated Universal Time]
            ISO-8601 [International Organization for Standardization]), 'mono_ns'
            (monotonic nanoseconds since logger start), 'op', call snapshot fields,
            and 'result' when captured.

        Notes
        -
        Ordering is guaranteed by 'version' and 'mono_ns'. The log is in-memory until exported.

        """
        if as_df:
            try:
                import polars as pl

                return pl.DataFrame(self._history)
            except Exception:
                try:
                    import pandas as pd

                    return pd.DataFrame.from_records(self._history)
                except Exception:
                    raise RuntimeError(
                        "Cannot return history as DataFrame: install polars (recommended) or pandas."
                    )
        return list(self._history)

    def export_history(self, path: str):
        """Write the mutation history to disk.

        Parameters
        --
        path : str
            Output path. Supported extensions: '.parquet', '.ndjson' (a.k.a. '.jsonl'),
            '.json', '.csv'. Unknown extensions default to Parquet by appending '.parquet'.

        Returns
        ---
        int
            Number of events written. Returns 0 if the history is empty.

        Raises
        --
        OSError
            If the file cannot be written.

        """
        if not self._history:
            return 0
        try:
            import polars as pl

            df = pl.DataFrame(self._history)
        except Exception:
            try:
                import pandas as pd

                df = pd.DataFrame.from_records(self._history)
            except Exception:
                raise RuntimeError(
                    "Cannot construct DataFrame from history: install polars (recommended) or pandas."
                )
        p = path.lower()
        if p.endswith(".parquet"):
            df.write_parquet(path)
            return len(df)
        if p.endswith(".ndjson") or p.endswith(".jsonl"):
            with open(path, "w", encoding="utf-8") as f:
                for r in df.iter_rows(named=True):
                    import json

                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            return len(df)
        if p.endswith(".json"):
            import json

            with open(path, "w", encoding="utf-8") as f:
                json.dump(df.to_dicts(), f, ensure_ascii=False)
            return len(df)
        if p.endswith(".csv"):
            df.write_csv(path)
            return len(df)
        # Default to Parquet if unknown
        df.write_parquet(path + ".parquet")
        return len(df)

    def enable_history(self, flag: bool = True):
        """Enable or disable in-memory mutation logging.

        Parameters
        --
        flag : bool, default True
            When True, start/continue logging; when False, pause logging.

        Returns
        ---
        None

        """
        self._history_enabled = bool(flag)

    def clear_history(self):
        """Clear the in-memory mutation log.

        Returns
        ---
        None

        Notes
        -
        This does not delete any files previously exported.

        """
        self._history.clear()

    def mark(self, label: str):
        """Insert a manual marker into the mutation history.

        Parameters
        --
        label : str
            Human-readable tag for the marker event.

        Returns
        ---
        None

        Notes
        -
        The event is recorded with 'op'='mark' alongside standard fields
        ('version', 'ts_utc', 'mono_ns'). Logging must be enabled for the
        marker to be recorded.

        """
        self._log_event("mark", label=label)
