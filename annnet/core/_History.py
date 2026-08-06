"""Mutation history, snapshots, and diffs."""

import json
import time
import inspect
from datetime import UTC, datetime
from functools import wraps

import numpy as np

from .._support.dataframe_backend import (
    dataframe_from_rows,
    dataframe_write_csv,
    dataframe_write_parquet,
)

_VAR_POSITIONAL = inspect.Parameter.VAR_POSITIONAL
_VAR_KEYWORD = inspect.Parameter.VAR_KEYWORD
_time = time.time


class GraphDiff:
    """Set difference between two graph snapshots (nodes/edges/slices added/removed)."""

    def __init__(self, snapshot_a, snapshot_b):
        self.snapshot_a = snapshot_a
        self.snapshot_b = snapshot_b
        self.nodes_added = snapshot_b['node_ids'] - snapshot_a['node_ids']
        self.nodes_removed = snapshot_a['node_ids'] - snapshot_b['node_ids']
        self.edges_added = snapshot_b['edge_ids'] - snapshot_a['edge_ids']
        self.edges_removed = snapshot_a['edge_ids'] - snapshot_b['edge_ids']
        self.slices_added = snapshot_b['slice_ids'] - snapshot_a['slice_ids']
        self.slices_removed = snapshot_a['slice_ids'] - snapshot_b['slice_ids']

    def summary(self):
        """Return a human-readable summary of differences.

        Returns
        -------
        str
            Summary text describing added/removed nodes, edges, and slices.
        """
        return '\n'.join(
            [
                f'Diff: {self.snapshot_a["label"]} - {self.snapshot_b["label"]}',
                '',
                f'Nodes: {len(self.nodes_added):+d} added, {len(self.nodes_removed)} removed',
                f'Edges: {len(self.edges_added):+d} added, {len(self.edges_removed)} removed',
                f'slices: {len(self.slices_added):+d} added, {len(self.slices_removed)} removed',
            ]
        )

    def is_empty(self):
        """Check whether the diff contains no changes.

        Returns
        -------
        bool
        """
        return not (
            self.nodes_added
            or self.nodes_removed
            or self.edges_added
            or self.edges_removed
            or self.slices_added
            or self.slices_removed
        )

    def __repr__(self):
        return self.summary()

    def to_dict(self):
        """Convert the diff to a serializable dictionary.

        Returns
        -------
        dict
        """
        return {
            'snapshot_a': self.snapshot_a['label'],
            'snapshot_b': self.snapshot_b['label'],
            'nodes_added': list(self.nodes_added),
            'nodes_removed': list(self.nodes_removed),
            'edges_added': list(self.edges_added),
            'edges_removed': list(self.edges_removed),
            'slices_added': list(self.slices_added),
            'slices_removed': list(self.slices_removed),
        }


class History:
    """Mutation logging, version counter, snapshots, and diffs (mixed into AnnNet)."""

    # The second the stamp of the last event fell in, and the text of it. Class
    # attributes so that no constructor and no copy has to know about them.
    _iso_second = -1
    _iso_prefix = ''

    def _bump_version(self) -> int:
        self._version += 1
        return self._version

    def _utcnow_iso(self) -> str:
        """The current instant, as ``YYYY-MM-DDTHH:MM:SS.ffffffZ``.

        Everything but the fraction is the same for every event of one second,
        and formatting a whole datetime is 1.1 of the 1.3 microseconds this used
        to cost — against a logged mutation of about four. So the second is
        formatted when it changes and the fraction is appended to it.
        """
        now = _time()
        whole = int(now)
        if whole != self._iso_second:
            self._iso_prefix = datetime.fromtimestamp(whole, UTC).isoformat(timespec='seconds')[:19]
            self._iso_second = whole
        return f'{self._iso_prefix}.{int((now - whole) * 1_000_000):06d}Z'

    def _jsonify(self, x):
        if x is None or isinstance(x, (bool, int, float, str)):
            return x
        if isinstance(x, (set, frozenset)):
            return sorted(self._jsonify(v) for v in x)
        if isinstance(x, (list, tuple)):
            return [self._jsonify(v) for v in x]
        if isinstance(x, dict):
            return {str(k): self._jsonify(v) for k, v in x.items()}
        if isinstance(x, np.generic):
            return x.item()
        return f'<<{type(x).__name__}>>'

    def _log_event(self, op: str, **fields):
        version = self._bump_version()
        if not self._history_enabled:
            return
        evt = {
            'version': version,
            'ts_utc': self._utcnow_iso(),
            'mono_ns': time.perf_counter_ns() - self._history_clock0,
            'op': op,
        }
        for k, v in fields.items():
            evt[k] = self._jsonify(v)
        self._history.append(evt)

    @staticmethod
    def _summarize_arg(v, _limit=32, _depth=1):
        """Cheap log value: large collections are summarized, not serialized.

        The walk matters as much as the test. A batch reaches a wrapped method
        through ``*args`` or ``**kwargs``, so the value bound to the parameter
        is a one-element tuple or a one-entry dict and the collection worth
        summarizing is one step inside it. Testing only the outer value let a
        bulk ``add_edges`` log a copy of every item it was given, which is the
        serialization the summary exists to avoid.

        One step is the whole of it: both wrappers hold the caller's own
        arguments directly, so nothing a caller passes sits deeper. Only
        ``list``, ``tuple`` and ``dict`` are walked — a set holds hashables, and
        a summarized element need not be one.

        A container with nothing to summarize is answered with itself. Every
        single write logs one, and rebuilding it would cost an allocation per
        call to arrive at the same value.
        """
        if not isinstance(v, (list, tuple, set, frozenset, dict)):
            return v
        if len(v) > _limit:
            return f'<{type(v).__name__}: {len(v)} items>'
        if not _depth or isinstance(v, (set, frozenset)):
            return v
        for entry in v.values() if isinstance(v, dict) else v:
            if isinstance(entry, (list, tuple, set, frozenset, dict)) and len(entry) > _limit:
                break
        else:
            return v
        summarize = History._summarize_arg
        if isinstance(v, dict):
            return {k: summarize(x, _limit, _depth - 1) for k, x in v.items()}
        return [summarize(x, _limit, _depth - 1) for x in v]

    def _log_mutation(self, name=None):
        def deco(fn):
            op = name or fn.__name__
            sig = inspect.signature(fn)
            # A method that takes nothing but ``*args`` and ``**kwargs`` binds to
            # exactly those two names, whatever the caller passes. Working that
            # out through ``inspect`` costs 1.8 microseconds a call, against a
            # single remove of about 10, and it arrives at the dict written here.
            kinds = [p.kind for p in sig.parameters.values()]
            star_only = kinds == [_VAR_POSITIONAL, _VAR_KEYWORD]

            @wraps(fn)
            def wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                if not self._history_enabled:
                    self._bump_version()
                    return result
                # Capture args only when logging; summarize big collections so a
                # bulk add_edges([...10k...]) does not pay an O(n) serialization.
                summ = self._summarize_arg
                if star_only:
                    payload = {'args': summ(args), 'kwargs': summ(kwargs)}
                else:
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    payload = {k: summ(v) for k, v in bound.arguments.items() if k != 'self'}
                payload['result'] = summ(result)
                self._log_event(op, **payload)
                return result

            return wrapper

        return deco

    def _install_history_hooks(self):
        to_wrap = [
            'add_nodes',
            'add_edges',
            'add_node',
            'add_edge',
            'add_hyperedge',
            'flatten_layers',
            'remove_edge',
            'remove_node',
            'set_node_attrs',
            'set_edge_attrs',
            'set_slice_attrs',
            'set_edge_slice_attrs',
            'register_slice',
            'unregister_slice',
        ]
        for name in to_wrap:
            fn = getattr(self, name, None)
            if fn and getattr(fn, '__wrapped__', None) is None:
                setattr(self, name, self._log_mutation(name)(fn))

    def history(self, as_df: bool = False):
        """Return the append-only mutation history.

        Parameters
        ----------
        as_df : bool, default False
            If True, return a DataFrame; otherwise return a list of dicts.

        Returns
        -------
        list[dict] | DataFrame
            Event records including `version`, `ts_utc`, `mono_ns`, `op`, and
            captured arguments/results.

        Notes
        -----
        Ordering is guaranteed by `version` and `mono_ns`. The log is in-memory
        until exported.
        """
        if as_df:
            return dataframe_from_rows(self._history)
        return list(self._history)

    def export_history(self, path: str):
        """Write the mutation history to disk.

        Parameters
        ----------
        path : str
            Output path. Supported extensions: `.parquet`, `.ndjson`/`.jsonl`,
            `.json`, `.csv`. Unknown extensions default to Parquet.

        Returns
        -------
        int
            Number of events written. Returns 0 if the history is empty.

        Raises
        ------
        OSError
            If the file cannot be written.

        Notes
        -----
        Unknown extensions default to Parquet by appending `.parquet`.
        """
        if not self._history:
            return 0
        suffix = str(path).lower()

        if suffix.endswith('.ndjson') or suffix.endswith('.jsonl'):
            with open(path, 'w', encoding='utf-8') as f:
                for r in self._history:
                    f.write(json.dumps(self._jsonify(r), ensure_ascii=False) + '\n')
            return len(self._history)
        if suffix.endswith('.json'):
            with open(path, 'w', encoding='utf-8') as f:
                json.dump([self._jsonify(r) for r in self._history], f, ensure_ascii=False)
            return len(self._history)

        def _flatten_row(row):
            flat = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple, set, frozenset)):
                    flat[key] = json.dumps(self._jsonify(value), ensure_ascii=False)
                else:
                    flat[key] = value
            return flat

        df = dataframe_from_rows([_flatten_row(r) for r in self._history])
        if suffix.endswith('.parquet'):
            dataframe_write_parquet(df, path)
        elif suffix.endswith('.csv'):
            dataframe_write_csv(df, path)
        else:
            dataframe_write_parquet(df, str(path) + '.parquet')
        return len(self._history)

    def enable_history(self, flag: bool = True):
        """Enable or disable in-memory mutation logging.

        Parameters
        ----------
        flag : bool, default True
            When True, start/continue logging; when False, pause logging.

        Returns
        -------
        None
        """
        self._history_enabled = bool(flag)

    def clear_history(self):
        """Clear the in-memory mutation log.

        Returns
        -------
        None

        Notes
        -----
        This does not delete any files previously exported.
        """
        self._history.clear()

    def mark(self, label: str):
        """Insert a manual marker into the mutation history.

        Parameters
        ----------
        label : str
            Human-readable tag for the marker event.

        Returns
        -------
        None

        Notes
        -----
        The event is recorded with `op='mark'` alongside standard fields
        (`version`, `ts_utc`, `mono_ns`). Logging must be enabled for the
        marker to be recorded.
        """
        self._log_event('mark', label=label)

    def _history_snapshot_impl(self, label=None):
        raw = self._current_snapshot()
        name = label if label is not None else f'snap_{len(self._snapshots)}'
        snap = {
            'label': name,
            'version': raw['version'],
            'node_ids': set(raw['node_ids']),
            'edge_ids': set(raw['edge_ids']),
            'slice_ids': set(raw['slice_ids']),
        }
        self._snapshots.append(snap)
        self._log_event('snapshot', label=name, version=snap['version'])
        return snap

    def _history_diff_impl(self, a=None, b=None):
        if a is None:
            if not self._snapshots:
                raise ValueError(
                    'history.diff() without arguments needs at least one snapshot; '
                    'call history.snapshot() first or pass an explicit reference.'
                )
            a = self._snapshots[-1]
        snap_a = self._resolve_snapshot(a)
        snap_b = self._resolve_snapshot(b) if b is not None else self._current_snapshot()
        return GraphDiff(snap_a, snap_b)

    def _history_list_snapshots_impl(self):
        return list(self._snapshots)


class HistoryAccessor:
    """Callable ``G.history`` namespace for logs and snapshots."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def __call__(self, *args, **kwargs):
        return History.history(self._G, *args, **kwargs)

    def enable(self, flag: bool = True):
        """Enable or disable mutation history recording."""
        return History.enable_history(self._G, flag)

    def clear(self):
        """Clear the recorded history log and stored snapshots."""
        return History.clear_history(self._G)

    def export(self, path: str):
        """Write the recorded history log to disk."""
        return History.export_history(self._G, path)

    def mark(self, label: str):
        """Append a labeled marker event to the history log."""
        return History.mark(self._G, label)

    def snapshot(self, label=None):
        """Capture and return a snapshot of the current graph state."""
        return self._G._history_snapshot_impl(label=label)

    def diff(self, a=None, b=None):
        """Return a diff between two snapshots or snapshot references."""
        return self._G._history_diff_impl(a, b=b)

    def list_snapshots(self):
        """Return the recorded snapshot list."""
        return self._G._history_list_snapshots_impl()
