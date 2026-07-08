"""Capture the environment so a benchmark report is reproducible.

A performance number is only an SSoT if you know the machine, the library
versions, and the exact source revision it came from. This module snapshots all
three into the results JSON header.
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
import platform
import subprocess


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def _lib_version(name: str) -> str | None:
    def _get():
        import importlib

        mod = importlib.import_module(name)
        return getattr(mod, '__version__', None)

    return _try(_get)


def _git_commit() -> str | None:
    def _get():
        out = subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()

    return _try(_get)


def _cpu_count() -> dict:
    import os

    return {
        'logical': os.cpu_count(),
        'affinity': _try(lambda: len(os.sched_getaffinity(0))),
    }


def capture() -> dict:
    """Return a JSON-serialisable snapshot of the benchmarking environment."""
    return {
        'timestamp_utc': datetime.now(UTC).isoformat(),
        'git_commit': _git_commit(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'processor': platform.processor() or platform.machine(),
        'cpu': _cpu_count(),
        'libraries': {
            'networkx': _lib_version('networkx'),
            'igraph': _lib_version('igraph'),
            'numpy': _lib_version('numpy'),
            'scipy': _lib_version('scipy'),
            'polars': _lib_version('polars'),
            'pandas': _lib_version('pandas'),
            'pyarrow': _lib_version('pyarrow'),
        },
    }
