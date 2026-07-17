"""Micro-benchmark primitives: adaptive-repeat timing + tracemalloc memory.

Single-shot timings and raw RSS deltas are too noisy for this suite. Everything
here is built so a number can be trusted as an SSoT:

* **Timing** — warmup, an adaptively sized inner loop so each batch runs long
  enough to dominate clock resolution, several batches, GC disabled during the
  timed region, and a reported distribution (min/median/mean/stdev/p95). ``min``
  is the least-contaminated estimate of intrinsic cost; ``median`` is the typical
  cost under real scheduling.
* **Memory** — ``tracemalloc`` gives a deterministic Python-level peak and a
  retained estimate (what is still referenced once the object is built), which is
  what "bytes per edge" should be derived from. ``psutil`` RSS is kept as a
  secondary, process-level cross-check.
"""

from __future__ import annotations

import gc
import os
import math
import time
import statistics
from dataclasses import asdict, dataclass
import tracemalloc
from collections.abc import Callable

import psutil

_PROC = psutil.Process(os.getpid())


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
@dataclass
class TimeStat:
    """Per-call timing distribution, all times in seconds."""

    min_s: float
    median_s: float
    mean_s: float
    stdev_s: float
    p95_s: float
    samples: int  # number of batches collected
    inner: int  # inner-loop iterations per batch (adaptive)
    total_calls: int  # samples * inner

    def as_dict(self) -> dict:
        return asdict(self)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float('nan')
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = q * (len(sorted_vals) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _stat_from(per_call: list[float], inner: int) -> TimeStat:
    s = sorted(per_call)
    return TimeStat(
        min_s=s[0],
        median_s=statistics.median(s),
        mean_s=statistics.fmean(s),
        stdev_s=statistics.pstdev(s) if len(s) > 1 else 0.0,
        p95_s=_percentile(s, 0.95),
        samples=len(s),
        inner=inner,
        total_calls=inner * len(s),
    )


def time_repeat(
    fn: Callable[[], object],
    *,
    warmup: int = 3,
    min_batch_s: float = 0.05,
    samples: int = 7,
    max_inner: int = 5_000_000,
) -> TimeStat:
    """Time a *repeatable, side-effect-free* callable (queries, lookups, reads).

    Calibrates an inner-loop count so each batch runs at least ``min_batch_s``,
    then collects ``samples`` batches with GC disabled. Per-call time is
    ``batch_time / inner``. Do not use for anything that mutates shared state —
    use :func:`time_oneshot` for construction / mutation.
    """
    for _ in range(warmup):
        fn()

    inner = 1
    while inner < max_inner:
        gc.collect()
        gc.disable()
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        dt = (time.perf_counter_ns() - t0) / 1e9
        gc.enable()
        if dt >= min_batch_s:
            break
        if dt <= 0:
            inner = min(inner * 8, max_inner)
            continue
        factor = max(2, int(min_batch_s / dt) + 1)
        inner = min(inner * factor, max_inner)

    per_call: list[float] = []
    for _ in range(samples):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter_ns()
        for _ in range(inner):
            fn()
        dt = (time.perf_counter_ns() - t0) / 1e9
        gc.enable()
        per_call.append(dt / inner)
    return _stat_from(per_call, inner)


def time_oneshot(
    make_fn: Callable[[], object],
    *,
    warmup: int = 1,
    samples: int = 5,
    teardown: Callable[[object], None] | None = None,
) -> TimeStat:
    """Time a *non-repeatable* callable (construction, copy, subgraph, mutation).

    Each sample runs ``make_fn()`` fresh and discards (or tears down) the result,
    so no state leaks between samples. ``inner`` is fixed at 1; the reported
    ``min_s`` is the cleanest estimate of build cost.
    """
    for _ in range(warmup):
        obj = make_fn()
        if teardown:
            teardown(obj)
        del obj

    per_call: list[float] = []
    for _ in range(samples):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter_ns()
        obj = make_fn()
        dt = (time.perf_counter_ns() - t0) / 1e9
        gc.enable()
        per_call.append(dt)
        if teardown:
            teardown(obj)
        del obj
    return _stat_from(per_call, inner=1)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------
@dataclass
class MemStat:
    peak_bytes: int  # tracemalloc high-water mark during build
    retained_bytes: int  # tracemalloc current with the object alive (footprint proxy)
    rss_delta_bytes: int  # psutil RSS delta (secondary, process-level cross-check)

    def as_dict(self) -> dict:
        return asdict(self)


def measure_memory(make_fn: Callable[[], object]) -> MemStat:
    """Peak + retained allocation for building one object.

    ``peak_bytes`` is the tracemalloc high-water mark attributable to the build.
    ``retained_bytes`` is what is still tracked once the object is built and a GC
    pass has run — the right basis for "bytes per edge". ``rss_delta_bytes`` is a
    coarse process-level cross-check. tracemalloc adds overhead, so never mix a
    memory pass into a timing pass.
    """
    gc.collect()
    rss0 = _PROC.memory_info().rss
    tracemalloc.start()
    base_cur, _base_peak = tracemalloc.get_traced_memory()
    obj = make_fn()
    _cur, peak = tracemalloc.get_traced_memory()
    gc.collect()
    cur2, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss1 = _PROC.memory_info().rss
    retained = max(cur2 - base_cur, 0)
    peak_delta = max(peak - base_cur, 0)
    del obj
    return MemStat(
        peak_bytes=peak_delta,
        retained_bytes=retained,
        rss_delta_bytes=rss1 - rss0,
    )
