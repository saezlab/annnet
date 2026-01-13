import os
import time
from contextlib import contextmanager

import psutil

_PROC = psutil.Process(os.getpid())


def rss_mb() -> float:
    return _PROC.memory_info().rss / 1024**2


@contextmanager
def measure():
    """
    Context manager yielding a dict with:
      - wall_time_s
      - rss_before_mb
      - rss_after_mb
      - rss_delta_mb
    """
    rss0 = rss_mb()
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        t1 = time.perf_counter()
        rss1 = rss_mb()
        result.update(
            wall_time_s=t1 - t0,
            rss_before_mb=rss0,
            rss_after_mb=rss1,
            rss_delta_mb=rss1 - rss0,
        )
