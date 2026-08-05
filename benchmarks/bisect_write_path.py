"""Attribute a write-path change to the commit that caused it.

The companion of :mod:`bisect_read_path`, and it follows the same two rules.
It measures whichever tree is on ``PYTHONPATH``, which is what lets it be
pointed at an old commit in a ``git worktree``:

.. code-block:: bash

    git worktree add /tmp/wt-<sha> <sha>
    PYTHONPATH=/tmp/wt-<sha> python benchmarks/bisect_write_path.py <sha>

**Run it by path and not as ``-m benchmarks.bisect_write_path``.** The ``-m``
form puts the working directory first on ``sys.path``, ahead of ``PYTHONPATH``,
so run from a checkout it measures that checkout and reports the label of the
worktree it never imported. The answer carries ``annnet`` under ``measured``,
so a run that reached the wrong tree says so.

**Only API that held across the whole cycle is used**, because this module is
not imported by the tree it measures and must not depend on anything that tree
may not have.

Where it differs from the read-path probe is that a write has no reference to
be measured against — networkx does not build the same thing — so the answers
here are absolute milliseconds. Alternate two trees and compare the runs; a
machine that drifts moves both, so run each tree more than once and read the
ranges rather than a single figure.

Every case is a pair: what to set up before the clock starts, and what to time.
The setup runs again every round, so nothing a round leaves behind — a built
matrix above all — is still there for the next one.
"""

from __future__ import annotations

import gc
import sys
import json
import time

N_EDGES_SMALL = 4000
N_EDGES_LARGE = 25600
N_APPENDS = 500
N_REMOVES = 100
N_KEPT = 600


def payload(count: int) -> list[dict]:
    """A batch of binary edges over a quarter as many vertices."""
    vertices = max(2, count // 4)
    return [
        {'source': f'n{i % vertices}', 'target': f'n{(i * 7 + 3) % vertices}'} for i in range(count)
    ]


def measure(label: str = '', rounds: int = 5) -> dict:
    """Return the best time of each write-path workload, in milliseconds."""
    import annnet

    small, large = payload(N_EDGES_SMALL), payload(N_EDGES_LARGE)

    def loaded(items):
        graph = annnet.Graph(directed=True)
        graph.add_edges(items)
        return graph

    held_small, held_large = loaded(small), loaded(large)
    victims = [f'edge_{i}' for i in range(N_REMOVES)]
    kept = [f'n{i}' for i in range(N_KEPT)]

    def appends(graph):
        for i in range(N_APPENDS):
            graph.add_edges([{'source': f'a{i}', 'target': f'b{i}'}])
            graph.X()

    def removes(graph):
        for edge_id in victims:
            graph.remove_edge(edge_id)
            graph.X()

    cases = {
        f'load {N_EDGES_LARGE} edges': (lambda: None, lambda _: loaded(large)),
        f'load {N_EDGES_SMALL} edges': (lambda: None, lambda _: loaded(small)),
        'copy': (lambda: held_large, lambda graph: graph.ops.copy()),
        'first matrix read after a load': (
            lambda: held_large.ops.copy(),
            lambda graph: graph.X(),
        ),
        f'{N_APPENDS} appends with a read each': (lambda: held_small.ops.copy(), appends),
        f'{N_REMOVES} removes with a read each': (lambda: held_small.ops.copy(), removes),
        f'subgraph of {N_KEPT} vertices': (
            lambda: held_small,
            lambda graph: graph.ops.subgraph(kept),
        ),
    }

    best = {}
    for name, (prepare, run) in cases.items():
        run(prepare())
        times = []
        for _ in range(rounds):
            subject = prepare()
            gc.collect()
            start = time.perf_counter()
            run(subject)
            times.append((time.perf_counter() - start) * 1e3)
        best[name] = min(times)

    return {'label': label, 'measured': annnet.__file__, 'best_ms': best}


if __name__ == '__main__':
    print(
        json.dumps(
            measure(
                sys.argv[1] if len(sys.argv) > 1 else '',
                int(sys.argv[2]) if len(sys.argv) > 2 else 5,
            )
        )
    )
