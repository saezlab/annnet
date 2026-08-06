"""Attribute a read-path change to the commit that caused it.

The rest of ``benchmarks`` measures the tree it lives in. This module measures
whichever tree is on ``PYTHONPATH``, which is what lets it be pointed at an old
commit in a ``git worktree``:

.. code-block:: bash

    git worktree add /tmp/wt-<sha> <sha>
    PYTHONPATH=/tmp/wt-<sha> python benchmarks/bisect_read_path.py <sha>

**Run it by path and not as ``-m benchmarks.bisect_read_path``.** The ``-m`` form
puts the working directory first on ``sys.path``, ahead of ``PYTHONPATH``, so
run from a checkout it measures that checkout and reports the label of the
worktree it never imported. Running the file by path puts ``benchmarks/`` first
instead, and no tree lives there. The answer carries ``annnet`` under
``measured``, so a run that reached the wrong tree says so.

Two rules make the answers comparable across commits and across machines.

**The reference is measured in the same process as the subject.** An absolute
number does not survive a change of machine, or a busy one — the reference
libraries in the T044 run were up to 1.9 times slower than in the baseline run,
which made every absolute comparison in that table invalid. A ratio against a
reference measured beside the subject survives both.

**Only API that held across the whole cycle is used.** This module is not
imported by the tree it measures, so it must not depend on anything that tree
may not have. Nothing here reaches past the public graph.

The scale is the ``small`` scale of ``benchmarks-baseline.md``: 1 000 nodes and
4 000 edges, a ring with a constant degree, so a query measures the cost of one
call rather than the size of an adjacency list.
"""

from __future__ import annotations

import gc
import sys
import json
import time
import inspect

N_NODES = 1000
N_EDGES = 4000


def make_data() -> tuple[list[str], list[tuple[str, str, float]]]:
    """The ring the baseline measured, shared by both engines."""
    names = [f'v{i}' for i in range(N_NODES)]
    edges = [(f'v{i % N_NODES}', f'v{(i + 1) % N_NODES}', 1.0) for i in range(N_EDGES)]
    return names, edges


def annnet_build(names, edges):
    """Return a callable that builds the graph, on whatever tree is imported."""
    from annnet.core.graph import AnnNet

    kwargs = {'directed': True}
    if 'annotations_backend' in inspect.signature(AnnNet.__init__).parameters:
        kwargs['annotations_backend'] = 'auto'

    def build():
        graph = AnnNet(**kwargs)
        graph.add_nodes(({'node_id': v} for v in names), slice='base')
        graph.add_edges({'source': u, 'target': v, 'weight': w} for (u, v, w) in edges)
        return graph

    return build


def networkx_build(names, edges):
    """Return a callable that builds the reference graph."""
    import networkx as nx

    def build():
        graph = nx.DiGraph()
        graph.add_nodes_from(names)
        graph.add_weighted_edges_from(edges)
        return graph

    return build


def _oneshot(fn, samples: int = 5) -> float:
    out = []
    for _ in range(samples):
        gc.collect()
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return min(out)


def _repeat(fn, *, calls: int, samples: int = 5) -> float:
    out = []
    for _ in range(samples):
        gc.collect()
        start = time.perf_counter()
        for _ in range(calls):
            fn()
        out.append((time.perf_counter() - start) / calls)
    return min(out)


def _measure(build, ops_of) -> dict:
    result = {'build': _oneshot(build)}
    handle = build()
    for name, fn in ops_of(handle).items():
        result[name] = _repeat(fn, calls=20 if name == 'enumerate_edges' else 2000)
    return result


def measure(label: str = '') -> dict:
    """Return the ratio of each operation against the reference, and the raw times."""
    names, edges = make_data()
    lo, hi = names[0], names[1]

    def annnet_ops(graph):
        return {
            'degree': lambda: graph.degree(lo),
            'neighbors': lambda: graph.neighbors(lo),
            'has_edge': lambda: graph.has_edge(lo, hi),
            'enumerate_edges': lambda: list(graph.edges()),
        }

    def networkx_ops(graph):
        return {
            'degree': lambda: graph.degree(lo),
            'neighbors': lambda: list(graph.neighbors(lo)),
            'has_edge': lambda: graph.has_edge(lo, hi),
            'enumerate_edges': lambda: list(graph.edges()),
        }

    # The reference runs on both sides of the subject, so a machine that drifts
    # during the run drifts across the reference as well as across the subject.
    reference = _measure(networkx_build(names, edges), networkx_ops)
    subject = _measure(annnet_build(names, edges), annnet_ops)
    after = _measure(networkx_build(names, edges), networkx_ops)
    reference = {op: min(value, after[op]) for op, value in reference.items()}

    import annnet

    return {
        'label': label,
        'measured': annnet.__file__,
        'ratio': {op: subject[op] / reference[op] for op in subject},
        'annnet_us': {op: subject[op] * 1e6 for op in subject},
        'reference_us': {op: reference[op] * 1e6 for op in reference},
    }


if __name__ == '__main__':
    print(json.dumps(measure(sys.argv[1] if len(sys.argv) > 1 else '')))
