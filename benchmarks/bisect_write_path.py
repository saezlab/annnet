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
    """A batch of binary edges over a quarter as many nodes."""
    nodes = max(2, count // 4)
    return [{'source': f'n{i % nodes}', 'target': f'n{(i * 7 + 3) % nodes}'} for i in range(count)]


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
            _ = graph.S

    def removes(graph):
        for edge_id in victims:
            graph.remove_edge(edge_id)
            _ = graph.S

    cases = {
        f'load {N_EDGES_LARGE} edges': (lambda: None, lambda _: loaded(large)),
        f'load {N_EDGES_SMALL} edges': (lambda: None, lambda _: loaded(small)),
        'copy': (lambda: held_large, lambda graph: graph.ops.copy()),
        'first matrix read after a load': (
            lambda: held_large.ops.copy(),
            lambda graph: graph.S,
        ),
        f'{N_APPENDS} appends with a read each': (lambda: held_small.ops.copy(), appends),
        f'{N_REMOVES} removes with a read each': (lambda: held_small.ops.copy(), removes),
        f'subgraph of {N_KEPT} nodes': (
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


# -- the bulk-build attribution -------------------------------------------
#
# `FR-024` asks where the wall time of a bulk build goes, by named phase, at the
# medium scale — 10 000 nodes and 40 000 edges, with the attribute counts the
# `medium` scale of the suite declares.
#
# The phases are told apart by a **profile**, grouped by a rule that maps a
# function to the phase of the data model it serves. Ablation was tried first and
# does not work here — see :func:`measure_phases` — and the store's own two
# phases are measured a second time without the profiler, so the distortion the
# profiler adds is visible rather than assumed.

MEDIUM_NODES = 10_000
MEDIUM_EDGES = 40_000
MEDIUM_NODE_ATTRS = 8
MEDIUM_EDGE_ATTRS = 4
MEDIUM_SLICES = 5


def _medium_rows(*, node_attrs: int, edge_attrs: int, slices: int):
    nodes = [
        {'node_id': f'v{i}', **{f'na{a}': float(i + a) for a in range(node_attrs)}}
        for i in range(MEDIUM_NODES)
    ]
    edges = []
    for i in range(MEDIUM_EDGES):
        row = {
            'source': f'v{i % MEDIUM_NODES}',
            'target': f'v{(i * 7 + 3) % MEDIUM_NODES}',
            'edge_id': f'e{i}',
            'weight': 1.0,
            **{f'ea{a}': float(i + a) for a in range(edge_attrs)},
        }
        if slices:
            row['slice'] = f's{i % slices}'
        edges.append(row)
    return nodes, edges


def _build_medium(*, node_attrs, edge_attrs, slices) -> float:
    import annnet

    nodes, edges = _medium_rows(node_attrs=node_attrs, edge_attrs=edge_attrs, slices=slices)

    def run():
        graph = annnet.Graph(directed=True)
        graph.add_nodes(nodes)
        graph.add_edges(edges)
        return graph

    return _oneshot_ms(run)


def _oneshot_ms(run, rounds: int = 5) -> float:
    run()
    times = []
    for _ in range(rounds):
        gc.collect()
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1e3)
    return min(times)


def _store_phases() -> tuple[float, float]:
    """Return the cost of registering the entities and of writing the edges.

    Both are measured on the canonical store directly, which is the one place
    the slot assignment and the member lists happen. Everything the gateway does
    around them — resolving an endpoint, deciding a direction, interning an id —
    is outside these two numbers by construction.
    """
    from annnet.core import _store as ST

    keys = [(f'v{i}', ('_',)) for i in range(MEDIUM_NODES)]
    specs = [
        (
            f'e{i}',
            [
                (keys[i % MEDIUM_NODES], 1.0, ST.SOURCE),
                (keys[(i * 7 + 3) % MEDIUM_NODES], -1.0, ST.TARGET),
            ],
            ST.BINARY,
            True,
            1.0,
            False,
            None,
            None,
            None,
        )
        for i in range(MEDIUM_EDGES)
    ]

    def entities():
        store = ST.CoreState(directed=True)
        for key in keys:
            store.add_entity(key)
        return store

    def edges():
        store = entities()
        store.add_edges(specs)
        return store

    identity = _oneshot_ms(entities)
    return identity, _oneshot_ms(edges) - identity


# Which named phase a function belongs to. A phase is a thing the data model
# requires, in the vocabulary of the model rather than of the call graph, so the
# mapping is from (module, function) to phase and not the other way round. A
# function no rule names lands in "the interpreter around it", which is counted
# and reported like the rest.
_PHASE_RULES = (
    (
        'identity registration',
        (
            '_identity',
            'resolve_',
            'ensure_endpoint',
            'add_entity',
            'register_entity',
            'entity_slot',
            'intern',
        ),
    ),
    (
        'slot assignment and member lists',
        (
            '_store',
            'add_edges',
            'add_edge',
            '_write_members',
            '_link_members',
            '_member_slots',
            '_grown',
        ),
    ),
    (
        'attribute cells',
        ('_attrs', 'set_node_attrs', 'set_edge_attrs', '_set', '_column_for', '_empty_column'),
    ),
    ('slice membership', ('_Slices', '_ensure_slice', 'slice', 'SliceRecord')),
    ('the mutation gateway', ('_mutate', '_build', 'graph.py')),
)


def _phase_of(filename: str, function: str) -> str:
    """Return the phase a profiled function belongs to, by the first rule that fits."""
    where = f'{filename}:{function}'
    for phase, markers in _PHASE_RULES:
        if any(marker in where for marker in markers):
            return phase
    return 'the interpreter around it'


def measure_phases() -> dict:
    """Attribute the wall time of a bulk build to named phases.

    The attribution is a profile rather than a set of ablations. Ablation was
    tried first and does not work here: there is no state in which a graph has no
    slice membership, because an edge that names no slice joins the default one,
    so "with the slice off" measures a different code path rather than the
    absence of one. A profile has its own price — the interpreter counts every
    call — so the unprofiled wall time is reported beside it and the shares are
    read against the profiled total.

    The self time of every function is summed, so the phases sum to the profiled
    total exactly and nothing is counted twice. Every function belongs to exactly
    one phase, including the ones no rule names.
    """
    import pstats
    import cProfile

    import annnet

    full = {
        'node_attrs': MEDIUM_NODE_ATTRS,
        'edge_attrs': MEDIUM_EDGE_ATTRS,
        'slices': MEDIUM_SLICES,
    }
    wall = _build_medium(**full)
    nodes, edges = _medium_rows(**full)

    def run():
        graph = annnet.Graph(directed=True)
        graph.add_nodes(nodes)
        graph.add_edges(edges)
        return graph

    profiler = cProfile.Profile()
    profiler.enable()
    run()
    profiler.disable()

    stats = pstats.Stats(profiler)
    phases: dict[str, float] = {}
    profiled = 0.0
    for (filename, _lineno, function), entry in stats.stats.items():
        self_time = entry[2] * 1e3
        profiled += self_time
        phase = _phase_of(str(filename), str(function))
        phases[phase] = phases.get(phase, 0.0) + self_time

    identity, member_lists = _store_phases()
    return {
        'label': 'phases',
        'measured': annnet.__file__,
        'nodes': MEDIUM_NODES,
        'edges': MEDIUM_EDGES,
        'wall_ms': wall,
        'profiled_ms': profiled,
        'phase_ms': dict(sorted(phases.items(), key=lambda kv: -kv[1])),
        'phase_share': {
            name: value / profiled for name, value in sorted(phases.items(), key=lambda kv: -kv[1])
        },
        'unprofiled_store_ms': {
            'identity registration': identity,
            'slot assignment and member lists': member_lists,
        },
    }


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--phases':
        print(json.dumps(measure_phases(), indent=2))
    else:
        print(
            json.dumps(
                measure(
                    sys.argv[1] if len(sys.argv) > 1 else '',
                    int(sys.argv[2]) if len(sys.argv) > 2 else 5,
                )
            )
        )
