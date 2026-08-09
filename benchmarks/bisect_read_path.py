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

## The derive probe

The module carries a second measurement, which is not a bisection. It re-measures
every number cycle 003 was planned against — the two attribute-column reads, the
O(1) probes on the store, the intrinsic edge column, and the three dataframe
baselines — so that any claim of that cycle can be checked with one command:

.. code-block:: bash

    python benchmarks/bisect_read_path.py --derive

The call counts are calibrated for the costs *after* cycle 003, which are
microseconds. A run against an older tree, where the same reads cost
milliseconds, should raise them or read the answer as an upper bound.

It prints the same JSON shape, under ``derive``. It reaches into ``annnet.core``
where the bisection probe deliberately does not, because the point of it is to
show which half of a read the cost sits in, and a public call cannot separate
the two.
"""

from __future__ import annotations

import gc
import sys
import json
import time
import inspect

N_NODES = 1000
N_EDGES = 4000

# The scales of the derive probe. The column read was measured at 100 000 nodes,
# the intrinsic edge column at 40 000 edges, and the dataframe baselines over
# 100 000 values.
DERIVE_NODES = 100_000
DERIVE_EDGES = 40_000

# The scale `SC-004` names for the rebuild of a cached matrix.
DERIVE_REBUILD_EDGES = 25_600


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


# -- the derive probe -----------------------------------------------------


def _derive_graph(nodes: int, edges: int):
    """A flat graph of ``nodes`` nodes and ``edges`` binary edges, both attributed.

    Every node carries ``score`` and every edge carries ``w2``, so the attribute
    column of each axis is full and the read has nothing to pad.
    """
    from annnet.core.graph import AnnNet

    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(nodes)])
    if edges:
        graph.add_edges(
            [
                {
                    'source': f'v{i % nodes}',
                    'target': f'v{(i + 1) % nodes}',
                    'weight': float(i),
                    'w2': float(i),
                }
                for i in range(edges)
            ]
        )
    return graph


def _column_reads() -> dict:
    """R1: what a column read costs, warm and after a structural write.

    The two writes differ in whether the element they add carries the attribute
    that is read. Neither is an attribute write: setting a cell does not advance
    the clock of the store, and it is the clock that the map from element id to
    slot is cached against.
    """
    graph = _derive_graph(DERIVE_NODES, 0)
    store = graph._store
    column = graph._attr_store.node_columns['score']
    count = store.entity_count
    fresh = iter(range(1_000_000))

    def after_carrying_write():
        graph.add_nodes([{'node_id': f'w{next(fresh)}', 'score': 0.0}])
        return graph.N['score']

    def after_bare_write():
        graph.add_nodes([{'node_id': f'x{next(fresh)}'}])
        return graph.N['score']

    graph.N['score']
    return {
        'read_warm': _repeat(lambda: graph.N['score'], calls=2000),
        'read_after_carrying_write': _repeat(after_carrying_write, calls=20),
        'read_after_bare_write': _repeat(after_bare_write, calls=20),
        'slice': _repeat(lambda: column[:count], calls=2000),
    }


def _store_probes() -> dict:
    """R2: the probes a contiguity predicate may and may not be built from."""
    graph = _derive_graph(DERIVE_NODES, 0)
    store = graph._store
    return {
        'entity_count': _repeat(lambda: store.entity_count, calls=20000),
        'len_entity_free': _repeat(lambda: len(store.entity_free), calls=20000),
        'len_entity_key': _repeat(lambda: len(store._entity_key), calls=20000),
        'live_entity_slots': _repeat(store.live_entity_slots, calls=5),
    }


def _edge_columns() -> dict:
    """R6: an intrinsic edge field against an edge attribute, and against the array."""
    graph = _derive_graph(DERIVE_EDGES // 4, DERIVE_EDGES)
    store = graph._store
    count = len(store._edge_id)
    return {
        'intrinsic_weight': _repeat(lambda: graph.E['weight'], calls=2000),
        'attribute_w2': _repeat(lambda: graph.E['w2'], calls=2000),
        'slice': _repeat(lambda: store.edge_weight[:count], calls=2000),
    }


def _dataframe_baselines() -> dict:
    """R7: the same sum over the same values, on each backend the package declares."""
    import numpy as np

    values = np.arange(DERIVE_NODES, dtype=np.float64)
    out = {'numpy': _repeat(lambda: float(values.sum()), calls=200)}
    for name, build in (('polars', _polars_column), ('pandas', _pandas_column)):
        column = build(values)
        out[name] = (
            None if column is None else _repeat(lambda held=column: float(held.sum()), calls=200)
        )
    return out


def _polars_column(values):
    try:
        import polars as pl
    except ImportError:
        return None
    return pl.DataFrame({'score': values})['score']


def _pandas_column(values):
    try:
        import pandas as pd
    except ImportError:
        return None
    return pd.DataFrame({'score': values})['score']


def _matrix_rebuild() -> dict:
    """R5: where the rebuild of a cached matrix spends its time.

    ``handover-07`` of cycle 002 names three causes for a rebuild costing about
    twice the one it replaced — the row lookup, the edge-kind selection, and the
    three array copies that seed the buffer — and only the third is a copy. The
    ceiling of `SC-004` cannot be set until the three are told apart.

    Each phase is timed on its own, on the same graph, so the parts are
    comparable with the whole rather than with each other alone.
    """
    from annnet.core import _matrices as M

    graph = _derive_graph(DERIVE_REBUILD_EDGES // 4, DERIVE_REBUILD_EDGES)
    store = graph._store
    kinds = None
    signed = True

    entity_slots, row_lookup = M._row_lookup(store)
    edge_slots = M._selected_edge_slots(store, kinds)
    built = M._incidence_matrix(store, entity_slots.size, edge_slots, row_lookup, signed)

    def whole():
        graph.matrices.cache.drop()
        return graph.S

    def seed():
        return M._CscBuffer.of(built)

    return {
        'whole_rebuild': _oneshot(whole, samples=7),
        'row_lookup': _repeat(lambda: M._row_lookup(store), calls=20),
        'edge_kind_selection': _repeat(lambda: M._selected_edge_slots(store, kinds), calls=20),
        'incidence_build': _repeat(
            lambda: M._incidence_matrix(store, entity_slots.size, edge_slots, row_lookup, signed),
            calls=5,
        ),
        'buffer_seed_copies': _repeat(seed, calls=20),
    }


def measure_rebuild() -> dict:
    """Return the attribution of a cached-matrix rebuild, in milliseconds."""
    import annnet

    parts = _matrix_rebuild()
    named = ('row_lookup', 'edge_kind_selection', 'incidence_build', 'buffer_seed_copies')
    attributed = sum(parts[name] for name in named)
    return {
        'label': 'rebuild',
        'measured': annnet.__file__,
        'edges': DERIVE_REBUILD_EDGES,
        'rebuild_ms': {name: value * 1e3 for name, value in parts.items()},
        'attributed_ms': attributed * 1e3,
        'unattributed_ms': (parts['whole_rebuild'] - attributed) * 1e3,
        'share_of_whole': {name: parts[name] / parts['whole_rebuild'] for name in named},
    }


def measure_derive() -> dict:
    """Return every reference measurement cycle 003 was planned against."""
    import annnet

    parts = {
        'column_read_us': _column_reads(),
        'store_probe_ns': _store_probes(),
        'edge_column_us': _edge_columns(),
        'dataframe_us': _dataframe_baselines(),
    }
    scaled = {
        'column_read_us': 1e6,
        'store_probe_ns': 1e9,
        'edge_column_us': 1e6,
        'dataframe_us': 1e6,
    }
    return {
        'label': 'derive',
        'measured': annnet.__file__,
        'nodes': DERIVE_NODES,
        'edges': DERIVE_EDGES,
        'derive': {
            group: {
                name: (None if value is None else value * scaled[group])
                for name, value in values.items()
            }
            for group, values in parts.items()
        },
    }


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--derive':
        print(json.dumps(measure_derive(), indent=2))
    elif len(sys.argv) > 1 and sys.argv[1] == '--rebuild':
        print(json.dumps(measure_rebuild(), indent=2))
    else:
        print(json.dumps(measure(sys.argv[1] if len(sys.argv) > 1 else '')))
