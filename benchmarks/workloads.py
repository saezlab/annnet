"""Benchmark workloads: comparable (AnnNet vs NetworkX vs igraph) + AnnNet-only.

Each workload yields flat *records* (plain dicts) so the report layer can pivot
them freely. A record is one measurement of one operation on one engine at one
scale:

    {engine, backend, scale, n_vertices, n_edges, group, op,
     time: {..}|None, memory: {..}|None, note}

Two groups matter:

* **comparable** — construction, queries and edge enumeration measured
  identically on all three engines. These are the head-to-head numbers.
* **annnet_only** — hyperedges, multilayer, incidence/adjacency materialisation,
  copy/subgraph, annotations and IO. NetworkX and igraph cannot express most of
  these, so they are reported as an expressiveness-cost dimension.
"""

from __future__ import annotations

import shutil
from pathlib import Path
import tempfile

from . import engines, harness


def _record(
    engine, group, op, scale, n_v, n_e, *, backend=None, time=None, memory=None, note=''
) -> dict:
    return {
        'engine': engine,
        'backend': backend,
        'scale': scale.name,
        'n_vertices': n_v,
        'n_edges': n_e,
        'group': group,
        'op': op,
        'time': time.as_dict() if time is not None else None,
        'memory': memory.as_dict() if memory is not None else None,
        'note': note,
    }


# ---------------------------------------------------------------------------
# Comparable workloads (all engines)
# ---------------------------------------------------------------------------
def comparable(
    engine: engines.Engine, scale, *, backend=None, do_memory=True, samples=5
) -> list[dict]:
    data = engines.make_data(scale.vertices, scale.edges)
    n_v, n_e = scale.vertices, scale.edges
    build = engine.build_factory(data)
    recs: list[dict] = []

    def rec(op, **kw):
        return _record(
            engine.name,
            'construction' if op == 'build' else 'query',
            op,
            scale,
            n_v,
            n_e,
            backend=backend,
            **kw,
        )

    # Construction (timed) --------------------------------------------------
    t_build = harness.time_oneshot(build, samples=samples)
    recs.append(rec('build', time=t_build, note='bulk construction with a weight attribute'))

    # Construction (memory footprint) --------------------------------------
    if do_memory:
        m_build = harness.measure_memory(build)
        recs.append(
            _record(
                engine.name,
                'memory',
                'footprint',
                scale,
                n_v,
                n_e,
                backend=backend,
                memory=m_build,
                note='retained bytes -> bytes/edge in report',
            )
        )

    # Queries (timed) -------------------------------------------------------
    handle = build()
    for op, fn in engine.query_ops(handle, data).items():
        t = harness.time_repeat(fn)
        recs.append(rec(op, time=t))
    del handle
    return recs


# ---------------------------------------------------------------------------
# AnnNet-only workloads (expressiveness cost)
# ---------------------------------------------------------------------------
def _annnet():
    from annnet.core.graph import AnnNet

    return AnnNet


def annnet_only(scale, *, backend='auto', samples=5) -> list[dict]:
    AnnNet = _annnet()
    recs: list[dict] = []

    def rec(op, n_v, n_e, *, group='annnet_only', time=None, memory=None, note=''):
        return _record(
            'annnet',
            group,
            op,
            scale,
            n_v,
            n_e,
            backend=backend,
            time=time,
            memory=memory,
            note=note,
        )

    # --- Hyperedge construction (arity-k, no binary-graph equivalent) ------
    n_he = scale.hyperedges
    arity = 4
    n_hv = max(arity, min(scale.vertices, n_he * 2))
    hv_names = [f'h{i}' for i in range(n_hv)]

    def build_hyper():
        G = AnnNet(directed=True, annotations_backend=backend)
        G.add_vertices(({'vertex_id': v} for v in hv_names), slice='base')
        G.add_edges(
            {
                'source': [hv_names[(i + j) % n_hv] for j in range(arity - 1)],
                'target': [hv_names[(i + arity) % n_hv]],
                'weight': 1.0,
            }
            for i in range(n_he)
        )
        return G

    recs.append(
        rec(
            'build_hyperedges',
            n_hv,
            n_he,
            time=harness.time_oneshot(build_hyper, samples=samples),
            note=f'arity-{arity} directed hyperedges',
        )
    )
    recs.append(
        rec(
            'build_hyperedges',
            n_hv,
            n_he,
            group='memory',
            memory=harness.measure_memory(build_hyper),
            note='hyperedge graph footprint',
        )
    )

    # --- Multilayer (Kivela) construction ---------------------------------
    aspects = ['aspect_0', 'aspect_1']
    elem = {a: [f'{a}_e{j}' for j in range(3)] for a in aspects}  # 3x3 = 9 layers
    n_ml_v = max(20, scale.vertices // 20)
    ml_names = [f'm{i}' for i in range(n_ml_v)]

    def build_multilayer():
        import warnings

        G = AnnNet(directed=True, annotations_backend=backend)
        G.layers.set_aspects(aspects, elem)
        layer_tuples = list(G.layers.iter_layers())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.add_vertices(({'vertex_id': v} for v in ml_names), slice='base')
        for aa in layer_tuples:
            for v in ml_names:
                G.add_vertices(v, layer=aa)
        # intra-layer edges per layer
        for aa in layer_tuples:
            G.add_edges(
                [
                    {
                        'source': (ml_names[i], aa),
                        'target': (ml_names[(i + 1) % n_ml_v], aa),
                        'weight': 1.0,
                    }
                    for i in range(n_ml_v)
                ]
            )
        return G

    recs.append(
        rec(
            'build_multilayer',
            n_ml_v,
            n_ml_v * len(aspects),
            time=harness.time_oneshot(build_multilayer, samples=max(3, samples // 2)),
            note=f'{len(elem[aspects[0]]) ** len(aspects)} layers, intra-layer edges',
        )
    )

    # --- Materialisation: incidence (dok -> csr) and adjacency ------------
    base = engines.AnnNetEngine(backend=backend).build_factory(
        engines.make_data(scale.vertices, scale.edges)
    )()
    recs.append(
        rec(
            'materialize_incidence_csr',
            scale.vertices,
            scale.edges,
            group='materialize',
            time=harness.time_repeat(lambda: base._matrix.tocsr()),
            note='sparse incidence dok -> csr',
        )
    )

    def adjacency():
        csr = base._matrix.tocsr()
        return csr @ csr.T

    recs.append(
        rec(
            'materialize_adjacency',
            scale.vertices,
            scale.edges,
            group='materialize',
            time=harness.time_repeat(adjacency),
            note='A = B @ B.T from incidence',
        )
    )

    # --- Copy / subgraph ---------------------------------------------------
    recs.append(
        rec(
            'copy',
            scale.vertices,
            scale.edges,
            group='structural',
            time=harness.time_oneshot(lambda: base.ops.copy(), samples=max(3, samples // 2)),
        )
    )

    sub_v = base.vertices()[: max(1, scale.vertices // 2)]
    recs.append(
        rec(
            'subgraph_half',
            len(sub_v),
            scale.edges,
            group='structural',
            time=harness.time_oneshot(
                lambda: base.ops.subgraph(sub_v), samples=max(3, samples // 2)
            ),
        )
    )

    # --- Annotations bulk write -------------------------------------------
    # Re-applying identical attrs is stable, so time_repeat isolates the write
    # path from construction (unlike timing a build-then-annotate closure).
    vids = base.vertices()
    payload = [(v, {'kind': 'gene', 'score': i % 7}) for i, v in enumerate(vids)]
    recs.append(
        rec(
            'annotations_vertex_bulk',
            scale.vertices,
            scale.edges,
            group='annotations',
            time=harness.time_repeat(lambda: base.attrs.set_vertex_attrs_bulk(payload)),
            note='set_vertex_attrs_bulk over all vertices (write path only)',
        )
    )

    # --- IO round-trip (.annnet archive) + on-disk size -------------------
    from annnet.io import annnet_format

    tmp = Path(tempfile.mkdtemp(prefix='annnet_bench_'))
    disk_bytes = 0
    try:
        arch = tmp / 'g.annnet'

        def write_once():
            if arch.exists():
                arch.unlink()
            annnet_format.write(base, arch, overwrite=True)

        recs.append(
            rec(
                'io_write',
                scale.vertices,
                scale.edges,
                group='io',
                time=harness.time_oneshot(write_once, samples=max(3, samples // 2)),
                note='.annnet zstd archive',
            )
        )
        disk_bytes = arch.stat().st_size if arch.exists() else 0
        recs.append(
            rec(
                'io_read',
                scale.vertices,
                scale.edges,
                group='io',
                time=harness.time_oneshot(
                    lambda: annnet_format.read(arch), samples=max(3, samples // 2)
                ),
                note=f'on-disk archive = {disk_bytes} bytes',
            )
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for r in recs:
        if r['op'] in ('io_read', 'io_write'):
            r['disk_bytes'] = disk_bytes
    return recs


# ---------------------------------------------------------------------------
# AnnNet feature workloads: slices, history, algorithms (also AnnNet-only)
# ---------------------------------------------------------------------------
def annnet_features(scale, *, backend='auto', samples=5) -> list[dict]:
    """Rigorous timings for slice ops, history, and traversal/algorithms."""
    import warnings

    AnnNet = _annnet()
    n_v, n_e = scale.vertices, scale.edges
    n_slices = max(2, scale.slices)
    recs: list[dict] = []

    def rec(group, op, *, time=None, note=''):
        return _record('annnet', group, op, scale, n_v, n_e, backend=backend, time=time, note=note)

    # --- base graph with edges/vertices distributed across N slices --------
    sids = [f's{k}' for k in range(n_slices)]
    per = max(1, n_e // n_slices)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True)
        for k, sid in enumerate(sids):
            base = k * per
            G.add_edges(
                (
                    {
                        'source': f'v{(base + i) % n_v}',
                        'target': f'v{(base + i + 1) % n_v}',
                        'weight': 1.0,
                    }
                    for i in range(per)
                ),
                slice=sid,
            )
    v0 = G.vertices()[0]

    # --- slices (set algebra + presence + slice-induced subgraph) ----------
    recs.append(
        rec(
            'slices',
            'slice_union',
            time=harness.time_repeat(lambda: G.slices.union(sids)),
            note=f'union of {n_slices} slices',
        )
    )
    recs.append(
        rec(
            'slices',
            'slice_intersect',
            time=harness.time_repeat(lambda: G.slices.intersect(sids)),
            note=f'intersection of {n_slices} slices',
        )
    )
    recs.append(
        rec(
            'slices',
            'slice_difference',
            time=harness.time_repeat(lambda: G.slices.difference(sids[0], sids[1])),
            note='two-slice difference',
        )
    )
    recs.append(
        rec(
            'slices',
            'slice_vertex_presence',
            time=harness.time_repeat(lambda: G.slices.vertex_presence(v0)),
            note='vertex membership across slices',
        )
    )
    sv = list(G.slices.vertices(sids[0]))
    recs.append(
        rec(
            'slices',
            'slice_subgraph',
            time=harness.time_oneshot(lambda: G.ops.subgraph(sv), samples=max(3, samples // 2)),
            note='subgraph induced by one slice',
        )
    )

    # --- history (logging overhead + snapshot/diff + history-preserving copy)
    def _build_hist():
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            H = AnnNet(directed=True)
            H.history.enable()
            H.add_edges(
                {'source': f'v{i % n_v}', 'target': f'v{(i + 1) % n_v}', 'weight': 1.0}
                for i in range(n_e)
            )
        return H

    recs.append(
        rec(
            'history',
            'build_with_history',
            time=harness.time_oneshot(_build_hist, samples=max(3, samples // 2)),
            note='build with mutation logging enabled',
        )
    )
    Gh = _build_hist()
    Gh.history.snapshot('base')
    recs.append(
        rec(
            'history',
            'snapshot',
            time=harness.time_oneshot(
                lambda: Gh.history.snapshot('s'), samples=max(3, samples // 2)
            ),
            note='capture a named snapshot',
        )
    )
    recs.append(
        rec(
            'history',
            'diff',
            time=harness.time_repeat(lambda: Gh.history.diff()),
            note='diff current vs last snapshot',
        )
    )
    recs.append(
        rec(
            'history',
            'copy_with_history',
            time=harness.time_oneshot(
                lambda: G.ops.copy(history=True), samples=max(3, samples // 2)
            ),
            note='deep copy preserving history',
        )
    )

    # --- algorithms / traversal (directional neighbor + edge queries) ------
    sample = G.vertices()[: min(len(G.vertices()), 1000)]
    for op, fn in (
        ('out_neighbors', lambda: G.out_neighbors(v0)),
        ('in_neighbors', lambda: G.in_neighbors(v0)),
        ('successors', lambda: G.successors(v0)),
        ('predecessors', lambda: G.predecessors(v0)),
        ('incident_edges', lambda: G.incident_edges(v0)),
        ('in_edges', lambda: list(G.in_edges([v0]))),
        ('out_edges', lambda: list(G.out_edges([v0]))),
    ):
        recs.append(
            rec('algorithms', op, time=harness.time_repeat(fn), note='per-call traversal overhead')
        )
    recs.append(
        rec(
            'algorithms',
            'traversal_sweep',
            time=harness.time_oneshot(
                lambda: [G.out_neighbors(v) for v in sample], samples=max(3, samples // 2)
            ),
            note=f'out-neighbors over {len(sample)} vertices',
        )
    )
    recs.append(
        rec(
            'algorithms',
            'incidence_lists',
            time=harness.time_repeat(lambda: G.ops.get_vertex_incidence_matrix_as_lists()),
            note='per-vertex incident edge lists',
        )
    )

    # --- layers: supra (vertex-layer) index + supra operations --------------
    # Guards the cached supra index: nl_to_row must stay ~O(1) as V_M grows.
    n_layers = 3
    n_lv = max(50, min(scale.vertices // 20, 20_000))  # vertices per layer
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Gml = AnnNet(directed=True)
        Gml.layers.set_aspects(['layer'], {'layer': [f'L{k}' for k in range(n_layers)]})
        coords = list(Gml.layers.iter_layers())
        for aa in coords:
            Gml.add_edges(
                [
                    {'source': (f'm{i}', aa), 'target': (f'm{(i + 1) % n_lv}', aa), 'weight': 1.0}
                    for i in range(n_lv)
                ]
            )
    aa0 = coords[0]
    v_m = n_lv * n_layers
    recs.append(
        rec(
            'layers',
            'nl_to_row',
            time=harness.time_repeat(lambda: Gml.layers.nl_to_row('m0', aa0)),
            note=f'single vertex-layer lookup (V_M={v_m}); must be ~O(1)',
        )
    )
    recs.append(
        rec(
            'layers',
            'build_supra_index',
            time=harness.time_repeat(lambda: Gml.layers._build_supra_index()),
            note='full supra index (cache hit after first build)',
        )
    )
    recs.append(
        rec(
            'layers',
            'supra_adjacency',
            time=harness.time_oneshot(
                lambda: Gml.layers.supra_adjacency(), samples=max(3, samples // 2)
            ),
            note=f'supra-adjacency over {v_m} vertex-layer nodes',
        )
    )
    recs.append(
        rec(
            'layers',
            'subgraph_from_layer',
            time=harness.time_oneshot(
                lambda: Gml.layers.subgraph_from_layer_tuple(aa0), samples=max(3, samples // 2)
            ),
            note='single-layer induced subgraph',
        )
    )

    # --- backend proxy: G.nx.* per-call wrapper tax on a cache hit ----------
    # Guards the O(1) output mapping: this must stay flat as V grows.
    try:
        import networkx as _nx  # noqa: F401

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.nx.number_of_nodes(G)  # prime the conversion cache
            recs.append(
                rec(
                    'backend_proxy',
                    'nx_call_cache_hit',
                    time=harness.time_repeat(lambda: G.nx.number_of_nodes(G)),
                    note='G.nx.* wrapper tax with conversion cached; must be ~O(1) in V',
                )
            )
    except ImportError:
        pass

    # G.ig.* wrapper tax with a vertex arg (guards the cached name->index map;
    # a bulk vertex arg was O(V*k) before caching).
    try:
        import igraph as _ig  # noqa: F401

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G.ig.degree(vertices=['v0'])  # prime the conversion + name-index cache
            recs.append(
                rec(
                    'backend_proxy',
                    'ig_call_cache_hit',
                    time=harness.time_repeat(lambda: G.ig.degree(vertices=['v0'])),
                    note='G.ig.* wrapper tax (1 vertex arg) with caches warm; ~O(1) in V',
                )
            )
    except ImportError:
        pass

    return recs


# ---------------------------------------------------------------------------
# The operation set the core refactor has to measure
# ---------------------------------------------------------------------------
# Each key is an operation the cycle promises to report. Each value is the record
# ``op`` name that carries the measurement, so a missing measurement is visible
# rather than silently absent.
REQUIRED_OPERATIONS = {
    'single_element_add': 'single_element_add',
    'single_element_remove': 'single_element_remove',
    'bulk_build': 'build',
    'traversal': 'neighbors',
    'subgraph': 'subgraph_half',
    'matrix_materialization': 'materialize_incidence_csr',
    'read_after_mutate': 'read_after_mutate',
}


# How many single writes one timed batch performs. A batch amortizes the clock
# over many writes, and the graph is built outside the clock, so the reported
# number is the cost of one write rather than the cost of a build.
MUTATION_BATCH = 64


def mutations(
    engine: engines.Engine, scale, *, backend=None, samples=5, count: int = MUTATION_BATCH
) -> list[dict]:
    """Measure the single-element writes and the write-then-read pair.

    A bulk build hides the per-element cost, and the per-element cost is what a
    user pays when a graph grows one edge at a time. The graph is built outside
    the measurement, and the clock covers a batch of single writes on it, so the
    reported time is per write.
    """
    data = engines.make_data(scale.vertices, scale.edges)
    recs: list[dict] = []
    for op, mutation in engine.mutation_ops(data, count=count).items():
        try:
            time = harness.time_batched(
                mutation.setup,
                mutation.step,
                count=count,
                samples=max(1, samples),
                warmup=0,
            )
        except Exception as exc:
            rec = _record(
                engine.name,
                'mutation',
                op,
                scale,
                scale.vertices,
                scale.edges,
                backend=backend,
                note=f'one write, batched over {count}',
            )
            rec['status'] = 'error'
            rec['error'] = f'{type(exc).__name__}: {exc}'
            recs.append(rec)
            continue
        rec = _record(
            engine.name,
            'mutation',
            op,
            scale,
            scale.vertices,
            scale.edges,
            backend=backend,
            time=time,
            note=f'one write, batched over {count}',
        )
        rec['status'] = 'ok'
        recs.append(rec)
    return recs


def _growth_sizes(scale, points: int = 4) -> list[int]:
    """Return a doubling sweep of edge counts, so a curve shape is readable."""
    top = max(8, min(scale.edges, 1_600))
    sizes = []
    size = top
    for _ in range(points):
        sizes.append(size)
        size = max(2, size // 2)
    return sorted(set(sizes))


def matrix_growth(scale, *, backend='auto', samples=3) -> list[dict]:
    """Measure whether reading the matrix after every write stays linear.

    Two curves come out. ``append_only`` appends N edges and reads nothing.
    ``append_then_read`` appends the same N edges and reads the matrix after each
    one. A core that rebuilds the matrix on a stale read makes the second curve
    grow with N squared, and the ratio between the curves grows with N.
    """
    AnnNet = _annnet()
    recs: list[dict] = []

    def run(n_edges: int, read_each: bool):
        def once():
            G = AnnNet(directed=True, annotations_backend=backend)
            G.add_vertices([f'v{i}' for i in range(n_edges + 1)])
            for i in range(n_edges):
                G.add_edges(f'v{i}', f'v{i + 1}', edge_id=f'e{i}')
                if read_each:
                    _ = G.S
            return G

        return once

    for n_edges in _growth_sizes(scale):
        for op, read_each in (('append_only', False), ('append_then_read', True)):
            time = harness.time_oneshot(run(n_edges, read_each), samples=max(1, samples), warmup=0)
            rec = _record(
                'annnet',
                'matrix_growth',
                op,
                scale,
                n_edges + 1,
                n_edges,
                backend=backend,
                time=time,
                note=f'{n_edges} single appends, matrix read after each one'
                if read_each
                else f'{n_edges} single appends, no read',
            )
            rec['status'] = 'ok'
            recs.append(rec)
    return recs


def matrix_cache_probe(scale, *, backend='auto', samples=5) -> list[dict]:
    """Compare extending a cached sparse matrix against re-mapping it.

    This sets the limits of the append-only cache. ``cache_remap`` builds the
    matrix from the whole member list, which is what a dropped cache costs.
    ``cache_extend`` appends the new columns to a matrix that already exists,
    which is what a kept cache costs. The point where the two cross is the
    number of appended columns that makes dropping the cache the cheaper move.
    """
    import numpy as np
    import scipy.sparse as sp

    recs: list[dict] = []
    n_rows = max(8, min(scale.vertices, 2_000))
    n_cols = max(8, min(scale.edges, 8_000))

    def column_block(count: int):
        rows = np.tile(np.arange(min(2, n_rows)), count)
        cols = np.repeat(np.arange(count), min(2, n_rows))
        data = np.ones(rows.size, dtype=np.float32)
        return sp.csc_array((data, (rows, cols)), shape=(n_rows, count))

    base = column_block(n_cols)

    for appended in (1, 8, 64, 512):
        block = column_block(appended)

        def remap(block=block):
            # A dropped cache rebuilds the whole matrix and then adds the new columns.
            return sp.csc_array(
                sp.hstack([column_block(n_cols), block], format='csc'), dtype=np.float32
            )

        def extend(block=block):
            # A kept cache adds the new columns to the matrix it already holds.
            return sp.csc_array(sp.hstack([base, block], format='csc'), dtype=np.float32)

        for op, fn in (('cache_remap', remap), ('cache_extend', extend)):
            time = harness.time_oneshot(fn, samples=max(1, samples), warmup=0)
            rec = _record(
                'annnet',
                'matrix_cache',
                op,
                scale,
                n_rows,
                n_cols + appended,
                backend=backend,
                time=time,
                note=f'{appended} appended columns onto {n_cols}',
            )
            rec['status'] = 'ok'
            recs.append(rec)
    return recs


def attribute_ops(scale, *, backend='auto', samples=5) -> list[dict]:
    """Compare one vectorized operation on an attribute column to a dataframe column.

    The requirement is that an attribute operation runs as fast as the same
    operation on a dataframe column of the same length. This pair of records is
    the evidence.
    """
    import numpy as np

    AnnNet = _annnet()
    recs: list[dict] = []
    n = max(16, min(scale.vertices, 20_000))
    values = np.arange(n, dtype=np.float64)

    G = AnnNet(directed=True, annotations_backend=backend)
    G.add_vertices([{'vertex_id': f'v{i}', 'score': float(i)} for i in range(n)])

    def attr_column_op():
        # The public column read, which is what a user does the operation on.
        # Reading the table instead would measure the materialization and not
        # the operation.
        return float(G.N['score'].sum())

    def dataframe_column_op():
        return float(values.sum())

    for op, fn in (
        ('attr_column_op', attr_column_op),
        ('dataframe_column_op', dataframe_column_op),
    ):
        time = harness.time_repeat(fn, samples=max(1, samples))
        rec = _record(
            'annnet',
            'attributes',
            op,
            scale,
            n,
            0,
            backend=backend,
            time=time,
            note=f'sum over {n} values',
        )
        rec['status'] = 'ok'
        recs.append(rec)
    return recs


def attribute_storage_options(scale, *, backend='auto', samples=5) -> list[dict]:
    """Measure the two candidate attribute stores against each other.

    ``columnar`` keeps one typed array per attribute, indexed by slot. A single
    write lands in one cell. ``journal`` appends a fact per write and folds the
    facts when a reader asks for a frame. The adverse case for the journal is a
    tight alternation of one write and one full-frame read, so both stores are
    measured on a single write and on a whole-frame read.
    """
    import numpy as np

    recs: list[dict] = []
    n = max(16, min(scale.vertices, 20_000))

    class Columnar:
        def __init__(self, size):
            self.column = np.zeros(size, dtype=np.float64)

        def write(self, slot, value):
            self.column[slot] = value

        def frame(self):
            return self.column

    class Journal:
        """An append-only journal with a vectorized fold.

        The fold uses one numpy scatter rather than a Python loop, so the
        comparison measures the cost of folding facts and not the cost of
        interpreting a loop.
        """

        def __init__(self, size):
            self.size = size
            self.slots: list[int] = []
            self.values: list[float] = []

        def write(self, slot, value):
            self.slots.append(slot)
            self.values.append(value)

        def frame(self):
            out = np.zeros(self.size, dtype=np.float64)
            # A duplicate index keeps the last value written, which is the
            # last-write-wins rule the journal needs.
            out[np.asarray(self.slots, dtype=np.intp)] = np.asarray(self.values, dtype=np.float64)
            return out

    def filled(factory):
        """A store that already holds one fact per slot, and nothing more."""
        store = factory(n)
        for slot in range(n):
            store.write(slot, float(slot))
        return store

    for label, factory in (('columnar', Columnar), ('journal', Journal)):
        # A write is timed with the store rebuilt outside the clock. Timing writes
        # on one shared store would let a journal accumulate the facts of every
        # calibration round, and the later read would then measure that instead of
        # the design.
        measurements = {
            'attr_write_single': harness.time_batched(
                lambda factory=factory: filled(factory),
                lambda store, i: store.write(i % n, 1.0),
                count=64,
                samples=max(1, samples),
                warmup=0,
            ),
            'attr_read_frame': harness.time_repeat(filled(factory).frame, samples=max(1, samples)),
        }

        for op, time in measurements.items():
            rec = _record(
                'annnet',
                'attribute_options',
                op,
                scale,
                n,
                0,
                backend=backend,
                time=time,
                note=f'{label}; {n} values',
            )
            rec['status'] = 'ok'
            recs.append(rec)
    return recs
