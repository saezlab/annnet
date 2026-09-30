"""What the general path costs: construction, counts, degree, build, promotion, exploration.

What a new interpreter pays before the first graph exists (import, the first
construction and the second, each in a fresh process) is measured apart from
everything else, because none of it is warm. Four costs of a flat binary graph
are measured against networkx on 5 000 nodes
/ 20 000 edges (construct an empty graph, count the nodes, the degree of one
node, a warm build); the count is measured at ten and a million nodes; the
degree at 5k and 50k nodes; what a flat graph pays to gain aspects or a
hyperedge; and what an exploration costs at 20k and 200k edges (a composed
selection, a table preview, a view and its materialization). The recorded
numbers and the decisions that rest on them are in
``docs/explanations/performance.md``.

Every number comes from :mod:`benchmarks.harness` (warm-up, adaptive inner
loop, GC off in the timed region, a distribution rather than one sample), and
every first call is recorded separately from the warm distribution.

Run from the repository root::

    python -m benchmarks.api_costs --out docs/explanations/performance-measurements.json
    python -m benchmarks.api_costs --tree /path/to/other/checkout --label previous --sections flat_costs,promotion

``--tree`` prepends a checkout to ``sys.path`` so the same script measures
another revision; ``--compare-tree`` runs that in a subprocess and merges the
result, so one file holds both revisions under matched conditions.
"""

from __future__ import annotations

import os
import sys
import json
import time
import random
from pathlib import Path
import argparse
import warnings
import statistics
import subprocess
import tracemalloc

SECTIONS = (
    'cold_start',
    'flat_costs',
    'count_scaling',
    'degree_scaling',
    'promotion',
    'exploration',
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def edge_specs(n_nodes: int, n_edges: int, seed: int = 4) -> list[dict]:
    """A deterministic random binary edge list on ``n_nodes`` string ids."""
    rng = random.Random(seed)
    ids = [f'n{i}' for i in range(n_nodes)]
    return [
        {
            'source': ids[rng.randrange(n_nodes)],
            'target': ids[rng.randrange(n_nodes)],
            'edge_id': f'e{k}',
        }
        for k in range(n_edges)
    ]


def node_ids(n_nodes: int) -> list[str]:
    return [f'n{i}' for i in range(n_nodes)]


def build_annnet(AnnNet, n_nodes: int, n_edges: int, *, seed: int = 4):
    G = AnnNet(directed=True)
    G.add_nodes(node_ids(n_nodes))
    G.add_edges(edge_specs(n_nodes, n_edges, seed))
    return G


def build_networkx(nx, n_nodes: int, n_edges: int, *, seed: int = 4):
    G = nx.DiGraph()
    G.add_nodes_from(node_ids(n_nodes))
    G.add_edges_from(
        (s['source'], s['target'], {'edge_id': s['edge_id']})
        for s in edge_specs(n_nodes, n_edges, seed)
    )
    return G


def median_degree_node(specs: list[dict]) -> tuple[str, int]:
    """The node whose degree sits at the median of the workload, and that degree."""
    degree: dict[str, set] = {}
    for s in specs:
        degree.setdefault(s['source'], set()).add(s['edge_id'])
        degree.setdefault(s['target'], set()).add(s['edge_id'])
    ranked = sorted(degree.items(), key=lambda kv: (len(kv[1]), kv[0]))
    node, edges = ranked[len(ranked) // 2]
    return node, len(edges)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def first_call(fn) -> tuple[float, object]:
    t0 = time.perf_counter_ns()
    out = fn()
    return (time.perf_counter_ns() - t0) / 1e9, out


def peak_memory(fn) -> tuple[int, int, object]:
    """(peak bytes, retained bytes, result) of one call, by tracemalloc."""
    import gc

    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    out = fn()
    _, peak = tracemalloc.get_traced_memory()
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return max(peak - base, 0), max(current - base, 0), out


def record(name: str, stat=None, **extra) -> dict:
    row = {'name': name}
    if stat is not None:
        row['warm'] = stat.as_dict()
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def section_flat_costs(harness, AnnNet, nx, *, n_nodes: int, n_edges: int) -> list[dict]:
    """The four flat-graph costs, AnnNet against networkx, at one size."""
    rows = []
    specs = edge_specs(n_nodes, n_edges)
    node, degree = median_degree_node(specs)

    # 1. construct an empty graph
    rows.append(
        record(
            'construct_empty', harness.time_repeat(lambda: AnnNet(directed=True)), library='annnet'
        )
    )
    if nx is not None:
        rows.append(
            record('construct_empty', harness.time_repeat(lambda: nx.DiGraph()), library='networkx')
        )

    # 2. count the nodes
    G = build_annnet(AnnNet, n_nodes, n_edges)
    rows.append(
        record(
            'count_nodes',
            harness.time_repeat(lambda: len(G.N)),
            library='annnet',
            n_nodes=n_nodes,
            n_edges=n_edges,
        )
    )
    rows.append(
        record(
            'count_edges',
            harness.time_repeat(lambda: len(G.E)),
            library='annnet',
            n_nodes=n_nodes,
            n_edges=n_edges,
        )
    )
    if nx is not None:
        H = build_networkx(nx, n_nodes, n_edges)
        rows.append(
            record(
                'count_nodes',
                harness.time_repeat(lambda: len(H.nodes)),
                library='networkx',
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
        rows.append(
            record(
                'count_edges',
                harness.time_repeat(lambda: H.number_of_edges()),
                library='networkx',
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )

    # 3. degree of one node
    rows.append(
        record(
            'degree_one_node',
            harness.time_repeat(lambda: G.degree(node)),
            library='annnet',
            node=node,
            degree=degree,
            n_nodes=n_nodes,
            n_edges=n_edges,
        )
    )
    if hasattr(G, 'incident_edges'):
        rows.append(
            record(
                'incident_edges_one_node',
                harness.time_repeat(lambda: G.incident_edges(node)),
                library='annnet',
                node=node,
                degree=degree,
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
    if nx is not None:
        rows.append(
            record(
                'degree_one_node',
                harness.time_repeat(lambda: H.degree(node)),
                library='networkx',
                node=node,
                degree=degree,
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )

    # 4. build the graph, warm (the specs are built outside the clock)
    rows.append(
        record(
            'build_warm',
            harness.time_oneshot(
                lambda: build_annnet(AnnNet, n_nodes, n_edges), warmup=2, samples=5
            ),
            library='annnet',
            n_nodes=n_nodes,
            n_edges=n_edges,
        )
    )
    mem = harness.measure_memory(lambda: build_annnet(AnnNet, n_nodes, n_edges))
    rows.append(
        record(
            'build_memory',
            None,
            library='annnet',
            n_nodes=n_nodes,
            n_edges=n_edges,
            **mem.as_dict(),
        )
    )
    if nx is not None:
        rows.append(
            record(
                'build_warm',
                harness.time_oneshot(
                    lambda: build_networkx(nx, n_nodes, n_edges), warmup=2, samples=5
                ),
                library='networkx',
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
        mem = harness.measure_memory(lambda: build_networkx(nx, n_nodes, n_edges))
        rows.append(
            record(
                'build_memory',
                None,
                library='networkx',
                n_nodes=n_nodes,
                n_edges=n_edges,
                **mem.as_dict(),
            )
        )
    return rows


_COLD_START = """
import json, sys, time
{preload}
t0 = time.perf_counter()
import annnet
t1 = time.perf_counter()
annnet.AnnNet(directed=True)
t2 = time.perf_counter()
annnet.AnnNet(directed=True)
t3 = time.perf_counter()
print(json.dumps({{
    'import_s': t1 - t0,
    'first_construct_s': t2 - t1,
    'second_construct_s': t3 - t2,
    'modules_loaded': len(sys.modules),
    'dataframe_libraries_loaded': sorted(m for m in ('polars', 'pandas', 'pyarrow') if m in sys.modules),
}}))
"""


def section_cold_start(tree: str | None, *, runs: int = 9) -> list[dict]:
    """What a new interpreter pays before the first graph exists.

    Each run is a fresh ``python`` process, so nothing is warm. ``import`` is the
    package and whatever it pulls in; ``first_construct`` is the first
    ``AnnNet()``, which pays for choosing and loading a dataframe backend;
    ``second_construct`` is the same call once that has happened. The
    ``libraries_preloaded`` variant imports polars, pandas and pyarrow before the
    clock starts, which measures the constructor alone and is not what a user
    who starts a session sees.
    """
    env = dict(os.environ)
    root = str(Path(tree).resolve()) if tree else str(Path(__file__).resolve().parents[1])
    env['PYTHONPATH'] = root + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('NUMBA_CACHE_DIR', '/tmp/annnet-numba-cache')
    rows = []
    for name, preload in (
        ('cold_start', ''),
        ('cold_start_libraries_preloaded', 'import polars, pandas, pyarrow'),
    ):
        samples = []
        for _ in range(runs):
            done = subprocess.run(
                [sys.executable, '-c', _COLD_START.format(preload=preload)],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            )
            samples.append(json.loads(done.stdout.strip().splitlines()[-1]))
        row = {'name': name, 'runs': runs, 'library': 'annnet'}
        for key in ('import_s', 'first_construct_s', 'second_construct_s'):
            values = sorted(sample[key] for sample in samples)
            row[key] = {
                'min': values[0],
                'median': statistics.median(values),
                'max': values[-1],
            }
        row['modules_loaded'] = samples[0]['modules_loaded']
        row['dataframe_libraries_loaded'] = samples[0]['dataframe_libraries_loaded']
        rows.append(row)
    return rows


def section_count_scaling(harness, AnnNet, *, sizes=(10, 1_000_000)) -> list[dict]:
    """Counting the nodes of a graph of a million nodes costs what ten cost."""
    rows = []
    for n in sizes:
        G = AnnNet(directed=True)
        first, _ = first_call(lambda: G.add_nodes(node_ids(n)))
        rows.append(
            record(
                'count_nodes',
                harness.time_repeat(lambda: len(G.N)),
                n_nodes=n,
                add_nodes_first_call_s=first,
            )
        )
        rows.append(record('count_edges', harness.time_repeat(lambda: len(G.E)), n_nodes=n))
        rows.append(record('shape', harness.time_repeat(lambda: G.shape), n_nodes=n))
    return rows


def section_degree_scaling(
    harness, AnnNet, *, sizes=((5_000, 20_000), (50_000, 200_000))
) -> list[dict]:
    """The degree of one node costs its incident edges, whatever the graph's size."""
    rows = []
    for n_nodes, n_edges in sizes:
        specs = edge_specs(n_nodes, n_edges)
        node, degree = median_degree_node(specs)
        G = AnnNet(directed=True)
        G.add_nodes(node_ids(n_nodes))
        G.add_edges(specs)
        rows.append(
            record(
                'degree_one_node',
                harness.time_repeat(lambda: G.degree(node)),
                node=node,
                degree=degree,
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
        rows.append(
            record(
                'incident_edges_one_node',
                harness.time_repeat(lambda: G.incident_edges(node)),
                node=node,
                degree=degree,
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
        # A self-loop, a zero-weight edge and a hyperedge each count once.
        loop = f'{node}_loop'
        G.add_edges(node, node, edge_id=loop)
        G.add_edges(node, 'n1', edge_id=f'{node}_zero', weight=0.0, parallel='parallel')
        G.add_edges([{'members': [node, 'n2', 'n3'], 'edge_id': f'{node}_hyper'}])
        rows.append(
            record(
                'degree_one_node_with_loop_zero_hyper',
                harness.time_repeat(lambda: G.degree(node)),
                node=node,
                degree=G.degree(node),
                n_nodes=n_nodes,
                n_edges=n_edges + 3,
            )
        )
    return rows


def section_promotion(harness, AnnNet, *, n_nodes: int, n_edges: int) -> list[dict]:
    """What a flat graph pays to gain aspects or a hyperedge."""
    rows = []
    specs = edge_specs(n_nodes, n_edges)

    def flat():
        G = AnnNet(directed=True)
        G.add_nodes(node_ids(n_nodes))
        G.add_edges(specs)
        return G

    # Gaining aspects: every held placement moves to the placeholder coordinate
    # and every held edge gains its (until now implicit) role record.
    def promote():
        G = flat()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            t, _ = first_call(lambda: G.layers.set_aspects(['cond'], {'cond': ['ctrl', 'stim']}))
        return t

    times = [promote() for _ in range(5)]
    rows.append(
        record(
            'gain_aspects',
            None,
            n_nodes=n_nodes,
            n_edges=n_edges,
            first_call_s=times[0],
            samples_s=times,
            median_s=statistics.median(times),
        )
    )

    # Gaining a hyperedge on a binary graph, against one more binary edge.
    G = flat()
    t_bin, _ = first_call(
        lambda: G.add_edges('n0', 'n1', edge_id='extra_binary', parallel='parallel')
    )
    t_hyper, _ = first_call(
        lambda: G.add_edges([{'members': ['n0', 'n1', 'n2'], 'edge_id': 'first_hyper'}])
    )
    t_hyper2, _ = first_call(
        lambda: G.add_edges([{'members': ['n3', 'n4', 'n5'], 'edge_id': 'second_hyper'}])
    )
    t_deg, _ = first_call(lambda: G.degree('n0'))
    rows.append(
        record(
            'first_hyperedge',
            None,
            n_nodes=n_nodes,
            n_edges=n_edges,
            one_more_binary_edge_s=t_bin,
            first_hyperedge_s=t_hyper,
            second_hyperedge_s=t_hyper2,
            degree_after_s=t_deg,
        )
    )

    # Mixed-batch order independence: the same specs in two orders describe the
    # same graph. The summary and the per-edge kinds are the witnesses.
    mixed = specs[:200] + [
        {'members': ['n0', 'n1', 'n2'], 'edge_id': 'h0'},
        {'members': ['n3', 'n4'], 'edge_id': 'h1'},
    ]
    A = AnnNet(directed=True)
    A.add_edges(mixed)
    B = AnnNet(directed=True)
    B.add_edges(list(reversed(mixed)))
    same = dict(A.edge_kind) == dict(B.edge_kind) and set(A.E) == set(B.E) and set(A.N) == set(B.N)
    if hasattr(A, 'summary'):
        same = same and A.summary()['edge_kinds'] == B.summary()['edge_kinds']
    rows.append(
        record('mixed_batch_order_independent', None, n_edges=len(mixed), independent=bool(same))
    )
    return rows


def section_exploration(
    harness, AnnNet, *, sizes=(20_000, 200_000), density: float = 0.10
) -> list[dict]:
    """A composed selection, a preview, a view and its copy at 20k / 200k edges.

    The contextual attributes are sparse: ``density`` of the nodes carry a
    score and a group, a tenth of those a flag; one edge in ten carries a
    confidence. That is the shape of a graph a user has annotated from one
    measurement, not one with a value in every cell.
    """
    rows = []
    for n_edges in sizes:
        n_nodes = n_edges // 4
        rng = random.Random(7)
        G = AnnNet(directed=True)
        G.add_nodes(node_ids(n_nodes))
        G.add_edges(edge_specs(n_nodes, n_edges))
        annotated = rng.sample(range(n_nodes), int(n_nodes * density))
        node_rows = {
            f'n{i}': {
                'score': rng.random(),
                'group': rng.choice('xyz'),
                **({'flag': True} if rng.random() < 0.1 else {}),
            }
            for i in annotated
        }
        G.attrs.update('nodes', node_rows)
        edge_rows = {
            f'e{k}': {'confidence': rng.random()} for k in rng.sample(range(n_edges), n_edges // 10)
        }
        G.attrs.update('edges', edge_rows)

        def compose():
            return (G.N.select(score__gte=0.5) | G.N.select(group='x')) - G.N.select(flag=True)

        t_first, sel = first_call(compose)
        n_sel = len(sel)
        rows.append(
            record(
                'composed_selection',
                harness.time_repeat(lambda: len(compose())),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                selected_nodes=n_sel,
            )
        )

        one = next(iter(node_rows))
        rows.append(
            record(
                'read_one_row',
                harness.time_repeat(lambda: G.attrs.row('nodes', one)['score']),
                n_nodes=n_nodes,
                n_edges=n_edges,
            )
        )
        t_first, preview = first_call(lambda: G.attrs.table('nodes', limit=20, backend='pandas'))
        rows.append(
            record(
                'table_preview_nodes',
                harness.time_repeat(lambda: G.attrs.table('nodes', limit=20, backend='pandas')),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                rows=int(len(preview)),
            )
        )
        t_first, preview = first_call(
            lambda: G.attrs.table('edges', derived=True, limit=20, backend='pandas')
        )
        rows.append(
            record(
                'table_preview_edges_derived',
                harness.time_repeat(
                    lambda: G.attrs.table('edges', derived=True, limit=20, backend='pandas')
                ),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                rows=int(len(preview)),
            )
        )

        def resolve_view():
            V = G.view(nodes=compose(), boundary='open')
            return len(V.E.ids)

        t_first, n_view_edges = first_call(resolve_view)
        rows.append(
            record(
                'view_open_boundary_edges',
                harness.time_repeat(resolve_view, warmup=1, samples=5),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                selected_nodes=n_sel,
                view_edges=n_view_edges,
            )
        )

        V = G.view(nodes=compose(), boundary='open')
        t_first, table = first_call(
            lambda: V.attrs.table('edges', derived=True, limit=20, backend='pandas')
        )
        rows.append(
            record(
                'view_table_preview_edges',
                harness.time_repeat(
                    lambda: V.attrs.table('edges', derived=True, limit=20, backend='pandas'),
                    warmup=1,
                    samples=5,
                ),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                rows=int(len(table)),
            )
        )
        t_first, H = first_call(V.materialize)
        peak, retained, _ = peak_memory(V.materialize)
        rows.append(
            record(
                'materialize',
                harness.time_oneshot(V.materialize, warmup=1, samples=3),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
                copy_nodes=len(H.N),
                copy_edges=len(H.E),
                peak_bytes=peak,
                retained_bytes=retained,
            )
        )
        t_first, summary = first_call(G.summary)
        rows.append(
            record(
                'summary',
                harness.time_repeat(G.summary, warmup=1, samples=5),
                n_nodes=n_nodes,
                n_edges=n_edges,
                first_call_s=t_first,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(sections, *, n_nodes: int, n_edges: int, exploration_sizes, label: str, tree=None) -> dict:
    import annnet
    from annnet import AnnNet
    from benchmarks import environment
    import benchmarks.harness as harness

    try:
        import networkx as nx
    except ImportError:  # pragma: no cover
        nx = None

    out = {
        'label': label,
        'annnet_version': getattr(annnet, '__version__', None),
        'environment': environment.capture(),
        'workload': {'n_nodes': n_nodes, 'n_edges': n_edges, 'seed': 4},
        'sections': {},
    }
    if 'cold_start' in sections:
        out['sections']['cold_start'] = section_cold_start(tree)
    if 'flat_costs' in sections:
        out['sections']['flat_costs'] = section_flat_costs(
            harness, AnnNet, nx, n_nodes=n_nodes, n_edges=n_edges
        )
    if 'count_scaling' in sections:
        out['sections']['count_scaling'] = section_count_scaling(harness, AnnNet)
    if 'degree_scaling' in sections:
        out['sections']['degree_scaling'] = section_degree_scaling(harness, AnnNet)
    if 'promotion' in sections:
        out['sections']['promotion'] = section_promotion(
            harness, AnnNet, n_nodes=n_nodes, n_edges=n_edges
        )
    if 'exploration' in sections:
        out['sections']['exploration'] = section_exploration(
            harness, AnnNet, sizes=exploration_sizes
        )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--out', type=Path, default=None, help='JSON file to write (default: stdout)'
    )
    parser.add_argument(
        '--tree',
        type=Path,
        default=None,
        help='a checkout to measure instead of the installed package',
    )
    parser.add_argument('--label', default='working', help='label for this tree in the record')
    parser.add_argument(
        '--compare-tree',
        type=Path,
        default=None,
        help='another checkout, measured in a subprocess and merged',
    )
    parser.add_argument('--compare-label', default='head')
    parser.add_argument('--compare-sections', default='flat_costs,promotion')
    parser.add_argument('--sections', default=','.join(SECTIONS))
    parser.add_argument('--nodes', type=int, default=5_000)
    parser.add_argument('--edges', type=int, default=20_000)
    parser.add_argument('--exploration-edges', default='20000,200000')
    args = parser.parse_args(argv)

    if args.tree is not None:
        sys.path.insert(0, str(args.tree.resolve()))
        for name in [m for m in sys.modules if m == 'annnet' or m.startswith('annnet.')]:
            del sys.modules[name]

    sections = [s.strip() for s in args.sections.split(',') if s.strip()]
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        parser.error(f'unknown sections {sorted(unknown)}; choose from {SECTIONS}')
    exploration_sizes = tuple(int(x) for x in args.exploration_edges.split(',') if x.strip())

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = run(
            sections,
            n_nodes=args.nodes,
            n_edges=args.edges,
            exploration_sizes=exploration_sizes,
            label=args.label,
            tree=args.tree,
        )
    payload = {'trees': {args.label: result}}

    if args.compare_tree is not None:
        cmd = [
            sys.executable,
            '-m',
            'benchmarks.api_costs',
            '--tree',
            str(args.compare_tree),
            '--label',
            args.compare_label,
            '--sections',
            args.compare_sections,
            '--nodes',
            str(args.nodes),
            '--edges',
            str(args.edges),
        ]
        env = dict(os.environ)
        env.setdefault('NUMBA_CACHE_DIR', '/tmp/annnet-numba-cache')
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
        if proc.returncode != 0:
            payload['trees'][args.compare_label] = {'error': proc.stderr[-4000:]}
        else:
            payload['trees'].update(json.loads(proc.stdout)['trees'])
        # The arguments of the run, with the paths a machine chose left out: the
        # record says what was measured, not where it was checked out.
        path_options = ('--out', '--tree', '--compare-tree')
        recorded = ['python -m benchmarks.api_costs']
        held = iter(sys.argv[1:])
        for token in held:
            recorded.append(token)
            if token in path_options:
                next(held, None)
                recorded.append(f'<{token.lstrip("-")}>')
        payload['command'] = ' '.join(recorded)

    text = json.dumps(payload, indent=2, default=str)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + '\n')
        print(f'wrote {args.out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    sys.exit(main())
