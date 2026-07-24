"""AnnNet spec-sheet benchmark: speed + memory across scales and edge types.

Produces a compact PDF (tables only) covering:

1. Head-to-head binary-graph ops (build + core queries) — AnnNet vs NetworkX,
   igraph and graph-tool, across three scales up to 1M vertices / 4M edges.
2. Memory footprint of the built binary graph for the same engines/scales.
3. AnnNet-only build time + memory for every edge type it can express
   (binary, vertex-edge / edge-node, hyperedge, multilayer), across scales.

Design for a machine with limited RAM:

* Every single measurement runs in an isolated subprocess so the OS reclaims
  memory between builds and each cell starts from a clean baseline.
* ``tracemalloc`` (precise retained bytes) is used only where the graph is small
  enough to trace cheaply; above ``MEM_TRACEMALLOC_MAX_EDGES`` we fall back to a
  cheap process-RSS delta so a multi-million-edge build never OOMs on tracing
  overhead. The memory method is recorded and shown in the report.

Run:  python -m benchmarks.reporting.specsheet            # full run + PDF
      python -m benchmarks.reporting.specsheet --quick    # small scales, fast check
"""

from __future__ import annotations

import gc
import os
import sys
import json
import time
from pathlib import Path
import argparse
import subprocess

# ---------------------------------------------------------------------------
# Scale + config
# ---------------------------------------------------------------------------
# The scale ladder is derived from the shared benchmarks.scales source so the
# spec sheet and the comparison suite (benchmarks.run) never drift apart.
DEFAULT_SCALE_KEYS = ['medium', 'large', 'xlarge']  # 10K / 100K / 1M vertices
QUICK_SCALE_KEYS = ['small', 'medium']  # fast layout check


def _scales(keys: list[str]) -> list[dict]:
    from ..scales import SCALES as _SHARED

    return [
        {'name': _si(_SHARED[k].vertices), 'v': _SHARED[k].vertices, 'e': _SHARED[k].edges}
        for k in keys
    ]


# Above this edge count, skip tracemalloc (tracing overhead is too heavy/RAM-hungry)
# and use a cheap RSS delta instead.
MEM_TRACEMALLOC_MAX_EDGES = 500_000

ENGINES = ['annnet', 'networkx', 'igraph', 'graph-tool']
EDGE_TYPES = ['binary', 'vertex_edge', 'hyper', 'multilayer']

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'results'


def _scale_samples(n_edges: int) -> tuple[int, int]:
    """(warmup, samples) for one-shot build timing, scaled to build cost."""
    if n_edges <= 50_000:
        return 1, 5
    if n_edges <= 800_000:
        return 1, 3
    return 0, 1


# ---------------------------------------------------------------------------
# Worker: run ONE measurement in this (isolated) process, print JSON to stdout
# ---------------------------------------------------------------------------
def _measure_memory(build, n_edges: int) -> dict:
    """Retained footprint of one build; tracemalloc when cheap, else RSS delta."""
    from .. import harness

    if n_edges <= MEM_TRACEMALLOC_MAX_EDGES:
        m = harness.measure_memory(build)
        return {
            'retained_bytes': m.retained_bytes,
            'peak_bytes': m.peak_bytes,
            'rss_delta_bytes': m.rss_delta_bytes,
            'method': 'tracemalloc',
        }
    # RSS-only path for very large builds.
    import psutil

    proc = psutil.Process(os.getpid())
    gc.collect()
    rss0 = proc.memory_info().rss
    obj = build()
    gc.collect()
    rss1 = proc.memory_info().rss
    retained = max(rss1 - rss0, 0)
    del obj
    gc.collect()
    return {
        'retained_bytes': retained,
        'peak_bytes': None,
        'rss_delta_bytes': retained,
        'method': 'rss',
    }


def _annnet_type_build(kind: str, n_v: int, n_e: int):
    """Return (build_fn, actual_n_v, actual_n_e, note) for an AnnNet edge type."""
    from annnet.core.graph import AnnNet

    if kind == 'binary':
        edges = [
            {'source': f'v{i % n_v}', 'target': f'v{(i + 1) % n_v}', 'weight': 1.0}
            for i in range(n_e)
        ]

        def build():
            G = AnnNet(directed=True)
            G.add_edges(iter(edges))
            return G

        return build, n_v, n_e, 'directed binary edges'

    if kind == 'vertex_edge':
        # Edge-node edges: each edge is also registered as an entity (as_entity).
        m = max(1, n_e // 2)  # heavier per edge; use ~half the edge budget
        edges = [
            {'source': f'v{i % n_v}', 'target': f'v{(i + 1) % n_v}', 'weight': 1.0}
            for i in range(m)
        ]

        def build():
            G = AnnNet(directed=True)
            G.add_edges(iter(edges), as_entity=True)
            return G

        return build, n_v, m, 'edge-node edges (as_entity)'

    if kind == 'hyper':
        arity = 4
        m = max(1, n_e // arity)  # arity-4 -> ~n_e incidences for a fair per-node cost
        edges = [
            {
                'source': [f'v{(i + j) % n_v}' for j in range(arity - 1)],
                'target': [f'v{(i + arity) % n_v}'],
                'weight': 1.0,
            }
            for i in range(m)
        ]

        def build():
            G = AnnNet(directed=True)
            G.add_edges(iter(edges))
            return G

        return build, n_v, m, f'arity-{arity} directed hyperedges'

    if kind == 'multilayer':
        import warnings

        L = 4
        base = max(1, n_v // L)
        layer_vals = [f'L{k}' for k in range(L)]

        def build():
            G = AnnNet(directed=True)
            G.layers.set_aspects(['layer'], {'layer': layer_vals})
            coords = list(G.layers.iter_layers())
            edges = []
            for aa in coords:
                for i in range(base):
                    edges.append(
                        {
                            'source': (f'm{i}', aa),
                            'target': (f'm{(i + 1) % base}', aa),
                            'weight': 1.0,
                        }
                    )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                G.add_edges(edges)
            return G

        return build, base * L, base * L, f'{L} layers, intra-layer edges'

    raise ValueError(f'unknown edge type {kind!r}')


def run_worker(task: dict) -> dict:
    from .. import engines, harness

    kind = task['kind']
    scale = task['scale']
    n_v, n_e = scale['v'], scale['e']
    warmup, samples = _scale_samples(n_e)

    if kind == 'cmp':
        engine_name = task['engine']
        if engine_name == 'annnet':
            engine = engines.AnnNetEngine(backend='auto')
        else:
            engine = engines.engine_by_name(engine_name)
        data = engines.make_data(n_v, n_e)
        build = engine.build_factory(data)

        out = {'kind': 'cmp', 'engine': engine_name, 'scale': scale['name'], 'n_v': n_v, 'n_e': n_e}
        out['build'] = harness.time_oneshot(build, warmup=warmup, samples=samples).as_dict()
        out['memory'] = _measure_memory(build, n_e)
        handle = build()
        q = {}
        for op, fn in engine.query_ops(handle, data).items():
            q[op] = harness.time_repeat(fn).as_dict()
        out['queries'] = q
        del handle
        return out

    if kind == 'annnet_type':
        etype = task['etype']
        build, an_v, an_e, note = _annnet_type_build(etype, n_v, n_e)
        out = {
            'kind': 'annnet_type',
            'etype': etype,
            'scale': scale['name'],
            'n_v': an_v,
            'n_e': an_e,
            'note': note,
        }
        out['build'] = harness.time_oneshot(build, warmup=warmup, samples=samples).as_dict()
        out['memory'] = _measure_memory(build, an_e)
        return out

    raise ValueError(f'unknown task kind {kind!r}')


# ---------------------------------------------------------------------------
# Orchestrator: spawn one subprocess per measurement, collect JSON
# ---------------------------------------------------------------------------
def _spawn(task: dict) -> dict:
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    env['OMP_NUM_THREADS'] = '1'
    payload = json.dumps(task)
    label = task.get('engine') or task.get('etype')
    print(f'  · {task["kind"]:11s} {label:12s} {task["scale"]["name"]:5s} ...', end='', flush=True)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, '-m', 'benchmarks.reporting.specsheet', '--worker', payload],
        env=env,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f' FAILED ({dt:.1f}s)')
        sys.stderr.write(proc.stderr[-2000:] + '\n')
        return {**task, 'error': proc.stderr[-500:]}
    line = [ln for ln in proc.stdout.splitlines() if ln.startswith('{')][-1]
    print(f' {dt:6.1f}s')
    return json.loads(line)


def orchestrate(scales: list[dict]) -> dict:
    results = {'cmp': [], 'annnet_type': [], 'scales': scales}
    print('Head-to-head (binary) + memory:')
    for sc in scales:
        for eng in ENGINES:
            results['cmp'].append(_spawn({'kind': 'cmp', 'engine': eng, 'scale': sc}))
    print('AnnNet edge types:')
    for sc in scales:
        for et in EDGE_TYPES:
            results['annnet_type'].append(_spawn({'kind': 'annnet_type', 'etype': et, 'scale': sc}))
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_time(s: float | None) -> str:
    if s is None:
        return '—'
    if s < 1e-6:
        return f'{s * 1e9:.0f} ns'
    if s < 1e-3:
        return f'{s * 1e6:.1f} µs'
    if s < 1.0:
        return f'{s * 1e3:.1f} ms'
    return f'{s:.2f} s'


def _fmt_bytes(b: float | None) -> str:
    if b is None:
        return '—'
    for unit, thr in (('GB', 1024**3), ('MB', 1024**2), ('KB', 1024)):
        if b >= thr:
            return f'{b / thr:.1f} {unit}'
    return f'{b:.0f} B'


def _median(d: dict | None) -> float | None:
    return d.get('median_s') if d else None


# ---------------------------------------------------------------------------
# PDF report
# ---------------------------------------------------------------------------
def render_pdf(results: dict, out_path: Path) -> None:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    scales = results['scales']
    cmp = {(r['engine'], r['scale']): r for r in results['cmp'] if 'error' not in r}
    types = {(r['etype'], r['scale']): r for r in results['annnet_type'] if 'error' not in r}

    env = _capture_env()

    def _table(ax, title, subtitle, col_labels, rows, col_widths=None):
        ax.axis('off')
        # Title + subtitle occupy the top strip; the table fills the rest via bbox
        # so rows never collide with the heading and there is no dead whitespace.
        ax.text(0, 1.0, title, transform=ax.transAxes, fontsize=11, fontweight='bold', va='top')
        if subtitle:
            ax.text(
                0, 0.885, subtitle, transform=ax.transAxes, fontsize=6.6, va='top', color='#555555'
            )
        tbl = ax.table(
            cellText=rows,
            colLabels=col_labels,
            cellLoc='center',
            colWidths=col_widths,
            bbox=[0, 0, 1, 0.80],
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        for (r, _c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor('#1f3b57')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f4f6f8')
        return tbl

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))  # US-Letter landscape
        fig.suptitle(
            'AnnNet — Speed & Memory Spec Sheet', x=0.5, y=0.988, fontsize=15, fontweight='bold'
        )
        fig.text(
            0.5,
            0.953,
            'incidence-graph library vs NetworkX, igraph & graph-tool · '
            'one isolated build per cell · median of adaptive-repeat timing',
            ha='center',
            fontsize=8,
            color='#444444',
        )
        fig.text(0.5, 0.02, env, ha='center', fontsize=6.2, color='#666666')

        # Row-count-proportional heights so row bands line up across tables.
        gs = fig.add_gridspec(
            3,
            1,
            height_ratios=[7.5, 7.5, 13.5],
            left=0.03,
            right=0.97,
            top=0.935,
            bottom=0.05,
            hspace=0.42,
        )

        # ---- Table 1: head-to-head binary ops -----------------------------
        ax1 = fig.add_subplot(gs[0])
        ops = [
            ('build', 'build'),
            ('degree', 'degree'),
            ('neighbors', 'neighbors'),
            ('has_edge', 'has_edge'),
            ('enumerate_edges', 'enumerate'),
        ]
        col_labels = ['scale (V / E)', 'engine'] + [lbl for _k, lbl in ops]
        rows = []
        for sc in scales:
            for eng in ENGINES:
                r = cmp.get((eng, sc['name']))
                v_e = f'{sc["name"]}  ({_si(sc["v"])} / {_si(sc["e"])})'
                cells = [v_e if eng == ENGINES[0] else '', eng]
                if r:
                    cells.append(_fmt_time(_median(r['build'])))
                    for k, _lbl in ops[1:]:
                        cells.append(_fmt_time(_median(r['queries'].get(k))))
                else:
                    cells += ['—'] * len(ops)
                rows.append(cells)
        _table(
            ax1,
            '1 · Binary graph — head-to-head (build + core queries)',
            'Median per-call time. Same vertex/edge input fed to all comparable engines.',
            col_labels,
            rows,
            col_widths=[0.20, 0.11] + [0.138] * 5,
        )

        # ---- Table 2: memory (binary) -------------------------------------
        ax2 = fig.add_subplot(gs[1])
        col_labels2 = ['scale (V / E)', 'engine', 'footprint', 'bytes/edge', 'method']
        rows2 = []
        for sc in scales:
            for eng in ENGINES:
                r = cmp.get((eng, sc['name']))
                v_e = f'{sc["name"]}  ({_si(sc["v"])} / {_si(sc["e"])})'
                if r and r.get('memory'):
                    mem = r['memory']
                    bpe = mem['retained_bytes'] / sc['e'] if sc['e'] else 0
                    rows2.append(
                        [
                            v_e if eng == ENGINES[0] else '',
                            eng,
                            _fmt_bytes(mem['retained_bytes']),
                            f'{bpe:.1f}',
                            mem['method'],
                        ]
                    )
                else:
                    rows2.append([v_e if eng == ENGINES[0] else '', eng, '—', '—', '—'])
        _table(
            ax2,
            '2 · Binary graph — memory footprint',
            'Retained allocation of the built graph. tracemalloc where cheap, '
            'process-RSS delta above 500K edges (heavier builds).',
            col_labels2,
            rows2,
            col_widths=[0.26, 0.14, 0.18, 0.18, 0.16],
        )

        # ---- Table 3: AnnNet edge types -----------------------------------
        ax3 = fig.add_subplot(gs[2])
        col_labels3 = [
            'scale',
            'edge type',
            'V (or supra)',
            'E',
            'build',
            'footprint',
            'bytes/edge',
            'mem method',
        ]
        rows3 = []
        pretty = {
            'binary': 'binary',
            'vertex_edge': 'vertex-edge (edge-node)',
            'hyper': 'hyperedge (arity-4)',
            'multilayer': 'multilayer (4 layers)',
        }
        for sc in scales:
            for et in EDGE_TYPES:
                r = types.get((et, sc['name']))
                first = et == EDGE_TYPES[0]
                if r:
                    mem = r.get('memory') or {}
                    bpe = (mem.get('retained_bytes') or 0) / r['n_e'] if r['n_e'] else 0
                    rows3.append(
                        [
                            sc['name'] if first else '',
                            pretty[et],
                            _si(r['n_v']),
                            _si(r['n_e']),
                            _fmt_time(_median(r['build'])),
                            _fmt_bytes(mem.get('retained_bytes')),
                            f'{bpe:.1f}',
                            mem.get('method', '—'),
                        ]
                    )
                else:
                    rows3.append(
                        [sc['name'] if first else '', pretty[et], '—', '—', '—', '—', '—', '—']
                    )
        _table(
            ax3,
            '3 · AnnNet edge types — build time + memory (all scales)',
            'Capabilities NetworkX / igraph / graph-tool cannot express. Hyperedges arity-4; '
            'multilayer = 4 layers, intra-layer ring edges.',
            col_labels3,
            rows3,
            col_widths=[0.08, 0.22, 0.12, 0.11, 0.12, 0.13, 0.12, 0.12],
        )

        pdf.savefig(fig)
        fig.savefig(out_path.with_suffix('.png'), dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Graphical report (charts)
# ---------------------------------------------------------------------------
ENGINE_COLORS = {
    'annnet': '#1f3b57',
    'networkx': '#d1495b',
    'igraph': '#2a9d8f',
    'graph-tool': '#8e5ea2',
}
TYPE_COLORS = {
    'binary': '#1f3b57',
    'vertex_edge': '#2a9d8f',
    'hyper': '#e08b3b',
    'multilayer': '#8e5ea2',
}
TYPE_LABELS = {
    'binary': 'binary',
    'vertex_edge': 'vertex-edge',
    'hyper': 'hyperedge (k=4)',
    'multilayer': 'multilayer (4L)',
}


def render_charts(results: dict, out_path: Path) -> None:
    """One-page graphical companion to the spec sheet (log-scaled charts)."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.backends.backend_pdf import PdfPages

    scales = results['scales']
    scale_names = [s['name'] for s in scales]
    edges_by_scale = [s['e'] for s in scales]
    cmp = {(r['engine'], r['scale']): r for r in results['cmp'] if 'error' not in r}
    types = {(r['etype'], r['scale']): r for r in results['annnet_type'] if 'error' not in r}
    env = _capture_env()

    with PdfPages(out_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(
            'AnnNet — Speed & Memory (graphical)', x=0.5, y=0.985, fontsize=15, fontweight='bold'
        )
        fig.text(
            0.5,
            0.951,
            'AnnNet vs NetworkX, igraph & graph-tool · log scales · one isolated build per point',
            ha='center',
            fontsize=8,
            color='#444444',
        )
        fig.text(0.5, 0.02, env, ha='center', fontsize=6.2, color='#666666')

        gs = fig.add_gridspec(
            2, 2, left=0.07, right=0.975, top=0.9, bottom=0.09, hspace=0.34, wspace=0.22
        )

        # --- (0,0) Build time vs scale (log-log) ---------------------------
        ax = fig.add_subplot(gs[0, 0])
        for eng in ENGINES:
            ys = [_median((cmp.get((eng, sn)) or {}).get('build')) for sn in scale_names]
            if not any(y is not None for y in ys):
                continue
            ax.plot(edges_by_scale, ys, marker='o', color=ENGINE_COLORS[eng], label=eng, lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Build time vs graph size', fontsize=10, fontweight='bold')
        ax.set_xlabel('edges')
        ax.set_ylabel('build time (s)')
        ax.set_xticks(edges_by_scale)
        ax.set_xticklabels([f'{_si(e)}' for e in edges_by_scale])
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=8, frameon=False)

        # --- (0,1) Memory footprint vs scale (log-log) ---------------------
        ax = fig.add_subplot(gs[0, 1])
        for eng in ENGINES:
            ys = [
                ((cmp.get((eng, sn)) or {}).get('memory') or {}).get('retained_bytes')
                for sn in scale_names
            ]
            ys = [(y / 1024**2 if y else None) for y in ys]
            if not any(y is not None for y in ys):
                continue
            ax.plot(edges_by_scale, ys, marker='s', color=ENGINE_COLORS[eng], label=eng, lw=2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('Memory footprint vs graph size', fontsize=10, fontweight='bold')
        ax.set_xlabel('edges')
        ax.set_ylabel('retained (MB)')
        ax.set_xticks(edges_by_scale)
        ax.set_xticklabels([f'{_si(e)}' for e in edges_by_scale])
        ax.grid(True, which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=8, frameon=False)

        # --- (1,0) Query latency at the largest scale (grouped bars) -------
        ax = fig.add_subplot(gs[1, 0])
        big = scale_names[-1]
        ops = ['degree', 'neighbors', 'has_edge', 'enumerate_edges']
        op_labels = ['degree', 'neighbors', 'has_edge', 'enumerate']
        import numpy as np

        x = np.arange(len(ops))
        w = min(0.8 / max(1, len(ENGINES)), 0.22)
        for i, eng in enumerate(ENGINES):
            r = cmp.get((eng, big))
            ys = []
            for op in ops:
                v = _median((r or {}).get('queries', {}).get(op)) if r else None
                ys.append(v * 1e6 if v else np.nan)  # microseconds
            if all(np.isnan(y) for y in ys):
                continue
            offset = (i - (len(ENGINES) - 1) / 2) * w
            ax.bar(x + offset, ys, w, color=ENGINE_COLORS[eng], label=eng)
        ax.set_yscale('log')
        ax.set_title(
            f'Query latency @ {big} ({_si(edges_by_scale[-1])} edges)',
            fontsize=10,
            fontweight='bold',
        )
        ax.set_ylabel('time per call (µs)')
        ax.set_xticks(x)
        ax.set_xticklabels(op_labels, fontsize=8)
        ax.grid(True, axis='y', which='both', ls=':', alpha=0.4)
        ax.legend(fontsize=8, frameon=False)

        # --- (1,1) AnnNet edge-type build time across scales ---------------
        ax = fig.add_subplot(gs[1, 1])
        x = np.arange(len(scale_names))
        w = 0.2
        for i, et in enumerate(EDGE_TYPES):
            ys = [_median((types.get((et, sn)) or {}).get('build')) for sn in scale_names]
            ys = [(y if y else np.nan) for y in ys]
            ax.bar(x + (i - 1.5) * w, ys, w, color=TYPE_COLORS[et], label=TYPE_LABELS[et])
        ax.set_yscale('log')
        ax.set_title('AnnNet build time by edge type', fontsize=10, fontweight='bold')
        ax.set_ylabel('build time (s)')
        ax.set_xticks(x)
        ax.set_xticklabels(scale_names)
        ax.grid(True, axis='y', which='both', ls=':', alpha=0.4)
        ax.legend(
            fontsize=7,
            frameon=False,
            ncol=2,
            handles=[Patch(color=TYPE_COLORS[et], label=TYPE_LABELS[et]) for et in EDGE_TYPES],
        )

        pdf.savefig(fig)
        fig.savefig(out_path.with_suffix('.png'), dpi=150)
        plt.close(fig)


def _si(n: int) -> str:
    if n >= 1_000_000:
        return f'{n / 1_000_000:.0f}M' if n % 1_000_000 == 0 else f'{n / 1_000_000:.1f}M'
    if n >= 1_000:
        return f'{n / 1_000:.0f}K' if n % 1_000 == 0 else f'{n / 1_000:.1f}K'
    return str(n)


def _capture_env() -> str:
    import platform

    try:
        from .. import environment

        cap = environment.capture()
        parts = [
            f'commit {cap.get("git_commit", "?")}',
            f'python {cap.get("python", platform.python_version())}',
            cap.get('platform', platform.platform()),
        ]
        libs = cap.get('libraries', {}) or {}
        lib_str = ' · '.join(
            f'{k} {libs[k]}'
            for k in ('networkx', 'igraph', 'graph_tool', 'numpy', 'scipy')
            if k in libs
        )
        if lib_str:
            parts.append(lib_str)
        parts.append(f'generated {cap.get("timestamp_utc", "")[:19]}')
        return '  |  '.join(p for p in parts if p)
    except Exception:
        return platform.platform()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', help=argparse.SUPPRESS)
    ap.add_argument('--quick', action='store_true', help='tiny scales for a fast check')
    ap.add_argument('--out', default=None, help='output PDF path')
    args = ap.parse_args()

    if args.worker:
        result = run_worker(json.loads(args.worker))
        print(json.dumps(result))
        return

    scales = _scales(QUICK_SCALE_KEYS if args.quick else DEFAULT_SCALE_KEYS)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / ('SPEC_SHEET_quick.pdf' if args.quick else 'SPEC_SHEET.pdf')
    )

    print(f'AnnNet spec-sheet | scales={[s["name"] for s in scales]}')
    results = orchestrate(scales)

    json_path = out_pdf.with_suffix('.json')
    json_path.write_text(json.dumps(results, indent=2))
    render_pdf(results, out_pdf)
    charts_pdf = out_pdf.with_name(out_pdf.stem + '_CHARTS.pdf')
    render_charts(results, charts_pdf)
    print(f'\nRaw JSON   -> {json_path}')
    print(f'Spec PDF   -> {out_pdf}')
    print(f'Charts PDF -> {charts_pdf}')


if __name__ == '__main__':
    main()
