"""Orchestrator + worker for the benchmark suite.

Run it as a module from the repository root::

    python -m benchmarks.run                 # full tier
    python -m benchmarks.run --tier quick     # tiny + small, fast
    python -m benchmarks.run --tier full      # tiny + xsmall + small + medium
    python -m benchmarks.run --tier heavy     # + large (100k V / 400k E)
    python -m benchmarks.run --tier huge      # + xlarge (1M V / 4M E)
    python -m benchmarks.run --backends polars,pandas

The parent process fans out one **subprocess per (engine, backend)** so that each
measurement runs in a clean interpreter with its own import graph and memory
baseline. Partial JSON results are merged, stamped with environment metadata, and
rendered to Markdown (+ scaling plots) as the SSoT report.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import argparse
import tempfile
import subprocess

# Support both `python -m benchmarks.run` and direct execution.
if __package__ in (None, ''):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from benchmarks import cases, engines, workloads, environment
    from benchmarks.scales import SCALES
    from benchmarks.reporting import render as render_report
else:
    from . import cases, engines, workloads, environment
    from .scales import SCALES
    from .reporting import render as render_report

TIERS = {
    'quick': ['tiny', 'small'],
    'full': ['tiny', 'xsmall', 'small', 'medium'],
    'heavy': ['tiny', 'xsmall', 'small', 'medium', 'large'],
    'huge': ['tiny', 'xsmall', 'small', 'medium', 'large', 'xlarge'],
}
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(__file__).resolve().parent / 'results'


# ---------------------------------------------------------------------------
# Worker: runs one engine, one backend, over the given scales.
# ---------------------------------------------------------------------------
def _worker(args) -> int:
    scales = [SCALES[s] for s in args.scales.split(',')]
    records: list[dict] = []

    if args.engine == 'annnet':
        eng = engines.AnnNetEngine(backend=args.backend)
        if not eng.available():
            print('[worker] annnet unavailable', file=sys.stderr)
            return 2
        for sc in scales:
            records += workloads.comparable(
                eng,
                sc,
                backend=args.backend,
                do_memory=not args.no_memory,
            )
            records += workloads.annnet_only(sc, backend=args.backend)
            records += workloads.annnet_features(sc, backend=args.backend)
            if not args.no_extra:
                records += cases.extra_dimensions(
                    sc,
                    backend=args.backend,
                    max_vertices=args.extra_max_vertices,
                    max_edges=args.extra_max_edges,
                    max_accessor_repeats=args.extra_max_accessor_repeats,
                )
    else:
        eng = engines.engine_by_name(args.engine)
        if not eng.available():
            print(f'[worker] {args.engine} unavailable', file=sys.stderr)
            return 2
        for sc in scales:
            records += workloads.comparable(
                eng,
                sc,
                backend=None,
                do_memory=not args.no_memory,
            )

    Path(args.out).write_text(json.dumps(records))
    return 0


# ---------------------------------------------------------------------------
# Parent: fan out workers, merge, report.
# ---------------------------------------------------------------------------
def _spawn(engine, backend, scales, no_memory, no_extra, extra_caps, tmp) -> list[dict]:
    out = tmp / f'part_{engine}_{backend}.json'
    cmd = [
        sys.executable,
        '-m',
        'benchmarks.run',
        '--worker',
        '--engine',
        engine,
        '--scales',
        ','.join(scales),
        '--out',
        str(out),
    ]
    if backend:
        cmd += ['--backend', backend]
    if no_memory:
        cmd += ['--no-memory']
    if no_extra:
        cmd += ['--no-extra']
    if extra_caps:
        max_vertices, max_edges, max_accessor_repeats = extra_caps
        cmd += [
            '--extra-max-vertices',
            str(max_vertices),
            '--extra-max-edges',
            str(max_edges),
            '--extra-max-accessor-repeats',
            str(max_accessor_repeats),
        ]

    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'

    label = f'{engine}/{backend or "-"}'
    print(f'  -> {label} [{", ".join(scales)}]', flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f'     SKIPPED ({label}): {proc.stderr.strip().splitlines()[-1:]}', flush=True)
        return []
    return json.loads(out.read_text())


def _spawn_json(module, key, tmp):
    """Run an io/adapter benchmark module in a subprocess and return its records."""
    out = tmp / f'part_{key}.json'
    env = dict(os.environ)
    env['MPLBACKEND'] = 'Agg'
    print(f'  -> {key}', flush=True)
    proc = subprocess.run(
        [sys.executable, '-m', module, '--json-out', str(out)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(f'     SKIPPED ({key}): {proc.stderr.strip().splitlines()[-1:]}', flush=True)
        return []
    return json.loads(out.read_text()).get(key, [])


def _parent(args) -> int:
    scales = args.scales.split(',') if args.scales else TIERS[args.tier]
    backends = [b for b in args.backends.split(',') if b]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'AnnNet benchmark | tier={args.tier} scales={scales} backends={backends}')

    all_records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix='annnet_bench_') as td:
        tmp = Path(td)
        # AnnNet across backends
        for backend in backends:
            all_records += _spawn(
                'annnet',
                backend,
                scales,
                args.no_memory,
                args.no_extra,
                (
                    args.extra_max_vertices,
                    args.extra_max_edges,
                    args.extra_max_accessor_repeats,
                ),
                tmp,
            )
        # Baselines once (backend independent)
        for engine in engines.BASELINE_ENGINE_NAMES:
            all_records += _spawn(
                engine,
                None,
                scales,
                args.no_memory,
                args.no_extra,
                None,
                tmp,
            )

    io_recs, adapter_recs = [], []
    if not args.no_io:
        with tempfile.TemporaryDirectory(prefix='annnet_ioadp_') as td2:
            tmp2 = Path(td2)
            print('IO formats + adapters:')
            io_recs = _spawn_json('benchmarks.io_formats', 'io_formats', tmp2)
            adapter_recs = _spawn_json('benchmarks.adapters', 'adapters', tmp2)

    payload = {
        'environment': environment.capture(),
        'config': {
            'tier': args.tier,
            'scales': scales,
            'backends': backends,
            'extra_dimensions': not args.no_extra,
            'extra_caps': {
                'max_vertices': args.extra_max_vertices,
                'max_edges': args.extra_max_edges,
                'max_accessor_repeats': args.extra_max_accessor_repeats,
            },
        },
        'records': all_records,
        'io_formats': io_recs,
        'adapters': adapter_recs,
    }
    results_path = out_dir / 'results.json'
    results_path.write_text(json.dumps(payload, indent=2))
    print(f'\nRaw results  -> {results_path}  ({len(all_records)} records)')

    report_path = out_dir / 'REPORT.md'
    render_report(payload, report_path, plots_dir=out_dir / 'plots')
    print(f'SSoT report  -> {report_path}')
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='AnnNet benchmark (SSoT)')
    p.add_argument('--tier', choices=list(TIERS), default='full')
    p.add_argument('--scales', default='', help='comma list, overrides --tier')
    p.add_argument(
        '--backends', default='auto', help='AnnNet annotation backends (polars,pandas,pyarrow,auto)'
    )
    p.add_argument('--out', default=str(DEFAULT_OUT), help='output directory')
    p.add_argument('--no-memory', action='store_true', help='skip memory passes')
    p.add_argument(
        '--no-io', action='store_true', help='skip the IO-format + adapter conversion sections'
    )
    p.add_argument('--no-extra', action='store_true', help='skip local extra benchmark dimensions')
    p.add_argument(
        '--extra-max-vertices',
        type=int,
        default=2_500,
        help='cap local extra-dimension workloads to this many vertices',
    )
    p.add_argument(
        '--extra-max-edges',
        type=int,
        default=10_000,
        help='cap local extra-dimension workloads to this many edges',
    )
    p.add_argument(
        '--extra-max-accessor-repeats',
        type=int,
        default=5,
        help='cap repeated accessor calls in local extra-dimension workloads',
    )
    # worker-only
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--engine', default='annnet')
    p.add_argument('--backend', default=None)
    args = p.parse_args(argv)

    if args.worker:
        return _worker(args)
    return _parent(args)


if __name__ == '__main__':
    raise SystemExit(main())
