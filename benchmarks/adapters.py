"""Adapter conversion benchmark — export/import time for the graph-library bridges.

AnnNet converts to/from NetworkX, igraph, graph-tool (round-trip) and exports to
PyTorch Geometric (one-way tensor bundle), all through the public
``annnet.adapters`` API and the rigorous shared harness. Optional backends
that are not installed are reported as ``skipped`` rather than failing the run.

Run:  python -m benchmarks.adapters [--vertices N] [--edges N] [--samples N]
"""

from __future__ import annotations

import json
from pathlib import Path
import argparse
import warnings

from . import harness, environment
from .io_formats import build_graph


def run(n_vertices: int = 2000, n_edges: int = 10000, samples: int = 5) -> list[dict]:
    import annnet.adapters as aad

    G = build_graph(n_vertices, n_edges)
    n_e = G.number_of_edges()
    recs: list[dict] = []

    def _rt(name, to_fn, from_fn, note):
        """Round-trip adapter: export (annnet -> lib) then import (lib -> annnet)."""
        rec = {'adapter': name, 'n_edges': n_e, 'note': note, 'direction': 'round-trip'}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                exported = to_fn(G)
                rec['export'] = harness.time_oneshot(lambda: to_fn(G), samples=samples).as_dict()
                rec['import'] = harness.time_oneshot(
                    lambda: from_fn(*exported), samples=samples
                ).as_dict()
                back = from_fn(*exported)
                rec['edges_ok'] = back.number_of_edges() == n_e
        except (ImportError, RuntimeError) as e:
            rec['skipped'] = f'{type(e).__name__}: {str(e)[:60]}'
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        recs.append(rec)

    def _export_only(name, to_fn, note):
        rec = {'adapter': name, 'n_edges': n_e, 'note': note, 'direction': 'export'}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                to_fn(G)
                rec['export'] = harness.time_oneshot(lambda: to_fn(G), samples=samples).as_dict()
        except (ImportError, RuntimeError) as e:
            rec['skipped'] = f'{type(e).__name__}: {str(e)[:60]}'
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        recs.append(rec)

    _rt('networkx', aad.to_nx, aad.from_nx, 'DiGraph + manifest')
    _rt('igraph', aad.to_igraph, aad.from_igraph, 'igraph.Graph + manifest')
    _rt('graphtool', aad.to_graphtool, aad.from_graphtool, 'gt.Graph + manifest')
    _export_only('pyg', aad.to_pyg, 'PyG Data (tensor bundle)')
    return recs


def _fmt_t(d):
    if not d:
        return '—'
    s = d['median_s']
    return f'{s * 1e3:.1f} ms' if s >= 1e-3 else f'{s * 1e6:.0f} µs'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vertices', type=int, default=2000)
    ap.add_argument('--edges', type=int, default=10000)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    recs = run(args.vertices, args.edges, args.samples)
    print(f'Adapters — {args.vertices} V / {args.edges} E  (median of {args.samples})\n')
    print(f'{"adapter":10s} {"export":>10s} {"import":>10s}  ok  note')
    print('-' * 60)
    for r in recs:
        if 'skipped' in r:
            print(f'{r["adapter"]:10s}  skipped ({r["skipped"]})')
            continue
        if 'error' in r:
            print(f'{r["adapter"]:10s}  ERROR: {r["error"]}')
            continue
        ok = '✓' if r.get('edges_ok') else ('—' if r['direction'] == 'export' else '✗')
        print(
            f'{r["adapter"]:10s} {_fmt_t(r.get("export")):>10s} {_fmt_t(r.get("import")):>10s} '
            f'  {ok}   {r["note"]}'
        )

    if args.json_out:
        payload = {'environment': environment.capture(), 'adapters': recs}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f'\nJSON -> {args.json_out}')


if __name__ == '__main__':
    main()
