"""IO-format round-trip benchmark — write time, read time, on-disk size, fidelity.

Every serialization format AnnNet ships is exercised through the public
``annnet.io`` facade on one identical graph, measured with the same rigorous
harness as the head-to-head suite (warmup + repeated samples, GC disabled).
Formats that cannot represent a feature (e.g. SIF has no edge weights) still
round-trip the structure; the ``edges_ok`` column reports whether the edge count
survived the round trip.

Run:  python -m benchmarks.io_formats [--vertices N] [--edges N] [--samples N]
"""

from __future__ import annotations

import os
import json
from pathlib import Path
import argparse
import tempfile
import warnings

from . import harness, environment


def build_graph(n_vertices: int, n_edges: int):
    """A deterministic directed graph with weights + a couple of attributes."""
    from annnet.core.graph import AnnNet

    G = AnnNet(directed=True)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G.add_edges(
            {
                'source': f'v{i % n_vertices}',
                'target': f'v{(i * 13 + 1) % n_vertices}',
                'weight': float(i % 7 + 1),
                'kind_attr': 'reg' if i % 2 else 'alt',
            }
            for i in range(n_edges)
        )
    return G


# ---------------------------------------------------------------------------
# Format specs: (name, write(G, dir)->path, read(path)->G, is_dir, note)
# ---------------------------------------------------------------------------
def _formats():
    import annnet.io as aio

    def _w(fn, fname):
        def write(G, d):
            p = os.path.join(d, fname)
            fn(G, p)
            return p

        return write

    return [
        ('annnet', lambda G, d: _annnet_write(aio, G, d), aio.read, False, 'native zstd archive'),
        ('json', _w(aio.to_json, 'g.json'), aio.from_json, False, 'JSON document'),
        (
            'parquet',
            lambda G, d: _parquet_write(aio, G, d),
            aio.from_parquet,
            True,
            'columnar Parquet dir',
        ),
        (
            'graphml',
            _w(aio.to_graphml, 'g.graphml'),
            aio.from_graphml,
            False,
            'GraphML (binary only)',
        ),
        ('sif', _w(aio.to_sif, 'g.sif'), aio.from_sif, False, 'SIF (no weights)'),
    ]


def _annnet_write(aio, G, d):
    p = os.path.join(d, 'g.annnet')
    aio.write(G, p, overwrite=True)
    return p


def _parquet_write(aio, G, d):
    p = os.path.join(d, 'gpq')
    aio.to_parquet(G, p)
    return p


def _path_size(p: str) -> int:
    if os.path.isfile(p):
        return os.path.getsize(p)
    if os.path.isdir(p):
        return sum(
            os.path.getsize(os.path.join(p, f))
            for f in os.listdir(p)
            if os.path.isfile(os.path.join(p, f))
        )
    return 0


def run(n_vertices: int = 2000, n_edges: int = 10000, samples: int = 5) -> list[dict]:
    G = build_graph(n_vertices, n_edges)
    n_e = G.number_of_edges()
    recs: list[dict] = []

    for name, write, read, _is_dir, note in _formats():
        rec = {'format': name, 'n_edges': n_e, 'note': note}
        tmp = Path(tempfile.mkdtemp(prefix=f'annnet_io_{name}_'))
        try:
            path = write(G, str(tmp))
            rec['size_bytes'] = _path_size(path)

            def _write_once(_p=path, _w=write, _t=str(tmp)):
                # rewrite to a fresh sub-path so each sample is independent
                return _w(G, _t)

            rec['write'] = harness.time_oneshot(_write_once, samples=samples).as_dict()
            rec['read'] = harness.time_oneshot(
                lambda _p=path, _r=read: _r(_p), samples=samples
            ).as_dict()
            G2 = read(path)
            rec['edges_ok'] = G2.number_of_edges() == n_e
        except Exception as e:
            rec['error'] = f'{type(e).__name__}: {e}'
        finally:
            import shutil

            shutil.rmtree(tmp, ignore_errors=True)
        recs.append(rec)

    return recs


def _fmt_t(d):
    if not d:
        return '—'
    s = d['median_s']
    return f'{s * 1e3:.1f} ms' if s >= 1e-3 else f'{s * 1e6:.0f} µs'


def _fmt_b(n):
    if n is None:
        return '—'
    return f'{n / 1024:.1f} KB' if n >= 1024 else f'{n} B'


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--vertices', type=int, default=2000)
    ap.add_argument('--edges', type=int, default=10000)
    ap.add_argument('--samples', type=int, default=5)
    ap.add_argument('--json-out', default=None)
    args = ap.parse_args()

    recs = run(args.vertices, args.edges, args.samples)
    print(f'IO formats — {args.vertices} V / {args.edges} E  (median of {args.samples})\n')
    print(f'{"format":10s} {"write":>10s} {"read":>10s} {"size":>10s}  ok  note')
    print('-' * 64)
    for r in recs:
        if 'error' in r:
            print(f'{r["format"]:10s}  ERROR: {r["error"]}')
            continue
        print(
            f'{r["format"]:10s} {_fmt_t(r.get("write")):>10s} {_fmt_t(r.get("read")):>10s} '
            f'{_fmt_b(r.get("size_bytes")):>10s}  {"✓" if r.get("edges_ok") else "✗"}   {r["note"]}'
        )

    if args.json_out:
        payload = {'environment': environment.capture(), 'io_formats': recs}
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(f'\nJSON -> {args.json_out}')


if __name__ == '__main__':
    main()
