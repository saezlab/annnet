# AnnNet Benchmark Suite

The **single source of truth** for AnnNet performance: build/query speed, memory,
IO-format cost, adapter conversion cost, and head-to-head comparison against
NetworkX and igraph. This package is the *entire* benchmark suite.

## Design principles (for trustworthiness)

The removed legacy harness timed each op **once** and reported a raw RSS delta, a
single shot is noise, and RSS delta is polluted by the allocator and other threads.
This suite is built so a number can be trusted as an SSoT:

| concern | legacy (removed) | this suite |
|---|---|---|
| timing | one shot | warmup + adaptive inner loop + N batches, GC disabled |
| reported stat | single `wall_time_s` | min / median / mean / stdev / p95 |
| memory | RSS delta only | `tracemalloc` peak + retained, bytes/edge, RSS cross-check |
| comparison | none | AnnNet vs **NetworkX** vs **igraph** on comparable ops |
| isolation | in-process | one subprocess per measurement (clean baseline) |
| reproducibility | none | env metadata: versions, CPU, git commit |

## What it covers

| module | entry point | what it measures |
|---|---|---|
| `run.py` | `python -m benchmarks.run` | **the comprehensive report**: head-to-head build/query vs nx/igraph, memory footprint, AnnNet-only ops, **slices / history / algorithms**, **IO formats**, **adapters**, scaling plots → `REPORT.md` |
| `specsheet.py` | `python -m benchmarks.specsheet` | one-page **spec-sheet PDF + charts** (build/query/memory across scales to 1M V / 4M edges, all edge types) → `SPEC_SHEET.pdf`, `SPEC_SHEET_CHARTS.pdf` |
| `io_formats.py` | `python -m benchmarks.io_formats` | round-trip **write/read time + on-disk size + fidelity** for annnet / json / parquet / graphml / sif |
| `adapters.py` | `python -m benchmarks.adapters` | **export/import time** for the NetworkX / igraph / graph-tool / PyG bridges (uninstalled backends are skipped) |

`run.py` invokes `io_formats` and `adapters` for you (subprocess); run
them standalone only for a focused check.

## Run

From the repository root (interpreter with the full stack):

```bash
python -m benchmarks.run                  # full report: comparison + io + adapters
python -m benchmarks.run --tier quick     # small only (~seconds)
python -m benchmarks.run --tier heavy     # + large scale (100K V / 400K E)
python -m benchmarks.run --tier huge      # + xlarge scale (1M V / 4M E)
python -m benchmarks.run --no-io          # skip io/adapter sections
python -m benchmarks.run --no-memory      # skip memory passes (faster)

python -m benchmarks.specsheet            # headline PDF + charts (long: to 1M/4M)
python -m benchmarks.specsheet --quick    # tiny scales, fast layout check

python -m benchmarks.io_formats --edges 20000
python -m benchmarks.adapters  --edges 20000
```

Outputs land in `benchmarks/results/` (override `run.py` with `--out`):
`results.json`, `REPORT.md`, `SPEC_SHEET.{pdf,png}`, `SPEC_SHEET_CHARTS.{pdf,png}`,
`plots/*.png`.

## Layout

```
run.py         orchestrator + worker; renders REPORT.md (+ io/adapters/plots)
specsheet.py   spec-sheet PDF + charts (self-contained, subprocess-per-cell)
harness.py     time_repeat / time_oneshot / measure_memory  (the ONE harness)
engines.py     AnnNet / NetworkX / igraph adapters over identical input
workloads.py   comparable + annnet_only measurements -> records
io_formats.py  serialization round-trip benchmark
adapters.py    graph-library conversion benchmark
environment.py reproducibility metadata
report.py      records -> Markdown + plots
```

## How to read it

- **Head-to-head** — comparable per-call medians; `ratio = AnnNet / igraph`. AnnNet
  carries an incidence matrix + annotation frames + multilayer state, so parity on
  plain ops is the honest bar, not a tie.
- **Memory** — `retained` bytes and `bytes/edge` for the built graph (nx/igraph store
  adjacency only).
- **AnnNet-only** — hyperedges, multilayer, incidence/adjacency materialisation,
  copy/subgraph, annotations, IO: capabilities the baselines cannot express.
- **Slices / History / Algorithms** — slice set-algebra + presence + induced
  subgraph; mutation-log / snapshot / diff cost; directional traversal and
  matrix-derived neighbour queries.

## Fairness notes

- All engines receive an identical vertex list and `(src, tgt, weight)` edge list.
- Query benchmarks issue a single representative call (repeated by the harness) to
  isolate per-call overhead; the ring input keeps degree constant across scale.
- Comparisons are only drawn between the three engine adapters; AnnNet-unique
  features are never scored as a "win" over engines that cannot express them.
