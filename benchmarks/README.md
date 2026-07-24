# AnnNet Benchmark Suite

The **single source of truth** for AnnNet performance: build/query speed, memory,
IO-format cost, adapter conversion cost, local mutation/accessor dimensions,
and head-to-head comparison against NetworkX, igraph, and graph-tool when the
Pixi `gt` environment is active.

## What it covers

| module | entry point | what it measures |
|---|---|---|
| `run.py` | `python -m benchmarks.run` | **the comprehensive report**: head-to-head build/query vs nx/igraph/graph-tool, memory footprint, AnnNet-only ops, **slices / history / algorithms**, **local mutation/accessor dimensions**, **IO formats**, **adapters**, scaling plots → `REPORT.md` |
| `cases/` | invoked by `run.py` | local non-overlapping benchmark cases: mutation primitives, annotation updates, backend mutation/annotation ops, lazy adapter accessors |
| `reporting/specsheet.py` | `python -m benchmarks.reporting.specsheet` | one-page **spec-sheet PDF + charts** (build/query/memory across scales to 1M V / 4M edges, all edge types) → `SPEC_SHEET.pdf`, `SPEC_SHEET_CHARTS.pdf` |
| `io_formats.py` | `python -m benchmarks.io_formats` | round-trip **write/read time + on-disk size + fidelity** for annnet / json / parquet / graphml / sif |
| `adapters.py` | `python -m benchmarks.adapters` | **export/import time** for the NetworkX / igraph / graph-tool / PyG bridges (uninstalled backends are skipped) |

`run.py` invokes `io_formats` and `adapters` for you (subprocess); run
them standalone only for a focused check.

## Run

From the repository root (interpreter with the full stack):

```bash
python -m benchmarks.run                  # full report: comparison + io + adapters
python -m benchmarks.run --tier quick     # tiny + small
python -m benchmarks.run --tier full      # tiny + xsmall + small + medium
python -m benchmarks.run --tier heavy     # + large scale (100K V / 400K E)
python -m benchmarks.run --tier huge      # + xlarge scale (1M V / 4M E)
python -m benchmarks.run --no-io          # skip io/adapter sections
python -m benchmarks.run --no-memory      # skip memory passes (faster)
python -m benchmarks.run --no-extra       # skip local mutation/accessor dimensions
python -m benchmarks.run --extra-max-edges 20000

pixi run -e gt benchmark-gt-quick         # same report with graph-tool enabled
pixi run -e gt benchmark-gt-full
pixi run -e gt benchmark-gt-heavy         # tiny -> large with graph-tool
pixi run -e gt benchmark-gt-huge          # tiny -> xlarge with graph-tool
python -m benchmarks.reporting.regenerate # rebuild REPORT.md + plots from existing JSON

python -m benchmarks.reporting.specsheet            # headline PDF + charts (long: to 1M/4M)
python -m benchmarks.reporting.specsheet --quick    # small scales, fast layout check

python -m benchmarks.io_formats --edges 20000
python -m benchmarks.adapters  --edges 20000
```

Outputs land in `benchmarks/results/` (override `run.py` with `--out`):
`results.json`, `REPORT.md`, `SPEC_SHEET.{pdf,png}`, `SPEC_SHEET_CHARTS.{pdf,png}`,
`plots/*.png`.

The local extra dimensions include intentionally expensive repeated public-call
cases. They are capped by default at 2,500 vertices / 10,000 edges / 5 accessor
repeats, while the upstream comparison and AnnNet-only workloads still use the
requested scale exactly. Capped rows are marked in the report notes.

## Layout

```
run.py         orchestrator + worker; renders REPORT.md (+ io/adapters/plots)
harness.py     time_repeat / time_oneshot / measure_memory  (the ONE harness)
scales.py      shared scale ladder and workload dimensions
engines.py     AnnNet / NetworkX / igraph / graph-tool adapters over identical input
workloads.py   comparable + annnet_only measurements -> records
cases/
  primitives.py    AnnNet public mutation primitives
  annotations.py   annotation initial-write/update variants
  backends.py      mutation/annotation comparisons across graph backends
  accessors.py     lazy adapter accessor cache and conversion costs
io_formats.py  serialization round-trip benchmark
adapters.py    graph-library conversion benchmark
environment.py reproducibility metadata
reporting/     Markdown renderer, plot renderer, artifact CSVs, regeneration CLI
  specsheet.py spec-sheet PDF + charts (self-contained, subprocess-per-cell)
```

## How to read it

- **Head-to-head** — comparable per-call medians; ratio columns and heatmaps are
  `AnnNet / baseline`. AnnNet carries an incidence matrix + annotation frames +
  multilayer state, so parity on plain ops is the honest bar, not a tie.
- **Memory** — `retained` bytes and `bytes/edge` for the built graph.
- **AnnNet-only** — hyperedges, multilayer, incidence/adjacency materialisation,
  copy/subgraph, annotations, IO: capabilities the baselines cannot express.
- **Slices / History / Algorithms** — slice set-algebra + presence + induced
  subgraph; mutation-log / snapshot / diff cost; directional traversal and
  matrix-derived neighbour queries.
- **Extra dimensions** — mutation primitives, annotation update paths,
  backend mutation/annotation operations, and lazy accessor cache overhead.
- **Plots / artifacts** — `plots/benchmark_records.csv`,
  `plots/status_counts.csv`, a comparable-operation scaling dashboard,
  AnnNet/NetworkX and AnnNet/igraph ratio heatmaps, optional AnnNet/graph-tool
  heatmap from the Pixi `gt` environment, memory scaling charts, wall-time
  summaries for every extra dimension, plus IO and adapter conversion plots.

## Fairness notes

- All engines receive an identical vertex list and `(src, tgt, weight)` edge list.
- Query benchmarks issue a single representative call (repeated by the harness) to
  isolate per-call overhead; the ring input keeps degree constant across scale.
- graph-tool is installed from conda-forge through `pixi run -e gt ...`; regular
  uv/pip runs skip it because graph-tool is not distributed on PyPI.
- Comparisons are only drawn between the engine adapters; AnnNet-unique
  features are never scored as a "win" over engines that cannot express them.
