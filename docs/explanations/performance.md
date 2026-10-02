# What the general path costs

AnnNet holds one representation for every graph — binary edges, hyperedges,
layers, slices, edge entities — and a flat binary graph uses the same code
path as a multilayer one. This page records what that path costs, how the
numbers were taken so they can be taken again, and the design decisions that
rest on them. The machine-readable evidence is
`docs/explanations/performance-measurements.json`.

Cold-start measurements include package import and the first graph construction
in a fresh interpreter. Warm-operation measurements use an initialized process.
The 53.7 µs warm constructor timing therefore excludes the roughly 317 ms first
construction cost reported below.

## How the numbers were taken

- Script: `benchmarks/api_costs.py`, built on `benchmarks/harness.py`
  (warm-up, an adaptive inner loop so each batch runs ≥ 50 ms, GC disabled in
  the timed region, seven batches, a distribution reported as min / median /
  p95 / stdev). A construction or a copy is timed one shot per sample, five
  samples, warm-up excluded. The first call of every exploration read is
  recorded separately from the warm distribution.
- Starting a session is measured in a **fresh interpreter for each of nine
  runs** (`cold_start` section): nothing is imported, nothing is cached, and the
  clock brackets `import annnet`, the first `AnnNet()` and the second. The
  `libraries preloaded` variant imports polars, pandas and pyarrow before the
  clock starts; it measures the constructor without their import and is **not**
  what a user who starts a session sees.
- Memory: `tracemalloc` peak and retained bytes attributable to one build
  (never mixed with a timing pass); `psutil` RSS delta as a cross-check.
- Command, from the repository root. `--compare-tree` measures another
  checkout of the package (for example `git archive <rev> annnet` exported to
  a scratch directory) in a subprocess with the same script, workload and
  interpreter, so two revisions are compared under matched conditions:

  ```
  NUMBA_CACHE_DIR=/tmp/annnet-numba-cache python -m benchmarks.api_costs \
      --out docs/explanations/performance-measurements.json \
      --compare-tree <exported previous revision> --compare-label previous \
      --compare-sections cold_start,flat_costs,promotion
  ```

- Environment of the recorded run: Python 3.13.9, Linux 6.6.87.2-microsoft-standard-WSL2-x86_64 (WSL2 on a Windows
  mount), x86_64, 6 logical CPUs; numpy 2.3.5, scipy 1.16.3,
  polars 1.35.2, pandas 2.3.3, pyarrow 22.0.0, networkx 3.5,
  igraph 1.0.0. Taken 2026-09-30 with the machine otherwise idle. The
  compared revision is the package at commit `8e6fc75` (AnnNet 0.3.0); "now" is
  0.4.0.
- Workload: `n0..n4999`, 20 000 random binary edges from `random.Random(4)`
  (`benchmarks.api_costs.edge_specs`), the degree read on the node whose
  degree is the median of the workload (`n2481`, degree 8).
- **Compare within a run.** The same script on the same code gives numbers that
  differ from run to run on this host: an earlier run measured `8e6fc75`'s
  `degree` at 1.6 µs where this one measured it at 2.97 µs. The ratios between
  two measurements of one run are the comparable quantity, and absolute numbers
  carry the host.

## 0. Starting a session

| in a fresh interpreter | at `8e6fc75` | now |
|---|---:|---:|
| `import annnet` | 85.1 ms | 88.1 ms |
| the first `AnnNet()` | 302 ms | 317 ms |
| the second `AnnNet()` | 114 µs | 118 µs |
| the first `AnnNet()`, with polars, pandas and pyarrow already imported | 240 ms | 240 ms |

`import annnet` is cheap because the package is lazy: it loads nothing it was
not asked for. The first graph is where the cost lands, 317 ms in
this run, because building it imports `annnet.core` and with it numpy and
`scipy.sparse`. Neither this release nor the previous one avoids that, and the
figures above show it did not change: what this release changed is what an
operation costs once the session has started, which is the rest of the page.
The 0.1 ms of the second construction is one call with cold caches; the warm
distribution of section 1 is 53.7 µs.

## 1. Four costs of a flat binary graph

| cost (warm) | at `8e6fc75` | now | networkx 3.5 | now vs networkx |
|---|---:|---:|---:|---:|
| construct an empty graph | 649 µs | **53.7 µs** | 0.92 µs | 58× |
| count the nodes (`len(G.N)`) | 260 µs | **0.41 µs** | 0.10 µs | 4× |
| degree of one node | 2.97 µs | **1.59 µs** | 0.24 µs | 7× |
| build 5 000 / 20 000 | 95.4 ms | **87.6 ms** (min 87.1 ms, p95 90.1 ms) | 30.4 ms | 2.9× |
| build, retained memory | 11.4 MB | **10.8 MB** (peak 17.1 MB) | 7.7 MB (peak 11.6 MB) | 1.4× |

- **Empty construction** builds nothing that is not asked for: 649 µs to
  53.7 µs, 12× faster once the session has started. What remains is the graph object itself: the
  store, the attribute store, the contextual store, the slice registry with its
  default slice, and the namespaces bound to the instance. Choosing a dataframe
  backend used to look every candidate module up on every construction; the
  answer is now remembered for the process, and asked again only when a backend
  is requested by name and was not found, or after
  `annnet._support.optional_components.refresh()`.
- **Counting** is constant time at every size: 0.37 µs at 10 nodes and 0.41 µs
  at 1 000 000 (`count_scaling` section). `len(G.N)`, `len(G.E)` and
  `G.shape` are maintained counters.
- **Degree** reads the incident-edge index the store maintains: 1.59 µs against
  44.2 µs for `G.incident_edges(node)` (which builds a record per edge) at
  5 000 / 20 000, and 1.59 µs at 50 000 / 200 000 (`degree_scaling`
  section). A self-loop, a zero-weight edge and a hyperedge added to the same
  node give degree 11 in 1.80 µs. The two revisions differ by
  1.9× in this run, which is within what the host does between runs; the
  claim is that it does not grow with the graph.
- **Build, warm: 2.9× networkx.** Attributed by profile (`cProfile` over one
  warm build, 5 000 / 20 000; the shares are of profiled time, which is
  inflated, so read them as proportions):
  1. `_mutate.batch_add_edges` — the per-edge normalization of each spec in
     Python: reading `source`/`target`/`edge_id`, resolving endpoints,
     deduplicating ids, deciding kind and direction, recording slice
     membership. About a third of the build in the function's own time.
  2. `CoreState._add_edges` — building the member lists and the incident-edge
     index per chunk (157 chunks of `BULK_CHUNK = 128`). About a quarter.
  3. `add_nodes`, about a tenth; the check that each spec is not a hyperedge,
     about 6 %; `sys.intern` of the ids; the default slice's membership sets.
     The rest.

  None of it is a per-edge record of a constant fact (section 2). What
  remains is identity and validation work networkx does not do: an edge id
  per edge, an incident-edge index that makes the degree 1.59 µs, membership in
  the default slice, and checked endpoints. The retained memory is 1.4× and
  the peak is transient.

  **Decision: no second "flat" mode.** A flat graph pays the validation and
  identity bookkeeping of the general path and nothing for layers, hyperedges
  or slices it does not use. A further gain would come from moving the spec
  normalization of `batch_add_edges` into one vectorized pass — an
  implementation change with no surface and no mode — and is left open.

## 2. No per-edge record of a constant fact

Every structural edge of a flat graph has the multilayer role `intra`. That is
a fact about the graph, so the store answers it from the aspects and records
nothing per edge; a write of the flat default stores nothing; and the record
is materialized once when aspects are declared, because only then can one
edge's role differ from another's (`tests/test_flat_defaults.py`). Before this
change the store held 20 000 identical `'intra'` entries for a 20 000-edge
flat graph; retained build memory fell from 11.4 MB to 10.8 MB in the run above.

Other constant bookkeeping on a flat graph, audited: `edge_ml_layers` is
empty (never written on a flat graph); `edge_directed` holds `INHERIT` per
edge as an int8 array column, which is the per-edge direction a caller may set
individually and is not constant by construction; `edge_policy` is written
only for edges that declare a policy; the default slice's membership sets hold
every node and edge id, which is what a slice is and is the cost of
`G.slices` answering without a scan.

## 3. The cost of gaining the full behaviour

The surface of a flat graph and of a multilayer graph is the same surface; a
flat graph gains aspects through `G.layers.set_aspects(...)` and hyperedges
through `G.add_edges(...)`. What that costs, on the 5 000 / 20 000 flat graph:

| promotion | at `8e6fc75` | now |
|---|---:|---:|
| `layers.set_aspects(['cond'], {...})`, first call | 2.6 ms | **4.7 ms** (median of 5: 4.7 ms) |
| one more binary edge, for scale | 96.4 µs | 59.4 µs |
| the first hyperedge on a binary graph | 72.8 µs | 93.6 µs |
| the second hyperedge | 75.7 µs | 35.6 µs |
| `degree` after the hyperedge | 2.0 ms | **9.81 µs** |

- Gaining aspects costs 4.7 ms at 20 000 edges, about twice what it cost at
  `8e6fc75` (2.6 ms): every placement moves to the placeholder coordinate, the
  bare-id index is built, and every held edge gains its role record (section 2,
  landing exactly once). That is the price of not storing the record on every
  flat graph, paid once, by a graph that is about to have layers.
- The first hyperedge costs about as much as one more binary edge; there is no
  mode switch and no rebuild. The degree read after it is 9.81 µs, where
  the previous revision paid 2.0 ms because its cache invalidation rebuilt a
  structure on the next read.
- A mixed batch of binary and hyperedges describes the same graph in either
  order (`mixed_batch_order_independent: true`).

**Decision: the caller needs no way to ask for the full behaviour up front.**

## 4. What an exploration costs

Random flat graphs of 20 000 and 200 000 edges (nodes = edges / 4), sparse
contextual attributes: 10 % of the nodes carry `score` and `group`, a tenth of
those `flag`; one edge in ten carries `confidence`. The composed selection is
`(G.N.select(score__gte=0.5) | G.N.select(group='x')) - G.N.select(flag=True)`
(298 nodes at 20k, 2,944 at 200k); the view is `G.view(nodes=that,
boundary='open')` (2,262 edges at 20k, 22,964 at 200k). "First" is the first call in the
process; "warm" is the median of the distribution.

| read | 20k edges, first | 20k, warm median | 200k, first | 200k, warm median |
|---|---:|---:|---:|---:|
| composed selection | 1.2 ms | 1.5 ms | 10.6 ms | 14.3 ms |
| one selected row, `G.attrs.row('nodes', id)['score']` | — | 1.95 µs | — | 2.02 µs |
| `attrs.table('nodes', limit=20)` | 7.7 ms | 911 µs | 6.2 ms | 3.6 ms |
| `attrs.table('edges', derived=True, limit=20)` | 16.1 ms | 1.6 ms | 65.0 ms | 5.2 ms |
| the open-boundary view, `len(V.E.ids)` (includes the selection) | 13.4 ms | 11.5 ms | 173 ms | 172 ms |
| `V.attrs.table('edges', derived=True, limit=20)` | 12.6 ms | 1.7 ms | 174 ms | 5.3 ms |
| `V.materialize()` | 27.3 ms | 19.7 ms | 241 ms | 221 ms |
| `G.summary()` | 593 µs | 322 µs | 10.2 ms | 4.3 ms |

Three properties of the exploration path are what the measurements protect:

1. **A table preview builds only the rows it shows.** `limit=` reaches the
   builders, which walk the address's keys lazily and stop, so the derived edges
   preview at 200 000 edges is bounded by the rows asked for.
2. **A view does not scan every edge.** A node-constrained selection reaches its
   candidate edges through the store's incident-edge index, the closure
   memoizes endpoint keys and entity kinds per resolution, and order comes from
   sorting the selected slots. What remains is the Python closure over the
   candidate edges (7.51 µs each), proportional to the selection, not the graph.
3. **A predicate over an object column does not call a function per cell**, and
   each leaf does not re-walk the axis for its ids. Equality, inequality,
   membership and `isnull` on an object column are one pass, and the axis ids
   are cached against the structure clock.

Reading one selected row costs 2.02 µs at either size: no contextual level is
scanned, no node × layer or slice × edge product is built. `materialize` is
~10 µs per copied edge and proportional to the copy; its peak allocation
is 26.8 MB for 20,743 nodes and 22,964 edges (retained 16.5 MB).

## 5. Left as it is, on the numbers

- `G.incident_edges` builds an `EdgeView` per edge (44.2 µs for degree 8).
  It answers a different question from `degree`; left as it is.
- `G.summary()` tallies edge kinds in one vectorized pass (4.3 ms at 200 000
  edges) and reports them as computed; left as it is.
- `V.materialize()` at 221 ms for 22,964 edges goes through the public
  `add_edges` path of the copy; a bulk transfer of store rows would be faster
  and is left open, because a copy is not an exploration read.
