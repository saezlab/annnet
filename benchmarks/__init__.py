"""AnnNet benchmark suite — the SSoT for speed / memory / overhead.

* every timing is repeated with warmup + GC control and reported as a
  distribution (min / median / mean / stdev / p95);
* memory is measured with ``tracemalloc`` (deterministic peak + retained) next
  to ``psutil`` RSS (process view), and normalised to bytes-per-edge;
* AnnNet is compared head-to-head against **NetworkX**, **igraph**, and
  **graph-tool** when the Pixi ``gt`` environment is active. AnnNet-only
  capabilities (hyperedges, multilayer, stoichiometry, slices, annotations) are
  reported separately as an expressiveness-cost dimension;
* the run captures full environment metadata (library versions, CPU, git commit)
  for reproducibility.

Entry point: ``python -m benchmarks.run`` (see ``run.py`` / ``README.md``).
"""

__all__ = [
    'harness',
    'engines',
    'workloads',
    'cases',
    'scales',
    'environment',
    'reporting',
    'run',
    'io_formats',
    'adapters',
]
