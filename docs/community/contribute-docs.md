# Contributing to the documentation

Edit Markdown pages under `docs/` and topic notebooks under `docs/guide/`.
The API reference under `docs/reference/` renders the package's NumPy-style
docstrings with mkdocstrings.

## Where content belongs

- **Quickstart:** one short import, inspect, select, and save workflow.
- **Guide:** five topic notebooks. Explain a concept, its mathematics where
  needed, and its practical API in the same notebook. Extend the relevant
  topic before adding another tutorial about it.
- **Tutorials:** complete analyses and integrations, all in `docs/tutorials/`.
  Each one has its environment file in `docs/tutorials/environments/`.
- **API Reference:** signatures, input forms, file-format details, and migration.
- **Community:** contribution instructions and implementation reports.

Prefer imported data for workflows. Use a small manually constructed graph when
it makes a specific structural rule or calculation easier to verify. State data
provenance and distinguish illustrative values from measured results.

## Preview locally

From the repository root:

```bash
uv sync --group docs
uv run --group docs python -m mkdocs serve --dev-addr 127.0.0.1:8000
```

Open <http://127.0.0.1:8000>. MkDocs rebuilds after edits. The default build
executes the five Guide notebooks and fails on execution errors. Case studies
and integration examples render their saved outputs using the
[example environments](../tutorials/index.md) when run separately.

For faster prose and layout work, render the notebooks' saved outputs:

```bash
ANNNET_EXECUTE_NOTEBOOKS=false uv run --group docs python -m mkdocs serve
```

Re-enable execution before validating changes to notebook code. Refresh stored
outputs in Jupyter with **Restart Kernel and Run All Cells**, then save.

## Check before submitting

```bash
uv run --group docs python -m mkdocs build --strict
uv run --group docs python -m mkdocs build --strict -f mkdocs-online.yml
```

The second command checks the public navigation, which excludes experimental
reference pages. Both write the site to `site/`. Check rendered formulas, table
outputs, downloads, and links as well as the build result. Notebook links should
use source-relative paths, such as `../reference/core/layers.md`; the build hook
rewrites them to website URLs.

Keep one authoritative explanation per topic. When moving a published page,
update internal links and its redirect in `tools/mkdocs_hooks.py` so existing
bookmarks still work.
