# Installation guide

`annnet` is designed to stay lightweight by default. The base package only requires `numpy`, `scipy`, and `narwhals`, and everything heavier is exposed as an optional extra.

## Requirements

- Python `3.10+`
- `pip` for standard installs
- `uv` if you want a fast local or development workflow
- `git` if you want to install from GitHub

Check your environment:

```bash
python --version
pip --version
uv --version
git --version
```

## Install from PyPI

If you just want the core package:

```bash
pip install annnet
```

This gives you the core graph data structure and algorithms without pulling in backend, plotting, storage, or ML dependencies you may never use.

## Choose your extras

Install only the extras you need. `annnet` exposes both single-purpose extras and broader install bundles.

### Tabular data backends

Single extras:

```bash
pip install "annnet[polars]"
pip install "annnet[pandas]"
pip install "annnet[pyarrow]"
```

- `polars`: recommended default dataframe backend for annnet tables and annotations
- `pandas`: pandas-native dataframe support
- `pyarrow`: Arrow table support and parquet-related tabular workflows

When AnnNet needs to create new annotation tables and no backend is specified,
`annotations_backend="auto"` selects the first installed backend in this order:
Polars, pandas, then PyArrow. DataFrame input remains Narwhals-compatible, so
tables from different supported eager backends can be imported together.

### Graph backends

Single extras:

```bash
pip install "annnet[networkx]"
pip install "annnet[igraph]"
pip install "annnet[pyg]"
```

Bundle:

```bash
pip install "annnet[backends]"
```

- `networkx`: interoperability and access to NetworkX algorithms
- `igraph`: interoperability with `python-igraph`
- `pyg`: PyTorch Geometric support
- `backends`: installs the pip-installable graph backends `networkx` and `igraph`

`graph-tool` is supported by the codebase, but it is not published on PyPI, so it must be installed separately with Pixi, conda, or your system package manager.

### Plotting backends

Single extras:

```bash
pip install "annnet[matplotlib]"
pip install "annnet[pydot]"
pip install "annnet[graphviz]"
```

Bundle:

```bash
pip install "annnet[plot]"
```

- `matplotlib`: plotting support for matplotlib-based rendering
- `pydot`: DOT export and pydot-based rendering
- `graphviz`: Python Graphviz bindings
- `plot`: installs all pip-installable plotting backends together

### I/O extras

Single extras:

```bash
pip install "annnet[excel]"
pip install "annnet[parquet]"
pip install "annnet[zarr_io]"
pip install "annnet[sbml]"
```

Bundles:

```bash
pip install "annnet[io]"
pip install "annnet[storage]"
pip install "annnet[metabo]"
```

- `excel`: Excel import/export support
- `parquet`: parquet I/O support via `pyarrow`
- `zarr_io`: `.annnet` storage support with `zarr` and `numcodecs`
- `sbml`: SBML parsing support
- `io`: broader I/O bundle including pandas, openpyxl, pyarrow, zarr, numcodecs, lxml, and toml
- `storage`: storage-oriented subset for parquet/zarr/pandas workflows
- `metabo`: metabolomics-oriented bundle with SBML, COBRA, and Graphviz

### Other install bundles

```bash
pip install "annnet[all]"
pip install "annnet[dev]"
```

- `all`: broad pip-installable runtime bundle, excluding non-PyPI dependencies and PyG
- `dev`: packaging and contributor tooling installable via pip
Use `all` if you want a batteries-included pip install and do not mind a larger environment.

## Install from GitHub with uv

Use this if you want the latest repository version before the next PyPI release:

```bash
uv venv
source .venv/bin/activate
uv pip install "git+https://github.com/saezlab/annnet.git"
```

If you want extras as well, use the package URL in the same way you would use an extra on PyPI, for example:

```bash
uv pip install "annnet[all] @ git+https://github.com/saezlab/annnet.git"
```

## Local editable install with uv

For development or documentation work with pip-installable dependencies only, clone the repository and install it in editable mode:

```bash
git clone https://github.com/saezlab/annnet.git
cd annnet
uv venv
source .venv/bin/activate
uv pip install -e .
```

For a fuller local environment:

```bash
uv sync --group dev --group tests --group docs
```

You can also add runtime extras you care about:

```bash
uv sync --group dev --group tests
uv pip install -e ".[networkx,igraph,polars,sbml,parquet]"
```

## Run tests with uv

From the repository root:

```bash
uv sync --group tests
uv run pytest -vv
```

## Use Pixi for non-pip dependencies

Use Pixi when you want the repository's fuller development environment, especially for dependencies that are not reliably available from PyPI such as `graph-tool`.

```bash
pixi install
pixi run test-gt
```

The Pixi environment:

- installs `annnet` editable with the published `all` and `dev` extras
- provides `graph-tool` from `conda-forge`
- is the right path for running tests that depend on non-pip packages

If you want the full test suite inside the Pixi environment, run:

```bash
pixi run test-all
```

## Build and serve the docs with uv

This repository defines a dedicated docs dependency group for MkDocs:

```bash
uv sync --group docs
uv run mkdocs serve
```

For a one-off strict build:

```bash
uv run mkdocs build --strict
```

The generated site is written to `site/`.
