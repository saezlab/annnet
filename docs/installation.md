# Installation guide

## Prerequisites

Before installing `annnet`, please ensure you have the following tools installed:


| Tool   | Minimum Version | Description                           | Installation Guide                                                           |
| ------ | --------------- | ------------------------------------- | ---------------------------------------------------------------------------- |
| Python | 3.10            | Programming language                  | [Install Python 3](https://docs.python.org/3/using/index.html)               |
| uv     | —               | Python packaging & dependency manager | [Install uv](https://docs.astral.sh/uv/getting-started/installation/)        |
| git    | —               | Version control system                | [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) |


!!! tip "Tip"
    If you are missing any of those pre-requisites, **please follow the installation guide in each resource before you continue**.


## Checking prerequisites

Verify that everything is installed by running:

```bash
python --version   # Should be 3.10 or higher
uv --version
git --version
```

## Installation

### From PyPI

```bash
pip install annnet

# Optional extras
pip install "annnet[networkx,igraph]"   # backends
pip install "annnet[io]"                # JSON/Parquet/Zarr, Excel, Narwhals
pip install "annnet[all]"               # common extras (graph‑tool not on PyPI)
```

Graph‑tool is supported if installed via your OS/package manager.

### From source (editable dev install)

This package is under active development. To try the latest, clone and install in editable mode so changes reflect immediately without reinstalling.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/saezlab/annnet.git
   ```

2. **Navigate into the project directory:**
   ```bash
   cd annnet
   ```

3. **Install the package in editable mode using `uv`:**
   ```bash
   uv pip install -e .
   ```

You can now start using `annnet` in your Python environment. Any changes you make to the source code will take effect immediately.
