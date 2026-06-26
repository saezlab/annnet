"""MkDocs startup hooks for the docs build."""

from __future__ import annotations

import os
import shutil
import warnings
from pathlib import Path

os.environ.setdefault("JUPYTER_PLATFORM_DIRS", "1")

try:
    from nbformat.validator import MissingIDFieldWarning
except Exception:  # pragma: no cover - docs-only fallback
    MissingIDFieldWarning = None

if MissingIDFieldWarning is not None:
    warnings.filterwarnings("ignore", category=MissingIDFieldWarning)

warnings.filterwarnings(
    "ignore",
    message="Jupyter is migrating its paths to use standard platformdirs",
    category=DeprecationWarning,
)


# Single source of truth: these notebooks/ subdirs are mirrored into the docs
# tree at build time so editing them auto-updates the published docs. The
# tutos/ notebooks already live under docs/ directly and are their own SSoT.
_MIRRORED_NOTEBOOK_DIRS = ("special", "use_cases")


def _sync_tutorial_notebooks() -> None:
    """Mirror the SSoT notebooks/ subdirs into docs/tutorials/notebooks for MkDocs."""
    repo_root = Path(__file__).resolve().parent.parent
    notebooks_root = repo_root / "notebooks"
    tutorials_root = repo_root / "docs" / "tutorials" / "notebooks"

    for name in _MIRRORED_NOTEBOOK_DIRS:
        source_dir = notebooks_root / name
        if not source_dir.is_dir():
            continue

        destination_dir = tutorials_root / name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for notebook in sorted(source_dir.glob("*.ipynb")):
            shutil.copy2(notebook, destination_dir / notebook.name)


_sync_tutorial_notebooks()
