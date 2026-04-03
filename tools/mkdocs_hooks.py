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


def _sync_tutorial_notebooks() -> None:
    """Mirror notebooks/tu*/ into docs/tutorials/notebooks for MkDocs."""
    repo_root = Path(__file__).resolve().parent.parent
    notebooks_root = repo_root / "notebooks"
    tutorials_root = repo_root / "docs" / "tutorials" / "notebooks"

    for source_dir in sorted(path for path in notebooks_root.glob("tu*") if path.is_dir()):
        destination_dir = tutorials_root / source_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for notebook in sorted(source_dir.glob("*.ipynb")):
            shutil.copy2(notebook, destination_dir / notebook.name)


_sync_tutorial_notebooks()
