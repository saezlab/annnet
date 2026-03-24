"""MkDocs startup hooks for the docs build."""

from __future__ import annotations

import os
import warnings

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
