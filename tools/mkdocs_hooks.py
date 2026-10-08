"""MkDocs startup hooks for the docs build."""

from __future__ import annotations

import os
import posixpath
import re
import shutil
import warnings
from html import escape
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from mkdocs.utils import get_relative_url

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


def _sync_case_studies() -> None:
    """Publish the two case studies; topic guides live directly in docs/guide."""
    repo_root = Path(__file__).resolve().parent.parent
    source_dir = repo_root / "notebooks" / "use_cases"
    destination_dir = repo_root / "docs" / "examples" / "use_cases"
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in ("UC1.ipynb", "UC2.ipynb"):
        shutil.copy2(source_dir / name, destination_dir / name)


def on_page_content(html, page, config, files):
    """Resolve source-relative notebook links just as MkDocs does for Markdown."""
    if not page.file.src_uri.endswith(".ipynb"):
        return html

    def replace(match):
        parts = urlsplit(match.group(3))
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith("/"):
            return match.group(0)
        source = posixpath.normpath(
            posixpath.join(posixpath.dirname(page.file.src_uri), parts.path)
        )
        target = files.get_file_from_path(source)
        if target is None:
            return match.group(0)
        url = urlunsplit(("", "", get_relative_url(target.url, page.url), parts.query, parts.fragment))
        return f"{match.group(1)}={match.group(2)}{escape(url, quote=True)}{match.group(2)}"

    return re.sub(r'''(href|src)=(["'])(.*?)\2''', replace, html)


# Preserve bookmarks while keeping superseded articles out of navigation/search.
_REDIRECTS = {
    "explanations": "guide",
    "tutorials": "examples",
    **{f"explanations/{name}": "guide/graph-model" for name in (
        "math-incidence", "edge-tables-and-formats",
    )},
    **{f"explanations/{name}": "guide/layers" for name in (
        "math-multilayer", "aspects-and-windows", "values-and-scale",
    )},
    "explanations/design-philosophy": "guide",
    "explanations/internal-representation": "community/internals",
    "explanations/architecture-overview": "community/internals",
    "explanations/mutation-and-derived-state": "community/internals",
    "explanations/reading-the-graph": "guide/annotations-and-views",
    "explanations/managers-and-views": "guide/slices",
    "explanations/history-and-diffs": "guide/computation-and-storage",
    "explanations/interoperability": "guide/computation-and-storage",
    "explanations/io-annnet": "guide/computation-and-storage",
    "explanations/format-semantics": "reference/io/format-comparison",
    "explanations/add-edges": "reference/core/adding-edges",
    "explanations/api-migration": "reference/migration",
    "explanations/performance": "community/performance",
    "tutorials/notebooks/special/sp03_flexible_edge_orientation": "reference/core/adding-edges",
    **{f"tutorials/notebooks/tutos/{name}": f"guide/{target}" for name, target in (
        ("01_quickstart", "annotations-and-views"),
        ("02_attributes_and_views", "annotations-and-views"),
        ("03_tables_and_storage", "computation-and-storage"),
        ("04_slices_and_subgraphs", "slices"),
        ("05_hyperedges_and_traversal", "graph-model"),
        ("06_multilayer", "layers"),
        ("07_history_and_reproducibility", "computation-and-storage"),
        ("08_backend_accessors", "computation-and-storage"),
    )},
    **{f"tutorials/notebooks/special/{name}": f"guide/{target}" for name, target in (
        ("sp01_directed_hyperedges", "graph-model"),
        ("sp02_multilayer_math", "layers"),
        ("sp04_backend_lazy_proxies", "computation-and-storage"),
        ("sp05_exploration_workflow", "annotations-and-views"),
    )},
    **{f"tutorials/notebooks/use_cases/{name}": f"examples/use_cases/{name}" for name in ("UC1", "UC2")},
    **{f"tutorials/notebooks/scenarios/{name}": f"examples/scenarios/{name}" for name in (
        "omnipath_table_ingestion", "cytoscape_cx2_export", "pyg_heterodata_export", "causal_activity_bridge",
    )},
}


def on_post_build(config):
    site = Path(config["site_dir"])
    for old, new in _REDIRECTS.items():
        destination = site / old / "index.html"
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = escape(posixpath.relpath(new, old) + "/", quote=True)
        destination.write_text(
            '<!doctype html><html lang="en"><meta charset="utf-8">'
            f'<meta http-equiv="refresh" content="0; url={url}">'
            '<title>Page moved</title>'
            f'<p>This page is now in the <a href="{url}">updated documentation</a>.</p></html>',
            encoding="utf-8",
        )


_sync_case_studies()
