#!/usr/bin/env python3
"""
Generate import/dependency graphs from the actual repository source tree.

Outputs:
  - import_edges.csv
  - private_imports.csv
  - module_symbols.csv
  - module_cycles.csv
  - clean_package_dependency.mmd/.dot/.svg
  - clean_module_dependency.mmd/.dot/.svg
  - symbol_labeled_module_dependency.mmd/.dot/.svg
  - comprehensive_symbol_import_graph.mmd/.dot/.svg
  - boundary_helper_graph.mmd/.dot/.svg

SVG rendering requires Graphviz `dot` to be installed.
Mermaid and CSV outputs require only the Python standard library.

Usage:
  python scripts/import_graphs.py --repo . --package annnet --out import_graphs --render-svg
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import shutil
import subprocess  # nosec B404 - used only for fixed-argv Graphviz execution
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Final
from types import MappingProxyType

# -----------------------------
# Architecture contract
# -----------------------------

# source prefix -> forbidden target prefixes
#
# Intended layer direction:
#
#   _support
#      <- core
#      <- algorithms
#      <- adapters
#      <- io / utils
#      <- top-level annnet API
#
# Imports should generally point downward only.
FORBIDDEN_IMPORT_PREFIXES: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    # Low-level support must stay dependency-free with respect to annnet internals.
    "{package}._support": (
        "{package}.core",
        "{package}.algorithms",
        "{package}.adapters",
        "{package}.io",
        "{package}.utils",
    ),

    # Core must not depend on higher-level packages.
    "{package}.core": (
        "{package}.algorithms",
        "{package}.adapters",
        "{package}.io",
        "{package}.utils",
    ),

    # Algorithms may operate on core, but should not know about adapters or I/O.
    "{package}.algorithms": (
        "{package}.adapters",
        "{package}.io",
        "{package}.utils",
    ),

    # Adapters convert to/from external graph backends; they should not depend on I/O.
    "{package}.adapters": (
        "{package}.io",
        "{package}.utils",
    ),

    # I/O may normally depend on core only.
    # Backend-specific exceptions are listed in ALLOWED_IMPORT_PAIRS.
    "{package}.io": (
        "{package}.adapters",
        "{package}.utils",
    ),

    # Utils should remain convenience-level code, not a base layer for other modules.
    "{package}.utils": (
        "{package}.io",
        "{package}.adapters",
    ),
})


# Exact source -> target exceptions.
#
# Keep this small. Every entry here is an intentional architectural exception.
ALLOWED_IMPORT_PAIRS: Final[frozenset[tuple[str, str]]] = frozenset({
    # GraphML can use NetworkX as an implementation detail.
    ("{package}.io.graphml", "{package}.adapters.networkx_adapter"),
})


# Private modules that are allowed to be imported across package boundaries.
#
# Do NOT include:
#   "{package}.core._records"
#
# If records are needed outside core internals, rename/promote:
#   annnet.core._records -> annnet.core.records
ALLOWED_PRIVATE_TARGET_PREFIXES: Final[tuple[str, ...]] = (
    "{package}._support",

    # Shared private helpers inside their own public boundary.
    "{package}.adapters._common",
    "{package}.io._common",
    "{package}.io._archive",

    # Allowed because this is an internal extension boundary.
    # Concrete backend accessors should ideally move out of core.
    "{package}.core.backend_accessors",
)


# Private helper modules that are allowed only for specific boundary-helper symbols.
BOUNDARY_HELPER_TARGET_PREFIXES: Final[tuple[str, ...]] = (
    "{package}.adapters._utils",
)


BOUNDARY_HELPER_SYMBOLS: Final[frozenset[str]] = frozenset({
    "_df_to_rows",
    "_safe_df_to_rows",
})

def expand(pattern: str, package: str) -> str:
    return pattern.format(package=package)


def matches_prefix(value: str, prefix: str) -> bool:
    return value == prefix or value.startswith(prefix + ".")


def matches_any_prefix(value: str, prefixes: tuple[str, ...], package: str) -> bool:
    return any(matches_prefix(value, expand(prefix, package)) for prefix in prefixes)


def matches_pair(
    edge: ImportEdge,
    pairs: frozenset[tuple[str, str]],
    package: str,
) -> bool:
    return (edge.source, edge.target) in {
        (expand(src, package), expand(tgt, package))
        for src, tgt in pairs
    }

# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class ImportEdge:
    source: str
    target: str
    imported: tuple[str, ...] = ()
    import_type: str = "import"
    lazy_kind: str = "eager"
    line: int = 0
    source_file: str = ""


@dataclass
class ModuleInfo:
    module: str
    path: Path
    is_package: bool
    definitions: set[str] = field(default_factory=set)


# -----------------------------
# Module discovery
# -----------------------------

def py_file_to_module(repo: Path, py_file: Path) -> tuple[str, bool]:
    rel = py_file.relative_to(repo)
    parts = list(rel.with_suffix("").parts)

    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]

    return ".".join(parts), is_package


def collect_modules(repo: Path, package: str) -> dict[str, ModuleInfo]:
    package_dir = repo / package.replace(".", "/")
    if not package_dir.exists():
        raise FileNotFoundError(f"Could not find package directory: {package_dir}")

    modules: dict[str, ModuleInfo] = {}

    for py_file in package_dir.rglob("*.py"):
        ignored_parts = {"__pycache__", ".ipynb_checkpoints"}
        if ignored_parts.intersection(py_file.parts):
            continue

        module, is_package = py_file_to_module(repo, py_file)

        # Keep only the requested package.
        if module != package and not module.startswith(package + "."):
            continue

        definitions = parse_top_level_definitions(py_file)
        modules[module] = ModuleInfo(
            module=module,
            path=py_file,
            is_package=is_package,
            definitions=definitions,
        )

    return modules


def parse_top_level_definitions(py_file: Path) -> set[str]:
    try:
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()

    names: set[str] = set()

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)

        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(extract_assigned_names(target))

        elif isinstance(node, ast.AnnAssign):
            names.update(extract_assigned_names(node.target))

    return names


def extract_assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        out: set[str] = set()
        for elt in target.elts:
            out.update(extract_assigned_names(elt))
        return out
    return set()


# -----------------------------
# Import resolution
# -----------------------------

def source_package(module: str, is_package: bool) -> str:
    if is_package:
        return module
    return module.rsplit(".", 1)[0] if "." in module else module


def resolve_relative_import(
    source_module: str,
    source_is_package: bool,
    level: int,
    imported_module: str | None,
) -> str | None:
    """
    Resolve `from ... import ...` into an absolute module path.

    Python semantics:
      level=1 means current package.
      level=2 means parent package.
    """
    if level == 0:
        return imported_module

    pkg = source_package(source_module, source_is_package)
    parts = pkg.split(".")

    up = level - 1
    if up > len(parts):
        return None

    base = ".".join(parts[: len(parts) - up])
    if imported_module:
        return base + "." + imported_module
    return base


def is_type_checking_if(node: ast.If) -> bool:
    """
    Return True for:

        if TYPE_CHECKING:
            ...

        if typing.TYPE_CHECKING:
            ...
    """
    test = node.test

    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True

    if (
        isinstance(test, ast.Attribute)
        and test.attr == "TYPE_CHECKING"
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
    ):
        return True

    return False


def longest_existing_module(candidate: str, known_modules: set[str]) -> str | None:
    """
    Return the longest prefix of candidate that is a known module/package.

    Example:
      candidate = annnet.core.graph.AnnNet
      known contains annnet.core.graph
      -> annnet.core.graph
    """
    parts = candidate.split(".")
    for i in range(len(parts), 0, -1):
        prefix = ".".join(parts[:i])
        if prefix in known_modules:
            return prefix
    return None


def is_internal_module_name(name: str, package: str) -> bool:
    return name == package or name.startswith(package + ".")


def detect_importlib_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """
    Return best-effort aliases for `importlib` and `import_module`.

    This keeps dynamic-import detection simple and intentionally only tracks
    module-level aliases.
    """
    importlib_module_aliases = {"importlib"}
    import_module_aliases: set[str] = set()

    if not isinstance(tree, ast.Module):
        return importlib_module_aliases, import_module_aliases

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    importlib_module_aliases.add(alias.asname or alias.name)

        elif isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    import_module_aliases.add(alias.asname or alias.name)

    return importlib_module_aliases, import_module_aliases


def dynamic_import_target(
    node: ast.Call,
    known_modules: set[str],
    package: str,
    importlib_module_aliases: set[str],
    import_module_aliases: set[str],
) -> str | None:
    func = node.func
    is_import_module_call = False

    if (
        isinstance(func, ast.Attribute)
        and func.attr == "import_module"
        and isinstance(func.value, ast.Name)
        and func.value.id in importlib_module_aliases
    ):
        is_import_module_call = True
    elif isinstance(func, ast.Name) and func.id in import_module_aliases:
        is_import_module_call = True

    if not is_import_module_call or not node.args:
        return None

    first_arg = node.args[0]
    if not isinstance(first_arg, ast.Constant) or not isinstance(first_arg.value, str):
        return None

    raw_target = first_arg.value
    if not is_internal_module_name(raw_target, package):
        return None

    return longest_existing_module(raw_target, known_modules)


def parse_imports(
    modules: dict[str, ModuleInfo],
    package: str,
) -> list[ImportEdge]:
    known = set(modules)
    edges: list[ImportEdge] = []

    def record_import_node(
        node: ast.Import | ast.ImportFrom,
        source: str,
        info: ModuleInfo,
        *,
        lazy_kind: str,
    ) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                raw = alias.name
                if not is_internal_module_name(raw, package):
                    continue

                target = longest_existing_module(raw, known)
                if target is None:
                    continue

                if target == source:
                    continue

                edges.append(
                    ImportEdge(
                        source=source,
                        target=target,
                        imported=(),
                        import_type="import",
                        lazy_kind=lazy_kind,
                        line=node.lineno,
                        source_file=str(info.path),
                    )
                )

        elif isinstance(node, ast.ImportFrom):
            raw_module = resolve_relative_import(
                source_module=source,
                source_is_package=info.is_package,
                level=node.level,
                imported_module=node.module,
            )

            if raw_module is None:
                return

            # Case: from . import graph
            # node.module is None, names may be submodules.
            if node.module is None:
                for alias in node.names:
                    candidate = raw_module + "." + alias.name

                    if candidate in known:
                        target = candidate
                        imported: tuple[str, ...] = ()
                    else:
                        target = longest_existing_module(raw_module, known)
                        if target is None:
                            continue
                        imported = (alias.name,)

                    if not is_internal_module_name(target, package):
                        continue

                    if target == source:
                        continue

                    edges.append(
                        ImportEdge(
                            source=source,
                            target=target,
                            imported=imported,
                            import_type="from",
                            lazy_kind=lazy_kind,
                            line=node.lineno,
                            source_file=str(info.path),
                        )
                    )

            else:
                if not is_internal_module_name(raw_module, package):
                    return

                target = longest_existing_module(raw_module, known)
                if target is None:
                    return

                imported = tuple(alias.name for alias in node.names)

                if target == source:
                    return

                edges.append(
                    ImportEdge(
                        source=source,
                        target=target,
                        imported=imported,
                        import_type="from",
                        lazy_kind=lazy_kind,
                        line=node.lineno,
                        source_file=str(info.path),
                    )
                )

    def visit_runtime_nodes(
        node: ast.AST,
        source: str,
        info: ModuleInfo,
        *,
        in_type_checking_block: bool = False,
        function_depth: int = 0,
        importlib_module_aliases: set[str],
        import_module_aliases: set[str],
    ) -> None:
        """
        Context-aware AST traversal.

        Imports inside `if TYPE_CHECKING:` are intentionally ignored because
        they are not runtime dependencies and must not participate in SCCs.
        """
        if isinstance(node, ast.If) and is_type_checking_if(node):
            # Skip the body entirely: imports here are type-only.
            # Still inspect orelse normally, though it is rare.
            for child in node.orelse:
                visit_runtime_nodes(
                    child,
                    source,
                    info,
                    in_type_checking_block=in_type_checking_block,
                    function_depth=function_depth,
                    importlib_module_aliases=importlib_module_aliases,
                    import_module_aliases=import_module_aliases,
                )
            return

        if in_type_checking_block:
            return

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            record_import_node(
                node,
                source,
                info,
                lazy_kind="lazy_local" if function_depth > 0 else "eager",
            )
            return

        if isinstance(node, ast.Call):
            dynamic_target = dynamic_import_target(
                node,
                known,
                package,
                importlib_module_aliases,
                import_module_aliases,
            )
            if dynamic_target is not None and dynamic_target != source:
                edges.append(
                    ImportEdge(
                        source=source,
                        target=dynamic_target,
                        imported=(),
                        import_type="dynamic_import",
                        lazy_kind="lazy_dynamic",
                        line=node.lineno,
                        source_file=str(info.path),
                    )
                )

        next_function_depth = function_depth
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            next_function_depth += 1

        for child in ast.iter_child_nodes(node):
            visit_runtime_nodes(
                child,
                source,
                info,
                in_type_checking_block=in_type_checking_block,
                function_depth=next_function_depth,
                importlib_module_aliases=importlib_module_aliases,
                import_module_aliases=import_module_aliases,
            )

    for source, info in modules.items():
        try:
            tree = ast.parse(info.path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue

        importlib_module_aliases, import_module_aliases = detect_importlib_aliases(tree)
        visit_runtime_nodes(
            tree,
            source,
            info,
            importlib_module_aliases=importlib_module_aliases,
            import_module_aliases=import_module_aliases,
        )

    return dedupe_edges(edges)


def dedupe_edges(edges: Iterable[ImportEdge]) -> list[ImportEdge]:
    seen: set[tuple[str, str, tuple[str, ...], str, str, int]] = set()
    out: list[ImportEdge] = []

    for edge in edges:
        key = (
            edge.source,
            edge.target,
            tuple(sorted(edge.imported)),
            edge.import_type,
            edge.lazy_kind,
            edge.line,
        )
        if key not in seen:
            seen.add(key)
            out.append(edge)

    return sorted(out, key=lambda e: (e.source, e.target, e.line, e.imported))


# -----------------------------
# Boundary/private-import logic
# -----------------------------

def top_level_area(module: str, package: str) -> str:
    """
    Architectural area used for boundary checks.

    Examples:
      annnet.core.graph -> annnet.core
      annnet.core.backend_accessors._base -> annnet.core
      annnet._support.dataframe_backend -> annnet._support
    """
    parts = module.split(".")
    if len(parts) >= 2 and parts[0] == package:
        return ".".join(parts[:2])
    return module


def display_area(module: str, package: str) -> str:
    """
    Area used for package-level graph display.

    Keeps backend_accessors separate because it is a useful façade.
    """
    parts = module.split(".")
    if len(parts) >= 3 and parts[:3] == [package, "core", "backend_accessors"]:
        return f"{package}.core.backend_accessors"
    if len(parts) >= 2 and parts[0] == package:
        return ".".join(parts[:2])
    return module


def has_private_module_segment(module: str, package: str) -> bool:
    parts = module.split(".")
    for part in parts[1:]:
        if part.startswith("_"):
            return True
    return False


def private_symbols(edge: ImportEdge) -> tuple[str, ...]:
    return tuple(name for name in edge.imported if name.startswith("_"))


def is_cross_area_private_import(edge: ImportEdge, package: str) -> bool:
    return (
        top_level_area(edge.source, package) != top_level_area(edge.target, package)
        and (has_private_module_segment(edge.target, package) or bool(private_symbols(edge)))
    )


def is_boundary_helper_edge(edge: ImportEdge, package: str) -> bool:
    if is_forbidden_import(edge, package):
        return True

    if edge.target == f"{package}.adapters._utils" or edge.target.startswith(
        f"{package}.adapters._utils."
    ):
        return True

    if any(sym in {"_df_to_rows", "_safe_df_to_rows"} for sym in edge.imported):
        return True

    if is_known_architecture_exception(edge, package):
        return True

    return False


# -----------------------------
# CSV outputs
# -----------------------------

def write_import_edges_csv(edges: list[ImportEdge], out: Path, package: str) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source",
                "target",
                "imported",
                "import_type",
                "lazy_kind",
                "line",
                "source_file",
                "source_area",
                "target_area",
                "target_has_private_segment",
                "private_symbols",
                "cross_area_private_import",
                "excluded_from_cycles",
            ],
        )
        writer.writeheader()

        for edge in edges:
            writer.writerow(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "imported": ";".join(edge.imported),
                    "import_type": edge.import_type,
                    "lazy_kind": edge.lazy_kind,
                    "line": edge.line,
                    "source_file": edge.source_file,
                    "source_area": top_level_area(edge.source, package),
                    "target_area": top_level_area(edge.target, package),
                    "target_has_private_segment": has_private_module_segment(edge.target, package),
                    "private_symbols": ";".join(private_symbols(edge)),
                    "cross_area_private_import": is_cross_area_private_import(edge, package),
                    "excluded_from_cycles": edge.lazy_kind == "lazy_dynamic",
                }
            )


def write_private_imports_csv(edges: list[ImportEdge], out: Path, package: str) -> None:
    rows = [
        edge for edge in edges
        if has_private_module_segment(edge.target, package) or private_symbols(edge)
    ]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "source",
                "target",
                "imported",
                "lazy_kind",
                "line",
                "source_area",
                "target_area",
                "cross_area_private_import",
            ],
        )
        writer.writeheader()

        for edge in rows:
            writer.writerow(
                {
                    "source": edge.source,
                    "target": edge.target,
                    "imported": ";".join(edge.imported),
                    "lazy_kind": edge.lazy_kind,
                    "line": edge.line,
                    "source_area": top_level_area(edge.source, package),
                    "target_area": top_level_area(edge.target, package),
                    "cross_area_private_import": is_cross_area_private_import(edge, package),
                }
            )


def write_module_symbols_csv(modules: dict[str, ModuleInfo], out: Path) -> None:
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["module", "symbol", "is_private", "file"])
        writer.writeheader()

        for module, info in sorted(modules.items()):
            for symbol in sorted(info.definitions):
                writer.writerow(
                    {
                        "module": module,
                        "symbol": symbol,
                        "is_private": symbol.startswith("_"),
                        "file": str(info.path),
                    }
                )


# -----------------------------
# Graph helpers
# -----------------------------

def mermaid_id(label: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", label)
    if re.match(r"^\d", safe):
        safe = "_" + safe
    return safe


def mermaid_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


def short_label(symbols: Iterable[str], max_chars: int = 80) -> str:
    items = [s for s in sorted(set(symbols)) if s]
    if not items:
        return ""

    text = ", ".join(items)
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…"


def aggregate_edges(
    edges: Iterable[ImportEdge],
    source_transform=lambda x: x,
    target_transform=lambda x: x,
    skip_self: bool = False,
) -> dict[tuple[str, str], set[str]]:
    agg: dict[tuple[str, str], set[str]] = defaultdict(set)

    for edge in edges:
        src = source_transform(edge.source)
        tgt = target_transform(edge.target)

        if skip_self and src == tgt:
            continue

        if edge.imported:
            agg[(src, tgt)].update(edge.imported)
        else:
            agg[(src, tgt)].add("")

    return agg

# -----------------------------
# Styled graph helpers
# -----------------------------

AREA_STYLES = {
    "root": {
        "label": "{package}",
        "class": "rootpkg",
        "fill": "#f8fafc",
        "stroke": "#334155",
    },
    "support": {
        "label": "{package}._support",
        "class": "support",
        "fill": "#e0f2fe",
        "stroke": "#0284c7",
    },
    "core": {
        "label": "{package}.core",
        "class": "core",
        "fill": "#dcfce7",
        "stroke": "#16a34a",
    },
    "backend_accessors": {
        "label": "{package}.core.backend_accessors",
        "class": "backend",
        "fill": "#f0fdf4",
        "stroke": "#15803d",
    },
    "adapters": {
        "label": "{package}.adapters",
        "class": "adapters",
        "fill": "#ffedd5",
        "stroke": "#ea580c",
    },
    "io": {
        "label": "{package}.io",
        "class": "io",
        "fill": "#fce7f3",
        "stroke": "#db2777",
    },
    "algorithms": {
        "label": "{package}.algorithms",
        "class": "algorithms",
        "fill": "#ede9fe",
        "stroke": "#7c3aed",
    },
    "utils": {
        "label": "{package}.utils",
        "class": "utils",
        "fill": "#f1f5f9",
        "stroke": "#475569",
    },
    "other": {
        "label": "other",
        "class": "other",
        "fill": "#f8fafc",
        "stroke": "#64748b",
    },
}

AREA_ORDER = [
    "root",
    "support",
    "algorithms",
    "core",
    "backend_accessors",
    "adapters",
    "io",
    "utils",
    "other",
]

EDGE_STYLES = {
    "normal": {
        "color": "#64748b",
        "width": "1",
        "dash": "",
        "dot_style": "solid",
    },
    "private": {
        "color": "#d97706",
        "width": "1.5",
        "dash": "5 4",
        "dot_style": "dashed",
    },
    "cross_private": {
        "color": "#dc2626",
        "width": "2",
        "dash": "6 4",
        "dot_style": "dashed",
    },
    "forbidden": {
        "color": "#991b1b",
        "width": "2.5",
        "dash": "3 3",
        "dot_style": "bold,dashed",
    },
    "exception": {
        "color": "#7c3aed",
        "width": "1.8",
        "dash": "6 3",
        "dot_style": "dashed",
    },
}

LOAD_STYLES = {
    "eager": {
        "label": "",
        "mermaid_dash": "",
        "dot_style": "",
        "penwidth_add": 0.0,
    },
    "lazy_local": {
        "label": "",
        "mermaid_dash": "2 6",
        "dot_style": "dotted",
        "penwidth_add": 0.2,
    },
    "lazy_dynamic": {
        "label": "dynamic",
        "mermaid_dash": "1 5",
        "dot_style": "dotted",
        "penwidth_add": 0.4,
    },
}

def area_key(module: str, package: str) -> str:
    """
    Return visual grouping area for a module.

    Keeps core.backend_accessors separate because it is an important façade.
    """
    if module == package:
        return "root"

    parts = module.split(".")

    if len(parts) >= 3 and parts[:3] == [package, "core", "backend_accessors"]:
        return "backend_accessors"

    if len(parts) >= 2 and parts[0] == package:
        name = parts[1]
        if name == "_support":
            return "support"
        if name in {"core", "adapters", "io", "algorithms", "utils"}:
            return name

    return "other"


def area_label(key: str, package: str) -> str:
    return AREA_STYLES.get(key, AREA_STYLES["other"])["label"].format(package=package)


def area_class(key: str) -> str:
    return AREA_STYLES.get(key, AREA_STYLES["other"])["class"]


def area_fill(key: str) -> str:
    return AREA_STYLES.get(key, AREA_STYLES["other"])["fill"]


def area_stroke(key: str) -> str:
    return AREA_STYLES.get(key, AREA_STYLES["other"])["stroke"]


def is_known_architecture_exception(edge: ImportEdge, package: str) -> bool:
    return matches_pair(edge, ALLOWED_IMPORT_PAIRS, package)


def is_allowed_facade_import(edge: ImportEdge, package: str) -> bool:
    return matches_any_prefix(edge.target, ALLOWED_PRIVATE_TARGET_PREFIXES, package)


def is_forbidden_import(edge: ImportEdge, package: str) -> bool:
    if is_known_architecture_exception(edge, package):
        return False

    for source_pattern, target_patterns in FORBIDDEN_IMPORT_PREFIXES.items():
        source_prefix = expand(source_pattern, package)

        if not matches_prefix(edge.source, source_prefix):
            continue

        for target_pattern in target_patterns:
            target_prefix = expand(target_pattern, package)

            if matches_prefix(edge.target, target_prefix):
                return True

    return False


def is_boundary_helper_edge(edge: ImportEdge, package: str) -> bool:
    if is_forbidden_import(edge, package):
        return True

    if matches_any_prefix(edge.target, BOUNDARY_HELPER_TARGET_PREFIXES, package):
        return True

    if any(symbol in BOUNDARY_HELPER_SYMBOLS for symbol in edge.imported):
        return True

    if is_known_architecture_exception(edge, package):
        return True

    return False


def graph_edge_kind(edge: ImportEdge, package: str) -> str:
    if is_forbidden_import(edge, package):
        return "forbidden"

    if is_known_architecture_exception(edge, package):
        return "exception"

    if is_allowed_facade_import(edge, package):
        return "normal"

    if has_private_module_segment(edge.target, package) or private_symbols(edge):
        return "private"

    return "normal"


def combine_edge_kinds(kinds: set[str]) -> str:
    """Pick the strongest visual style for aggregated edges."""
    priority = ["forbidden", "exception", "cross_private", "private", "normal"]

    for kind in priority:
        if kind in kinds:
            return kind

    return "normal"


def edge_load_label(lazy_kind: str) -> str:
    return str(LOAD_STYLES[lazy_kind]["label"])


def merge_dot_styles(*styles: str) -> str:
    out: list[str] = []

    for style in styles:
        if not style:
            continue

        for part in style.split(","):
            part = part.strip()
            if part and part not in out:
                out.append(part)

    return ",".join(out) if out else "solid"


def aggregate_edges_styled(
    edges: Iterable[ImportEdge],
    package: str,
    source_transform=lambda x: x,
    target_transform=lambda x: x,
    skip_self: bool = False,
) -> dict[tuple[str, str], dict[str, object]]:
    agg: dict[tuple[str, str, str], dict[str, object]] = {}

    for edge in edges:
        src = source_transform(edge.source)
        tgt = target_transform(edge.target)

        if skip_self and src == tgt:
            continue

        key = (src, tgt, edge.lazy_kind)

        if key not in agg:
            agg[key] = {
                "symbols": set(),
                "kinds": set(),
                "count": 0,
                "lazy_kind": edge.lazy_kind,
            }

        symbols: set[str] = agg[key]["symbols"]  # type: ignore[assignment]
        kinds: set[str] = agg[key]["kinds"]  # type: ignore[assignment]

        if edge.imported:
            symbols.update(edge.imported)
        else:
            symbols.add("")

        kinds.add(graph_edge_kind(edge, package))
        agg[key]["count"] = int(agg[key]["count"]) + 1

    return agg


def grouped_nodes(nodes: Iterable[str], package: str) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)

    for node in sorted(nodes):
        groups[area_key(node, package)].append(node)

    return groups


def mermaid_id(label: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", label)
    if re.match(r"^\d", safe):
        safe = "_" + safe
    return safe


def mermaid_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


def dot_label(label: str) -> str:
    return label.replace("\\", "\\\\").replace('"', '\\"')


def short_label(symbols: Iterable[str], max_chars: int = 80) -> str:
    items = [s for s in sorted(set(symbols)) if s]
    if not items:
        return ""

    text = ", ".join(items)
    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1] + "…"


def write_symbol_import_mermaid(
    edges: list[ImportEdge],
    modules: dict[str, ModuleInfo] | None,
    out: Path,
    package: str,
    max_edges: int | None = None,
) -> None:
    """
    Styled comprehensive symbol-import Mermaid graph.

    This is not a runtime call graph. It shows imported symbols:
      source module -> target module::symbol
    """
    symbol_edges: list[tuple[str, str, str, str, str]] = []

    for edge in edges:
        kind = graph_edge_kind(edge, package)
        lazy_kind = edge.lazy_kind

        if not edge.imported:
            symbol_edges.append((edge.source, edge.target, "module", kind, lazy_kind))
            continue

        for symbol in edge.imported:
            if symbol == "*":
                symbol_edges.append((edge.source, edge.target + "::*", "star", kind, lazy_kind))
            else:
                symbol_edges.append((edge.source, edge.target + "::" + symbol, "symbol", kind, lazy_kind))

    symbol_edges = sorted(set(symbol_edges))

    if max_edges is not None and max_edges > 0:
        symbol_edges = symbol_edges[:max_edges]

    module_nodes = sorted({src for src, _, _, _, _ in symbol_edges})
    symbol_nodes = sorted({tgt for _, tgt, _, _, _ in symbol_edges})

    lines: list[str] = [
        "flowchart LR",
        "  %% Comprehensive symbol-import graph",
        "  %% This is an import graph, not a runtime call graph.",
        "",
        '  subgraph importing_modules["Importing modules"]',
        "    direction TB",
    ]

    for node in module_nodes:
        cls = area_class(area_key(node, package))
        lines.append(f'    {mermaid_id("module:" + node)}["{mermaid_label(node)}"]')
        lines.append(f"    class {mermaid_id('module:' + node)} {cls};")

    lines.extend([
        "  end",
        "",
        '  subgraph imported_symbols["Imported modules/symbols"]',
        "    direction TB",
    ])

    for node in symbol_nodes:
        module_name = node.split("::", 1)[0]
        cls = area_class(area_key(module_name, package))
        module_name, _, symbol_name = node.partition("::")
        display = symbol_name if symbol_name else module_name
        lines.append(f'    {mermaid_id("symbol:" + node)}["{mermaid_label(display)}"]')
        lines.append(f"    class {mermaid_id('symbol:' + node)} {cls};")

    lines.extend([
        "  end",
        "",
    ])

    link_styles: list[str] = []
    edge_index = 0

    for src, tgt, import_kind, edge_kind, lazy_kind in symbol_edges:
        src_id = mermaid_id("module:" + src)
        tgt_id = mermaid_id("symbol:" + tgt)
        style = EDGE_STYLES[edge_kind]
        load_style = LOAD_STYLES[lazy_kind]
        edge_label = edge_load_label(lazy_kind)

        if import_kind == "star":
            combined_label = "*" if not edge_label else f"* {edge_label}"
            lines.append(f'  {src_id} -->|"{mermaid_label(combined_label)}"| {tgt_id}')
        elif edge_label:
            lines.append(f'  {src_id} -->|"{mermaid_label(edge_label)}"| {tgt_id}')
        else:
            lines.append(f"  {src_id} --> {tgt_id}")

        if edge_kind != "normal" or lazy_kind != "eager":
            dash = load_style["mermaid_dash"] or style["dash"]
            dash_part = f",stroke-dasharray:{dash}" if dash else ""
            link_styles.append(
                f"  linkStyle {edge_index} "
                f"stroke:{style['color']},stroke-width:{float(style['width']) + float(load_style['penwidth_add'])}px{dash_part};"
            )

        edge_index += 1

    lines.append("")

    for key in AREA_ORDER:
        cls = area_class(key)
        lines.append(
            f"  classDef {cls} "
            f"fill:{area_fill(key)},"
            f"stroke:{area_stroke(key)},"
            "stroke-width:1px,"
            "color:#111827;"
        )

    if link_styles:
        lines.append("")
        lines.extend(link_styles)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mermaid_graph(
    edges: Iterable[ImportEdge],
    out: Path,
    package: str,
    title: str,
    labeled: bool = False,
    package_level: bool = False,
    filter_fn=None,
) -> None:
    """
    Styled Mermaid graph with subgraphs/classes.

    Mermaid layout is less controllable than Graphviz, but this gives
    package/module grouping and edge highlighting.
    """
    edge_list = list(edges)

    if filter_fn is not None:
        edge_list = [edge for edge in edge_list if filter_fn(edge)]

    if package_level:
        agg = aggregate_edges_styled(
            edge_list,
            package=package,
            source_transform=lambda m: display_area(m, package),
            target_transform=lambda m: display_area(m, package),
            skip_self=True,
        )
    else:
        agg = aggregate_edges_styled(edge_list, package=package)

    nodes = sorted({node for src, tgt, _ in agg for node in (src, tgt)})
    groups = grouped_nodes(nodes, package)

    lines: list[str] = [
        "flowchart LR",
        f"  %% {title}",
        "  %% A --> B means A imports/depends on B",
        "",
    ]

    # Nodes grouped by architectural area.
    for key in AREA_ORDER:
        group_nodes = groups.get(key, [])
        if not group_nodes:
            continue

        cluster_id = f"cluster_{area_class(key)}"
        lines.append(f'  subgraph {cluster_id}["{mermaid_label(area_label(key, package))}"]')
        lines.append("    direction TB")

        for node in group_nodes:
            lines.append(f'    {mermaid_id(node)}["{mermaid_label(node)}"]')

        lines.append("  end")
        lines.append("")

    # Edge lines and Mermaid linkStyle directives.
    link_styles: list[str] = []
    edge_index = 0

    for (src, tgt, lazy_kind), payload in sorted(agg.items()):
        symbols: set[str] = payload["symbols"]  # type: ignore[assignment]
        kinds: set[str] = payload["kinds"]  # type: ignore[assignment]
        count = int(payload["count"])
        kind = combine_edge_kinds(kinds)
        style = EDGE_STYLES[kind]
        load_style = LOAD_STYLES[lazy_kind]

        if labeled:
            label = short_label(symbols)
            if not label:
                label = f"{count} import{'s' if count != 1 else ''}"
        elif package_level:
            label = f"{count} edge{'s' if count != 1 else ''}"
        else:
            label = ""

        lazy_label = edge_load_label(lazy_kind)
        if lazy_label:
            label = f"{label} [{lazy_label}]".strip() if label else lazy_label

        if label:
            lines.append(
                f'  {mermaid_id(src)} -->|"{mermaid_label(label)}"| {mermaid_id(tgt)}'
            )
        else:
            lines.append(f"  {mermaid_id(src)} --> {mermaid_id(tgt)}")

        if kind != "normal" or lazy_kind != "eager":
            dash = load_style["mermaid_dash"] or style["dash"]
            dash_part = f",stroke-dasharray:{dash}" if dash else ""
            link_styles.append(
                f"  linkStyle {edge_index} "
                f"stroke:{style['color']},stroke-width:{float(style['width']) + float(load_style['penwidth_add'])}px{dash_part};"
            )

        edge_index += 1

    lines.append("")

    # Node classes.
    for node in nodes:
        cls = area_class(area_key(node, package))
        lines.append(f"  class {mermaid_id(node)} {cls};")

    lines.append("")

    # Class definitions.
    for key in AREA_ORDER:
        cls = area_class(key)
        lines.append(
            f"  classDef {cls} "
            f"fill:{area_fill(key)},"
            f"stroke:{area_stroke(key)},"
            "stroke-width:1px,"
            "color:#111827;"
        )

    if link_styles:
        lines.append("")
        lines.extend(link_styles)

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_dot_graph(
    edges: Iterable[ImportEdge],
    out: Path,
    package: str,
    labeled: bool = False,
    package_level: bool = False,
    filter_fn=None,
) -> None:
    """
    Styled Graphviz DOT graph with real clusters.

    This is the closest to the previous online SVG plots because Graphviz
    supports explicit cluster grouping.
    """
    edge_list = list(edges)

    if filter_fn is not None:
        edge_list = [edge for edge in edge_list if filter_fn(edge)]

    if package_level:
        agg = aggregate_edges_styled(
            edge_list,
            package=package,
            source_transform=lambda m: display_area(m, package),
            target_transform=lambda m: display_area(m, package),
            skip_self=True,
        )
    else:
        agg = aggregate_edges_styled(edge_list, package=package)

    nodes = sorted({node for src, tgt, _ in agg for node in (src, tgt)})
    groups = grouped_nodes(nodes, package)

    lines = [
        "digraph G {",
        '  graph [rankdir="LR", bgcolor="white", compound=true, splines=false, pad=0.15, nodesep=0.25, ranksep=0.45];',
        '  node [shape="box", style="rounded,filled", fontname="Helvetica", fontsize=10, color="#64748b", fillcolor="white"];',
        '  edge [fontname="Helvetica", fontsize=9, color="#64748b", arrowsize=0.7];',
        "",
    ]

    # Clustered nodes.
    for key in AREA_ORDER:
        group_nodes = groups.get(key, [])
        if not group_nodes:
            continue

        lines.extend(
            [
                f"  subgraph cluster_{area_class(key)} {{",
                f'    label="{dot_label(area_label(key, package))}";',
                '    style="rounded,filled";',
                f'    color="{area_stroke(key)}";',
                f'    fillcolor="{area_fill(key)}";',
                '    penwidth=1.2;',
                '    fontname="Helvetica-Bold";',
                "    fontsize=12;",
                "",
            ]
        )

        for node in group_nodes:
            lines.append(
                f'    "{dot_label(node)}" '
                f'[label="{dot_label(node)}", '
                f'fillcolor="white", '
                f'color="{area_stroke(key)}"];'
            )

        lines.extend(["  }", ""])

    # Edges.
    for (src, tgt, lazy_kind), payload in sorted(agg.items()):
        symbols: set[str] = payload["symbols"]  # type: ignore[assignment]
        kinds: set[str] = payload["kinds"]  # type: ignore[assignment]
        count = int(payload["count"])
        kind = combine_edge_kinds(kinds)
        style = EDGE_STYLES[kind]
        load_style = LOAD_STYLES[lazy_kind]

        attrs = [
            f'color="{style["color"]}"',
            f'penwidth="{float(style["width"]) + float(load_style["penwidth_add"])}"',
            f'style="{merge_dot_styles(str(style["dot_style"]), str(load_style["dot_style"]))}"',
        ]

        if labeled:
            label = short_label(symbols)
            if not label:
                label = f"{count} import{'s' if count != 1 else ''}"
        elif package_level:
            label = f"{count} edge{'' if count == 1 else 's'}"
        else:
            label = ""

        lazy_label = edge_load_label(lazy_kind)
        if lazy_label:
            label = f"{label} [{lazy_label}]".strip() if label else lazy_label

        if label:
            attrs.append(f'label="{dot_label(label)}"')

        attr_text = ", ".join(attrs)
        lines.append(
            f'  "{dot_label(src)}" -> "{dot_label(tgt)}" [{attr_text}];'
        )

    lines.append("}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_symbol_import_dot(
    edges: list[ImportEdge],
    out: Path,
    package: str,
    max_edges: int | None = None,
) -> None:
    """
    Styled comprehensive symbol-import DOT graph.

    This is still not a call graph. It shows imported symbols:
      source module -> target module::symbol

    It is grouped by source module area and target symbol area.
    """
    symbol_edges: list[tuple[str, str, str, str, str]] = []

    for edge in edges:
        kind = graph_edge_kind(edge, package)
        lazy_kind = edge.lazy_kind

        if not edge.imported:
            symbol_edges.append((edge.source, edge.target, "module", kind, lazy_kind))
            continue

        for symbol in edge.imported:
            if symbol == "*":
                symbol_edges.append((edge.source, edge.target + "::*", "star", kind, lazy_kind))
            else:
                symbol_edges.append((edge.source, edge.target + "::" + symbol, "symbol", kind, lazy_kind))

    symbol_edges = sorted(set(symbol_edges))

    if max_edges is not None and max_edges > 0:
        symbol_edges = symbol_edges[:max_edges]

    module_nodes = sorted({src for src, _, _, _, _ in symbol_edges})
    symbol_nodes = sorted({tgt for _, tgt, _, _, _ in symbol_edges})

    module_rank_nodes = [f"module:{node}" for node in module_nodes]
    symbol_rank_nodes = [f"symbol:{node}" for node in symbol_nodes]

    module_groups = grouped_nodes(module_nodes, package)

    def symbol_module_name(symbol_node: str) -> str:
        return symbol_node.split("::", 1)[0]

    symbol_groups: dict[str, list[str]] = defaultdict(list)
    for symbol_node in symbol_nodes:
        symbol_groups[area_key(symbol_module_name(symbol_node), package)].append(symbol_node)

    lines = [
        "digraph G {",
        '  graph [rankdir="LR", bgcolor="white", compound=true, splines=spline, pad=0.5, nodesep=0.45, ranksep=2.5, concentrate=false];',
        '  node [shape="box", style="rounded,filled", fontname="Helvetica", fontsize=8, margin="0.04,0.02", color="#64748b", fillcolor="white"];',
        '  edge [fontname="Helvetica", fontsize=7, color="#94a3b8", arrowsize=0.45];',
        "",
    ]

    # Left side: importing modules.
    lines.extend(
        [
            "  subgraph cluster_importing_modules {",
            '    label="Importing modules";',
            '    style="rounded,filled";',
            '    color="#94a3b8";',
            '    fillcolor="#f8fafc";',
            "",
        ]
    )

    for key in AREA_ORDER:
        group_nodes = module_groups.get(key, [])
        if not group_nodes:
            continue

        lines.extend(
            [
                f"    subgraph cluster_importing_{area_class(key)} {{",
                f'      label="{dot_label(area_label(key, package))}";',
                '      style="rounded,filled";',
                f'      color="{area_stroke(key)}";',
                f'      fillcolor="{area_fill(key)}";',
            ]
        )

        for node in group_nodes:
            node_id = "module:" + node
            lines.append(
                f'      "{dot_label(node_id)}" '
                f'[label="{dot_label(node)}", '
                f'color="{area_stroke(key)}", '
                f'fillcolor="{area_fill(key)}"];'
            )

        lines.append("    }")

    lines.extend(["  }", ""])

    # Right side: imported symbols.
    lines.extend(
        [
            "  subgraph cluster_imported_symbols {",
            '    label="Imported modules/symbols";',
            '    style="rounded,filled";',
            '    color="#94a3b8";',
            '    fillcolor="#f8fafc";',
            "",
        ]
    )

    for key in AREA_ORDER:
        group_nodes = sorted(symbol_groups.get(key, []))
        if not group_nodes:
            continue

        lines.extend(
            [
                f"    subgraph cluster_symbols_{area_class(key)} {{",
                f'      label="{dot_label(area_label(key, package))}";',
                '      style="rounded,filled";',
                f'      color="{area_stroke(key)}";',
                f'      fillcolor="{area_fill(key)}";',
            ]
        )

        for node in group_nodes:
            node_id = "symbol:" + node
            module_name, _, symbol_name = node.partition("::")
            display = symbol_name if symbol_name else module_name
            lines.append(
                f'      "{dot_label(node_id)}" '
                f'[label="{dot_label(display)}", '
                f'tooltip="{dot_label(node.replace("::", "."))}", '
                f'color="{area_stroke(key)}", '
                f'fillcolor="{area_fill(key)}"];'
            )

        lines.append("    }")

    lines.extend(["  }", ""])

    if module_rank_nodes:
        lines.append("  { rank=source; " + " ".join(f'"{dot_label(n)}";' for n in module_rank_nodes) + " }")
    if symbol_rank_nodes:
        lines.append("  { rank=sink; " + " ".join(f'"{dot_label(n)}";' for n in symbol_rank_nodes) + " }")
    lines.append("")

    # Edges.
    for src, tgt, import_kind, edge_kind, lazy_kind in symbol_edges:
        src_id = "module:" + src
        tgt_id = "symbol:" + tgt
        style = EDGE_STYLES[edge_kind]
        load_style = LOAD_STYLES[lazy_kind]

        edge_color = style["color"]
        if edge_kind == "normal":
            edge_color = "#94a3b8"

        attrs = [
            f'color="{edge_color}55"',  # transparent-ish in SVG Graphviz
            f'penwidth="{0.8 + float(load_style["penwidth_add"])}"',
            f'style="{merge_dot_styles(str(style["dot_style"]), str(load_style["dot_style"]))}"',
        ]

        # Do not label every imported-symbol edge as "symbol"; it creates visual noise.
        # Keep only exceptional import labels if useful.
        edge_label = edge_load_label(lazy_kind)
        if import_kind == "star":
            attrs.append(f'label="{("* " + edge_label).strip()}"' if edge_label else 'label="*"')
        elif edge_label:
            attrs.append(f'label="{dot_label(edge_label)}"')

        lines.append(
            f'  "{dot_label(src_id)}" -> "{dot_label(tgt_id)}" '
            f'[{", ".join(attrs)}];'
        )

    lines.append("}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_svg(dot_file: Path) -> None:
    dot = shutil.which("dot")
    if dot is None:
        print(f"[skip] Graphviz dot not found; not rendering {dot_file.name}")
        return

    svg_file = dot_file.with_suffix(".svg")

    commands = [
        [dot, "-Gsplines=spline", "-Tsvg", str(dot_file), "-o", str(svg_file)],
        [dot, "-Gsplines=curved", "-Tsvg", str(dot_file), "-o", str(svg_file)],
        [dot, "-Gsplines=polyline", "-Tsvg", str(dot_file), "-o", str(svg_file)],
        [dot, "-Gsplines=false", "-Tsvg", str(dot_file), "-o", str(svg_file)],
    ]

    for i, cmd in enumerate(commands, start=1):
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )  # nosec B603 - argv is constructed from trusted local paths and a resolved `dot` binary
            if i > 1:
                print(f"[ok] Rendered {svg_file.name} with Graphviz fallback {i}")
            return
        except subprocess.CalledProcessError as err:
            if i == len(commands):
                print(f"[warn] Could not render {dot_file.name} to SVG.")
                print(err.stderr.strip())
                return


# -----------------------------
# SCC / circularity detection
# -----------------------------

def strongly_connected_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, list[str]] = defaultdict(list)
    for src, tgt in edges:
        graph[src].append(tgt)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    sccs: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for neighbor in graph.get(node, []):
            if neighbor not in indices:
                visit(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in on_stack:
                lowlink[node] = min(lowlink[node], indices[neighbor])

        if lowlink[node] == indices[node]:
            component: list[str] = []
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            sccs.append(sorted(component))

    for node in sorted(nodes):
        if node not in indices:
            visit(node)

    return sorted(sccs, key=lambda c: (len(c), c))


def write_cycles_csv(modules: dict[str, ModuleInfo], edges: list[ImportEdge], out: Path) -> list[list[str]]:
    edge_pairs = {
        (edge.source, edge.target)
        for edge in edges
        if edge.lazy_kind != "lazy_dynamic"
    }
    sccs = strongly_connected_components(modules.keys(), edge_pairs)
    cycles = [c for c in sccs if len(c) > 1]

    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["component_id", "size", "modules"])
        writer.writeheader()

        for i, component in enumerate(cycles, start=1):
            writer.writerow(
                {
                    "component_id": i,
                    "size": len(component),
                    "modules": ";".join(component),
                }
            )

    return cycles


# -----------------------------
# Report
# -----------------------------

def write_summary_report(
    modules: dict[str, ModuleInfo],
    edges: list[ImportEdge],
    cycles: list[list[str]],
    out: Path,
    package: str,
) -> None:
    private_edges = [
        e for e in edges
        if has_private_module_segment(e.target, package) or private_symbols(e)
    ]
    cross_private = [e for e in edges if is_cross_area_private_import(e, package)]
    boundary_edges = [e for e in edges if is_boundary_helper_edge(e, package)]
    lazy_local_edges = [e for e in edges if e.lazy_kind == "lazy_local"]
    lazy_dynamic_edges = [e for e in edges if e.lazy_kind == "lazy_dynamic"]

    area_edges = aggregate_edges(
        edges,
        source_transform=lambda m: display_area(m, package),
        target_transform=lambda m: display_area(m, package),
        skip_self=True,
    )

    lines = [
        "# Import graph report",
        "",
        f"- Package: `{package}`",
        f"- Modules parsed: `{len(modules)}`",
        f"- Internal import edges: `{len(edges)}`",
        f"- Lazy local edges: `{len(lazy_local_edges)}`",
        f"- Lazy dynamic edges: `{len(lazy_dynamic_edges)}`",
        f"- Package-level edges: `{len(area_edges)}`",
        f"- Strongly connected components >1: `{len(cycles)}`",
        f"- Private import edges: `{len(private_edges)}`",
        f"- Cross-area private import edges: `{len(cross_private)}`",
        f"- Boundary/helper cleanup candidate edges: `{len(boundary_edges)}`",
        "",
        "## Cycles",
        "",
    ]

    if not cycles:
        lines.append("No module-level circular imports detected.")
    else:
        for i, component in enumerate(cycles, start=1):
            lines.append(f"### Component {i}")
            for module in component:
                lines.append(f"- `{module}`")
            lines.append("")

    lines.extend(["", "## Cross-area private imports", ""])

    if not cross_private:
        lines.append("No cross-area private imports detected.")
    else:
        for edge in cross_private:
            imported = ", ".join(edge.imported) if edge.imported else "(module)"
            lines.append(
                f"- `{edge.source}` -> `{edge.target}` "
                f"at line `{edge.line}` importing `{imported}`"
            )

    lines.extend(["", "## Boundary/helper cleanup candidates", ""])

    if not boundary_edges:
        lines.append("No boundary/helper cleanup candidates detected.")
    else:
        for edge in boundary_edges:
            imported = ", ".join(edge.imported) if edge.imported else "(module)"
            lines.append(
                f"- `{edge.source}` -> `{edge.target}` "
                f"at line `{edge.line}` importing `{imported}`"
            )

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--package", default="annnet", help="Top-level package name")
    parser.add_argument("--out", default="import_graphs", help="Output directory")
    parser.add_argument(
        "--render-svg",
        action="store_true",
        help="Render SVG files using Graphviz dot, if installed",
    )
    parser.add_argument(
        "--max-symbol-edges",
        type=int,
        default=400,
        help="Limit comprehensive symbol graph edges. 0 means no limit. Default 400.",
    )
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    modules = collect_modules(repo, args.package)
    edges = parse_imports(modules, args.package)

    write_import_edges_csv(edges, out / "import_edges.csv", args.package)
    write_private_imports_csv(edges, out / "private_imports.csv", args.package)
    write_module_symbols_csv(modules, out / "module_symbols.csv")
    cycles = write_cycles_csv(modules, edges, out / "module_cycles.csv")

    graph_specs = [
        {
            "stem": "clean_package_dependency",
            "package_level": True,
            "labeled": False,
            "filter_fn": None,
        },
        {
            "stem": "clean_module_dependency",
            "package_level": False,
            "labeled": False,
            "filter_fn": None,
        },
        {
            "stem": "symbol_labeled_module_dependency",
            "package_level": False,
            "labeled": True,
            "filter_fn": None,
        },
        {
            "stem": "boundary_helper_graph",
            "package_level": False,
            "labeled": True,
            "filter_fn": lambda e: is_boundary_helper_edge(e, args.package),
        },
    ]

    for spec in graph_specs:
        mmd = out / f"{spec['stem']}.mmd"
        dot = out / f"{spec['stem']}.dot"

        write_mermaid_graph(
            edges=edges,
            out=mmd,
            package=args.package,
            title=spec["stem"],
            labeled=spec["labeled"],
            package_level=spec["package_level"],
            filter_fn=spec["filter_fn"],
        )

        write_dot_graph(
            edges=edges,
            out=dot,
            package=args.package,
            labeled=spec["labeled"],
            package_level=spec["package_level"],
            filter_fn=spec["filter_fn"],
        )

        if args.render_svg:
            render_svg(dot)

    write_symbol_import_mermaid(
        package=args.package,
        edges=edges,
        modules=modules,
        out=out / "comprehensive_symbol_import_graph.mmd",
        max_edges=args.max_symbol_edges or None,
    )

    comprehensive_dot = out / "comprehensive_symbol_import_graph.dot"

    write_symbol_import_dot(
        edges=edges,
        out=comprehensive_dot,
        package=args.package,
        max_edges=args.max_symbol_edges or None,
    )

    if args.render_svg:
        render_svg(comprehensive_dot)

    write_summary_report(
        modules=modules,
        edges=edges,
        cycles=cycles,
        out=out / "import_graph_report.md",
        package=args.package,
    )

    print(f"Wrote outputs to: {out}")
    print(f"Modules parsed: {len(modules)}")
    print(f"Internal import edges: {len(edges)}")
    print(f"Cycles detected: {len(cycles)}")

    if cycles:
        print("Circular components:")
        for component in cycles:
            print("  - " + " -> ".join(component))


if __name__ == "__main__":
    main()
