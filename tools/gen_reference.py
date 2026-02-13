from __future__ import annotations

from pathlib import Path

import mkdocs_gen_files

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "annnet"


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT)
    parts = list(rel.with_suffix("").parts)
    if path.name == "__init__.py":
        parts = parts[:-1]
    return ".".join(parts)


def _out_path(path: Path) -> Path:
    rel = path.relative_to(ROOT)
    parts = list(rel.with_suffix("").parts)
    if path.name == "__init__.py":
        parts = parts[:-1]
        return Path("reference/generated").joinpath(*parts, "index.md")
    return Path("reference/generated").joinpath(*parts).with_suffix(".md")


def _is_internal(module: str) -> bool:
    parts = module.split(".")[1:]  # skip top-level package name
    return any(p.startswith("_") for p in parts)


def iter_modules():
    for path in sorted(PKG.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module = _module_name(path)
        if not module:
            continue
        yield module, path


def write_module_page(module: str, path: Path) -> Path:
    out_path = _out_path(path)
    with mkdocs_gen_files.open(out_path, "w") as f:
        f.write(f"# {module}\n\n")
        f.write(f"::: {module}\n")
        f.write("    options:\n")
        f.write("      members: true\n")
        f.write("      inherited_members: false\n")
        f.write("      show_root_heading: true\n")
        f.write("      show_root_toc_entry: false\n")
        f.write("      filters:\n")
        f.write("      - \"!^_\"\n")
    return out_path


def write_index(modules: list[tuple[str, Path]]):
    public = []
    internal = []
    for mod, path in modules:
        out_path = _out_path(path)
        rel = out_path.relative_to("reference")
        if _is_internal(mod):
            internal.append((mod, rel))
        else:
            public.append((mod, rel))

    with mkdocs_gen_files.open("reference/index.md", "w") as f:
        f.write("# Full API Index\n\n")
        f.write(
            "This page is generated from the package source at build time. "
            "If something is missing, add a module to the package or check "
            "`tools/gen_reference.py`.\n\n"
        )

        f.write("## Public Modules\n\n")
        for mod, rel in public:
            f.write(f"- [{mod}]({rel.as_posix()})\n")

        f.write("\n## Internal Modules\n\n")
        for mod, rel in internal:
            f.write(f"- [{mod}]({rel.as_posix()})\n")


def write_summary(modules: list[tuple[str, Path]]):
    nav = mkdocs_gen_files.Nav()
    nav["Full API Index"] = "index.md"
    for mod, path in modules:
        out_path = _out_path(path)
        rel = out_path.relative_to("reference")
        nav[("Modules", *mod.split("."))] = rel.as_posix()
    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as f:
        f.writelines(nav.build_literate_nav())


def main():
    modules = list(iter_modules())
    for mod, path in modules:
        write_module_page(mod, path)
    write_index(modules)
    write_summary(modules)


main()
