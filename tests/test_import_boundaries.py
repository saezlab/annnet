from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / 'annnet'
ALLOWED_PRIVATE_IMPORTS = {
    ('annnet.io', 'annnet._support.graph_records'),
    ('annnet.adapters', 'annnet._support.graph_records'),
}


@dataclass(frozen=True)
class ImportRef:
    source_module: str
    source_file: Path
    target_module: str
    imported_name: str | None
    lineno: int


def _module_name(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix('')
    parts = list(rel.parts)
    if parts[-1] == '__init__':
        parts = parts[:-1]
    return '.'.join(parts)


def _resolve_from_import(source_module: str, level: int, module: str | None) -> str:
    package_parts = source_module.split('.')[:-1]
    if level:
        package_parts = package_parts[: len(package_parts) - level + 1]
    target_parts = package_parts + ([module] if module else [])
    return '.'.join(part for part in target_parts if part)


def _subpackage_name(module_name: str) -> str:
    parts = module_name.split('.')
    return parts[1] if len(parts) > 1 else ''


def _is_private_cross_boundary(
    source_module: str, target_module: str, imported_name: str | None
) -> bool:
    source_prefix = _subpackage_name(source_module)
    target_prefix = _subpackage_name(target_module)
    if not source_prefix or not target_prefix or source_prefix == target_prefix:
        return False
    if any(segment.startswith('_') for segment in target_module.split('.')[2:]):
        return True
    return bool(imported_name and imported_name.startswith('_'))


def _is_allowed_private_import(source_module: str, target_module: str) -> bool:
    source_prefix = f'annnet.{_subpackage_name(source_module)}'
    target_prefix = '.'.join(target_module.split('.')[:3])
    return (source_prefix, target_prefix) in ALLOWED_PRIVATE_IMPORTS


def _iter_internal_imports() -> list[ImportRef]:
    refs: list[ImportRef] = []
    for path in PKG_ROOT.rglob('*.py'):
        source_module = _module_name(path)
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith('annnet.'):
                        refs.append(
                            ImportRef(
                                source_module, path, alias.name, None, getattr(node, 'lineno', 0)
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                target_module = _resolve_from_import(source_module, node.level, node.module)
                if not target_module.startswith('annnet'):
                    continue
                for alias in node.names:
                    refs.append(
                        ImportRef(
                            source_module,
                            path,
                            target_module,
                            alias.name,
                            getattr(node, 'lineno', 0),
                        )
                    )
    return refs


def test_internal_import_boundaries():
    violations: list[str] = []

    for ref in _iter_internal_imports():
        src = ref.source_module
        dst = ref.target_module
        imported = f'.{ref.imported_name}' if ref.imported_name else ''
        location = f'{ref.source_file.relative_to(ROOT)}:{ref.lineno}'

        if src.startswith('annnet._support') and (
            dst.startswith('annnet.core')
            or dst.startswith('annnet.adapters')
            or dst.startswith('annnet.io')
        ):
            violations.append(
                f'{location}: {src} imports {dst}{imported} but _support must not depend on core/adapters/io'
            )

        if (
            src.startswith('annnet.core')
            and not src.startswith('annnet.core.backend_accessors')
            and (dst.startswith('annnet.adapters') or dst.startswith('annnet.io'))
        ):
            violations.append(
                f'{location}: {src} imports {dst}{imported} but core must not depend on adapters/io'
            )

        if src.startswith('annnet.adapters') and dst.startswith('annnet.io'):
            violations.append(
                f'{location}: {src} imports {dst}{imported} but adapters must not depend on io'
            )

        if src.startswith('annnet.io') and dst == 'annnet.adapters._utils':
            violations.append(
                f'{location}: {src} imports {dst}{imported} but io must not depend on adapters._utils'
            )

        if _is_private_cross_boundary(
            src, dst, ref.imported_name
        ) and not _is_allowed_private_import(src, dst):
            violations.append(
                f'{location}: {src} imports private target {dst}{imported} across subpackage boundary'
            )

    assert not violations, '\n'.join(violations)
