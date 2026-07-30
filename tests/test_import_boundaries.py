from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

from annnet.core._state import (
    CO_MAINTAINED_FIELDS,
    DERIVED_FIELDS,
    SOT_FIELDS,
)


ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / 'annnet'
CORE_ROOT = PKG_ROOT / 'core'
ALLOWED_PRIVATE_IMPORTS = {
    ('annnet.io', 'annnet._support.graph_records'),
    ('annnet.adapters', 'annnet._support.graph_records'),
}

# The structural query facade is the sanctioned door into the core. Every module
# of the package may import it, and no module may reach past it.
FACADE_MODULE = 'annnet.core._structure'

# The canonical store of a graph. The field inventory in ``annnet.core._state`` is
# the source of this list, so a new canonical field is covered without a second
# edit here. The extra names below hold the same state under a different door.
EXTRA_STORE_FIELDS = (
    '_store',
    '_matrix_cache',
    '_matrix_shape',
    '_matrix_dirty',
    '_edge_indexes_built',
    '_supra_index_cache',
    '_vertex_key_index',
)
PRIVATE_STORE_FIELDS = frozenset(
    name
    for name in SOT_FIELDS + CO_MAINTAINED_FIELDS + DERIVED_FIELDS + EXTRA_STORE_FIELDS
    if name.startswith('_')
)

# Functions that still touch the canonical store from outside the core. Each one
# rebuilds a graph from a file or from another library, so it constructs the store
# rather than querying it. The new store gives them a proper entry point and this
# list then goes away. Nothing may join it.
STORE_RESTORE_LEDGER = frozenset(
    {
        'annnet.io.annnet_format:_load_structure',
        'annnet.io.annnet_format:_recover_legacy_coeffs',
        'annnet.io.annnet_format:_load_multilayers',
        'annnet.io.annnet_format:_load_slices',
        'annnet.io.cx2:from_cx2',
        'annnet.adapters.graphtool_adapter:from_graphtool',
        'annnet._support.serialization:restore_multilayer_manifest',
    }
)


@dataclass(frozen=True)
class ImportRef:
    source_module: str
    source_file: Path
    target_module: str
    imported_name: str | None
    lineno: int


@dataclass(frozen=True)
class StoreAccess:
    source_module: str
    source_file: Path
    field: str
    lineno: int
    is_write: bool
    scope: str

    @property
    def entry(self) -> str:
        """The ledger key of this access, which is ``module:function``."""
        return f'{self.source_module}:{self.scope}'


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


def _is_allowed_private_import(
    source_module: str, target_module: str, imported_name: str | None = None
) -> bool:
    # ``import annnet.core._structure`` and ``from annnet.core import _structure``
    # name the same module, so both forms have to resolve to the facade.
    if FACADE_MODULE in (target_module, f'{target_module}.{imported_name}'):
        return True
    source_prefix = f'annnet.{_subpackage_name(source_module)}'
    target_prefix = '.'.join(target_module.split('.')[:3])
    return (source_prefix, target_prefix) in ALLOWED_PRIVATE_IMPORTS


def _non_core_modules():
    """Yield every package module that sits outside ``annnet/core``."""
    for path in sorted(PKG_ROOT.rglob('*.py')):
        if CORE_ROOT in path.parents or path == CORE_ROOT:
            continue
        yield path


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


class _StoreAccessVisitor(ast.NodeVisitor):
    """Collect store accesses together with the function that holds them."""

    def __init__(self, source_module: str, source_file: Path):
        self.source_module = source_module
        self.source_file = source_file
        self.accesses: list[StoreAccess] = []
        self._scope: list[str] = []

    def _in_scope(self, node):
        self._scope.append(node.name)
        self.generic_visit(node)
        self._scope.pop()

    visit_FunctionDef = _in_scope
    visit_AsyncFunctionDef = _in_scope
    visit_ClassDef = _in_scope

    def visit_Attribute(self, node):
        if node.attr in PRIVATE_STORE_FIELDS:
            self.accesses.append(
                StoreAccess(
                    source_module=self.source_module,
                    source_file=self.source_file,
                    field=node.attr,
                    lineno=getattr(node, 'lineno', 0),
                    is_write=isinstance(node.ctx, (ast.Store, ast.Del)),
                    scope='.'.join(self._scope) or '<module>',
                )
            )
        self.generic_visit(node)


def _iter_store_accesses() -> list[StoreAccess]:
    """Find every access to a canonical store field outside the core."""
    accesses: list[StoreAccess] = []
    for path in _non_core_modules():
        visitor = _StoreAccessVisitor(_module_name(path), path)
        visitor.visit(ast.parse(path.read_text(encoding='utf-8'), filename=str(path)))
        accesses.extend(visitor.accesses)
    return accesses


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
        ) and not _is_allowed_private_import(src, dst, ref.imported_name):
            violations.append(
                f'{location}: {src} imports private target {dst}{imported} across subpackage boundary'
            )

    assert not violations, '\n'.join(violations)


def test_no_code_outside_the_core_touches_the_private_store():
    """Structure is read through ``annnet.core._structure`` and nowhere else.

    A read of a canonical store field outside the core ties that module to one
    store layout. The refactor replaces the layout, so every such read has to go
    through the facade instead. The only exception is a function on the ledger,
    which rebuilds a store rather than querying one.
    """
    violations = [
        f'{access.source_file.relative_to(ROOT)}:{access.lineno}: '
        f'{access.entry} touches the private store field {access.field!r}. '
        f'Read structure through annnet.core._structure instead.'
        for access in _iter_store_accesses()
        if access.entry not in STORE_RESTORE_LEDGER
    ]
    assert not violations, '\n'.join(sorted(violations))


def test_every_ledger_entry_still_touches_the_store():
    """A ledger entry that no longer touches the store must leave the ledger.

    Without this check the ledger would keep growing stale exemptions, and the
    boundary would weaken one forgotten line at a time.
    """
    live = {access.entry for access in _iter_store_accesses()}
    stale = sorted(STORE_RESTORE_LEDGER - live)
    assert not stale, (
        'these ledger entries no longer touch the canonical store, so remove them '
        'from STORE_RESTORE_LEDGER: ' + ', '.join(stale)
    )


def test_the_store_field_list_is_not_empty():
    """A silent drop of the field inventory would make the boundary test vacuous."""
    assert {'_entities', '_edges', '_matrix'} <= PRIVATE_STORE_FIELDS
