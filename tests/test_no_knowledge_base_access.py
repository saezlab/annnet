"""The package holds no access to any one knowledge base.

Constitution Principle IV says that prior knowledge is reached through the
`omnipath-client` package. A generic network data structure that downloads one
knowledge base is not generic, and every consumer of it pays for a dependency it
does not use.

Three rules, and each one is checked over the source rather than by importing,
so a module that fails to import for another reason cannot hide a breach:

- no module of the package imports an HTTP client,
- every third-party module the package imports is a declared dependency,
- no loader for one named knowledge base reaches the public surface.
"""

from __future__ import annotations

import ast
import sys
import tomllib
from pathlib import Path

import pytest

import annnet

PACKAGE = Path(annnet.__file__).parent
PROJECT = PACKAGE.parent

# An HTTP client the package would have to depend on. It downloads nothing, so
# it imports none of these, in any module and under any name.
HTTP_CLIENTS = frozenset(
    {
        'requests',
        'httpx',
        'urllib3',
        'aiohttp',
    }
)

# The standard library speaks HTTP too, and declaring it buys nothing. So the
# rule over it is different: the package may hold a call that reaches the
# network, and nothing inside the package may make one. This is the whole list,
# and each entry is a function a user calls on purpose.
OPT_IN_NETWORK_CALLS = {
    'get_latest_version': 'reads the version on the default branch, when a user asks for it',
}

# The knowledge bases the package must not reach for. The word may appear in a
# comment or in a test; what it must not do is name a module or a public
# function of the package.
KNOWLEDGE_BASES = ('omnipath',)


def python_files() -> list[Path]:
    return sorted(PACKAGE.rglob('*.py'))


def imported_modules(path: Path) -> set[str]:
    """Return the top-level module of every import in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                found.add(node.module)
    return found


def declared_dependencies() -> set[str]:
    """Return every distribution the project declares, required or optional."""
    manifest = tomllib.loads((PROJECT / 'pyproject.toml').read_text())
    project = manifest['project']
    specs = list(project.get('dependencies', ()))
    for extra in project.get('optional-dependencies', {}).values():
        specs.extend(extra)
    names = set()
    for spec in specs:
        name = spec.split(';')[0]
        for separator in ('>=', '<=', '==', '!=', '~=', '>', '<', '['):
            name = name.split(separator)[0]
        names.add(name.strip().lower().replace('-', '_'))
    return names


# The import name of a distribution, where the two differ.
IMPORT_NAME = {
    'python_igraph': 'igraph',
    'python_libsbml': 'libsbml',
}

# graph-tool is not on PyPI and cannot be installed by pip, so no manifest can
# declare it. The adapter imports it inside the call that needs it and says so
# when it is absent, which is the whole of what a declaration would buy.
UNDECLARABLE = frozenset({'graph_tool'})


def test_no_module_imports_an_http_client():
    offenders = []
    for path in python_files():
        for module in imported_modules(path):
            root = module.split('.')[0]
            if root in HTTP_CLIENTS or module in HTTP_CLIENTS:
                offenders.append(f'{path.relative_to(PROJECT)} imports {module}')
    assert not offenders, 'the package downloads nothing and needs no HTTP client:\n' + '\n'.join(
        offenders
    )


def called_names(path: Path) -> set[str]:
    """Return the name of every call in one file, as it is written there."""
    tree = ast.parse(path.read_text(), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Name):
            found.add(target.id)
        elif isinstance(target, ast.Attribute):
            found.add(target.attr)
    return found


def test_nothing_in_the_package_reaches_the_network_on_its_own():
    """A call that reaches the network is one a user makes, never the package."""
    callers = []
    for path in python_files():
        for name in called_names(path) & set(OPT_IN_NETWORK_CALLS):
            callers.append(f'{path.relative_to(PROJECT)} calls {name}')
    assert not callers, 'the package reaches the network without being asked:\n' + '\n'.join(
        callers
    )


def test_every_third_party_import_is_declared():
    declared = {IMPORT_NAME.get(name, name) for name in declared_dependencies()}
    standard = sys.stdlib_module_names
    offenders = []
    for path in python_files():
        for module in imported_modules(path):
            root = module.split('.')[0]
            if root in {'annnet', ''} or module.startswith('.'):
                continue
            # A distribution name is case-insensitive and a module name is not,
            # so the two are compared in one case.
            if root in standard or root.lower() in declared or root in UNDECLARABLE:
                continue
            offenders.append(f'{path.relative_to(PROJECT)} imports {root}')
    assert not offenders, 'these are imported and not declared in pyproject.toml:\n' + '\n'.join(
        sorted(set(offenders))
    )


@pytest.mark.parametrize('knowledge_base', KNOWLEDGE_BASES)
def test_no_module_is_named_after_a_knowledge_base(knowledge_base):
    named = [
        str(path.relative_to(PROJECT))
        for path in python_files()
        if knowledge_base in path.stem.lower()
    ]
    assert not named, (
        f'{knowledge_base} access belongs in the omnipath-client package, not here: {named}'
    )


@pytest.mark.parametrize('knowledge_base', KNOWLEDGE_BASES)
def test_no_public_name_reaches_a_knowledge_base(knowledge_base):
    import annnet.io

    surfaces = {
        'annnet': annnet.__all__,
        'annnet.io': getattr(annnet.io, '__all__', dir(annnet.io)),
    }
    offenders = [
        f'{module}.{name}'
        for module, names in surfaces.items()
        for name in names
        if knowledge_base in name.lower()
    ]
    assert not offenders, (
        f'the public surface reaches {knowledge_base} directly: {offenders}. '
        f'Use omnipath_client, which returns an AnnNet object.'
    )
