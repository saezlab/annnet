"""The canonical store holds no matrix object and imports no matrix library.

A matrix is derived state. A store that builds one at construction time drags a
matrix library into every graph, including a graph that never runs linear algebra.
The rule is checkable, so it is checked.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STORE_MODULES = ('annnet/core/_store.py', 'annnet/core/_state.py')
MATRIX_LIBRARIES = {'scipy', 'scipy.sparse'}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f'{node.module}.{alias.name}' for alias in node.names)
    return found


def test_the_store_modules_import_no_matrix_library():
    violations = []
    for relative in STORE_MODULES:
        path = ROOT / relative
        if not path.exists():
            continue
        for name in _imported_modules(path):
            root = name.split('.')[0]
            if root in {library.split('.')[0] for library in MATRIX_LIBRARIES}:
                violations.append(f'{relative} imports {name}')
    assert not violations, 'the canonical store must hold no matrix object: ' + ', '.join(
        sorted(violations)
    )


def test_the_store_module_builds_no_matrix_at_construction():
    from annnet.core import _store

    state = _store.CoreState(directed=True)
    for name in dir(state):
        if name.startswith('__'):
            continue
        value = getattr(state, name, None)
        assert not hasattr(value, 'tocsr'), f'{name} is a materialized matrix in the store'
