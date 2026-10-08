"""The foundation imports no domain package.

``annnet`` sits at the bottom of the ecosystem's one import direction,
``sysbioverse → biology → annnet``. Nothing in it may import either package
above it — not lazily, not under ``TYPE_CHECKING``, not in a test.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOMAIN = ('biology', 'sysbioverse')


def _files():
    yield from sorted((ROOT / 'annnet').rglob('*.py'))
    yield from sorted((ROOT / 'tests').glob('*.py'))


@pytest.mark.parametrize('path', list(_files()), ids=lambda p: str(p.relative_to(ROOT)))
def test_nothing_imports_a_domain_package(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names = [node.module or '']
        else:
            continue
        for name in names:
            if name.split('.')[0] in DOMAIN:
                found.append(f'{node.lineno}: imports {name}')
    assert not found, '\n'.join(found)
