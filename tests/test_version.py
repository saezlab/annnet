"""The version lives in one file, and a bump moves one line.

``pyproject.toml`` carries the version. ``annnet.__version__`` reads it from
there through ``_support.metadata``, so nothing in the package repeats the
number and nothing can disagree with it.

``.bumpversion.cfg`` is what moves it. bump2version rewrites that file on every
bump and drops the comments, so the reason its search is scoped cannot live in
the file itself. It lives here.

**The reason.** The sibling packages write ``files = pyproject.toml``, which
replaces every occurrence of the version string anywhere in the file. In annnet
``0.2.0`` was a substring of the ``codecov-cli>=10.2.0`` pin, so that form
rewrites the pin to ``>=10.3.0`` on a minor bump, silently, and the next install
resolves a dependency nobody asked for. The scoped ``search`` and ``replace``
are what stop it, and this module fails when somebody takes them out.
"""

from __future__ import annotations

import configparser
import re
import tomllib
from pathlib import Path

import pytest

import annnet

PROJECT = Path(annnet.__file__).parent.parent
PYPROJECT = PROJECT / 'pyproject.toml'
BUMPVERSION = PROJECT / '.bumpversion.cfg'

pytestmark = pytest.mark.skipif(
    not (PYPROJECT.is_file() and BUMPVERSION.is_file()),
    reason='the project files are not beside the package',
)

SEMVER = re.compile(r'^(\d+)\.(\d+)\.(\d+)$')


def declared_version() -> str:
    return tomllib.loads(PYPROJECT.read_text())['project']['version']


def bump_config() -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read_string(BUMPVERSION.read_text())
    return parser


def test_the_declared_version_is_a_semantic_version():
    assert SEMVER.match(declared_version()), declared_version()


def test_the_package_reports_the_version_the_project_declares():
    assert annnet.__version__ == declared_version()


def test_bumpversion_agrees_with_the_project():
    """A bump reads its own current version, so a disagreement bumps the wrong one."""
    assert bump_config()['bumpversion']['current_version'] == declared_version()


def test_the_bump_is_scoped_to_the_version_assignment():
    """The bare ``files =`` form rewrites any pin the version is a substring of."""
    config = bump_config()
    assert 'files' not in config['bumpversion'], (
        'the bare `files =` form replaces every occurrence of the version string '
        'in the file. Use a [bumpversion:file:...] section with `search` and '
        '`replace` scoped to the assignment, and read this module for why.'
    )
    section = config['bumpversion:file:pyproject.toml']
    assert section['search'] == 'version = "{current_version}"'
    assert section['replace'] == 'version = "{new_version}"'


@pytest.mark.parametrize('part', ['major', 'minor', 'patch'])
def test_a_bump_of_each_part_moves_exactly_the_version_line(part):
    """Apply the configured replacement by hand and count what moves.

    This is the check that matters. It does not trust the shape of the config;
    it performs the substitution the config describes and asserts that one line
    of ``pyproject.toml`` differs, and that the line is the version.
    """
    major, minor, patch = (int(value) for value in SEMVER.match(declared_version()).groups())
    bumped = {
        'major': f'{major + 1}.0.0',
        'minor': f'{major}.{minor + 1}.0',
        'patch': f'{major}.{minor}.{patch + 1}',
    }[part]

    section = bump_config()['bumpversion:file:pyproject.toml']
    search = section['search'].format(current_version=declared_version())
    replace = section['replace'].format(new_version=bumped)

    before = PYPROJECT.read_text().splitlines()
    after = PYPROJECT.read_text().replace(search, replace).splitlines()

    moved = [(old, new) for old, new in zip(before, after, strict=True) if old != new]
    assert len(moved) == 1, f'a {part} bump moves {len(moved)} lines: {moved}'
    assert moved[0] == (f'version = "{declared_version()}"', f'version = "{bumped}"')


def test_no_other_tracked_file_carries_the_version_as_a_source():
    """``uns``, the metadata module and the docs read the version; none states it.

    A lockfile records a resolved version, which its own tool regenerates, so
    those are the one place a stale copy is expected and harmless.
    """
    version = declared_version()
    stating = []
    for path in PROJECT.rglob('*.py'):
        if '.venv' in path.parts or 'site-packages' in path.parts:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(rf'^\s*(__version__|version)\s*=\s*[\'"]{re.escape(version)}', line):
                stating.append(f'{path.relative_to(PROJECT)}:{number}')
    assert not stating, f'these files state the version instead of reading it: {stating}'
