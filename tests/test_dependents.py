"""The names other packages call must keep resolving.

AnnNet removes a public name without a deprecation period, which is the right
choice before a first stable release and a real cost to every package that
bridges to us. None of those packages is in this test suite, so a rename here
lands in them silently and somebody finds it by running into it.

``dependents.toml`` is the register: one entry per package, with the AnnNet
names that package calls. This module asserts that each of them still resolves
on the public surface. A rename therefore fails the build in this repository,
where the person making it can see who to tell, rather than in a repository they
are not looking at.

What this does not do is prove a dependent works. The register is written by
hand, so a bridge may call more than it lists. It catches the common case, which
is a rename or a removal of a name somebody already told us about.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

import pytest

import annnet
from annnet import AnnNet
from annnet.core._records import _EDGE_RESERVED, _node_RESERVED

REGISTER = Path(annnet.__file__).parent.parent / 'dependents.toml'

# Where a dependent's migration stands. The order is the order of confidence.
#
#   required_before_release  no migration is known to pass its tests
#   verified_locally         a migration applied to ``upstream_revision`` passed the
#                            package's own tests against AnnNet ``verified_annnet``;
#                            the package's repository has not merged it
#   merged_upstream          the migration is on the package's default branch
STATUSES = ('required_before_release', 'verified_locally', 'merged_upstream')
VERIFIED = ('verified_locally', 'merged_upstream')

# The register sits beside the package, so it is there in a checkout and in an
# unpacked sdist, and absent when the tests run against an installed wheel.
pytestmark = pytest.mark.skipif(
    not REGISTER.is_file(), reason='dependents.toml is not beside the package'
)

# The namespaces a dotted name may start from. A dependent writes
# ``G.layers.node_attrs``, and the register spells it the same way.
NAMESPACES = (
    'attrs',
    'ops',
    'slices',
    'layers',
    'history',
    'idx',
    'cache',
    'N',
    'E',
)


def register() -> dict:
    return tomllib.loads(REGISTER.read_text())


def packages() -> list[dict]:
    return register()['package']


def resolves(graph: AnnNet, dotted: str) -> bool:
    """Whether one dotted name of the register resolves on a graph."""
    head, *rest = dotted.split('.')
    if head == 'AnnNet':
        return not rest
    target = graph
    for part in (head, *rest):
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


@pytest.fixture(scope='module')
def graph() -> AnnNet:
    """One graph with both axes and a layer, so every namespace answers."""
    g = AnnNet(directed=True, aspects={'condition': ['a', 'b']})
    g.add_nodes(['x', 'y'], layer=('a',))
    g.add_edges({'source': ('x', ('a',)), 'target': ('y', ('a',)), 'edge_id': 'e'})
    return g


def test_the_register_parses_and_is_not_empty():
    entries = packages()
    assert entries, 'dependents.toml lists no package'
    for entry in entries:
        for field in (
            'name',
            'owner',
            'repository',
            'contact',
            'modules',
            'calls',
            'previous_calls',
            'migration',
            'migration_status',
        ):
            assert entry.get(field), f'{entry.get("name", entry)!r} is missing {field!r}'
        assert entry['migration_status'] in STATUSES


@pytest.mark.parametrize('entry', packages(), ids=lambda entry: entry['name'])
def test_every_name_a_dependent_calls_still_resolves(entry, graph):
    """A rename fails here, and the failure names the package to update.

    Add the new spelling to ``dependents.toml`` in the same change that renames
    the name, and open a pull request against the repository the entry names.
    """
    missing = [name for name in entry['calls'] if not resolves(graph, name)]
    assert not missing, (
        f'{entry["name"]} calls {missing}, which the public surface no longer carries. '
        f'It bridges through {", ".join(entry["modules"])}. '
        f'Owner: {entry["owner"]} ({entry["contact"]}) at {entry["repository"]}. '
        f'Name the replacement in CHANGELOG.md, update dependents.toml, '
        f'and see DEPENDENTS.md.'
    )


@pytest.mark.parametrize('entry', packages(), ids=lambda entry: entry['name'])
def test_every_previous_call_that_no_longer_resolves_has_a_migration_line(entry, graph):
    """A removed name the package was seen
    calling must be named in its migration record. This proves the record,
    not the package — the package is edited in its own repository."""
    gone = [name for name in entry['previous_calls'] if not resolves(graph, name)]
    lines = ' '.join(entry['migration'])
    missing = [name for name in gone if name.split('.')[-1] not in lines]
    assert not missing, (
        f'{entry["name"]} was observed calling {missing}, which no longer resolve, '
        f'and the register names no migration for them'
    )


@pytest.mark.parametrize('entry', packages(), ids=lambda entry: entry['name'])
def test_a_verified_entry_says_what_it_was_verified_against(entry):
    """Verified means something specific, so the entry has to say what.

    ``verified_locally`` and ``merged_upstream`` name the upstream commit the
    migration was applied to, the AnnNet version it was run against, the digest
    of the patch that was tested, and the command that ran. ``merged_upstream``
    also names the commit that carries the migration.
    """
    if entry['migration_status'] not in VERIFIED:
        return
    required = ['upstream_revision', 'verified_annnet', 'verification', 'patch_sha256']
    if entry['migration_status'] == 'merged_upstream':
        required.append('merged_revision')
    missing = [field for field in required if not entry.get(field)]
    assert not missing, f'{entry["name"]} is {entry["migration_status"]} but lacks {missing}'
    assert len(entry['upstream_revision']) >= 7


def gate() -> str:
    return os.environ.get('ANNNET_RELEASE_GATE', '').strip().lower()


@pytest.mark.skipif(
    gate() not in ('1', 'compat', 'public'),
    reason='the compatibility gate runs with ANNNET_RELEASE_GATE=1',
)
def test_compatibility_gate_every_dependent_has_a_verified_migration():
    """A migration exists, and passed the dependent's own tests against this AnnNet.

    The gate fails while any entry is ``required_before_release``, and while a
    verified entry was verified against a different AnnNet version than the one
    in this tree, because a change of version is a change that has to be tested
    again. It does not say the dependent's repository carries the change; that is
    the public gate.
    """
    unresolved = [e['name'] for e in packages() if e['migration_status'] not in VERIFIED]
    assert not unresolved, f'dependents with no verified migration: {unresolved}'
    stale = [
        f'{e["name"]} (verified against {e["verified_annnet"]})'
        for e in packages()
        if e['verified_annnet'] != annnet.__version__
    ]
    assert not stale, (
        f'these were verified against another AnnNet than {annnet.__version__}: {stale}. '
        f'Run their tests against this version and record it.'
    )


@pytest.mark.skipif(
    gate() != 'public',
    reason='the public-release gate runs with ANNNET_RELEASE_GATE=public',
)
def test_public_release_gate_every_dependent_has_merged_its_migration():
    """The dependents' repositories carry the change, so a user of both keeps working.

    A migration that passes only as a local patch means a user who upgrades
    AnnNet and keeps the released version of a dependent meets the break the
    register exists to prevent. This gate fails until each package's default
    branch has merged its migration.
    """
    waiting = [
        f'{e["name"]} ({e["migration_status"]})'
        for e in packages()
        if e['migration_status'] != 'merged_upstream'
    ]
    assert not waiting, f'dependents whose repository has not merged the migration: {waiting}'


def test_every_namespace_the_register_uses_exists(graph):
    """A register entry that starts from a namespace we dropped is a stale entry."""
    used = {name.split('.')[0] for entry in packages() for name in entry['calls'] if '.' in name}
    used |= {
        '.'.join(name.split('.')[:1])
        for entry in packages()
        for name in entry['previous_calls']
        if '.' in name and name.split('.')[0] != 'views'
    }
    unknown = sorted(used - set(NAMESPACES))
    assert not unknown, f'the register reaches through namespaces that are not listed: {unknown}'


def test_the_structural_keys_a_dependent_writes_still_mean_what_they_did(graph):
    """A key name is as much a contract as a method name.

    A bridge builds a graph from a table, so it writes ``node_id`` into a node
    spec and ``source``, ``target`` and ``weight`` into an edge spec. Those are
    the names the mutation gateway reads as structure, and renaming one moves a
    caller's value into an ordinary attribute without raising.
    """
    keys = register()['keys']

    missing_node = sorted(set(keys['node']) - set(_node_RESERVED))
    assert not missing_node, (
        f'a node key a dependent writes is no longer structural: {missing_node}'
    )

    missing_edge = sorted(set(keys['edge']) - set(_EDGE_RESERVED))
    assert not missing_edge, (
        f'an edge key a dependent writes is no longer structural: {missing_edge}'
    )


def test_the_edge_view_a_dependent_reads_still_carries_its_fields(graph):
    """``E.at`` hands back a record, and a bridge reads it by name."""
    view = graph.E.at('e')
    for field in ('edge_id', 'kind', 'source', 'target', 'weight', 'directed'):
        assert hasattr(view, field), f'EdgeView no longer carries {field!r}'
    source, target = view
    assert source and target, 'unpacking an edge into two sides no longer works'
