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

import tomllib
from pathlib import Path

import pytest

import annnet
from annnet import AnnNet
from annnet.core._records import _EDGE_RESERVED, _node_RESERVED

REGISTER = Path(annnet.__file__).parent.parent / 'dependents.toml'

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
    'views',
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
        for field in ('name', 'owner', 'repository', 'contact', 'modules', 'calls'):
            assert entry.get(field), f'{entry.get("name", entry)!r} is missing {field!r}'


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


def test_every_namespace_the_register_uses_exists(graph):
    """A register entry that starts from a namespace we dropped is a stale entry."""
    used = {name.split('.')[0] for entry in packages() for name in entry['calls'] if '.' in name}
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
    """``get_edge`` hands back a record, and a bridge reads it by name."""
    view = graph.get_edge('e')
    for field in ('edge_id', 'kind', 'source', 'target', 'weight', 'directed'):
        assert hasattr(view, field), f'EdgeView no longer carries {field!r}'
    source, target = view
    assert source and target, 'unpacking an edge into two sides no longer works'
