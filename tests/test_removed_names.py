"""A caller holding a removed name meets the name that replaced it.

The package removes a public name without a deprecation and without an alias,
so the error the old name raises is the whole of the migration help a caller
gets at the moment they need it. These tests hold every removed name to that:
it raises, the message says what to write instead, and ``dev/migration.md``
carries the same replacement for a reader who wants the full map.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import annnet
from annnet import AnnNet
from annnet.core._removed import GRAPH, NAMESPACES

MIGRATION = Path(annnet.__file__).parent.parent / 'dev' / 'migration.md'


@pytest.fixture
def graph() -> AnnNet:
    anet = AnnNet(directed=True)
    anet.add_nodes(['A', 'B'])
    anet.add_edges('A', 'B', edge_id='e0')
    return anet


@pytest.fixture
def holders(graph) -> dict:
    """One object per holder the record names."""
    return {
        'G': graph,
        'G.attrs': graph.attrs,
        'G.layers': graph.layers,
        'G.slices': graph.slices,
        'G.cache': graph.cache,
        'V': graph.view(),
    }


def _names() -> list[tuple[str, str, str]]:
    """Every removed namespace name, as holder, name and replacement."""
    return [
        (holder, name, replacement)
        for holder, names in NAMESPACES.items()
        for name, replacement in names.items()
    ]


def test_the_record_names_a_holder_the_graph_carries(holders):
    """Every holder of the record is reachable, so no entry is dead."""
    assert set(NAMESPACES) <= set(holders)


@pytest.mark.parametrize(('holder', 'name', 'replacement'), _names(), ids=lambda part: str(part))
def test_a_removed_namespace_name_names_its_replacement(holders, holder, name, replacement):
    """The error says the name went, and what to write instead."""
    with pytest.raises(AttributeError, match='was removed') as caught:
        getattr(holders[holder], name)

    message = str(caught.value)
    assert f'{holder}.{name}' in message
    assert replacement in message


@pytest.mark.parametrize(('holder', 'name', 'replacement'), _names(), ids=lambda part: str(part))
def test_a_removed_namespace_name_is_not_reachable(holders, holder, name, replacement):
    """``hasattr`` answers False, so a caller can ask before they call."""
    assert not hasattr(holders[holder], name)


def test_a_removed_graph_name_names_its_replacement(graph):
    """The graph names were already held to this rule. They stay."""
    for name, replacement in GRAPH.items():
        with pytest.raises(AttributeError, match='was removed') as caught:
            getattr(graph, name)
        assert replacement.split('(')[0].split(',')[0] in str(caught.value)


@pytest.mark.skipif(not MIGRATION.is_file(), reason='dev/migration.md is not beside the package')
def test_every_removed_name_is_on_the_migration_page_with_its_holder():
    """The record and the page say the same, so neither drifts alone."""
    migration = MIGRATION.read_text()
    missing = [
        f'{holder}.{name}'
        for holder, name, _replacement in _names()
        if re.search(rf'\b{name}\b', migration) is None
    ]
    assert missing == []
