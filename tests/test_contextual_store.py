"""Contextual attributes are dicts; a table is what a reader gets.

Three of the six contextual levels used to be stored as backend dataframes on the
graph. That made the type of canonical state depend on which table library was
installed, and it made every single attribute write filter and rebuild the whole
frame — so the per-write cost grew with the table and the total was quadratic.

These tests pin the properties that replaced it: one store, dicts inside, tables
built on demand in whatever backend the caller names, and a write that costs the
same whether the graph holds ten pairs or ten thousand.
"""

from __future__ import annotations

import time

import pytest

from annnet import AnnNet
from annnet.core._contextual import LEVELS, ContextualStore


def _graph_with_edges(count):
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    G.add_edges([{'source': 'a', 'target': 'b', 'edge_id': f'e{i}'} for i in range(count)])
    G.slices.add('s')
    return G


# ---------------------------------------------------------------------------
# Canonical state
# ---------------------------------------------------------------------------


def test_no_contextual_level_is_stored_as_a_dataframe():
    """The whole point: a graph's own state must not depend on a table library."""
    G = _graph_with_edges(2)
    G.attrs.set_slice_attrs('s', kind='curated')
    G.attrs.set_edge_slice_attrs('s', 'e0', confidence=0.9)
    G.layers.set_aspects(['phase'], {'phase': ['t0']})
    G.layers.set_elementary_attrs('phase', 't0', colour='blue')
    for level in LEVELS:
        held = getattr(G._contextual, level)
        assert isinstance(held, dict), f'{level} is a {type(held).__name__}, not a dict'


def test_every_level_lives_in_one_store():
    G = AnnNet(directed=True)
    assert {*LEVELS} == {name for name in ContextualStore.__slots__ if name != 'version'}
    assert isinstance(G._contextual, ContextualStore)


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('backend', ['polars', 'pandas', 'pyarrow'])
def test_a_table_is_rendered_in_the_backend_the_caller_names(backend):
    """The backend belongs to the read, not to the graph."""
    G = _graph_with_edges(3)
    G.attrs.set_edge_slice_attrs('s', 'e0', confidence=0.9)
    table = G.contextual_table('edge_slice_attrs', backend=backend)
    assert type(table).__module__.split('.')[0] == backend


def test_the_table_round_trips_through_the_property():
    G = _graph_with_edges(2)
    G.attrs.set_slice_attrs('s', kind='curated')
    assert G.slice_attributes.to_dicts() == [{'slice_id': 's', 'kind': 'curated'}]

    other = AnnNet(directed=True)
    other.slice_attributes = G.slice_attributes
    assert other._contextual.slice_attrs == {'s': {'kind': 'curated'}}


def test_a_second_read_is_the_cached_table_and_a_write_rebuilds_it():
    G = _graph_with_edges(4)
    G.attrs.set_edge_slice_attrs('s', 'e0', confidence=0.1)
    first = G.edge_slice_attributes
    assert G.edge_slice_attributes is first

    # An update in place changes no count, so the store carries a version.
    G.attrs.set_edge_slice_attrs('s', 'e0', confidence=0.9)
    rebuilt = G.edge_slice_attributes
    assert rebuilt is not first
    assert {row['edge_id']: row['confidence'] for row in rebuilt.to_dicts()}['e0'] == 0.9


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_a_contextual_write_does_not_cost_the_size_of_the_table():
    """The defect this replaced was quadratic: per-write cost grew with the table.

    The check is a ratio rather than a wall-clock bound, so it says the same
    thing on a slow machine. A linear-per-write implementation shows a ratio near
    the size ratio (8x here); a constant one stays near 1.
    """
    small, large = 250, 2000
    timings = {}
    for count in (small, large):
        G = _graph_with_edges(count)
        start = time.perf_counter()
        for i in range(count):
            G.attrs.set_edge_slice_attrs('s', f'e{i}', conf=0.5)
        timings[count] = (time.perf_counter() - start) / count

    ratio = timings[large] / timings[small]
    assert ratio < 3.0, (
        f'per-write cost grew {ratio:.1f}x when the table grew '
        f'{large // small}x — the write is not constant-time'
    )


def test_the_store_forgets_what_an_element_carried():
    G = _graph_with_edges(2)
    G.attrs.set_edge_slice_attrs('s', 'e0', confidence=0.9)
    assert ('s', 'e0') in G._contextual.edge_slice_attrs
    G.remove_edge('e0')
    assert G.edge_slice_attributes is not None
    assert ('s', 'e0') not in G._contextual.edge_slice_attrs


def test_copy_shares_no_dict_with_the_original():
    store = ContextualStore()
    store.set('slice_attrs', 's', {'kind': 'curated'})
    clone = store.copy()
    clone.set('slice_attrs', 's', {'kind': 'changed'})
    assert store.slice_attrs['s'] == {'kind': 'curated'}
    assert clone.slice_attrs['s'] == {'kind': 'changed'}
