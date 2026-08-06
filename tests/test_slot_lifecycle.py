"""Slot allocate, free, and reuse tests for the slot-addressed store.

A slot is a stable integer address. The store assigns it on insert and frees it on
delete, and it never renumbers one. That is the property the whole refactor rests
on, so these tests pin it down.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet.core import _store as ST

FLAT = ('_',)


def key(node_id):
    return (node_id, FLAT)


@pytest.fixture
def store():
    return ST.CoreState(directed=True)


def edge(store, edge_id, source, target):
    return store.add_edge(
        edge_id,
        [(key(source), 1.0, ST.SOURCE), (key(target), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.0,
    )


# ---------------------------------------------------------------------------
# Allocate
# ---------------------------------------------------------------------------


def test_the_first_slots_start_at_zero_and_rise_by_one(store):
    assert [store.add_entity(key(n)) for n in 'ABC'] == [0, 1, 2]


def test_a_slot_maps_back_to_its_identity(store):
    slot = store.add_entity(key('A'))
    assert store.entity_key(slot) == key('A')
    assert store.entity_slot(key('A')) == slot


def test_adding_the_same_identity_twice_returns_the_same_slot(store):
    first = store.add_entity(key('A'))
    assert store.add_entity(key('A')) == first
    assert store.entity_count == 1


def test_an_edge_gets_its_own_slot_space(store):
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    assert edge(store, 'e0', 'A', 'B') == 0
    assert edge(store, 'e1', 'A', 'B') == 1


# ---------------------------------------------------------------------------
# Free
# ---------------------------------------------------------------------------


def test_deleting_an_entity_frees_its_slot(store):
    for n in 'ABC':
        store.add_entity(key(n))
    store.remove_entity(key('B'))
    assert store.entity_slot(key('B')) is None
    assert store.entity_key(1) is None
    assert 1 in store.entity_free


def test_deleting_an_entity_renumbers_no_other_entity(store):
    slots = {n: store.add_entity(key(n)) for n in 'ABC'}
    store.remove_entity(key('B'))
    assert store.entity_slot(key('A')) == slots['A']
    assert store.entity_slot(key('C')) == slots['C']


def test_deleting_an_edge_renumbers_no_other_edge(store):
    for n in 'ABC':
        store.add_entity(key(n))
    slots = {eid: edge(store, eid, 'A', 'B') for eid in ('e0', 'e1', 'e2')}
    store.remove_edge('e1')
    assert store.edge_slot('e0') == slots['e0']
    assert store.edge_slot('e2') == slots['e2']
    assert store.edge_slot('e1') is None


def test_deleting_an_edge_clears_its_member_list(store):
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    slot = edge(store, 'e0', 'A', 'B')
    store.remove_edge('e0')
    assert store.members(slot).entities.size == 0


def test_deleting_an_unknown_element_raises(store):
    with pytest.raises(KeyError):
        store.remove_entity(key('nope'))
    with pytest.raises(KeyError):
        store.remove_edge('nope')


# ---------------------------------------------------------------------------
# Reuse
# ---------------------------------------------------------------------------


def test_a_freed_entity_slot_is_reused_by_the_next_insert(store):
    for n in 'ABC':
        store.add_entity(key(n))
    freed = store.entity_slot(key('B'))
    store.remove_entity(key('B'))
    assert store.add_entity(key('D')) == freed


def test_a_freed_edge_slot_is_reused_by_the_next_insert(store):
    store.add_entity(key('A'))
    store.add_entity(key('B'))
    for eid in ('e0', 'e1'):
        edge(store, eid, 'A', 'B')
    freed = store.edge_slot('e0')
    store.remove_edge('e0')
    assert edge(store, 'e2', 'A', 'B') == freed


def test_a_reused_slot_carries_the_new_identity_only(store):
    store.add_entity(key('A'))
    slot = store.add_entity(key('B'))
    store.remove_entity(key('B'))
    store.add_entity(key('C'))
    assert store.entity_key(slot) == key('C')
    assert store.entity_slot(key('B')) is None


def test_a_stale_slot_reference_does_not_resolve_to_the_reused_element(store):
    """The rule that makes a slot safe to hold inside the core."""
    stale = store.add_entity(key('B'))
    store.remove_entity(key('B'))
    store.add_entity(key('C'))
    # The slot now belongs to C. Asking the store for B must not answer with the
    # slot, and asking for the identity of the slot must not answer with B.
    assert store.entity_slot(key('B')) is None
    assert store.entity_key(stale) != key('B')


def test_a_reused_edge_slot_carries_the_new_member_list(store):
    for n in 'ABC':
        store.add_entity(key(n))
    slot = edge(store, 'e0', 'A', 'B')
    store.remove_edge('e0')
    edge(store, 'e1', 'B', 'C')
    members = store.members(slot)
    assert set(members.entities) == {store.entity_slot(key('B')), store.entity_slot(key('C'))}


# ---------------------------------------------------------------------------
# The clock
# ---------------------------------------------------------------------------


def test_every_write_advances_the_clock(store):
    seen = [store.structure_version]
    store.add_entity(key('A'))
    seen.append(store.structure_version)
    store.add_entity(key('B'))
    seen.append(store.structure_version)
    edge(store, 'e0', 'A', 'B')
    seen.append(store.structure_version)
    store.remove_edge('e0')
    seen.append(store.structure_version)
    store.remove_entity(key('A'))
    seen.append(store.structure_version)
    assert seen == sorted(set(seen)), 'the clock must rise on every write'


def test_a_read_leaves_the_clock_alone(store):
    store.add_entity(key('A'))
    before = store.structure_version
    store.entity_slot(key('A'))
    store.entity_key(0)
    assert store.structure_version == before


# ---------------------------------------------------------------------------
# Capacity
# ---------------------------------------------------------------------------


def test_the_arrays_grow_when_the_freelist_is_empty(store):
    for i in range(300):
        store.add_entity(key(f'v{i}'))
    assert store.entity_count == 300
    assert store.entity_key(299) == key('v299')


def test_a_long_churn_keeps_identity_and_address_in_step(store):
    for i in range(50):
        store.add_entity(key(f'v{i}'))
    for i in range(0, 50, 2):
        store.remove_entity(key(f'v{i}'))
    for i in range(50, 75):
        store.add_entity(key(f'v{i}'))
    for slot, entity_key in store.live_entities():
        assert store.entity_slot(entity_key) == slot
        assert store.entity_key(slot) == entity_key


# ---------------------------------------------------------------------------
# Rekey
# ---------------------------------------------------------------------------
# An identity may change while an address does not. That is what lets a graph
# declare aspects over the nodes it already holds without touching a member
# list or a matrix position.


def test_rekeying_an_entity_keeps_its_slot_and_its_edges(store):
    for n in 'AB':
        store.add_entity(key(n))
    slot = store.entity_slot(key('A'))
    edge(store, 'e0', 'A', 'B')
    store.rekey({key('A'): ('A', ('_', '_'))})
    assert store.entity_slot(('A', ('_', '_'))) == slot
    assert store.entity_slot(key('A')) is None
    assert store.entity_key(slot) == ('A', ('_', '_'))
    assert store.entity_edge_slots(('A', ('_', '_'))) == [store.edge_slot('e0')]


def test_rekeying_leaves_the_member_lists_alone(store):
    for n in 'AB':
        store.add_entity(key(n))
    slot = edge(store, 'e0', 'A', 'B')
    before = store.members(slot).entities.copy()
    store.rekey({key('A'): ('A', ('x',)), key('B'): ('B', ('x',))})
    assert np.array_equal(store.members(slot).entities, before)


def test_rekeying_may_permute_the_keys_of_two_entities(store):
    for n in 'AB':
        store.add_entity(key(n))
    slots = (store.entity_slot(key('A')), store.entity_slot(key('B')))
    store.rekey({key('A'): key('B'), key('B'): key('A')})
    assert (store.entity_slot(key('B')), store.entity_slot(key('A'))) == slots


def test_rekeying_onto_an_entity_the_store_holds_raises(store):
    for n in 'AB':
        store.add_entity(key(n))
    with pytest.raises(KeyError, match='Rekeying'):
        store.rekey({key('A'): key('B')})


def test_rekeying_names_a_bare_id_at_its_new_coordinate(store):
    store._aspects = ('cond',)
    store.add_entity(('A', ('ctrl',)))
    store.rekey({('A', ('ctrl',)): ('A', ('treated',))})
    assert store.entity_keys_of_id('A') == [('A', ('treated',))]


# ---------------------------------------------------------------------------
# Copy
# ---------------------------------------------------------------------------


def _churned(store):
    """A store with holes in both freelists and a policy on one edge."""
    for n in 'ABCD':
        store.add_entity(key(n))
    edge(store, 'e0', 'A', 'B')
    edge(store, 'e1', 'B', 'C')
    edge(store, 'e2', 'C', 'D')
    store.remove_edge('e1')
    store.remove_entity(key('D'))
    store.edge_policy[store.edge_slot('e0')] = {'mode': 'flexible'}
    store.edge_ml_kind[store.edge_slot('e2')] = 'inter'
    return store


def test_a_copy_holds_the_same_graph_at_the_same_addresses(store):
    _churned(store)
    other = store.copy()
    assert list(other.live_entities()) == list(store.live_entities())
    assert list(other.live_edges()) == list(store.live_edges())
    for slot, _edge_id in store.live_edges():
        mine, theirs = store.members(slot), other.members(slot)
        assert np.array_equal(theirs.entities, mine.entities)
        assert np.allclose(theirs.coefficients, mine.coefficients)
        assert np.array_equal(theirs.roles, mine.roles)
    assert other.entity_free == store.entity_free
    assert other.edge_free == store.edge_free
    assert other.structure_version == store.structure_version


def test_a_write_to_a_copy_leaves_the_original_alone(store):
    _churned(store)
    other = store.copy()
    other.add_entity(key('E'))
    other.remove_edge('e0')
    other.edge_policy[other.edge_slot('e2')] = {'mode': 'fixed'}
    assert store.entity_slot(key('E')) is None
    assert store.edge_slot('e0') is not None
    assert store.edge_policy[store.edge_slot('e0')] == {'mode': 'flexible'}
    assert store.edge_policy.get(store.edge_slot('e2')) is None


def test_a_copy_carries_no_hook_of_the_store_it_came_from(store):
    _churned(store)
    freed = []
    store.edge_freed_hooks.append(lambda slot, edge_id: freed.append(edge_id))
    other = store.copy()
    other.remove_edge('e0')
    assert freed == [], 'a hook belongs to one graph, not to every copy of its store'


def test_the_named_array_list_covers_every_array_a_store_owns(store):
    _churned(store)
    owned = {
        name
        for name, value in vars(store).items()
        if isinstance(value, np.ndarray) and not name.startswith('__')
    }
    assert owned == set(ST._ARRAYS), 'a copy would leave a new array behind'
