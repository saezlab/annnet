"""The bulk edge write must give the store the single writes give.

:meth:`CoreState.add_edges` writes a batch of edges in one pass, growing each
array once and assigning each member pool once. It is a second way to write the
same thing, so what these tests pin is that it is not a different thing: for
every shape of edge, the store it leaves is the store the same edges added one
at a time leave, down to the incidence index and the append log.

Two shapes take a path of their own inside it. A batch of two entries per edge
is walked as pairs, because that is what a bulk load is made of; a batch too
small to pay for the vectorized write goes through the single write instead. The
cases below cross both with every edge shape.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet.core import _store as ST

FLAT = ('_',)
KEYS = [(f'n{i}', FLAT) for i in range(6)]

# What a store holds outside its arrays, and what a comparison has to cover.
_MAPS = (
    '_entity_slot',
    '_entity_key',
    '_edge_slot',
    '_edge_id',
    '_entity_edges',
    'edge_ml_kind',
    'edge_ml_layers',
    'edge_policy',
    'edge_free',
    '_member_used',
    'structure_version',
    'append_log',
    'append_log_from_version',
)


def seeded():
    """A store holding the six entities every case below draws its members from."""
    store = ST.CoreState(directed=True)
    for key in KEYS:
        store.add_entity(key, ST.NODE)
    return store


def singly(specs, store=None):
    """Write the specs one at a time, the way the store wrote them before."""
    store = seeded() if store is None else store
    for spec in specs:
        spec = ST.EdgeSpec(*spec)
        store.add_edge(
            spec.id,
            spec.members,
            kind=spec.kind,
            directed=spec.directed,
            weight=spec.weight,
            explicit_coefficients=spec.explicit_coefficients,
            ml_kind=spec.ml_kind,
            ml_layers=spec.ml_layers,
            direction_policy=spec.direction_policy,
        )
    return store


def in_bulk(specs, store=None):
    """Write the specs in one pass."""
    store = seeded() if store is None else store
    store.add_edges([ST.EdgeSpec(*spec) for spec in specs])
    return store


def assert_same_store(one, many):
    """Assert the two stores are the same store, array by array and map by map."""
    for name in ST._ARRAYS:
        left, right = getattr(one, name), getattr(many, name)
        width = min(left.size, right.size)
        assert np.array_equal(left[:width], right[:width]), name
    for name in _MAPS:
        assert getattr(one, name) == getattr(many, name), name


def binary(edge_id, source, target, weight=1.0):
    return (
        edge_id,
        ((KEYS[source], weight, ST.SOURCE), (KEYS[target], -weight, ST.TARGET)),
    )


# One entry per shape an edge can take, at a size that reaches the vectorized
# path, and again at a size that does not.
SHAPES = {
    'binary edges': [binary(f'e{i}', i % 6, (i + 1) % 6) for i in range(12)],
    'self loops': [binary(f'e{i}', i % 6, i % 6) for i in range(12)],
    'boundary edges': [(f'e{i}', ((KEYS[i % 6], 1.0, ST.MEMBER),)) for i in range(12)],
    'hyperedges': [
        (
            f'e{i}',
            tuple((KEYS[(i + j) % 6], 1.0, ST.SOURCE if j < 2 else ST.TARGET) for j in range(4)),
        )
        for i in range(12)
    ],
    'placeholders': [(f'e{i}', (), ST.PLACEHOLDER, False, 1.0) for i in range(12)],
    'one entity in three roles': [
        (
            f'e{i}',
            (
                (KEYS[i % 6], 1.0, ST.SOURCE),
                (KEYS[i % 6], 1.0, ST.SOURCE),
                (KEYS[i % 6], -1.0, ST.TARGET),
            ),
        )
        for i in range(12)
    ],
    'widths mixed within one batch': [
        binary('a', 0, 1),
        ('b', ((KEYS[1], 1.0, ST.MEMBER),)),
        ('c', tuple((KEYS[j], 1.0, ST.SOURCE) for j in range(5))),
        binary('d', 2, 2),
        ('e', ()),
        binary('f', 3, 4),
        binary('g', 4, 5),
        binary('h', 5, 0),
        binary('i', 0, 2),
        binary('j', 1, 3),
        binary('k', 2, 4),
        binary('l', 3, 5),
    ],
    'the rare per-edge state': [
        ('a', ((KEYS[0], 1.0, ST.SOURCE), (KEYS[1], -1.0, ST.TARGET)), ST.BINARY, False, 2.5, True),
        ('b', ((KEYS[1], 1.0, ST.SOURCE),), ST.HYPER, None, 3.0, False, 'inter', ('x', 'y')),
        (
            'c',
            ((KEYS[2], 1.0, ST.SOURCE), (KEYS[3], -1.0, ST.TARGET)),
            ST.BINARY,
            True,
            1.0,
            False,
            None,
            None,
            {'mode': 'flexible'},
        ),
    ]
    + [binary(f'e{i}', i % 6, (i + 1) % 6) for i in range(9)],
}


@pytest.mark.parametrize('shape', list(SHAPES))
def test_a_bulk_write_gives_the_store_the_single_writes_give(shape):
    specs = SHAPES[shape]
    assert_same_store(singly(specs), in_bulk(specs))


@pytest.mark.parametrize('shape', list(SHAPES))
def test_a_batch_below_the_bulk_minimum_gives_the_same_store_too(shape):
    specs = SHAPES[shape][: ST._BULK_MINIMUM - 1]
    assert_same_store(singly(specs), in_bulk(specs))


def test_a_bulk_write_reuses_a_freed_slot_before_it_grows_the_frontier():
    """The batch after a removal fills the holes it left, as single adds do."""
    specs = [binary(f'q{i}', i % 6, (i + 1) % 6) for i in range(10)]

    def prepared():
        store = seeded()
        for i in range(4):
            store.add_edge(f'p{i}', binary(f'p{i}', i, i + 1)[1])
        store.remove_edge('p1')
        store.remove_edge('p2')
        return store

    assert_same_store(singly(specs, prepared()), in_bulk(specs, prepared()))


def test_a_bulk_write_leaves_the_append_log_a_run_of_frontier_appends():
    """A cached matrix extends over a bulk write, because the log accounts for it."""
    store = in_bulk([binary(f'e{i}', i % 6, (i + 1) % 6) for i in range(10)])
    assert store.append_log == list(range(10))
    assert len(store.append_log) == store.structure_version - store.append_log_from_version


def test_a_bulk_write_after_a_removal_drops_the_reused_head_from_the_log():
    store = seeded()
    store.add_edges([ST.EdgeSpec(*binary(f'p{i}', i % 6, (i + 1) % 6)) for i in range(10)])
    store.remove_edge('p3')
    store.add_edges([ST.EdgeSpec(*binary(f'q{i}', i % 6, (i + 1) % 6)) for i in range(10)])
    # The first of the ten took the freed slot and the other nine went to the
    # frontier, so the log holds those nine and nothing before them.
    assert store.append_log == list(range(10, 19))
    assert len(store.append_log) == store.structure_version - store.append_log_from_version


def test_an_empty_batch_writes_nothing():
    store = seeded()
    before = store.structure_version
    assert store.add_edges([]) == []
    assert store.structure_version == before


def test_a_bulk_write_returns_the_slot_of_every_edge_in_order():
    store = seeded()
    specs = [ST.EdgeSpec(*binary(f'e{i}', i % 6, (i + 1) % 6)) for i in range(10)]
    slots = store.add_edges(specs)
    assert slots == [store.edge_slot(spec.id) for spec in specs]


# ---------------------------------------------------------------------------
# A batch that cannot be written must leave the store as it was
# ---------------------------------------------------------------------------

BAD = {
    'an entity the store does not hold': (
        [ST.EdgeSpec('x', ((('ghost', FLAT), 1.0, ST.SOURCE),))],
        'names an entity the store does not hold',
    ),
    'an id the store already holds': ([ST.EdgeSpec('kept', ())], 'Duplicate edge id'),
    'the same id twice in one batch': (
        [ST.EdgeSpec('y', ()), ST.EdgeSpec('y', ())],
        'Duplicate edge id',
    ),
}


def _snapshot(store):
    return (
        store.structure_version,
        dict(store._edge_slot),
        list(store._edge_id),
        store._member_used,
        {slot: dict(edges) for slot, edges in store._entity_edges.items()},
    )


@pytest.mark.parametrize('padding', [0, ST._BULK_MINIMUM], ids=['below the minimum', 'in bulk'])
@pytest.mark.parametrize('case', list(BAD))
def test_a_batch_that_cannot_be_written_writes_none_of_itself(case, padding):
    specs, message = BAD[case]
    store = seeded()
    store.add_edge('kept', binary('kept', 0, 1)[1])
    before = _snapshot(store)
    padded = [ST.EdgeSpec(*binary(f'pad{i}', i % 6, (i + 1) % 6)) for i in range(padding)] + specs
    with pytest.raises(KeyError, match=message):
        store.add_edges(padded)
    assert _snapshot(store) == before


def test_the_raise_names_the_edge_and_the_entry_that_caused_it():
    store = seeded()
    specs = [ST.EdgeSpec(*binary(f'e{i}', i % 6, (i + 1) % 6)) for i in range(10)]
    specs[7] = ST.EdgeSpec(
        'broken', ((KEYS[0], 1.0, ST.SOURCE), (('ghost', FLAT), -1.0, ST.TARGET))
    )
    with pytest.raises(KeyError, match="Edge 'broken' names an entity"):
        store.add_edges(specs)


def test_the_raise_names_the_id_the_batch_repeats():
    store = seeded()
    specs = [ST.EdgeSpec(*binary(f'e{i}', i % 6, (i + 1) % 6)) for i in range(10)]
    specs[8] = ST.EdgeSpec(*binary('e2', 0, 1))
    with pytest.raises(KeyError, match="Duplicate edge id: 'e2'"):
        store.add_edges(specs)
