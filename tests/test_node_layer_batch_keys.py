"""A batch of node-layer updates names each placement once, however it spells it.

The mapping form cannot repeat a key, so it never had to say what a repeat means.
The iterable form can, and a repeat there used to keep the last row and drop the
others without a word. Whatever spelling the repeated placement takes (a full
key, a bare node id with ``layer=``, a layer given as a label or a tuple), the
batch is refused before anything is written.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet


@pytest.fixture
def graph():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(aspects={'condition': ['ctrl', 'stim']})
        G.add_nodes(['A', 'B'], layer=('ctrl',))
        G.add_nodes(['A'], layer=('stim',))
    return G


def _untouched(graph):
    return graph.attrs.rows('node_layers') == {
        ('A', ('ctrl',)): {},
        ('B', ('ctrl',)): {},
        ('A', ('stim',)): {},
    }


def test_a_repeated_full_key_in_an_iterable_is_refused(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update(
            'node_layers', [(('A', ('ctrl',)), {'x': 1}), (('A', ('ctrl',)), {'x': 2})]
        )
    assert _untouched(graph)


def test_two_spellings_of_one_placement_are_refused(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update('node_layers', [(('A', 'ctrl'), {'x': 1}), (('A', ('ctrl',)), {'x': 2})])
    assert _untouched(graph)


def test_a_bare_id_and_the_full_key_it_stands_for_are_refused(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update(
            'node_layers',
            [('A', {'x': 1}), (('A', ('ctrl',)), {'x': 2})],
            layer=('ctrl',),
        )
    assert _untouched(graph)


def test_a_repeated_bare_id_with_a_shorthand_layer_is_refused(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update('node_layers', [('A', 1), ('A', 2)], layer=('ctrl',), key='x')
    assert _untouched(graph)


def test_a_duplicate_after_valid_rows_writes_none_of_them(graph):
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update(
            'node_layers',
            [
                (('B', ('ctrl',)), {'x': 0}),
                (('A', ('ctrl',)), {'x': 1}),
                (('A', ('ctrl',)), {'x': 2}),
            ],
        )
    assert _untouched(graph)
    assert graph.attrs.schema('node_layers').fields == ()


def test_the_same_node_on_different_layers_is_not_a_duplicate(graph):
    written = graph.attrs.update(
        'node_layers',
        [(('A', ('ctrl',)), {'x': 1}), (('A', ('stim',)), {'x': 2})],
    )
    assert written == 2
    assert graph.attrs.row('node_layers', ('A', ('ctrl',))) == {'x': 1}
    assert graph.attrs.row('node_layers', ('A', ('stim',))) == {'x': 2}


def test_the_mapping_form_is_unchanged(graph):
    assert graph.attrs.update('node_layers', {'A': 1, 'B': 2}, layer=('ctrl',), key='x') == 2
    assert graph.attrs.row('node_layers', ('B', ('ctrl',))) == {'x': 2}


@pytest.mark.parametrize(
    'address,rows',
    [
        ('nodes', [('A', {'x': 1}), ('A', {'x': 2})]),
        ('layers', [(('ctrl',), {'x': 1}), ('ctrl', {'x': 2})]),
        ('slices', [('default', {'x': 1}), ('default', {'x': 2})]),
    ],
)
def test_every_address_refuses_a_repeated_key_in_an_iterable(graph, address, rows):
    before = graph.attrs.rows(address)
    with pytest.raises(ValueError, match='duplicate key'):
        graph.attrs.update(address, rows)
    assert graph.attrs.rows(address) == before
