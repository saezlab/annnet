"""A failed write puts back the schema as well as the values.

An attribute batch can add a column, widen the type of one, or (through a delete)
remove one. If it fails, the graph has to answer as it did before: the same
fields in the same order with the same types, the topology and indexes the batch
had touched, and caches that describe the restored graph. The batch used to put
the values back and leave the column it introduced behind, so the schema said a
field existed that no row carried.
"""

from __future__ import annotations

import pandas as pd
import pytest

from annnet import AnnNet
from annnet.core import _store as ST, _attribute_api


def _boom(self, edge_id):
    raise RuntimeError('policy failed')


@pytest.fixture
def graph():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.attrs.update(
        'nodes',
        {
            'a': {'count': 1, 'score': 0.5, 'label': 'x', 'x': 0.9},
            'b': {'count': 2, 'score': 1.5, 'x': 0.1},
            'c': {'count': 3},
        },
    )
    G.add_edges('a', 'b', edge_id='ab', flexible={'var': 'x', 'threshold': 0.5, 'scope': 'node'})
    G.add_edges('b', 'c', edge_id='bc')
    G.attrs.update('edges', {'ab': {'kind_note': 'first'}, 'bc': {'weight_note': 2.0}})
    return G


def _state(G):
    """Everything observable about the attributes and the topology."""
    return {
        'node_schema': G.attrs.schema('nodes'),
        'edge_schema': G.attrs.schema('edges'),
        'node_columns': list(G.attrs.table('nodes', backend='pandas').columns),
        'edge_columns': list(G.attrs.table('edges', backend='pandas').columns),
        'nodes': G.attrs.rows('nodes'),
        'edges': G.attrs.rows('edges'),
        'ends': {e: (sorted(G.E.at(e).source), sorted(G.E.at(e).target)) for e in G.E},
        'key_index': dict(G._node_key_index),
    }


def test_a_column_a_failed_row_batch_introduced_does_not_survive(graph, monkeypatch):
    before = _state(graph)
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    with pytest.raises(RuntimeError, match='policy failed'):
        graph.attrs.update('nodes', {'a': {'x': 0.1, 'brand_new': 1}, 'b': {'x': 0.9, 'other': 2}})
    assert _state(graph) == before
    assert 'brand_new' not in graph.attrs.schema('nodes').names()
    assert 'other' not in list(graph.attrs.table('nodes', backend='pandas').columns)


def test_a_column_an_edge_batch_introduced_does_not_survive(graph, monkeypatch):
    graph.add_edges(
        'a', 'c', edge_id='ac', flexible={'var': 'q', 'threshold': 0.5, 'scope': 'edge'}
    )
    before = _state(graph)
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    with pytest.raises(RuntimeError, match='policy failed'):
        graph.attrs.update('edges', {'ac': {'q': 0.9, 'brand_new': 1}})
    assert _state(graph) == before
    assert 'brand_new' not in graph.attrs.schema('edges').names()


def test_a_failed_batch_restores_the_order_and_type_of_existing_columns(graph, monkeypatch):
    before = _state(graph)
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    with pytest.raises(RuntimeError):
        # A string into a numeric column widens it before the policy fails.
        graph.attrs.update('nodes', {'a': {'score': 'not a number', 'x': 0.1}})
    assert _state(graph) == before
    assert dict(graph.attrs.schema('nodes').fields)['score'] == 'float'
    assert dict(graph.attrs.schema('nodes').fields)['count'] == 'int'


def test_a_failed_table_replacement_restores_columns_values_and_topology(graph, monkeypatch):
    before = _state(graph)
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    frame = pd.DataFrame({'node_id': ['a', 'b'], 'x': [0.1, 0.9], 'added': ['p', 'q']})
    with pytest.raises(RuntimeError, match='policy failed'):
        graph.attrs.replace('nodes', frame)
    assert _state(graph) == before
    assert 'added' not in graph.attrs.schema('nodes').names()
    assert graph.attrs.schema('nodes').names() == ['node_id', 'count', 'score', 'label', 'x']


def test_a_failed_delete_restores_the_dropped_column_in_place(graph, monkeypatch):
    before = _state(graph)
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    with pytest.raises(RuntimeError):
        graph.attrs.delete('nodes', names=['x'])
    assert _state(graph) == before
    assert graph.attrs.schema('nodes').names() == ['node_id', 'count', 'score', 'label', 'x']


def test_a_batch_that_fails_validation_never_touches_the_schema(graph):
    before = _state(graph)
    with pytest.raises(KeyError):
        graph.attrs.update('nodes', {'a': {'fresh': 1}, 'ghost': {'fresh': 2}})
    with pytest.raises(ValueError, match='reserved'):
        graph.attrs.update('nodes', {'a': {'fresh': 1, 'node_id': 'z'}})
    assert _state(graph) == before


def test_a_null_write_to_an_unknown_field_creates_no_column(graph):
    before = _state(graph)
    graph.attrs.update('nodes', {'a': {'never_held': None}})
    graph.attrs.update('edges', {'ab': {'never_held': float('nan')}})
    assert _state(graph) == before


def test_caches_describe_the_restored_graph(graph, monkeypatch):
    matrix = graph.A.toarray().copy()
    view = graph.view(nodes=graph.N.select(count__gte=1))
    assert view.N.ids == ('a', 'b', 'c')
    monkeypatch.setattr(type(graph), '_apply_flexible_direction', _boom)
    with pytest.raises(RuntimeError):
        graph.attrs.update('nodes', {'a': {'x': 0.1, 'count': 0}, 'b': {'x': 0.9}})
    assert (graph.A.toarray() == matrix).all()
    assert view.N.ids == ('a', 'b', 'c')
    assert graph.N.select(count__gte=1).ids == ('a', 'b', 'c')
    assert graph.degree('a') == 1 and graph.degree('b') == 2
    # And the graph still works: the next write goes through.
    monkeypatch.undo()
    graph.attrs.update('nodes', {'a': {'count': 5}})
    assert graph.attrs.row('nodes', 'a')['count'] == 5


def test_a_contextual_address_drops_fields_a_failed_batch_introduced(monkeypatch):
    G = AnnNet()
    G.add_nodes(['n'])
    G.slices.add('s')
    G.attrs.update('slices', {'s': {'note': 'kept'}})
    before = (G.attrs.rows('slices'), G.attrs.schema('slices'))
    original = _attribute_api.Attrs._commit

    def write_then_fail(self, address, updates):
        original(self, address, updates)
        raise RuntimeError('commit failed')

    monkeypatch.setattr(_attribute_api.Attrs, '_commit', write_then_fail)
    with pytest.raises(RuntimeError, match='commit failed'):
        G.attrs.update('slices', {'s': {'introduced': 1, 'note': 'changed'}})
    monkeypatch.undo()
    assert (G.attrs.rows('slices'), G.attrs.schema('slices')) == before


def test_a_generic_address_drops_columns_a_failed_commit_introduced(graph, monkeypatch):
    before = _state(graph)
    original = _attribute_api.Attrs._commit

    def write_then_fail(self, address, updates):
        original(self, address, updates)
        raise RuntimeError('commit failed')

    monkeypatch.setattr(_attribute_api.Attrs, '_commit', write_then_fail)
    with pytest.raises(RuntimeError, match='commit failed'):
        graph.attrs.update('nodes', {'a': {'introduced': 1}, 'c': {'introduced': 2, 'count': 30}})
    with pytest.raises(RuntimeError, match='commit failed'):
        graph.attrs.replace('nodes', pd.DataFrame({'node_id': ['a'], 'introduced': [1]}))
    monkeypatch.undo()
    assert _state(graph) == before


def test_an_ordinary_scalar_write_copies_no_graph(graph, monkeypatch):
    def forbidden(self):
        raise AssertionError('a write copied the whole store')

    monkeypatch.setattr(ST.CoreState, 'copy', forbidden)
    monkeypatch.setattr(ST.CoreState, 'select', forbidden)
    graph.attrs.update('nodes', {'a': {'count': 9}})
    graph.attrs.update('nodes', {'a': {'fresh': 1}})
    graph.attrs.delete('nodes', names=['fresh'])
    graph.attrs.delete('nodes', keys=['a'], names=['count'])
    assert 'fresh' not in graph.attrs.schema('nodes').names()
