"""An attribute batch is all or nothing.

A write can re-orient an edge through a flexible-direction policy, so a batch
that fails partway must restore the attribute values, the incidence columns
the policies already rewrote, the node-key index and every derived cache —
and a scalar write must do that without copying the graph. Each test is
checked against an independent reading of the graph.
"""

from __future__ import annotations


import numpy as np
import pytest

from annnet import AnnNet
from annnet.core import _store as ST
from annnet.core import _structure as S


# ---------------------------------------------------------------------------
# 1. transactions are atomic across attributes, topology, indexes and caches
# ---------------------------------------------------------------------------


def _column(G, edge_id):
    """The incidence column of one edge as ``{node_id: coefficient}``, read afresh."""
    matrix = G.S.toarray()
    col = G.idx.edge_to_col(edge_id)
    sides = S.edge_endpoints(G, edge_id)
    return {
        key[0]: float(matrix[G.idx.entity_to_row(key[0]), col])
        for key in sides.source | sides.target
    }


def _flexible(policy, *, second=False):
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.add_edges('a', 'b', edge_id='e', flexible=dict(policy))
    if second:
        G.add_edges('b', 'c', edge_id='f', flexible=dict(policy))
    return G


def _raise_after(monkeypatch, G, *, on_call=1):
    """Let the policy run, then raise — on the ``on_call``-th application."""
    original = type(G)._apply_flexible_direction
    calls = {'n': 0}

    def wrapped(self, edge_id):
        original(self, edge_id)
        calls['n'] += 1
        if calls['n'] == on_call:
            raise RuntimeError('policy failed after rewriting the edge')

    monkeypatch.setattr(type(G), '_apply_flexible_direction', wrapped)
    return calls


def test_a_failed_single_write_restores_the_edge_orientation(monkeypatch):
    G = _flexible({'var': 'x', 'threshold': 0.5, 'scope': 'edge'})
    G.attrs.update('edges', {'e': {'x': 1.0}})
    before = _column(G, 'e')
    assert before == {'a': 1.0, 'b': -1.0}
    clock = G._state_clock()
    _raise_after(monkeypatch, G)
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.update('edges', {'e': {'x': 0.0}})
    assert G.attrs.row('edges', 'e')['x'] == 1.0
    assert _column(G, 'e') == before, 'the rewritten orientation must be rolled back'
    assert S.edge_ref(G, 'e').directed is True
    assert G.validate() == []
    # The caches were re-derived after the rollback, not left holding the
    # intermediate state: a fresh matrix and the cached one agree.
    assert np.array_equal(G.S.toarray(), G.matrices.signed_matrix().toarray())
    # A rollback may move the clocks (it writes through the gateway); it must
    # never leave them where the failed commit had them while the state differs.
    assert G._state_clock() != clock or _column(G, 'e') == before


def test_a_failure_partway_through_a_batch_restores_every_edge(monkeypatch):
    G = _flexible({'var': 'x', 'threshold': 0.5, 'scope': 'edge'}, second=True)
    G.attrs.update('edges', {'e': {'x': 1.0}, 'f': {'x': 1.0}})
    before = {eid: _column(G, eid) for eid in ('e', 'f')}
    rows = G.attrs.rows('edges')
    calls = _raise_after(monkeypatch, G, on_call=2)
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.update('edges', {'e': {'x': 0.0, 'note': 'n'}, 'f': {'x': 0.0}})
    assert calls['n'] == 2, 'the first policy had already rewritten its edge'
    assert G.attrs.rows('edges') == rows
    assert {eid: _column(G, eid) for eid in ('e', 'f')} == before
    assert G.validate() == []


def test_a_failed_node_write_restores_edges_oriented_from_node_attributes(monkeypatch):
    G = _flexible({'var': 'level', 'threshold': 0.0, 'scope': 'node'})
    G.attrs.update('nodes', {'a': {'level': 5.0}, 'b': {'level': 1.0}})
    before = _column(G, 'e')
    assert before == {'a': 1.0, 'b': -1.0}
    G.set_node_key('level')
    index = dict(G._node_key_index)
    _raise_after(monkeypatch, G)
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.update('nodes', {'a': {'level': 0.5}, 'b': {'level': 9.0}})
    assert G.attrs.row('nodes', 'a')['level'] == 5.0 and G.attrs.row('nodes', 'b')['level'] == 1.0
    assert _column(G, 'e') == before
    assert G._node_key_index == index, 'the composite-key index is part of the transaction'


def test_a_failed_table_replacement_restores_topology_too(monkeypatch):
    import pandas as pd

    G = _flexible({'var': 'x', 'threshold': 0.5, 'scope': 'edge'})
    G.attrs.update('edges', {'e': {'x': 1.0}})
    before = _column(G, 'e')
    _raise_after(monkeypatch, G)
    with pytest.raises(RuntimeError, match='policy failed'):
        G.attrs.replace('edges', pd.DataFrame({'edge_id': ['e'], 'x': [0.0]}))
    assert dict(G.attrs.row('edges', 'e')) == {'x': 1.0}
    assert _column(G, 'e') == before


def test_an_ordinary_scalar_write_copies_no_graph(monkeypatch):
    """The transaction snapshots what a policy may touch, never the whole store."""
    G = AnnNet(directed=True)
    G.add_nodes([f'n{i}' for i in range(50)])
    G.add_edges('n0', 'n1', edge_id='e')

    def forbidden(self):
        raise AssertionError('a scalar write copied the whole store')

    monkeypatch.setattr(ST.CoreState, 'copy', forbidden)
    monkeypatch.setattr(ST.CoreState, 'select', forbidden)
    G.attrs.update('nodes', {'n3': {'score': 1.0}})
    G.attrs.update('edges', {'e': {'w': 2.0}})
    G.attrs.update('nodes', {f'n{i}': {'score': float(i)} for i in range(50)})
    assert G.attrs.row('nodes', 'n49')['score'] == 49.0
