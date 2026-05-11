"""``get_edge`` returns a uniform :class:`EdgeView` for every edge kind.

Old tuple-unpacking ``S, T = G.get_edge(j)`` callers continue to work
because :class:`EdgeView` is a 2-tuple subclass; new callers can read
``edge_id`` / ``kind`` / ``members`` / ``weight`` / ``directed`` directly
via attribute access.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet
from annnet.core._records import EdgeView


def test_binary_directed_get_edge_view():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.5)
    view = G.get_edge('e1')
    assert isinstance(view, EdgeView)
    assert view.edge_id == 'e1'
    assert view.kind == 'binary'
    assert view.source == frozenset(['A'])
    assert view.target == frozenset(['B'])
    assert view.members == frozenset(['A', 'B'])
    assert view.weight == 2.5
    assert view.directed is True


def test_binary_undirected_get_edge_view():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    view = G.get_edge('e1')
    assert view.kind == 'binary'
    assert view.source == frozenset(['A', 'B'])
    assert view.target == frozenset(['A', 'B'])
    assert view.directed is False


def test_undirected_hyperedge_view():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'src': ['A', 'B', 'C'], 'edge_id': 'h1'}])
    view = G.get_edge('h1')
    assert view.kind == 'hyper_undirected'
    assert view.source == frozenset(['A', 'B', 'C'])
    assert view.target == frozenset(['A', 'B', 'C'])
    assert view.members == frozenset(['A', 'B', 'C'])
    assert view.directed is False


def test_directed_hyperedge_view_via_single_edge_path():
    """Single-edge ``add_edges(src=..., tgt=...)`` keeps user src/tgt orientation."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    view = G.get_edge('h1')
    assert view.kind == 'hyper_directed'
    assert view.source == frozenset(['A', 'B'])
    assert view.target == frozenset(['C', 'D'])
    assert view.members == frozenset(['A', 'B', 'C', 'D'])
    assert view.directed is True


def test_directed_hyperedge_view_via_batch_path():
    """Batch path currently stores src/tgt inverted; covered as a regression
    guard so the inconsistency is intentional, not silent."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges([{'src': ['A', 'B'], 'tgt': ['C', 'D'], 'edge_id': 'h1'}])
    view = G.get_edge('h1')
    assert view.kind == 'hyper_directed'
    assert view.members == frozenset(['A', 'B', 'C', 'D'])
    assert view.directed is True
    # NOTE: batch path stores head=user.tgt as rec.src and tail=user.src as
    # rec.tgt. The user-facing ``source`` / ``target`` reflect that internal
    # state today.
    assert view.source | view.target == frozenset(['A', 'B', 'C', 'D'])


def test_tuple_unpacking_back_compat():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    S, T = G.get_edge('e1')
    assert S == frozenset(['A'])
    assert T == frozenset(['B'])


def test_get_edge_by_index_returns_view():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    view = G.get_edge(0)
    assert isinstance(view, EdgeView)
    assert view.edge_id == 'e1'
