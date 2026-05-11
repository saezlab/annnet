"""Standard Python protocols on AnnNet: __repr__, __len__, __iter__, __contains__.

`len(G)` and `for v in G` follow the NetworkX convention (vertex iteration).
`in` is vertex-membership only — edges are not supported as a `__contains__`
argument because edge identities can be int/str/tuple depending on kind.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet


def test_repr_empty_graph_is_informative():
    G = AnnNet(directed=True)
    s = repr(G)
    assert 'AnnNet' in s
    assert 'n_vertices' in s
    assert 'n_edges' in s
    assert '0' in s


def test_repr_populated_graph_lists_columns_and_uns():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'], kind='gene')
    G.add_edges([{'source': 'A', 'target': 'B', 'edge_id': 'e1', 'confidence': 0.9}])
    G.uns['study'] = 'demo'
    s = repr(G)
    assert '3' in s and '1' in s
    assert 'directed: True' in s
    assert "'kind'" in s
    assert "'confidence'" in s
    assert "'study'" in s


def test_len_returns_vertex_count():
    G = AnnNet(directed=False)
    assert len(G) == 0
    G.add_vertices(['A', 'B', 'C'])
    assert len(G) == 3


def test_iter_yields_vertex_ids():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    assert set(iter(G)) == {'A', 'B', 'C'}
    assert [v for v in G] == list(G.vertices())


def test_contains_checks_vertex_membership():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    assert 'A' in G
    assert 'B' in G
    assert 'Z' not in G


def test_contains_returns_false_for_edge_ids():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    assert 'e1' not in G
