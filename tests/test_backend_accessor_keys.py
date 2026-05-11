"""``G.nx.<func>(G)`` must return vertex-ID-keyed results.

The README promises: "AnnNet ... maps vertex identifiers back where
supported." Algorithms that return ``{vertex: value}`` dicts must yield
vertex-ID strings as keys, not integer row indices.
"""

from __future__ import annotations

import pytest

nx = pytest.importorskip('networkx')

from annnet.core.graph import AnnNet


def _build_graph():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'source': 'A', 'target': 'B'}, {'source': 'B', 'target': 'C'}])
    return G


def test_degree_centrality_returns_vertex_id_keys():
    G = _build_graph()
    result = G.nx.degree_centrality(G)
    assert set(result.keys()) == {'A', 'B', 'C'}


def test_pagerank_returns_vertex_id_keys():
    G = _build_graph()
    result = G.nx.pagerank(G)
    assert set(result.keys()) == {'A', 'B', 'C'}


def test_results_match_native_nx_call_on_backend():
    G = _build_graph()
    backend = G.nx.backend(directed=True)
    expected = nx.degree_centrality(backend)
    actual = G.nx.degree_centrality(G)
    assert actual == expected
