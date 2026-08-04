"""``obs`` and ``var`` row counts must equal ``nv`` and ``ne``.

Mirrors AnnData's contract: every observation has a row in ``obs``,
every variable has a row in ``var`` — even if no extra attributes have
been set. Cells without user attributes are null.
"""

from __future__ import annotations

from annnet.core.graph import AnnNet
from annnet.core import _structure as S


def test_obs_row_count_matches_nv():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    assert G.obs.shape[0] == G.nv_supra == 3


def test_var_row_count_matches_ne_after_binary_edges_with_no_attrs():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([('A', 'B'), ('B', 'C')])
    assert G.var.shape[0] == G.ne == 2


def test_var_row_count_matches_ne_after_single_edge_with_attrs():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', confidence=0.9)
    assert G.var.shape[0] == G.ne == 1


def test_var_row_count_matches_ne_after_hyperedge():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'src': ['A', 'B', 'C'], 'edge_id': 'h1'}])
    assert G.var.shape[0] == G.ne == 1


def test_var_row_count_matches_registry_after_edge_entity():
    G = AnnNet(directed=False)
    G.add_edges([{'edge_id': 'EE1'}], as_entity=True)
    # An edge entity is an edge the graph knows the name of, and it has no
    # structural incidence (no matrix column), so ne stays at 0 while var still
    # mirrors every edge the graph holds.
    assert G.var.shape[0] == len(list(S.iter_edges(G, include_placeholders=True))) == 1
    assert G.ne == 0


def test_var_row_decreases_when_edge_removed():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([('A', 'B'), ('B', 'C')])
    G.remove_edges('edge_0')
    assert G.var.shape[0] == G.ne == 1
