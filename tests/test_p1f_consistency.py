"""P1-F: API consistency cleanups.

- ``incident_edges`` returns a list, matching ``vertices``/``edges``/
  ``edge_list``.
- Multilayer ``add_edges`` with bare vertex strings warns and falls back
  to the placeholder layer, mirroring ``add_vertices``.
- ``layers.list_layers`` hides the synthetic ``'_'`` placeholder by
  default; ``include_placeholder=True`` opts in.
"""

from __future__ import annotations

import warnings

from annnet.core.graph import AnnNet


def test_incident_edges_returns_list():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([('A', 'B'), ('B', 'C')])
    out = G.incident_edges('B')
    assert isinstance(out, list)
    assert len(out) == 2


def test_incident_edges_empty_returns_empty_list():
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    assert G.incident_edges('A') == []


def test_multilayer_add_edges_with_bare_ids_warns_and_falls_back():
    G = AnnNet(directed=False)
    G.layers.set_aspects(['condition'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        G.add_edges('a', 'b')
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any('placeholder layer' in m for m in msgs), msgs
    assert 'a' in G.vertices()
    assert 'b' in G.vertices()


def test_list_layers_hides_placeholder_by_default():
    G = AnnNet(directed=False)
    G.layers.set_aspects(['condition'], {'condition': ['healthy']})
    G.add_vertices(['a'])  # falls back to placeholder
    G.add_vertices(['b'], layer={'condition': 'healthy'})
    layers = G.layers.list_layers('condition')
    assert layers == ['healthy']


def test_list_layers_include_placeholder_opts_in():
    G = AnnNet(directed=False)
    G.layers.set_aspects(['condition'], {'condition': ['healthy']})
    G.add_vertices(['a'])  # placeholder
    G.add_vertices(['b'], layer={'condition': 'healthy'})
    layers = G.layers.list_layers('condition', include_placeholder=True)
    assert layers == ['_', 'healthy']
