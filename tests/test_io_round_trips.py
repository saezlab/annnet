"""IO round-trip basics.

- JSON IO must preserve ``uns``.
- ``from_dataframes(to_dataframes(G))`` must round-trip.
- ``to_cx2(G, path)`` must accept a path argument like its siblings.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet
from annnet.io import json_format, dataframes
from annnet.io import cx2 as cx2_io


@pytest.fixture
def graph():
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'], kind='gene')
    G.add_edges(
        [
            {'source': 'A', 'target': 'B', 'edge_id': 'e1', 'confidence': 0.9},
            {'source': 'B', 'target': 'C', 'edge_id': 'e2'},
        ]
    )
    G.uns['study'] = 'test'
    G.uns['thresholds'] = {'low': 0.1, 'high': 0.9}
    return G


# ── JSON ``uns`` round-trip ───────────────────────────────────────────────


def test_json_round_trip_preserves_uns(graph, tmp_path):
    p = tmp_path / 'graph.json'
    json_format.to_json(graph, str(p))
    loaded = json_format.from_json(str(p))
    assert loaded.uns.get('study') == 'test'
    assert loaded.uns.get('thresholds') == {'low': 0.1, 'high': 0.9}


# ── ``from_dataframes(to_dataframes(G))`` ─────────────────────────────────


def test_dataframes_round_trip(graph):
    dfs = dataframes.to_dataframes(graph)
    loaded = dataframes.from_dataframes(dfs)
    assert loaded.nv == graph.nv
    assert loaded.ne == graph.ne
    assert set(loaded.edges()) == set(graph.edges())


# ── ``to_cx2(G, path)`` accepts a path ────────────────────────────────────


def test_to_cx2_accepts_path_argument(graph, tmp_path):
    p = tmp_path / 'graph.cx2'
    cx2_io.to_cx2(graph, str(p))
    assert p.exists()
    assert p.stat().st_size > 0
