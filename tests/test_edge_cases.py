import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import pytest

from annnet.core.graph import AnnNet
from annnet.io.json_format import from_json, to_json  # JSON (JavaScript Object Notation)
from annnet.io.parquet import (
    from_parquet,
    to_parquet,
)  # Parquet (columnar storage)
from annnet.io.sif import from_sif, to_sif  # SIF (Simple Interaction Format)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _roundtrip_json(G, tmpdir, name='g'):
    p = tmpdir / f'{name}.json'
    to_json(G, p)
    return from_json(p)


def _roundtrip_parquet(G, tmpdir, name='g'):
    p = tmpdir / f'{name}_dir'
    to_parquet(G, p)
    return from_parquet(p)


def _roundtrip_sif(G, tmpdir, name='g'):
    sif_p = tmpdir / f'{name}.sif'
    man_p = tmpdir / f'{name}.manifest.json'
    to_sif(G, sif_p, lossless=True, manifest_path=man_p)
    return from_sif(sif_p, manifest=man_p)


# ---------------------------------------------------------------------------
# Empty-graph — each adapter is a separate parametrized case
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('adapter', ['json', 'parquet', 'dataframe'])
def test_empty_graph(adapter, tmpdir_fixture):
    G = AnnNet()

    if adapter == 'json':
        G2 = _roundtrip_json(G, tmpdir_fixture, 'empty')
    elif adapter == 'parquet':
        G2 = _roundtrip_parquet(G, tmpdir_fixture, 'empty')
    else:
        from annnet.io.dataframes import from_dataframes, to_dataframes

        dfs = to_dataframes(G)
        G2 = from_dataframes(**dfs)

    assert len(list(G2.vertices())) == 0


# ---------------------------------------------------------------------------
# Special characters in vertex IDs — json and parquet tested separately
# ---------------------------------------------------------------------------

SPECIAL_IDS = [
    'node with spaces',
    'node-with-dashes',
    'node_with_underscores',
    'node.with.dots',
    'α',
    'β',
    'γ',
    'node\twith\ttabs',
]


def _build_special_graph():
    G = AnnNet()
    for vid in SPECIAL_IDS:
        G.add_vertices(vid)
    G.add_edges(SPECIAL_IDS[0], SPECIAL_IDS[1], edge_id='e1')
    G.add_edges('α', 'β', edge_id='e2')
    return G


@pytest.mark.parametrize('adapter', ['json', 'parquet'])
def test_special_characters_in_ids(adapter, tmpdir_fixture):
    G = _build_special_graph()

    if adapter == 'json':
        G2 = _roundtrip_json(G, tmpdir_fixture, 'special')
    else:
        G2 = _roundtrip_parquet(G, tmpdir_fixture, 'special')

    assert set(G.vertices()) == set(G2.vertices())


# ---------------------------------------------------------------------------
# Extreme weights — json only
# ---------------------------------------------------------------------------


def test_large_weights_and_extreme_values(tmpdir_fixture):
    G = AnnNet()
    G.add_vertices('A')
    G.add_vertices('B')
    G.add_edges('A', 'B', edge_id='e1', weight=1e10)
    G.add_edges('A', 'B', edge_id='e2', weight=1e-10, parallel='parallel')
    G.add_edges('A', 'B', edge_id='e3', weight=0.0, parallel='parallel')

    G2 = _roundtrip_json(G, tmpdir_fixture, 'extreme')

    assert abs(G2.edge_weights.get('e1', 0) - 1e10) < 1e-6
    assert abs(G2.edge_weights.get('e2', 0) - 1e-10) < 1e-15
    assert abs(G2.edge_weights.get('e3', 1) - 0.0) < 1e-10


# ---------------------------------------------------------------------------
# Self-loops — json, parquet, sif tested separately
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('adapter', ['json', 'parquet', 'sif'])
def test_self_loops(adapter, tmpdir_fixture):
    G = AnnNet()
    G.add_vertices('A')
    G.add_edges('A', 'A', edge_id='loop', weight=2.5)

    if adapter == 'json':
        G2 = _roundtrip_json(G, tmpdir_fixture, 'loop')
    elif adapter == 'parquet':
        G2 = _roundtrip_parquet(G, tmpdir_fixture, 'loop')
    else:
        G2 = _roundtrip_sif(G, tmpdir_fixture, 'loop')

    assert 'loop' in G2.edge_to_idx


# ---------------------------------------------------------------------------
# Parallel edges — json
# ---------------------------------------------------------------------------


def test_parallel_edges(tmpdir_fixture):
    G = AnnNet()
    G.add_vertices('A')
    G.add_vertices('B')
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('A', 'B', edge_id='e2', weight=2.0, parallel='parallel')
    G.add_edges('A', 'B', edge_id='e3', weight=3.0, parallel='parallel')

    G2 = _roundtrip_json(G, tmpdir_fixture, 'parallel')

    assert 'e1' in G2.edge_to_idx
    assert 'e2' in G2.edge_to_idx
    assert 'e3' in G2.edge_to_idx
    assert G2.ne == 3


# ---------------------------------------------------------------------------
# None / null attribute handling — json
# ---------------------------------------------------------------------------


def test_null_and_none_handling(tmpdir_fixture):
    G = AnnNet()
    G.add_vertices('A')
    G.attrs.set_vertex_attrs('A', present='value', missing=None, zero=0, empty_string='')

    G2 = _roundtrip_json(G, tmpdir_fixture, 'nulls')

    attrs = G2.attrs.get_vertex_attrs('A') or {}
    assert attrs.get('present') == 'value'
    assert attrs.get('zero') == 0
    assert 'missing' not in attrs or attrs.get('missing') is None


# ---------------------------------------------------------------------------
# Large graph — parquet only
# ---------------------------------------------------------------------------


def test_very_large_graph(tmpdir_fixture):
    import random

    G = AnnNet()
    n_vertices = 1000
    n_edges = 2000
    for i in range(n_vertices):
        G.add_vertices(f'v{i}')
    random.seed(42)
    for i in range(n_edges):
        u = f'v{random.randint(0, n_vertices - 1)}'  # nosec B311
        v = f'v{random.randint(0, n_vertices - 1)}'  # nosec B311
        G.add_edges(u, v, edge_id=f'e{i}', weight=random.random(), parallel='parallel')  # nosec B311

    G2 = _roundtrip_parquet(G, tmpdir_fixture, 'large')

    assert len(list(G2.vertices())) == n_vertices
    assert G2.ne == n_edges
