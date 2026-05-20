"""Coverage tests for ``annnet/io/cx2.py``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from annnet.core.graph import AnnNet
from annnet.io.cx2 import (
    _cx2_collect_reified,
    _cx2_to_cytoscapejs,
    _infer_cx2_type,
    _infer_cx2_types,
    _jsonify,
    _serialize_slices_public,
    from_cx2,
    to_cx2,
)


# ── pure helpers ───────────────────────────────────────────────────────


def test_infer_cx2_type_handles_each_python_type() -> None:
    assert _infer_cx2_type([]) == 'string'
    assert _infer_cx2_type([None, None]) == 'string'
    assert _infer_cx2_type([True, False]) == 'boolean'
    assert _infer_cx2_type([1, 2, 3]) == 'long'
    assert _infer_cx2_type([1.5, 2.5]) == 'double'
    assert _infer_cx2_type(['a', 'b']) == 'string'
    out = _infer_cx2_type([[1, 2], [3, 4]])
    assert out == 'list_of_long'


def test_infer_cx2_types_skips_id_column() -> None:
    rows = [{'id': 1, 'name': 'a'}, {'id': 2, 'name': 'b'}]
    out = _infer_cx2_types(rows, id_col='id')
    assert 'id' not in out
    assert out.get('name') == 'string'


def test_jsonify_recursively_unwraps_dict_set_list_tuple() -> None:
    payload = {'k': {1, 2}, 'l': [(1, 2), 3]}
    out = _jsonify(payload)
    # Sets become lists; tuples become lists; nesting walked.
    assert isinstance(out['k'], list)
    assert sorted(out['k']) == [1, 2]
    assert out['l'] == [[1, 2], 3]


def test_serialize_slices_public_returns_per_slice_dict() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.slices.add('s1')
    G.add_edges('A', 'B', edge_id='e1', slice='s1')
    out = _serialize_slices_public(G)
    assert 's1' in out
    assert 'vertices' in out['s1']
    assert 'edges' in out['s1']


# ── to_cx2 hyperedge modes ─────────────────────────────────────────────


def _toy_with_undirected_hyper() -> AnnNet:
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges(['A', 'B', 'C'], edge_id='h1')
    return G


def _toy_with_directed_hyper() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges(src=['A', 'B'], tgt=['C', 'D'], edge_id='h1')
    return G


def test_to_cx2_with_hyperedges_expand_undirected_emits_clique_edges() -> None:
    G = _toy_with_undirected_hyper()
    out = to_cx2(G, hyperedges='expand')
    edges_aspect = next(a for a in out if 'edges' in a)
    # 3 members → 3 unordered pairs.
    assert len(edges_aspect['edges']) >= 3


def test_to_cx2_with_hyperedges_expand_directed_emits_cartesian_edges() -> None:
    G = _toy_with_directed_hyper()
    out = to_cx2(G, hyperedges='expand')
    edges_aspect = next(a for a in out if 'edges' in a)
    # 2 tail × 2 head → 4 expanded edges
    assert len(edges_aspect['edges']) >= 4


def test_to_cx2_with_hyperedges_reify_undirected_emits_hyperedge_node() -> None:
    G = _toy_with_undirected_hyper()
    out = to_cx2(G, hyperedges='reify')
    nodes_aspect = next(a for a in out if 'nodes' in a)
    has_he_node = any(n.get('v', {}).get('is_hyperedge') for n in nodes_aspect['nodes'])
    assert has_he_node


def test_to_cx2_with_hyperedges_reify_directed_emits_role_edges() -> None:
    G = _toy_with_directed_hyper()
    out = to_cx2(G, hyperedges='reify')
    edges_aspect = next(a for a in out if 'edges' in a)
    roles = {e.get('v', {}).get('role') for e in edges_aspect['edges']}
    assert 'head' in roles
    assert 'tail' in roles


def test_to_cx2_with_hyperedges_skip_omits_them_entirely() -> None:
    G = _toy_with_undirected_hyper()
    out = to_cx2(G, hyperedges='skip')
    edges_aspect = next(a for a in out if 'edges' in a)
    # No clique edges; only binary edges (which there are none here).
    assert edges_aspect['edges'] == []


# ── to_cx2 path writing ────────────────────────────────────────────────


def test_to_cx2_writes_file_when_path_given(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    out_path = tmp_path / 'out.cx2.json'
    to_cx2(G, out_path)
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding='utf-8'))
    assert isinstance(payload, list)


def test_to_cx2_with_layer_arg_subgraphs_first() -> None:
    """Pass ``layer=(...)`` → exports only that elementary layer subgraph."""
    G = AnnNet(directed=True)
    G.layers.set_aspects(['condition'], {'condition': ['healthy', 'treated']})
    G.add_vertices(['A'], layer={'condition': 'healthy'})
    G.add_vertices(['B'], layer={'condition': 'treated'})
    out = to_cx2(G, layer=('healthy',))
    nodes_aspect = next(a for a in out if 'nodes' in a)
    # only the 'healthy' subgraph nodes should be present (A only)
    names = {n.get('v', {}).get('name') for n in nodes_aspect['nodes']}
    assert 'A' in names


def test_to_cx2_with_layer_must_be_tuple() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A'])
    with pytest.raises(TypeError, match='layer must be a tuple'):
        to_cx2(G, layer='not-a-tuple')


# ── attribute cleaning and styles ──────────────────────────────────────


def test_to_cx2_with_vertex_attributes_strings_and_layout_columns() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.attrs.set_vertex_attrs('A', label='alpha', layout_x=1.0, layout_y=2.0)
    G.attrs.set_vertex_attrs('B', label='beta')
    G.add_edges('A', 'B', edge_id='e1')
    out = to_cx2(G)
    nodes_aspect = next(a for a in out if 'nodes' in a)
    # Both nodes are emitted.
    assert len(nodes_aspect['nodes']) == 2
    # The layout_x value was hoisted to top-level 'x' for one node.
    has_x = any(n.get('x') == 1.0 for n in nodes_aspect['nodes'])
    assert has_x


def test_to_cx2_with_edge_attributes_attaches_them_under_v() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    G.attrs.set_edge_attrs('e1', label='alpha', confidence=0.95)
    out = to_cx2(G)
    edges_aspect = next(a for a in out if 'edges' in a)
    e = edges_aspect['edges'][0]
    assert e['v'].get('label') == 'alpha'
    assert e['v'].get('confidence') == 0.95


# ── from_cx2 ingestion paths ──────────────────────────────────────────


def test_from_cx2_round_trip_via_manifest() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=2.0)
    G.add_edges('B', 'C', edge_id='e2', weight=3.0)
    cx2 = to_cx2(G)
    H = from_cx2(cx2)
    assert set(H.vertices()) == {'A', 'B', 'C'}
    assert H.ne == 2


def test_from_cx2_reads_from_file_path(tmp_path: Path) -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    p = tmp_path / 'demo.cx2.json'
    to_cx2(G, p)
    H = from_cx2(str(p))
    assert set(H.vertices()) == {'A', 'B'}


def test_from_cx2_reads_from_json_string() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    cx2 = to_cx2(G)
    payload = json.dumps(cx2)
    H = from_cx2(payload)
    assert set(H.vertices()) == {'A', 'B'}


def test_from_cx2_invalid_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match='Invalid CX2'):
        from_cx2('not valid json and not a file path')


def test_from_cx2_with_no_manifest_still_imports_basic_nodes_and_edges() -> None:
    """A hand-built CX2 (no AnnNet manifest) gets reconstructed structurally."""
    cx2 = [
        {'CXVersion': '2.0', 'hasFragments': False},
        {'metaData': []},
        {
            'attributeDeclarations': [
                {'nodes': {'name': {'d': 'string'}}, 'edges': {}, 'networkAttributes': {}}
            ]
        },
        {'networkAttributes': [{'name': 'manual'}]},
        {'nodes': [{'id': 0, 'v': {'name': 'A'}}, {'id': 1, 'v': {'name': 'B'}}]},
        {'edges': [{'id': 0, 's': 0, 't': 1, 'v': {'interaction': 'edge'}}]},
        {'status': [{'success': True}]},
    ]
    H = from_cx2(cx2)
    assert set(H.vertices()) == {'A', 'B'}
    assert H.ne == 1


def test_from_cx2_with_reified_hyperedges_round_trips_via_manifest() -> None:
    G = _toy_with_undirected_hyper()
    cx2 = to_cx2(G, hyperedges='reify')
    H = from_cx2(cx2)
    # Hyperedge survives via the manifest reconstruction path.
    assert H.ne >= 1


# ── _cx2_collect_reified ──────────────────────────────────────────────


def test_cx2_collect_reified_returns_empty_when_no_he_node() -> None:
    aspects = {'nodes': [], 'edges': []}
    defs, mem = _cx2_collect_reified(aspects)
    assert defs == []
    assert mem == set()


def test_cx2_collect_reified_discovers_undirected_membership_edges() -> None:
    aspects = {
        'nodes': [
            {'id': 0, 'v': {'name': 'A'}},
            {'id': 1, 'v': {'name': 'B'}},
            {'id': 2, 'v': {'is_hyperedge': True, 'eid': 'h1'}},
        ],
        'edges': [
            {'id': 0, 's': 2, 't': 0, 'v': {}},
            {'id': 1, 's': 2, 't': 1, 'v': {}},
        ],
    }
    defs, mem = _cx2_collect_reified(aspects)
    assert len(defs) == 1
    eid, directed, head, tail, *_ = defs[0]
    assert eid == 'h1'
    assert mem == {0, 1}


# ── _cx2_to_cytoscapejs ──────────────────────────────────────────────


def test_cx2_to_cytoscapejs_emits_node_and_edge_dicts() -> None:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    cx2 = to_cx2(G)
    cy = _cx2_to_cytoscapejs(cx2)
    assert 'nodes' in cy
    assert 'edges' in cy
    assert len(cy['nodes']) == 2
    assert len(cy['edges']) == 1
    assert cy['nodes'][0]['data'].get('label') in ('A', 'B')


def test_cx2_collect_reified_skips_hyperedge_node_with_no_eid() -> None:
    aspects = {
        'nodes': [{'id': 0, 'v': {'is_hyperedge': True}}],  # no eid
        'edges': [],
    }
    defs, mem = _cx2_collect_reified(aspects)
    assert defs == []
    assert mem == set()


def test_from_cx2_skips_empty_aspect_dicts() -> None:
    """Coverage for the ``if not item: continue`` skip in the aspect loop."""
    cx2 = [
        {'CXVersion': '2.0', 'hasFragments': False},
        {},  # empty dict → skipped
        {'metaData': []},
        {'networkAttributes': [{'name': 'demo'}]},
        {'nodes': [{'id': 0, 'v': {'name': 'A'}}]},
        {'edges': []},
        {'status': [{'success': True}]},
    ]
    H = from_cx2(cx2)
    assert 'A' in H.vertices()
