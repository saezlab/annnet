"""Every subgraph path shares one materializer.

``layers.subgraph_from_layer_*``, ``ops.subgraph`` and ``V.materialize()``
resolve and copy through the same code, so a layer subgraph keeps every
attribute address, its structure and the meaning of its boundary options.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet
from annnet.core import _structure as S


# ---------------------------------------------------------------------------
# 2. one materializer behind every subgraph path
# ---------------------------------------------------------------------------


@pytest.fixture
def layered():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True, aspects={'c': ['a', 'b']})
        G.layers.set_ordered('c')
        G.layers.place(['n', 'm', 'k'], [('a',), ('b',)])
        G.add_edges(('n', ('a',)), ('m', ('a',)), edge_id='e1', weight=2.5)
        G.add_edges(('m', ('a',)), ('k', ('a',)), edge_id='e2', directed=False)
        G.add_edges(('n', ('a',)), ('n', ('b',)), edge_id='couple')
        G.add_edges(('m', ('a',)), ('k', ('b',)), edge_id='inter')
        G.add_edges(
            [{'head': [('n', ('a',))], 'tail': [('m', ('a',)), ('k', ('a',))], 'edge_id': 'h'}]
        )
        G.set_edge_coeffs('h', {('n', ('a',)): 2.0, ('m', ('a',)): -1.0, ('k', ('a',)): -3.0})
        G.slices.add('s', edges=['e1', 'h'])
        G.attrs.update('nodes', {'n': {'gene': 'TP53'}})
        G.attrs.update('edges', {'e1': {'score': 7}})
        G.attrs.update('slices', {'s': {'role': 'prior'}})
        G.attrs.update('aspects', {'c': {'unit': 'cond'}})
        G.attrs.update('layers', {('a',): {'note': 'baseline'}})
        G.attrs.update('edge_slices', {('s', 'e1'): {'activity': 0.4}})
        G.attrs.update('node_layers', {('n', ('a',)): {'value': 42}})
        G.attrs.update('elementary_layers', {('c', 'a'): {'label': 'control'}})
        G.uns['study'] = 'demo'
    return G


def _all_addresses(G):
    return {address: G.attrs.rows(address) for address in G.attrs}


def _structure_of(G):
    store = G._store
    out = {}
    for eid in G.E:
        ref = S.edge_ref(G, eid)
        slot = store.edge_slot(eid)
        members = store.members(slot)
        out[eid] = (
            ref.kind,
            ref.directed,
            ref.declared_weight,
            ref.ml_kind,
            ref.ml_layers,
            sorted(
                (store.entity_key(int(e)), float(c), int(r))
                for e, c, r in zip(
                    members.entities, members.coefficients, members.roles, strict=True
                )
            ),
        )
    return out


def test_layer_subgraph_keeps_every_address_and_the_structure(layered):
    H = layered.layers.subgraph_from_layer_tuple(('a',))
    V = layered.view(layers=[('a',)]).materialize()
    assert list(H.E) == list(V.E) == ['e1', 'e2', 'h']
    assert list(H.supra_nodes()) == list(V.supra_nodes())
    assert _all_addresses(H) == _all_addresses(V)
    assert H.attrs.row('edges', 'e1')['score'] == 7
    assert H.attrs.row('node_layers', ('n', ('a',)))['value'] == 42
    assert H.attrs.row('nodes', 'n')['gene'] == 'TP53'
    assert H.attrs.row('slices', 's')['role'] == 'prior'
    assert H.attrs.row('aspects', 'c')['unit'] == 'cond'
    assert H.attrs.row('layers', ('a',))['note'] == 'baseline'
    assert H.attrs.row('edge_slices', ('s', 'e1'))['activity'] == 0.4
    assert H.attrs.row('elementary_layers', ('c', 'a'))['label'] == 'control'
    assert H.uns['study'] == 'demo'
    assert _structure_of(H) == {
        eid: spec for eid, spec in _structure_of(layered).items() if eid in H.E
    }
    assert H.layers.aspect('c').ordered is True
    assert H.validate() == []


@pytest.mark.parametrize(
    'kwargs, expected_edges',
    [
        ({}, ['e1', 'e2', 'h']),
        ({'include_inter': True}, ['e1', 'e2', 'h']),
        ({'include_inter': True, 'boundary': 'open'}, ['e1', 'e2', 'inter', 'h']),
        ({'include_coupling': True, 'boundary': 'open'}, ['e1', 'e2', 'couple', 'h']),
    ],
)
def test_layer_subgraph_options_keep_their_meaning(layered, kwargs, expected_edges):
    H = layered.layers.subgraph_from_layer_tuple(('a',), **kwargs)
    assert list(H.E) == expected_edges
    touched = set()
    for eid in expected_edges:
        touched.update(
            S.edge_endpoints(layered, eid).source | S.edge_endpoints(layered, eid).target
        )
    assert set(H.supra_nodes()) == {k for k in S.node_keys(layered) if k[1] == ('a',)} | touched
    assert _structure_of(H) == {
        eid: spec for eid, spec in _structure_of(layered).items() if eid in H.E
    }
    assert H.attrs.row('edges', 'e1')['score'] == 7
    for key in H.supra_nodes():
        assert dict(H.attrs.row('node_layers', key)) == dict(layered.attrs.row('node_layers', key))


def test_layer_union_intersection_difference_subgraphs_share_the_materializer(layered):
    union = layered.layers.subgraph_from_layer_union([('a',), ('b',)])
    assert set(union.E) == {'e1', 'e2', 'h'}, 'intra edges alone by default'
    union = layered.layers.subgraph_from_layer_union(
        [('a',), ('b',)], include_inter=True, include_coupling=True
    )
    assert set(union.E) == {'e1', 'e2', 'couple', 'inter', 'h'}
    assert union.attrs.row('node_layers', ('n', ('a',)))['value'] == 42
    assert union.attrs.row('edges', 'e1')['score'] == 7
    inter = layered.layers.subgraph_from_layer_intersection([('a',), ('b',)])
    assert set(inter.N) == {'n', 'm', 'k'}
    assert set(inter.E) == set()
    assert inter.attrs.row('nodes', 'n')['gene'] == 'TP53'
    diff = layered.layers.subgraph_from_layer_difference(('a',), ('b',))
    assert list(diff.N) == []
    assert diff.attrs.rows('nodes') == {}
    for H in (union, inter, diff):
        assert H.validate() == []
        assert H.uns['study'] == 'demo'


def test_ops_subgraphs_and_views_agree_on_the_layered_graph(layered):
    for nodes in (['n', 'm'], [('n', ('a',)), ('m', ('a',)), ('k', ('a',))]):
        H = layered.ops.subgraph(nodes)
        V = layered.view(nodes=nodes).materialize()
        assert list(H.E) == list(V.E)
        assert _all_addresses(H) == _all_addresses(V)
        assert _structure_of(H) == _structure_of(V)
    H = layered.ops.edge_subgraph(['h', 'inter'])
    V = layered.view(edges=['h', 'inter']).materialize()
    assert list(H.supra_nodes()) == list(V.supra_nodes())
    assert _all_addresses(H) == _all_addresses(V)
