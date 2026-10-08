"""View and selection semantics that are easy to get wrong, pinned as regressions.

Each test states one rule of the composable view — a false filter selects
nothing, a scalar id is one node, nested views only restrict, materialization
agrees with the view — on a flat and a layered fixture.

Flat fixture: nodes ``AA, BB, CC``; edges ``AA→BB``, ``BB→CC``.
Layered fixture: nodes ``A, B`` on ``ctrl`` and ``stim``; a valid ``stim`` edge.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet
from annnet.core import _structure as S


@pytest.fixture
def flat():
    G = AnnNet(directed=True)
    G.add_nodes(['AA', 'BB', 'CC'])
    G.attrs.update('nodes', {'AA': {'score': 1.0}, 'BB': {'score': 0.5}, 'CC': {'score': 0.2}})
    G.add_edges('AA', 'BB', edge_id='e1')
    G.add_edges('BB', 'CC', edge_id='e2')
    return G


@pytest.fixture
def layered():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True, aspects={'cond': ['ctrl', 'stim']})
        for cond in ('ctrl', 'stim'):
            G.add_nodes(['A', 'B'], layer=(cond,))
        G.add_edges(('A', ('stim',)), ('B', ('stim',)), edge_id='e_stim')
        G.add_edges([{'members': [('A', ('ctrl',)), ('B', ('ctrl',))], 'edge_id': 'h_ctrl'}])
    return G


# 1. A false node predicate selects nothing, not everything.
def test_a_false_predicate_selects_no_nodes(flat):
    V = flat.view(predicate=lambda n: False)
    assert len(V.N) == 0
    assert V.N.ids == ()
    assert len(V.E) == 0


# 2. A false edge predicate on a nested view selects no edges.
def test_a_false_edge_filter_on_a_nested_view_selects_no_edges(flat):
    V = flat.view().view(edges=lambda e: False)
    assert len(V.E) == 0
    # An edge-only filter keeps the selected edges and their endpoint entities,
    # not every unrelated node of the parent.
    assert len(V.N) == 0
    W = flat.view().view(edges=lambda e: e == 'e1')
    assert W.E.ids == ('e1',)
    assert W.N.ids == ('AA', 'BB')


# 3. A scalar node ID is one node, not a string of characters.
def test_a_scalar_node_id_is_one_node(flat):
    V = flat.view(nodes='AA')
    assert V.N.ids == ('AA',)
    assert len(V.N) == 1


# 4. The view and its materialization agree on the edges.
def test_view_and_materialization_agree_on_edges(flat):
    V = flat.view(nodes=['AA'])
    assert len(V.E) == 0
    H = V.materialize()
    assert len(H.E) == 0
    assert list(H.N) == ['AA']


# 5. A structured predicate is live: the membership follows the attribute.
def test_a_structured_selection_follows_the_attribute(flat):
    V = flat.view(nodes=flat.N.select(score__gt=0.4))
    assert V.N.ids == ('AA', 'BB')
    flat.attrs.update('nodes', {'BB': {'score': 0.1}})
    assert V.N.ids == ('AA',)
    flat.attrs.update('nodes', {'CC': {'score': 0.9}})
    assert V.N.ids == ('AA', 'CC')


# 6. An exception inside a predicate propagates with the offending ID.
def test_an_error_inside_a_predicate_propagates(flat):
    def broken(node_id):
        raise TypeError('boom')

    V = flat.view(predicate=broken)
    with pytest.raises(TypeError, match=r"boom.*'AA'|'AA'.*boom"):
        _ = V.N.ids


# 7. Bare IDs on a layered graph keep the valid edge between their placements.
def test_bare_ids_keep_the_edge_between_their_placements(layered):
    V = layered.view(nodes=['A', 'B'], edges=['e_stim'])
    assert V.E.ids == ('e_stim',)
    assert len(V.E) == 1


# 8. Explicit (node_id, layer) keys select those placements.
def test_explicit_placements_are_selected(layered):
    V = layered.view(nodes=[('A', ('stim',)), ('B', ('stim',))])
    assert V.N.ids == ('A', 'B')
    assert set(V.supra_nodes()) == {('A', ('stim',)), ('B', ('stim',))}
    assert V.E.ids == ('e_stim',)


# 9. Materialization inserts no placeholder layer and no placeholder placement.
def test_materialization_inserts_no_placeholder_placement(layered):
    V = layered.view(nodes=[('A', ('stim',)), ('B', ('stim',))])
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        H = V.materialize()
    assert set(H.supra_nodes()) == {('A', ('stim',)), ('B', ('stim',))}
    assert '_' not in H.layers.list_layers('cond', include_placeholder=True)
    assert list(H.E) == ['e_stim']


# 10. The kind of a hyperedge reads the same on every surface.
def test_edge_kind_agrees_between_record_column_and_frame(layered):
    assert layered.E.at('h_ctrl').kind == 'hyper'
    assert layered.E.at('h_ctrl').directed is False
    kinds = dict(zip(layered.E.ids, layered.E['kind'], strict=True))
    assert kinds['h_ctrl'] == 'hyper'
    assert kinds['e_stim'] == 'binary'
    frame = layered.attrs.table('edges', derived=True, backend='pandas')
    by_id = frame.set_index('edge_id')
    assert by_id.loc['h_ctrl', 'kind'] == 'hyper'
    assert bool(by_id.loc['h_ctrl', 'directed']) is False
    assert layered.E.select(kind='hyper').ids == ('h_ctrl',)
    assert S.edge_ref(layered, 'h_ctrl').kind == S.HYPER


# Zero-result, scalar and nested composition, on top of the ten.
def test_zero_result_views_stay_empty_through_every_read(flat):
    V = flat.view(nodes=[])
    assert len(V.N) == 0 and len(V.E) == 0
    assert V.shape == (0, 0)
    assert V.attrs.table('nodes', backend='pandas').shape[0] == 0
    assert V.materialize().shape == (0, 0)


def test_nested_views_only_restrict(flat):
    V = flat.view(nodes=['AA', 'BB'])
    W = V.view(nodes=['BB', 'CC'])
    assert W.N.ids == ('BB',)
    X = W.view(nodes=['AA', 'BB', 'CC'])
    assert X.N.ids == ('BB',), 'a nested view cannot broaden its parent'


# An edge entity on a multilayer graph can be named as an endpoint. It sits on
# the placeholder coordinate, which registering it must declare — the way a
# node added without ``layer=`` has it declared — or the next add_edges that
# names the entity fails resolving ('_', ...) against the aspect labels.
def test_an_edge_entity_is_a_valid_endpoint_on_a_layered_graph(layered):
    layered.add_edges(('A', ('ctrl',)), ('B', ('ctrl',)), edge_id='backing', as_entity=True)
    layered.add_edges(('backing', ('_',)), ('A', ('stim',)), edge_id='meta')
    sides = S.edge_sides(layered, 'meta')
    assert sides.source == {('backing', ('_',))} and sides.target == {('A', ('stim',))}
    assert layered.entity_kinds()['backing'] == 'edge'
    # The placeholder is a declaration detail, not a label a reader sees.
    assert layered.layers.aspect('cond').values == ('ctrl', 'stim')
    assert set(layered.view(edges=['meta']).E.ids) == {'meta', 'backing'}, 'referential closure'
