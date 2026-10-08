"""One resolver, closed and open boundaries, live read-only views.

The reference enumerator at the top is deliberately naive: it re-derives the
closed and open selections from the endpoints alone, without the resolver.
Every fixture family — flat, multilayer, hypergraph, edge-entity — is checked
against it, and the materialized graph is checked against the view.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pytest

from annnet import AnnNet
from annnet.core import _structure as S
from annnet.core._attribute_api import ReadOnlyViewError

# ---------------------------------------------------------------------------
# the reference
# ---------------------------------------------------------------------------


def endpoints(G, eid):
    """Every endpoint key of an edge, whatever the graph's identity form."""
    store = G._store
    sides = store.endpoints(store.edge_slot(eid))
    return set(sides.source | sides.target)


def is_edge_entity(G, key):
    slot = G._store.entity_slot(key)
    return slot is not None and int(G._store.entity_kind[slot]) == 1


def reference_closed(G, selected_keys, candidate_edges):
    """Edges whose every node endpoint is selected and whose edge-entity
    endpoints refer to kept edges (a fixpoint)."""
    kept = set()
    while True:
        added = False
        for eid in candidate_edges:
            if eid in kept:
                continue
            ok = True
            for key in endpoints(G, eid):
                if is_edge_entity(G, key):
                    if key[0] not in kept:
                        ok = False
                elif key not in selected_keys:
                    ok = False
            if ok:
                kept.add(eid)
                added = True
        if not added:
            return kept


def reference_open(G, seeds, candidate_edges):
    """Edges touching a seed placement, with their full endpoints.

    An edge-entity endpoint needs its backing edge to be represented: it is
    brought along when the candidates hold it, and the dependent edge is
    dropped when they do not.
    """
    candidates = set(candidate_edges)
    kept = {
        eid
        for eid in candidate_edges
        if any((not is_edge_entity(G, key)) and key in seeds for key in endpoints(G, eid))
    }
    changed = True
    while changed:
        changed = False
        for eid in list(kept):
            for key in endpoints(G, eid):
                if is_edge_entity(G, key) and key[0] not in kept:
                    if key[0] in candidates:
                        kept.add(key[0])
                    else:
                        kept.discard(eid)
                    changed = True
    keys = set(seeds)
    for eid in kept:
        keys.update(key for key in endpoints(G, eid) if not is_edge_entity(G, key))
    return kept, keys


def row_order(G, keys):
    wanted = set(keys)
    return [key for key in S.node_keys(G) if key in wanted]


def edge_order(G, ids):
    wanted = set(ids)
    return [eid for eid in S.edge_ids(G) if eid in wanted]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c', 'd', 'iso'])
    G.attrs.update('nodes', {'a': {'g': 'x'}, 'b': {'g': 'x'}, 'c': {'g': 'y'}, 'd': {'g': 'y'}})
    G.add_edges('a', 'b', edge_id='ab')
    G.add_edges('b', 'c', edge_id='bc')
    G.add_edges('c', 'c', edge_id='loop')
    G.add_edges([{'members': ['a', 'b', 'c'], 'edge_id': 'h3'}])
    G.add_edges('c', 'd', edge_id='cd', directed=False)
    G.slices.add('s1', edges=['ab', 'bc'])
    G.slices.add('s2', edges=['bc', 'cd'])
    return G


@pytest.fixture
def layered():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = AnnNet(directed=True, aspects={'t': ['t1', 't2', 't3']})
        G.layers.set_ordered('t')
        G.layers.place(['a', 'b', 'c'], [('t1',), ('t2',)])
        G.add_nodes('lonely', layer=('t3',))
        G.add_edges(('a', ('t1',)), ('b', ('t1',)), edge_id='ab1')
        G.add_edges(('a', ('t2',)), ('b', ('t2',)), edge_id='ab2')
        G.add_edges(('a', ('t1',)), ('a', ('t2',)), edge_id='couple_a')
        G.add_edges(('b', ('t1',)), ('c', ('t2',)), edge_id='cross_bc')
        G.add_edges(
            [{'head': [('a', ('t2',))], 'tail': [('b', ('t2',)), ('c', ('t2',))], 'edge_id': 'h2'}]
        )
    return G


@pytest.fixture
def entity():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.add_edges('a', 'b', edge_id='ab', as_entity=True)
    G.add_edges('ab', 'c', edge_id='meta')
    G.add_edges('b', 'c', edge_id='bc')
    return G


# ---------------------------------------------------------------------------
# closed and open, against the reference
# ---------------------------------------------------------------------------


def check_agreement(G, V, selected_keys, candidate_edges, boundary):
    if boundary == 'closed':
        kept = reference_closed(G, selected_keys, candidate_edges)
        keys = set(selected_keys)
    else:
        kept, keys = reference_open(G, selected_keys, candidate_edges)
    assert list(V.E.ids) == edge_order(G, kept)
    assert list(V.supra_nodes()) == row_order(G, keys)
    H = V.materialize()
    assert list(H.E) == list(V.E.ids)
    assert list(H.supra_nodes()) == list(V.supra_nodes())
    assert list(H.N) == list(V.N.ids)
    for eid in H.E:
        assert endpoints(H, eid) == endpoints(G, eid)


@pytest.mark.parametrize('boundary', ['closed', 'open'])
def test_random_node_filters_agree_with_the_reference(flat, layered, entity, boundary):
    rng = random.Random(4)
    for G in (flat, layered, entity):
        keys = S.node_keys(G)
        for _ in range(25):
            chosen = [key for key in keys if rng.random() < 0.5]
            V = G.view(nodes=chosen, boundary=boundary)
            check_agreement(G, V, set(chosen), S.edge_ids(G), boundary)


@pytest.mark.parametrize('boundary', ['closed', 'open'])
def test_random_node_and_edge_filters_agree_with_the_reference(flat, layered, entity, boundary):
    rng = random.Random(11)
    for G in (flat, layered, entity):
        keys = S.node_keys(G)
        edges = S.edge_ids(G)
        for _ in range(25):
            chosen = [key for key in keys if rng.random() < 0.6]
            chosen_edges = [eid for eid in edges if rng.random() < 0.6]
            V = G.view(nodes=chosen, edges=chosen_edges, boundary=boundary)
            check_agreement(G, V, set(chosen), chosen_edges, boundary)


def test_hyperedges_are_kept_whole_or_dropped(flat):
    assert 'h3' in flat.view(nodes=['a', 'b', 'c']).E.ids
    assert 'h3' not in flat.view(nodes=['a', 'b']).E.ids
    V = flat.view(nodes=['a'], boundary='open')
    assert 'h3' in V.E.ids
    assert V.N.ids == ('a', 'b', 'c'), 'open expands to the whole hyperedge'
    H = V.materialize()
    assert endpoints(H, 'h3') == endpoints(flat, 'h3')


def test_self_loop_has_one_endpoint(flat):
    assert flat.view(nodes=['c']).E.ids == ('loop',)
    assert flat.view(nodes=['c'], boundary='open').E.ids == ('bc', 'loop', 'h3', 'cd')
    assert flat.view(nodes=['c']).degree('c') == 1


def test_isolated_nodes_survive_node_filters(flat):
    V = flat.view(nodes=['iso', 'a'])
    assert V.N.ids == ('a', 'iso')
    assert V.E.ids == ()
    assert list(V.materialize().N) == ['a', 'iso']


def test_edge_only_filters_keep_endpoints_only(flat):
    V = flat.view(edges=['cd'])
    assert V.E.ids == ('cd',)
    assert V.N.ids == ('c', 'd')
    assert 'iso' not in V


def test_layer_only_filters_keep_isolated_placements(layered):
    V = layered.view(layers=('t3',))
    assert V.supra_nodes() == [('lonely', ('t3',))]
    assert V.E.ids == ()
    W = layered.view(layers=layered.layers.where(t__lte='t1'))
    assert set(W.supra_nodes()) == {('a', ('t1',)), ('b', ('t1',)), ('c', ('t1',))}
    assert W.E.ids == ('ab1',), 'a crossing edge is out of a closed window'
    X = layered.view(layers=layered.layers.where(t__lte='t1'), boundary='open')
    assert set(X.E.ids) == {'ab1', 'couple_a', 'cross_bc'}
    assert ('a', ('t2',)) in X and ('c', ('t2',)) in X
    assert X.summary()['expanded'] == {'nodes': 2, 'edges': 0}


def test_slice_only_views_are_membership_not_induction(flat):
    flat.slices.add('n_only', nodes=['a', 'b'])
    V = flat.view(slices='n_only')
    assert V.N.ids == ('a', 'b')
    assert V.E.ids == (), 'a slice is a membership set; it does not induce edges'
    W = flat.view(slices=['s1', 's2'])
    assert set(W.E.ids) == {'ab', 'bc', 'cd'}, 'several slices mean their union'
    both = flat.view(slices=flat.slices.intersect(['s1', 's2']))
    assert both.E.ids == ('bc',)
    assert both.N.ids == ('b', 'c')
    assert list(both.slices.list()) == ['default', 's1', 's2', 'n_only']
    with pytest.raises(KeyError, match='unknown slice'):
        flat.view(slices='nope')


def test_node_and_edge_filters_both_apply(flat):
    V = flat.view(nodes=['a', 'b', 'c'], edges=['ab', 'cd'])
    assert V.E.ids == ('ab',)
    H = V.materialize()
    assert list(H.E) == ['ab']
    assert list(H.N) == ['a', 'b', 'c']


def test_bare_ids_and_explicit_placements(layered):
    V = layered.view(nodes=['a', 'b'])
    assert set(V.supra_nodes()) == {('a', ('t1',)), ('b', ('t1',)), ('a', ('t2',)), ('b', ('t2',))}
    assert set(V.E.ids) == {'ab1', 'ab2', 'couple_a'}
    W = layered.view(nodes=[('a', ('t1',)), ('b', ('t1',))])
    assert W.E.ids == ('ab1',)
    with pytest.raises(KeyError, match='not placed'):
        layered.view(nodes=[('a', ('t3',))])
    with pytest.raises(KeyError, match='unknown node'):
        layered.view(nodes=['zz'])


def test_edge_entity_closure(entity):
    # The edge that names the edge entity needs its backing edge.
    V = entity.view(edges=['meta'])
    assert V.E.ids == ('ab', 'meta'), 'edge-only: the backing edge is retained'
    assert V.N.ids == ('a', 'b', 'c')
    W = entity.view(nodes=['c', 'b'])
    assert W.E.ids == ('bc',), 'closed: the entity is not selected, so meta is dropped'
    X = entity.view(nodes=['c'], boundary='open')
    assert set(X.E.ids) == {'ab', 'meta', 'bc'}
    H = X.materialize()
    assert set(H.E) == {'ab', 'meta', 'bc'}
    assert S.entity_ref(H, ('ab', ('_',))).kind == S.EDGE_ENTITY
    assert H.validate() == []


def test_half_edge_uses_its_actual_endpoints():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    # A one-sided hyperedge, as the SBML reader builds a boundary reaction.
    G.add_edges([{'head': ['a'], 'tail': [], 'edge_id': 'half'}])
    G.add_edges('a', 'b', edge_id='ab')
    assert G._store.is_boundary(G._store.edge_slot('half'))
    assert G.view(nodes=['a']).E.ids == ('half',)
    assert G.view(nodes=['b']).E.ids == ()
    assert G.view(nodes=['a'], boundary='open').N.ids == ('a', 'b')
    H = G.view(nodes=['a']).materialize()
    assert list(H.E) == ['half']
    assert H._store.is_boundary(H._store.edge_slot('half'))


# ---------------------------------------------------------------------------
# nesting, liveness and read-only
# ---------------------------------------------------------------------------


def test_nested_views_intersect_every_predicate(flat):
    V = flat.view(nodes=flat.N.select(g='x'))
    W = V.view(nodes=['b', 'c'])
    assert W.N.ids == ('b',)
    X = W.view(slices='s2')
    assert X.N.ids == ('b',) and X.E.ids == ()
    Y = V.view(edges=V.E.select(directed=True))
    assert Y.E.ids == ('ab',)
    assert V.view().N.ids == V.N.ids
    with pytest.raises(ValueError, match='different graph'):
        other = AnnNet()
        other.add_nodes('q')
        flat.view(nodes=other.N.select())


def test_open_boundary_cannot_leave_the_parent(flat):
    V = flat.view(nodes=['a', 'b'])
    W = V.view(nodes=['a'], boundary='open')
    assert W.N.ids == ('a', 'b')
    assert W.E.ids == ('ab',), 'h3 needs c, which the parent does not hold'


def test_views_are_live(flat):
    V = flat.view(nodes=flat.N.select(g='x'))
    assert V.N.ids == ('a', 'b')
    flat.attrs.update('nodes', {'c': {'g': 'x'}})
    assert V.N.ids == ('a', 'b', 'c')
    assert V.E.ids == ('ab', 'bc', 'loop', 'h3')
    flat.remove_nodes('b')
    assert V.N.ids == ('a', 'c')
    flat.add_nodes('e', g='x')
    # Row order is the parent's order: e took the slot b freed.
    assert V.N.ids == ('a', 'e', 'c')
    assert list(flat.N) == ['a', 'e', 'c', 'd', 'iso']
    fixed = flat.view(nodes=['a', 'c'])
    flat.remove_nodes('c')
    assert fixed.N.ids == ('a',)
    flat.add_nodes('c')
    assert fixed.N.ids == ('a', 'c'), 'an explicit id names the id, not a slot'


def test_views_follow_every_clock(flat):
    by_slice = flat.view(slices='s1')
    assert set(by_slice.E.ids) == {'ab', 'bc'}
    flat.slices.add_edges('s1', ['cd'])
    assert set(by_slice.E.ids) == {'ab', 'bc', 'cd'}
    rows = flat.view(nodes=flat.attrs.select('nodes', g='y').project('nodes'))
    assert rows.N.ids == ('c', 'd')
    flat.attrs.replace('nodes', flat.attrs.table('nodes', backend='pandas').assign(g='y'))
    assert rows.N.ids == ('a', 'b', 'c', 'd', 'iso')


def test_a_view_refuses_every_write_route(flat):
    V = flat.view(nodes=['a', 'b'])
    with pytest.raises(ReadOnlyViewError, match='materialize'):
        V.N['g'] = 'z'
    with pytest.raises(ReadOnlyViewError):
        V.E['weight'] = 2.0
    with pytest.raises(ReadOnlyViewError):
        V.attrs.update('nodes', {'a': {'g': 'z'}})
    with pytest.raises(ReadOnlyViewError):
        V.attrs.update('nodes', {'a': {'g': 'z'}})
    with pytest.raises(ReadOnlyViewError):
        V.attrs.replace('nodes', flat.attrs.nodes)
    with pytest.raises(ReadOnlyViewError):
        V.attrs.replace('nodes', flat.attrs.nodes)
    with pytest.raises(ReadOnlyViewError):
        V.attrs.delete('nodes', keys=['a'])
    with pytest.raises(ReadOnlyViewError):
        V.uns['x'] = 1
    flat.uns['nested'] = {'k': [1, 2]}
    with pytest.raises(ReadOnlyViewError):
        V.uns['nested']['k'] = 3
    assert V.uns['nested']['k'] == (1, 2)
    with pytest.raises(AttributeError, match='materialize'):
        V.add_nodes('z')
    with pytest.raises(AttributeError, match='materialize'):
        V.slices.add('z')
    with pytest.raises(AttributeError, match='materialize'):
        V.layers.set_aspects(['x'])
    with pytest.raises(AttributeError, match='read-only'):
        _ = V.ops
    with pytest.raises(ReadOnlyViewError):
        V.attrs.backend = 'pandas'
    record = V.N.at('a')
    record.attrs['g'] = 'z'
    assert flat.attrs.row('nodes', 'a')['g'] == 'x', 'a record hands back a detached copy'
    column = V.N['g']
    with pytest.raises(ValueError):
        column[0] = 'z'
    assert flat.attrs.row('nodes', 'a')['g'] == 'x'
    for name in ('obs', 'var', 'nodes_df', 'edges_df', 'node_count', 'edge_count', 'subview', 'X'):
        with pytest.raises(AttributeError, match='use'):
            getattr(V, name)


def test_parent_stays_editable_and_the_copy_is_independent(flat):
    V = flat.view(nodes=['a', 'b'])
    H = V.materialize()
    H.add_nodes('new')
    H.attrs.update('nodes', {'a': {'g': 'changed'}})
    assert 'new' not in flat.N
    assert flat.attrs.row('nodes', 'a')['g'] == 'x'
    flat.attrs.update('nodes', {'b': {'g': 'parent'}})
    assert H.attrs.row('nodes', 'b')['g'] == 'x'
    assert H.uns['selection']['nodes'] == 2


# ---------------------------------------------------------------------------
# what the view exposes
# ---------------------------------------------------------------------------


def test_scoped_attrs_and_tables(flat):
    flat.attrs.update('edge_slices', {('s1', 'ab'): {'w': 1}, ('s2', 'cd'): {'w': 2}})
    flat.attrs.update('slices', {'s2': {'role': 'fit'}})
    V = flat.view(nodes=['a', 'b'], slices='s1')
    assert V.attrs.rows('edges') == {'ab': {}}
    assert V.attrs.select('edge_slices').keys == (('s1', 'ab'),)
    assert list(V.attrs.table('slices', backend='pandas')['slice_id']) == ['s1']
    with pytest.raises(KeyError, match='outside this view'):
        V.attrs.row('slices', 's2')
    frame = V.attrs.table('edges', derived=True, backend='pandas')
    assert frame['edge_id'].tolist() == ['ab']
    assert V.attrs.table('nodes', derived=True, limit=1, backend='pandas').shape[0] == 1
    assert len(V.N) == 2, 'a preview limit changes no membership'
    assert V.attrs.schema('nodes').rows == 2
    assert V.slices.edges('s1') == {'ab'}
    assert V.slices.nodes('s1') == {'a', 'b'}


def test_matrices_and_traversal_see_only_the_view(flat):
    V = flat.view(nodes=['a', 'b', 'c'])
    # Each named matrix keeps its documented projection: B holds the binary
    # edges alone, S every structural edge, H the hyperedges.
    assert V.B.shape == (3, 3)
    assert V.S.shape == (3, 4)
    assert V.H.shape == (3, 1)
    labels = V.idx
    assert [labels.col_to_edge(j) for j in range(4)] == ['ab', 'bc', 'loop', 'h3']
    assert [labels.row_to_entity(i) for i in range(3)] == ['a', 'b', 'c']
    assert V.shape == (3, 4)
    assert V.supra_shape == (3, 4)
    assert set(V.neighbors('c')) <= {'a', 'b', 'c'}
    assert 'd' not in V.neighbors('c')
    assert 'd' in flat.neighbors('c')
    assert V.degree('c') == 3
    assert V.degree('d') == 0
    assert V.has_edge('c', 'd') == (False, [])
    assert V.has_edge(edge_id='cd') is False


def test_view_b_rows_include_edge_entities(entity):
    V = entity.view(edges=['meta'])
    assert V.B.shape == (4, 2)
    rows = [V.idx.row_to_entity(i) for i in range(4)]
    assert rows == ['a', 'b', 'c', 'ab'], 'row order is the parent row order'
    assert [V.idx.col_to_edge(j) for j in range(2)] == ['ab', 'meta']


def test_consistent_resolution_within_one_operation(flat, monkeypatch):
    V = flat.view(nodes=flat.N.select(g='x'))
    calls = {'n': 0}
    original = flat._state_clock

    def ticking():
        calls['n'] += 1
        return original() if calls['n'] > 40 else (calls['n'],) + original()[1:]

    monkeypatch.setattr(flat, '_state_clock', ticking)
    with pytest.raises(RuntimeError, match='changed while'):
        V.materialize()


def test_masks_and_generators_as_view_inputs(flat):
    mask = np.array([True, True, False, False, False])
    assert flat.view(nodes=mask).N.ids == ('a', 'b')
    assert flat.view(nodes=(n for n in ['b', 'a'])).N.ids == ('a', 'b')
    assert flat.view(edges='ab').E.ids == ('ab',)
    with pytest.raises(TypeError, match='project'):
        flat.view(edges=flat.attrs.select('edge_slices', slice_id='s1'))
    assert flat.view(
        edges=flat.attrs.select('edge_slices', slice_id='s1').project('edges')
    ).E.ids == ('ab', 'bc')
