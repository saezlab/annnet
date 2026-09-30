"""Coverage of the attribute write paths and the flexible-direction policies.

Written against the current surface: ``G.attrs.update`` for every batch,
``G.attrs.row(address, key)`` for one row, ``G.E.effective_weight`` for the
resolved weight, ``G.attrs.audit()`` for the audit. The semantics these tests
pin are the ones the old ``AttributesClass`` methods had where they were
correct; where the old route was lenient about an unknown key it is now
strict, and the test says so.
"""

from __future__ import annotations

import pytest

from annnet.core._Annotation import AttributesClass
from annnet.core.graph import AnnNet


def _toy() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    G.slices.add('s1')
    return G


def _annotation_snapshot(G):
    """Everything a no-op write must leave alone."""
    return (
        G.attrs.rows('nodes'),
        G.attrs.rows('edges'),
        G.attrs.rows('edge_slices'),
        G._state_clock(),
    )


# ── batch writes ───────────────────────────────────────────────────────


def test_node_batch_dict_input_writes_each_row() -> None:
    G = _toy()
    assert G.attrs.update('nodes', {'A': {'color': 'red'}, 'B': {'color': 'blue'}}) == 2
    assert G.attrs.row('nodes', 'A')['color'] == 'red'
    assert G.attrs.row('nodes', 'B')['color'] == 'blue'


def test_node_batch_accepts_iterable_of_pairs() -> None:
    G = _toy()
    G.attrs.update('nodes', [('A', {'color': 'red'}), ('B', {'color': 'blue'})])
    assert G.attrs.row('nodes', 'A')['color'] == 'red'


def test_node_batch_rejects_non_mapping_rows() -> None:
    G = _toy()
    with pytest.raises(TypeError, match='mapping'):
        G.attrs.update('nodes', {'A': 'not-a-dict'})


def test_node_batch_rejects_reserved_keys() -> None:
    G = _toy()
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.update('nodes', {'A': {'node_id': 'X'}})


def test_node_batch_is_a_noop_on_empty_input() -> None:
    G = _toy()
    before = _annotation_snapshot(G)
    assert G.attrs.update('nodes', {}) == 0
    assert G.attrs.update('nodes', []) == 0
    assert _annotation_snapshot(G) == before


def test_edge_batch_dict_input_writes_each_row() -> None:
    G = _toy()
    G.attrs.update('edges', {'e1': {'label': 'alpha'}, 'e2': {'label': 'beta'}})
    assert G.attrs.row('edges', 'e1')['label'] == 'alpha'
    assert G.attrs.row('edges', 'e2')['label'] == 'beta'


def test_edge_batch_accepts_iterable_of_pairs() -> None:
    G = _toy()
    G.attrs.update('edges', [('e1', {'label': 'alpha'})])
    assert G.attrs.row('edges', 'e1')['label'] == 'alpha'


def test_edge_batch_rejects_non_mapping_rows_and_reserved_keys() -> None:
    G = _toy()
    with pytest.raises(TypeError, match='mapping'):
        G.attrs.update('edges', {'e1': 'nope'})
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.update('edges', {'e1': {'source': 'X'}})


def test_an_empty_row_in_a_batch_writes_nothing() -> None:
    G = _toy()
    before = _annotation_snapshot(G)
    G.attrs.update('edges', {'e1': {}})
    G.attrs.update('nodes', {'A': {}})
    assert G.attrs.rows('nodes') == before[0]
    assert G.attrs.rows('edges') == before[1]


# ── edge-slice writes ──────────────────────────────────────────────────


def test_edge_slice_weight_is_coerced_to_float() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.update('edge_slices', {('s1', 'e1'): {'weight': 99}})
    out = G.attrs.row('edge_slices', ('s1', 'e1'))['weight']
    assert out == 99.0
    assert isinstance(out, float)


def test_edge_slice_non_weight_attrs_are_written() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.update('edge_slices', {('s1', 'e1'): {'confidence': 0.95}})
    assert G.attrs.row('edge_slices', ('s1', 'e1'))['confidence'] == 0.95


def test_an_edge_slice_row_appears_only_once_something_is_written() -> None:
    G = _toy()
    before = _annotation_snapshot(G)
    G.attrs.update('edge_slices', {('s1', 'e1'): {}})
    assert G.attrs.rows('edge_slices') == before[2]
    assert G.attrs.row('edge_slices', ('s1', 'e1')).get('weight') is None
    assert ('s1', 'e1') not in G._contextual.edge_slice_attrs


def test_edge_slice_batch_writes_every_present_attr() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1', 'e2'])
    G.attrs.update(
        'edge_slices', {('s1', 'e1'): {'weight': 5.0}, ('s1', 'e2'): {'confidence': 0.9}}
    )
    assert G.attrs.row('edge_slices', ('s1', 'e1'))['weight'] == 5.0
    assert G.attrs.row('edge_slices', ('s1', 'e2'))['confidence'] == 0.9


def test_an_override_needs_an_existing_slice_and_edge() -> None:
    G = _toy()
    with pytest.raises(KeyError, match='slice'):
        G.attrs.update('edge_slices', {('not-a-slice', 'e1'): {'weight': 1.0}})
    with pytest.raises(KeyError, match='edge'):
        G.attrs.update('edge_slices', {('s1', 'not-an-edge'): {'weight': 1.0}})
    # An override can be prepared before the edge is a member of the slice.
    G.attrs.update('edge_slices', {('s1', 'e1'): {'weight': 42.0}})
    assert G.attrs.row('edge_slices', ('s1', 'e1'))['weight'] == 42.0
    assert 'e1' not in G.slices.edges('s1')


# ── effective weight ───────────────────────────────────────────────────


def test_effective_weight_falls_back_to_the_stored_weight() -> None:
    G = _toy()
    assert G.E.effective_weight('e1') == 1.0
    assert G.E.effective_weight('e2', slice='s1') == 2.0


def test_effective_weight_uses_the_slice_override_when_present() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.update('edge_slices', {('s1', 'e1'): {'weight': 42.0}})
    assert G.E.effective_weight('e1', slice='s1') == 42.0
    assert G.E.effective_weight('e1') == 1.0, 'the active slice carries no override'
    G.slices.active = 's1'
    assert G.E.effective_weight('e1') == 42.0


def test_effective_weight_is_one_for_an_unknown_edge() -> None:
    G = _toy()
    assert G.E.effective_weight('no-such-edge') == 1.0


# ── audit ──────────────────────────────────────────────────────────────


def test_audit_returns_the_documented_shape_on_a_clean_graph() -> None:
    G = _toy()
    for out in (G.attrs.audit(), AttributesClass.audit_attributes(G)):
        for key in (
            'extra_node_rows',
            'extra_edge_rows',
            'missing_node_rows',
            'missing_edge_rows',
            'invalid_edge_slice_rows',
            'invalid_slice_rows',
            'invalid_node_layer_rows',
        ):
            assert isinstance(out[key], list)
            assert out[key] == []


def test_audit_finds_a_contextual_row_the_structure_no_longer_holds() -> None:
    G = _toy()
    G.slices.add_edges('s1', ['e1'])
    G.attrs.update('edge_slices', {('s1', 'e1'): {'weight': 3.0}})
    # Reach behind the API, as a broken loader might.
    G._contextual.edge_slice_attrs[('s1', 'ghost')] = {'weight': 1.0}
    G._contextual.touch('edge_slice_attrs')
    assert G.attrs.audit()['invalid_edge_slice_rows'] == [('s1', 'ghost')]


# ── one row ────────────────────────────────────────────────────────────


def test_a_row_by_id_reads_what_was_written() -> None:
    G = _toy()
    G.attrs.update('edges', {'e1': {'label': 'alpha'}})
    assert dict(G.attrs.row('edges', 'e1')) == {'label': 'alpha'}
    G.attrs.update('nodes', {'A': {'color': 'red'}})
    assert dict(G.attrs.row('nodes', 'A')) == {'color': 'red'}


def test_a_row_for_an_unknown_key_raises_rather_than_answering_empty() -> None:
    """The old getters answered ``{}`` for an unknown id; a typo now fails."""
    G = _toy()
    with pytest.raises(KeyError, match='unknown edge'):
        G.attrs.row('edges', 'no-such')
    with pytest.raises(KeyError, match='unknown node'):
        G.attrs.row('nodes', 'no-such')
    with pytest.raises(TypeError, match='edge id'):
        G.attrs.row('edges', 0)


def test_rows_of_many_keys() -> None:
    G = _toy()
    G.attrs.update('edges', {'e1': {'label': 'alpha'}, 'e2': {'label': 'beta'}})
    assert G.attrs.rows('edges') == {'e1': {'label': 'alpha'}, 'e2': {'label': 'beta'}}
    assert G.attrs.rows('edges', ['e1']) == {'e1': {'label': 'alpha'}}
    G.attrs.update('nodes', {'A': {'color': 'red'}, 'B': {'color': 'blue'}})
    assert set(G.attrs.rows('nodes')) == {'A', 'B', 'C'}
    assert G.attrs.rows('nodes', {'A'}) == {'A': {'color': 'red'}}


# ── columns and selections replace the old lookups ─────────────────────


def test_a_column_read_with_a_default_replaces_get_attr_from_edges() -> None:
    G = _toy()
    G.attrs.update('edges', {'e1': {'label': 'alpha'}})
    assert dict(zip(G.E.ids, G.E.column('label', default='??'), strict=True)) == {
        'e1': 'alpha',
        'e2': '??',
    }
    with pytest.raises(KeyError, match='no attribute'):
        G.E.column('not-a-column', default='??')


def test_a_selection_replaces_get_edges_by_attr() -> None:
    G = _toy()
    G.attrs.update('edges', {'e1': {'label': 'alpha'}, 'e2': {'label': 'alpha'}})
    assert set(G.E.select(label='alpha').ids) == {'e1', 'e2'}
    with pytest.raises(KeyError, match='unknown field'):
        G.E.select(**{'not-a-column': 'x'})


# ── graph attributes ──────────────────────────────────────────────────


def test_uns_is_the_graph_attribute_mapping() -> None:
    G = _toy()
    G.uns['study'] = 'demo'
    out = dict(G.uns)
    out['mutated'] = True
    assert 'mutated' not in G.uns
    assert G.uns['study'] == 'demo'


# ── composite node key ─────────────────────────────────────────────────


def test_composite_key_writes_and_indexes() -> None:
    G = _toy()
    G.set_node_key('name')
    G.attrs.update('nodes', {'A': {'name': 'alice'}})
    assert G.attrs.row('nodes', 'A')['name'] == 'alice'
    assert G._node_key_index == {('alice',): 'A'}


def test_composite_key_rejects_a_collision() -> None:
    G = _toy()
    G.set_node_key('name')
    G.attrs.update('nodes', {'A': {'name': 'alice'}})
    with pytest.raises(ValueError, match='Composite key collision'):
        G.attrs.update('nodes', {'B': {'name': 'alice'}})


def test_composite_key_batch_writes_all() -> None:
    G = _toy()
    G.set_node_key('name')
    G.attrs.update('nodes', {'A': {'name': 'alice'}, 'B': {'name': 'bob'}})
    assert G.attrs.row('nodes', 'A')['name'] == 'alice'
    assert G.attrs.row('nodes', 'B')['name'] == 'bob'
    assert G._node_key_index == {('alice',): 'A', ('bob',): 'B'}


# ── flexible edge direction policy ────────────────────────────────────
#
# The policy rewrites the member coefficients of the edge, so what it did is
# visible in the incidence column: +w on the source side, -w on the target side.


def _incidence_column(G, edge_id='e1'):
    """The incidence column of one edge, as ``{node_id: coefficient}``."""
    matrix = G.S.toarray()
    column = G.idx.edge_to_col(edge_id)
    return {n: float(matrix[G.idx.entity_to_row(n), column]) for n in ('A', 'B')}


def _flexible_graph(policy):
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0, flexible=policy)
    return G


def test_flexible_edge_scope_policy_orients_the_edge_from_the_attribute() -> None:
    policy = {'var': 'temperature', 'threshold': 10.0, 'scope': 'edge', 'above': 's->t'}

    above = _flexible_graph(policy)
    above.attrs.update('edges', {'e1': {'temperature': 20.0}})
    assert _incidence_column(above) == {'A': 1.0, 'B': -1.0}

    below = _flexible_graph(policy)
    below.attrs.update('edges', {'e1': {'temperature': 2.0}})
    assert _incidence_column(below) == {'A': -1.0, 'B': 1.0}


def test_flexible_node_scope_policy_orients_from_the_endpoint_attributes() -> None:
    policy = {'var': 'level', 'threshold': 5.0, 'scope': 'node', 'above': 's->t'}

    forward = _flexible_graph(policy)
    forward.attrs.update('nodes', {'A': {'level': 10.0}})
    forward.attrs.update('nodes', {'B': {'level': 2.0}})
    assert _incidence_column(forward) == {'A': 1.0, 'B': -1.0}

    backward = _flexible_graph(policy)
    backward.attrs.update('nodes', {'A': {'level': 2.0}})
    backward.attrs.update('nodes', {'B': {'level': 10.0}})
    assert _incidence_column(backward) == {'A': -1.0, 'B': 1.0}


def test_flexible_edge_tie_keep_leaves_the_orientation_alone() -> None:
    G = _flexible_graph({'var': 'x', 'threshold': 5.0, 'scope': 'edge', 'tie': 'keep'})
    before = _incidence_column(G)
    G.attrs.update('edges', {'e1': {'x': 5.0}})
    assert _incidence_column(G) == before == {'A': 1.0, 'B': -1.0}


def test_flexible_edge_tie_undirected_puts_the_weight_on_both_sides() -> None:
    G = _flexible_graph({'var': 'x', 'threshold': 5.0, 'scope': 'edge', 'tie': 'undirected'})
    G.attrs.update('edges', {'e1': {'x': 5.0}})
    assert _incidence_column(G) == {'A': 1.0, 'B': 1.0}


def test_flexible_policy_applies_on_the_batch_edge_write() -> None:
    policy = {'var': 'x', 'threshold': 5.0, 'scope': 'edge'}

    above = _flexible_graph(policy)
    above.attrs.update('edges', {'e1': {'x': 10.0}})
    assert _incidence_column(above) == {'A': 1.0, 'B': -1.0}

    below = _flexible_graph(policy)
    below.attrs.update('edges', {'e1': {'x': 2.0}})
    assert _incidence_column(below) == {'A': -1.0, 'B': 1.0}


def test_flexible_policy_applies_on_the_batch_node_write() -> None:
    policy = {'var': 'level', 'threshold': 5.0, 'scope': 'node'}

    forward = _flexible_graph(policy)
    forward.attrs.update('nodes', {'A': {'level': 10.0}, 'B': {'level': 2.0}})
    assert _incidence_column(forward) == {'A': 1.0, 'B': -1.0}

    backward = _flexible_graph(policy)
    backward.attrs.update('nodes', {'A': {'level': 2.0}, 'B': {'level': 10.0}})
    assert _incidence_column(backward) == {'A': -1.0, 'B': 1.0}


def test_flexible_policy_applies_through_a_column_write() -> None:
    policy = {'var': 'level', 'threshold': 5.0, 'scope': 'node'}
    G = _flexible_graph(policy)
    G.N['level'] = [2.0, 10.0]
    assert _incidence_column(G) == {'A': -1.0, 'B': 1.0}
