"""The two public removes of an edge must leave the same graph.

``remove_edge`` takes one edge and ``remove_edges`` takes a set of them, and one
edge is a set of one: the graph a single remove leaves has to be the graph the
set of one leaves, down to both attribute tables, the slice memberships and the
per-slice weight cache.

The edge-by-slice table takes its removal buffered, so what these also pin is
the buffer: a row is gone from the first read after the remove, a table written
whole owes nothing against the table it replaced, and a removal recorded before
the id set of a table is built is still true of that set when it is.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def _graph():
    """Four edges over four vertices, two of them weighted in a second slice."""
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B', 'C', 'D'])
    G.add_edges('A', 'B', edge_id='e1', weight=1.0)
    G.add_edges('B', 'C', edge_id='e2', weight=2.0)
    G.add_edges('C', 'D', edge_id='e3', weight=3.0)
    G.add_edges('D', 'A', edge_id='e4', weight=4.0)
    G.slices.add('treated')
    G.attrs.set_edge_slice_attrs('treated', 'e1', weight=99.0)
    G.attrs.set_edge_slice_attrs('treated', 'e2', weight=98.0)
    return G


def _state(G):
    """Everything a removal touches, in a form two graphs can be compared by."""
    return {
        'edges': sorted(G.edges()),
        'vertices': sorted(G.vertices()),
        'edge_rows': sorted(r['edge_id'] for r in G._edge_table.to_dicts()),
        'edge_slice_rows': sorted(
            (r['slice_id'], r['edge_id']) for r in G.edge_slice_attributes.to_dicts()
        ),
        'slice_edges': {sid: sorted(d['edges']) for sid, d in G._slices.items()},
        'slice_weights': {sid: dict(d) for sid, d in G.slice_edge_weights.items() if d},
    }


def test_the_two_removes_leave_the_same_graph():
    single, batched = _graph(), _graph()
    single.remove_edge('e1')
    batched.remove_edges(['e1'])
    assert _state(single) == _state(batched)


def test_a_remove_takes_the_edge_out_of_the_edge_slice_table():
    G = _graph()
    G.remove_edge('e1')
    rows = {(r['slice_id'], r['edge_id']) for r in G.edge_slice_attributes.to_dicts()}
    assert ('treated', 'e1') not in rows
    assert ('treated', 'e2') in rows


def test_a_remove_takes_the_edge_out_of_the_slice_weight_cache():
    G = _graph()
    assert G.slice_edge_weights['treated']['e1'] == 99.0
    G.remove_edge('e1')
    assert 'e1' not in G.slice_edge_weights['treated']
    assert G.slice_edge_weights['treated']['e2'] == 98.0


def test_a_table_written_whole_owes_nothing_against_the_one_it_replaced():
    G = _graph()
    G.remove_edge('e1')
    replacement = _graph().edge_slice_attributes
    G.edge_slice_attributes = replacement
    rows = {(r['slice_id'], r['edge_id']) for r in G.edge_slice_attributes.to_dicts()}
    assert ('treated', 'e1') in rows


def test_a_removal_recorded_before_the_id_set_is_built_survives_the_build():
    G = _graph()
    # Nothing has asked for the id set of the edge table yet, and the removal
    # must not have to build one to be recorded against it.
    G._edge_attr_ids = None
    G.remove_edge('e1')
    G.add_edges('A', 'C', edge_id='e5')
    ids = sorted(r['edge_id'] for r in G._edge_table.to_dicts())
    assert ids == ['e2', 'e3', 'e4', 'e5']


def test_an_unknown_edge_is_still_a_key_error():
    G = _graph()
    with pytest.raises(KeyError):
        G.remove_edge('nope')
    assert sorted(G.edges()) == ['e1', 'e2', 'e3', 'e4']
