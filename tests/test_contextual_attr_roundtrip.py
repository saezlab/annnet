"""Both contextual attribute levels survive a round trip, whichever store holds them.

`SC-007`, for the two levels decision `D4` of cycle 003 is about: the attributes
of a slice, and the attributes of one edge inside one slice. `D4` records that
they move to the mapping the canonical store already holds, and that the move is
a later cycle's work. These tests are written against the *behaviour* and not
against the storage, so they hold on either side of that move and will say so if
the move loses anything.

This module deliberately does not share a file with the round-trip tests of
User Story 7. The two stories are independent and either may land alone.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from annnet import AnnNet
from annnet.io.annnet_format import read as annnet_read
from annnet.io.annnet_format import write as annnet_write


def _graph() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C'])
    graph.add_edges('A', 'B', edge_id='e1', slice='s1')
    graph.add_edges('B', 'C', edge_id='e2', slice='s2')
    graph.attrs.set_slice_attrs('s1', label='first', score=1.5)
    graph.attrs.set_slice_attrs('s2', label='second')
    graph.attrs.set_edge_slice_attrs('s1', 'e1', confidence=0.75, note='seen')
    graph.attrs.set_edge_slice_attrs('s2', 'e2', confidence=0.25)
    return graph


def _round_trip(graph, directory) -> AnnNet:
    path = Path(directory) / 'g.annnet'
    annnet_write(graph, path)
    return annnet_read(path)


@pytest.fixture
def restored():
    graph = _graph()
    with tempfile.TemporaryDirectory() as tmp:
        yield graph, _round_trip(graph, tmp)


class TestTheSliceLevel:
    def test_every_slice_keeps_its_attributes(self, restored):
        _before, after = restored
        assert after.attrs.get_slice_attr('s1', 'label') == 'first'
        assert after.attrs.get_slice_attr('s1', 'score') == 1.5
        assert after.attrs.get_slice_attr('s2', 'label') == 'second'

    def test_a_slice_that_carried_nothing_gains_nothing(self, restored):
        _before, after = restored
        assert after.attrs.get_slice_attr('s2', 'score') is None


class TestTheEdgeSliceLevel:
    def test_every_pair_keeps_its_attributes(self, restored):
        _before, after = restored
        assert after.attrs.get_edge_slice_attr('s1', 'e1', 'confidence') == 0.75
        assert after.attrs.get_edge_slice_attr('s1', 'e1', 'note') == 'seen'
        assert after.attrs.get_edge_slice_attr('s2', 'e2', 'confidence') == 0.25

    def test_a_pair_that_carried_nothing_gains_nothing(self, restored):
        _before, after = restored
        assert after.attrs.get_edge_slice_attr('s2', 'e1', 'confidence') is None

    def test_the_level_is_the_pair_and_not_the_edge(self, restored):
        """An edge that carries a value in one slice carries none in another."""
        _before, after = restored
        assert after.attrs.edge_slice('s1', 'e1')
        assert after.attrs.edge_slice('s2', 'e1') == {}


def test_the_two_levels_come_back_together(restored):
    before, after = restored
    for slice_id, name in (('s1', 'label'), ('s1', 'score'), ('s2', 'label'), ('s2', 'score')):
        assert after.attrs.get_slice_attr(slice_id, name) == before.attrs.get_slice_attr(
            slice_id, name
        )
    for slice_id, edge_id in (('s1', 'e1'), ('s2', 'e2'), ('s1', 'e2')):
        assert after.attrs.edge_slice(slice_id, edge_id) == before.attrs.edge_slice(
            slice_id, edge_id
        )


def test_a_graph_that_carries_neither_level_still_round_trips():
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B'])
    graph.add_edges('A', 'B', edge_id='e1')
    with tempfile.TemporaryDirectory() as tmp:
        after = _round_trip(graph, tmp)
    assert after.attrs.get_slice_attr('default', 'label') is None
    assert after.attrs.edge_slice('default', 'e1') == {}


def test_a_per_slice_weight_override_survives_beside_them():
    """The override is read by a matrix build, so it is structural as well."""
    graph = _graph()
    graph.attrs.set_slice_edge_weight('s1', 'e1', 4.0)
    with tempfile.TemporaryDirectory() as tmp:
        after = _round_trip(graph, tmp)
    assert after.attrs.get_edge_slice_attr('s1', 'e1', 'weight') == 4.0
