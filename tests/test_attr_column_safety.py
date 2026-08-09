"""What a caller may do with a column, and how long it is good for.

Two rules, and `D2` of cycle 003 settles the second.

**A column is read-only.** `FR-011`. A borrowing read hands back a window onto
the canonical store, and a write through that window would reach the graph with
no validation, no clock bump and no history entry — the one outcome worse than
copying. So it is refused, and it is refused on every path, so that a caller
never has to ask which one answered.

**A column is good until the next write to the graph.** `FR-010`. A view onto the
storage would otherwise mean that a cell write shows through it and a growth does
not, because a growth allocates a new array, and a caller cannot see which of the
two happened. The package therefore states one lifetime instead of two, and
`.copy()` is the documented way to hold values across a change.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet


def _graph(nodes: int = 8, *, edges: int = 4) -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes([{'node_id': f'v{i}', 'score': float(i)} for i in range(nodes)])
    if edges:
        graph.add_edges(
            [
                {'source': f'v{i}', 'target': f'v{i + 1}', 'edge_id': f'e{i}', 'w2': float(i)}
                for i in range(edges)
            ]
        )
    return graph


class TestAColumnIsReadOnly:
    """`FR-011`. The behaviour is stated, so it is tested."""

    def test_writing_into_a_node_column_is_refused(self):
        column = _graph().N['score']
        with pytest.raises(ValueError, match='read-only'):
            column[0] = 99.0

    def test_writing_into_an_edge_column_is_refused(self):
        column = _graph().E['w2']
        with pytest.raises(ValueError, match='read-only'):
            column[0] = 99.0

    def test_the_refusal_holds_on_the_gathering_path_too(self):
        """A graph with freed slots answers from a gather, and is refused alike."""
        graph = _graph()
        graph.remove_node('v3')
        assert graph._store.node_axis_contiguous is False
        with pytest.raises(ValueError, match='read-only'):
            graph.N['score'][0] = 99.0

    def test_the_refusal_holds_for_a_subsequence(self):
        graph = _graph()
        with pytest.raises(ValueError, match='read-only'):
            graph.N[0:4]['score'][0] = 99.0

    def test_the_refusal_holds_for_an_intrinsic_field(self):
        graph = _graph()
        for name in ('weight', 'directed', 'kind'):
            with pytest.raises(ValueError, match='read-only'):
                graph.E[name][0] = graph.E[name][0]

    def test_a_refused_write_leaves_the_graph_alone(self):
        graph = _graph()
        before = np.asarray(graph.N['score']).tolist()
        with pytest.raises(ValueError):
            graph.N['score'][0] = 99.0
        assert np.asarray(graph.N['score']).tolist() == before

    def test_reading_and_computing_still_work(self):
        column = _graph().N['score']
        assert float(column.sum()) == sum(float(i) for i in range(8))
        doubled = column * 2
        assert doubled.flags.writeable is True
        doubled[0] = 1.0


class TestACopyIsTheCallersOwn:
    """The one documented way to get an array that survives a change."""

    def test_a_copy_is_writable(self):
        column = _graph().N['score'].copy()
        column[0] = 99.0
        assert column[0] == 99.0

    def test_a_copy_does_not_reach_the_graph(self):
        graph = _graph()
        column = graph.N['score'].copy()
        column[0] = 99.0
        assert float(graph.N['score'][0]) == 0.0

    def test_a_copy_survives_every_kind_of_change(self):
        graph = _graph()
        snapshot = graph.N['score'].copy()
        graph.attrs.set_node_attrs('v0', score=-1.0)
        graph.add_nodes([{'node_id': f'w{i}', 'score': 5.0} for i in range(64)])
        graph.remove_node('v1')
        assert snapshot.tolist() == [float(i) for i in range(8)]


class TestTheStatedLifetime:
    """`FR-010` and `D2`: a column is good until the next write to the graph."""

    def test_a_column_holds_the_values_of_the_moment_it_was_read(self):
        graph = _graph()
        column = graph.N['score']
        assert column.tolist() == [float(i) for i in range(8)]

    def test_a_read_after_a_write_gives_the_new_values(self):
        graph = _graph()
        graph.attrs.set_node_attrs('v0', score=99.0)
        assert float(graph.N['score'][0]) == 99.0

    def test_the_lifetime_does_not_depend_on_whether_the_store_grew(self):
        """The point of `D2`. Both sequences end at the same stated place.

        A cell write lands in place and a growth allocates, so a live window
        would show the first and miss the second. The package promises neither:
        a column read before a write is stale after it, and a caller who needs
        the values copies. Both branches below are therefore the same rule, and
        neither asserts what a stale column shows.
        """
        for grow_first in (False, True):
            graph = _graph()
            column = graph.N['score']
            snapshot = column.copy()
            if grow_first:
                graph.add_nodes([{'node_id': f'w{i}', 'score': 0.0} for i in range(64)])
            graph.attrs.set_node_attrs('v0', score=99.0)

            # What the package promises after a write: the copy is untouched,
            # and a fresh read is right.
            assert snapshot.tolist() == [float(i) for i in range(8)]
            assert float(graph.N['score'][0]) == 99.0

    def test_a_stale_column_is_never_a_way_into_the_graph(self):
        """Whatever a stale column shows, it stays refused."""
        graph = _graph()
        column = graph.N['score']
        graph.add_nodes([{'node_id': 'w0', 'score': 0.0}])
        with pytest.raises(ValueError, match='read-only'):
            column[0] = 99.0
