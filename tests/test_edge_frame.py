"""Building a graph from an edge table, and what happens when an id is taken.

Two defects motivated this, and neither is verbosity. A **positional id** means
re-sorting the input renames every edge, so two runs over the same data disagree
about what anything is called. A **side list** of ids exists only because nothing
on the graph could answer *which edges came from this table* — which slices do.
"""

from __future__ import annotations

import pytest

import annnet as an
from annnet import ON_CONFLICT, EdgeIdConflict, edge_id_for

ROWS = [
    {'source': 'TF1', 'target': 'G1', 'effect': 1, 'resource': 'demo'},
    {'source': 'TF1', 'target': 'G2', 'effect': -1, 'resource': 'demo'},
]


@pytest.fixture
def G():
    return an.from_edge_frame(ROWS, sign='effect', slice='prior')


class TestDerivedIds:
    """An id says what the edge is, not where it sat in the file."""

    def test_the_same_payload_gives_the_same_id(self):
        assert edge_id_for({'source': 'A', 'target': 'B'}) == edge_id_for(
            {'target': 'B', 'source': 'A'}
        )

    def test_a_changed_value_gives_a_different_id(self):
        assert edge_id_for({'source': 'A', 'target': 'B', 'sign': 1}) != edge_id_for(
            {'source': 'A', 'target': 'B', 'sign': -1}
        )

    def test_re_sorting_the_table_renames_nothing(self):
        """The whole point. A positional id fails this."""
        first = an.from_edge_frame(ROWS, sign='effect', slice='prior')
        second = an.from_edge_frame(list(reversed(ROWS)), sign='effect', slice='prior')
        assert sorted(first.edges()) == sorted(second.edges())

    def test_the_prefix_is_readable_in_the_id(self, G):
        assert all(edge_id.startswith('e:') for edge_id in G.edges())

    def test_a_prefix_can_be_named(self):
        graph = an.from_edge_frame(ROWS, sign='effect', id_prefix='prior')
        assert all(edge_id.startswith('prior:') for edge_id in graph.edges())

    def test_an_id_column_is_used_when_given(self):
        rows = [{'source': 'A', 'target': 'B', 'eid': 'mine'}]
        graph = an.from_edge_frame(rows, edge_id='eid')
        assert list(graph.edges()) == ['mine']

    def test_the_returned_ids_are_in_row_order(self, G):
        ids = an.add_edges_from_frame(
            G, [{'source': 'X', 'target': 'Y'}, {'source': 'Y', 'target': 'Z'}]
        )
        assert len(ids) == 2
        assert G.get_edge(ids[0]).source_id == 'X'
        assert G.get_edge(ids[1]).source_id == 'Y'


class TestWhatLands:
    def test_the_edges_are_there(self, G):
        assert len(list(G.edges())) == 2

    def test_the_slice_answers_which_edges_came_from_the_table(self, G):
        """The side list this replaces existed only because nothing else could."""
        assert G.slices.edges('prior') == set(G.edges())

    def test_the_sign_column_lands_under_the_reserved_name(self, G):
        signs = {G.attrs.get_edge_attrs(e).get('sign') for e in G.edges()}
        assert signs == {1, -1}

    def test_other_columns_are_carried(self, G):
        one = next(iter(G.edges()))
        assert G.attrs.get_edge_attrs(one)['resource'] == 'demo'

    def test_attrs_restricts_what_is_carried(self):
        graph = an.from_edge_frame(ROWS, sign='effect', attrs=[])
        one = next(iter(graph.edges()))
        assert 'resource' not in graph.attrs.get_edge_attrs(one)

    def test_a_missing_column_raises(self):
        with pytest.raises(KeyError, match='is not a column'):
            an.from_edge_frame(ROWS, sign='absent')

    def test_an_empty_table_adds_nothing(self):
        graph = an.Graph()
        assert an.add_edges_from_frame(graph, []) == []

    def test_aspects_are_declared_before_the_edges_land(self):
        graph = an.from_edge_frame(ROWS, aspects={'cond': ['a', 'b']})
        assert graph.layers.list_aspects() == ('cond',)


class TestConflictPolicy:
    """A batch that half-lands is worse than one that does not."""

    def test_the_policies_are_declared(self):
        assert ON_CONFLICT == ('error', 'skip', 'rename', 'replace')

    def test_error_is_the_default_and_refuses(self, G):
        with pytest.raises(EdgeIdConflict):
            an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior')

    def test_nothing_lands_on_a_refusal(self, G):
        before = sorted(G.edges())
        with pytest.raises(EdgeIdConflict):
            an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior')
        assert sorted(G.edges()) == before

    def test_the_refusal_names_the_ids_and_the_way_out(self, G):
        with pytest.raises(EdgeIdConflict) as caught:
            an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior')
        assert caught.value.ids
        assert 'skip' in str(caught.value)

    def test_the_exception_is_catchable_by_its_public_name(self):
        """A lazily exported class that is not the class cannot be caught."""
        assert isinstance(EdgeIdConflict, type)
        assert issubclass(EdgeIdConflict, ValueError)

    def test_skip_keeps_what_the_graph_has(self, G):
        before = sorted(G.edges())
        an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior', on_conflict='skip')
        assert sorted(G.edges()) == before

    def test_replace_keeps_what_the_batch_brings(self, G):
        before = sorted(G.edges())
        an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior', on_conflict='replace')
        assert sorted(G.edges()) == before

    def test_rename_keeps_both(self, G):
        before = set(G.edges())
        ids = an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior', on_conflict='rename')
        assert set(G.edges()) > before
        assert all(edge_id.endswith('~2') for edge_id in ids)

    def test_rename_tells_the_caller_the_new_ids(self, G):
        ids = an.add_edges_from_frame(G, ROWS, sign='effect', slice='prior', on_conflict='rename')
        assert all(G.has_edge(edge_id=edge_id) for edge_id in ids)

    def test_an_unknown_policy_raises(self, G):
        with pytest.raises(ValueError, match='on_conflict must be one of'):
            an.add_edges_from_frame(G, ROWS, sign='effect', on_conflict='shrug')
