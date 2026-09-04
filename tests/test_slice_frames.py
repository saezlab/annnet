"""Reading many slices at once, and diffing two of them.

Three helpers callers wrote by hand. ``edge_frame`` is the per-slice attribute
cube — the shape behind *which interactions carried signal in which condition*,
which a ``source``/``target``/``weight`` frame structurally cannot express.
``compare`` is the diff that three set expressions and a loop produced. And
``induce_edges`` is the missing half of a slice built by naming nodes, which
otherwise reads as an edgeless graph.
"""

from __future__ import annotations

import pytest

import annnet as an


@pytest.fixture
def G():
    """A path of three edges, two overlapping slices, per-slice activity."""
    graph = an.Graph(directed=True)
    graph.add_nodes(['A', 'B', 'C', 'D'])
    for source, target, edge_id in (('A', 'B', 'e1'), ('B', 'C', 'e2'), ('C', 'D', 'e3')):
        graph.add_edges(source, target, edge_id=edge_id)
    graph.slices.add('prior', edges=['e1', 'e2'], role='input')
    graph.slices.add('fit', edges=['e2', 'e3'])
    graph.attrs.set_edge_slice_attrs('prior', 'e1', activity=1.0)
    graph.attrs.set_edge_slice_attrs('fit', 'e2', activity=-1.0)
    graph.attrs.set_edge_slice_attrs('fit', 'e3', activity=1.0)
    return graph


# ---------------------------------------------------------------------------
# slices.add(nodes=, edges=)
# ---------------------------------------------------------------------------


class TestAddWithMembers:
    """The three-call opening every caller wrote, as one call."""

    def test_edges_land_in_the_slice(self, G):
        assert G.slices.edges('prior') == {'e1', 'e2'}

    def test_their_incident_nodes_come_with_them(self, G):
        assert G.slices.nodes('prior') == {'A', 'B', 'C'}

    def test_attributes_still_work_alongside(self, G):
        assert G.slices.attrs('prior')['role'] == 'input'

    def test_nodes_can_be_given_instead(self, G):
        G.slices.add('picked', nodes=['A', 'B'])
        assert G.slices.nodes('picked') == {'A', 'B'}
        assert G.slices.edges('picked') == set()

    def test_a_slice_with_neither_is_empty_as_before(self, G):
        G.slices.add('bare')
        assert G.slices.nodes('bare') == set()


# ---------------------------------------------------------------------------
# slices.edge_frame
# ---------------------------------------------------------------------------


class TestEdgeFrame:
    """The per-slice attribute cube."""

    def test_wide_is_one_row_per_slice(self, G):
        rows = _rows(G.slices.edge_frame(attrs=['activity']))
        assert [row['slice_id'] for row in rows] == ['prior', 'fit']

    def test_one_column_per_edge_when_one_attribute_is_asked_for(self, G):
        frame = G.slices.edge_frame(attrs=['activity'])
        assert set(_columns(frame)) == {'slice_id', 'e1', 'e2', 'e3'}

    def test_the_values_land_on_the_right_cells(self, G):
        rows = {row['slice_id']: row for row in _rows(G.slices.edge_frame(attrs=['activity']))}
        assert rows['prior']['e1'] == 1.0
        assert rows['fit']['e2'] == -1.0
        assert rows['prior']['e2'] is None

    def test_the_differentiator_one_edge_active_in_a_named_subset(self, G):
        """A source/target/weight frame cannot express this at all."""
        rows = {row['slice_id']: row for row in _rows(G.slices.edge_frame(attrs=['activity']))}
        assert rows['fit']['e3'] == 1.0
        assert rows['prior']['e3'] is None

    def test_long_is_one_row_per_cell(self, G):
        rows = _rows(G.slices.edge_frame(attrs=['activity'], format='long'))
        assert len(rows) == 6  # 2 slices x 3 edges
        assert set(rows[0]) == {'edge_id', 'slice_id', 'attr', 'value'}

    def test_slices_restricts_the_rows(self, G):
        rows = _rows(G.slices.edge_frame(slices=['fit'], attrs=['activity']))
        assert [row['slice_id'] for row in rows] == ['fit']

    def test_edges_restricts_the_columns(self, G):
        frame = G.slices.edge_frame(edges=['e2'], attrs=['activity'])
        assert set(_columns(frame)) == {'slice_id', 'activity'}

    def test_pairs_cost_the_pairs_asked_for(self, G):
        frame = G.slices.edge_frame(pairs={'second': ('e2', 'activity')})
        assert set(_columns(frame)) == {'slice_id', 'second'}

    def test_missing_names_what_an_absent_cell_holds(self, G):
        rows = {
            row['slice_id']: row
            for row in _rows(G.slices.edge_frame(attrs=['activity'], missing=0.0))
        }
        assert rows['prior']['e2'] == 0.0

    def test_an_unknown_format_raises(self, G):
        with pytest.raises(ValueError, match="format must be 'wide' or 'long'"):
            G.slices.edge_frame(format='tall')

    def test_a_graph_with_no_per_slice_attributes_gives_an_empty_frame(self):
        bare = an.Graph()
        bare.add_nodes(['A'])
        frame = bare.slices.edge_frame()
        assert _rows(frame) == []
        assert 'slice_id' in _columns(frame)


# ---------------------------------------------------------------------------
# slices.compare
# ---------------------------------------------------------------------------


class TestCompare:
    """Which, and on which side."""

    def test_every_element_of_either_slice_gets_a_row(self, G):
        rows = _rows(G.slices.compare('prior', 'fit'))
        assert {row['edge_id'] for row in rows} == {'e1', 'e2', 'e3'}

    def test_the_status_names_the_side(self, G):
        found = {row['edge_id']: row['status'] for row in _rows(G.slices.compare('prior', 'fit'))}
        assert found == {'e1': 'a_only', 'e2': 'both', 'e3': 'b_only'}

    def test_it_is_not_symmetric_and_says_so(self, G):
        forward = {r['edge_id']: r['status'] for r in _rows(G.slices.compare('prior', 'fit'))}
        backward = {r['edge_id']: r['status'] for r in _rows(G.slices.compare('fit', 'prior'))}
        assert forward['e1'] == 'a_only'
        assert backward['e1'] == 'b_only'

    def test_nodes_is_the_other_axis(self, G):
        found = {
            row['node_id']: row['status']
            for row in _rows(G.slices.compare('prior', 'fit', axis='nodes'))
        }
        assert found == {'A': 'a_only', 'B': 'both', 'C': 'both', 'D': 'b_only'}

    def test_it_agrees_with_the_set_operations(self, G):
        rows = _rows(G.slices.compare('prior', 'fit'))
        both = {row['edge_id'] for row in rows if row['status'] == 'both'}
        assert both == G.slices.intersect(['prior', 'fit'])['edges']

    def test_an_unknown_axis_raises(self, G):
        with pytest.raises(ValueError, match="axis must be 'edges' or 'nodes'"):
            G.slices.compare('prior', 'fit', axis='sideways')

    def test_two_identical_slices_are_all_both(self, G):
        rows = _rows(G.slices.compare('prior', 'prior'))
        assert {row['status'] for row in rows} == {'both'}


# ---------------------------------------------------------------------------
# slices.induce_edges
# ---------------------------------------------------------------------------


class TestInduceEdges:
    """A slice named by nodes otherwise reads as an edgeless graph."""

    def test_both_attaches_the_induced_subgraph(self, G):
        G.slices.add('picked', nodes=['A', 'B', 'C'])
        assert G.slices.induce_edges('picked') == 2
        assert G.slices.edges('picked') == {'e1', 'e2'}

    def test_both_leaves_out_an_edge_reaching_outside(self, G):
        G.slices.add('picked', nodes=['A', 'B'])
        G.slices.induce_edges('picked')
        assert 'e2' not in G.slices.edges('picked')

    def test_any_reaches_outside(self, G):
        G.slices.add('picked', nodes=['A', 'B'])
        assert G.slices.induce_edges('picked', mode='any') == 2
        assert G.slices.edges('picked') == {'e1', 'e2'}

    def test_it_returns_how_many_it_attached(self, G):
        G.slices.add('picked', nodes=['A', 'B', 'C'])
        assert G.slices.induce_edges('picked') == 2
        assert G.slices.induce_edges('picked') == 0

    def test_an_edge_already_in_the_slice_is_not_counted_twice(self, G):
        assert G.slices.induce_edges('prior') == 0

    def test_skip_leaves_hyperedges_out(self):
        graph = an.Graph()
        graph.add_nodes(['A', 'B', 'C'])
        graph.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h1'}])
        graph.slices.add('picked', nodes=['A', 'B', 'C'])
        assert graph.slices.induce_edges('picked', hyper='skip') == 0
        assert graph.slices.induce_edges('picked', hyper='all') == 1

    @pytest.mark.parametrize(
        ('kwargs', 'message'),
        [
            ({'mode': 'either'}, "mode must be 'both' or 'any'"),
            ({'hyper': 'maybe'}, "hyper must be 'all' or 'skip'"),
        ],
    )
    def test_an_unknown_policy_raises(self, G, kwargs, message):
        G.slices.add('picked', nodes=['A'])
        with pytest.raises(ValueError, match=message):
            G.slices.induce_edges('picked', **kwargs)


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)


def _columns(frame):
    from annnet._support.dataframe_backend import dataframe_columns

    return list(dataframe_columns(frame))
