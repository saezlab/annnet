"""An attached array survives a write and a read.

This is the test that would have caught a silent wrong answer. A node-layer value
that arrived as a matrix lives in an attached backing, and
``graph.layers.node_attrs`` — which the manifest serializer read — sees only the
contextual store. So every value an attach had joined was written **nowhere**,
and the file read back null in its place without saying anything.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annnet as an
from annnet import Aspect

CONDITIONS = ('c0', 'c1')
NODES = ('n0', 'n1')
VALUES = np.array([[1.5, 2.5], [3.5, 4.5]])


@pytest.fixture
def G():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.Graph(directed=True)
        graph.layers.set_aspects(['cond'], {'cond': Aspect(CONDITIONS, ordered=True)})
        graph.layers.place(NODES, [(c,) for c in CONDITIONS])
        graph.add_edges(('n0', ('c0',)), ('n1', ('c0',)), edge_id='e1')
    graph.layers.attach_values(
        {'expr': VALUES}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES)
    )
    return graph


def _read_back(graph, tmp_path, **kwargs):
    path = tmp_path / 'g.annnet'
    graph.write(path, **kwargs)
    return an.read(path)


def _cells(graph):
    resolver = graph.layers.values()
    return np.array(
        [[resolver.get(n, (c,), 'expr', np.nan) for n in NODES] for c in CONDITIONS],
        dtype=float,
    )


class TestMaterialise:
    """The default, and the fix."""

    def test_the_cells_come_back(self, G, tmp_path):
        assert np.array_equal(_cells(_read_back(G, tmp_path)), VALUES)

    def test_it_is_the_default(self, G, tmp_path):
        explicit = tmp_path / 'explicit'
        explicit.mkdir()
        assert np.array_equal(
            _cells(_read_back(G, tmp_path)),
            _cells(_read_back(G, explicit, attached='materialise')),
        )

    def test_they_read_back_as_ordinary_node_layer_attributes(self, G, tmp_path):
        back = _read_back(G, tmp_path)
        assert back.layers.node_attrs('n0', ('c0',))['expr'] == 1.5

    def test_the_rest_of_the_graph_still_round_trips(self, G, tmp_path):
        back = _read_back(G, tmp_path)
        assert sorted(back.edges()) == ['e1']
        assert back.layers.aspect('cond').values == CONDITIONS

    def test_a_graph_with_no_attached_array_is_unaffected(self, tmp_path):
        graph = an.Graph(directed=True)
        graph.add_nodes(['A', 'B'])
        graph.add_edges('A', 'B', edge_id='e1')
        back = _read_back(graph, tmp_path)
        assert sorted(back.edges()) == ['e1']


class TestTheTwoWaysOut:
    """Neither of them is quiet."""

    def test_drop_leaves_them_out(self, G, tmp_path):
        back = _read_back(G, tmp_path, attached='drop')
        assert np.isnan(_cells(back)).all()

    def test_drop_keeps_the_rest(self, G, tmp_path):
        back = _read_back(G, tmp_path, attached='drop')
        assert sorted(back.edges()) == ['e1']

    def test_error_refuses(self, G, tmp_path):
        with pytest.raises(ValueError, match="attached='error' refuses"):
            G.write(tmp_path / 'g.annnet', attached='error')

    def test_the_refusal_names_what_it_refused(self, G, tmp_path):
        with pytest.raises(ValueError, match="'expr'"):
            G.write(tmp_path / 'g.annnet', attached='error')

    def test_error_is_fine_when_there_is_nothing_attached(self, tmp_path):
        graph = an.Graph(directed=True)
        graph.add_nodes(['A'])
        graph.write(tmp_path / 'g.annnet', attached='error')
        assert (tmp_path / 'g.annnet').exists()

    def test_an_unknown_policy_raises(self, G, tmp_path):
        with pytest.raises(ValueError, match='attached must be'):
            G.write(tmp_path / 'g.annnet', attached='maybe')


class TestWhatMaterialisingMeans:
    def test_only_node_layers_the_graph_holds_are_written(self, tmp_path):
        """A value on a pair nothing placed has no row to be written on."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = an.Graph(directed=True)
            graph.layers.set_aspects(['cond'], {'cond': list(CONDITIONS)})
            graph.layers.place(['n0'], [('c0',)])
        graph.layers.attach_values(
            {'expr': VALUES}, layers=[(c,) for c in CONDITIONS], nodes=list(NODES)
        )
        back = _read_back(graph, tmp_path)
        assert back.layers.values().get('n0', ('c0',), 'expr') == 1.5
        assert back.layers.values().get('n1', ('c1',), 'expr', None) is None

    def test_a_later_backing_wins_on_the_way_out_too(self, G, tmp_path):
        G.layers.attach_values(
            {'expr': np.full((2, 2), 9.0)},
            layers=[(c,) for c in CONDITIONS],
            nodes=list(NODES),
        )
        assert (_cells(_read_back(G, tmp_path)) == 9.0).all()
