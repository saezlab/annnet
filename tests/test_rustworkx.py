"""The rustworkx adapter: export, round trip, slices and hyperedge modes."""

from __future__ import annotations

import pytest

rx = pytest.importorskip('rustworkx')

import annnet
from annnet import AnnNet
from annnet.adapters.rustworkx_adapter import from_rustworkx, to_rustworkx

from ._equivalence import compare


def _small():
    g = AnnNet(directed=True)
    g.add_nodes(['a', 'b', 'c', 'd'])
    g.attrs.update('nodes', {'a': {'score': 1.5, 'name': 'alpha'}})
    g.add_edges('a', 'b', edge_id='e1', weight=2.0)
    g.attrs.update('edges', {'e1': {'relation': 'activates'}})
    g.add_edges('b', 'c', edge_id='e2', directed=False)
    g.add_edges('c', 'c', edge_id='loop')
    g.add_edges([{'head': ['a', 'b'], 'tail': ['c', 'd'], 'edge_id': 'h1', 'weight': 0.5}])
    g.add_edges([{'members': ['a', 'c', 'd'], 'edge_id': 'h2'}])
    g.slices.add('s', edges=['e1'])
    return g


class TestExport:
    def test_payloads_carry_ids_weights_and_attributes(self):
        rxG, _ = to_rustworkx(_small())
        index_of = rxG.attrs['index_of']
        assert rxG[index_of['a']] == {'score': 1.5, 'name': 'alpha', 'node_id': 'a'}
        e1 = next(p for p in rxG.edges() if p['edge_id'] == 'e1')
        assert e1['weight'] == 2.0 and e1['relation'] == 'activates' and e1['directed'] is True

    def test_directed_export_doubles_an_undirected_edge(self):
        rxG, _ = to_rustworkx(_small())
        assert isinstance(rxG, rx.PyDiGraph)
        assert sum(p['edge_id'] == 'e2' for p in rxG.edges()) == 2

    def test_undirected_export(self):
        rxG, _ = to_rustworkx(_small(), directed=False)
        assert isinstance(rxG, rx.PyGraph)
        assert sum(p['edge_id'] == 'e2' for p in rxG.edges()) == 1

    @pytest.mark.parametrize(
        ('mode', 'nodes', 'edges'), [('skip', 4, 4), ('expand', 4, 14), ('reify', 6, 14)]
    )
    def test_hyperedge_modes(self, mode, nodes, edges):
        rxG, _ = to_rustworkx(_small(), hyperedge_mode=mode)
        assert (rxG.num_nodes(), rxG.num_edges()) == (nodes, edges)

    def test_a_slice_limits_nodes_and_edges(self):
        rxG, manifest = to_rustworkx(_small(), slice='s')
        assert (rxG.num_nodes(), rxG.num_edges()) == (2, 1)
        assert list(manifest['edges']) == ['e1']


class TestRoundTrip:
    def test_lossless_with_the_manifest(self):
        g = _small()
        assert compare(g, from_rustworkx(*to_rustworkx(g))) == []

    def test_lossless_on_the_shared_fixture(self, complex_graph):
        assert compare(complex_graph, from_rustworkx(*to_rustworkx(complex_graph))) == []

    def test_a_multilayer_graph(self):
        g = AnnNet(directed=True, aspects={'t': ['0h', '1h']})
        g.add_nodes(['x', 'y'], layer=('0h',))
        g.add_nodes(['x', 'y'], layer=('1h',))
        g.add_edges(('x', ('0h',)), ('y', ('0h',)), edge_id='e')
        g.layers.couple('t', kind='categorical')
        back = from_rustworkx(*to_rustworkx(g))
        assert compare(g, back) == []

    def test_without_a_manifest_reads_the_payloads(self):
        g = _small()
        back = from_rustworkx(to_rustworkx(g)[0])
        assert sorted(back.E) == ['e1', 'e2', 'loop']
        assert back.E.at('e2').directed is False
        assert back.attrs.row('edges', 'e1')['relation'] == 'activates'
        assert back.attrs.row('nodes', 'a')['score'] == 1.5

    def test_reified_hyperedges_come_back(self):
        g = _small()
        rxG, _ = to_rustworkx(g, hyperedge_mode='reify')
        back = from_rustworkx(rxG, hyperedge='reified')
        assert back.E.at('h1').source == frozenset({'a', 'b'})
        assert back.E.at('h1').target == frozenset({'c', 'd'})
        assert set(back.E.at('h2').members) == {'a', 'c', 'd'}

    def test_a_plain_rustworkx_graph(self):
        rxG = rx.PyDiGraph()
        a, b = rxG.add_node('a'), rxG.add_node('b')
        rxG.add_edge(a, b, 3.0)
        back = from_rustworkx(rxG)
        assert sorted(back.N) == ['a', 'b']
        assert len(back.E) == 1


def test_top_level_exports():
    assert annnet.to_rustworkx.__name__ == 'to_rustworkx'
    assert annnet.from_rustworkx.__name__ == 'from_rustworkx'
    assert compare(_small(), annnet.from_rustworkx(*annnet.to_rustworkx(_small()))) == []
    assert annnet.adapters.available_backends()['rustworkx'] is True
