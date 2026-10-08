"""What every exchange format returns for the parts of a graph it is easiest to lose.

A round trip is judged on structure, not on counts: the direction of an edge,
the kind of each entity and edge, an edge entity's own attributes, the ids an
edge was given and the slices it is in. The fixture carries a parallel pair, a
self-loop, an undirected edge, both kinds of hyperedge and an edge entity with an
edge that joins it to a node.

Formats that keep a record of their own inside the file (JSON, Parquet, CX2, the
native format, the adapter manifests) must return the graph exactly. Formats that
cannot (SIF, CSV, GraphML, GEXF) return it exactly too, through the sidecar they
write beside the file, and say so with an :class:`AnnNetLossWarning`; read
without the sidecar they return the part of the graph they can hold and never
fail.
"""

from __future__ import annotations

import warnings
import importlib.util

import pytest

import annnet as an
import annnet.io as aio
from annnet.adapters import igraph_adapter, networkx_adapter
from annnet.io._shared.sidecar import AnnNetLossWarning


def build():
    graph = an.AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C', 'D'])
    graph.add_edges('A', 'B', edge_id='ab', weight=2.0, relation='activates')
    graph.add_edges('A', 'B', edge_id='ab2', weight=5.0, relation='alternative')
    graph.add_edges('D', 'D', edge_id='loop', weight=0.5, relation='self')
    graph.add_edges('B', 'C', edge_id='bc', directed=False, relation='binds')
    graph.add_edges(['A', 'C', 'D'], edge_id='h_und', directed=False, tag='complex')
    graph.add_edges(['A', 'B'], ['C', 'D'], edge_id='h_dir', directed=True, reaction='r')
    graph.add_edges(edge_id='ent', as_entity=True, description='signal')
    graph.add_edges('ent', 'C', edge_id='ent_c', directed=True, as_entity=True, channel='x')
    graph.attrs.update('nodes', {'A': {'label': 'a', 'n': 1}, 'B': {'label': 'b'}})
    graph.slices.add('fit')
    graph.slices.add_edges('fit', ['ab', 'bc', 'ent_c'])
    return graph


def structure(graph):
    edges = {}
    for edge_id in graph.E:
        record = graph.E.at(edge_id)
        edges[edge_id] = (
            record.kind,
            tuple(sorted(record.source)),
            tuple(sorted(record.target)),
            bool(record.directed),
            float(record.weight),
        )
    rows = graph._attr_store.edge_attr_rows()
    edge_attrs = {
        key: {name: value for name, value in row.items() if value is not None}
        for key, row in rows.items()
    }
    return {
        'entities': dict(graph.entity_kinds()),
        'edges': edges,
        'edge_attrs': {key: row for key, row in edge_attrs.items() if row},
        'slices': {sid: sorted(graph.slices.edges(sid)) for sid in graph.slices.list(False)},
    }


def _via_json(graph, tmp_path):
    aio.to_json(graph, tmp_path / 'g.json')
    return aio.from_json(tmp_path / 'g.json')


def _via_ndjson(graph, tmp_path):
    aio.write_ndjson(graph, tmp_path / 'nd')
    return aio.read_ndjson(tmp_path / 'nd')


def _via_parquet(graph, tmp_path):
    aio.to_parquet(graph, tmp_path / 'p')
    return aio.from_parquet(tmp_path / 'p')


def _via_cx2(graph, tmp_path):
    aio.to_cx2(graph, tmp_path / 'g.cx2')
    return aio.from_cx2(tmp_path / 'g.cx2')


def _via_native(graph, tmp_path):
    aio.write(graph, tmp_path / 'g.annnet')
    return aio.read(tmp_path / 'g.annnet')


def _via_dataframes(graph, tmp_path):
    return aio.from_dataframes(**aio.to_dataframes(graph))


def _via_networkx(graph, tmp_path):
    return networkx_adapter.from_nx(
        *networkx_adapter.to_nx(graph, hyperedge_mode='reify'), hyperedge='reified'
    )


def _via_igraph(graph, tmp_path):
    return igraph_adapter.from_igraph(
        *igraph_adapter.to_igraph(graph, hyperedge_mode='reify'), hyperedge='reified'
    )


def _via_graphtool(graph, tmp_path):
    from annnet.adapters import graphtool_adapter

    return graphtool_adapter.from_graphtool(*graphtool_adapter.to_graphtool(graph))


def _via_graphml(graph, tmp_path):
    aio.to_graphml(graph, tmp_path / 'g.graphml')
    return aio.from_graphml(tmp_path / 'g.graphml')


def _via_gexf(graph, tmp_path):
    aio.to_gexf(graph, tmp_path / 'g.gexf')
    return aio.from_gexf(tmp_path / 'g.gexf')


def _via_sif(graph, tmp_path):
    aio.to_sif(graph, str(tmp_path / 'g.sif'))
    return aio.from_sif(str(tmp_path / 'g.sif'))


def _via_excel(graph, tmp_path):
    aio.to_excel(graph, tmp_path / 'g.xlsx')
    return aio.from_excel(tmp_path / 'g.xlsx')


def _via_csv(graph, tmp_path):
    aio.to_csv(graph, tmp_path / 'g.csv')
    return aio.from_csv(tmp_path / 'g.csv')


FORMATS = {
    'json': _via_json,
    'ndjson': _via_ndjson,
    'parquet': _via_parquet,
    'cx2': _via_cx2,
    'native': _via_native,
    'dataframes': _via_dataframes,
    'networkx': _via_networkx,
    'igraph': _via_igraph,
    'graphml': _via_graphml,
    'gexf': _via_gexf,
    'sif': _via_sif,
    'csv': _via_csv,
}
if importlib.util.find_spec('openpyxl') is not None:
    FORMATS['excel'] = _via_excel

if importlib.util.find_spec('graph_tool') is not None:
    FORMATS['graphtool'] = _via_graphtool

# Formats that hold the whole graph in one file (or in the pair the adapter
# returns) write no loss warning for it.
SELF_CONTAINED = {
    'json',
    'ndjson',
    'parquet',
    'cx2',
    'native',
    'dataframes',
    'networkx',
    'igraph',
    'graphtool',
}


@pytest.mark.parametrize('name', sorted(FORMATS))
def test_a_round_trip_returns_the_same_structure(name, tmp_path):
    graph = build()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        back = FORMATS[name](graph, tmp_path)
    expected, found = structure(graph), structure(back)
    for key in ('entities', 'edges', 'edge_attrs', 'slices'):
        assert found[key] == expected[key], f'{name}: {key} differ'
    warned = any(issubclass(item.category, AnnNetLossWarning) for item in caught)
    if name in SELF_CONTAINED:
        assert not warned, f'{name} holds the whole graph and should not warn'
    else:
        assert warned, f'{name} cannot hold the whole graph and must say so'


@pytest.mark.parametrize('name', sorted(FORMATS))
def test_a_directed_edge_keeps_its_direction(name, tmp_path):
    graph = an.AnnNet(directed=True)
    graph.add_nodes(['A', 'B'])
    graph.add_edges('B', 'A', edge_id='ba')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        back = FORMATS[name](graph, tmp_path)
    record = back.E.at('ba')
    assert (sorted(record.source), sorted(record.target)) == (['B'], ['A'])


@pytest.mark.parametrize('name', ['sif', 'csv', 'graphml', 'gexf'])
def test_read_without_the_sidecar_never_reinterprets_silently(name, tmp_path):
    """Without the sidecar the file holds nodes and edges, and reads as such.

    The read succeeds and the edge that named the entity is still an edge. What
    the format could not hold stays in the sidecar, which is how the entity comes
    back when it is read.
    """
    graph = build()
    writers = {
        'sif': (
            lambda p: aio.to_sif(graph, str(p)),
            lambda p: aio.from_sif(str(p), sidecar='ignore'),
        ),
        'csv': (lambda p: aio.to_csv(graph, p), lambda p: aio.from_csv(p, sidecar='ignore')),
        'graphml': (
            lambda p: aio.to_graphml(graph, p),
            lambda p: aio.from_graphml(p, sidecar='ignore'),
        ),
        'gexf': (lambda p: aio.to_gexf(graph, p), lambda p: aio.from_gexf(p, sidecar='ignore')),
    }
    write, read = writers[name]
    path = tmp_path / f'g.{name}'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        write(path)
        back = read(path)
    assert 'ent_c' in list(back.E) or name == 'sif'
    assert (path.parent / (path.name + '.annnet-sidecar')).exists()


def test_a_manifest_without_edge_entities_reads_without_failing():
    """A manifest written before edge entities were recorded still reads.

    The attributes of an identity that is not an edge have nowhere to go; they are
    reported as a loss and the rest of the graph is returned.
    """
    graph = build()
    nx_graph, manifest = networkx_adapter.to_nx(graph)
    manifest.pop('edge_entities')
    with pytest.warns(AnnNetLossWarning, match='not edges'):
        back = networkx_adapter.from_nx(nx_graph, manifest)
    assert 'ab' in list(back.E)
    assert back.entity_kinds()['ent'] == 'node'


def test_a_csv_edge_keeps_its_id_and_is_one_edge_in_every_slice(tmp_path):
    graph = an.AnnNet(directed=True)
    graph.add_nodes(['A', 'B'])
    graph.add_edges('A', 'B', edge_id='ab', weight=2.0, slice='s1')
    graph.slices.add('s2')
    graph.slices.add_edges('s2', ['ab'])
    aio.to_csv(graph, tmp_path / 'g.csv')
    back = aio.from_csv(tmp_path / 'g.csv')
    assert list(back.E) == ['ab']
    assert sorted(back.slices.edges('s1')) == ['ab']
    assert sorted(back.slices.edges('s2')) == ['ab']


def test_an_igraph_round_trip_keeps_a_multi_aspect_graph():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.AnnNet(directed=True, aspects={'cond': ['ctrl', 'stim'], 'time': ['t0', 't1']})
        for cond in ('ctrl', 'stim'):
            for time in ('t0', 't1'):
                graph.add_nodes(['A', 'B', 'C'], layer=(cond, time))
        graph.add_edges(('A', ('stim', 't0')), ('B', ('stim', 't0')), edge_id='e1')
        graph.add_edges(
            [
                {
                    'members': [
                        ('A', ('ctrl', 't0')),
                        ('B', ('ctrl', 't0')),
                        ('C', ('ctrl', 't0')),
                    ],
                    'edge_id': 'h1',
                }
            ]
        )
        back = igraph_adapter.from_igraph(
            *igraph_adapter.to_igraph(graph, hyperedge_mode='reify'), hyperedge='reified'
        )
    assert back.nv_supra == graph.nv_supra
    assert set(back.E) == set(graph.E)
