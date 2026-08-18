"""Every reader offers the same destination, and every lossy writer the same sidecar.

Two things used to depend on who wrote the adapter. Whether you could read into a
graph you already had: five of thirteen readers took a ``graph=``, three took a
slice, one took a layer. And what happened to the state a format could not hold:
five different manifest conventions, none of which said *what* was lost.

Both are one contract now, and these tests are what keeps them one.
"""

from __future__ import annotations

import inspect
import warnings

import pytest

from annnet import AnnNet
import annnet.io as aio
from annnet.core import _structure as S
from annnet.io._shared.sidecar import (
    FORMAT_CAPABILITIES,
    AnnNetLossWarning,
    SidecarIntegrityError,
    losses_for,
    present,
    sidecar_path,
)

READERS = (
    'read',
    'from_csv',
    'from_dataframe',
    'from_excel',
    'from_json',
    'from_parquet',
    'from_graphml',
    'from_gexf',
    'from_sif',
    'from_cx2',
    'from_sbml',
    'from_sbml_cobra',
    'from_cobra_model',
    'from_dataframes',
)

# (format, filename, writer, reader)
ROUND_TRIPS = (
    ('annnet', 'g.annnet', 'write', 'read'),
    ('json', 'g.json', 'to_json', 'from_json'),
    ('parquet', 'g.parquet', 'to_parquet', 'from_parquet'),
    ('graphml', 'g.graphml', 'to_graphml', 'from_graphml'),
    ('gexf', 'g.gexf', 'to_gexf', 'from_gexf'),
    ('sif', 'g.sif', 'to_sif', 'from_sif'),
    ('cx2', 'g.cx2', 'to_cx2', 'from_cx2'),
)


def rich_graph() -> AnnNet:
    """A graph carrying something at every level a format might drop."""
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.add_edges('a', 'b', edge_id='e0', weight=2.0)
    G.add_edges([{'head': ['a', 'b'], 'tail': ['c'], 'edge_id': 'h0'}])
    G.slices.add('core')
    G.slices.add_edges('core', ['e0'])
    G.attrs.set_slice_attrs('core', curated=True)
    G.attrs.set_edge_slice_attrs('core', 'e0', confidence=0.9)
    G.uns['src'] = 'x'
    return G


def plain_graph() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b'])
    G.add_edges('a', 'b', edge_id='e0')
    return G


# ---------------------------------------------------------------------------
# One reader contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', READERS)
def test_every_reader_offers_the_same_destination(name):
    parameters = inspect.signature(getattr(aio, name)).parameters
    for option in ('into', 'slice', 'layer', 'on_conflict'):
        assert option in parameters, f'{name} does not offer {option}='


@pytest.mark.parametrize(('fmt', 'filename', 'writer', 'reader'), ROUND_TRIPS)
def test_any_format_can_be_read_into_an_existing_graph_as_a_slice(
    fmt, filename, writer, reader, tmp_path
):
    """The case that used to work for CSV and SBML and no one else."""
    target = plain_graph()
    target.add_nodes(['x'])
    path = tmp_path / filename
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(aio, writer)(rich_graph(), path)
        getattr(aio, reader)(path, into=target, slice='imported', on_conflict='skip')
    assert 'imported' in target.slices.list()
    assert target.slices.nodes('imported'), 'the slice was created but left empty'


@pytest.mark.parametrize(('fmt', 'filename', 'writer', 'reader'), ROUND_TRIPS)
def test_any_format_can_be_read_onto_a_layer(fmt, filename, writer, reader, tmp_path):
    target = AnnNet(directed=True)
    target.layers.set_aspects(['src'], {'src': ['one', 'two']})
    target.add_nodes(['seed'], layer=('one',))
    path = tmp_path / filename
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(aio, writer)(plain_graph(), path)
        getattr(aio, reader)(path, into=target, layer=('two',), on_conflict='skip')
    layers = {key[1] for key in S.entity_keys(target)}
    assert ('two',) in layers, 'imported entities did not land on the named layer'
    assert ('one',) in layers, 'the destination lost what it already held'


def test_a_colliding_id_is_refused_unless_told_otherwise(tmp_path):
    path = tmp_path / 'g.json'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aio.to_json(plain_graph(), path)
        with pytest.raises(Exception, match='already in the destination'):
            aio.from_json(path, into=plain_graph())
        kept = aio.from_json(path, into=plain_graph(), on_conflict='skip')
    assert kept.ne == 1


# ---------------------------------------------------------------------------
# One sidecar
# ---------------------------------------------------------------------------


def test_a_format_reports_only_what_this_graph_would_actually_lose():
    """A loss is reported against the graph in hand, not the format's reputation.

    A plain graph through GraphML loses nothing; the same graph through SIF loses
    node identity, because SIF is an edge list and cannot name a node on its own.
    """
    assert losses_for(plain_graph(), 'graphml') == ()
    assert losses_for(rich_graph(), 'annnet') == ()
    assert 'nodes' in losses_for(plain_graph(), 'sif')
    assert 'contextual_attributes' in losses_for(rich_graph(), 'sif')


def test_the_capability_vocabulary_is_shared():
    for fmt, held in FORMAT_CAPABILITIES.items():
        unknown = held - set(present(rich_graph())) - set(FORMAT_CAPABILITIES['annnet'])
        assert not unknown, f'{fmt} claims capabilities outside the vocabulary: {unknown}'


@pytest.mark.parametrize(
    ('fmt', 'filename', 'writer', 'reader'),
    [c for c in ROUND_TRIPS if c[0] in {'graphml', 'gexf', 'sif'}],
)
def test_a_lossy_format_round_trips_everything_through_its_sidecar(
    fmt, filename, writer, reader, tmp_path
):
    path = tmp_path / filename
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        getattr(aio, writer)(rich_graph(), path)
        assert any(isinstance(c.message, AnnNetLossWarning) for c in caught), (
            f'{fmt} dropped state without saying so'
        )
    assert sidecar_path(path).exists()
    back = getattr(aio, reader)(path)
    assert back.slices.attrs('core') == {'curated': True}
    assert back.attrs.edge_slice('core', 'e0') == {'confidence': 0.9}
    assert dict(back.uns) == {'src': 'x'}


@pytest.mark.parametrize(
    ('writer', 'filename'),
    [('write', 'g.annnet'), ('to_json', 'g.json'), ('to_parquet', 'g.parquet')],
)
def test_a_lossless_format_leaves_no_sidecar(writer, filename, tmp_path):
    """JSON and Parquet carry contextual attributes themselves now."""
    path = tmp_path / filename
    getattr(aio, writer)(rich_graph(), path)
    assert not sidecar_path(path).exists()
    back = getattr(
        aio, {'write': 'read', 'to_json': 'from_json', 'to_parquet': 'from_parquet'}[writer]
    )(path, sidecar='ignore')
    assert back.slices.attrs('core') == {'curated': True}
    assert back.attrs.edge_slice('core', 'e0') == {'confidence': 0.9}


# ---------------------------------------------------------------------------
# Both directions exist
# ---------------------------------------------------------------------------


# Path-based IO carries a sidecar; an object bridge has no file for one to sit
# beside, so it does not offer the switch at all.
PATH_WRITERS = (
    'write',
    'to_json',
    'write_ndjson',
    'to_parquet',
    'to_graphml',
    'to_gexf',
    'to_sif',
    'to_cx2',
    'to_csv',
    'to_excel',
    'to_sbml',
)
PATH_READERS = (
    'read',
    'from_json',
    'read_ndjson',
    'from_parquet',
    'from_graphml',
    'from_gexf',
    'from_sif',
    'from_cx2',
    'from_csv',
    'from_excel',
    'from_sbml',
    'from_sbml_cobra',
)
OBJECT_BRIDGES = ('to_pyg', 'from_pyg', 'to_dataframes', 'from_dataframes')


@pytest.mark.parametrize('name', PATH_WRITERS + PATH_READERS)
def test_every_path_based_entry_point_offers_the_sidecar_switch(name):
    assert 'sidecar' in inspect.signature(getattr(aio, name)).parameters


@pytest.mark.parametrize('name', OBJECT_BRIDGES)
def test_an_object_bridge_does_not_pretend_to_have_a_sidecar(name):
    assert 'sidecar' not in inspect.signature(getattr(aio, name)).parameters


@pytest.mark.parametrize(
    ('writer', 'reader', 'filename'),
    [
        ('to_csv', 'from_csv', 'g.csv'),
        ('to_excel', 'from_excel', 'g.xlsx'),
        ('to_sbml', 'from_sbml', 'g.xml'),
        ('write_ndjson', 'read_ndjson', 'nd'),
    ],
)
def test_a_format_that_could_only_be_read_can_now_be_written(writer, reader, filename, tmp_path):
    """Five directions were missing; a round trip is what proves they arrived."""
    graph = AnnNet(directed=True)
    graph.add_nodes(['a', 'b', 'c'])
    graph.add_edges('a', 'b', edge_id='e0', weight=2.0)
    path = tmp_path / filename
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        getattr(aio, writer)(graph, path)
        back = getattr(aio, reader)(path)
    assert sorted(back.nodes()) == ['a', 'b', 'c']
    assert back.ne >= 1


def test_pyg_round_trips_through_its_manifest():
    pytest.importorskip('torch_geometric')
    graph = AnnNet(directed=True)
    graph.add_nodes(['a', 'b', 'c'])
    graph.add_edges('a', 'b', edge_id='e0', weight=2.0)
    graph.add_edges('b', 'c', edge_id='e1')
    back = aio.from_pyg(aio.to_pyg(graph))
    assert sorted(back.edges()) == ['e0', 'e1']
    assert back.get_edge('e0').weight == pytest.approx(2.0)


def test_a_sidecar_refuses_a_primary_that_changed(tmp_path):
    path = tmp_path / 'g.sif'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aio.to_sif(rich_graph(), path)
    path.write_text(path.read_text(encoding='utf-8') + 'a\tinteracts_with\tz\n', encoding='utf-8')
    with pytest.raises(SidecarIntegrityError, match='different version'):
        aio.from_sif(path)
    # and the escape hatch still reads the primary alone
    assert aio.from_sif(path, sidecar='ignore') is not None


def test_a_writer_can_be_told_not_to_leave_one(tmp_path):
    path = tmp_path / 'g.sif'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        aio.to_sif(rich_graph(), path, sidecar=False)
    assert not sidecar_path(path).exists()
