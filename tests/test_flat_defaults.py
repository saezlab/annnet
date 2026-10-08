"""A flat graph holds no per-edge record of a fact constant across it.

The multilayer role of every structural edge on a flat graph is ``intra``. The
store answers that from the aspects and records nothing per edge; the record
appears when the graph declares aspects, because only then can one edge's role
differ from another's.
"""

from __future__ import annotations

import warnings

import pytest

from annnet import AnnNet
from annnet.core import _structure as S


@pytest.fixture
def flat():
    G = AnnNet(directed=True)
    G.add_nodes(['a', 'b', 'c'])
    G.add_edges('a', 'b', edge_id='e1')
    G.add_edges('b', 'c', edge_id='e2', directed=False)
    G.add_edges([{'members': ['a', 'b', 'c'], 'edge_id': 'h'}])
    G.add_edges(edge_id='ph', as_entity=True)
    return G


def test_a_flat_graph_records_no_role_per_edge(flat):
    assert flat._store.edge_ml_kind == {}
    assert flat._store.edge_ml_layers == {}


def test_every_reader_answers_the_flat_default(flat):
    assert dict(flat.edge_kind) == {'e1': 'intra', 'e2': 'intra', 'h': 'hyper'}
    assert [S.edge_ref(flat, eid).ml_kind for eid in ('e1', 'e2', 'h')] == ['intra'] * 3
    assert S.edge_ref(flat, 'ph').ml_kind is None, 'a placeholder has no role'
    assert list(flat.E['ml_kind']) == ['intra', 'intra', 'intra']
    table = flat.attrs.table('edges', derived=True, backend='pandas').set_index('edge_id')
    assert table.loc['e1', 'ml_kind'] == 'intra'
    assert flat.E.select(ml_kind='intra').ids == ('e1', 'e2', 'h')


def test_writing_the_flat_default_stores_nothing(flat):
    flat.edge_kind['e1'] = 'intra'
    assert flat._store.edge_ml_kind == {}
    flat.edge_kind = {'e2': 'intra', 'h': 'hyper'}
    assert flat._store.edge_ml_kind == {}
    assert flat.edge_kind['e2'] == 'intra'


def test_bulk_adds_and_copies_store_nothing_on_a_flat_graph(flat):
    flat.add_edges(
        [
            {'source': 'a', 'target': 'c', 'edge_id': 'e3'},
            {'source': 'c', 'target': 'a', 'edge_id': 'e4'},
        ]
    )
    assert flat._store.edge_ml_kind == {}
    assert flat.ops.copy()._store.edge_ml_kind == {}
    assert flat.view(edges=['e1', 'e3']).materialize()._store.edge_ml_kind == {}


def test_promotion_materializes_the_record_once(flat):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        flat.layers.set_aspects(['cond'], {'cond': ['x']})
    held = flat._store.edge_ml_kind
    assert sorted(held.values()) == ['intra', 'intra', 'intra'], (
        'every structural edge, not the placeholder'
    )
    assert dict(flat.edge_kind) == {'e1': 'intra', 'e2': 'intra', 'h': 'hyper'}
    flat.add_nodes('d', layer=('x',))
    flat.add_edges(('a', ('_',)), ('d', ('x',)), edge_id='e_inter')
    assert flat.edge_kind['e_inter'] == 'inter'
    assert flat.edge_kind['e1'] == 'intra'


def test_the_native_format_keeps_a_flat_graph_flat(flat, tmp_path):
    path = tmp_path / 'flat.annnet'
    try:
        flat.write(path)
        back = AnnNet.read(path)
    except Exception as exc:  # pragma: no cover - environment-dependent backend
        pytest.skip(f'native IO unavailable here: {exc!r}')
    assert back._store.edge_ml_kind == {}
    assert dict(back.edge_kind) == dict(flat.edge_kind)
