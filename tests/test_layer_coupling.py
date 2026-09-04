"""Coupling the layers of one aspect, in one call.

A multilayer graph has two families of coupling, and which one applies follows
from whether the aspect is ordered. Both were reachable before only by building
the value pairs by hand from a list kept beside the graph::

    G.layers.add_categorical_coupling('time', [[t, n] for t, n in zip(TIMES, TIMES[1:])])

``TIMES`` there is the fact :meth:`LayerAccessor.aspect` now holds, and the
comprehension is :meth:`Aspect.consecutive_pairs`.
"""

from __future__ import annotations

import warnings

import pytest

import annnet as an
from annnet import Aspect

TIMES = ('t0', 't1', 't2')
MECHANISMS = ('x', 'y', 'z')


def _graph(aspects, nodes=('A', 'B')):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = an.Graph(directed=True)
        G.layers.set_aspects(list(aspects), aspects)
        for coordinate in G.layers._all_layers:
            G.add_nodes([{'node_id': n} for n in nodes], layer=coordinate)
    return G


@pytest.fixture
def timed():
    return _graph({'time': Aspect(TIMES, ordered=True)})


@pytest.fixture
def categorical():
    return _graph({'mechanism': list(MECHANISMS)})


class TestOrdinal:
    """An ordered aspect couples consecutive values."""

    def test_it_couples_each_value_to_the_next(self, timed):
        assert timed.layers.couple('time') == 4  # 2 nodes x 2 consecutive pairs

    def test_the_pairs_are_the_aspects_own(self, timed):
        timed.layers.couple('time')
        rows = [r for r in _rows(timed.views.edges()) if r['ml_kind'] == 'coupling']
        pairs = {(r['src_layer'], r['dst_layer']) for r in rows}
        assert pairs == set(timed.layers.aspect('time').consecutive_pairs())

    def test_it_does_not_couple_across_a_gap(self, timed):
        timed.layers.couple('time')
        rows = [r for r in _rows(timed.views.edges()) if r['ml_kind'] == 'coupling']
        assert ('t0', 't2') not in {(r['src_layer'], r['dst_layer']) for r in rows}

    def test_a_categorical_aspect_refuses_ordinal(self, categorical):
        with pytest.raises(ValueError, match='categorical'):
            categorical.layers.couple('mechanism')


class TestCategorical:
    """A categorical aspect couples across values."""

    def test_it_couples_every_pair(self, categorical):
        assert categorical.layers.couple('mechanism', kind='categorical') == 6  # 2 x C(3,2)

    def test_an_ordered_aspect_may_still_be_coupled_categorically(self, timed):
        assert timed.layers.couple('time', kind='categorical') == 6

    def test_an_unknown_kind_raises(self, timed):
        with pytest.raises(ValueError, match="'ordinal' or 'categorical'"):
            timed.layers.couple('time', kind='sideways')


class TestExplicitPairs:
    """A coupling neither family describes."""

    def test_pairs_override_the_kind(self, timed):
        assert timed.layers.couple('time', pairs=[('t0', 't2')]) == 2

    def test_a_pair_naming_an_absent_value_raises(self, timed):
        with pytest.raises(KeyError, match='not a value of aspect'):
            timed.layers.couple('time', pairs=[('t0', 't9')])


class TestTheFamilyIsInTheId:
    """Two schemes over one node pair used to produce one id, and collide."""

    def test_the_id_carries_the_family(self, timed):
        timed.layers.couple('time')
        assert all(eid.startswith('ordinal:') for eid in timed.edges())

    def test_edge_kind_renames_it(self, timed):
        timed.layers.couple('time', edge_kind='timecourse')
        assert all(eid.startswith('timecourse:') for eid in timed.edges())

    def test_the_family_is_also_an_attribute(self, timed):
        timed.layers.couple('time', edge_kind='timecourse')
        one = next(iter(timed.edges()))
        assert timed.attrs.get_edge_attrs(one)['edge_kind'] == 'timecourse'

    def test_two_families_over_one_node_pair_coexist(self, timed):
        first = timed.layers.couple('time', edge_kind='a')
        second = timed.layers.couple('time', edge_kind='b')
        assert first == second == 4
        assert len(list(timed.edges())) == 8

    def test_the_older_generators_carry_a_family_too(self, categorical):
        categorical.layers.add_categorical_coupling('mechanism', [list(MECHANISMS)])
        assert all(eid.startswith('categorical:') for eid in categorical.edges())

    def test_and_take_edge_kind(self, categorical):
        categorical.layers.add_categorical_coupling(
            'mechanism', [list(MECHANISMS)], edge_kind='mine'
        )
        assert all(eid.startswith('mine:') for eid in categorical.edges())


class TestJoiningOnAnAttribute:
    """Two node ids may denote one entity."""

    @pytest.fixture
    def omics(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G = an.Graph(directed=True)
            G.layers.set_aspects(['omic'], {'omic': ['rna', 'prot']})
            G.add_nodes([{'node_id': 'gene:X', 'sym': 'X'}], layer=('rna',))
            G.add_nodes([{'node_id': 'prot:X', 'sym': 'X'}], layer=('prot',))
            G.add_nodes([{'node_id': 'gene:Q', 'sym': 'Q'}], layer=('rna',))
        return G

    def test_it_couples_two_ids_that_share_the_key(self, omics):
        assert omics.layers.couple('omic', kind='categorical', on='sym') == 1

    def test_the_edge_runs_between_the_two_ids(self, omics):
        omics.layers.couple('omic', kind='categorical', on='sym')
        row = next(iter(_rows(omics.views.edges())))
        assert {row['source'], row['target']} == {'gene:X', 'prot:X'}
        assert {row['src_layer'], row['dst_layer']} == {'rna', 'prot'}

    def test_it_is_an_inter_layer_edge_not_a_coupling_one(self, omics):
        """`ml_kind` is structural: coupling means one entity, and these are two.

        A join on an attribute says two *different* node ids denote one thing,
        which the graph records as an inter-layer edge. The ``edge_kind``
        attribute is what says the two were coupled deliberately.
        """
        omics.layers.couple('omic', kind='categorical', on='sym')
        row = next(iter(_rows(omics.views.edges())))
        assert row['ml_kind'] == 'inter'
        assert row['edge_kind'] == 'categorical'

    def test_an_entity_present_on_one_side_only_is_not_coupled(self, omics):
        omics.layers.couple('omic', kind='categorical', on='sym')
        assert 'gene:Q' not in {r['source'] for r in _rows(omics.views.edges())}

    def test_without_the_key_nothing_couples(self, omics):
        """The two ids differ, so a coupling on identity finds no pair."""
        assert omics.layers.couple('omic', kind='categorical') == 0


class TestWithinAndPresence:
    def test_within_restricts_the_other_aspects(self):
        G = _graph({'time': Aspect(TIMES, ordered=True), 'mechanism': list(MECHANISMS)})
        every = G.layers.couple('time', edge_kind='all')
        some = G.layers.couple('time', within={'mechanism': 'x'}, edge_kind='one')
        assert some * len(MECHANISMS) == every

    def test_within_takes_a_collection(self):
        G = _graph({'time': Aspect(TIMES, ordered=True), 'mechanism': list(MECHANISMS)})
        two = G.layers.couple('time', within={'mechanism': {'x', 'y'}})
        assert two == 8  # 2 nodes x 2 pairs x 2 mechanisms

    def test_both_present_skips_a_missing_node_layer(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G = an.Graph(directed=True)
            G.layers.set_aspects(['time'], {'time': Aspect(('t0', 't1'), ordered=True)})
            G.add_nodes([{'node_id': 'A'}], layer=('t0',))
        assert G.layers.couple('time') == 0

    def test_both_present_false_places_it(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G = an.Graph(directed=True)
            G.layers.set_aspects(['time'], {'time': Aspect(('t0', 't1'), ordered=True)})
            G.add_nodes([{'node_id': 'A'}], layer=('t0',))
        assert G.layers.couple('time', both_present=False) == 1
        assert G.exists('A', time='t1')

    def test_both_present_false_refuses_a_join_key(self, timed):
        """There is no answer to which node id a missing node would have."""
        with pytest.raises(ValueError, match='no answer to which node id'):
            timed.layers.couple('time', on='sym', both_present=False)


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)
