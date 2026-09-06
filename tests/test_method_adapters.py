"""The adapters, and the one property that makes them trustworthy.

**No adapter reimplements any arithmetic.** Each test class below pins numeric
parity against calling the method package directly on the equivalent DataFrame,
because an adapter that quietly recomputed something would be indistinguishable
from one that passed through — right up until the numbers diverged.

What an adapter contributes is the three things around the arithmetic: the
declaration before, reading the input off the graph, and writing the result back
additively.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annnet as an
from annnet import exp

pytest.importorskip('anndata')
pd = pytest.importorskip('pandas')
ad = pytest.importorskip('anndata')

CONDITIONS = ('ctrl', 'stim', 'late')


@pytest.fixture
def regulon_graph():
    """Four regulators over thirty targets, signed, one slice."""
    rng = np.random.default_rng(0)
    regulators = [f'TF{i}' for i in range(4)]
    targets = [f'G{i}' for i in range(30)]
    rows = [
        {'source': r, 'target': t, 'effect': int(rng.choice([-1, 1]))}
        for r in regulators
        for t in rng.choice(targets, 12, replace=False)
    ]
    # One under-powered regulator, so `tmin` has something to drop that is not
    # everything: decoupler refuses outright when no source survives.
    rows += [{'source': 'TFsmall', 'target': t, 'effect': 1} for t in targets[:3]]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return an.from_edge_frame(
            rows,
            directed=True,
            sign='effect',
            slice='regulon',
            aspects={'condition': list(CONDITIONS)},
        )


@pytest.fixture
def measurements():
    rng = np.random.default_rng(1)
    targets = [f'G{i}' for i in range(30)]
    return ad.AnnData(
        X=rng.normal(size=(3, 30)),
        obs=pd.DataFrame(index=list(CONDITIONS)),
        var=pd.DataFrame(index=targets),
    )


# ---------------------------------------------------------------------------
# decoupler
# ---------------------------------------------------------------------------

dc = pytest.importorskip('decoupler')
D = exp.sysbio.methods.decoupler


class TestDecouplerSpec:
    def test_it_needs_a_sign(self):
        assert 'sign' in D.SPEC.requires_edge

    def test_it_does_not_declare_bipartite(self):
        """921 of CollecTRI's 1,185 regulators are themselves targets.

        A spec that declared `bipartite` would refuse the resource the method
        exists to score, and what a user learns from that is to skip the check.
        """
        assert D.SPEC.bipartite is False
        assert 'bipartite' not in D.SPEC.capabilities()

    def test_it_does_not_declare_acyclic(self):
        """A regulator regulating itself is biology, not a malformed input."""
        assert D.SPEC.acyclic is None

    def test_a_regulon_with_a_cycle_still_passes(self, regulon_graph):
        cyclic = an.from_edge_frame(
            [
                {'source': 'A', 'target': 'B', 'effect': 1},
                {'source': 'B', 'target': 'A', 'effect': 1},
            ],
            directed=True,
            sign='effect',
        )
        assert exp.vocabulary.check(cyclic, method=D.SPEC) == []

    def test_an_unsigned_graph_is_refused(self):
        unsigned = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        with pytest.raises(exp.vocabulary.ContractViolation):
            D.run(unsigned, None, contract=True)


class TestRegulon:
    def test_it_reads_the_signed_regulon_off_the_graph(self, regulon_graph):
        net = D.regulon(regulon_graph, slice='regulon')
        assert list(net.columns) == ['source', 'target', 'weight']
        assert set(net['weight']) <= {-1, 1}

    def test_one_row_per_pair(self, regulon_graph):
        net = D.regulon(regulon_graph, slice='regulon')
        assert not net.duplicated(subset=['source', 'target']).any()

    def test_an_absent_weight_column_raises(self, regulon_graph):
        with pytest.raises(KeyError, match='is not a column of the edge view'):
            D.regulon(regulon_graph, slice='regulon', weight='absent')

    def test_labels_can_come_from_a_node_attribute(self, regulon_graph):
        regulon_graph.attrs.set_node_attrs_bulk(
            {n: {'symbol': f'sym::{n}'} for n in regulon_graph.nodes()}
        )
        net = D.regulon(regulon_graph, slice='regulon', source_attr='symbol')
        assert all(str(s).startswith('sym::') for s in net['source'])


class TestDecouplerParity:
    """The numbers are decoupler's."""

    def test_the_scores_match_calling_decoupler_directly(self, regulon_graph, measurements):
        result = D.run(regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5)
        net = D.regulon(regulon_graph, slice='regulon')
        direct = measurements.copy()
        dc.mt.ulm(direct, net, tmin=5)
        assert np.allclose(result.scores.values, direct.obsm['score_ulm'].values)

    def test_the_columns_match(self, regulon_graph, measurements):
        result = D.run(regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5)
        net = D.regulon(regulon_graph, slice='regulon')
        direct = measurements.copy()
        dc.mt.ulm(direct, net, tmin=5)
        assert list(result.scores.columns) == list(direct.obsm['score_ulm'].columns)

    def test_it_does_not_annotate_the_callers_anndata(self, regulon_graph, measurements):
        """Scoring is not the adapter's licence to write into the caller's object."""
        D.run(regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5)
        assert 'score_ulm' not in measurements.obsm


class TestDecouplerWriteBack:
    """Additive: a new slice and new attributes, nothing replaced."""

    @pytest.fixture
    def result(self, regulon_graph, measurements):
        return D.run(
            regulon_graph,
            measurements,
            aspect='condition',
            slice='regulon',
            into_slice='scored',
            tmin=5,
        )

    def test_the_scores_land_on_the_graph(self, regulon_graph, result):
        one = result.sources[0]
        assert regulon_graph.layers.values().get(one, ('stim',), 'score') == pytest.approx(
            float(result.scores.loc['stim', one])
        )

    def test_it_places_the_node_layers_the_scores_need(self, regulon_graph, result):
        """A regulator the assay never measured has nowhere to be scored otherwise."""
        assert result.placed > 0
        assert regulon_graph.exists(result.sources[0], condition='stim')

    def test_the_per_condition_summary_lands_on_the_layer(self, regulon_graph, result):
        assert result.summary_key == 'n_active'
        for condition in CONDITIONS:
            assert 'n_active' in regulon_graph.layers.attrs((condition,))

    def test_the_scored_edges_land_in_a_slice(self, regulon_graph, result):
        assert result.slice == 'scored'
        assert regulon_graph.slices.edges('scored')

    def test_the_prior_slice_is_untouched(self, regulon_graph, result):
        assert regulon_graph.slices.edges('regulon')

    def test_dropped_regulators_come_back_named(self, regulon_graph, measurements):
        """One missing for want of targets looks exactly like one that scored zero."""
        result = D.run(regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5)
        assert 'TFsmall' in result.dropped
        assert 'TFsmall' not in result.sources

    def test_an_unknown_method_raises(self, regulon_graph, measurements):
        with pytest.raises(ValueError, match='is not a function decoupler exposes'):
            D.run(regulon_graph, measurements, method='telepathy', aspect='condition')


# ---------------------------------------------------------------------------
# corneto / CARNIVAL
# ---------------------------------------------------------------------------

pytest.importorskip('corneto')
K = exp.sysbio.methods.corneto


@pytest.fixture
def signalling():
    """A cascade with two routes to the same measured node."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = an.from_edge_frame(
            [
                {'source': 'R', 'target': 'A', 'e': 1},
                {'source': 'A', 'target': 'B', 'e': 1},
                {'source': 'B', 'target': 'C', 'e': -1},
                {'source': 'A', 'target': 'D', 'e': 1},
                {'source': 'D', 'target': 'C', 'e': 1},
            ],
            directed=True,
            sign='e',
            slice='signalling',
            aspects={'condition': ['ctrl', 'stim']},
        )
        G.layers.place(sorted(G.nodes()), [('ctrl',), ('stim',)])
    G.layers.set_node_attrs_bulk({('R', ('ctrl',)): 1.0, ('R', ('stim',)): 1.0}, key='perturbation')
    G.layers.set_node_attrs_bulk(
        {('C', ('stim',)): 1.0, ('B', ('stim',)): -1.0, ('C', ('ctrl',)): -1.0},
        key='measured',
    )
    return G


class TestCarnivalSpec:
    def test_it_needs_a_signed_directed_dyadic_graph(self):
        assert K.SPEC.requires_edge == ('sign',)
        assert set(K.SPEC.capabilities()) == {'directed', 'dyadic'}

    def test_it_does_not_demand_an_acyclic_prior(self):
        """It takes a cyclic prior and constrains the *solution*."""
        assert K.SPEC.acyclic is None

    def test_the_solver_is_named_rather_than_defaulted(self):
        """So a run cannot silently start depending on a commercial licence."""
        assert K.SOLVER == 'HIGHS'


class TestPkn:
    def test_it_carries_the_annnet_edge_id(self, signalling):
        """CORNETO adds synthetic edges, so position is not a mapping."""
        network = K.pkn(signalling, slice='signalling')
        ids = {network.get_attr_edge(i).get(K.ID_ATTR) for i in range(network.num_edges)}
        assert ids == set(signalling.slices.edges('signalling'))

    def test_an_unsigned_edge_is_refused(self):
        unsigned = an.from_edge_frame([{'source': 'A', 'target': 'B'}], directed=True)
        with pytest.raises(ValueError, match='carry no sign'):
            K.pkn(unsigned)


class TestCarnivalRun:
    @pytest.fixture
    def fit(self, signalling):
        return K.run(
            signalling,
            inputs='perturbation',
            outputs='measured',
            aspect='condition',
            slice='signalling',
            into_slice='fit',
        )

    def test_one_slice_per_condition(self, fit):
        assert fit.slices == ['fit__ctrl', 'fit__stim']

    def test_each_condition_selected_something(self, fit):
        assert all(count > 0 for count in fit.selected.values())

    def test_a_selected_edge_carries_its_sign_in_that_slice(self, signalling, fit):
        frame = signalling.slices.edge_frame(slices=fit.slices, attrs=['activity'])
        values = {
            v for row in _rows(frame) for k, v in row.items() if k != 'slice_id' and v is not None
        }
        assert values <= {-1.0, 1.0}

    def test_the_two_conditions_differ(self, signalling, fit):
        """The whole point: a value per edge, per condition."""
        found = {row['status'] for row in _rows(signalling.slices.compare(*fit.slices))}
        assert found != {'both'}

    def test_the_prior_is_untouched(self, signalling, fit):
        assert len(signalling.slices.edges('signalling')) == 5

    def test_an_objective_is_reported_per_condition(self, fit):
        assert len(fit.objective) == len(fit.conditions)

    def test_a_node_the_network_does_not_hold_is_reported(self, signalling):
        signalling.add_nodes([{'node_id': 'GHOST'}])
        signalling.layers.place(['GHOST'], [('stim',)])
        signalling.layers.set_node_attrs_bulk({('GHOST', ('stim',)): 1.0}, key='measured')
        fit = K.run(
            signalling,
            inputs='perturbation',
            outputs='measured',
            aspect='condition',
            slice='signalling',
            into_slice='f2',
        )
        assert 'GHOST' in fit.absent['stim']


# ---------------------------------------------------------------------------
# usecases — composition, and nothing else
# ---------------------------------------------------------------------------


class TestUseCases:
    def test_tf_activity_runs_the_whole_loop(self, regulon_graph, measurements):
        result = exp.sysbio.usecases.tf_activity(
            regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5
        )
        assert result.scores is not None
        assert 'n_active' in measurements.obs

    def test_it_gives_the_same_numbers_as_the_adapter(self, regulon_graph, measurements):
        """A use case is a composition; a second code path would be a fork."""
        composed = exp.sysbio.usecases.tf_activity(
            regulon_graph, measurements.copy(), aspect='condition', slice='regulon', tmin=5
        )
        exp.sysbio.attach(
            regulon_graph, measurements, aspect='condition', values={'expression': None}
        )
        direct = D.run(regulon_graph, measurements, aspect='condition', slice='regulon', tmin=5)
        assert np.allclose(composed.scores.values, direct.scores.values)

    def test_causal_subnetwork_wraps_carnival(self, signalling):
        fit = exp.sysbio.usecases.causal_subnetwork(
            signalling,
            inputs='perturbation',
            outputs='measured',
            aspect='condition',
            slice='signalling',
            into_slice='uc',
        )
        assert fit.slices == ['uc__ctrl', 'uc__stim']

    def test_activity_to_obs_is_the_last_step_alone(self, regulon_graph, measurements):
        for condition in CONDITIONS:
            regulon_graph.layers.set_attrs((condition,), fit_score=1.5)
        out = exp.sysbio.usecases.activity_to_obs(
            regulon_graph, measurements, key='fit_score', aspect='condition'
        )
        assert list(out) == [1.5] * len(CONDITIONS)
        assert 'fit_score' in measurements.obs


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)
