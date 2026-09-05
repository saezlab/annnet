"""The loop: an AnnData, joined onto a graph, and projected back out.

    AnnData -> attach -> AnnNet -> method -> AnnNet (annotated) -> AnnData

Four properties of that loop are what these tests pin, because none of them is
visible from inside a single call:

*Attach is a join.* The entities a network names and the entities an assay
measures do not correspond one to one, and what does not match comes back as
data rather than as silence.

*The measurements stay in the AnnData.* Attaching holds two index maps onto the
caller's array. Reading never consults presence, so a graph can answer for
twenty million cells while holding a few hundred node-layers.

*Results are additive.* A method annotates; nothing is replaced.

*The write-back is lossy, by design.* An AnnData holds a number per entity per
condition. The graph keeps the rest.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import annnet as an
from annnet import exp

ad = pytest.importorskip('anndata')
pd = pytest.importorskip('pandas')

CONDITIONS = ('ctrl', 'stim')
SYMBOLS = ('TF1', 'TG1', 'TG2', 'SUBA', 'SUBB')
X = np.array([[1.0, 2.0, 3.0, 4.0, 6.0], [10.0, 20.0, 30.0, 5.0, 7.0]])


@pytest.fixture
def adata():
    """What a workflow hands over: conditions by measured entities."""
    return ad.AnnData(
        X=X.copy(),
        obs=pd.DataFrame(index=list(CONDITIONS)),
        var=pd.DataFrame(index=list(SYMBOLS)),
    )


@pytest.fixture
def G():
    """A prior with three proteins and one complex made of two measured things."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = an.from_edge_frame(
            [
                {'source': 'TF1', 'target': 'TG1', 'effect': 1},
                {'source': 'TF1', 'target': 'TG2', 'effect': -1},
            ],
            sign='effect',
            aspects={'condition': list(CONDITIONS)},
        )
        graph.add_nodes([{'node_id': 'CPX', 'kind': 'complex', 'sym': 'SUBA_SUBB'}])
        graph.attrs.set_node_attrs_bulk(
            {n: {'sym': n, 'kind': 'protein'} for n in ('TF1', 'TG1', 'TG2')}
        )
    return graph


@pytest.fixture
def mapper():
    return exp.vocabulary.SymbolMapper(separator='_', known=set(SYMBOLS))


def _attach(G, adata, mapper, **kwargs):
    kwargs.setdefault('multiplicity', {'complex': 'min', 'default': 'error'})
    return exp.sysbio.attach(
        G, adata, aspect='condition', on='sym', values={'expr': None}, mapper=mapper, **kwargs
    )


class TestTheJoin:
    def test_a_measured_entity_reaches_its_node(self, G, adata, mapper):
        _attach(G, adata, mapper)
        assert G.layers.values().get('TF1', ('stim',), 'expr') == 10.0

    def test_coverage_is_reported(self, G, adata, mapper):
        assert _attach(G, adata, mapper).coverage == 1.0

    def test_what_did_not_match_comes_back_as_data(self, G, adata, mapper):
        adata.var_names = [*SYMBOLS[:-1], 'GHOST']
        report = _attach(G, adata, mapper)
        assert report.unmapped_ids == ['GHOST']

    def test_the_shortfall_is_a_frame_and_a_list(self, G, adata, mapper):
        """A list answers *how many*; a frame answers *which, and why*."""
        adata.var_names = [*SYMBOLS[:-1], 'GHOST']
        report = _attach(G, adata, mapper)
        rows = _rows(report.unmapped)
        assert [row['input'] for row in rows] == ['GHOST']
        assert rows[0]['status'] == 'unmapped'

    def test_unmapped_ids_compares_against_a_list(self, G, adata, mapper):
        """`report.unmapped == ['Q']` against a frame is silently False."""
        adata.var_names = [*SYMBOLS[:-1], 'GHOST']
        assert _attach(G, adata, mapper).unmapped_ids == ['GHOST']

    def test_the_mapping_names_every_pair(self, G, adata, mapper):
        report = _attach(G, adata, mapper)
        rows = _rows(report.mapping)
        assert {row['input'] for row in rows} == set(SYMBOLS)
        assert {row['status'] for row in rows} <= {'ok', 'ambiguous', 'unmapped'}

    def test_unmapped_error_refuses(self, G, adata, mapper):
        adata.var_names = [*SYMBOLS[:-1], 'GHOST']
        with pytest.raises(ValueError, match='reached no node'):
            _attach(G, adata, mapper, unmapped='error')

    def test_an_undeclared_condition_is_refused(self, G, adata, mapper):
        adata.obs_names = ['ctrl', 'late']
        with pytest.raises(ValueError, match='not values of aspect'):
            _attach(G, adata, mapper)

    def test_an_unknown_aspect_raises(self, G, adata, mapper):
        with pytest.raises(KeyError, match='unknown aspect'):
            exp.sysbio.attach(G, adata, aspect='timepoint', on='sym', mapper=mapper)


class TestMultiplicity:
    """Two directions, named apart by type."""

    def test_a_composite_reduces_over_its_members(self, G, adata, mapper):
        _attach(G, adata, mapper)
        assert G.layers.values().get('CPX', ('ctrl',), 'expr') == 4.0  # min(4, 6)
        assert G.layers.values().get('CPX', ('stim',), 'expr') == 5.0  # min(5, 7)

    def test_a_kind_nobody_named_stops_the_join(self, G, adata, mapper):
        with pytest.raises(ValueError, match='how to combine them is not stated'):
            _attach(G, adata, mapper, multiplicity={'default': 'error'})

    def test_the_refusal_names_a_way_out(self, G, adata, mapper):
        with pytest.raises(ValueError, match='min'):
            _attach(G, adata, mapper, multiplicity={'default': 'error'})

    def test_never_means_this_kind_takes_no_measurement(self, G, adata, mapper):
        _attach(G, adata, mapper, multiplicity={'complex': 'never', 'default': 'error'})
        assert G.layers.values().get('CPX', ('ctrl',), 'expr', None) is None

    @pytest.mark.parametrize(
        ('rule', 'expected'),
        [('min', 4.0), ('max', 6.0), ('mean', 5.0), ('sum', 10.0), ('first', 4.0)],
    )
    def test_every_reducer(self, G, adata, mapper, rule, expected):
        _attach(G, adata, mapper, multiplicity={'complex': rule, 'default': 'error'})
        assert G.layers.values().get('CPX', ('ctrl',), 'expr') == pytest.approx(expected)

    def test_one_entity_reaching_several_nodes_is_the_other_direction(self, G, adata, mapper):
        """Every node takes the same value — nothing is combined."""
        G.add_nodes([{'node_id': 'TF1_copy', 'sym': 'TF1', 'kind': 'protein'}])
        report = _attach(G, adata, mapper, multiplicity={'complex': 'min', 'default': 'first'})
        assert report.multiplicity.get('TF1') == 2
        assert G.layers.values().get('TF1_copy', ('ctrl',), 'expr') == 1.0

    def test_a_bare_string_answers_only_the_entity_direction(self, G, adata, mapper):
        """And the refusal says so, rather than repeating "not stated"."""
        with pytest.raises(ValueError, match='this is the other direction'):
            _attach(G, adata, mapper, multiplicity='allow')

    def test_error_refuses_the_entity_direction(self, adata, mapper):
        """On a graph with no composite, so only that direction is in play."""
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = an.from_edge_frame(
                [{'source': 'TF1', 'target': 'TG1', 'effect': 1}],
                sign='effect',
                aspects={'condition': list(CONDITIONS)},
            )
            graph.attrs.set_node_attrs_bulk(
                {n: {'sym': n, 'kind': 'protein'} for n in ('TF1', 'TG1')}
            )
            graph.add_nodes([{'node_id': 'TF1_copy', 'sym': 'TF1', 'kind': 'protein'}])
        with pytest.raises(ValueError, match='reaches 2 nodes'):
            exp.sysbio.attach(
                graph,
                adata,
                aspect='condition',
                on='sym',
                values={'expr': None},
                mapper=mapper,
                multiplicity='error',
            )

    def test_an_unknown_policy_is_refused_before_any_work(self, G, adata, mapper):
        with pytest.raises(ValueError, match='multiplicity must be'):
            _attach(G, adata, mapper, multiplicity='sideways')


class TestPartialComposites:
    """A composite only some of whose members were measured."""

    @pytest.fixture
    def half(self, adata):
        adata.var_names = ['TF1', 'TG1', 'TG2', 'SUBA', 'OTHER']
        return adata

    def test_reduce_combines_what_there_is(self, G, half, mapper):
        _attach(G, half, mapper, partial='reduce')
        assert G.layers.values().get('CPX', ('ctrl',), 'expr') == 4.0

    def test_skip_leaves_the_node_unmeasured(self, G, half, mapper):
        _attach(G, half, mapper, partial='skip')
        assert G.layers.values().get('CPX', ('ctrl',), 'expr', None) is None

    def test_error_refuses_and_names_the_counts(self, G, half, mapper):
        with pytest.raises(ValueError, match='only 1 were measured'):
            _attach(G, half, mapper, partial='error')

    def test_an_unknown_policy_raises(self, G, adata, mapper):
        with pytest.raises(ValueError, match='partial must be'):
            _attach(G, adata, mapper, partial='maybe')


class TestPlacement:
    """The measurements stay in the AnnData; placement is for the network."""

    def test_matched_places_only_the_nodes_the_join_reached(self, G, adata, mapper):
        report = _attach(G, adata, mapper)
        assert report.placed == 4 * len(CONDITIONS)  # TF1, TG1, TG2, CPX

    def test_none_places_nothing_and_still_reads(self, G, adata, mapper):
        report = _attach(G, adata, mapper, place='none')
        assert report.placed == 0
        assert G.layers.values().get('TF1', ('stim',), 'expr') == 10.0

    def test_reading_never_consults_presence(self, G, adata, mapper):
        """The property that lets the AnnData keep the matrix."""
        _attach(G, adata, mapper, place='none')
        assert not G.exists('TF1', condition='stim')
        assert G.layers.matrix('expr', nodes=['TF1']).values[1, 0] == 10.0

    def test_an_unknown_placement_raises(self, G, adata, mapper):
        with pytest.raises(ValueError, match='place must be'):
            _attach(G, adata, mapper, place='everywhere')

    def test_a_gate_places_only_what_was_measured(self, G, adata, mapper):
        gate = np.ones_like(X, dtype=bool)
        gate[1, :] = False  # nothing measured in `stim`
        adata.layers['measured'] = gate
        _attach(G, adata, mapper, gate='measured')
        assert G.exists('TF1', condition='ctrl')
        assert not G.exists('TF1', condition='stim')

    def test_a_gated_cell_holds_no_value(self, G, adata, mapper):
        gate = np.ones_like(X, dtype=bool)
        gate[1, 0] = False
        adata.layers['measured'] = gate
        _attach(G, adata, mapper, gate='measured')
        assert G.layers.values().get('TF1', ('stim',), 'expr', None) is None


class TestScope:
    def test_unmeasured_is_scoped_to_the_nodes_in_play(self, G, adata, mapper):
        G.add_nodes([{'node_id': 'OTHER_SLICE_NODE'}])
        report = _attach(G, adata, mapper, scope=['TF1', 'TG1', 'TG2', 'CPX'])
        assert report.unmeasured_ids == []

    def test_without_scope_every_unreached_node_is_reported(self, G, adata, mapper):
        G.add_nodes([{'node_id': 'OTHER_SLICE_NODE'}])
        report = _attach(G, adata, mapper)
        assert 'OTHER_SLICE_NODE' in report.unmeasured_ids


class TestWriteBack:
    """A projection, not a round trip."""

    def test_layers_gives_a_conditions_by_entities_matrix(self, G, adata, mapper):
        _attach(G, adata, mapper)
        out = exp.sysbio.write_back(
            G, adata, key='expr', aspect='condition', on='sym', mapper=mapper
        )
        assert out.shape == (adata.n_obs, adata.n_vars)

    def test_the_measured_columns_come_back_unchanged(self, G, adata, mapper):
        _attach(G, adata, mapper)
        out = exp.sysbio.write_back(
            G, adata, key='expr', aspect='condition', on='sym', mapper=mapper
        )
        assert np.allclose(out[:, :3], X[:, :3])

    def test_a_composites_members_report_the_composites_value(self, G, adata, mapper):
        _attach(G, adata, mapper)
        out = exp.sysbio.write_back(
            G, adata, key='expr', aspect='condition', on='sym', mapper=mapper
        )
        assert out[0, 3] == out[0, 4] == 4.0

    def test_it_lands_in_the_anndata(self, G, adata, mapper):
        _attach(G, adata, mapper)
        exp.sysbio.write_back(
            G, adata, key='expr', aspect='condition', on='sym', mapper=mapper, name='back'
        )
        assert 'back' in adata.layers

    def test_obs_takes_a_per_condition_number(self, G, adata, mapper):
        _attach(G, adata, mapper)
        G.layers.set_attrs(('ctrl',), n_active=3.0)
        G.layers.set_attrs(('stim',), n_active=7.0)
        out = exp.sysbio.write_back(G, adata, key='n_active', aspect='condition', into='obs')
        assert list(out) == [3.0, 7.0]
        assert 'n_active' in adata.obs

    def test_var_takes_a_per_entity_number(self, G, adata, mapper):
        G.attrs.set_node_attrs_bulk({'TF1': {'score': 9.0}})
        out = exp.sysbio.write_back(G, adata, key='score', on='sym', mapper=mapper, into='var')
        assert out[0] == 9.0
        assert np.isnan(out[1])

    def test_an_unknown_target_raises(self, G, adata):
        with pytest.raises(ValueError, match='into must be'):
            exp.sysbio.write_back(G, adata, key='x', into='somewhere')

    def test_layers_and_obs_need_the_aspect(self, G, adata):
        with pytest.raises(ValueError, match='aspect its obs rows'):
            exp.sysbio.write_back(G, adata, key='x', into='obs')

    def test_an_unknown_combiner_raises(self, G, adata):
        with pytest.raises(ValueError, match='combine must be'):
            exp.sysbio.write_back(G, adata, key='x', aspect='condition', combine='median')


class TestConnected:
    def test_a_graph_with_no_measurements_is_not_connected(self, G):
        assert exp.sysbio.connected(G) is False
        assert exp.sysbio.measurements(G) == []

    def test_attaching_makes_it_connected(self, G, adata, mapper):
        _attach(G, adata, mapper)
        assert exp.sysbio.connected(G) is True
        assert exp.sysbio.measurements(G) == ['expr']


class TestScale:
    """The property that makes the AnnData the right place for the matrix.

    Twenty thousand entities across a thousand conditions is 2e7 cells. Attaching
    holds two index maps onto the caller's array; reading never consults
    presence. If either stopped being true this test would take minutes.
    """

    def test_a_whole_assay_attaches_and_reads_without_placement(self):
        n_entities, n_cond, n_nodes = 20_000, 1_000, 200
        conditions = [f'c{i}' for i in range(n_cond)]
        symbols = [f'n{i}' for i in range(n_entities)]
        network = symbols[:n_nodes]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            graph = an.from_edge_frame(
                [{'source': network[i], 'target': network[i + 1]} for i in range(n_nodes - 1)],
                aspects={'condition': conditions},
            )
        matrix = np.random.rand(n_cond, n_entities).astype('float32')
        big = ad.AnnData(
            X=matrix,
            obs=pd.DataFrame(index=conditions),
            var=pd.DataFrame(index=symbols),
        )
        report = exp.sysbio.attach(
            graph, big, aspect='condition', values={'expr': None}, place='none'
        )
        # The network is small; the assay is not. Only the network's own
        # entities matched, and nothing was placed.
        assert report.placed == 0
        assert len(report.matched) == n_nodes
        assert len(report.unmapped_ids) == n_entities - n_nodes

        block = graph.layers.matrix(
            'expr', nodes=network[:64], layers=[(c,) for c in conditions[:64]]
        )
        assert block.shape == (64, 64)
        assert np.allclose(block.values, matrix[:64, :64])


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)
