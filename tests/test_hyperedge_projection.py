"""One vocabulary for turning a hyperedge into pairs.

Every exporter had to decide what to do with a hyperedge, and each grew its own
spelling — the adapters said ``hyperedge_mode``, CX2 said ``hyperedges`` — with
the words behind them documented three times and agreeing by coincidence.
"""

from __future__ import annotations

import pathlib

import pytest

import annnet as an
from annnet.core import _structure
from annnet._support import projection as P


@pytest.fixture
def hyper():
    G = an.Graph(directed=True)
    G.add_nodes(['A', 'B', 'C'])
    G.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h1'}])
    return G


@pytest.fixture
def flat():
    G = an.Graph(directed=True)
    G.add_nodes(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    return G


class TestVocabulary:
    def test_the_projections_are_the_words_already_in_use(self):
        assert P.PROJECTIONS == ('skip', 'reify', 'expand')

    def test_the_literature_names_are_aliases_not_new_words(self):
        assert P.normalise('star') == 'reify'
        assert P.normalise('clique') == 'expand'

    def test_a_projection_name_passes_through(self):
        for name in P.PROJECTIONS:
            assert P.normalise(name) == name

    def test_none_means_the_default(self):
        assert P.normalise(None) == 'skip'
        assert P.normalise(None, default='reify') == 'reify'

    def test_an_unknown_name_raises_and_lists_both_spellings(self):
        with pytest.raises(ValueError, match='star'):
            P.normalise('sideways')


class TestShapeQuestions:
    def test_is_flat_is_true_when_every_edge_is_a_pair(self, flat):
        assert _structure.is_flat(flat)

    def test_is_flat_is_false_with_a_hyperedge(self, hyper):
        assert not _structure.is_flat(hyper)

    def test_a_derived_coefficient_is_not_an_explicit_one(self, hyper):
        """An edge whose coefficients follow from its weight loses nothing."""
        assert _structure.hyperedges_with_coefficients(hyper) == []

    def test_a_flat_graph_has_none_either(self, flat):
        assert _structure.hyperedges_with_coefficients(flat) == []


class TestCoefficientGuard:
    def test_the_policies_are_declared(self):
        assert P.COEFFICIENT_POLICIES == ('error', 'drop')

    def test_an_unknown_policy_raises(self):
        with pytest.raises(ValueError, match='coefficients must be'):
            P.check_coefficients(['h1'], 'expand', 'maybe')

    def test_skip_and_reify_never_refuse(self):
        for projection in ('skip', 'reify'):
            P.check_coefficients(['h1'], projection)

    def test_drop_accepts_the_loss(self):
        P.check_coefficients(['h1'], 'expand', 'drop')

    def test_no_carrying_edges_means_nothing_to_lose(self):
        P.check_coefficients([], 'expand')

    def test_expand_refuses_when_coefficients_would_go(self):
        with pytest.raises(P.CoefficientsWouldBeLost):
            P.check_coefficients(['h1'], 'expand')

    def test_the_vocabulary_module_reads_no_graph(self):
        """`_support` is a leaf: it holds the words and nothing that reads a graph."""
        source = pathlib.Path(P.__file__).read_text()
        assert 'from ..core' not in source
        assert 'import annnet.core' not in source

    def test_the_exception_carries_the_ids(self):
        raised = P.CoefficientsWouldBeLost(['h1', 'h2'], 'expand')
        assert raised.edge_ids == ['h1', 'h2']
        assert 'reify' in str(raised)


class TestExportersShareIt:
    """``hyperedges=`` means the same thing wherever it is written."""

    def test_the_adapters_take_the_shared_name(self, hyper):
        out = an.to_nx(hyper, hyperedges='star')
        graph = out[0] if isinstance(out, tuple) else out
        assert graph.number_of_nodes() == 4  # three members plus the relation

    def test_an_alias_works_the_same_as_the_word_it_names(self, hyper):
        star = an.to_nx(hyper, hyperedges='star')
        reify = an.to_nx(hyper, hyperedges='reify')
        left = star[0] if isinstance(star, tuple) else star
        right = reify[0] if isinstance(reify, tuple) else reify
        assert left.number_of_nodes() == right.number_of_nodes()
        assert left.number_of_edges() == right.number_of_edges()

    def test_the_old_spelling_still_works(self, hyper):
        out = an.to_nx(hyper, hyperedge_mode='reify')
        graph = out[0] if isinstance(out, tuple) else out
        assert graph.number_of_nodes() == 4

    def test_the_shared_name_wins_when_both_are_given(self, hyper):
        out = an.to_nx(hyper, hyperedge_mode='skip', hyperedges='reify')
        graph = out[0] if isinstance(out, tuple) else out
        assert graph.number_of_nodes() == 4

    def test_an_unknown_name_raises_at_the_exporter(self, hyper):
        with pytest.raises(ValueError, match='hyperedges must be'):
            an.to_nx(hyper, hyperedges='sideways')

    def test_igraph_takes_it_too(self, hyper):
        pytest.importorskip('igraph')
        out = an.to_igraph(hyper, hyperedges='star')
        graph = out[0] if isinstance(out, tuple) else out
        assert graph.vcount() == 4
