"""What a number in a graph means, and whether it means it.

`validate` asks whether the object is internally consistent. This asks whether
the data means what a method reading it will assume — and a failure here is a
graph that is fine and will be misread, which is the worse of the two because it
produces an answer.
"""

from __future__ import annotations

import pytest

import annnet as an
from annnet import exp

V = exp.vocabulary


def _signed(rows=None):
    return an.from_edge_frame(
        rows
        or [
            {'source': 'TF1', 'target': 'G1', 'effect': 1},
            {'source': 'TF1', 'target': 'G2', 'effect': -1},
        ],
        sign='effect',
        directed=True,
    )


class TestReservedNames:
    """`weight` was doing three jobs and is now one."""

    def test_the_three_are_distinct_names(self):
        assert len({V.WEIGHT, V.SIGN, V.CONFIDENCE}) == 3

    def test_sign_has_two_values_and_zero_is_not_one(self):
        assert V.SIGN_DOMAIN == (-1, 1)
        assert 0 not in V.SIGN_DOMAIN

    def test_confidence_is_a_unit_range(self):
        assert V.CONFIDENCE_RANGE == (0.0, 1.0)

    def test_each_axis_reserves_its_own_names(self):
        assert V.WEIGHT in V.EDGE_RESERVED
        assert V.GENE_SYMBOL in V.NODE_RESERVED
        assert V.WEIGHT not in V.NODE_RESERVED


class TestContract:
    def test_a_clean_graph_reports_nothing(self):
        assert V.check(_signed()) == []

    def test_the_default_contract_is_named(self):
        assert V.DEFAULT_CONTRACT in V.contracts()

    def test_an_unknown_contract_raises(self):
        with pytest.raises(KeyError, match='unknown contract'):
            V.check(_signed(), 'nonesuch')

    def test_the_rules_are_listable(self):
        assert 'sign_domain' in V.rules()

    def test_zero_is_refused_as_a_sign(self):
        """Unknown is null. Zero is a number and it will be averaged."""
        graph = _signed([{'source': 'A', 'target': 'B', 'effect': 0}])
        found = V.check(graph)
        assert any('sign 0' in message for message in found)

    def test_a_continuous_sign_is_refused_and_named(self):
        graph = _signed([{'source': 'A', 'target': 'B', 'effect': 0.5}])
        assert any(V.CONFIDENCE in message for message in V.check(graph))

    def test_confidence_out_of_range_is_refused(self):
        graph = an.from_edge_frame([{'source': 'A', 'target': 'B', 'confidence': 1.5}])
        assert any('range is' in message for message in V.check(graph))

    def test_a_sign_column_wearing_the_structural_name_is_caught(self):
        """Every weight ±1 and no sign anywhere reads as a mislabelled column."""
        graph = an.from_edge_frame(
            [
                {'source': 'A', 'target': 'B', 'w': 1},
                {'source': 'B', 'target': 'C', 'w': -1},
            ],
            weight='w',
        )
        assert any('structural name' in message for message in V.check(graph))

    def test_a_real_weight_range_is_not_caught(self):
        graph = an.from_edge_frame(
            [
                {'source': 'A', 'target': 'B', 'w': 2.5},
                {'source': 'B', 'target': 'C', 'w': -0.3},
            ],
            weight='w',
        )
        assert not any('structural name' in message for message in V.check(graph))

    def test_an_edge_name_on_the_node_axis_is_caught(self):
        graph = _signed()
        graph.attrs.set_node_attrs_bulk({'TF1': {'confidence': 0.5}})
        assert any('reserves for the edge axis' in m for m in V.check(graph))

    def test_strict_raises_instead_of_reporting(self):
        graph = _signed([{'source': 'A', 'target': 'B', 'effect': 0}])
        with pytest.raises(V.ContractViolation):
            V.check(graph, strict=True)


class TestCapabilities:
    def test_the_capabilities_are_declared(self):
        assert 'signed' in V.capabilities()
        assert 'dyadic' in V.capabilities()

    def test_a_signed_directed_graph_meets_both(self):
        assert V.requires(_signed(), 'signed', 'directed') == []

    def test_an_unsigned_graph_falls_short_and_says_how_many(self):
        graph = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        found = V.requires(graph, 'signed', strict=False)
        assert found and '1 of 1 edges' in found[0]

    def test_requires_is_strict_by_default(self):
        """A caller writing `requires` is stating a precondition, not asking."""
        graph = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        with pytest.raises(V.ContractViolation):
            V.requires(graph, 'signed')

    def test_dyadic_is_met_by_a_flat_graph(self):
        assert V.requires(_signed(), 'dyadic') == []

    def test_dyadic_is_not_met_with_a_hyperedge(self):
        graph = an.Graph(directed=True)
        graph.add_nodes(['A', 'B', 'C'])
        graph.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h1'}])
        assert V.requires(graph, 'dyadic', strict=False)

    def test_acyclic_finds_a_cycle(self):
        graph = an.from_edge_frame(
            [
                {'source': 'A', 'target': 'B'},
                {'source': 'B', 'target': 'C'},
                {'source': 'C', 'target': 'A'},
            ],
            directed=True,
        )
        assert V.requires(graph, 'acyclic', strict=False)

    def test_acyclic_passes_on_a_path(self):
        graph = an.from_edge_frame(
            [{'source': 'A', 'target': 'B'}, {'source': 'B', 'target': 'C'}], directed=True
        )
        assert V.requires(graph, 'acyclic') == []

    def test_bipartite_catches_an_entity_on_both_sides(self):
        assert V.requires(_signed(), 'bipartite', strict=False) == []
        chained = an.from_edge_frame(
            [{'source': 'A', 'target': 'B'}, {'source': 'B', 'target': 'C'}], directed=True
        )
        assert V.requires(chained, 'bipartite', strict=False)

    def test_an_unknown_capability_raises(self):
        with pytest.raises(ValueError, match='unknown capability'):
            V.requires(_signed(), 'purple', strict=False)


class TestMethodSpec:
    def test_it_reads_as_a_sentence(self):
        spec = V.MethodSpec(name='decoupler', requires_edge=('sign',), directed=True)
        assert str(spec) == 'decoupler needs: sign on every edge, directed'

    def test_an_empty_spec_demands_nothing(self):
        assert V.MethodSpec().capabilities() == ()
        assert (
            V.check(an.from_edge_frame([{'source': 'A', 'target': 'B'}]), method=V.MethodSpec())
            == []
        )

    def test_hyperedges_none_becomes_the_dyadic_capability(self):
        assert 'dyadic' in V.MethodSpec(hyperedges='none').capabilities()

    def test_an_unknown_hyperedge_policy_raises(self):
        with pytest.raises(ValueError, match='hyperedges must be'):
            V.MethodSpec(hyperedges='sometimes')

    def test_it_is_hashable_and_frozen(self):
        spec = V.MethodSpec(name='x', requires_edge=['sign'])
        assert hash(spec)
        assert spec.requires_edge == ('sign',)

    def test_a_missing_edge_attribute_is_named_with_a_count(self):
        graph = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        spec = V.MethodSpec(name='decoupler', requires_edge=('sign',))
        found = V.check(graph, method=spec)
        assert found and "needs all 1 edges to carry 'sign'" in found[0]

    def test_the_refusal_names_the_method(self):
        graph = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        spec = V.MethodSpec(name='decoupler', requires_edge=('sign',))
        assert "'decoupler'" in V.check(graph, method=spec)[0]

    def test_a_met_spec_reports_nothing(self):
        spec = V.MethodSpec(
            name='decoupler', requires_edge=('sign',), directed=True, hyperedges='none'
        )
        assert V.check(_signed(), method=spec) == []

    def test_an_over_declared_spec_refuses_a_graph_the_method_would_handle(self):
        """Which is why every field defaults to 'does not care'.

        A regulon holds regulator-to-regulator edges, so a spec that declared
        `bipartite` would refuse the very resource the method exists to score —
        and what a user learns from that is to skip the check.
        """
        chained = an.from_edge_frame(
            [
                {'source': 'A', 'target': 'B', 'effect': 1},
                {'source': 'B', 'target': 'C', 'effect': 1},
            ],
            sign='effect',
            directed=True,
        )
        permissive = V.MethodSpec(name='m', requires_edge=('sign',), directed=True)
        strict = V.MethodSpec(name='m', requires_edge=('sign',), directed=True, bipartite=True)
        assert V.check(chained, method=permissive) == []
        assert V.check(chained, method=strict) != []


class TestIdentifiers:
    def test_a_qualified_name_parses_into_two_parts(self):
        found = V.parse('uniprot:P15056')
        assert (found.namespace, found.local_id) == ('uniprot', 'P15056')
        assert found.qualified

    def test_a_bare_name_has_no_namespace(self):
        found = V.parse('BRAF')
        assert found.namespace is None
        assert not found.qualified

    def test_only_the_first_colon_splits(self):
        assert V.parse('a:b:c').local_id == 'b:c'

    def test_rendering_is_the_inverse(self):
        for text in ('uniprot:P15056', 'BRAF'):
            assert V.render(V.parse(text)) == text

    def test_the_null_mapper_is_the_identity(self):
        mapper = V.as_mapper(None)
        assert [str(i) for i in mapper.resolve('BRAF')] == ['BRAF']
        assert mapper.members('A_B') is None

    def test_a_symbol_mapper_splits_a_known_composite(self):
        mapper = V.SymbolMapper(separator='_', known={'A', 'B'})
        assert mapper.members('A_B') == ['A', 'B']

    def test_it_does_not_split_a_name_that_merely_contains_the_separator(self):
        """The failure mode splitting invites, and the reason `known` exists."""
        mapper = V.SymbolMapper(separator='_', known={'A', 'B'})
        assert mapper.members('SLC2A1_X') is None

    def test_a_table_resolves_a_name_to_another(self):
        mapper = V.SymbolMapper({'BRAF': 'uniprot:P15056'})
        assert str(mapper.resolve('BRAF')[0]) == 'uniprot:P15056'

    def test_an_absent_name_resolves_to_itself(self):
        assert str(V.SymbolMapper({}).resolve('X')[0]) == 'X'

    def test_names_of_gives_the_members_or_the_name(self):
        mapper = V.SymbolMapper(separator='_', known={'A', 'B'})
        assert list(V.names_of(mapper, 'A_B')) == ['A', 'B']
        assert list(V.names_of(mapper, 'C')) == ['C']
