"""One predicate engine behind ``N.select``, ``E.select``, ``attrs.select`` and
``layers.where``.

The reference implementation at the bottom is a plain Python loop over row
dictionaries. Every selection surface is checked against it on one fixture, so
a surface that answers differently is caught by a second opinion rather than by
another call into the same resolver.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from annnet import AnnNet
from annnet.core._predicate import (
    OPERATORS,
    Condition,
    parse_conditions,
    satisfies,
)


# ---------------------------------------------------------------------------
# fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def G():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = AnnNet(directed=True, aspects={'time': ['0h', '1h', '12h'], 'cond': ['ctrl', 'stim']})
        g.layers.set_ordered('time')
        g.layers.place(['a', 'b', 'c', 'd'], [('0h', 'ctrl'), ('1h', 'ctrl'), ('12h', 'stim')])
        g.add_nodes('e', layer=('0h', 'ctrl'))
        g.attrs.update(
            'nodes',
            {
                'a': {'score': 0.9, 'group': 'x', 'weird__name': 1},
                'b': {'score': 0.4, 'group': 'y', 'weird__name': 2},
                'c': {'score': 2.5, 'group': 'x', 'excluded': True},
                'd': {'group': None, 'excluded': False},
                # e carries nothing
            },
        )
        g.add_edges(('a', ('0h', 'ctrl')), ('b', ('0h', 'ctrl')), edge_id='ab', weight=1.0)
        g.add_edges(('b', ('1h', 'ctrl')), ('c', ('1h', 'ctrl')), edge_id='bc', weight=2.0)
        g.add_edges(
            ('c', ('12h', 'stim')), ('a', ('12h', 'stim')), edge_id='ca', weight=0.0, directed=False
        )
        g.add_edges(
            [
                {
                    'head': [('a', ('0h', 'ctrl'))],
                    'tail': [('b', ('0h', 'ctrl')), ('c', ('0h', 'ctrl'))],
                    'edge_id': 'h1',
                }
            ]
        )
        g.attrs.update(
            'edges',
            {
                'ab': {'confidence': 0.9, 'kind_label': 'ppi'},
                'bc': {'confidence': 0.3, 'kind_label': 'ppi'},
                'ca': {'confidence': 0.8},
                'h1': {'confidence': 0.95, 'kind_label': 'complex'},
            },
        )
        g.slices.add('prior', edges=['ab', 'bc'])
        g.slices.add('fit', edges=['bc', 'ca'])
        g.attrs.update(
            'edge_slices', {('fit', 'bc'): {'activity': 1.5}, ('fit', 'ca'): {'activity': -0.2}}
        )
        g.attrs.update(
            'node_layers',
            {
                ('a', ('0h', 'ctrl')): {'score': 0.2},
                ('a', ('12h', 'stim')): {'score': 0.95},
                ('c', ('1h', 'ctrl')): {'score': 0.85},
            },
        )
    return g


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


class TestParsing:
    def test_bare_keyword_is_equality(self):
        assert parse_conditions({'group': 'x'}, fields=['group']) == [Condition('group', 'eq', 'x')]

    def test_every_operator_parses(self):
        for op in OPERATORS:
            value = [1] if op in ('in', 'not_in') else (True if op == 'isnull' else 1)
            [(field, operator, _)] = parse_conditions({f'score__{op}': value}, fields=['score'])
            assert (field, operator) == ('score', op)

    def test_a_literal_column_name_with_dunder_wins_over_suffix_parsing(self):
        found = parse_conditions({'weird__name': 2}, fields=['weird__name'])
        assert found == [Condition('weird__name', 'eq', 2)]

    def test_an_unknown_field_raises_with_the_known_fields(self):
        with pytest.raises(KeyError, match="unknown field 'nope'.*score"):
            parse_conditions({'nope__gt': 1}, fields=['score'])
        with pytest.raises(KeyError, match="unknown field 'nope'"):
            parse_conditions({'nope': 1}, fields=['score'])

    def test_an_unknown_operator_raises(self):
        with pytest.raises(ValueError, match="unknown operator 'between'"):
            parse_conditions({'score__between': 1}, fields=['score'])

    def test_explicit_tuple_keys_remove_ambiguity(self):
        found = parse_conditions({('weird__name', 'gt'): 1}, fields=['weird__name'])
        assert found == [Condition('weird__name', 'gt', 1)]
        with pytest.raises(ValueError, match='unknown operator'):
            parse_conditions({('score', 'nope'): 1}, fields=['score'])

    def test_in_needs_a_collection(self):
        with pytest.raises(TypeError, match="'in' needs a collection"):
            parse_conditions({'group__in': 'x'}, fields=['group'])
        [(_, _, value)] = parse_conditions({'group__in': ['x', 'y']}, fields=['group'])
        assert set(value) == {'x', 'y'}

    def test_isnull_needs_a_bool(self):
        with pytest.raises(TypeError, match='isnull'):
            parse_conditions({'group__isnull': 'yes'}, fields=['group'])


class TestSatisfies:
    @pytest.mark.parametrize('missing', [None, float('nan'), np.nan])
    def test_missing_values_are_excluded_by_every_comparison_but_isnull(self, missing):
        for op in ('eq', 'ne', 'in', 'not_in', 'lt', 'lte', 'gt', 'gte'):
            wanted = [1] if op in ('in', 'not_in') else 1
            assert satisfies(op, missing, wanted) is False
        assert satisfies('isnull', missing, True) is True
        assert satisfies('isnull', missing, False) is False
        assert satisfies('isnull', 3, True) is False
        assert satisfies('isnull', 3, False) is True

    def test_incompatible_comparison_raises_with_context(self):
        with pytest.raises(TypeError, match="score.*'lt'"):
            satisfies('lt', 'text', 3, field='score')

    def test_numeric_and_string_equality(self):
        assert satisfies('eq', 1.0, 1)
        assert satisfies('ne', 'x', 'y')
        assert satisfies('gte', 2, 2)
        assert satisfies('in', 'x', {'x', 'y'})
        assert satisfies('not_in', 'z', {'x', 'y'})


# ---------------------------------------------------------------------------
# the surfaces, each against the reference loop
# ---------------------------------------------------------------------------


def reference(rows: dict, **conditions) -> list:
    """The independent implementation: a loop over ``{key: row}``."""
    fields = {name for row in rows.values() for name in row}
    parsed = parse_conditions(
        conditions, fields=fields | {c.rsplit('__', 1)[0] for c in conditions}
    )
    out = []
    for key, row in rows.items():
        if all(satisfies(op, row.get(field), wanted) for field, op, wanted in parsed):
            out.append(key)
    return out


def node_rows(G):
    return {n: {'id': n, **G.attrs.rows('nodes')[n]} for n in G.N}


def edge_rows(G):
    out = {}
    for e in G.E:
        record = G.E.at(e)
        out[e] = {
            'id': e,
            'kind': record.kind,
            'directed': record.directed,
            'weight': record.weight,
            **G.attrs.rows('edges')[e],
        }
    return out


class TestNodeAndEdgeSelect:
    def test_threshold(self, G):
        assert G.N.select(score__gt=0.5).ids == tuple(reference(node_rows(G), score__gt=0.5))
        assert G.N.select(score__gt=0.5).ids == ('a', 'c')

    def test_membership_and_null(self, G):
        assert G.N.select(group__in=['x', 'y']).ids == ('a', 'b', 'c')
        assert G.N.select(group__isnull=True).ids == ('d', 'e')
        assert G.N.select(group__isnull=False).ids == ('a', 'b', 'c')
        assert G.N.select(group__not_in=['x']).ids == ('b',), (
            'missing values are excluded by not_in'
        )
        assert G.N.select(group__ne='x').ids == ('b',)

    def test_and_within_keywords(self, G):
        assert G.N.select(group='x', score__gte=2).ids == ('c',)
        assert G.N.select(group='x', score__gte=2).ids == tuple(
            reference(node_rows(G), group='x', score__gte=2)
        )

    def test_algebra_keeps_parent_order(self, G):
        chosen = (G.N.select(group='x') | G.N.select(score__gt=0.3)) - G.N.select(excluded=True)
        assert chosen.ids == ('a', 'b')
        assert list(chosen) == ['a', 'b']
        assert len(chosen) == 2
        assert 'a' in chosen and 'c' not in chosen
        both = G.N.select(group='x') & G.N.select(score__gt=1)
        assert both.ids == ('c',)

    def test_edge_intrinsic_and_attribute_fields(self, G):
        assert G.E.select(confidence__gte=0.8, kind='binary').ids == ('ab', 'ca')
        assert G.E.select(kind='hyper').ids == ('h1',)
        assert G.E.select(directed=False).ids == ('ca',)
        assert G.E.select(weight=0).ids == ('ca',)
        for conditions in ({'confidence__gte': 0.8}, {'kind': 'binary', 'confidence__lt': 0.5}):
            assert G.E.select(**conditions).ids == tuple(reference(edge_rows(G), **conditions))

    def test_literal_dunder_column(self, G):
        assert G.N.select(weird__name=2).ids == ('b',)
        assert G.N.select({('weird__name', 'gte'): 1}).ids == ('a', 'b')

    def test_empty_result_is_empty_not_no_filter(self, G):
        empty = G.N.select(score__gt=100)
        assert empty.ids == ()
        assert len(empty) == 0
        assert (empty | G.N.select(group='y')).ids == ('b',)
        assert G.view(nodes=empty).N.ids == ()

    def test_unknown_field_and_bad_comparison_raise(self, G):
        with pytest.raises(KeyError, match='unknown field'):
            G.N.select(nope=1)
        with pytest.raises(TypeError, match="group.*'gt'"):
            G.N.select(group__gt=1)

    def test_find_is_one_or_raises(self, G):
        assert G.N.find(group='y') == 'b'
        with pytest.raises(KeyError, match='nothing matches'):
            G.N.find(group='z')
        with pytest.raises(ValueError, match='2 elements match'):
            G.N.find(group='x')

    def test_cross_root_algebra_raises(self, G):
        other = AnnNet()
        other.add_nodes(['a'])
        with pytest.raises(ValueError, match='different graph'):
            _ = G.N.select(group='x') | other.N.select()
        with pytest.raises(TypeError, match='node.*edge'):
            _ = G.N.select(group='x') | G.E.select()

    def test_repeated_filters_compose(self, G):
        assert G.N.select(group='x').select(score__gt=1).ids == ('c',)
        assert G.E.select(kind='binary').select(directed=True).ids == ('ab', 'bc')

    def test_selection_is_live(self, G):
        chosen = G.N.select(score__gt=0.5)
        assert chosen.ids == ('a', 'c')
        G.attrs.update('nodes', {'b': {'score': 0.6}})
        assert chosen.ids == ('a', 'b', 'c')
        G.remove_nodes('a')
        assert chosen.ids == ('b', 'c')

    def test_explicit_ids_are_fixed_and_validated(self, G):
        fixed = G.N.select(['c', 'a'])
        assert fixed.ids == ('a', 'c'), 'parent order, not the order given'
        with pytest.raises(KeyError, match="'zz'"):
            G.N.select(['a', 'zz'])
        G.remove_nodes('a')
        assert fixed.ids == ('c',), 'an id removed later is dropped, not resurrected'
        G.add_nodes('a', layer=('0h', 'ctrl'))
        assert fixed.ids == ('a', 'c') or fixed.ids == ('c', 'a')

    def test_a_mask_binds_to_the_ids_at_selection_time(self, G):
        mask = np.array([True, False, True, False, False])
        chosen = G.N[mask]
        assert chosen.ids == ('a', 'c')
        with pytest.raises(ValueError, match='mask of 5'):
            _ = G.N[np.array([True, False])]
        G.remove_nodes('a')
        G.add_nodes('zz', layer=('0h', 'ctrl'))
        assert 'zz' not in chosen.ids, 'a reused position must not retarget a new id'
        assert chosen.ids == ('c',)

    def test_duplicate_ids_collapse(self, G):
        assert G.N.select(['a', 'a', 'c']).ids == ('a', 'c')

    def test_scalar_and_generator_inputs(self, G):
        assert G.N.select('a').ids == ('a',)
        assert G.N.select(x for x in ['c', 'a']).ids == ('a', 'c')


class TestAttrsSelect:
    def test_node_layer_rows_keep_their_keys(self, G):
        rows = G.attrs.select('node_layers', score__gte=0.8)
        # Placement (row) order: ``place`` fills layer by layer, so c@1h comes
        # before a@12h.
        assert rows.keys == (('c', ('1h', 'ctrl')), ('a', ('12h', 'stim')))
        assert len(rows) == 2
        assert ('c', ('1h', 'ctrl')) in rows

    def test_projection_is_existential_and_in_parent_order(self, G):
        rows = G.attrs.select('node_layers', score__gte=0.8)
        assert rows.project('nodes').ids == ('a', 'c')
        with pytest.raises(ValueError, match='project'):
            rows.project('slices')

    def test_ordered_aspect_comparison_on_placements(self, G):
        early = G.attrs.select('node_layers', time__lte='1h')
        assert all(key[1][0] in ('0h', '1h') for key in early.keys)
        with pytest.raises(ValueError, match='categorical'):
            G.attrs.select('node_layers', cond__lt='stim')

    def test_edge_slice_rows_and_projection(self, G):
        fit = G.attrs.select('edge_slices', slice_id='fit', activity__gt=0)
        assert fit.keys == (('fit', 'bc'),)
        assert fit.project('edges').ids == ('bc',)
        assert fit.project('slices').ids == ('fit',)
        everything = G.attrs.select('edge_slices')
        # The domain is the explicit memberships plus the stored pairs, slice by
        # slice in registry order and edge by edge in E order — the default
        # slice holds every edge added while it was active.
        assert everything.keys == (
            ('default', 'ab'),
            ('default', 'bc'),
            ('default', 'ca'),
            ('default', 'h1'),
            ('prior', 'ab'),
            ('prior', 'bc'),
            ('fit', 'bc'),
            ('fit', 'ca'),
        )
        assert G.attrs.select('edge_slices', slice_id__ne='default').keys == (
            ('prior', 'ab'),
            ('prior', 'bc'),
            ('fit', 'bc'),
            ('fit', 'ca'),
        )

    def test_rows_of_every_address_have_a_domain(self, G):
        assert G.attrs.select('slices').keys == ('default', 'prior', 'fit')
        assert G.attrs.select('aspects').keys == ('time', 'cond')
        assert G.attrs.select('layers').keys == (('0h', 'ctrl'), ('1h', 'ctrl'), ('12h', 'stim'))
        assert G.attrs.select('elementary_layers', aspect='cond').keys == (
            ('cond', 'ctrl'),
            ('cond', 'stim'),
        )
        assert G.attrs.select('nodes', score__gt=0.5).keys == ('a', 'c')
        assert G.attrs.select('edges', confidence__gte=0.8).keys == ('ab', 'ca', 'h1')

    def test_row_algebra_and_mixing(self, G):
        left = G.attrs.select('edge_slices', slice_id='fit')
        right = G.attrs.select('edge_slices', activity__gt=0)
        assert (left & right).keys == (('fit', 'bc'),)
        assert (left - right).keys == (('fit', 'ca'),)
        with pytest.raises(TypeError, match='edge_slices.*node_layers'):
            _ = left | G.attrs.select('node_layers')


class TestLayersWhere:
    def test_where_uses_the_same_engine(self, G):
        # ``where`` selects over the declared coordinates (the product), not
        # only the occurring ones; the view and the address domains narrow.
        window = G.layers.where(time__lte='1h')
        assert window.layers == (('0h', 'ctrl'), ('0h', 'stim'), ('1h', 'ctrl'), ('1h', 'stim'))
        with pytest.raises(KeyError, match='unknown aspect'):
            G.layers.where(nope='x')
        with pytest.raises(ValueError, match='categorical'):
            G.layers.where(cond__gt='ctrl')
        assert G.layers.where(cond__in=['stim']).layers == (
            ('0h', 'stim'),
            ('1h', 'stim'),
            ('12h', 'stim'),
        )
        assert G.layers.where(time__ne='0h', cond='ctrl').layers == (
            ('1h', 'ctrl'),
            ('12h', 'ctrl'),
        )

    def test_where_reports_an_unknown_value(self, G):
        with pytest.raises(KeyError, match='not a value'):
            G.layers.where(time__lte='99h')


def test_nan_is_missing_in_a_float_column():
    G = AnnNet()
    G.add_nodes(['a', 'b', 'c'])
    G.attrs.update('nodes', {'a': {'x': 1.0}, 'b': {'x': math.nan}, 'c': {'x': 2.0}})
    assert G.N.select(x__isnull=True).ids == ('b',)
    assert G.N.select(x__gte=0).ids == ('a', 'c')
    assert G.N.select(x__ne=1.0).ids == ('c',)
