"""An aspect knows its values, in the order they were declared.

A ``set`` held them before, and a set has no order — so a graph declared with
``['basal', 'stim', 'late']`` read its layers back as
``['basal', 'late', 'stim']``. For a categorical aspect that is cosmetic. For an
ordinal one it is the wrong order, and nothing said so.

The tests below pin *agreement with the declaration* rather than a literal list,
so they keep meaning something if the declaration changes.
"""

from __future__ import annotations

import warnings

import pytest

import annnet as an
from annnet import Aspect, OrderedLabels, as_aspect

TIMES = ('0h', '1h', '12h', '24h')
MECHANISMS = ('mapk', 'pi3k')


@pytest.fixture
def timed():
    """One ordered aspect, declared out of alphabetical order on purpose."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = an.Graph(directed=True)
        G.layers.set_aspects(['time'], {'time': Aspect(TIMES, ordered=True)})
    return G


@pytest.fixture
def categorical():
    """One aspect whose values do not come one before another."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        G = an.Graph(directed=True)
        G.layers.set_aspects(['mechanism'], {'mechanism': list(MECHANISMS)})
    return G


# ---------------------------------------------------------------------------
# OrderedLabels
# ---------------------------------------------------------------------------


class TestOrderedLabels:
    """A set that remembers the order it was given."""

    def test_it_keeps_declaration_order(self):
        assert list(OrderedLabels(['b', 'a', 'c'])) == ['b', 'a', 'c']

    def test_adding_one_already_held_does_not_move_it(self):
        labels = OrderedLabels(['b', 'a'])
        labels.add('b')
        assert list(labels) == ['b', 'a']

    def test_adding_a_new_one_appends(self):
        labels = OrderedLabels(['b', 'a'])
        labels.add('c')
        assert list(labels) == ['b', 'a', 'c']

    def test_discard_removes_and_is_forgiving(self):
        labels = OrderedLabels(['b', 'a'])
        labels.discard('b')
        labels.discard('absent')
        assert list(labels) == ['a']

    def test_update_appends_the_new_and_keeps_the_held(self):
        labels = OrderedLabels(['b'])
        labels.update(['a', 'b', 'c'])
        assert list(labels) == ['b', 'a', 'c']

    def test_membership_and_length_read_like_a_set(self):
        labels = OrderedLabels(['b', 'a'])
        assert 'b' in labels and 'z' not in labels
        assert len(labels) == 2

    def test_it_compares_equal_to_a_set_ignoring_order(self):
        assert OrderedLabels(['b', 'a']) == {'a', 'b'}

    def test_it_compares_equal_to_a_list_respecting_order(self):
        assert OrderedLabels(['b', 'a']) == ['b', 'a']
        assert OrderedLabels(['b', 'a']) != ['a', 'b']


# ---------------------------------------------------------------------------
# Aspect
# ---------------------------------------------------------------------------


class TestAspect:
    """Order is answered by the aspect, or refused by it."""

    def test_a_bare_list_is_categorical(self):
        assert as_aspect(['a', 'b']).ordered is False

    def test_as_aspect_leaves_an_aspect_alone(self):
        one = Aspect(['a', 'b'], ordered=True)
        assert as_aspect(one) is one

    def test_duplicate_values_collapse_keeping_the_first_position(self):
        assert Aspect(['a', 'b', 'a']).values == ('a', 'b')

    def test_index_is_the_declared_position(self):
        assert Aspect(TIMES, ordered=True).index('12h') == 2

    def test_consecutive_pairs_are_what_ordinal_coupling_couples(self):
        assert Aspect(TIMES, ordered=True).consecutive_pairs() == [
            ('0h', '1h'),
            ('1h', '12h'),
            ('12h', '24h'),
        ]

    def test_normalized_position_spans_zero_to_one(self):
        aspect = Aspect(TIMES, ordered=True)
        assert aspect.normalized_position(TIMES[0]) == 0.0
        assert aspect.normalized_position(TIMES[-1]) == 1.0

    def test_a_one_value_aspect_does_not_divide_by_zero(self):
        assert Aspect(['only'], ordered=True).normalized_position('only') == 0.0

    def test_before_and_after_split_around_a_value(self):
        aspect = Aspect(TIMES, ordered=True)
        assert aspect.before('12h') == ['0h', '1h']
        assert aspect.after('12h') == ['24h']

    def test_inclusive_keeps_the_value_itself(self):
        aspect = Aspect(TIMES, ordered=True)
        assert aspect.before('12h', inclusive=True) == ['0h', '1h', '12h']
        assert aspect.after('12h', inclusive=True) == ['12h', '24h']

    @pytest.mark.parametrize(
        'call',
        [
            lambda a: a.index('b'),
            lambda a: a.consecutive_pairs(),
            lambda a: a.normalized_position('b'),
            lambda a: a.before('b'),
            lambda a: a.after('b'),
        ],
    )
    def test_a_categorical_aspect_refuses_every_order_question(self, call):
        """The answer would be the declaration order pretending to be a meaning."""
        with pytest.raises(ValueError, match='categorical'):
            call(Aspect(['a', 'b', 'c']))

    def test_the_refusal_says_how_to_fix_it(self):
        with pytest.raises(ValueError, match='ordered=True'):
            Aspect(['a', 'b']).index('a')

    def test_a_value_it_does_not_hold_raises(self):
        with pytest.raises(KeyError, match='not a value of this aspect'):
            Aspect(TIMES, ordered=True).index('99h')

    def test_membership_and_iteration_work_either_way(self):
        for aspect in (Aspect(TIMES, ordered=True), Aspect(TIMES)):
            assert '12h' in aspect
            assert list(aspect) == list(TIMES)
            assert len(aspect) == len(TIMES)


# ---------------------------------------------------------------------------
# The graph keeps the order
# ---------------------------------------------------------------------------


class TestDeclarationOrderSurvives:
    """The regression this whole module exists for."""

    def test_elem_layers_reads_back_in_declaration_order(self, timed):
        assert timed.layers.elem_layers['time'] == list(TIMES)

    def test_it_is_not_merely_sorted(self, timed):
        """Sorted, `TIMES` would read ['0h', '12h', '1h', '24h'] — a different order."""
        assert timed.layers.elem_layers['time'] != sorted(TIMES)

    def test_list_layers_agrees_with_the_declaration(self, timed):
        assert timed.layers.list_layers('time') == list(TIMES)

    def test_the_layer_product_follows_it(self, timed):
        assert timed.layers._all_layers == tuple((value,) for value in TIMES)

    def test_a_second_aspect_keeps_its_own_order(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            G = an.Graph()
            G.layers.set_aspects(
                ['time', 'mechanism'],
                {'time': Aspect(TIMES, ordered=True), 'mechanism': list(MECHANISMS)},
            )
        assert G.layers.elem_layers['time'] == list(TIMES)
        assert G.layers.elem_layers['mechanism'] == list(MECHANISMS)

    def test_a_layer_added_later_goes_on_the_end(self, timed):
        timed.layers.add_elementary_layer('time', '48h')
        assert timed.layers.list_layers('time') == [*TIMES, '48h']


class TestOrderedness:
    """Whether an aspect is ordinal is a property of the aspect."""

    def test_an_aspect_declared_ordered_reads_back_ordered(self, timed):
        assert timed.layers.aspect('time').ordered is True

    def test_a_bare_list_reads_back_categorical(self, categorical):
        assert categorical.layers.aspect('mechanism').ordered is False

    def test_the_aspect_carries_its_values(self, timed):
        assert timed.layers.aspect('time').values == TIMES

    def test_set_ordered_declares_it_after_the_fact(self, categorical):
        categorical.layers.set_ordered('mechanism')
        assert categorical.layers.aspect('mechanism').ordered is True

    def test_set_ordered_takes_it_back(self, timed):
        timed.layers.set_ordered('time', False)
        assert timed.layers.aspect('time').ordered is False

    def test_an_unknown_aspect_raises_on_both(self, timed):
        with pytest.raises(KeyError, match='unknown aspect'):
            timed.layers.aspect('absent')
        with pytest.raises(KeyError, match='unknown aspect'):
            timed.layers.set_ordered('absent')
