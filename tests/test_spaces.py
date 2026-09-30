"""Generic sets and external-axis bindings."""

import pytest

import annnet as an


def test_set_scalar_and_algebra():
    keys = an.Set('alpha')
    assert list(keys) == ['alpha']
    assert keys | {'beta'} == {'alpha', 'beta'}
    assert {'alpha', 'gamma'} & keys == {'alpha'}
    assert keys - 'alpha' == set()
    assert len(an.Set(['alpha', 'alpha'])) == 1
    assert an.Set(7) == {7}
    assert an.Set([('a', 1)]) == {('a', 1)}


def test_empty_binding_and_explicit_translation():
    graph = an.Graph()
    graph.add_nodes(['n1', 'n2'])
    space = an.Space(graph, ['x', 'y'], name='features', id_map={'n1': 'x', 'n2': ['x', 'y']})
    assert len(space) == 0
    assert space.features('n1') == {'x'}
    assert space.nodes('x') == {'n1', 'n2'}
    space.populate('n1')
    assert graph.spaces.of('n1') == {'features'}
    assert graph.spaces.nodes('features') == {'n1'}
    assert graph.spaces.of('n2') == set()
    assert space & {'x'} == {'n1'}
    assert 'x' in space
    assert 'y' not in space


def test_invalid_population_is_atomic():
    graph = an.Graph()
    graph.add_nodes('n1')
    space = an.Space(graph, ['x'], name='features', id_map={'n1': 'x'})
    with pytest.raises(KeyError):
        space.populate(['n1', 'absent'])
    assert len(space) == 0
    assert graph.spaces.of('n1') == set()


def test_membership_survives_copy_and_tracks_removal():
    graph = an.Graph()
    graph.add_nodes('n1')
    an.Space(graph, ['n1'], name='a').populate('n1')
    copy = graph.ops.copy()
    assert copy.spaces.of('n1') == {'a'}
    graph.remove_nodes('n1')
    assert graph.spaces.nodes('a') == set()
    assert copy.spaces.nodes('a') == {'n1'}


def test_registered_accessor_uses_bound_space():
    ad = pytest.importorskip('anndata')
    import numpy as np

    graph = an.Graph()
    graph.add_nodes('n1')
    data = ad.AnnData(np.zeros((2, 1)))
    data.var_names = ['feature1']
    space = an.Space(graph, data, name='features', id_map={'n1': 'feature1'})
    space.populate('n1')
    assert data.annnet.spaces['features'] is space
    assert 'feature1' in space
    assert 'n1' in space


def test_live_space_set_operations_use_current_membership():
    graph = an.Graph()
    graph.add_nodes(['a', 'b'])
    space = an.Space(graph, ['x', 'y'], id_map={'a': 'x', 'b': 'y'})
    space.populate('a')
    assert space == {'a'}
    assert repr(space) == "Space({'a'})"
    assert space ^ {'b'} == {'a', 'b'}
    assert {'b'} ^ space == {'a', 'b'}
    assert space.isdisjoint({'b'})
    assert space | {'y'} == {'a', 'b'}
    graph.remove_nodes('a')
    assert space == set()
    assert space ^ {'b'} == {'b'}


def test_one_shot_axis_is_retained():
    graph = an.Graph()
    graph.add_nodes('n')
    space = an.Space(graph, iter(['feature']), id_map={'n': 'feature'})
    assert space.features('n') == {'feature'}
    assert space.features('n') == {'feature'}


def test_explicit_translation_allows_overlapping_key_spaces():
    graph = an.Graph()
    graph.add_nodes(['a', 'b'])
    space = an.Space(graph, ['a', 'x'], id_map={'a': 'x', 'b': 'a'})
    assert space.features('a') == {'x'}
    assert space.nodes('a') == {'b'}
    space.populate('a')
    with pytest.raises(ValueError, match='ambiguous'):
        _ = 'a' in space


def test_multiple_bindings_do_not_replace_one_another():
    ad = pytest.importorskip('anndata')
    import numpy as np

    data = ad.AnnData(np.zeros((1, 1)))
    data.var_names = ['a']
    graph = an.Graph()
    graph.add_nodes('a')
    first = an.Space(graph, data, name='first')
    second = an.Space(graph, data, name='second')
    assert data.annnet.spaces['first'] is first
    assert data.annnet.spaces['second'] is second
    with pytest.raises(ValueError, match='already bound'):
        an.Space(graph, data, name='first')
