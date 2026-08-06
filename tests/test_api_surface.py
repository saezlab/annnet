"""Surface tests for the public AnnNet API.

The public surface is the part of the package a user reads first, and it is the
part that a refactor is least free to change later. These tests hold it to the
contract: named matrices, two element sequences, short counts, and lookups that
take an id and nothing else.

The sequences are the new part. ``G.N`` and ``G.E`` are the node sequence and
the edge sequence. Iterating one yields ids. A string key is an attribute name,
because that is what almost every read of a sequence wants. An integer key is a
position in the order the sequence currently holds, which is the one place a
position is a legitimate argument: the user asked for the n-th of a sequence
they can see, rather than for an internal address.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet import AnnNet


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def chain() -> AnnNet:
    """Three nodes, two directed edges, one attribute on each axis."""
    G = AnnNet(directed=True)
    G.add_vertices(
        [
            {'vertex_id': 'a', 'kind': 'left'},
            {'vertex_id': 'b', 'kind': 'middle'},
            {'vertex_id': 'c', 'kind': 'right'},
        ]
    )
    G.add_edges('a', 'b', edge_id='e_ab', weight=1.0, label='first')
    G.add_edges('b', 'c', edge_id='e_bc', weight=2.0, label='second')
    return G


# ---------------------------------------------------------------------------
# The named matrices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['A', 'B', 'H', 'S', 'L'])
def test_every_named_matrix_is_a_property(chain, name):
    """Each matrix has a one-letter name and needs no call."""
    matrix = getattr(chain, name)
    assert matrix.shape[0] > 0


def test_the_matrices_namespace_holds_the_parameterized_ones(chain):
    """``G.matrices`` is the namespace for everything that takes an argument."""
    assert hasattr(chain, 'matrices')


def test_the_incidence_matrix_is_named_b_and_not_x(chain):
    """``G.B`` replaced ``G.X``, and no alias stays behind."""
    assert chain.B.shape == (3, 2)
    assert not hasattr(chain, 'X')


# ---------------------------------------------------------------------------
# The counts
# ---------------------------------------------------------------------------


def test_the_counts_are_short_names(chain):
    """``ncount`` and ``ecount`` are the count methods."""
    assert chain.ncount() == 3
    assert chain.ecount() == 2


def test_the_length_of_a_graph_is_its_node_count(chain):
    assert len(chain) == chain.ncount()


def test_the_supra_count_is_an_option_of_the_node_count(chain):
    """A flat graph holds one supra-node per node, so the two counts agree."""
    assert chain.ncount(supra=True) == 3


@pytest.mark.parametrize(
    'name',
    ['num_vertices', 'num_edges', 'number_of_vertices', 'number_of_edges', 'global_vertex_count'],
)
def test_the_old_count_aliases_are_gone(chain, name):
    """One name per count. The aliases the old code carried are removed."""
    assert not hasattr(chain, name)


# ---------------------------------------------------------------------------
# The two sequences
# ---------------------------------------------------------------------------


def test_iterating_a_sequence_yields_ids(chain):
    assert list(chain.N) == ['a', 'b', 'c']
    assert list(chain.E) == ['e_ab', 'e_bc']


def test_a_sequence_has_a_length(chain):
    assert len(chain.N) == 3
    assert len(chain.E) == 2


def test_an_integer_key_is_a_position_in_the_sequence(chain):
    assert chain.N[0] == 'a'
    assert chain.E[1] == 'e_bc'


def test_a_string_key_is_an_attribute_column(chain):
    assert list(chain.N['kind']) == ['left', 'middle', 'right']
    assert list(chain.E['label']) == ['first', 'second']


def test_an_attribute_column_is_a_vector(chain):
    """The column is a vector, so a vectorized operation runs on it directly."""
    column = chain.E['weight']
    assert isinstance(column, np.ndarray)
    assert (column * 2).tolist() == [2.0, 4.0]


def test_assigning_a_string_key_sets_the_whole_column(chain):
    chain.N['kind'] = ['x', 'y', 'z']
    assert list(chain.N['kind']) == ['x', 'y', 'z']


def test_assigning_a_column_that_does_not_exist_creates_it(chain):
    chain.E['score'] = [0.1, 0.2]
    assert list(chain.E['score']) == [0.1, 0.2]


def test_assigning_a_column_of_the_wrong_length_is_an_error(chain):
    with pytest.raises(ValueError):
        chain.N['kind'] = ['x', 'y']


def test_an_unknown_attribute_name_is_an_error(chain):
    with pytest.raises(KeyError):
        chain.N['absent']


def test_select_returns_a_subsequence(chain):
    """A filter selects a subsequence, which is a sequence in its own right."""
    selected = chain.E.select(directed=True)
    assert list(selected) == ['e_ab', 'e_bc']
    assert len(chain.E.select(label='first')) == 1


def test_a_subsequence_answers_the_same_questions(chain):
    selected = chain.N.select(kind='middle')
    assert list(selected) == ['b']
    assert list(selected['kind']) == ['middle']


def test_find_returns_one_element(chain):
    assert chain.N.find(id='a') == 'a'
    assert chain.E.find(label='second') == 'e_bc'


def test_find_refuses_an_ambiguous_answer(chain):
    """``find`` names one element, so more than one match is an error."""
    chain.N['kind'] = ['same', 'same', 'same']
    with pytest.raises(ValueError):
        chain.N.find(kind='same')


def test_find_refuses_an_empty_answer(chain):
    with pytest.raises(KeyError):
        chain.N.find(id='absent')


# ---------------------------------------------------------------------------
# No internal position map, and no position in a lookup
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('name', ['entity_to_idx', 'idx_to_entity', 'entity_types'])
def test_the_public_namespace_holds_no_position_map(chain, name):
    """A position map is internal. The public object does not carry one."""
    assert not hasattr(chain, name)
    assert name not in dir(chain)


def test_an_edge_lookup_refuses_a_position(chain):
    """``get_edge`` takes an id. A column number is not an id."""
    with pytest.raises(TypeError):
        chain.get_edge(0)


def test_an_edge_lookup_takes_an_id(chain):
    assert chain.get_edge('e_ab').edge_id == 'e_ab'


def test_the_position_lookup_of_a_node_is_gone(chain):
    """``G.N[0]`` is how a caller asks for the n-th node now."""
    assert not hasattr(chain, 'get_vertex')


# ---------------------------------------------------------------------------
# One element and a collection of one
# ---------------------------------------------------------------------------


def test_one_node_and_a_list_of_one_node_agree():
    """A single element is accepted wherever a collection is."""
    single = AnnNet(directed=True)
    single.add_vertices('a')
    listed = AnnNet(directed=True)
    listed.add_vertices(['a'])
    assert list(single.N) == list(listed.N) == ['a']


def test_one_edge_and_a_list_of_one_edge_agree():
    single = AnnNet(directed=True)
    single.add_edges({'source': 'a', 'target': 'b', 'edge_id': 'e'})
    listed = AnnNet(directed=True)
    listed.add_edges([{'source': 'a', 'target': 'b', 'edge_id': 'e'}])
    assert list(single.E) == list(listed.E) == ['e']
    assert list(single.N) == list(listed.N)


def test_one_node_and_a_list_of_one_node_agree_on_removal(chain):
    single = chain.ops.copy()
    single.remove_vertices('a')
    listed = chain.ops.copy()
    listed.remove_vertices(['a'])
    assert list(single.N) == list(listed.N)


# ---------------------------------------------------------------------------
# obs and var are derived, not stored
# ---------------------------------------------------------------------------


def test_obs_and_var_are_materialized_each_time(chain):
    """Each read builds a table. Two reads are equal and not the same object."""
    assert chain.obs is not chain.obs
    assert chain.var is not chain.var


def test_the_stored_attribute_frames_are_gone(chain):
    """``obs`` and ``var`` are the only names for the two tables."""
    assert not hasattr(chain, 'vertex_attributes')
    assert not hasattr(chain, 'edge_attributes')


def test_a_column_write_shows_up_in_the_next_materialization(chain):
    """A write through the cheap path reaches the table the slow path builds."""
    chain.N['kind'] = ['x', 'y', 'z']
    rows = {row['vertex_id']: row for row in chain.obs.to_dicts()}
    assert [rows[v]['kind'] for v in ('a', 'b', 'c')] == ['x', 'y', 'z']


def test_a_cell_write_shows_up_in_the_next_materialization(chain):
    chain.attrs.set_edge_attrs('e_ab', label='changed')
    rows = {row['edge_id']: row for row in chain.var.to_dicts()}
    assert rows['e_ab']['label'] == 'changed'
