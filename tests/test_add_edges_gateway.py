"""How the public ``add_edges`` classifies a batch before it writes one.

The gateway decides whether a batch is all binary, all hyper, or mixed, and it
decides it by reading one item at a time — so the decision is walked once per
edge of a bulk load and every probe it makes is paid that many times.

What makes the reading subtle is the aliases. ``src`` takes precedence over
``source``, so a list-shaped ``src`` names a hyperedge however the plain key
beside it is shaped, and an alias explicitly set to ``None`` falls back to the
plain key rather than deciding anything. The cheap answer for the item a bulk
load actually carries must not lose any of that.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet, _is_hyper_item


@pytest.mark.parametrize(
    ('item', 'is_hyper'),
    [
        ({'source': 'A', 'target': 'B'}, False),
        ({'source': 'A', 'target': 'B', 'weight': 2.0, 'edge_id': 'e'}, False),
        ({'src': 'A', 'tgt': 'B'}, False),
        ({'members': ['A', 'B', 'C']}, True),
        ({'head': ['A'], 'tail': ['B']}, True),
        ({'tail': ['B']}, True),
        ({'source': ['A', 'B'], 'target': 'C'}, True),
        ({'source': 'A', 'target': ['B', 'C']}, True),
        # An alias takes precedence over the plain key, so a list-shaped alias
        # names a hyperedge however the plain key beside it is shaped.
        ({'src': ['A', 'B'], 'source': 'A', 'target': 'B'}, True),
        ({'source': 'A', 'tgt': ['B', 'C'], 'target': 'B'}, True),
        # An alias explicitly set to None falls back to the plain key.
        ({'src': None, 'source': 'A', 'target': 'B'}, False),
        ({'src': None, 'source': ['A', 'B'], 'target': 'B'}, True),
        # A multilayer endpoint is a tuple and is not a member list.
        ({'source': ('A', ('L1',)), 'target': ('B', ('L1',))}, False),
        (('A', 'B'), False),
        (['A', 'B'], False),
        ('A', False),
    ],
)
def test_an_item_is_classified_by_its_keys_and_its_endpoint_shapes(item, is_hyper):
    assert _is_hyper_item(item) is is_hyper


def _kinds(graph, ids):
    return [graph.get_edge(eid).kind for eid in ids]


def test_a_uniform_binary_batch_reaches_the_binary_writer():
    graph = AnnNet(directed=True)
    ids = graph.add_edges([{'source': 'A', 'target': 'B'}, {'source': 'B', 'target': 'C'}])
    assert _kinds(graph, ids) == ['binary', 'binary']


def test_a_uniform_hyper_batch_reaches_the_hyper_writer():
    graph = AnnNet(directed=True)
    ids = graph.add_edges([{'members': ['A', 'B', 'C']}, {'members': ['B', 'C', 'D']}])
    assert _kinds(graph, ids) == ['hyper_undirected', 'hyper_undirected']


def test_a_mixed_batch_keeps_the_order_it_was_given():
    graph = AnnNet(directed=True)
    ids = graph.add_edges(
        [
            {'source': 'A', 'target': 'B'},
            {'members': ['A', 'B', 'C'], 'edge_id': 'h1'},
            {'source': 'C', 'target': 'D'},
        ]
    )
    assert ids[1] == 'h1'
    assert _kinds(graph, ids) == ['binary', 'hyper_undirected', 'binary']
