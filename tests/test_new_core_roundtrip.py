"""Every graph kind and every attribute store survives the native format.

The core refactor replaced the store behind the graph. The native ``.annnet``
format writes that store to disk and reads it back, so it is the one place
where every field of the store has to be named. A field the writer forgets is
a field the graph loses, and nothing else in the suite would notice.

This module asks two questions.

The first is about shape. Every case of the operation matrix is written, read,
and compared to the graph it came from, by the equivalence harness that the
refactor uses everywhere else. The invariant checker then runs on the graph
that came back, because a graph can answer every question the same way and
still hold a store that contradicts itself.

The second is about attributes. One graph carries a value in every attribute
store the core has, generic and contextual alike, and each store is read back
by name after the roundtrip.

Both parts run two cycles. A field that the writer drops and the reader
invents a default for is equal to itself on the second cycle and not on the
first, so one cycle would hide it; a field the reader mangles a little on
every pass shows up on the second.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from annnet import AnnNet
from annnet.core import _validate as V
from annnet.io import read, write

from ._equivalence import compare
from ._fixtures import CASE_NAMES, build_case

FLAT = ('_',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def roundtrip(graph: AnnNet, path: Path) -> AnnNet:
    """Write a graph to the native format and read it back."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        write(graph, str(path))
        return read(str(path))


def table_rows(table) -> list[dict]:
    """Return a dataframe as a sorted list of row dictionaries, backend independent."""
    if table is None:
        return []
    rows = table.to_dicts() if hasattr(table, 'to_dicts') else list(table.rows(named=True))
    return sorted(
        ({k: v for k, v in row.items() if v is not None} for row in rows),
        key=repr,
    )


def assert_no_violation(graph: AnnNet, label: str) -> None:
    """Fail when the invariant checker reports anything about ``graph``."""
    problems = V.validate_internal_consistency(graph, strict=False)
    assert not problems, f'{label} violates the invariants:\n  ' + '\n  '.join(problems)


# ---------------------------------------------------------------------------
# Every graph kind
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case', CASE_NAMES)
def test_every_graph_kind_survives_the_native_format(case, tmp_path):
    """A written and re-read graph answers every structural question the same way."""
    original = build_case(case)
    returned = roundtrip(original, tmp_path / f'{case}.annnet')

    problems = compare(original, returned)
    assert not problems, f'{case} changed across the roundtrip:\n  ' + '\n  '.join(problems)


@pytest.mark.parametrize('case', CASE_NAMES)
def test_every_graph_kind_comes_back_consistent(case, tmp_path):
    """The store the reader builds satisfies every invariant of the data model."""
    returned = roundtrip(build_case(case), tmp_path / f'{case}.annnet')
    assert_no_violation(returned, f'{case} after a roundtrip')


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_second_cycle_changes_nothing(case, tmp_path):
    """The second write and read agree with the first, field for field."""
    first = roundtrip(build_case(case), tmp_path / f'{case}-1.annnet')
    second = roundtrip(first, tmp_path / f'{case}-2.annnet')

    problems = compare(first, second)
    assert not problems, f'{case} changed on the second cycle:\n  ' + '\n  '.join(problems)


# ---------------------------------------------------------------------------
# Every attribute store
# ---------------------------------------------------------------------------


def graph_with_every_store() -> AnnNet:
    """Build a graph that carries a value in every attribute store of the core.

    The stores are the generic ones, which belong to one element, and the
    contextual ones, which belong to a pair. The graph is multilayer and
    sliced, because half the stores are keyed by a layer or a slice and have
    nowhere to live otherwise.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        G = AnnNet(directed=True)
        G.layers.set_aspects(['phase'], {'phase': ['t0', 't1']})
        G.add_vertices(
            [
                {'vertex_id': 'A', 'kind': 'left', 'score': 0.25},
                {'vertex_id': 'B', 'kind': 'right', 'score': 0.75},
            ],
            layer=('t0',),
        )
        G.add_vertices([{'vertex_id': 'A', 'kind': 'left', 'score': 0.5}], layer=('t1',))
        G.add_edges(
            [
                {
                    'source': ('A', ('t0',)),
                    'target': ('B', ('t0',)),
                    'edge_id': 'e_ab',
                    'weight': 2.0,
                    'edge_kind': 'plain',
                },
                {
                    'source': ('A', ('t0',)),
                    'target': ('A', ('t1',)),
                    'edge_id': 'e_couple',
                    'edge_kind': 'coupling',
                },
            ],
            default_edge_directed=True,
        )

        # Generic: one element, one row.
        G.attrs.set_vertex_attrs('B', note='a node value')
        G.attrs.set_edge_attrs('e_ab', note='an edge value')

        # Contextual: a pair, and no row for the pairs that carry nothing.
        G.slices.add('left')
        G.slices.add_vertex_to_slice('left', 'A')
        G.slices.add_edges('left', ['e_ab'])
        G.attrs.set_slice_attrs('left', note='a slice value')
        G.attrs.set_edge_slice_attrs('left', 'e_ab', note='an edge-in-slice value')
        G.attrs.set_slice_edge_weight('left', 'e_ab', 9.5)
        G.layers.set_aspect_attrs('phase', note='an aspect value')
        G.layers.set_elementary_attrs('phase', 't0', note='an elementary-layer value')
        G.layers.set_attrs(('t0',), note='a layer value')
        G.layers.set_node_attrs('A', ('t0',), note='a node-in-layer value')

        # The graph itself.
        G.uns['note'] = 'a graph value'
        return G


def attribute_snapshot(graph: AnnNet) -> dict:
    """Read back every attribute store by name."""
    return {
        'node_attributes': table_rows(graph.obs),
        'edge_attributes': table_rows(graph.var),
        'slice_attributes': table_rows(graph.slice_attributes),
        'edge_slice_attributes': table_rows(graph.edge_slice_attributes),
        'layer_attributes': table_rows(graph.layer_attributes),
        'aspect_attrs': graph.layers.aspect_attrs('phase'),
        'elementary_layer_attrs': graph.layers.elementary_attrs('phase', 't0'),
        'layer_attrs': graph.layers.attrs(('t0',)),
        'vertex_layer_attrs': graph.layers.node_attrs('A', ('t0',)),
        'slice_attr': graph.attrs.get_slice_attr('left', 'note'),
        'edge_slice_attr': graph.attrs.get_edge_slice_attr('left', 'e_ab', 'note'),
        'weight_in_the_slice': graph.attrs.get_effective_edge_weight('e_ab', slice='left'),
        'weight_outside_it': graph.attrs.get_effective_edge_weight('e_ab'),
        'uns': dict(graph.uns),
    }


@pytest.fixture
def stores_after_a_roundtrip(tmp_path):
    """Return the attribute snapshot before the roundtrip and after two cycles."""
    original = graph_with_every_store()
    first = roundtrip(original, tmp_path / 'stores-1.annnet')
    second = roundtrip(first, tmp_path / 'stores-2.annnet')
    return (
        attribute_snapshot(original),
        attribute_snapshot(first),
        attribute_snapshot(second),
    )


@pytest.mark.parametrize('store', list(attribute_snapshot(graph_with_every_store())))
def test_every_attribute_store_survives_the_native_format(store, stores_after_a_roundtrip):
    """Each store holds the value it held before the graph was written."""
    original, first, second = stores_after_a_roundtrip
    assert first[store] == original[store], f'{store} changed across the roundtrip'
    assert second[store] == first[store], f'{store} changed on the second cycle'


def test_the_graph_with_every_store_stays_consistent(tmp_path):
    """The invariant checker passes on the graph the reader builds."""
    returned = roundtrip(graph_with_every_store(), tmp_path / 'stores.annnet')
    assert_no_violation(returned, 'the graph with every store, after a roundtrip')


def test_the_graph_with_every_store_answers_the_same_way(tmp_path):
    """The structural questions have the same answers after the roundtrip."""
    original = graph_with_every_store()
    returned = roundtrip(original, tmp_path / 'stores.annnet')

    problems = compare(original, returned)
    assert not problems, 'the graph with every store changed:\n  ' + '\n  '.join(problems)
