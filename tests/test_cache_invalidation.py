"""Read-mutate-read invariants for version-keyed derived caches.

``validate_internal_consistency`` checks canonical-vs-derived agreement *at rest*
and so cannot catch a stale cache: the canonical records are always correct, and a
cache only diverges once a read has warmed it before a mutation. These tests
exercise the sequence instead.

The shared template is the G1/G2 divergence that surfaced the original bug: build
two identical graphs, warm a cache on one, apply the same mutation to both, and
assert the warm graph answers exactly like the cold one (and like a graph built
from scratch in the post-mutation state).

Regression: hyperedge neighbor lists and the CSR/CSC/adjacency caches keyed on
``_version``, a history counter that does not advance on removes — so anything
read before a removal survived it.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


# --- graph builders --------------------------------------------------------


def _hyper_graph():
    G = AnnNet(directed=False)
    G.add_vertices(['a', 'b', 'c', 'd'])
    G.add_edges([{'src': {'a', 'b', 'c'}, 'edge_id': 'h1'}])
    return G


def _binary_graph():
    G = AnnNet(directed=False)
    G.add_vertices(['a', 'b', 'c', 'd'])
    G.add_edges([('a', 'b'), ('b', 'c')])
    return G


def _mixed_graph():
    G = AnnNet(directed=False)
    G.add_vertices(['a', 'b', 'c', 'd'])
    G.add_edges([('a', 'b')])
    G.add_edges([{'src': {'a', 'c', 'd'}, 'edge_id': 'h1'}])
    return G


# --- mutations -------------------------------------------------------------


def _remove_hyperedge(G):
    G.remove_edges(['h1'])


def _remove_binary_edge(G):
    G.remove_edges([G.edges()[0]])


def _remove_vertex(G):
    G.remove_vertices(['a'])


def _add_binary_edge(G):
    G.add_edges([('a', 'd')])


def _add_hyperedge(G):
    G.add_edges([{'src': {'b', 'c', 'd'}, 'edge_id': 'h_new'}])


def _add_vertex(G):
    G.add_vertices(['z'])
    G.add_edges([('a', 'z')])


# --- cache probes ----------------------------------------------------------
#
# Each probe both warms a cache and produces a comparable value.


def _probe_neighbors(G):
    return {v: sorted(G.neighbors(v)) for v in sorted(G.vertices())}


def _probe_csr(G):
    m = G.cache.csr
    return (m.shape, sorted(zip(*m.nonzero(), strict=False)))


def _probe_csc(G):
    m = G.cache.csc
    return (m.shape, sorted(zip(*m.nonzero(), strict=False)))


def _probe_adjacency(G):
    m = G.cache.adjacency
    return (m.shape, sorted(zip(*m.nonzero(), strict=False)))


def _probe_supra_adjacency(G):
    m = G.layers.supra_adjacency()
    return (m.shape, sorted(zip(*m.nonzero(), strict=False)))


PROBES = {
    'neighbors': _probe_neighbors,
    'csr': _probe_csr,
    'csc': _probe_csc,
    'adjacency': _probe_adjacency,
    'supra_adjacency': _probe_supra_adjacency,
}

CASES = {
    'hyper/remove_hyperedge': (_hyper_graph, _remove_hyperedge),
    'hyper/add_hyperedge': (_hyper_graph, _add_hyperedge),
    'hyper/add_binary': (_hyper_graph, _add_binary_edge),
    'hyper/remove_vertex': (_hyper_graph, _remove_vertex),
    'binary/remove_edge': (_binary_graph, _remove_binary_edge),
    'binary/add_edge': (_binary_graph, _add_binary_edge),
    'binary/add_vertex': (_binary_graph, _add_vertex),
    'binary/remove_vertex': (_binary_graph, _remove_vertex),
    'mixed/remove_hyperedge': (_mixed_graph, _remove_hyperedge),
    'mixed/remove_binary': (_mixed_graph, _remove_binary_edge),
    'mixed/add_hyperedge': (_mixed_graph, _add_hyperedge),
    'mixed/remove_vertex': (_mixed_graph, _remove_vertex),
}


@pytest.mark.parametrize('case', sorted(CASES))
@pytest.mark.parametrize('probe_name', sorted(PROBES))
def test_warm_cache_agrees_with_cold_after_mutation(case, probe_name):
    """A cache warmed before a mutation must not outlive it."""
    build, mutate = CASES[case]
    probe = PROBES[probe_name]

    warm = build()
    probe(warm)  # warm the cache while the pre-mutation state is live
    mutate(warm)

    cold = build()
    mutate(cold)

    assert probe(warm) == probe(cold), (
        f'{probe_name} diverged after {case}: cache warmed before the mutation survived it'
    )


@pytest.mark.parametrize('case', sorted(CASES))
def test_all_caches_warm_then_mutate(case):
    """Warming every cache at once must not leave any of them stale."""
    build, mutate = CASES[case]

    warm = build()
    for probe in PROBES.values():
        probe(warm)
    mutate(warm)

    cold = build()
    mutate(cold)

    for name, probe in PROBES.items():
        assert probe(warm) == probe(cold), f'{name} stale after {case}'


def test_hyperedge_removal_regression():
    """The original report: neighbors through a removed hyperedge."""
    warm = _hyper_graph()
    assert sorted(warm.neighbors('a')) == ['b', 'c']
    warm.remove_edges(['h1'])

    assert warm.edges() == []
    assert warm.neighbors('a') == []


def test_repeated_read_mutate_cycles():
    """Interleaved reads and mutations stay consistent across many rounds."""
    G = AnnNet(directed=False)
    G.add_vertices(['a', 'b', 'c', 'd'])

    for i in range(5):
        eid = f'h{i}'
        G.add_edges([{'src': {'a', 'b', 'c'}, 'edge_id': eid}])
        assert sorted(G.neighbors('a')) == ['b', 'c']
        G.cache.csr  # noqa: B018 - warm between mutations
        G.remove_edges([eid])
        assert G.neighbors('a') == [], f'stale after cycle {i}'
        assert G.cache.csr.nnz == 0, f'stale csr after cycle {i}'


def test_structure_version_advances_on_removal():
    """The structural clock moves on removes; the history clock deliberately does not."""
    G = _hyper_graph()
    struct_before = G._structure_version
    hist_before = G._version

    G.remove_edges(['h1'])

    assert G._structure_version > struct_before
    assert G._version == hist_before, (
        '_version is a history/audit counter surfaced in snapshots and diffs; '
        'it must not be repurposed as a structural clock'
    )


def test_copy_gets_own_structure_version():
    """``copy()`` must not inherit the ``history=False`` version reset."""
    G = _hyper_graph()
    G.neighbors('a')  # warm before copying

    for kwargs in ({}, {'history': True}):
        c = G.ops.copy(**kwargs)
        assert c._structure_version > 0
        assert sorted(c.neighbors('a')) == ['b', 'c']
        c.remove_edges(['h1'])
        assert c.neighbors('a') == []
        assert sorted(G.neighbors('a')) == ['b', 'c'], 'copy mutation leaked to source'
