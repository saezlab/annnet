"""One cache answers a matrix read.

`FR-019`. Cycle 002 left two. ``MatrixCache`` holds a built matrix against the
clock of the store and extends it when a write only appended edges at the
frontier. ``CacheManager`` held the CSR form, the CSC form and a
boundary-filtered adjacency against a *second* clock, one the graph maintained
itself, so a matrix could be current in one and stale in the other, and the two
between them held the same entries twice.

The named formats stay. What goes is the second holding of them, and the second
clock they were kept against.
"""

from __future__ import annotations

import scipy.sparse as sp

from annnet.core.graph import AnnNet


def _graph() -> AnnNet:
    graph = AnnNet(directed=True)
    graph.add_nodes(['A', 'B', 'C'])
    graph.add_edges('A', 'B', edge_id='e1', weight=2.5)
    graph.add_edges('B', 'C', edge_id='e2', weight=1.5)
    return graph


def _entries(graph) -> dict:
    return graph.matrices.cache._entries


class TestOneCacheHoldsThemAll:
    def test_the_named_formats_land_in_the_matrix_cache(self):
        graph = _graph()
        assert not _entries(graph)
        _ = graph.cache.csr
        _ = graph.cache.csc
        _ = graph.cache.adjacency
        held = set(_entries(graph))
        assert any('csr' in str(key) for key in held)
        assert any('csc' in str(key) for key in held)
        assert any('adjacency' in str(key) for key in held)

    def test_the_manager_holds_nothing_of_its_own(self):
        graph = _graph()
        _ = graph.cache.csr
        _ = graph.cache.csc
        _ = graph.cache.adjacency
        kept = {
            name: value
            for name, value in vars(graph.cache).items()
            if sp.issparse(value) or (isinstance(value, tuple) and any(map(sp.issparse, value)))
        }
        assert kept == {}

    def test_two_reads_give_the_same_object(self):
        graph = _graph()
        assert graph.cache.csr is graph.cache.csr
        assert graph.cache.csc is graph.cache.csc
        assert graph.cache.adjacency is graph.cache.adjacency

    def test_dropping_the_one_cache_drops_the_named_formats_too(self):
        graph = _graph()
        first = graph.cache.csr
        graph.matrices.cache.drop()
        assert graph.cache.has_csr() is False
        assert graph.cache.csr is not first


class TestTheFormatsStillAnswer:
    def test_they_are_the_formats_they_say_they_are(self):
        graph = _graph()
        assert sp.issparse(graph.cache.csr) and graph.cache.csr.format == 'csr'
        assert sp.issparse(graph.cache.csc) and graph.cache.csc.format == 'csc'
        adjacency = graph.cache.adjacency
        assert sp.issparse(adjacency)
        assert adjacency.shape[0] == adjacency.shape[1] == graph.cache.csr.shape[0]

    def test_they_carry_what_the_incidence_matrix_carries(self):
        graph = _graph()
        assert (graph.cache.csr != graph._matrix.tocsr()).nnz == 0
        assert (graph.cache.csc != graph._matrix.tocsc()).nnz == 0

    def test_a_write_moves_them(self):
        graph = _graph()
        before = graph.cache.csr.shape
        graph.add_edges('A', 'C', edge_id='e3')
        assert graph.cache.csr.shape[1] == before[1] + 1

    def test_the_has_flags_follow_the_one_clock(self):
        graph = _graph()
        assert graph.cache.has_csr() is False
        _ = graph.cache.csr
        assert graph.cache.has_csr() is True
        graph.add_edges('A', 'C', edge_id='e3')
        assert graph.cache.has_csr() is False

    def test_invalidate_forgets_what_it_names_and_nothing_else(self):
        graph = _graph()
        _ = graph.cache.csr
        _ = graph.cache.csc
        graph.cache.invalidate(['csr'])
        assert graph.cache.has_csr() is False
        assert graph.cache.has_csc() is True
        graph.cache.invalidate()
        assert graph.cache.has_csc() is False
