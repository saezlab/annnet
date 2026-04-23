"""Unit tests for annnet/algorithms/traversal.py."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from annnet.core.graph import AnnNet


class TestNeighborsBinary(unittest.TestCase):
    """neighbors() on binary edges."""

    def test_directed_out_edge(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        # A has outgoing edge → B is a neighbor of A
        self.assertIn('B', G.neighbors('A'))

    def test_directed_no_reverse(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        # directed: B does not see A as neighbor (not an edge-entity)
        self.assertNotIn('A', G.neighbors('B'))

    def test_undirected_both_directions(self):
        G = AnnNet(directed=False)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertIn('B', G.neighbors('A'))
        self.assertIn('A', G.neighbors('B'))

    def test_multi_neighbors(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges('A', 'B')
        G.add_edges('A', 'C')
        nbrs = G.neighbors('A')
        self.assertIn('B', nbrs)
        self.assertIn('C', nbrs)
        self.assertEqual(len(nbrs), 2)

    def test_unknown_vertex_returns_empty(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        self.assertEqual(G.neighbors('Z'), [])

    def test_isolated_vertex_no_neighbors(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertEqual(G.neighbors('B'), [])

    def test_self_loop(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_edges('A', 'A')
        # Self-loop: A is its own neighbor
        self.assertIn('A', G.neighbors('A'))


class TestOutNeighbors(unittest.TestCase):
    """out_neighbors() under directed semantics."""

    def test_directed_source(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertIn('B', G.out_neighbors('A'))

    def test_directed_target_has_none(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertEqual(G.out_neighbors('B'), [])

    def test_undirected_both_are_out_neighbors(self):
        G = AnnNet(directed=False)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertIn('B', G.out_neighbors('A'))
        self.assertIn('A', G.out_neighbors('B'))

    def test_unknown_vertex_returns_empty(self):
        G = AnnNet(directed=True)
        self.assertEqual(G.out_neighbors('X'), [])


class TestInNeighbors(unittest.TestCase):
    """in_neighbors() under directed semantics."""

    def test_directed_target_sees_source(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertIn('A', G.in_neighbors('B'))

    def test_directed_source_sees_none(self):
        G = AnnNet(directed=True)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertEqual(G.in_neighbors('A'), [])

    def test_undirected_both_are_in_neighbors(self):
        G = AnnNet(directed=False)
        G.add_vertices('A')
        G.add_vertices('B')
        G.add_edges('A', 'B')
        self.assertIn('A', G.in_neighbors('B'))
        self.assertIn('B', G.in_neighbors('A'))

    def test_chain(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges('A', 'B')
        G.add_edges('B', 'C')
        # B has in-neighbor A; C has in-neighbor B
        self.assertIn('A', G.in_neighbors('B'))
        self.assertIn('B', G.in_neighbors('C'))
        self.assertEqual(G.in_neighbors('A'), [])

    def test_unknown_vertex_returns_empty(self):
        G = AnnNet(directed=True)
        self.assertEqual(G.in_neighbors('X'), [])


class TestSuccessorsPredecessors(unittest.TestCase):
    """successors() and predecessors() mirror out/in_neighbors."""

    def test_successors_match_out_neighbors(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges('A', 'B')
        G.add_edges('A', 'C')
        self.assertEqual(set(G.successors('A')), set(G.out_neighbors('A')))

    def test_predecessors_match_in_neighbors(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges('B', 'A')
        G.add_edges('C', 'A')
        self.assertEqual(set(G.predecessors('A')), set(G.in_neighbors('A')))


class TestHyperedgeTraversal(unittest.TestCase):
    """Traversal through directed and undirected hyperedges."""

    def test_directed_hyperedge_head_to_tail(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        # A is in head, B and C are in tail
        G.add_edges(src=['A'], tgt=['B', 'C'])
        nbrs = G.neighbors('A')
        self.assertIn('B', nbrs)
        self.assertIn('C', nbrs)

    def test_directed_hyperedge_tail_sees_head(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges(src=['A'], tgt=['B', 'C'])
        nbrs = G.neighbors('B')
        self.assertIn('A', nbrs)

    def test_undirected_hyperedge_all_see_each_other(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges(src=['A', 'B', 'C'])
        for v in ['A', 'B', 'C']:
            nbrs = G.neighbors(v)
            others = {'A', 'B', 'C'} - {v}
            for other in others:
                self.assertIn(other, nbrs)

    def test_directed_hyperedge_out_neighbors(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges(src=['A'], tgt=['B', 'C'])
        out = G.out_neighbors('A')
        self.assertIn('B', out)
        self.assertIn('C', out)
        self.assertEqual(G.out_neighbors('B'), [])

    def test_directed_hyperedge_in_neighbors(self):
        G = AnnNet(directed=True)
        for v in ['A', 'B', 'C']:
            G.add_vertices(v)
        G.add_edges(src=['A'], tgt=['B', 'C'])
        inn = G.in_neighbors('B')
        self.assertIn('A', inn)


if __name__ == '__main__':
    unittest.main()
