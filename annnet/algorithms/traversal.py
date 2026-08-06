"""Local neighborhood traversal.

The traversal reads structure through the query facade of the core. It holds no
knowledge of how the core stores a graph.
"""

from ..core import _structure as S


class Traversal:
    """Local neighborhood traversal over the incidence-backed graph.

    Every method here delegates to the structural query facade of the core, which
    owns the traversal itself. The facade answers a binary edge from the adjacency
    index in time proportional to the degree, and it answers a hyperedge from a
    list it caches against the structural clock. A graph with no hyperedge
    therefore never pays a full edge scan.
    """

    def neighbors(self, entity_id):
        """Return adjacent entities for a node or an edge-entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable through incident edges.
        """
        return S.neighbors(self, entity_id, 'both')

    def out_neighbors(self, node_id):
        """Return outward neighbors of a node.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable via outgoing or undirected edges.
        """
        return S.neighbors(self, node_id, 'out')

    def successors(self, node_id):
        """Alias for :meth:`out_neighbors`.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[str]
            Successor identifiers.
        """
        return self.out_neighbors(node_id)

    def in_neighbors(self, node_id):
        """Return inward neighbors of a node.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable via incoming or undirected edges.
        """
        return S.neighbors(self, node_id, 'in')

    def predecessors(self, node_id):
        """Alias for :meth:`in_neighbors`.

        Parameters
        ----------
        node_id : str
            Node identifier.

        Returns
        -------
        list[str]
            Predecessor identifiers.
        """
        return self.in_neighbors(node_id)
