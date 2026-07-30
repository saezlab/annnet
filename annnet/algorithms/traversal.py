"""Local neighborhood traversal.

The traversal reads structure through the query facade of the core. It holds no
knowledge of how the core stores a graph.
"""

from ..core import _structure as S


class Traversal:
    """Local neighborhood traversal over the incidence-backed graph.

    A query asks the facade which edges touch the entity, then reads the two
    sides of each of those edges. The facade answers a binary edge from the
    adjacency index in time proportional to the degree, and it answers a
    hyperedge from a list it caches against the structural clock. A graph with
    no hyperedge therefore never pays a full edge scan.
    """

    def _neighbors(self, entity_id, direction):
        """Collect the neighbors of one entity in one direction.

        ``direction`` is ``"out"``, ``"in"``, or ``"both"``. An undirected edge
        answers in both directions. A directed edge answers on the side that
        holds the entity.
        """
        if not S.has_entity(self, entity_id):
            return []
        key = S.entity_key(self, entity_id)
        entity_kind = S.entity_ref(self, key).kind
        wants_out = direction in ('out', 'both')
        wants_in = direction in ('in', 'both')

        found = set()
        for edge_id in S.entity_edges(self, key, direction):
            edge = S.edge_ref(self, edge_id)
            sides = S.edge_endpoints(self, edge_id)

            if not edge.directed:
                # Every other member is a neighbor, whatever the shape of the edge.
                members = set(S.edge_members(self, edge_id))
                if key in members:
                    found |= members - {key}
                continue

            if edge.kind == S.HYPER:
                if wants_out and key in sides.source:
                    found |= sides.target
                elif wants_in and key in sides.target:
                    found |= sides.source
                continue

            if wants_out and key in sides.source:
                found |= sides.target
            if wants_in and key in sides.target:
                # A query for the inward side reaches back along a directed edge.
                # An unqualified query does not, unless the entity is the edge
                # itself, because both sides of that edge describe it.
                if direction == 'in' or entity_kind == S.EDGE_ENTITY:
                    found |= sides.source

        return [S.endpoint_form(self, member) for member in found]

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
        return self._neighbors(entity_id, 'both')

    def out_neighbors(self, vertex_id):
        """Return outward neighbors of a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable via outgoing or undirected edges.
        """
        return self._neighbors(vertex_id, 'out')

    def successors(self, vertex_id):
        """Alias for :meth:`out_neighbors`.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        list[str]
            Successor identifiers.
        """
        return self.out_neighbors(vertex_id)

    def in_neighbors(self, vertex_id):
        """Return inward neighbors of a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable via incoming or undirected edges.
        """
        return self._neighbors(vertex_id, 'in')

    def predecessors(self, vertex_id):
        """Alias for :meth:`in_neighbors`.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        list[str]
            Predecessor identifiers.
        """
        return self.in_neighbors(vertex_id)
