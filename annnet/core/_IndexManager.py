class IndexManager:
    """Namespace for index operations.
    Provides clean API over existing dicts.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Entity (vertex) Indexes ====================

    def entity_to_row(self, entity_id):
        """Map entity ID to matrix row index."""
        if entity_id not in self._G.entity_to_idx:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_to_idx[entity_id]

    def row_to_entity(self, row):
        """Map matrix row index to entity ID."""
        if row not in self._G.idx_to_entity:
            raise KeyError(f"Row {row} not found")
        return self._G.idx_to_entity[row]

    def entities_to_rows(self, entity_ids):
        """Batch convert entity IDs to row indices."""
        return [self._G.entity_to_idx[eid] for eid in entity_ids]

    def rows_to_entities(self, rows):
        """Batch convert row indices to entity IDs."""
        return [self._G.idx_to_entity[r] for r in rows]

    # ==================== Edge Indexes ====================

    def edge_to_col(self, edge_id):
        """Map edge ID to matrix column index."""
        if edge_id not in self._G.edge_to_idx:
            raise KeyError(f"Edge '{edge_id}' not found")
        return self._G.edge_to_idx[edge_id]

    def col_to_edge(self, col):
        """Map matrix column index to edge ID."""
        if col not in self._G.idx_to_edge:
            raise KeyError(f"Column {col} not found")
        return self._G.idx_to_edge[col]

    def edges_to_cols(self, edge_ids):
        """Batch convert edge IDs to column indices."""
        return [self._G.edge_to_idx[eid] for eid in edge_ids]

    def cols_to_edges(self, cols):
        """Batch convert column indices to edge IDs."""
        return [self._G.idx_to_edge[c] for c in cols]

    # ==================== Utilities ====================

    def entity_type(self, entity_id):
        """Get entity type ('vertex' or 'edge')."""
        if entity_id not in self._G.entity_types:
            raise KeyError(f"Entity '{entity_id}' not found")
        return self._G.entity_types[entity_id]

    def is_vertex(self, entity_id):
        """Check if entity is a vertex."""
        return self.entity_type(entity_id) == "vertex"

    def is_edge_entity(self, entity_id):
        """Check if entity is an edge-entity (vertex-edge hybrid)."""
        return self.entity_type(entity_id) == "edge"

    def has_entity(self, entity_id: str) -> bool:
        """True if the ID exists as any entity (vertex or edge-entity)."""
        return entity_id in self._G.entity_to_idx

    def has_vertex(self, vertex_id: str) -> bool:
        """True if the ID exists and is a vertex (not an edge-entity)."""
        return self._G.entity_types.get(vertex_id) == "vertex"

    def has_edge_id(self, edge_id: str) -> bool:
        """True if an edge with this ID exists."""
        return edge_id in self._G.edge_to_idx

    def edge_count(self) -> int:
        """Number of edges (columns in incidence)."""
        return len(self._G.edge_to_idx)

    def entity_count(self) -> int:
        """Number of entities (vertices + edge-entities)."""
        return len(self._G.entity_to_idx)

    def vertex_count(self) -> int:
        """Number of true vertices (excludes edge-entities)."""
        return sum(1 for t in self._G.entity_types.values() if t == "vertex")

    def stats(self):
        """Get index statistics."""
        return {
            "n_entities": len(self._G.entity_to_idx),
            "n_vertices": sum(1 for t in self._G.entity_types.values() if t == "vertex"),
            "n_edge_entities": sum(1 for t in self._G.entity_types.values() if t == "edge"),
            "n_edges": len(self._G.edge_to_idx),
            "max_row": max(self._G.idx_to_entity.keys()) if self._G.idx_to_entity else -1,
            "max_col": max(self._G.idx_to_edge.keys()) if self._G.idx_to_edge else -1,
        }
