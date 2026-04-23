from __future__ import annotations

from typing import TYPE_CHECKING

from .._dataframe_backend import (
    empty_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_append_rows,
)

if TYPE_CHECKING:
    pass


class CacheManager:
    """Materialized matrix cache manager.

    The cache manager owns derived sparse representations such as CSR, CSC, and
    adjacency. Cached values are invalidated against the graph version counter.
    """

    def __init__(self, graph):
        self._G = graph
        self._csr = None
        self._csc = None
        self._adjacency = None
        self._csr_version = None
        self._csc_version = None
        self._adjacency_version = None

    # ==================== CSR/CSC Properties ====================

    @property
    def csr(self):
        """Return the CSR (Compressed Sparse Row) matrix.

        Returns
        -------
        scipy.sparse.csr_matrix

        Notes
        -----
        Built and cached on first access.
        """
        if self._csr is None or self._csr_version != self._G._version:
            self._csr = self._G._matrix.tocsr()
            self._csr_version = self._G._version
        return self._csr

    @property
    def csc(self):
        """Return the CSC (Compressed Sparse Column) matrix.

        Returns
        -------
        scipy.sparse.csc_matrix

        Notes
        -----
        Built and cached on first access.
        """
        if self._csc is None or self._csc_version != self._G._version:
            self._csc = self._G._matrix.tocsc()
            self._csc_version = self._G._version
        return self._csc

    @property
    def adjacency(self):
        """Return the adjacency matrix computed from incidence.

        Returns
        -------
        scipy.sparse.spmatrix

        Notes
        -----
        For incidence matrix `B`, adjacency is computed as `A = B @ B.T`.
        """
        if self._adjacency is None or self._adjacency_version != self._G._version:
            csr = self.csr
            # Adjacency from incidence: A = B @ B.T
            self._adjacency = csr @ csr.T
            self._adjacency_version = self._G._version
        return self._adjacency

    def has_csr(self) -> bool:
        """Check whether a valid CSR cache exists.

        Returns
        -------
        bool
        """
        return self._csr is not None and self._csr_version == self._G._version

    def has_csc(self) -> bool:
        """Check whether a valid CSC cache exists.

        Returns
        -------
        bool
        """
        return self._csc is not None and self._csc_version == self._G._version

    def has_adjacency(self) -> bool:
        """Check whether a valid adjacency cache exists.

        Returns
        -------
        bool
        """
        return self._adjacency is not None and self._adjacency_version == self._G._version

    def get_csr(self):
        """Return the cached CSR matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self.csr

    def get_csc(self):
        """Return the cached CSC matrix.

        Returns
        -------
        scipy.sparse.csc_matrix
        """
        return self.csc

    def get_adjacency(self):
        """Return the cached adjacency matrix.

        Returns
        -------
        scipy.sparse.spmatrix
        """
        return self.adjacency

    # ==================== Cache Management ====================

    def invalidate(self, formats=None):
        """Invalidate cached formats.

        Parameters
        ----------
        formats : list[str], optional
            Formats to invalidate (`'csr'`, `'csc'`, `'adjacency'`).
            If None, invalidate all.

        Returns
        -------
        None
        """
        if formats is None:
            formats = ['csr', 'csc', 'adjacency']

        for fmt in formats:
            if fmt == 'csr':
                self._csr = None
                self._csr_version = None
            elif fmt == 'csc':
                self._csc = None
                self._csc_version = None
            elif fmt == 'adjacency':
                self._adjacency = None
                self._adjacency_version = None

    def build(self, formats=None):
        """Pre-build specified formats (eager caching).

        Parameters
        ----------
        formats : list[str], optional
            Formats to build (`'csr'`, `'csc'`, `'adjacency'`).
            If None, build all.

        Returns
        -------
        None
        """
        if formats is None:
            formats = ['csr', 'csc', 'adjacency']

        for fmt in formats:
            if fmt == 'csr':
                _ = self.csr
            elif fmt == 'csc':
                _ = self.csc
            elif fmt == 'adjacency':
                _ = self.adjacency

    def clear(self):
        """Clear all caches.

        Returns
        -------
        None
        """
        self.invalidate()

    def info(self):
        """Get cache status and memory usage.

        Returns
        -------
        dict
            Status and size information for each cached format.
        """

        def _format_info(matrix, version):
            if matrix is None:
                return {'cached': False}

            size_bytes = 0
            if hasattr(matrix, 'data'):
                size_bytes += matrix.data.nbytes
            if hasattr(matrix, 'indices'):
                size_bytes += matrix.indices.nbytes
            if hasattr(matrix, 'indptr'):
                size_bytes += matrix.indptr.nbytes

            return {
                'cached': True,
                'version': version,
                'size_mb': size_bytes / (1024**2),
                'nnz': matrix.nnz if hasattr(matrix, 'nnz') else 0,
                'shape': matrix.shape,
            }

        return {
            'csr': _format_info(self._csr, self._csr_version),
            'csc': _format_info(self._csc, self._csc_version),
            'adjacency': _format_info(self._adjacency, self._adjacency_version),
        }


def _entity_kind(rec):
    """Return the external kind string for an EntityRecord."""
    k = rec.kind
    if k == 'vertex':
        return 'vertex'
    if k in ('edge', 'edge_entity'):
        return 'edge'
    return k


class IndexManager:
    """Namespace for entity-row and edge-column lookups.

    The manager exposes the incidence-matrix indexing layer without surfacing
    the internal storage dicts directly. It is the preferred public path for
    translating between graph identifiers and matrix coordinates.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Entity (vertex) Indexes ====================

    def entity_to_row(self, entity_id):
        """Map an entity ID to its matrix row index.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        int
            Row index for the entity.

        Raises
        ------
        KeyError
            If the entity is not found.
        """
        ekey = self._G._resolve_entity_key(entity_id)
        rec = self._G._entities.get(ekey)
        if rec is None:
            raise KeyError(f"Entity '{entity_id}' not found")
        return rec.row_idx

    def row_to_entity(self, row):
        """Map a matrix row index to its entity ID.

        Parameters
        ----------
        row : int
            Row index.

        Returns
        -------
        str
            Entity identifier.

        Raises
        ------
        KeyError
            If the row index is not found.
        """
        ekey = self._G._row_to_entity.get(row)
        if ekey is None:
            raise KeyError(f'Row {row} not found')
        return ekey[0]  # bare vid

    def entities_to_rows(self, entity_ids):
        """Batch convert entity IDs to row indices.

        Parameters
        ----------
        entity_ids : Iterable[str]
            Entity identifiers.

        Returns
        -------
        list[int]
        """
        return [self._G._entities[self._G._resolve_entity_key(eid)].row_idx for eid in entity_ids]

    def rows_to_entities(self, rows):
        """Batch convert row indices to entity IDs.

        Parameters
        ----------
        rows : Iterable[int]
            Row indices.

        Returns
        -------
        list[str]
        """
        return [self._G._row_to_entity[r][0] for r in rows]

    # ==================== Edge Indexes ====================

    def edge_to_col(self, edge_id):
        """Map an edge ID to its matrix column index.

        Parameters
        ----------
        edge_id : str
            Edge identifier.

        Returns
        -------
        int
            Column index for the edge.

        Raises
        ------
        KeyError
            If the edge is not found.
        """
        rec = self._G._edges.get(edge_id)
        if rec is None or rec.col_idx < 0:
            raise KeyError(f"Edge '{edge_id}' not found")
        return rec.col_idx

    def col_to_edge(self, col):
        """Map a matrix column index to its edge ID.

        Parameters
        ----------
        col : int
            Column index.

        Returns
        -------
        str
            Edge identifier.

        Raises
        ------
        KeyError
            If the column index is not found.
        """
        eid = self._G._col_to_edge.get(col)
        if eid is None:
            raise KeyError(f'Column {col} not found')
        return eid

    def edges_to_cols(self, edge_ids):
        """Batch convert edge IDs to column indices.

        Parameters
        ----------
        edge_ids : Iterable[str]
            Edge identifiers.

        Returns
        -------
        list[int]
        """
        return [self._G._edges[eid].col_idx for eid in edge_ids]

    def cols_to_edges(self, cols):
        """Batch convert column indices to edge IDs.

        Parameters
        ----------
        cols : Iterable[int]
            Column indices.

        Returns
        -------
        list[str]
        """
        return [self._G._col_to_edge[c] for c in cols]

    # ==================== Utilities ====================

    def entity_type(self, entity_id):
        """Get the entity type for an ID.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        str
            `'vertex'` or `'edge'`.

        Raises
        ------
        KeyError
            If the entity is not found.
        """
        ekey = self._G._resolve_entity_key(entity_id)
        rec = self._G._entities.get(ekey)
        if rec is None:
            raise KeyError(f"Entity '{entity_id}' not found")
        return _entity_kind(rec)

    def is_vertex(self, entity_id):
        """Check whether an entity ID refers to a vertex.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        bool
        """
        return self.entity_type(entity_id) == 'vertex'

    def is_edge_entity(self, entity_id):
        """Check whether an entity ID refers to an edge-entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        bool
        """
        return self.entity_type(entity_id) == 'edge'

    def has_entity(self, entity_id: str) -> bool:
        """Check if an ID exists as any entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        bool
        """
        ekey = self._G._resolve_entity_key(entity_id)
        return ekey in self._G._entities

    def has_vertex(self, vertex_id: str) -> bool:
        """Check if an ID exists and is a vertex.

        Parameters
        ----------
        vertex_id : str
            Vertex identifier.

        Returns
        -------
        bool
        """
        ekey = self._G._resolve_entity_key(vertex_id)
        rec = self._G._entities.get(ekey)
        return rec is not None and rec.kind == 'vertex'

    def has_edge_id(self, edge_id: str) -> bool:
        """Check if an edge ID exists.

        Parameters
        ----------
        edge_id : str
            Edge identifier.

        Returns
        -------
        bool
        """
        rec = self._G._edges.get(edge_id)
        return rec is not None and rec.col_idx >= 0

    def edge_count(self) -> int:
        """Return the number of edges in the graph.

        Returns
        -------
        int
        """
        return len(self._G._col_to_edge)

    def entity_count(self) -> int:
        """Return the number of entities (vertices + edge-entities).

        Returns
        -------
        int
        """
        return len(self._G._entities)

    def vertex_count(self) -> int:
        """Return the number of true vertices (excludes edge-entities).

        Returns
        -------
        int
        """
        return sum(1 for rec in self._G._entities.values() if rec.kind == 'vertex')

    def stats(self):
        """Return index statistics for entities and edges.

        Returns
        -------
        dict
        """
        counts = {'vertex': 0, 'edge': 0}
        for rec in self._G._entities.values():
            k = _entity_kind(rec)
            counts[k] = counts.get(k, 0) + 1
        n_ents = len(self._G._entities)
        n_edges = len(self._G._col_to_edge)
        return {
            'n_entities': n_ents,
            'n_vertices': counts['vertex'],
            'n_edge_entities': counts['edge'],
            'n_edges': n_edges,
            'max_row': n_ents - 1,
            'max_col': n_edges - 1,
        }


class IndexMapping:
    """Internal indexing helpers used by :class:`annnet.core.graph.AnnNet`."""

    def _get_next_edge_id(self) -> str:
        """Generate a fresh edge identifier.

        Returns
        -------
        str
            Fresh ``edge_<n>`` identifier from the monotonic internal counter.
        """
        edge_id = f'edge_{self._next_edge_id}'
        self._next_edge_id += 1
        return edge_id

    def _ensure_vertex_table(self) -> None:
        """Ensure the vertex attribute table exists with a canonical schema.

        Notes
        -----
        Creates an empty dataframe with a canonical ``vertex_id`` column when
        the current vertex attribute table is missing or malformed.
        """
        df = getattr(self, 'vertex_attributes', None)

        needs_init = df is None or 'vertex_id' not in dataframe_columns(df)

        if needs_init:
            self.vertex_attributes = empty_dataframe({'vertex_id': 'text'})

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        """INTERNAL: Ensure a row for ``vertex_id`` exists in the vertex attribute DF."""
        try:
            import sys as _sys

            if isinstance(vertex_id, str):
                vertex_id = _sys.intern(vertex_id)
        except Exception:
            pass

        df = self.vertex_attributes

        # Build/refresh cached id-set (auto-invalidates on DF object change)
        try:
            cached_ids = getattr(self, '_vertex_attr_ids', None)
            cached_df_id = getattr(self, '_vertex_attr_df_id', None)
            if cached_ids is None or cached_df_id != id(df):
                ids = set()
                if df is not None and 'vertex_id' in dataframe_columns(df):
                    ids = {row.get('vertex_id') for row in dataframe_to_rows(df)}
                    ids.discard(None)
                self._vertex_attr_ids = ids
                self._vertex_attr_df_id = id(df)
        except Exception:
            self._vertex_attr_ids = None
            self._vertex_attr_df_id = None

        ids = getattr(self, '_vertex_attr_ids', None)
        if ids is not None and vertex_id in ids:
            return

        is_empty = dataframe_height(df) == 0

        if is_empty:
            self.vertex_attributes = empty_dataframe({'vertex_id': 'text'})
            self.vertex_attributes = dataframe_append_rows(
                self.vertex_attributes,
                [{'vertex_id': vertex_id}],
            )
            try:
                if isinstance(self._vertex_attr_ids, set):
                    self._vertex_attr_ids.add(vertex_id)
                else:
                    self._vertex_attr_ids = {vertex_id}
                self._vertex_attr_df_id = id(self.vertex_attributes)
            except Exception:
                pass
            return

        row = dict.fromkeys(dataframe_columns(df))
        row['vertex_id'] = vertex_id
        self.vertex_attributes = dataframe_append_rows(df, [row])

        try:
            if isinstance(self._vertex_attr_ids, set):
                self._vertex_attr_ids.add(vertex_id)
            else:
                self._vertex_attr_ids = {vertex_id}
            self._vertex_attr_df_id = id(self.vertex_attributes)
        except Exception:
            pass

    def _vertex_key_enabled(self) -> bool:
        return bool(self._vertex_key_fields)

    def _build_key_from_attrs(self, attrs: dict) -> tuple | None:
        if not self._vertex_key_fields:
            return None
        vals = []
        for f in self._vertex_key_fields:
            if f not in attrs or attrs[f] is None:
                return None
            vals.append(attrs[f])
        return tuple(vals)

    def _current_key_of_vertex(self, vertex_id) -> tuple | None:
        if not self._vertex_key_fields:
            return None
        cur = {f: self.attrs.get_attr_vertex(vertex_id, f, None) for f in self._vertex_key_fields}
        return self._build_key_from_attrs(cur)
