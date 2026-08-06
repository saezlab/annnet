"""Matrix caches, index lookups, and attribute-table row helpers."""

from __future__ import annotations

import sys as _sys

from . import _store as ST, _matrices, _structure
from ._records import _external_entity_kind
from ._stored_kinds import STORED_ENTITY_KIND
from .._support.dataframe_backend import (
    empty_dataframe,
    dataframe_columns,
    dataframe_drop_rows,
    dataframe_append_rows,
    dataframe_column_values,
)

# Tells a missing key from one whose value is ``None``, which every pending row
# carries: the pending buffers are dicts used as insertion-ordered sets.
_ABSENT = object()


class CacheManager:
    """Derived sparse-matrix cache (CSR/CSC/adjacency), keyed on the structural clock.

    Keys on ``_structure_version`` (see ``_derive.bump_structure``), never on
    ``_version``: that is a history counter which does not advance on removes.
    """

    def __init__(self, graph):
        self._G = graph
        self._csr = None
        self._csc = None
        self._adjacency = None
        self._csr_version = None
        self._csc_version = None
        self._adjacency_version = None

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
        if self._csr is None or self._csr_version != self._G._structure_version:
            self._csr = self._G._matrix.tocsr()
            self._csr_version = self._G._structure_version
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
        if self._csc is None or self._csc_version != self._G._structure_version:
            self._csc = self._G._matrix.tocsc()
            self._csc_version = self._G._structure_version
        return self._csc

    @property
    def adjacency(self):
        """Return the adjacency matrix computed from incidence.

        Returns
        -------
        scipy.sparse.sparray

        Notes
        -----
        For incidence matrix `B`, adjacency is computed as `A = B @ B.T`.
        Boundary (half) edges — exchange/demand reactions flagged with the
        ``is_boundary`` edge attribute, which carry a single real endpoint and no
        partner — are dropped from `B` first, so they contribute neither spurious
        inter-metabolite links nor self-loop-like diagonal terms.
        """
        if self._adjacency is None or self._adjacency_version != self._G._structure_version:
            csr = self.csr
            keep = self._non_boundary_cols()
            B = csr if keep is None else csr[:, keep]
            self._adjacency = B @ B.T
            self._adjacency_version = self._G._structure_version
        return self._adjacency

    def _non_boundary_cols(self):
        """Integer column indices to keep for adjacency (boundary columns removed).

        Boundary edges are identified by the ``is_boundary`` edge attribute (set by
        the stoichiometric importers, and round-tripped like any other attribute).
        Returns ``None`` when there are no boundary edges, so the full incidence is
        reused without copying — the overwhelmingly common case.
        """
        g = self._G
        boundary_eids = g.attrs.get_edges_by_attr('is_boundary', True)
        if not boundary_eids:
            return None
        boundary_cols = {
            _structure.edge_column(g, e)
            for e in boundary_eids
            if _structure.has_edge(g, e) and _structure.carries_structure(g, e)
        }
        if not boundary_cols:
            return None
        ncols = _structure.edge_count(g)
        return [c for c in range(ncols) if c not in boundary_cols]

    def has_csr(self) -> bool:
        """Check whether a valid CSR cache exists.

        Returns
        -------
        bool
        """
        return self._csr is not None and self._csr_version == self._G._structure_version

    def has_csc(self) -> bool:
        """Check whether a valid CSC cache exists.

        Returns
        -------
        bool
        """
        return self._csc is not None and self._csc_version == self._G._structure_version

    def has_adjacency(self) -> bool:
        """Check whether a valid adjacency cache exists.

        Returns
        -------
        bool
        """
        return self._adjacency is not None and self._adjacency_version == self._G._structure_version

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
        scipy.sparse.sparray
        """
        return self.adjacency

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
                self._csr = self._csr_version = None
            elif fmt == 'csc':
                self._csc = self._csc_version = None
            elif fmt == 'adjacency':
                self._adjacency = self._adjacency_version = None

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
            if fmt in ('csr', 'csc', 'adjacency'):
                getattr(self, fmt)

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

        def _fmt(matrix, version):
            if matrix is None:
                return {'cached': False}
            size = sum(
                getattr(matrix, a).nbytes
                for a in ('data', 'indices', 'indptr')
                if hasattr(matrix, a)
            )
            return {
                'cached': True,
                'version': version,
                'size_mb': size / (1024**2),
                'nnz': getattr(matrix, 'nnz', 0),
                'shape': matrix.shape,
            }

        return {
            'csr': _fmt(self._csr, self._csr_version),
            'csc': _fmt(self._csc, self._csc_version),
            'adjacency': _fmt(self._adjacency, self._adjacency_version),
        }


# One member list feeds several purpose-built matrices. Each selection below names
# the edges that belong in one of them, so no matrix has to carry a convention
# that another one needs. The graph exposes these as G.B, G.H and G.S.
BINARY = {'kinds': (ST.BINARY, ST.NODE_EDGE), 'signed': True}
HYPERGRAPH = {'kinds': (ST.HYPER,), 'signed': False}
SIGNED = {'kinds': None, 'signed': True}


class MatrixNamespace:
    """The parameterized matrices of one graph, over a cache keyed to its store.

    Each name comes in two forms. The plain one returns a
    :class:`_matrices.MatrixView`, which carries the materialized matrix and the
    maps between an identity and the position it holds in that matrix. A position
    belongs to one matrix and to nothing else, which is why it comes with the
    matrix rather than from the graph.

    The ``_matrix`` form returns the matrix alone. The maps cost an entry per row
    and per column, which is more than the matrix costs to build, so a caller
    that only wants to do arithmetic must not pay for them. The named properties
    of the graph — ``G.B``, ``G.H``, ``G.S``, ``G.A`` and ``G.L`` — are the
    default parameterization of these, and they take the ``_matrix`` form.
    """

    def __init__(self, graph):
        self._G = graph
        self._store = None
        self._cache = None

    @property
    def cache(self):
        """Return the matrix cache, rebound when the graph takes a new store.

        A copy, a selection and a load each install a store of their own, and a
        matrix cached against the old one names nothing in the new one.
        """
        store = _structure.store_of(self._G)
        if self._cache is None or self._store is not store:
            self._store = store
            self._cache = _matrices.MatrixCache(store)
        return self._cache

    def incidence(self, *, kinds=None, signed: bool = True):
        """Return an incidence matrix over the selected edge kinds.

        Parameters
        ----------
        kinds : iterable of int | None, optional
            The stored edge kinds to include. None includes every edge.
        signed : bool, default True
            Carry the coefficient of each member entry. Set False to report
            membership alone.

        Returns
        -------
        MatrixView
        """
        return self.cache.incidence(kinds=kinds, signed=signed)

    def incidence_matrix(self, *, kinds=None, signed: bool = True):
        """Return an incidence matrix alone, without the maps a view carries.

        Returns
        -------
        scipy.sparse.csc_array
        """
        return self.cache.incidence_matrix(kinds=kinds, signed=signed)

    def binary(self):
        """Return the incidence matrix of the binary edges, with its maps."""
        return self.incidence(**BINARY)

    def binary_matrix(self):
        """Return the incidence matrix of the binary edges (``G.B``)."""
        return self.incidence_matrix(**BINARY)

    def hypergraph(self):
        """Return the incidence matrix of the hyperedges, with its maps."""
        return self.incidence(**HYPERGRAPH)

    def hypergraph_matrix(self):
        """Return the incidence matrix of the hyperedges (``G.H``)."""
        return self.incidence_matrix(**HYPERGRAPH)

    def signed(self):
        """Return the coefficient incidence matrix of every edge, with its maps."""
        return self.incidence(**SIGNED)

    def signed_matrix(self):
        """Return the coefficient incidence matrix of every edge (``G.S``)."""
        return self.incidence_matrix(**SIGNED)

    def adjacency(self, **kwargs):
        """Return the adjacency matrix with its maps (``G.A``).

        A self-loop lands on the diagonal. A boundary edge joins nothing, so it
        is left out, and so is a hyperedge: projecting one onto pairs is a choice
        that belongs to the caller. Pass ``kinds`` to make that choice.

        Returns
        -------
        MatrixView
        """
        return self.cache.adjacency(**kwargs)

    def adjacency_matrix(self, **kwargs):
        """Return the adjacency matrix alone (``G.A``).

        Returns
        -------
        scipy.sparse.csr_array
        """
        return self.cache.adjacency_matrix(**kwargs)

    def laplacian(self, **kwargs):
        """Return the Laplacian with its maps.

        It is the degree matrix minus the adjacency.

        Returns
        -------
        MatrixView
        """
        return self.cache.laplacian(**kwargs)

    def laplacian_matrix(self, **kwargs):
        """Return the Laplacian alone (``G.L``).

        Returns
        -------
        scipy.sparse.csr_array
        """
        return self.cache.laplacian_matrix(**kwargs)

    def drop(self) -> None:
        """Forget every cached matrix. A cache is always safe to drop."""
        if self._cache is not None:
            self._cache.drop()


def _entity_kind(kind):
    """Return the word this API uses for one entity kind of the facade."""
    return _external_entity_kind(STORED_ENTITY_KIND.get(kind, kind))


class IndexManager:
    """Read-only selectors translating graph ids to/from matrix coordinates."""

    def __init__(self, graph):
        self._G = graph

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
        return _structure.entity_row(self._G, entity_id)

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
        return _structure.entity_key_of_row(self._G, row)[0]

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
        return [_structure.entity_row(self._G, eid) for eid in entity_ids]

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
        return [_structure.entity_key_of_row(self._G, r)[0] for r in rows]

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
        column = (
            _structure.edge_column(self._G, edge_id)
            if _structure.has_edge(self._G, edge_id)
            else -1
        )
        if column < 0:
            raise KeyError(f"Edge '{edge_id}' not found")
        return column

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
        return _structure.edge_at_column(self._G, col)

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
        return [_structure.edge_column(self._G, eid) for eid in edge_ids]

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
        return [_structure.edge_at_column(self._G, c) for c in cols]

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
        return _entity_kind(_structure.entity_ref(self._G, entity_id).kind)

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
        return _structure.has_entity(self._G, entity_id)

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
        if not _structure.has_entity(self._G, vertex_id):
            return False
        return _structure.entity_ref(self._G, vertex_id).kind == _structure.NODE

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
        return _structure.has_edge(self._G, edge_id) and _structure.carries_structure(
            self._G, edge_id
        )

    def edge_count(self) -> int:
        """Return the number of edges in the graph.

        Returns
        -------
        int
        """
        return _structure.edge_count(self._G)

    def entity_count(self) -> int:
        """Return the number of entities (vertices + edge-entities).

        Returns
        -------
        int
        """
        return _structure.entity_count(self._G)

    def vertex_count(self) -> int:
        """Return the number of true vertices (excludes edge-entities).

        Returns
        -------
        int
        """
        return _structure.node_count(self._G)

    def stats(self):
        """Return index statistics for entities and edges.

        Returns
        -------
        dict
        """
        counts = {'vertex': 0, 'edge': 0}
        for ref in _structure.iter_entities(self._G):
            k = _entity_kind(ref.kind)
            counts[k] = counts.get(k, 0) + 1
        n_ents = _structure.entity_count(self._G)
        n_edges = _structure.edge_count(self._G)
        return {
            'n_entities': n_ents,
            'n_vertices': counts['vertex'],
            'n_edge_entities': counts['edge'],
            'n_edges': n_edges,
            'max_row': n_ents - 1,
            'max_col': n_edges - 1,
        }


class IndexMapping:
    """Internal id-generation and attribute-row helpers mixed into ``AnnNet``."""

    def _get_next_edge_id(self) -> str:
        edge_id = f'edge_{self._next_edge_id}'
        self._next_edge_id += 1
        return edge_id

    # --- Buffered attribute-row insertion and removal --------------------------
    # ``add_vertices``/``add_edges`` must guarantee a row per entity in obs/var
    # (the anndata symmetry). Appending one row to a columnar (Polars) table per
    # call is O(n) -> O(n^2). Instead we buffer new id-only rows and flush them in
    # one batch when the table is read (via the vertex_attributes/edge_attributes
    # property getters on AnnNet). Membership stays O(1) via a maintained id-set.
    #
    # A removal is buffered the same way and for the same reason. Filtering a
    # frame costs the call rather than the rows, so a hundred single removes paid
    # a hundred filters where the set they name needs one. The two buffers flush
    # together: the drops go first, then the appends, so an id removed and added
    # again before a read ends with the fresh row and not the stale one.

    def _ensure_vertex_table(self) -> None:
        df = self._vertex_attributes
        if df is None or 'vertex_id' not in dataframe_columns(df):
            self.vertex_attributes = empty_dataframe({'vertex_id': 'text'})

    def _vertex_id_set(self):
        ids = self._built_vertex_id_set()
        if ids is None:
            df = self._vertex_attributes
            ids = set(dataframe_column_values(df, 'vertex_id'))
            ids.discard(None)
            ids.difference_update(self._pending_vertex_drops)
            ids.update(self._pending_vertex_ids)
            self._vertex_attr_ids = ids
            self._vertex_attr_df_id = id(df)
        return ids

    def _built_vertex_id_set(self):
        """The id set if it is built and current, and ``None`` if it is neither.

        Building it is a pass over the table, and a drop does not need one: the
        build subtracts what is pending, so a drop recorded against a set that
        was never built is still true of the set when it is.
        """
        ids = getattr(self, '_vertex_attr_ids', None)
        if ids is None or getattr(self, '_vertex_attr_df_id', None) != id(self._vertex_attributes):
            return None
        return ids

    def _ensure_vertex_row(self, vertex_id: str) -> None:
        if isinstance(vertex_id, str):
            try:
                vertex_id = _sys.intern(vertex_id)
            except (AttributeError, TypeError):
                pass
        ids = self._vertex_id_set()
        if vertex_id in ids:
            return
        ids.add(vertex_id)
        self._pending_vertex_ids[vertex_id] = None

    def _drop_vertex_rows(self, vertex_ids) -> None:
        """Take the rows of these vertices out of the table, at the next read."""
        drop = set(vertex_ids)
        if not drop:
            return
        ids = self._built_vertex_id_set()
        if ids is not None:
            ids.difference_update(drop)
        pend = self._pending_vertex_ids
        # A row that was never written needs no filter to take it away again, so
        # only a vertex the table already holds is worth a drop. The walk goes
        # over what is being dropped and not over what is pending, because a set
        # operation against the keys of a dict copies the whole dict first.
        for vertex_id in drop:
            if pend.pop(vertex_id, _ABSENT) is _ABSENT:
                self._pending_vertex_drops.add(vertex_id)

    def _flush_vertex_rows(self) -> None:
        drop = self._pending_vertex_drops
        pend = self._pending_vertex_ids
        if not drop and not pend:
            return
        self._pending_vertex_drops = set()
        self._pending_vertex_ids = {}
        df = self._vertex_attributes
        if drop:
            df = dataframe_drop_rows(df, 'vertex_id', drop)
        if pend:
            if df is None or 'vertex_id' not in dataframe_columns(df):
                df = empty_dataframe({'vertex_id': 'text'}, backend=self._annotations_backend)
                cols = ['vertex_id']
            else:
                cols = list(dataframe_columns(df))
            rows = [{**dict.fromkeys(cols), 'vertex_id': v} for v in pend]
            df = dataframe_append_rows(df, rows)
        self._vertex_attributes = df
        self._vertex_attr_df_id = id(self._vertex_attributes)

    def _edge_id_set(self):
        ids = self._built_edge_id_set()
        if ids is None:
            df = self._edge_attributes
            ids = set(dataframe_column_values(df, 'edge_id'))
            ids.discard(None)
            ids.difference_update(self._pending_edge_drops)
            ids.update(self._pending_edge_ids)
            self._edge_attr_ids = ids
            self._edge_attr_df_id = id(df)
        return ids

    def _built_edge_id_set(self):
        """The id set if it is built and current, and ``None`` if it is neither."""
        ids = getattr(self, '_edge_attr_ids', None)
        if ids is None or getattr(self, '_edge_attr_df_id', None) != id(self._edge_attributes):
            return None
        return ids

    def _ensure_edge_row(self, edge_id: str) -> None:
        if isinstance(edge_id, str):
            try:
                edge_id = _sys.intern(edge_id)
            except (AttributeError, TypeError):
                pass
        ids = self._edge_id_set()
        if edge_id in ids:
            return
        ids.add(edge_id)
        self._pending_edge_ids[edge_id] = None

    def _ensure_edge_rows_bulk(self, edge_ids) -> None:
        if not edge_ids:
            return
        ids = self._edge_id_set()
        new = [eid for eid in edge_ids if eid not in ids]
        if not new:
            return
        ids.update(new)
        self._pending_edge_ids.update(dict.fromkeys(new))

    def _drop_edge_rows(self, edge_ids) -> None:
        """Take the rows of these edges out of the table, at the next read."""
        drop = set(edge_ids)
        if not drop:
            return
        ids = self._built_edge_id_set()
        if ids is not None:
            ids.difference_update(drop)
        pend = self._pending_edge_ids
        # A row that was never written needs no filter to take it away again, so
        # only an edge the table already holds is worth a drop. The walk goes
        # over what is being dropped and not over what is pending, because a set
        # operation against the keys of a dict copies the whole dict first.
        for edge_id in drop:
            if pend.pop(edge_id, _ABSENT) is _ABSENT:
                self._pending_edge_drops.add(edge_id)

    def _flush_edge_rows(self) -> None:
        drop = self._pending_edge_drops
        pend = self._pending_edge_ids
        if not drop and not pend:
            return
        self._pending_edge_drops = set()
        self._pending_edge_ids = {}
        df = self._edge_attributes
        if drop:
            df = dataframe_drop_rows(df, 'edge_id', drop)
        if pend:
            if df is None or 'edge_id' not in dataframe_columns(df):
                df = empty_dataframe({'edge_id': 'text'}, backend=self._annotations_backend)
                cols = ['edge_id']
            else:
                cols = list(dataframe_columns(df))
            rows = [{**dict.fromkeys(cols), 'edge_id': eid} for eid in pend]
            df = dataframe_append_rows(df, rows)
        self._edge_attributes = df
        self._edge_attr_df_id = id(self._edge_attributes)

    def _drop_edge_slice_rows(self, edge_ids) -> None:
        """Take the rows of these edges out of the edge-slice table, at the next read.

        The table has no insertion buffer to match this one, because a row of it
        is written whole rather than owed by an id, and the write that owns it
        reads the table first.
        """
        self._pending_edge_slice_drops.update(edge_ids)

    def _flush_edge_slice_rows(self) -> None:
        drop = self._pending_edge_slice_drops
        if not drop:
            return
        self._pending_edge_slice_drops = set()
        self._edge_slice_attributes = dataframe_drop_rows(
            self._edge_slice_attributes, 'edge_id', drop
        )

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
