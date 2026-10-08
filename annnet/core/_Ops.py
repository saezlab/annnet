"""Subgraph extraction, copy, reverse, and incidence materialization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

from . import _build, _mutate, _structure
from ._state import GraphState
from ._records import SliceRecord
from .._support.dataframe_backend import (
    clone_dataframe,
    dataframe_memory_usage,
)

if TYPE_CHECKING:
    from .graph import AnnNet


def _as_graph(mixin: Any) -> AnnNet:
    """Say to a type checker what is true at runtime.

    ``Operations`` is a mixin of ``AnnNet``, so the two are one object. A method
    of the mixin that reads a field of the graph passes ``self`` through here.
    """
    return cast('AnnNet', mixin)


def _new_graph(source: Any, *, aspects: Any = None) -> AnnNet:
    """Return an empty graph of the class and the direction of ``source``.

    ``type(source)`` is the graph class, because the mixin is never instantiated
    on its own. Saying so here keeps every construction site in this module one
    call, and typed.
    """
    graph_class = cast('type[AnnNet]', type(source))
    if aspects is None:
        return graph_class(directed=source.directed)
    return graph_class(directed=source.directed, aspects=aspects)


def _share_or_clone_table(df):
    return None if df is None else clone_dataframe(df)


def _require_one_layer_registry(left, right) -> None:
    """Refuse set algebra between two graphs that place their nodes differently.

    A layer coordinate is part of the identity of a node, so two graphs that
    declare different aspects do not name the same nodes even when the bare ids
    match. There is no answer to give, rather than a costly one.
    """
    if left._aspects != right._aspects:
        raise ValueError(
            f'set algebra needs one layer registry, got {left._aspects!r} and {right._aspects!r}'
        )


def _take_attributes(target, source, node_ids, edge_ids) -> None:
    """Copy the attributes of the named elements from one graph to another."""
    if node_ids:
        rows = source._attr_store.node_attr_rows(node_ids)
        if rows:
            target.attrs.update('nodes', rows)
    if edge_ids:
        rows = source._attr_store.edge_attr_rows(edge_ids)
        if rows:
            target.attrs.update('edges', rows)


def _take_slices(target, source) -> None:
    """Add the slice memberships of one graph to another, keeping the target's.

    A slice both graphs declare keeps the attributes it has in the target, and
    takes the members it has in the source. A slice only the source declares
    arrives whole.
    """
    for slice_id, record in source._slices.items():
        held = target._slices.get(slice_id)
        if held is None:
            target._slices[slice_id] = SliceRecord(
                set(record['nodes']), set(record['edges']), dict(record['attributes'])
            )
            continue
        held['nodes'].update(record['nodes'])
        held['edges'].update(record['edges'])
        for key, value in record['attributes'].items():
            held['attributes'].setdefault(key, value)


class Operations(GraphState):
    """Topology materialization and graph-copy operations (mixed into AnnNet)."""

    def _constructor_aspects(self):
        if self._aspects == ('_',):
            return None
        return {aspect: list(self._layers.get(aspect, ())) for aspect in self._aspects}

    def _materialized(self, view, **kwargs) -> AnnNet:
        """Materialize one view through the shared resolver (no selection record)."""
        from . import _materialize

        return view._stable(
            lambda resolved: _materialize.materialize(resolved, record=False, **kwargs)
        )

    def _known_nodes(self, nodes) -> list:
        """The node references the graph holds, in the order given; unknown ones are skipped.

        The subgraph operations have always ignored a node the graph does not
        hold, and a reader that builds a selection from a file relies on it.
        ``G.view`` itself rejects an unknown id.
        """
        graph = _as_graph(self)
        found: list = []
        for item in nodes:
            if isinstance(item, str):
                if graph.has_node(item):
                    found.append(item)
            elif _structure.is_entity_key(item):
                if graph._has_node_layer((item[0], tuple(item[1]))):
                    found.append((item[0], tuple(item[1])))
        return found

    def _known_edges(self, edges) -> list:
        graph = _as_graph(self)
        if edges and all(isinstance(e, int) for e in edges):
            edges = [_structure.edge_at_column(graph, e) for e in edges]
        return [
            eid
            for eid in edges
            if _structure.has_edge(graph, eid) and _structure.carries_structure(graph, eid)
        ]

    def edge_subgraph(self, edges) -> AnnNet:
        """Create a subgraph containing only a specified subset of edges.

        Parameters
        ----------
        edges : Iterable[str] | Iterable[int]
            Edge identifiers or edge indices to retain. Unknown ids are skipped.

        Returns
        -------
        AnnNet
            The selected edges with their full endpoint entities — a
            hyperedge keeps every member, an edge entity brings its backing
            edge — and nothing else. The same selection as
            ``G.view(edges=edges).materialize()``.
        """
        graph = _as_graph(self)
        return self._materialized(graph.view(edges=self._known_edges(list(edges))))

    def subgraph(self, nodes) -> AnnNet:
        """Create a node-induced subgraph.

        Parameters
        ----------
        nodes : Iterable[str | tuple]
            Node identifiers, or explicit ``(node_id, layer)`` placements, to
            retain. A bare id keeps every placement of the node. Unknown ids
            are skipped.

        Returns
        -------
        AnnNet
            The selected nodes and every edge whose complete endpoint set is
            selected (a hyperedge is kept whole or dropped). The same
            selection as ``G.view(nodes=nodes).materialize()``.
        """
        graph = _as_graph(self)
        return self._materialized(graph.view(nodes=self._known_nodes(list(nodes))))

    def extract_subgraph(self, nodes=None, edges=None) -> AnnNet:
        """Create a subgraph based on node and/or edge filters.

        Parameters
        ----------
        nodes : Iterable[str], optional
            Node IDs to include. If None, no node filtering is applied.
        edges : Iterable[str] | Iterable[int], optional
            Edge IDs or indices to include. If None, no edge filtering is applied.

        Returns
        -------
        AnnNet
            The same selection as ``G.view(nodes=..., edges=...).materialize()``:
            both constraints apply, with the closed boundary.
        """
        if nodes is None and edges is None:
            return Operations.copy(self)
        graph = _as_graph(self)
        return self._materialized(
            graph.view(
                nodes=None if nodes is None else self._known_nodes(list(nodes)),
                edges=None if edges is None else self._known_edges(list(edges)),
            )
        )

    # ── Set algebra between two graphs ────────────────────────────────────────

    def merge(self, other) -> AnnNet:
        """Take every element of ``other`` that this graph does not hold.

        This is the in-place union, and it is what ``G |= H`` runs. The graph on
        the left is the answer wherever the two disagree: an element both graphs
        hold keeps the attributes it has here, and only an element this graph
        does not hold arrives with the attributes of ``other``.

        Parameters
        ----------
        other : AnnNet
            The graph to take from. It is not changed.

        Returns
        -------
        AnnNet
            This graph.
        """
        _require_one_layer_registry(self, other)

        entities, edges = _structure.definitions_of(self)
        their_entities, their_edges = _structure.definitions_of(other)

        known_keys = {ref.key for ref in entities}
        new_entities = [ref for ref in their_entities if ref.key not in known_keys]
        known_edges = {edge.id for edge in edges}
        new_edges = [edge for edge in their_edges if edge.id not in known_edges]

        if new_entities or new_edges:
            _build.install_structure(self, definitions=(entities + new_entities, edges + new_edges))

        _take_attributes(
            self, other, {ref.key[0] for ref in new_entities}, {e.id for e in new_edges}
        )
        _take_slices(self, other)
        for key, value in other.graph_attributes.items():
            self.graph_attributes.setdefault(key, value)
        return _as_graph(self)

    def union(self, other) -> AnnNet:
        """Return a graph holding every element of this graph and of ``other``.

        Where the two disagree about one element, this graph is the answer. See
        :meth:`merge`, which is the same operation without the copy.
        """
        return Operations.merge(Operations.copy(self), other)

    def intersection(self, other) -> AnnNet:
        """Return a graph holding the elements that both graphs hold.

        An edge survives only when every node it names does, so an edge both
        graphs hold is dropped when one of its endpoints is not shared.
        """
        _require_one_layer_registry(self, other)
        return Operations.extract_subgraph(
            self,
            nodes=set(_structure.node_ids(self)) & set(_structure.node_ids(other)),
            edges=set(_structure.edge_ids(self)) & set(_structure.edge_ids(other)),
        )

    def difference(self, other) -> AnnNet:
        """Return a graph holding the elements ``other`` does not hold.

        An edge survives only when every node it names does, so an edge that
        keeps its own id loses its place when an endpoint goes.
        """
        _require_one_layer_registry(self, other)
        return Operations.extract_subgraph(
            self,
            nodes=set(_structure.node_ids(self)) - set(_structure.node_ids(other)),
            edges=set(_structure.edge_ids(self)) - set(_structure.edge_ids(other)),
        )

    def symmetric_difference(self, other) -> AnnNet:
        """Return a graph holding the elements exactly one of the two holds."""
        return Operations.merge(
            Operations.difference(self, other), Operations.difference(other, self)
        )

    def reverse(self) -> AnnNet:
        """Return a new graph with all directed edges reversed.

        Returns
        -------
        AnnNet
            A new `AnnNet` instance with reversed directionality where applicable.

        Behavior
        --------
        - **Binary edges:** direction is flipped by swapping source and target.
        - **Directed hyperedges:** `head` and `tail` sets are swapped.
        - **Undirected edges/hyperedges:** unaffected.
        - Edge attributes and metadata are preserved.

        Notes
        -----
        - This operation does not modify the original graph.
        - If the graph is undirected (`self.directed == False`), the result is
          identical to the original.
        - For mixed graphs (directed + undirected edges), only the directed
          ones are reversed.
        """
        g = Operations.copy(self)
        _mutate.reverse_directions(g)
        return g

    def subgraph_from_slice(self, slice_id, *, resolve_slice_weights=True):
        """Create a subgraph induced by a single slice.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        resolve_slice_weights : bool, optional
            If True, an edge whose weight this slice overrides takes the
            override as its stored weight in the copy.

        Returns
        -------
        AnnNet
            The slice's recorded nodes and edges (a slice is a membership
            set, so no edge is induced), with that slice active.

        Raises
        ------
        KeyError
            If the slice does not exist.
        """
        graph = _as_graph(self)
        if slice_id not in graph._slices:
            raise KeyError(f'slice {slice_id} not found')
        weights = {}
        if resolve_slice_weights:
            members = set(graph._slices[slice_id].edges)
            for (sid, eid), attrs in graph._contextual.edge_slice_attrs.items():
                if sid != slice_id or eid not in members:
                    continue
                weight = attrs.get('weight')
                if weight is not None and not (isinstance(weight, float) and weight != weight):
                    weights[eid] = float(weight)
        new = self._materialized(graph.view(slices=slice_id), edge_weights=weights)
        new.slices.active = slice_id
        return new

    def copy(self, history: bool = False):
        """Deep copy of the entire AnnNet.

        Parameters
        ----------
        history : bool, optional
            If True, copy the mutation history and snapshot timeline.
            If False, the new graph starts with a clean history.

        Returns
        -------
        AnnNet
            A new graph with full structural and attribute fidelity.

        Notes
        -----
        O(N) Python, O(nnz) matrix; this path is optimized for speed.
        """
        new_aspects = self._constructor_aspects()
        new = _new_graph(self, aspects=new_aspects)

        _build.install_structure(
            new,
            # A copy of the slot arrays keeps every slot at the address it had,
            # and it costs a memory copy rather than a pass over every edge.
            store=self._store.copy(),
        )
        new.node_aligned = self.node_aligned
        new._next_edge_id = self._next_edge_id

        _build.install_slices(
            new,
            _build.clone_slices(self._slices, drop_attributes=True),
            default=self._default_slice,
            current=self._current_slice,
        )

        new.slice_edge_weights = {lid: m.copy() for lid, m in self.slice_edge_weights.items()}

        # A copy keeps every slot at the address it had, so the columns are
        # copied as they stand rather than replayed row by row.
        new._attr_store.copy_columns_from(self._attr_store)
        new.slice_attributes = _share_or_clone_table(self.slice_attributes)
        new.edge_slice_attributes = _share_or_clone_table(self.edge_slice_attributes)
        new.layer_attributes = _share_or_clone_table(self.layer_attributes)

        new.layers._all_layers = (
            tuple(tuple(x) for x in self.layers._all_layers) if self.layers.list_aspects() else ()
        )
        new.layers._aspect_attrs = {a: m.copy() for a, m in self.layers._aspect_attrs.items()}
        new.layers._layer_attrs = {aa: m.copy() for aa, m in self.layers._layer_attrs.items()}
        new.layers._state_attrs = {k: m.copy() for k, m in self.layers._state_attrs.items()}

        new.graph_attributes = self.graph_attributes.copy()

        new._history_enabled = self._history_enabled
        if history:
            new._history = [h.copy() for h in self._history]
            new._version = self._version
            new._snapshots = list(self._snapshots)
        else:
            new._history = []
            new._version = 0
            new._snapshots = []
        new._history_clock0 = time.perf_counter_ns()
        new._install_history_hooks()
        return new

    def memory_usage(self):
        """Approximate total memory usage in bytes.

        Returns
        -------
        int
            Estimated bytes for the incidence matrix, dictionaries, and attribute DFs.
        """
        matrix_bytes = self._matrix.nnz * (4 + 4 + 4)
        dict_bytes = (
            _structure.entity_count(self)
            + _structure.edge_count(self)
            + sum(
                1
                for ref in _structure.iter_edges(self, include_placeholders=True)
                if ref.declared_weight is not None
            )
        ) * 100
        df_bytes = 0
        for df in (self._node_table, self._edge_table):
            if df is not None:
                df_bytes += dataframe_memory_usage(df)
        return matrix_bytes + dict_bytes + df_bytes

    def get_node_incidence_matrix_as_lists(self, values: bool = False) -> dict:
        """Materialize the node–edge incidence structure as Python lists.

        Parameters
        ----------
        values : bool, optional (default=False)
            - If `False`, returns edge indices incident to each node.
            - If `True`, returns the **matrix values** (usually weights or 1/0) for
            each incident edge instead of the indices.

        Returns
        -------
        dict[str, list]
            A mapping from `node_id` - list of incident edges (indices or values),
            where:
            - Keys are node IDs.
            - Values are lists of edge indices (if `values=False`) or numeric values
            from the incidence matrix (if `values=True`).

        Notes
        -----
        - Internally uses the sparse incidence matrix `self._matrix`, which is stored
        as a SciPy CSR (compressed sparse row) matrix or similar.
        - The incidence matrix `M` is defined as:
            - Rows: nodes
            - Columns: edges
            - Entry `M[i, j]` non-zero ⇨ node `i` is incident to edge `j`.
        - This is a convenient method when you want a native-Python structure for
        downstream use (e.g., exporting, iterating, or visualization).
        """
        result = {}
        graph = _as_graph(self)
        csr = graph._get_csr()
        for i in range(graph._num_entities):
            entry = _structure.entity_key_of_row(self, i)
            node_id = entry[0] if isinstance(entry, tuple) else entry
            start, end = csr.indptr[i], csr.indptr[i + 1]
            result[node_id] = (csr.data[start:end] if values else csr.indices[start:end]).tolist()
        return result

    def node_incidence_matrix(self, values: bool = False, sparse: bool = False):
        """Return the node–edge incidence matrix in sparse or dense form.

        Parameters
        ----------
        values : bool, optional (default=False)
            If `True`, include the numeric values stored in the matrix
            (e.g., weights or signed incidence values). If `False`, convert the
            matrix to a binary mask (1 if incident, 0 if not).
        sparse : bool, optional (default=False)
            - If `True`, return the underlying sparse matrix (CSR).
            - If `False`, return a dense NumPy ndarray.

        Returns
        -------
        scipy.sparse.csr_matrix | numpy.ndarray
            The node–edge incidence matrix `M`:
            - Rows correspond to nodes.
            - Columns correspond to edges.
            - `M[i, j]` ≠ 0 indicates that node `i` is incident to edge `j`.

        Notes
        -----
        - If `values=False`, the returned matrix is binarized before returning.
        - Use `sparse=True` for large graphs to avoid memory blowups.
        - This is the canonical low-level structure that most algorithms (e.g.,
        spectral clustering, Laplacian construction, hypergraph analytics) rely on.
        """
        M = self._matrix.tocsr()
        if not values:
            M = M.copy()
            M.data[:] = 1
        if sparse:
            return M
        rows, cols = M.shape
        estimated_gb = rows * cols * 4 / 1024**3
        if estimated_gb > 2.0:
            raise MemoryError(
                f'Dense conversion would require ~{estimated_gb:.1f} GB '
                f'({rows:,} × {cols:,} float32). Use sparse=True instead.'
            )
        return M.toarray()


_OPS_DELEGATED = {
    'subgraph': 'subgraph',
    'edge_subgraph': 'edge_subgraph',
    'extract': 'extract_subgraph',
    'extract_subgraph': 'extract_subgraph',
    'copy': 'copy',
    'merge': 'merge',
    'union': 'union',
    'intersection': 'intersection',
    'difference': 'difference',
    'symmetric_difference': 'symmetric_difference',
    'reverse': 'reverse',
    'memory_usage': 'memory_usage',
    'incidence': 'node_incidence_matrix',
    'node_incidence_matrix': 'node_incidence_matrix',
    'incidence_as_lists': 'get_node_incidence_matrix_as_lists',
    'get_node_incidence_matrix_as_lists': 'get_node_incidence_matrix_as_lists',
}


class OperationsAccessor:
    """Namespace for structural graph operations (``G.ops``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def __hash__(self) -> int:
        """Structural hash over nodes, edge endpoints/direction, and graph attrs."""
        G = self._G
        node_ids = tuple(sorted(_structure.node_ids(G)))
        edge_defs = []
        for eid in _structure.edge_ids(G):
            S, T = _structure.edge_sides(G, eid)
            edge_defs.append((eid, tuple(sorted(S)), tuple(sorted(T)), G._is_directed_edge(eid)))
        ordered_defs = tuple(sorted(edge_defs))
        graph_meta = (
            tuple(sorted(G.graph_attributes.items())) if hasattr(G, 'graph_attributes') else ()
        )
        return hash((node_ids, ordered_defs, graph_meta))


def _install_ops_delegators():
    for name, target_name in _OPS_DELEGATED.items():

        def _make(tname):
            target = getattr(Operations, tname)

            def _delegator(self, *args, **kwargs):
                return target(self._G, *args, **kwargs)

            _delegator.__name__ = tname
            _delegator.__qualname__ = f'OperationsAccessor.{tname}'
            _delegator.__doc__ = target.__doc__
            return _delegator

        setattr(OperationsAccessor, name, _make(target_name))


_install_ops_delegators()
