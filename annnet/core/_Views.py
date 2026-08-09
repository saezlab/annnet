"""Lazy graph views and materialized table builders."""

import numpy as np
import scipy.sparse as sp

from . import _mutate, _structure
from ._attrs import read_only
from ._state import GraphState
from ._records import _external_entity_kind
from ._stored_kinds import STORED_EDGE_KIND, STORED_ENTITY_KIND
from .._support.dataframe_backend import (
    clone_dataframe,
    empty_dataframe,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_in,
    dataframe_from_rows,
)


class GraphView:
    """Lazy, filtered view into a graph; materialize() for a concrete subgraph."""

    def __init__(self, graph, nodes=None, edges=None, slices=None, predicate=None):
        self._graph = graph
        self._nodes_filter = nodes
        self._edges_filter = edges
        self._predicate = predicate
        if slices is None:
            self._slices = None
        elif isinstance(slices, str):
            self._slices = [slices]
        else:
            self._slices = list(slices)
        self._node_ids_cache = None
        self._edge_ids_cache = None
        self._computed = False

    @property
    def obs(self):
        """Return the filtered node attribute table for this view.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Materialized from the node table of the graph and filtered by the
        node ids of this view. It is a table for the caller, not the storage
        of the graph.
        """
        node_ids = self.node_ids
        if node_ids is None:
            return clone_dataframe(self._graph._node_table)
        return dataframe_filter_in(self._graph._node_table, 'node_id', node_ids)

    @property
    def var(self):
        """Return the filtered edge attribute table for this view.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Materialized from the edge table of the graph and filtered by the edge
        ids of this view. It is a table for the caller, not the storage of the
        graph.
        """
        edge_ids = self.edge_ids
        if edge_ids is None:
            return clone_dataframe(self._graph._edge_table)
        return dataframe_filter_in(self._graph._edge_table, 'edge_id', edge_ids)

    @property
    def X(self):
        """Return the filtered incidence matrix subview.

        Returns
        -------
        scipy.sparse.dok_matrix
        """
        node_ids = self.node_ids
        edge_ids = self.edge_ids
        if node_ids is not None:
            rows = [
                _structure.entity_row(self._graph, nid)
                for nid in node_ids
                if _structure.has_entity(self._graph, nid)
            ]
        else:
            rows = list(range(self._graph._matrix.shape[0]))
        if edge_ids is not None:
            cols = []
            for eid in edge_ids:
                if not _structure.has_edge(self._graph, eid):
                    continue
                column = _structure.edge_column(self._graph, eid)
                if column >= 0:
                    cols.append(column)
        else:
            cols = list(range(self._graph._matrix.shape[1]))
        if rows and cols:
            return self._graph._matrix[rows, :][:, cols]
        return sp.dok_array((len(rows), len(cols)), dtype=self._graph._matrix.dtype)

    @property
    def node_ids(self):
        """Get filtered node IDs (cached).

        Returns
        -------
        set[str] | None
            None means no node filter (full graph).
        """
        if not self._computed:
            self._compute_ids()
        return self._node_ids_cache

    @property
    def edge_ids(self):
        """Get filtered edge IDs (cached).

        Returns
        -------
        set[str] | None
            None means no edge filter (full graph).
        """
        if not self._computed:
            self._compute_ids()
        return self._edge_ids_cache

    @property
    def node_count(self):
        """Return the number of nodes in this view.

        Returns
        -------
        int
        """
        node_ids = self.node_ids
        if node_ids is None:
            return _structure.node_count(self._graph)
        return len(node_ids)

    @property
    def edge_count(self):
        """Return the number of edges in this view.

        Returns
        -------
        int
        """
        edge_ids = self.edge_ids
        if edge_ids is None:
            return _structure.edge_count(self._graph)
        return len(edge_ids)

    def _compute_ids(self):
        node_ids = None
        edge_ids = None

        if self._slices is not None:
            node_ids = set()
            edge_ids = set()
            for slice_id in self._slices:
                if slice_id in self._graph._slices:
                    node_ids.update(self._graph._slices[slice_id]['nodes'])
                    edge_ids.update(self._graph._slices[slice_id]['edges'])

        if self._nodes_filter is not None:
            candidate_nodes = (
                node_ids
                if node_ids is not None
                else {
                    ref.id
                    for ref in _structure.iter_entities(self._graph)
                    if ref.kind == _structure.NODE
                }
            )
            if callable(self._nodes_filter):
                filtered = set()
                for vid in candidate_nodes:
                    try:
                        if self._nodes_filter(vid):
                            filtered.add(vid)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                node_ids = filtered
            else:
                specified = set(self._nodes_filter)
                node_ids = (
                    (node_ids & specified)
                    if node_ids is not None
                    else (specified & candidate_nodes)
                )

        if self._edges_filter is not None:
            candidate_edges = (
                edge_ids
                if edge_ids is not None
                else {ref.id for ref in _structure.iter_edges(self._graph)}
            )
            if callable(self._edges_filter):
                filtered = set()
                for eid in candidate_edges:
                    try:
                        if self._edges_filter(eid):
                            filtered.add(eid)
                    except (AttributeError, KeyError, TypeError, ValueError):
                        pass
                edge_ids = filtered
            else:
                specified = set(self._edges_filter)
                edge_ids = (
                    (edge_ids & specified)
                    if edge_ids is not None
                    else (specified & candidate_edges)
                )

        if self._predicate is not None and node_ids is not None:
            filtered = set()
            for vid in node_ids:
                try:
                    if self._predicate(vid):
                        filtered.add(vid)
                except (AttributeError, KeyError, TypeError, ValueError):
                    pass
            node_ids = filtered

        if node_ids is not None and edge_ids is not None:
            filtered = set()
            for eid in edge_ids:
                if not _structure.has_edge(self._graph, eid):
                    continue
                if not _structure.carries_structure(self._graph, eid):
                    continue
                sides = _structure.edge_sides(self._graph, eid)
                if not (sides.source <= node_ids and sides.target <= node_ids):
                    continue
                # A binary edge needs both of its sides. A hyperedge with no target
                # side is undirected, and its one side is the whole edge.
                is_hyper = _structure.edge_ref(self._graph, eid).kind == _structure.HYPER
                if is_hyper or (sides.source and sides.target):
                    filtered.add(eid)
            edge_ids = filtered

        self._node_ids_cache = node_ids
        self._edge_ids_cache = edge_ids
        self._computed = True

    def edges_df(self, **kwargs):
        """Return an edge DataFrame view filtered to this view's edges.

        Parameters
        ----------
        **kwargs
            Passed through to `AnnNet.edges_view()`.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.edges_view()` and then filters by the view's edge IDs.
        """
        df = self._graph.views.edges(**kwargs)
        edge_ids = self.edge_ids
        if edge_ids is not None:
            df = dataframe_filter_in(df, 'edge_id', edge_ids)
        return df

    def nodes_df(self, **kwargs):
        """Return a node DataFrame view filtered to this view's nodes.

        Parameters
        ----------
        **kwargs
            Passed through to `AnnNet.nodes_view()`.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Uses `AnnNet.nodes_view()` and then filters by the view's node IDs.
        """
        df = self._graph.views.nodes(**kwargs)
        node_ids = self.node_ids
        if node_ids is not None:
            df = dataframe_filter_in(df, 'node_id', node_ids)
        return df

    def materialize(self, copy_attributes=True):
        """Create a concrete subgraph from this view.

        Parameters
        ----------
        copy_attributes : bool, optional
            If True, copy node/edge attributes into the new graph.

        Returns
        -------
        AnnNet
            Materialized subgraph.
        """
        subG = self._graph.ops.extract_subgraph(nodes=self.node_ids, edges=self.edge_ids)
        if copy_attributes:
            return subG

        # The rows stay, because an element of the subgraph is a row of its
        # tables. What goes is every attribute the caller asked not to carry
        # over, which is every column beside the id.
        subG._attr_store.drop_node_columns()
        subG._attr_store.drop_edge_columns()
        return subG

    def subview(self, nodes=None, edges=None, slices=None, predicate=None):
        """Create a new GraphView by further restricting this view.

        Parameters
        ----------
        nodes : Iterable[str] | callable | None
            Node IDs or predicate; intersects with current view if provided.
        edges : Iterable[str] | callable | None
            Edge IDs or predicate; intersects with current view if provided.
        slices : Iterable[str] | None
            Slice IDs to include. Defaults to current view's slices if None.
        predicate : callable | None
            Additional node predicate applied in conjunction with existing filters.

        Returns
        -------
        GraphView

        Notes
        -----
        Predicates are combined with logical AND.
        """
        base_nodes = self.node_ids
        base_edges = self.edge_ids

        if nodes is None:
            new_nodes, node_pred = base_nodes, None
        elif callable(nodes):
            new_nodes, node_pred = base_nodes, nodes
        else:
            to_set = set(nodes)
            new_nodes = (set(base_nodes) & to_set) if base_nodes is not None else to_set
            node_pred = None

        if edges is None or callable(edges):
            new_edges = base_edges
        else:
            to_set = set(edges)
            new_edges = (set(base_edges) & to_set) if base_edges is not None else to_set

        new_slices = slices if slices is not None else (self._slices if self._slices else None)

        def combined_pred(v):
            ok = True
            for pred in (self._predicate, predicate, node_pred):
                if pred:
                    try:
                        ok = ok and bool(pred(v))
                    except (AttributeError, TypeError, ValueError):
                        ok = False
            return ok

        final_pred = combined_pred if (self._predicate or predicate or node_pred) else None
        return GraphView(
            self._graph,
            nodes=new_nodes,
            edges=new_edges,
            slices=new_slices,
            predicate=final_pred,
        )

    def summary(self):
        """Return a human-readable summary of this view.

        Returns
        -------
        str
        """
        lines = [
            'GraphView Summary',
            '─' * 30,
            f'nodes: {self.node_count}',
            f'Edges: {self.edge_count}',
        ]
        filters = []
        if self._slices:
            filters.append(f'slices={self._slices}')
        if self._nodes_filter:
            filters.append(
                'nodes=<predicate>'
                if callable(self._nodes_filter)
                else f'nodes={len(list(self._nodes_filter))} specified'
            )
        if self._edges_filter:
            filters.append(
                'edges=<predicate>'
                if callable(self._edges_filter)
                else f'edges={len(list(self._edges_filter))} specified'
            )
        if self._predicate:
            filters.append('predicate=<function>')
        lines.append(f'Filters: {", ".join(filters)}' if filters else 'Filters: None (full graph)')
        return '\n'.join(lines)

    def __repr__(self):
        return f'GraphView(nodes={self.node_count}, edges={self.edge_count})'

    def __len__(self):
        return self.node_count


class ViewsClass(GraphState):
    """Materialized table builders mixed into ``AnnNet``."""

    def edges_view(
        self,
        slice=None,
        include_directed=True,
        include_weight=True,
        resolved_weight=True,
        copy=True,
    ):
        """Build a DataFrame view of edges with optional slice join.

        Parameters
        ----------
        slice : str, optional
            Slice ID to join per-slice attributes.
        include_directed : bool, optional
            Include directedness column.
        include_weight : bool, optional
            Include global weight column.
        resolved_weight : bool, optional
            Include effective weight (slice override if present).
        copy : bool, optional
            Return a cloned DataFrame if True.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Vectorized implementation avoids per-edge scans.
        """
        _edge_refs = list(_structure.iter_edges(self))
        if not _edge_refs:
            return empty_dataframe({'edge_id': 'text', 'kind': 'text', 'ml_kind': 'text'})

        eids_raw = [ref.id for ref in _edge_refs]
        eids_str = [str(eid) for eid in eids_raw]

        kinds = ['hyper' if ref.kind == _structure.HYPER else 'binary' for ref in _edge_refs]
        ml_kinds = [ref.ml_kind for ref in _edge_refs]

        need_global = include_weight or resolved_weight
        global_w = [ref.weight for ref in _edge_refs] if need_global else None
        dirs = [ref.directed for ref in _edge_refs] if include_directed else None

        src, tgt, etype, head, tail, members = [], [], [], [], [], []
        for ref in _edge_refs:
            sides = _structure.edge_sides(self, ref.id)
            if ref.kind == _structure.HYPER:
                src_vals = tuple(str(x) for x in sorted(sides.source, key=str))
                if sides.target:
                    tgt_vals = tuple(str(x) for x in sorted(sides.target, key=str))
                    head.append(src_vals)
                    tail.append(tgt_vals)
                    members.append(None)
                    src.append('|'.join(src_vals))
                    tgt.append('|'.join(tgt_vals))
                else:
                    head.append(None)
                    tail.append(None)
                    members.append(src_vals)
                    src.append('|'.join(src_vals))
                    tgt.append(None)
                etype.append(None)
            else:
                one_source = next(iter(sides.source), None)
                one_target = next(iter(sides.target), None)
                src.append(str(one_source) if one_source is not None else None)
                tgt.append(str(one_target) if one_target is not None else None)
                etype.append(STORED_EDGE_KIND.get(ref.kind, ref.kind))
                head.append(None)
                tail.append(None)
                members.append(None)

        edge_attrs_map = self._attr_store.edge_attr_rows()
        slice_attrs_map = {}
        if slice is not None:
            for row in dataframe_to_rows(self.edge_slice_attributes):
                if row.get('slice_id') != slice:
                    continue
                eid = row.get('edge_id')
                if eid is None:
                    continue
                slice_attrs_map[str(eid)] = {
                    f'slice_{k}': v for k, v in row.items() if k not in {'slice_id', 'edge_id'}
                }

        out_rows = []
        for idx, eid in enumerate(eids_str):
            row = {
                'edge_id': eid,
                'kind': kinds[idx],
                'ml_kind': ml_kinds[idx],
                'source': src[idx],
                'target': tgt[idx],
                'edge_type': etype[idx],
                'head': list(head[idx]) if head[idx] is not None else None,
                'tail': list(tail[idx]) if tail[idx] is not None else None,
                'members': list(members[idx]) if members[idx] is not None else None,
            }
            if include_directed:
                row['directed'] = dirs[idx]
            if include_weight:
                row['global_weight'] = global_w[idx]
            elif resolved_weight:
                row['_gw_tmp'] = global_w[idx]

            row.update(edge_attrs_map.get(eid, {}))
            row.update(slice_attrs_map.get(eid, {}))

            if resolved_weight:
                gw_col = 'global_weight' if include_weight else '_gw_tmp'
                row['effective_weight'] = row.get('slice_weight', row.get(gw_col))
                if not include_weight:
                    row.pop('_gw_tmp', None)

            out_rows.append(row)

        out = dataframe_from_rows(out_rows)
        return clone_dataframe(out) if copy else out

    def nodes_view(self, copy=True):
        """Read-only node attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like
            Columns include `node_id` plus pure attributes.
        """
        df = self._node_table
        if df is None or 'node_id' not in dataframe_columns(df):
            out = empty_dataframe({'node_id': 'text'})
        else:
            out = clone_dataframe(df)
        return clone_dataframe(out) if copy else out

    def slices_view(self, copy=True):
        """Read-only slice attribute table.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like
            One row per slice (including the default slice), keyed by
            ``slice_id``. User-set slice attributes appear as additional
            columns; slices without user attrs still appear, with null
            cells.
        """
        all_slice_ids = list(self.slices.list(include_default=True))
        attr_df = self.slice_attributes
        attr_rows: dict = {}
        if attr_df is not None and 'slice_id' in dataframe_columns(attr_df):
            for row in dataframe_to_rows(attr_df):
                sid = row.get('slice_id')
                if sid is not None:
                    attr_rows[sid] = {k: v for k, v in row.items() if k != 'slice_id'}
        rows = [{'slice_id': sid, **attr_rows.get(sid, {})} for sid in all_slice_ids]
        out = dataframe_from_rows(rows) if rows else empty_dataframe({'slice_id': 'text'})
        return clone_dataframe(out) if copy else out

    def aspects_view(self, copy=True):
        """Return a view of Kivela aspects and their metadata.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Columns include `aspect`, `elem_layers`, and any aspect attribute keys.
        """
        if not getattr(self, 'aspects', None):
            return empty_dataframe({'aspect': 'text', 'elem_layers': 'list_text'})
        rows = []
        for a in self.aspects:
            base = {'aspect': a, 'elem_layers': list(self.elem_layers.get(a, []))}
            base.update(self.layers._aspect_attrs.get(a, {}))
            rows.append(base)
        df = dataframe_from_rows(rows)
        return clone_dataframe(df) if copy else df

    def layers_view(self, copy=True):
        """Return a read-only table of multi-aspect layers.

        Parameters
        ----------
        copy : bool, optional
            Return a cloned DataFrame.

        Returns
        -------
        DataFrame-like

        Notes
        -----
        Columns include `layer_tuple`, `layer_id`, aspect columns, layer attributes,
        and prefixed elementary layer attributes.
        """
        if not self.aspects or not getattr(self.layers, '_all_layers', ()):
            return empty_dataframe({'layer_tuple': 'list_text', 'layer_id': 'text'})

        elem_attr_rows = {}
        if self.layer_attributes is not None and 'layer_id' in dataframe_columns(
            self.layer_attributes
        ):
            for row in dataframe_to_rows(self.layer_attributes):
                layer_id = row.get('layer_id')
                if layer_id is not None:
                    elem_attr_rows[str(layer_id)] = {
                        k: v for k, v in row.items() if k != 'layer_id'
                    }

        rows = []
        for aa in self.layers._all_layers:
            aa = tuple(aa)
            base = {'layer_tuple': list(aa), 'layer_id': self.layers.layer_tuple_to_id(aa)}
            for i, aspect in enumerate(self.aspects):
                base[aspect] = aa[i]
            base.update(self.layers._layer_attrs.get(aa, {}))
            for i, aspect in enumerate(self.aspects):
                for k, v in elem_attr_rows.get(f'{aspect}_{aa[i]}', {}).items():
                    base[f'{aspect}__{k}'] = v
            rows.append(base)
        df = dataframe_from_rows(rows)
        return clone_dataframe(df) if copy else df


class ViewsAccessor:
    """Namespace for materialized graph tables (``G.views``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def edges(self, *args, **kwargs):
        """Materialize the edge table view."""
        return ViewsClass.edges_view(self._G, *args, **kwargs)

    def entity_kinds(self) -> dict:
        """Return the kind of every entity, as a mapping from its id.

        An entity is a node, or an edge that is a node in its own right. The
        answer is built on each call, so changing it changes nothing.
        """
        return {
            ref.id: _external_entity_kind(STORED_ENTITY_KIND[ref.kind])
            for ref in _structure.iter_entities(self._G)
        }

    def nodes(self, *args, **kwargs):
        """Materialize the node table view."""
        return ViewsClass.nodes_view(self._G, *args, **kwargs)

    def slices(self, *args, **kwargs):
        """Materialize the slice table view."""
        return ViewsClass.slices_view(self._G, *args, **kwargs)

    def aspects(self, *args, **kwargs):
        """Materialize the aspect table view."""
        return ViewsClass.aspects_view(self._G, *args, **kwargs)

    def layers(self, *args, **kwargs):
        """Materialize the layer table view."""
        return ViewsClass.layers_view(self._G, *args, **kwargs)

    def layers_view(self, copy=True):
        """Materialize the layer table view."""
        return ViewsClass.layers_view(self._G, copy=copy)


# ---------------------------------------------------------------------------
# The node sequence and the edge sequence
# ---------------------------------------------------------------------------


_MISSING = object()


class ElementSequence:
    """One axis of a graph, read and written as a sequence.

    ``G.N`` is the node sequence and ``G.E`` is the edge sequence. Both hold ids
    in the order the graph holds them, and both answer three kinds of key:

    - an integer is a position in this sequence, and gives back the id there
    - a slice is a range of positions, and gives back a subsequence
    - a string is an attribute name, and gives back the column as a vector

    A subsequence is a sequence in its own right, so a filter and a column read
    compose. It holds the ids it selected and reads through the same graph.
    """

    id_key = 'id'
    id_column = 'id'
    intrinsic_names: tuple[str, ...] = ('id',)

    def __init__(self, graph, ids=None):
        self._graph = graph
        self._ids = None if ids is None else tuple(ids)

    # -- the ids ----------------------------------------------------------

    def _all_ids(self) -> tuple:
        raise NotImplementedError

    @property
    def ids(self) -> tuple:
        """The ids of this sequence, in order."""
        return self._all_ids() if self._ids is None else self._ids

    def _subsequence(self, ids):
        return type(self)(self._graph, ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids)

    def __contains__(self, item) -> bool:
        return item in self.ids

    def __repr__(self) -> str:
        return f'<{type(self).__name__} of {len(self)}>'

    # -- the keys ---------------------------------------------------------

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.column(key)
        if isinstance(key, slice):
            return self._subsequence(self.ids[key])
        if isinstance(key, (int, np.integer)):
            return self.ids[key]
        raise TypeError(
            f'a sequence key is an attribute name, a position, or a range of '
            f'positions, not {type(key).__name__}'
        )

    def __setitem__(self, key, values):
        if not isinstance(key, str):
            raise TypeError('only an attribute column can be assigned to a sequence')
        self.set_column(key, values)

    # -- the columns ------------------------------------------------------

    def _attribute_map(self, name: str) -> dict | None:
        """Return the value of one attribute per element, or None when unknown.

        An attribute no element carries is not the same as an attribute every
        element leaves empty. The first is a mistake by the caller and the
        second is an ordinary graph, so this says which of the two it is.
        """
        raise NotImplementedError

    def _intrinsic(self, name: str, ids):
        """Return one structural field of the named elements, or ``_MISSING``."""
        if name in (self.id_key, self.id_column):
            return list(ids)
        return _MISSING

    def _attribute_vector(self, name: str):
        """Return the whole column of this axis as the store holds it, or None."""
        raise NotImplementedError

    def _intrinsic_vector(self, name: str):
        """Return one structural field of the whole axis as a vector, or None.

        ``None`` means this axis has no such column to read off its arrays, and
        the caller falls back to reading the elements. The id of an element is
        always ``None`` here: the id column *is* the ids, and the caller has
        them.
        """
        return None

    def column(self, name: str, default=None):
        """Return one attribute of every element of this sequence, as a vector.

        A read of the whole axis is a slice of the array the store holds, so it
        costs no walk over the elements. A subsequence, and a caller that names
        a value for the elements that carry none, are read element by element.

        **Every path gives back a read-only array**, so that a caller never has
        to ask which one answered. A read of the whole axis borrows the array the
        store holds, and a write into it would reach the graph with no validation
        and no history entry. A caller who means to change values copies.
        """
        if self._ids is None and default is None:
            # Neither branch asks for the ids. Building the id tuple of the whole
            # axis is itself a walk, so a read that never needs them must not
            # trigger one.
            vector = (
                self._intrinsic_vector(name)
                if name in self.intrinsic_names
                else self._attribute_vector(name)
            )
            if vector is not None:
                return read_only(vector)
        ids = self.ids
        found = self._intrinsic(name, ids)
        if found is not _MISSING:
            if isinstance(found, np.ndarray):
                return read_only(found)
            return read_only(np.array(found, dtype=object if not found else None))
        values = self._attribute_map(name)
        if values is None:
            raise KeyError(f'no attribute named {name!r} on this sequence')
        return read_only(np.array([values.get(element, default) for element in ids]))

    def set_column(self, name: str, values) -> None:
        """Set one attribute of every element of this sequence."""
        ids = self.ids
        if isinstance(values, (str, bytes)) or not hasattr(values, '__len__'):
            values = [values] * len(ids)
        values = list(values)
        if len(values) != len(ids):
            raise ValueError(f'a column of {len(ids)} values is needed, {len(values)} were given')
        self._write_column(name, dict(zip(ids, values, strict=True)))

    def _write_column(self, name: str, values: dict) -> None:
        raise NotImplementedError

    # -- the filters ------------------------------------------------------

    def _matches(self, conditions: dict) -> list:
        ids = self.ids
        columns = {name: self.column(name) for name in conditions}
        keep = []
        for position, element in enumerate(ids):
            if all(columns[name][position] == want for name, want in conditions.items()):
                keep.append(element)
        return keep

    def select(self, **conditions):
        """Return the subsequence whose elements match every condition."""
        if not conditions:
            return self._subsequence(self.ids)
        return self._subsequence(self._matches(conditions))

    def find(self, **conditions):
        """Return the one element that matches every condition.

        A filter that matches nothing, and a filter that matches more than one
        element, are both errors. A caller that wants either of those wants
        :meth:`select`.
        """
        if not conditions:
            raise TypeError('find needs at least one condition')
        matched = self._matches(conditions)
        if not matched:
            raise KeyError(f'nothing matches {conditions!r}')
        if len(matched) > 1:
            raise ValueError(f'{len(matched)} elements match {conditions!r}, expected one')
        return matched[0]


class NodeSequence(ElementSequence):
    """The nodes of a graph, in the order the graph holds them."""

    id_key = 'id'
    id_column = 'node_id'
    intrinsic_names = ('id', 'node_id')

    def _all_ids(self) -> tuple:
        return tuple(self._graph.nodes())

    def _attribute_map(self, name: str) -> dict | None:
        return self._graph._attr_store.node_attr_map(name)

    def _attribute_vector(self, name: str):
        return self._graph._attr_store.node_vector(name)

    def _write_column(self, name: str, values: dict) -> None:
        if name in self.intrinsic_names:
            raise KeyError(
                f'{name!r} is the id of a node, not an attribute of one. '
                'Renaming a node is a structural change, not a column write.'
            )
        self._graph.attrs.set_node_attrs_bulk(
            {element: {name: value} for element, value in values.items()}
        )


# The two intrinsic edge fields a caller may write. ``kind`` is not one: it
# follows from how many members an edge holds and on which sides.
_EDGE_STRUCTURAL_WRITES = frozenset({'weight', 'directed'})

# The three that read as a column. All three come off the edge arrays, so all
# three read as one pass over them rather than as one record per edge.
_EDGE_INTRINSIC_COLUMNS = ('directed', 'weight', 'kind')

# The kind of an edge, by the code the store holds for it, in the words the
# public record uses. The store holds the code and this is the vocabulary, which
# is why the table is passed down rather than kept there.
#
# ``hyper`` stands in for the two names a hyperedge takes. Which of them it takes
# depends on whether its members hold roles, so the direction column decides it,
# and the two are substituted after the table is applied.
STORED_EDGE_KIND_NAMES = tuple(
    STORED_EDGE_KIND[_structure._SLOT_EDGE_KIND[code]]
    for code in sorted(_structure._SLOT_EDGE_KIND)
)
_HYPER_KIND_NAMES = ('hyper_undirected', 'hyper_directed')
_HYPER_KIND_LABEL = STORED_EDGE_KIND[_structure.HYPER]


class EdgeSequence(ElementSequence):
    """The edges of a graph, in the order the graph holds them.

    An edge carries three fields that are not attributes: its direction, its
    weight, and its kind. They read like a column, because a filter over them
    is as common as a filter over an attribute.
    """

    id_key = 'id'
    id_column = 'edge_id'
    intrinsic_names = ('id', 'edge_id', 'directed', 'weight', 'kind')

    def _all_ids(self) -> tuple:
        return tuple(self._graph.edges())

    def _intrinsic(self, name: str, ids):
        if name in (self.id_key, self.id_column):
            return list(ids)
        if name in _EDGE_INTRINSIC_COLUMNS:
            return [getattr(self._graph.get_edge(element), name) for element in ids]
        return _MISSING

    def _intrinsic_vector(self, name: str):
        """Return the whole intrinsic column from the edge arrays, or None.

        The three fields are held differently and so they are read differently.
        ``weight`` is the array the store holds, so the answer is a slice of it.
        The other two are **derived from an array rather than held in one**, so
        each is one vectorized pass that the store keeps against its clock:

        - ``directed`` because an edge that declares nothing inherits the default
          of the graph, and a hyperedge takes neither, resolving its direction
          from whether its members hold roles,
        - ``kind`` because it follows from the shape of the edge — the array
          holds a code, the record shows a name, and a hyperedge shows one of two
          names depending on that same direction.

        The slice addresses the *structural* edges, which is what this sequence
        holds. A placeholder edge occupies no column and is not one of them, so a
        store that holds any falls back to the read element by element.
        """
        if name not in _EDGE_INTRINSIC_COLUMNS:
            return None
        store = self._graph._store
        if not store.edge_axis_contiguous:
            return None
        count = store.edge_count
        if name == 'weight':
            return store.edge_weight[:count]
        directed = store.edge_directed_column()[:count]
        if name == 'directed':
            return directed
        kinds = store.edge_kind_column(STORED_EDGE_KIND_NAMES)[:count]
        hyper = kinds == _HYPER_KIND_LABEL
        if not hyper.any():
            return kinds
        named = kinds.astype(object)
        named[hyper] = [_HYPER_KIND_NAMES[bool(value)] for value in directed[hyper]]
        return named

    def _attribute_map(self, name: str) -> dict | None:
        return self._graph._attr_store.edge_attr_map(name)

    def _attribute_vector(self, name: str):
        return self._graph._attr_store.edge_vector(name)

    def _write_column(self, name: str, values: dict) -> None:
        # ``weight`` and ``directed`` read like a column and are not attributes,
        # so a write of either reaches the field of the edge rather than the
        # attribute store, which reserves both names.
        if name in _EDGE_STRUCTURAL_WRITES:
            for element, value in values.items():
                _mutate.set_edge_field(self._graph, element, name, value)
            self._graph._mark_structure_changed()
            return
        if name in self.intrinsic_names:
            raise KeyError(
                f'{name!r} follows from the shape of an edge, so it cannot be written. '
                'Set the members of the edge instead.'
            )
        self._graph.attrs.set_edge_attrs_bulk(
            {element: {name: value} for element, value in values.items()}
        )
