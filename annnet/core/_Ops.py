"""Subgraph extraction, copy, reverse, and incidence materialization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from . import _build, _mutate, _structure
from ._stored_kinds import STORED_EDGE_KIND
from .._support.dataframe_backend import (
    clone_dataframe,
    dataframe_height,
    dataframe_columns,
    dataframe_to_rows,
    dataframe_filter_in,
    dataframe_memory_usage,
)

if TYPE_CHECKING:
    from .graph import AnnNet


def _hyper_def(graph, edge_id):
    """Return the definition of one hyperedge, as the bulk add API states it.

    A directed hyperedge has a head and a tail. An undirected one has members on
    one side alone.
    """
    sides = _structure.edge_sides(graph, edge_id)
    if _structure.edge_ref(graph, edge_id).directed:
        return {'directed': True, 'head': set(sides.source), 'tail': set(sides.target)}
    return {'directed': False, 'members': set(sides.source)}


def _edge_payload(graph, edge_id):
    """Return one edge in the form the bulk add API takes.

    A directed hyperedge states a head and a tail, and an undirected one states
    members. A binary edge states a source and a target, and an open side of one
    is ``None``, which is what the store holds there.
    """
    ref = _structure.edge_ref(graph, edge_id)
    sides = _structure.edge_sides(graph, edge_id)
    if ref.kind == _structure.HYPER:
        if ref.directed:
            return {
                'head': list(sides.source),
                'tail': list(sides.target),
                'edge_id': edge_id,
                'weight': ref.declared_weight,
            }
        return {
            'members': list(sides.source),
            'edge_id': edge_id,
            'weight': ref.declared_weight,
        }
    default_directed = True if graph.directed is None else graph.directed
    declared = ref.declared_directed
    return {
        'source': next(iter(sides.source)) if sides.source else None,
        'target': next(iter(sides.target)) if sides.target else None,
        'edge_id': edge_id,
        'edge_type': STORED_EDGE_KIND[ref.kind],
        'edge_directed': declared if declared is not None else default_directed,
        'weight': ref.declared_weight,
    }


def _payload_endpoints(payload) -> set:
    """Return every identity one edge payload names."""
    if 'source' in payload:
        return {payload['source'], payload['target']}
    if 'members' in payload:
        return set(payload['members'])
    return set(payload['head']) | set(payload['tail'])


def _payload_has_both_sides(payload) -> bool:
    """Return False for a binary edge that leaves one of its sides open.

    Such an edge names one endpoint alone, so a graph built from the payload
    cannot hold it.
    """
    if 'source' not in payload:
        return True
    return payload['source'] is not None and payload['target'] is not None


def _payload_inside(payload, vertex_ids, bare) -> bool:
    """Return True when every identity of one edge payload lies in a vertex set."""
    if not _payload_has_both_sides(payload):
        return False
    return {bare(member) for member in _payload_endpoints(payload)} <= vertex_ids


def _is_hyper(graph, eid):
    return _structure.has_edge(graph, eid) and (
        _structure.edge_ref(graph, eid).kind == _structure.HYPER
    )


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


def _take_attributes(target, source, vertex_ids, edge_ids) -> None:
    """Copy the attributes of the named elements from one graph to another."""
    if vertex_ids:
        rows = Operations._rows_attr_map(source, source.vertex_attributes, 'vertex_id', vertex_ids)
        if rows:
            target.attrs.set_vertex_attrs_bulk(rows)
    if edge_ids:
        rows = Operations._rows_attr_map(source, source.edge_attributes, 'edge_id', edge_ids)
        if rows:
            target.attrs.set_edge_attrs_bulk(rows)


def _take_slices(target, source) -> None:
    """Add the slice memberships of one graph to another, keeping the target's.

    A slice both graphs declare keeps the attributes it has in the target, and
    takes the members it has in the source. A slice only the source declares
    arrives whole.
    """
    for slice_id, record in source._slices.items():
        held = target._slices.get(slice_id)
        if held is None:
            target._slices[slice_id] = {
                'vertices': set(record['vertices']),
                'edges': set(record['edges']),
                'attributes': dict(record['attributes']),
            }
            continue
        held['vertices'].update(record['vertices'])
        held['edges'].update(record['edges'])
        for key, value in record['attributes'].items():
            held['attributes'].setdefault(key, value)


class Operations:
    """Topology materialization and graph-copy operations (mixed into AnnNet)."""

    def _constructor_aspects(self):
        if self._aspects == ('_',):
            return None
        return {aspect: list(self._layers.get(aspect, ())) for aspect in self._aspects}

    def _copy_graph_attributes(self, new) -> None:
        new.graph_attributes = self.graph_attributes.copy()

    def _rows_attr_map(self, df, key_col: str, keys=None) -> dict:
        if df is None or key_col not in dataframe_columns(df) or dataframe_height(df) == 0:
            return {}
        cache = getattr(self, '_row_attr_cache', None)
        if cache is None:
            cache = {}
            self._row_attr_cache = cache
        cache_key = (id(df), key_col)
        mapping = cache.get(cache_key)
        if mapping is None:
            mapping = {}
            for row in dataframe_to_rows(df):
                kval = row.get(key_col)
                if kval is None:
                    continue
                d = dict(row)
                d.pop(key_col, None)
                mapping[kval] = d
            cache[cache_key] = mapping
        if keys is None:
            return mapping
        wanted = set(keys)
        return {k: v for k, v in mapping.items() if k in wanted}

    def _filter_attr_table(self, df, key_col: str, keys):
        if df is None or key_col not in dataframe_columns(df):
            return df
        return dataframe_filter_in(df, key_col, keys)

    def _flat_edge_vertices(self, edge_ids) -> set[str]:
        vertices = set()
        for eid in edge_ids:
            if not _structure.has_edge(self, eid) or not _structure.carries_structure(self, eid):
                continue
            sides = _structure.edge_sides(self, eid)
            if not sides.source:
                continue
            vertices.update(sides.source)
            vertices.update(sides.target)
        return vertices

    def _ordered_flat_vertex_ids(self, vertex_ids) -> list[str]:
        wanted = set(vertex_ids)
        return [key[0] for key in _structure.node_keys(self) if key[0] in wanted]

    def _ordered_edge_ids(self, edge_ids) -> list[str]:
        wanted = set(edge_ids)
        return [edge_id for edge_id in _structure.edge_ids(self) if edge_id in wanted]

    def _ordered_selection_rows(self, vertex_ids, edge_ids) -> list:
        """Return the entities a selection holds, in the row order they had.

        An edge-entity is one identity on both axes, so a selection that keeps
        the edge keeps the entity. Leaving it behind gives the new graph an edge
        that is an edge-entity with nothing to name it, which is the same shape a
        removal used to leave and which data-model rule 6 forbids.
        """
        wanted_vertices = set(vertex_ids)
        wanted_edges = set(edge_ids)
        return [
            ref.key
            for ref in _structure.iter_entities(self)
            if ref.id in (wanted_edges if ref.kind == _structure.EDGE_ENTITY else wanted_vertices)
        ]

    def _build_flat_graph_from_selection(
        self,
        *,
        vertex_ids,
        edge_ids,
        slice_specs,
        active_slice=None,
        edge_weight_overrides=None,
    ) -> AnnNet:
        ordered_vertices = self._ordered_flat_vertex_ids(vertex_ids)
        ordered_edges = self._ordered_edge_ids(edge_ids)
        row_keys = self._ordered_selection_rows(ordered_vertices, ordered_edges)

        new = self.__class__(directed=self.directed)

        weight_overrides = edge_weight_overrides or {}
        _build.install_structure(
            new,
            # The store selects the same elements in the same order, so it numbers
            # its slots as the new graph numbers its rows and columns.
            store=self._store.select(row_keys, ordered_edges, weights=weight_overrides),
        )
        new.vertex_aligned = self.vertex_aligned
        new._next_edge_id = self._next_edge_id

        _build.install_slices(
            new,
            _build.slices_from_specs(slice_specs),
            current=active_slice if active_slice is not None else self._default_slice,
        )

        new.vertex_attributes = self._filter_attr_table(
            self.vertex_attributes, 'vertex_id', ordered_vertices
        )
        new.edge_attributes = self._filter_attr_table(
            self.edge_attributes, 'edge_id', ordered_edges
        )
        new.slice_attributes = self._filter_attr_table(
            self.slice_attributes, 'slice_id', list(new._slices.keys())
        )
        new.edge_slice_attributes = self._filter_attr_table(
            self.edge_slice_attributes, 'edge_id', []
        )
        new.layer_attributes = _share_or_clone_table(self.layer_attributes)
        new.slice_edge_weights = type(self.slice_edge_weights)()
        self._copy_graph_attributes(new)
        new._install_history_hooks()
        return new

    @staticmethod
    def _bare_vid(node):
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
            return node[0]
        return node

    def edge_subgraph(self, edges) -> AnnNet:
        """Create a subgraph containing only a specified subset of edges.

        Parameters
        ----------
        edges : Iterable[str] | Iterable[int]
            Edge identifiers or edge indices to retain.

        Returns
        -------
        AnnNet
            Subgraph containing selected edges and their incident vertices.

        Notes
        -----
        Hyperedges are supported and retain all member vertices.
        """
        if all(isinstance(e, int) for e in edges):
            E = {_structure.edge_at_column(self, e) for e in edges}
        else:
            E = set(edges)

        if self._aspects == ('_',):
            E = {eid for eid in E if _structure.has_edge(self, eid)}
            E = {eid for eid in E if _structure.carries_structure(self, eid)}
            V = self._flat_edge_vertices(E)
            slice_specs = {}
            for lid, meta in self._slices.items():
                slice_specs[lid] = {
                    'vertices': set(meta['vertices']) & V if lid == self._default_slice else set(),
                    'edges': set(meta['edges']) & E,
                    'attributes': dict(meta['attributes']),
                }
            return self._build_flat_graph_from_selection(
                vertex_ids=V, edge_ids=E, slice_specs=slice_specs
            )

        V = set()
        bin_payload, hyper_payload = [], []
        for eid in E:
            if not _structure.has_edge(self, eid) or not _structure.carries_structure(self, eid):
                continue
            payload = _edge_payload(self, eid)
            if not _payload_has_both_sides(payload):
                continue
            V.update(_payload_endpoints(payload))
            if 'source' in payload:
                bin_payload.append(payload)
            else:
                hyper_payload.append(payload)

        G = self.__class__
        new_aspects = self._constructor_aspects()
        if new_aspects is not None:
            g = G(
                directed=self.directed,
                aspects=new_aspects,
            )
            bare_vid_attrs = self._rows_attr_map(
                self.vertex_attributes, 'vertex_id', {self._bare_vid(v) for v in V}
            )
            for node in V:
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                    bare_vid, layer_coord = node
                else:
                    bare_vid, layer_coord = node, None
                g.add_vertices(bare_vid, layer=layer_coord, **bare_vid_attrs.get(bare_vid, {}))
        else:
            g = G(directed=self.directed)
            va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
            v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]
            g._add_vertices_bulk(v_rows, slice=g._default_slice)

        if bin_payload:
            g._add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=g._default_slice)

        for lid, meta in self._slices.items():
            if not g.slices.exists(lid):
                g.slices.add(lid, **meta['attributes'])
            kept_edges = set(meta['edges']) & E
            if kept_edges:
                g.slices.add_edges(lid, kept_edges)

        self._copy_graph_attributes(g)
        return g

    def subgraph(self, vertices) -> AnnNet:
        """Create a vertex-induced subgraph.

        Parameters
        ----------
        vertices : Iterable[str]
            Vertex identifiers to retain.

        Returns
        -------
        AnnNet
            Subgraph containing only the specified vertices and their internal edges.

        Notes
        -----
        For hyperedges, all member vertices must be included to retain the edge.
        """
        V = set(vertices)

        if self._aspects == ('_',):
            E = set()
            for ref in _structure.iter_edges(self):
                sides = _structure.edge_sides(self, ref.id)
                if not sides.source:
                    continue
                if ref.kind == _structure.HYPER:
                    if sides.source <= V and sides.target <= V:
                        E.add(ref.id)
                elif sides.target and sides.source <= V and sides.target <= V:
                    E.add(ref.id)
            slice_specs = {}
            for lid, meta in self._slices.items():
                slice_specs[lid] = {
                    'vertices': set(meta['vertices']) & V if lid == self._default_slice else set(),
                    'edges': set(meta['edges']) & E,
                    'attributes': dict(meta['attributes']),
                }
            return self._build_flat_graph_from_selection(
                vertex_ids=V, edge_ids=E, slice_specs=slice_specs
            )

        bare = self._bare_vid
        bin_payload, hyper_payload = [], []
        for ref in _structure.iter_edges(self):
            payload = _edge_payload(self, ref.id)
            if not _payload_inside(payload, V, bare):
                continue
            if 'source' in payload:
                bin_payload.append(payload)
            else:
                hyper_payload.append(payload)

        va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
        v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]

        G = self.__class__
        new_aspects = self._constructor_aspects()
        if new_aspects is not None:
            g = G(
                directed=self.directed,
                aspects=new_aspects,
            )
            by_id = _structure.entities_by_id(self)
            for vid in V:
                attrs = va_lookup.get(vid, {})
                placed = False
                for ref in by_id.get(vid, ()):
                    g.add_vertices(ref.id, layer=ref.layer, **attrs)
                    placed = True
                if not placed:
                    g.add_vertices(vid, **attrs)
        else:
            g = G(directed=self.directed)
            g._add_vertices_bulk(v_rows, slice=g._default_slice)
        if bin_payload:
            g._add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=g._default_slice)

        for lid, meta in self._slices.items():
            if not g.slices.exists(lid):
                g.slices.add(lid, **meta['attributes'])
            keep = set()
            for eid in meta['edges']:
                if not _structure.has_edge(self, eid) or not _structure.carries_structure(
                    self, eid
                ):
                    continue
                payload = _edge_payload(self, eid)
                if _payload_inside(payload, V, bare):
                    keep.add(eid)
            if keep:
                g.slices.add_edges(lid, keep)

        self._copy_graph_attributes(g)
        return g

    def extract_subgraph(self, vertices=None, edges=None) -> AnnNet:
        """Create a subgraph based on vertex and/or edge filters.

        Parameters
        ----------
        vertices : Iterable[str] | None, optional
            Vertex IDs to include. If None, no vertex filtering is applied.
        edges : Iterable[str] | Iterable[int] | None, optional
            Edge IDs or indices to include. If None, no edge filtering is applied.

        Returns
        -------
        AnnNet
            Filtered subgraph.

        Notes
        -----
        This is a convenience method that delegates to `subgraph()` and
        `edge_subgraph()` internally.
        """
        if vertices is None and edges is None:
            return Operations.copy(self)

        if edges is not None:
            E = (
                {_structure.edge_at_column(self, e) for e in edges}
                if all(isinstance(e, int) for e in edges)
                else set(edges)
            )
        else:
            E = None
        V = set(vertices) if vertices is not None else None

        if self._aspects == ('_',) and V is not None and E is not None:
            kept_edges = set()
            for eid in E:
                if not _structure.has_edge(self, eid) or not _structure.carries_structure(
                    self, eid
                ):
                    continue
                sides = _structure.edge_sides(self, eid)
                if not sides.source:
                    continue
                if _structure.edge_ref(self, eid).kind == _structure.HYPER:
                    if sides.source <= V and sides.target <= V:
                        kept_edges.add(eid)
                elif sides.target and sides.source <= V and sides.target <= V:
                    kept_edges.add(eid)
            slice_specs = {}
            for lid, meta in self._slices.items():
                slice_specs[lid] = {
                    'vertices': set(meta['vertices']) & V if lid == self._default_slice else set(),
                    'edges': set(meta['edges']) & kept_edges,
                    'attributes': dict(meta['attributes']),
                }
            return self._build_flat_graph_from_selection(
                vertex_ids=V, edge_ids=kept_edges, slice_specs=slice_specs
            )

        if V is not None and E is None:
            return Operations.subgraph(self, V)
        if V is None and E is not None:
            return Operations.edge_subgraph(self, E)

        bare = self._bare_vid
        kept_edges = set()
        for eid in E:
            if not _structure.has_edge(self, eid) or not _structure.carries_structure(self, eid):
                continue
            payload = _edge_payload(self, eid)
            if _payload_inside(payload, V, bare):
                kept_edges.add(eid)

        return Operations.subgraph(Operations.edge_subgraph(self, kept_edges), set(V))

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
        return self

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
            vertices=set(self.vertices()) & set(other.vertices()),
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
            vertices=set(self.vertices()) - set(other.vertices()),
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
            If True, use per-slice edge weights when available.

        Returns
        -------
        AnnNet
            Subgraph containing the slice vertices and edges.

        Raises
        ------
        KeyError
            If the slice does not exist.
        """
        if slice_id not in self._slices:
            raise KeyError(f'slice {slice_id} not found')

        slice_meta = self._slices[slice_id]
        V = set(slice_meta['vertices'])
        E = set(slice_meta['edges'])

        if self._aspects == ('_',):
            E = {eid for eid in E if _structure.has_edge(self, eid)}
            E = {eid for eid in E if _structure.carries_structure(self, eid)}
            weight_overrides = {}
            if resolve_slice_weights:
                df = self.edge_slice_attributes
                if df is not None and {'slice_id', 'edge_id', 'weight'}.issubset(
                    dataframe_columns(df)
                ):
                    for row in dataframe_to_rows(dataframe_filter_in(df, 'edge_id', E)):
                        if row.get('slice_id') != slice_id:
                            continue
                        weight = row.get('weight')
                        if weight is not None:
                            weight_overrides[row['edge_id']] = float(weight)
            return self._build_flat_graph_from_selection(
                vertex_ids=V,
                edge_ids=E,
                slice_specs={
                    self._default_slice: {
                        'vertices': set(),
                        'edges': set(),
                        'attributes': dict(self._slices[self._default_slice]['attributes']),
                    },
                    slice_id: {
                        'vertices': V,
                        'edges': E,
                        'attributes': dict(slice_meta['attributes']),
                    },
                },
                active_slice=slice_id,
                edge_weight_overrides=weight_overrides,
            )

        G = self.__class__
        new_aspects = self._constructor_aspects()
        if new_aspects is not None:
            g = G(
                directed=self.directed,
                aspects=new_aspects,
            )
        else:
            g = G(directed=self.directed)
        g.slices.add(slice_id, **slice_meta['attributes'])
        g.slices.active = slice_id

        va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
        if new_aspects is not None:
            by_id = _structure.entities_by_id(self)
            for vid in V:
                attrs = va_lookup.get(vid, {})
                placed = False
                for ref in by_id.get(vid, ()):
                    if ref.kind != _structure.NODE:
                        continue
                    g.add_vertices(ref.id, layer=ref.layer, slice=slice_id, **attrs)
                    placed = True
                if not placed:
                    g.add_vertices(vid, slice=slice_id, **attrs)
        else:
            v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]
            g._add_vertices_bulk(v_rows, slice=slice_id)

        e_attrs = self._rows_attr_map(self.edge_attributes, 'edge_id', E)
        eff_w = {}
        if resolve_slice_weights:
            df = self.edge_slice_attributes
            if df is not None and {'slice_id', 'edge_id', 'weight'}.issubset(dataframe_columns(df)):
                for row in dataframe_to_rows(dataframe_filter_in(df, 'edge_id', E)):
                    if row.get('slice_id') != slice_id:
                        continue
                    weight = row.get('weight')
                    if weight is not None:
                        eff_w[row['edge_id']] = float(weight)

        bin_payload, hyper_payload = [], []
        for eid in E:
            if not _structure.has_edge(self, eid) or not _structure.carries_structure(self, eid):
                continue
            payload = _edge_payload(self, eid)
            base_weight = _structure.edge_ref(self, eid).weight
            payload['weight'] = (
                eff_w.get(eid, base_weight) if resolve_slice_weights else base_weight
            )
            payload['attributes'] = e_attrs.get(eid, {})
            if 'source' in payload:
                bin_payload.append(payload)
            else:
                hyper_payload.append(payload)

        if bin_payload:
            g._add_edges_bulk(bin_payload, slice=slice_id)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=slice_id)

        self._copy_graph_attributes(g)
        return g

    def _row_attrs(self, df, key_col: str, key):
        if df is None or key_col not in dataframe_columns(df) or dataframe_height(df) == 0:
            return {}
        cache = getattr(self, '_row_attr_cache', None)
        if cache is None:
            cache = {}
            self._row_attr_cache = cache
        cache_key = (id(df), key_col)
        mapping = cache.get(cache_key)
        if mapping is None:
            mapping = {}
            for row in dataframe_to_rows(df):
                kval = row.get(key_col)
                if kval is None:
                    continue
                d = dict(row)
                d.pop(key_col, None)
                mapping[kval] = d
            cache[cache_key] = mapping
        return mapping.get(key, {})

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
        G = self.__class__
        new_aspects = self._constructor_aspects()
        new = G(directed=self.directed, aspects=new_aspects)

        _build.install_structure(
            new,
            # A copy of the slot arrays keeps every slot at the address it had,
            # and it costs a memory copy rather than a pass over every edge.
            store=self._store.copy(),
        )
        new.vertex_aligned = self.vertex_aligned
        new._next_edge_id = self._next_edge_id

        _build.install_slices(
            new,
            _build.clone_slices(self._slices, drop_attributes=True),
            default=self._default_slice,
            current=self._current_slice,
        )

        new.slice_edge_weights = {lid: m.copy() for lid, m in self.slice_edge_weights.items()}

        new.vertex_attributes = _share_or_clone_table(self.vertex_attributes)
        new.edge_attributes = _share_or_clone_table(self.edge_attributes)
        new.slice_attributes = _share_or_clone_table(self.slice_attributes)
        new.edge_slice_attributes = _share_or_clone_table(self.edge_slice_attributes)
        new.layer_attributes = _share_or_clone_table(self.layer_attributes)

        new.layers._all_layers = (
            tuple(tuple(x) for x in self.layers._all_layers) if self.layers.aspects else ()
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
        for df in (self.vertex_attributes, self.edge_attributes):
            if df is not None:
                df_bytes += dataframe_memory_usage(df)
        return matrix_bytes + dict_bytes + df_bytes

    def get_vertex_incidence_matrix_as_lists(self, values: bool = False) -> dict:
        """Materialize the vertex–edge incidence structure as Python lists.

        Parameters
        ----------
        values : bool, optional (default=False)
            - If `False`, returns edge indices incident to each vertex.
            - If `True`, returns the **matrix values** (usually weights or 1/0) for
            each incident edge instead of the indices.

        Returns
        -------
        dict[str, list]
            A mapping from `vertex_id` - list of incident edges (indices or values),
            where:
            - Keys are vertex IDs.
            - Values are lists of edge indices (if `values=False`) or numeric values
            from the incidence matrix (if `values=True`).

        Notes
        -----
        - Internally uses the sparse incidence matrix `self._matrix`, which is stored
        as a SciPy CSR (compressed sparse row) matrix or similar.
        - The incidence matrix `M` is defined as:
            - Rows: vertices
            - Columns: edges
            - Entry `M[i, j]` non-zero ⇨ vertex `i` is incident to edge `j`.
        - This is a convenient method when you want a native-Python structure for
        downstream use (e.g., exporting, iterating, or visualization).
        """
        result = {}
        csr = self._get_csr()
        for i in range(self._num_entities):
            entry = _structure.entity_key_of_row(self, i)
            vertex_id = entry[0] if isinstance(entry, tuple) else entry
            start, end = csr.indptr[i], csr.indptr[i + 1]
            result[vertex_id] = (csr.data[start:end] if values else csr.indices[start:end]).tolist()
        return result

    def vertex_incidence_matrix(self, values: bool = False, sparse: bool = False):
        """Return the vertex–edge incidence matrix in sparse or dense form.

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
            The vertex–edge incidence matrix `M`:
            - Rows correspond to vertices.
            - Columns correspond to edges.
            - `M[i, j]` ≠ 0 indicates that vertex `i` is incident to edge `j`.

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
    'incidence': 'vertex_incidence_matrix',
    'vertex_incidence_matrix': 'vertex_incidence_matrix',
    'incidence_as_lists': 'get_vertex_incidence_matrix_as_lists',
    'get_vertex_incidence_matrix_as_lists': 'get_vertex_incidence_matrix_as_lists',
}


class OperationsAccessor:
    """Namespace for structural graph operations (``G.ops``)."""

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def __hash__(self) -> int:
        """Structural hash over vertices, edge endpoints/direction, and graph attrs."""
        G = self._G
        vertex_ids = tuple(sorted(G.vertices()))
        edge_defs = []
        for j in range(G.ne):
            S, T = G.get_edge(j)
            eid = _structure.edge_at_column(G, j)
            edge_defs.append((eid, tuple(sorted(S)), tuple(sorted(T)), G._is_directed_edge(eid)))
        edge_defs = tuple(sorted(edge_defs))
        graph_meta = (
            tuple(sorted(G.graph_attributes.items())) if hasattr(G, 'graph_attributes') else ()
        )
        return hash((vertex_ids, edge_defs, graph_meta))


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
