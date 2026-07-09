"""Subgraph extraction, copy, reverse, and incidence materialization."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from . import _build, _mutate
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


def _hyper_def(rec):
    if rec.tgt is not None:
        return {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
    return {'directed': False, 'members': set(rec.src)}


def _is_hyper(graph, eid):
    rec = graph._edges.get(eid)
    return rec is not None and rec.etype == 'hyper'


def _share_or_clone_table(df):
    return None if df is None else clone_dataframe(df)


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
            rec = self._edges.get(eid)
            if rec is None or rec.col_idx < 0 or rec.src is None:
                continue
            if rec.etype == 'hyper':
                vertices.update(rec.src)
                if rec.tgt is not None:
                    vertices.update(rec.tgt)
            else:
                vertices.add(rec.src)
                vertices.add(rec.tgt)
        return vertices

    def _ordered_flat_vertex_ids(self, vertex_ids) -> list[str]:
        wanted = set(vertex_ids)
        ordered = []
        for row_idx in range(len(self._row_to_entity)):
            ekey = self._row_to_entity.get(row_idx)
            if ekey is None:
                continue
            rec = self._entities.get(ekey)
            if rec is None or rec.kind != 'vertex':
                continue
            if ekey[0] in wanted:
                ordered.append(ekey[0])
        return ordered

    def _ordered_edge_ids(self, edge_ids) -> list[str]:
        wanted = set(edge_ids)
        return [
            self._col_to_edge[col]
            for col in range(len(self._col_to_edge))
            if self._col_to_edge[col] in wanted
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
        row_keys = [(vid, ('_',)) for vid in ordered_vertices]
        row_indexes = [self._entities[ekey].row_idx for ekey in row_keys]
        col_indexes = [self._edges[eid].col_idx for eid in ordered_edges]

        new = self.__class__(directed=self.directed)
        matrix = self._get_csr()[row_indexes, :][:, col_indexes].todok()

        entities = {ekey: _build.new_entity_record(i, 'vertex') for i, ekey in enumerate(row_keys)}
        weight_overrides = edge_weight_overrides or {}
        edges = {
            eid: _build.clone_edge_record(
                self._edges[eid], col_idx=new_col, weight=weight_overrides.get(eid)
            )
            for new_col, eid in enumerate(ordered_edges)
        }
        _build.install_structure(new, entities=entities, edges=edges, matrix=matrix)
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
            E = {self._col_to_edge[e] for e in edges}
        else:
            E = set(edges)

        if self._aspects == ('_',):
            E = {eid for eid in E if eid in self._edges and self._edges[eid].col_idx >= 0}
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

        default_dir = True if self.directed is None else self.directed
        V = set()
        bin_payload, hyper_payload = [], []
        for eid in E:
            rec = self._edges.get(eid)
            if rec is None or rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                h = _hyper_def(rec)
                if h.get('members'):
                    V.update(h['members'])
                    hyper_payload.append(
                        {'members': list(h['members']), 'edge_id': eid, 'weight': rec.weight}
                    )
                else:
                    V.update(h.get('head', ()))
                    V.update(h.get('tail', ()))
                    hyper_payload.append(
                        {
                            'head': list(h.get('head', ())),
                            'tail': list(h.get('tail', ())),
                            'edge_id': eid,
                            'weight': rec.weight,
                        }
                    )
            else:
                s, t = rec.src, rec.tgt
                if s is None or t is None:
                    continue
                V.add(s)
                V.add(t)
                bin_payload.append(
                    {
                        'source': s,
                        'target': t,
                        'edge_id': eid,
                        'edge_type': rec.etype,
                        'edge_directed': rec.directed if rec.directed is not None else default_dir,
                        'weight': rec.weight,
                    }
                )

        G = self.__class__
        new_aspects = self._constructor_aspects()
        if new_aspects is not None:
            g = G(directed=self.directed, v=len(V), e=len(E), aspects=new_aspects)
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
            g = G(directed=self.directed, v=len(V), e=len(E))
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
            for eid, rec in self._edges.items():
                if rec.col_idx < 0 or rec.src is None:
                    continue
                if rec.etype == 'hyper':
                    if set(rec.src).issubset(V) and (rec.tgt is None or set(rec.tgt).issubset(V)):
                        E.add(eid)
                elif rec.src in V and rec.tgt in V:
                    E.add(eid)
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
        E_bin, E_hyper_members, E_hyper_dir = [], [], []
        for eid, rec in self._edges.items():
            if rec.col_idx < 0 or rec.src is None:
                continue
            if rec.etype == 'hyper':
                h = _hyper_def(rec)
                if h.get('members'):
                    if {bare(m) for m in h['members']}.issubset(V):
                        E_hyper_members.append(eid)
                elif {bare(m) for m in h.get('head', ())}.issubset(V) and {
                    bare(m) for m in h.get('tail', ())
                }.issubset(V):
                    E_hyper_dir.append(eid)
            else:
                s, t = rec.src, rec.tgt
                if s is not None and t is not None and bare(s) in V and bare(t) in V:
                    E_bin.append(eid)

        va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
        v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]
        default_dir = True if self.directed is None else self.directed

        bin_payload = []
        for eid in E_bin:
            rec = self._edges[eid]
            bin_payload.append(
                {
                    'source': rec.src,
                    'target': rec.tgt,
                    'edge_id': eid,
                    'edge_type': rec.etype,
                    'edge_directed': rec.directed if rec.directed is not None else default_dir,
                    'weight': rec.weight,
                }
            )
        hyper_payload = []
        for eid in E_hyper_members:
            rec = self._edges[eid]
            h = _hyper_def(rec)
            hyper_payload.append(
                {'members': list(h['members']), 'edge_id': eid, 'weight': rec.weight}
            )
        for eid in E_hyper_dir:
            rec = self._edges[eid]
            h = _hyper_def(rec)
            hyper_payload.append(
                {
                    'head': list(h.get('head', ())),
                    'tail': list(h.get('tail', ())),
                    'edge_id': eid,
                    'weight': rec.weight,
                }
            )

        G = self.__class__
        edge_count = len(E_bin) + len(E_hyper_members) + len(E_hyper_dir)
        new_aspects = self._constructor_aspects()
        if new_aspects is not None:
            g = G(directed=self.directed, v=len(V), e=edge_count, aspects=new_aspects)
            for vid in V:
                attrs = va_lookup.get(vid, {})
                placed = False
                for ekey in self._vid_to_ekeys.get(vid, []):
                    g.add_vertices(ekey[0], layer=ekey[1], **attrs)
                    placed = True
                if not placed:
                    g.add_vertices(vid, **attrs)
        else:
            g = G(directed=self.directed, v=len(V), e=edge_count)
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
                rec = self._edges.get(eid)
                if rec is None or rec.col_idx < 0:
                    continue
                if rec.etype == 'hyper':
                    h = _hyper_def(rec)
                    if h.get('members'):
                        if {bare(m) for m in h['members']}.issubset(V):
                            keep.add(eid)
                    elif {bare(m) for m in h.get('head', ())}.issubset(V) and {
                        bare(m) for m in h.get('tail', ())
                    }.issubset(V):
                        keep.add(eid)
                else:
                    s, t = rec.src, rec.tgt
                    if s is not None and t is not None and bare(s) in V and bare(t) in V:
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
                {self._col_to_edge[e] for e in edges}
                if all(isinstance(e, int) for e in edges)
                else set(edges)
            )
        else:
            E = None
        V = set(vertices) if vertices is not None else None

        if self._aspects == ('_',) and V is not None and E is not None:
            kept_edges = set()
            for eid in E:
                rec = self._edges.get(eid)
                if rec is None or rec.col_idx < 0 or rec.src is None:
                    continue
                if rec.etype == 'hyper':
                    if set(rec.src).issubset(V) and (rec.tgt is None or set(rec.tgt).issubset(V)):
                        kept_edges.add(eid)
                elif rec.src in V and rec.tgt in V:
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
            rec = self._edges.get(eid)
            if rec is None or rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                h = _hyper_def(rec)
                if h.get('members'):
                    if {bare(m) for m in h['members']}.issubset(V):
                        kept_edges.add(eid)
                elif {bare(m) for m in h.get('head', ())}.issubset(V) and {
                    bare(m) for m in h.get('tail', ())
                }.issubset(V):
                    kept_edges.add(eid)
            else:
                s, t = rec.src, rec.tgt
                if s is not None and t is not None and bare(s) in V and bare(t) in V:
                    kept_edges.add(eid)

        return Operations.subgraph(Operations.edge_subgraph(self, kept_edges), set(V))

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
            E = {eid for eid in E if eid in self._edges and self._edges[eid].col_idx >= 0}
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
            g = G(directed=self.directed, v=len(V), e=len(E), aspects=new_aspects)
        else:
            g = G(directed=self.directed, v=len(V), e=len(E))
        g.slices.add(slice_id, **slice_meta['attributes'])
        g.slices.active = slice_id

        va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
        if new_aspects is not None:
            for vid in V:
                attrs = va_lookup.get(vid, {})
                placed = False
                for ekey in self._vid_to_ekeys.get(vid, []):
                    rec = self._entities.get(ekey)
                    if rec is None or rec.kind != 'vertex':
                        continue
                    g.add_vertices(ekey[0], layer=ekey[1], slice=slice_id, **attrs)
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
            rec = self._edges.get(eid)
            if rec is None or rec.col_idx < 0:
                continue
            base_weight = rec.weight if rec.weight is not None else 1.0
            w = eff_w.get(eid, base_weight) if resolve_slice_weights else base_weight
            attrs = e_attrs.get(eid, {})
            if rec.etype == 'hyper':
                if rec.tgt is None:
                    hyper_payload.append(
                        {'members': list(rec.src), 'edge_id': eid, 'weight': w, 'attributes': attrs}
                    )
                else:
                    hyper_payload.append(
                        {
                            'head': list(rec.src),
                            'tail': list(rec.tgt),
                            'edge_id': eid,
                            'weight': w,
                            'attributes': attrs,
                        }
                    )
            else:
                bin_payload.append(
                    {
                        'source': rec.src,
                        'target': rec.tgt,
                        'edge_id': eid,
                        'edge_type': rec.etype,
                        'edge_directed': rec.directed
                        if rec.directed is not None
                        else (True if self.directed is None else self.directed),
                        'weight': w,
                        'attributes': attrs,
                    }
                )

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
        if new_aspects is not None:
            new = G(
                directed=self.directed,
                v=self._matrix.shape[0],
                e=self._matrix.shape[1],
                aspects=new_aspects,
            )
        else:
            new = G(directed=self.directed, v=self._matrix.shape[0], e=self._matrix.shape[1])

        _build.install_structure(
            new,
            entities=_build.clone_entities(self._entities),
            edges={eid: _build.clone_edge_record(rec) for eid, rec in self._edges.items()},
            matrix=self._matrix.copy(),
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
            len(self._entities)
            + len(self._col_to_edge)
            + sum(1 for r in self._edges.values() if r.weight is not None)
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
        row_to_entity = self._row_to_entity
        for i in range(self._num_entities):
            entry = row_to_entity[i]
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
            eid = G._col_to_edge[j]
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
