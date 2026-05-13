from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ._records import EdgeRecord, SliceRecord, EntityRecord
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
    """Build hyperedge metadata dict from an EdgeRecord (no compat proxy)."""
    if rec.tgt is not None:
        return {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
    return {'directed': False, 'members': set(rec.src)}


def _is_hyper(graph, eid):
    rec = graph._edges.get(eid)
    return rec is not None and rec.etype == 'hyper'


def _clone_edge_record(rec, *, col_idx=None, weight=None):
    """Clone an EdgeRecord without generic copy machinery."""
    return EdgeRecord(
        src=rec.src,
        tgt=rec.tgt,
        weight=rec.weight if weight is None else weight,
        directed=rec.directed,
        etype=rec.etype,
        col_idx=rec.col_idx if col_idx is None else col_idx,
        ml_kind=rec.ml_kind,
        ml_layers=rec.ml_layers,
        direction_policy=rec.direction_policy,
    )


def _share_or_clone_table(df):
    """Reuse immutable columnar tables when safe, else copy."""
    if df is None:
        return None
    return clone_dataframe(df)


class Operations:
    """Topology materialization and graph-copy operations."""

    def _rows_attr_map(self, df, key_col: str, keys=None) -> dict:
        """Return a key -> attrs mapping from an attribute table."""
        if df is None or key_col not in dataframe_columns(df):
            return {}
        if dataframe_height(df) == 0:
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
        """Return a filtered attribute table preserving the input backend."""
        if df is None or key_col not in dataframe_columns(df):
            return df

        return dataframe_filter_in(df, key_col, keys)

    def _flat_edge_vertices(self, edge_ids) -> set[str]:
        """Return flat vertex ids incident to the selected edges."""
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
        """Return selected flat vertex ids in row order."""
        wanted = set(vertex_ids)
        ordered = []
        for row_idx in range(len(self._row_to_entity)):
            ekey = self._row_to_entity.get(row_idx)
            if ekey is None:
                continue
            rec = self._entities.get(ekey)
            if rec is None or rec.kind != 'vertex':
                continue
            vid = ekey[0]
            if vid in wanted:
                ordered.append(vid)
        return ordered

    def _ordered_edge_ids(self, edge_ids) -> list[str]:
        """Return selected edge ids in column order."""
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
        """Materialize a flat subgraph by slicing the internal IR directly."""
        ordered_vertices = self._ordered_flat_vertex_ids(vertex_ids)
        ordered_edges = self._ordered_edge_ids(edge_ids)
        row_keys = [(vid, ('_',)) for vid in ordered_vertices]
        row_indexes = [self._entities[ekey].row_idx for ekey in row_keys]
        col_indexes = [self._edges[eid].col_idx for eid in ordered_edges]

        G = self.__class__
        new = G(directed=self.directed)

        new._matrix = self._get_csr()[row_indexes, :][:, col_indexes].todok()

        new._entities = {}
        new._row_to_entity = {}
        new._vid_to_ekeys = {}
        for i, ekey in enumerate(row_keys):
            new._entities[ekey] = EntityRecord(row_idx=i, kind='vertex')
            new._row_to_entity[i] = ekey
            new._vid_to_ekeys.setdefault(ekey[0], []).append(ekey)
        new.vertex_aligned = self.vertex_aligned

        weight_overrides = edge_weight_overrides or {}
        new._edges = {}
        new._col_to_edge = {}
        new._src_to_edges = {}
        new._tgt_to_edges = {}
        new._pair_to_edges = {}
        for new_col, eid in enumerate(ordered_edges):
            rec = self._edges[eid]
            new_rec = _clone_edge_record(
                rec,
                col_idx=new_col,
                weight=weight_overrides.get(eid),
            )
            new._edges[eid] = new_rec
            new._col_to_edge[new_col] = eid
            if new_rec.etype != 'hyper' and new_rec.src is not None and new_rec.tgt is not None:
                new._src_to_edges.setdefault(new_rec.src, []).append(eid)
                new._tgt_to_edges.setdefault(new_rec.tgt, []).append(eid)
        new._next_edge_id = self._next_edge_id
        new._edge_indexes_built = True

        new._slices = {}
        for slice_id, spec in slice_specs.items():
            new._slices[slice_id] = SliceRecord(
                vertices=set(spec.get('vertices', ())),
                edges=set(spec.get('edges', ())),
                attributes=dict(spec.get('attributes', {})),
            )
        if new._default_slice not in new._slices:
            new._slices[new._default_slice] = SliceRecord()
        new._current_slice = active_slice if active_slice is not None else self._default_slice

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
        new.layer_attributes = self.layer_attributes.clone()
        new.slice_edge_weights = type(self.slice_edge_weights)()
        new.graph_attributes = {}
        new._install_history_hooks()
        return new

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
        # normalize to edge_id set
        if all(isinstance(e, int) for e in edges):
            E = {self._col_to_edge[e] for e in edges}
        else:
            E = set(edges)

        if self._aspects == ('_',):
            if all(isinstance(e, int) for e in edges):
                E = {self._col_to_edge[e] for e in edges}
            else:
                E = set(edges)
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
                vertex_ids=V,
                edge_ids=E,
                slice_specs=slice_specs,
            )

        default_dir = True if self.directed is None else self.directed

        # collect incident vertices and partition edges
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
                        {
                            'members': list(h['members']),
                            'edge_id': eid,
                            'weight': rec.weight,
                        }
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
                s, t, etype = rec.src, rec.tgt, rec.etype
                if s is None or t is None:
                    continue
                V.add(s)
                V.add(t)
                bin_payload.append(
                    {
                        'source': s,
                        'target': t,
                        'edge_id': eid,
                        'edge_type': etype,
                        'edge_directed': rec.directed if rec.directed is not None else default_dir,
                        'weight': rec.weight,
                    }
                )

        # new graph prealloc — preserve aspects when self is multilayer so
        # supra-node tuple endpoints remain valid in the new graph.
        G = self.__class__
        if self._aspects != ('_',):
            new_aspects = {a: list(self._layers.get(a, [])) for a in self._aspects}
            g = G(directed=self.directed, n=len(V), e=len(E), aspects=new_aspects)
            # Place vertices in their original supra-node coordinates.
            bare_vid_attrs = self._rows_attr_map(
                self.vertex_attributes, 'vertex_id', {self._bare_vid(v) for v in V}
            )
            for node in V:
                if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                    bare_vid, layer_coord = node
                else:
                    bare_vid, layer_coord = node, None
                attrs = bare_vid_attrs.get(bare_vid, {})
                g.add_vertices(bare_vid, layer=layer_coord, **attrs)
        else:
            g = G(directed=self.directed, n=len(V), e=len(E))
            va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
            v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]
            g.add_vertices_bulk(v_rows, slice=g._default_slice)

        # edges
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=g._default_slice)

        # copy slice memberships for retained edges & incident vertices
        for lid, meta in self._slices.items():
            if not g.slices.exists(lid):
                g.slices.add(lid, **meta['attributes'])
            kept_edges = set(meta['edges']) & E
            if kept_edges:
                g.slices.add_edges(lid, kept_edges)

        return g

    @staticmethod
    def _bare_vid(node):
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
            return node[0]
        return node

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
                else:
                    if rec.src in V and rec.tgt in V:
                        E.add(eid)
            slice_specs = {}
            for lid, meta in self._slices.items():
                kept_edges = set(meta['edges']) & E
                slice_specs[lid] = {
                    'vertices': set(meta['vertices']) & V if lid == self._default_slice else set(),
                    'edges': kept_edges,
                    'attributes': dict(meta['attributes']),
                }
            return self._build_flat_graph_from_selection(
                vertex_ids=V,
                edge_ids=E,
                slice_specs=slice_specs,
            )

        # collect edges fully inside V (V can be a set of bare vertex IDs even
        # when self is multilayer — endpoints are normalized via _bare_vid).
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
                else:
                    if {bare(m) for m in h.get('head', ())}.issubset(V) and {
                        bare(m) for m in h.get('tail', ())
                    }.issubset(V):
                        E_hyper_dir.append(eid)
            else:
                s, t = rec.src, rec.tgt
                if s is not None and t is not None and bare(s) in V and bare(t) in V:
                    E_bin.append(eid)

        # payloads
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

        # build new graph — preserve aspects when self is multilayer so the
        # supra-node-tuple endpoints in payloads stay valid.
        G = self.__class__
        edge_count = len(E_bin) + len(E_hyper_members) + len(E_hyper_dir)
        if self._aspects != ('_',):
            new_aspects = {a: list(self._layers.get(a, [])) for a in self._aspects}
            g = G(directed=self.directed, n=len(V), e=edge_count, aspects=new_aspects)
            # Place each retained vertex back at its original supra-node coords.
            for vid in V:
                attrs = va_lookup.get(vid, {})
                placed = False
                for ekey in self._vid_to_ekeys.get(vid, []):
                    g.add_vertices(ekey[0], layer=ekey[1], **attrs)
                    placed = True
                if not placed:
                    g.add_vertices(vid, **attrs)
        else:
            g = G(directed=self.directed, n=len(V), e=edge_count)
            g.add_vertices_bulk(v_rows, slice=g._default_slice)
        if bin_payload:
            g.add_edges_bulk(bin_payload, slice=g._default_slice)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=g._default_slice)

        # slice memberships restricted to V
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
                    else:
                        if {bare(m) for m in h.get('head', ())}.issubset(V) and {
                            bare(m) for m in h.get('tail', ())
                        }.issubset(V):
                            keep.add(eid)
                else:
                    s, t = rec.src, rec.tgt
                    if s is not None and t is not None and bare(s) in V and bare(t) in V:
                        keep.add(eid)
            if keep:
                g.slices.add_edges(lid, keep)

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
            if all(isinstance(e, int) for e in edges):
                E = {self._col_to_edge[e] for e in edges}
            else:
                E = set(edges)
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
                else:
                    if rec.src in V and rec.tgt in V:
                        kept_edges.add(eid)
            slice_specs = {}
            for lid, meta in self._slices.items():
                slice_specs[lid] = {
                    'vertices': set(meta['vertices']) & V if lid == self._default_slice else set(),
                    'edges': set(meta['edges']) & kept_edges,
                    'attributes': dict(meta['attributes']),
                }
            return self._build_flat_graph_from_selection(
                vertex_ids=V,
                edge_ids=kept_edges,
                slice_specs=slice_specs,
            )

        # If only one filter, delegate to optimized path
        if V is not None and E is None:
            return Operations.subgraph(self, V)
        if V is None and E is not None:
            return Operations.edge_subgraph(self, E)

        # Both filters: keep only edges in E whose endpoints (or members) lie in V
        def _bare_vid(node):
            # Multilayer endpoints are (vertex_id, layer_coord_tuple); bare vids
            # are plain strings. Normalize to the underlying vertex ID so V can
            # be expressed as a set of bare vertex IDs regardless of multilayer.
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                return node[0]
            return node

        kept_edges = set()
        kept_vertices = set(V)
        for eid in E:
            rec = self._edges.get(eid)
            if rec is None or rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                h = _hyper_def(rec)
                if h.get('members'):
                    if {_bare_vid(m) for m in h['members']}.issubset(V):
                        kept_edges.add(eid)
                else:
                    if {_bare_vid(m) for m in h.get('head', ())}.issubset(V) and {
                        _bare_vid(m) for m in h.get('tail', ())
                    }.issubset(V):
                        kept_edges.add(eid)
            else:
                s, t = rec.src, rec.tgt
                if s is not None and t is not None and _bare_vid(s) in V and _bare_vid(t) in V:
                    kept_edges.add(eid)

        return Operations.subgraph(Operations.edge_subgraph(self, kept_edges), kept_vertices)

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

        for rec in g._edges.values():
            if rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                if rec.tgt is not None:
                    rec.src, rec.tgt = rec.tgt, rec.src
                continue
            edge_is_directed = (
                rec.directed
                if rec.directed is not None
                else (True if g.directed is None else g.directed)
            )
            if edge_is_directed:
                rec.src, rec.tgt = rec.tgt, rec.src

        g._rebuild_edge_indexes()

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
        g = G(directed=self.directed, n=len(V), e=len(E))
        g.slices.add(slice_id, **slice_meta['attributes'])
        g.slices.active = slice_id

        # vertices with attrs (edge-entities share same table)
        va_lookup = self._rows_attr_map(self.vertex_attributes, 'vertex_id', V)
        v_rows = [{'vertex_id': v, **va_lookup.get(v, {})} for v in V]
        g.add_vertices_bulk(v_rows, slice=slice_id)

        # edge attrs
        e_attrs = self._rows_attr_map(self.edge_attributes, 'edge_id', E)

        # weights
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

        # partition edges
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
                        {
                            'members': list(rec.src),
                            'edge_id': eid,
                            'weight': w,
                            'attributes': attrs,
                        }
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
            g.add_edges_bulk(bin_payload, slice=slice_id)
        if hyper_payload:
            g.add_edges(hyper_payload, slice=slice_id)

        return g

    def _row_attrs(self, df, key_col: str, key):
        """INTERNAL: return a dict of attributes for the row in `df` where `key_col == key`,

        excluding the key column itself. If not found or df empty, return {}.
        Caches per (id(df), key_col) for speed; cache auto-refreshes when the df object changes.
        """

        # Basic guards
        if df is None or key_col not in dataframe_columns(df):
            return {}
        if dataframe_height(df) == 0:
            return {}

        # Cache setup
        cache = getattr(self, '_row_attr_cache', None)
        if cache is None:
            cache = {}
            self._row_attr_cache = cache

        cache_key = (id(df), key_col)
        mapping = cache.get(cache_key)

        # Build the mapping once per df object
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

        # ---------------------------------------------------------------
        # 1) Construct empty graph with same capacity (fast path)
        # ---------------------------------------------------------------
        G = self.__class__
        new = G(directed=self.directed)

        # ---------------------------------------------------------------
        # 2) Clone incidence matrix (DOK → DOK copy is fast)
        # ---------------------------------------------------------------
        new._matrix = self._matrix.copy()

        # ---------------------------------------------------------------
        # 3) Clone entity/index mappings
        # ---------------------------------------------------------------
        new._entities = {}
        new._row_to_entity = {}
        new._vid_to_ekeys = {}
        for k, v in self._entities.items():
            rec = EntityRecord(row_idx=v.row_idx, kind=v.kind)
            new._entities[k] = rec
            new._row_to_entity[rec.row_idx] = k
            if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], str):
                new._vid_to_ekeys.setdefault(k[0], []).append(k)

        new.vertex_aligned = self.vertex_aligned

        # ---------------------------------------------------------------
        # 4) Clone edge/index mappings
        # ---------------------------------------------------------------
        new._edges = {}
        new._col_to_edge = dict(self._col_to_edge)
        new._src_to_edges = {}
        new._tgt_to_edges = {}
        new._pair_to_edges = {}
        for eid, rec in self._edges.items():
            new_rec = _clone_edge_record(rec)
            new._edges[eid] = new_rec
            if new_rec.etype != 'hyper' and new_rec.src is not None and new_rec.tgt is not None:
                new._src_to_edges.setdefault(new_rec.src, []).append(eid)
                new._tgt_to_edges.setdefault(new_rec.tgt, []).append(eid)
        new._next_edge_id = self._next_edge_id
        new._edge_indexes_built = True

        # ---------------------------------------------------------------
        # 5) Clone slice structure (vertices, edges, attributes)
        # ---------------------------------------------------------------
        new._slices = {}
        for lid, meta in self._slices.items():
            new._slices[lid] = SliceRecord(
                vertices=meta['vertices'].copy(),
                edges=meta['edges'].copy(),
                attributes={},
            )

        new._default_slice = self._default_slice
        new._current_slice = self._current_slice

        # ---------------------------------------------------------------
        # 6) Clone slice_edge_weights
        # ---------------------------------------------------------------
        new.slice_edge_weights = {lid: m.copy() for lid, m in self.slice_edge_weights.items()}

        # ---------------------------------------------------------------
        # 7) Clone attribute tables (Polars DF → clone is fast / zero-copy)
        # ---------------------------------------------------------------
        new.vertex_attributes = _share_or_clone_table(self.vertex_attributes)
        new.edge_attributes = _share_or_clone_table(self.edge_attributes)
        new.slice_attributes = _share_or_clone_table(self.slice_attributes)
        new.edge_slice_attributes = _share_or_clone_table(self.edge_slice_attributes)
        new.layer_attributes = _share_or_clone_table(self.layer_attributes)

        # ---------------------------------------------------------------
        # 8) Clone Kivela metadata
        # ---------------------------------------------------------------
        if self.layers.aspects:
            new.layers.set_aspects(
                list(self.layers.aspects),
                {k: list(v) for k, v in self.layers.elem_layers.items()},
            )
            new.layers._all_layers = tuple(tuple(x) for x in self.layers._all_layers)
        else:
            new.layers._all_layers = ()
        new.layers._aspect_attrs = {a: m.copy() for a, m in self.layers._aspect_attrs.items()}
        new.layers._layer_attrs = {aa: m.copy() for aa, m in self.layers._layer_attrs.items()}
        new.layers._state_attrs = {k: m.copy() for k, m in self.layers._state_attrs.items()}

        # ---------------------------------------------------------------
        # 10) Copy global graph attributes
        # ---------------------------------------------------------------
        new.graph_attributes = self.graph_attributes.copy()

        # ---------------------------------------------------------------
        # 11) History / versioning
        # ---------------------------------------------------------------
        if history:
            new._history_enabled = self._history_enabled
            new._history = [h.copy() for h in self._history]
            new._version = self._version
            new._snapshots = list(self._snapshots)
            new._history_clock0 = time.perf_counter_ns()
        else:
            new._history_enabled = self._history_enabled
            new._history = []
            new._version = 0
            new._snapshots = []
            new._history_clock0 = time.perf_counter_ns()

        # ---------------------------------------------------------------
        # 12) Reinstall hooks (fresh)
        # ---------------------------------------------------------------
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

        va = self.vertex_attributes
        if va is not None:
            df_bytes += dataframe_memory_usage(va)

        ea = self.edge_attributes
        if ea is not None:
            df_bytes += dataframe_memory_usage(ea)
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
            start = csr.indptr[i]
            end = csr.indptr[i + 1]
            if values:
                result[vertex_id] = csr.data[start:end].tolist()
            else:
                result[vertex_id] = csr.indices[start:end].tolist()
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
        else:
            rows, cols = M.shape
            estimated_gb = rows * cols * 4 / 1024**3
            if estimated_gb > 2.0:
                raise MemoryError(
                    f'Dense conversion would require ~{estimated_gb:.1f} GB '
                    f'({rows:,} × {cols:,} float32). '
                    'Use sparse=True instead, or call .toarray() explicitly if you are certain.'
                )
            return M.toarray()


class OperationsAccessor:
    """Namespace for structural graph operations.

    Returned by ``G.ops``. Flat methods remain during the migration window.
    """

    __slots__ = ('_G',)

    def __init__(self, graph):
        self._G = graph

    def subgraph(self, *args, **kwargs):
        return Operations.subgraph(self._G, *args, **kwargs)

    def edge_subgraph(self, *args, **kwargs):
        return Operations.edge_subgraph(self._G, *args, **kwargs)

    def extract(self, *args, **kwargs):
        return Operations.extract_subgraph(self._G, *args, **kwargs)

    def extract_subgraph(self, *args, **kwargs):
        return Operations.extract_subgraph(self._G, *args, **kwargs)

    def copy(self, *args, **kwargs):
        return Operations.copy(self._G, *args, **kwargs)

    def reverse(self, *args, **kwargs):
        return Operations.reverse(self._G, *args, **kwargs)

    def memory_usage(self, *args, **kwargs):
        return Operations.memory_usage(self._G, *args, **kwargs)

    def incidence(self, *args, **kwargs):
        return Operations.vertex_incidence_matrix(self._G, *args, **kwargs)

    def vertex_incidence_matrix(self, *args, **kwargs):
        return Operations.vertex_incidence_matrix(self._G, *args, **kwargs)

    def incidence_as_lists(self, *args, **kwargs):
        return Operations.get_vertex_incidence_matrix_as_lists(self._G, *args, **kwargs)

    def get_vertex_incidence_matrix_as_lists(self, *args, **kwargs):
        return Operations.get_vertex_incidence_matrix_as_lists(self._G, *args, **kwargs)

    def __hash__(self) -> int:
        """Return a stable hash representing the current graph structure and metadata.

        Returns
        -------
        int
            A hash value that uniquely (within high probability) identifies the graph
            based on its topology and attributes.

        Behavior
        --------

        - Includes the set of vertices, edges, and directedness in the hash.
        - Includes graph-level attributes (if any) to capture metadata changes.
        - Does **not** depend on memory addresses or internal object IDs, so the same
        graph serialized/deserialized or reconstructed with identical structure
        will produce the same hash.

        Notes
        -----
        - This method enables `AnnNet` objects to be used in hash-based containers
        (like `set` or `dict` keys).
        - If the graph is **mutated** after hashing (e.g., vertices or edges are added
        or removed), the hash will no longer reflect the new state.
        - The method uses a deterministic representation: sorted vertex/edge sets
        ensure that ordering does not affect the hash.

        """
        G = self._G
        vertex_ids = tuple(sorted(G.vertices()))
        edge_defs = []

        for j in range(G.ne):
            S, T = G.get_edge(j)
            eid = G._col_to_edge[j]
            directed = G._is_directed_edge(eid)
            edge_defs.append((eid, tuple(sorted(S)), tuple(sorted(T)), directed))

        edge_defs = tuple(sorted(edge_defs))

        graph_meta = (
            tuple(sorted(G.graph_attributes.items())) if hasattr(G, 'graph_attributes') else ()
        )

        return hash((vertex_ids, edge_defs, graph_meta))
