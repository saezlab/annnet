# Traversal (neighbors)


def _hyper_meta(rec):
    """Build the hyperedge metadata dict from an EdgeRecord."""
    if rec.tgt is not None:
        return {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
    return {'directed': False, 'members': set(rec.src)}


class Traversal:
    """Local neighborhood traversal over the incidence-backed graph.

    Binary adjacency is answered from the ``_src_to_edges`` / ``_tgt_to_edges`` indices in
    O(degree); hyperedges are handled from a per-version cached list, so a graph with no
    hyperedges never pays a full edge scan.
    """

    def _iter_hyperedges(self):
        """Return ``(eid, rec)`` for live hyperedges, cached against the structural clock.

        Keys on ``_structure_version``, not ``_version``: the latter is a history
        counter that does not move on removes, so a hyperedge list warmed before a
        removal would survive it and report neighbors through a deleted edge.
        """
        version = getattr(self, '_structure_version', None)
        cache = getattr(self, '_hyper_items_cache', None)
        if cache is None or cache[0] != version:
            items = [
                (eid, rec)
                for eid, rec in self._edges.items()
                if rec.etype == 'hyper' and rec.col_idx >= 0
            ]
            self._hyper_items_cache = (version, items)
            return items
        return cache[1]

    def neighbors(self, entity_id):
        """Return adjacent entities for a vertex or edge-entity.

        Parameters
        ----------
        entity_id : str
            Entity identifier.

        Returns
        -------
        list[str]
            Neighbor identifiers reachable through incident edges.
        """
        ekey = self._resolve_entity_key(entity_id)
        if ekey not in self._entities:
            return []
        self._ensure_edge_indexes()

        out = set()
        default_dir = self.directed if self.directed is not None else True
        ekey_kind = self._entities[ekey].kind
        probe = ekey if self._aspects != ('_',) else entity_id
        edges = self._edges

        for eid in self._src_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx >= 0:
                out.add(rec.tgt)
        for eid in self._tgt_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx < 0:
                continue
            edir = rec.directed if rec.directed is not None else default_dir
            if (not edir) or ekey_kind == 'edge_entity':
                out.add(rec.src)

        for _eid, rec in self._iter_hyperedges():
            meta = _hyper_meta(rec)
            if meta['directed']:
                if probe in meta['head']:
                    out |= meta['tail']
                elif probe in meta['tail']:
                    out |= meta['head']
            else:
                members = meta['members']
                if probe in members:
                    out |= members - {probe}
        return list(out)

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
        ekey = self._resolve_entity_key(vertex_id)
        if ekey not in self._entities:
            return []
        self._ensure_edge_indexes()

        out = set()
        default_dir = self.directed if self.directed is not None else True
        probe = ekey if self._aspects != ('_',) else vertex_id
        edges = self._edges

        for eid in self._src_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx >= 0:
                out.add(rec.tgt)
        for eid in self._tgt_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx < 0:
                continue
            edir = rec.directed if rec.directed is not None else default_dir
            if not edir:
                out.add(rec.src)

        for _eid, rec in self._iter_hyperedges():
            meta = _hyper_meta(rec)
            if meta['directed']:
                if probe in meta['head']:
                    out |= meta['tail']
            else:
                members = meta['members']
                if probe in members:
                    out |= members - {probe}
        return list(out)

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
        ekey = self._resolve_entity_key(vertex_id)
        if ekey not in self._entities:
            return []
        self._ensure_edge_indexes()

        inn = set()
        default_dir = self.directed if self.directed is not None else True
        probe = ekey if self._aspects != ('_',) else vertex_id
        edges = self._edges

        for eid in self._tgt_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx >= 0:
                inn.add(rec.src)
        for eid in self._src_to_edges.get(probe, ()):
            rec = edges[eid]
            if rec.col_idx < 0:
                continue
            edir = rec.directed if rec.directed is not None else default_dir
            if not edir:
                inn.add(rec.tgt)

        for _eid, rec in self._iter_hyperedges():
            meta = _hyper_meta(rec)
            if meta['directed']:
                if probe in meta['tail']:
                    inn |= meta['head']
            else:
                members = meta['members']
                if probe in members:
                    inn |= members - {probe}
        return list(inn)

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
