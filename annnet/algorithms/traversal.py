# Traversal (neighbors)


def _hyper_meta(rec):
    """Build the hyperedge metadata dict from an EdgeRecord."""
    if rec.tgt is not None:
        return {'directed': True, 'head': set(rec.src), 'tail': set(rec.tgt)}
    return {'directed': False, 'members': set(rec.src)}


class Traversal:
    """Local neighborhood traversal helpers over the incidence-backed graph."""

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

        out = set()
        default_dir = self.directed if self.directed is not None else True
        ekey_kind = self._entities[ekey].kind

        for eid, rec in self._edges.items():
            if rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                meta = _hyper_meta(rec)
                if meta['directed']:
                    if entity_id in meta['head']:
                        out |= meta['tail']
                    elif entity_id in meta['tail']:
                        out |= meta['head']
                else:
                    members = meta.get('members', set())
                    if entity_id in members:
                        out |= members - {entity_id}
            else:
                s, t = rec.src, rec.tgt
                if s is None or t is None:
                    continue
                edir = rec.directed if rec.directed is not None else default_dir
                if s == entity_id:
                    out.add(t)
                elif t == entity_id and (not edir or ekey_kind == 'edge_entity'):
                    out.add(s)
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

        out = set()
        default_dir = self.directed if self.directed is not None else True

        for eid, rec in self._edges.items():
            if rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                meta = _hyper_meta(rec)
                if meta['directed']:
                    if vertex_id in meta['head']:
                        out |= meta['tail']
                else:
                    members = meta.get('members', set())
                    if vertex_id in members:
                        out |= members - {vertex_id}
            else:
                s, t = rec.src, rec.tgt
                if s is None or t is None:
                    continue
                edir = rec.directed if rec.directed is not None else default_dir
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
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

        inn = set()
        default_dir = self.directed if self.directed is not None else True

        for eid, rec in self._edges.items():
            if rec.col_idx < 0:
                continue
            if rec.etype == 'hyper':
                meta = _hyper_meta(rec)
                if meta['directed']:
                    if vertex_id in meta['tail']:
                        inn |= meta['head']
                else:
                    members = meta.get('members', set())
                    if vertex_id in members:
                        inn |= members - {vertex_id}
            else:
                s, t = rec.src, rec.tgt
                if s is None or t is None:
                    continue
                edir = rec.directed if rec.directed is not None else default_dir
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
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
