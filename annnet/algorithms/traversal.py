# Traversal (neighbors)
class Traversal:
    def neighbors(self, entity_id):
        """Neighbors of an entity (vertex or edge-entity).

        Parameters
        --
        entity_id : str

        Returns
        ---
        list[str]
            Adjacent entities. For hyperedges, uses head/tail orientation.

        """
        if entity_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if entity_id in meta["head"]:
                        out |= meta["tail"]
                    elif entity_id in meta["tail"]:
                        out |= meta["head"]
                else:
                    if ("members" in meta) and (entity_id in meta["members"]):
                        out |= meta["members"] - {entity_id}
            else:
                # binary / vertex_edge
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == entity_id:
                    out.add(t)
                elif t == entity_id and (not edir or self.entity_types.get(entity_id) == "edge"):
                    out.add(s)
        return list(out)

    def out_neighbors(self, vertex_id):
        """Out-neighbors of a vertex under directed semantics.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["head"]:
                        out |= meta["tail"]
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def successors(self, vertex_id):
        """Successors of a vertex under directed semantics.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        out = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["head"]:
                        out |= meta["tail"]
                else:
                    if vertex_id in meta.get("members", ()):
                        out |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if s == vertex_id:
                    out.add(t)
                elif t == vertex_id and not edir:
                    out.add(s)
        return list(out)

    def in_neighbors(self, vertex_id):
        """In-neighbors of a vertex under directed semantics.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        inn = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["tail"]:
                        inn |= meta["head"]
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)

    def predecessors(self, vertex_id):
        """In-neighbors of a vertex under directed semantics.

        Parameters
        --
        vertex_id : str

        Returns
        ---
        list[str]

        """
        if vertex_id not in self.entity_to_idx:
            return []
        inn = set()
        for eid in self.edge_to_idx.keys():
            kind = self.edge_kind.get(eid, None)
            if kind == "hyper":
                meta = self.hyperedge_definitions[eid]
                if meta["directed"]:
                    if vertex_id in meta["tail"]:
                        inn |= meta["head"]
                else:
                    if vertex_id in meta.get("members", ()):
                        inn |= meta["members"] - {vertex_id}
            else:
                s, t, _ = self.edge_definitions[eid]
                edir = self.edge_directed.get(eid, True if self.directed is None else self.directed)
                if t == vertex_id:
                    inn.add(s)
                elif s == vertex_id and not edir:
                    inn.add(t)
        return list(inn)
