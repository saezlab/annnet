class LayerManager:
    """Manager for Kivela multi-layer operations.

    Provides organized namespace for layer operations by delegating to Graph methods.
    All heavy lifting is done by the Graph class; this is just a clean API surface.

    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Multi-aspect awareness ====================

    def aspects(self):
        """List aspect names."""
        return list(self._G.aspects)

    def elementary_layers(self):
        """Dict[aspect -> list of elementary labels]."""
        return {a: list(v) for a, v in self._G.elem_layers.items()}

    def layer_tuples(self):
        """List all aspect-tuples (Cartesian product)."""
        return list(self._G.iter_layers())

    def tuple_id(self, aa):
        """Canonical string id for a layer tuple (matches Graph’s synthetic id)."""
        aa = tuple(aa)
        if len(self._G.aspects) == 1:
            return aa[0]
        return "×".join(aa)

    # ==================== Presence utilities ====================

    def vertex_layers(self, u: str):
        """All layer-tuples where vertex u is present."""
        return list(self._G.iter_vertex_layers(u))

    def has_presence(self, u: str, aa):
        """True if (u, aa) ∈ V_M."""
        return self._G.has_presence(u, tuple(aa))

    # ==================== Aspect / layer / vertex-layer attributes ===========

    def set_aspect_attrs(self, aspect: str, **attrs):
        """Attach metadata to an aspect (delegates to Graph.set_aspect_attrs)."""
        return self._G.set_aspect_attrs(aspect, **attrs)

    def aspect_attrs(self, aspect: str) -> dict:
        """Get metadata dict for an aspect."""
        return self._G.get_aspect_attrs(aspect)

    def set_layer_attrs(self, aa, **attrs):
        """Attach metadata to a Kivela layer aa (aspect tuple)."""
        return self._G.set_layer_attrs(tuple(aa), **attrs)

    def layer_attrs(self, aa) -> dict:
        """Get metadata dict for a Kivela layer aa (aspect tuple)."""
        return self._G.get_layer_attrs(tuple(aa))

    def set_vertex_layer_attrs(self, u: str, aa, **attrs):
        """Attach metadata to a vertex–layer pair (u, aa)."""
        return self._G.set_vertex_layer_attrs(u, tuple(aa), **attrs)

    def vertex_layer_attrs(self, u: str, aa) -> dict:
        """Get metadata dict for a vertex–layer pair (u, aa)."""
        return self._G.get_vertex_layer_attrs(u, tuple(aa))

    # ==================== Elementary layer attributes ===========

    def elem_layer_id(self, aspect: str, label: str) -> str:
        """Canonical '{aspect}_{label}' id used in Graph.layer_attributes."""
        return self._G._elem_layer_id(aspect, label)

    def set_elem_layer_attrs(self, aspect: str, label: str, **attrs):
        """
        Upsert attributes for elementary Kivela layer (aspect, label).

        Writes into Graph.layer_attributes with layer_id = "{aspect}_{label}".
        """
        return self._G.set_elementary_layer_attrs(aspect, label, **attrs)

    def elem_layer_attrs(self, aspect: str, label: str) -> dict:
        """
        Read attributes for elementary Kivela layer (aspect, label) as dict.

        Reads from Graph.layer_attributes.
        """
        return self._G.get_elementary_layer_attrs(aspect, label)

    # ==================== Algebra on Kivela layers =========================

    def vertex_set(self, aa):
        """Vertices present in Kivela layer aa (tuple)."""
        return self._G.layer_vertex_set(tuple(aa))

    def edge_set(self, aa, include_inter=False, include_coupling=False):
        """Edges associated with Kivela layer aa."""
        return self._G.layer_edge_set(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-union over Kivela layers; returns {'vertices', 'edges'}."""
        return self._G.layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-intersection over Kivela layers; returns {'vertices', 'edges'}."""
        return self._G.layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Set-difference layer_a layer_b; returns {'vertices', 'edges'}."""
        return self._G.layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Layer-slice bridge ===============================

    def to_slice(self, aa, slice_id=None, include_inter=False, include_coupling=False, **attrs):
        """
        Create a slice from a single Kivela layer aa and return its slice_id.
        """
        aa = tuple(aa)
        sid = slice_id or self.tuple_id(aa)
        return self._G.create_slice_from_layer(
            sid,
            aa,
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def union_to_slice(self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from union of several Kivela layers.
        """
        return self._G.create_slice_from_layer_union(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def intersection_to_slice(self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from intersection of several Kivela layers.
        """
        return self._G.create_slice_from_layer_intersection(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def difference_to_slice(self, layer_a, layer_b, slice_id, include_inter=False, include_coupling=False, **attrs):
        """
        Create slice from set-difference layer_a layer_b.
        """
        return self._G.create_slice_from_layer_difference(
            slice_id,
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    # ==================== Subgraphs =======================================

    def subgraph(self, aa, include_inter=False, include_coupling=False):
        """Concrete subgraph induced by Kivela layer aa."""
        return self._G.subgraph_from_layer_tuple(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by union of several Kivela layers."""
        return self._G.subgraph_from_layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by intersection of several Kivela layers."""
        return self._G.subgraph_from_layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Subgraph induced by layer_a layer_b."""
        return self._G.subgraph_from_layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Intra/inter/coupling surfacing ====================

    def intra_edges_tuple(self, aa):
        """Edge IDs of intra edges inside tuple-layer aa."""
        aa = tuple(aa)
        # intra appear in Graph.edge_kind with edge_layers[eid] == aa
        return {eid for eid, k in self._G.edge_kind.items() if k == "intra" and self._G.edge_layers[eid] == aa}

    def inter_edges_between(self, aa, bb):
        """Edge IDs of inter edges between tuple-layers aa and bb."""
        aa = tuple(aa); bb = tuple(bb)
        return {eid for eid, k in self._G.edge_kind.items() if k == "inter" and self._G.edge_layers[eid] == (aa, bb)}

    def coupling_edges_between(self, aa, bb):
        """Edge IDs of coupling edges connecting same-vertex (aa)↔(bb)."""
        aa = tuple(aa); bb = tuple(bb)
        return {eid for eid, k in self._G.edge_kind.items() if k == "coupling" and self._G.edge_layers[eid] == (aa, bb)}

    # ==================== Supra / blocks ====================

    def supra_adjacency(self, layers=None):
        """Proxy to full supra A over selected layer-tuples."""
        return self._G.supra_adjacency(layers)

    def blocks(self, layers=None):
        """Return dict of diagonal/off-diagonal blocks."""
        return {
            "intra": self._G.build_intra_block(layers),
            "inter": self._G.build_inter_block(layers),
            "coupling": self._G.build_coupling_block(layers),
        }
