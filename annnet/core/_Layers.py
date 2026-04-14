from __future__ import annotations

import copy
import itertools
import warnings
from typing import TYPE_CHECKING

import narwhals as nw
import numpy as np

try:
    import polars as pl
except Exception:
    pl = None
import scipy.sparse as sp

if TYPE_CHECKING:
    from .graph import AnnNet

from ._helpers import EdgeRecord, EntityRecord, build_dataframe_from_rows


class LayerManager:
    """Namespace for multilayer operations.

    ``LayerManager`` is the primary public entry point for aspect declaration,
    elementary layer metadata, supra-node presence, layer-induced set algebra,
    and supra-matrix constructions. Methods delegate to the underlying
    :class:`annnet.core.graph.AnnNet` instance.
    """

    def __init__(self, graph):
        self._G = graph

    # ==================== Multi-aspect awareness ====================

    def aspects(self):
        """List aspect names.

        Returns
        -------
        list[str]
            Aspect identifiers in configured order.
        """
        return list(self._G.aspects)

    def elementary_layers(self):
        """Return elementary layer labels per aspect.

        Returns
        -------
        dict[str, list[str]]
            Mapping of aspect name to list of elementary labels.
        """
        return {a: list(v) for a, v in self._G.elem_layers.items()}

    def set_aspects(self, aspects):
        """Declare the aspect dimensions for the graph."""
        return self._G.set_aspects(aspects)

    def set_elementary_layers(self, layers_by_aspect: dict[str, list[str]]):
        """Declare concrete elementary layer values for existing aspects."""
        return self._G.set_elementary_layers(layers_by_aspect)

    def layers(self):
        """List all realized layer tuples."""
        return list(self._G.iter_layers())

    def layer_id(self, aa):
        """Canonical string id for a layer tuple."""
        aa = tuple(aa)
        if len(self._G.aspects) == 1:
            return aa[0]
        return "×".join(aa)

    # ==================== Presence utilities ====================

    def vertex_layers(self, u: str):
        """List all layer-tuples where vertex ``u`` is present.

        Parameters
        ----------
        u : str
            Vertex identifier.

        Returns
        -------
        list[tuple[str, ...]]
            Layer tuples where ``(u, aa)`` is in ``V_M``.
        """
        return list(self._G.iter_vertex_layers(u))

    def has_presence(self, u: str, aa):
        """Check whether ``(u, aa)`` is in ``V_M``.

        Parameters
        ----------
        u : str
            Vertex identifier.
        aa : Iterable[str]
            Layer tuple.

        Returns
        -------
        bool
        """
        return self._G.has_presence(u, tuple(aa))

    # ==================== Aspect / layer / vertex-layer attributes ===========

    def set_aspect_attrs(self, aspect: str, **attrs):
        """Attach metadata to an aspect.

        Parameters
        ----------
        aspect : str
            Aspect identifier.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        return self._G.set_aspect_attrs(aspect, **attrs)

    def aspect_attrs(self, aspect: str) -> dict:
        """Get metadata dict for an aspect.

        Parameters
        ----------
        aspect : str
            Aspect identifier.

        Returns
        -------
        dict
        """
        return self._G.get_aspect_attrs(aspect)

    def set_layer_attrs(self, aa, **attrs):
        """Attach metadata to a Kivela layer (aspect tuple).

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        return self._G.set_layer_attrs(tuple(aa), **attrs)

    def layer_attrs(self, aa) -> dict:
        """Get metadata dict for a Kivela layer (aspect tuple).

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.

        Returns
        -------
        dict
        """
        return self._G.get_layer_attrs(tuple(aa))

    def set_vertex_layer_attrs(self, u: str, aa, **attrs):
        """Attach metadata to a vertex–layer pair.

        Parameters
        ----------
        u : str
            Vertex identifier.
        aa : Iterable[str]
            Layer tuple.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        return self._G.set_vertex_layer_attrs(u, tuple(aa), **attrs)

    def vertex_layer_attrs(self, u: str, aa) -> dict:
        """Get metadata dict for a vertex–layer pair.

        Parameters
        ----------
        u : str
            Vertex identifier.
        aa : Iterable[str]
            Layer tuple.

        Returns
        -------
        dict
        """
        return self._G.get_vertex_layer_attrs(u, tuple(aa))

    # ==================== Elementary layer attributes ===========

    def elementary_layer_id(self, aspect: str, label: str) -> str:
        """Canonical id for an elementary layer."""
        return self._G._elem_layer_id(aspect, label)

    def set_elementary_layer_attrs(self, aspect: str, label: str, **attrs):
        """Upsert attributes for a declared elementary layer.

        Parameters
        ----------
        aspect : str
            Aspect name.
        label : str
            Elementary layer label within ``aspect``.
        **attrs
            Attribute key-value pairs to store.

        Returns
        -------
        None
        """
        return self._G.set_elementary_layer_attrs(aspect, label, **attrs)

    def elementary_layer_attrs(self, aspect: str, label: str) -> dict:
        """Return attributes stored for an elementary layer.

        Parameters
        ----------
        aspect : str
            Aspect name.
        label : str
            Elementary layer label.

        Returns
        -------
        dict
            Attribute mapping for the elementary layer.
        """
        return self._G.get_elementary_layer_attrs(aspect, label)

    # ==================== Algebra on Kivela layers =========================

    def vertex_set(self, aa):
        """Vertices present in a Kivela layer.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.

        Returns
        -------
        set[str]
        """
        return self._G.layer_vertex_set(tuple(aa))

    def edge_set(self, aa, include_inter=False, include_coupling=False):
        """Edges associated with a Kivela layer.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``aa``.
        include_coupling : bool, optional
            Include coupling edges touching ``aa``.

        Returns
        -------
        set[str]
        """
        return self._G.layer_edge_set(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-union over Kivela layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        return self._G.layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Set-intersection over Kivela layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        return self._G.layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Set-difference over two layers.

        Parameters
        ----------
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        return self._G.layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Layer-slice bridge ===============================

    def to_slice(self, aa, slice_id=None, include_inter=False, include_coupling=False, **attrs):
        """Create a slice from a single Kivela layer.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.
        slice_id : str, optional
            Slice identifier. Defaults to the canonical layer id.
        include_inter : bool, optional
            Include inter-layer edges touching ``aa``.
        include_coupling : bool, optional
            Include coupling edges touching ``aa``.
        **attrs
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.

        Examples
        --------
        ```python
        G.layers.to_slice(("t1", "F"), slice_id="t1_F")
        ```
        """
        aa = tuple(aa)
        sid = slice_id or self.layer_id(aa)
        return self._G.create_slice_from_layer(
            sid,
            aa,
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def union_to_slice(
        self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs
    ):
        """Create a slice from the union of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        slice_id : str
            Slice identifier.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.
        **attrs
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
        """
        return self._G.create_slice_from_layer_union(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def intersection_to_slice(
        self, layer_tuples, slice_id, include_inter=False, include_coupling=False, **attrs
    ):
        """Create a slice from the intersection of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        slice_id : str
            Slice identifier.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.
        **attrs
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
        """
        return self._G.create_slice_from_layer_intersection(
            slice_id,
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
            **attrs,
        )

    def difference_to_slice(
        self, layer_a, layer_b, slice_id, include_inter=False, include_coupling=False, **attrs
    ):
        """Create a slice from a set-difference of two layers.

        Parameters
        ----------
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        slice_id : str
            Slice identifier.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.
        **attrs
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
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
        """Concrete subgraph induced by a single layer.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``aa``.
        include_coupling : bool, optional
            Include coupling edges touching ``aa``.

        Returns
        -------
        AnnNet
        """
        return self._G.subgraph_from_layer_tuple(
            tuple(aa),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_union(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by the union of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.

        Returns
        -------
        AnnNet
        """
        return self._G.subgraph_from_layer_union(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_intersection(self, layer_tuples, include_inter=False, include_coupling=False):
        """Subgraph induced by the intersection of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.

        Returns
        -------
        AnnNet
        """
        return self._G.subgraph_from_layer_intersection(
            [tuple(a) for a in layer_tuples],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    def subgraph_difference(self, layer_a, layer_b, include_inter=False, include_coupling=False):
        """Subgraph induced by a set-difference of two layers.

        Parameters
        ----------
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.

        Returns
        -------
        AnnNet
        """
        return self._G.subgraph_from_layer_difference(
            tuple(layer_a),
            tuple(layer_b),
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

    # ==================== Intra/inter/coupling surfacing ====================

    def intra_edges_tuple(self, aa):
        """Edge IDs of intra edges inside a layer.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple.

        Returns
        -------
        set[str]
        """
        aa = tuple(aa)
        return {
            eid
            for eid, rec in self._G._edges.items()
            if rec.ml_kind == "intra" and rec.ml_layers == aa
        }

    def inter_edges_between(self, aa, bb):
        """Edge IDs of inter edges between two layers.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple A.
        bb : Iterable[str]
            Layer tuple B.

        Returns
        -------
        set[str]
        """
        aa = tuple(aa)
        bb = tuple(bb)
        return {
            eid
            for eid, rec in self._G._edges.items()
            if rec.ml_kind == "inter" and rec.ml_layers == (aa, bb)
        }

    def coupling_edges_between(self, aa, bb):
        """Edge IDs of coupling edges between two layers.

        Parameters
        ----------
        aa : Iterable[str]
            Layer tuple A.
        bb : Iterable[str]
            Layer tuple B.

        Returns
        -------
        set[str]
        """
        aa = tuple(aa)
        bb = tuple(bb)
        return {
            eid
            for eid, rec in self._G._edges.items()
            if rec.ml_kind == "coupling" and rec.ml_layers == (aa, bb)
        }

    # ==================== Supra / blocks ====================

    def supra_adjacency(self, layers=None):
        """Proxy to full supra adjacency over selected layers.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Layer subset. In single-aspect mode, string IDs are accepted.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self._G.supra_adjacency(layers)

    def supra_incidence(
        self,
        layers: list[str] | list[tuple[str, ...]] | None = None,
        include_inter: bool = True,
        include_coupling: bool = True,
    ) -> tuple[sp.csr_matrix, list[str]]:
        """Build the supra-incidence matrix over selected layers.

            Unlike supra_adjacency, this preserves the full hyperedge structure —
            a k-ary hyperedge becomes a single column with k nonzero entries, with
            stoichiometric coefficients intact. Binary intra, inter, coupling, and
            hyperedges are all handled in a unified column-oriented representation.

            Rows  : vertex-layer pairs (u, aa) — identical index to supra_adjacency,
                    built by ensure_vertex_layer_index.
            Cols  : one per selected edge, ordered as: intra edges (per layer, sorted
                    by eid), then inter/coupling edges, then unassigned hyperedges last.

            Column sign convention (matches _matrix):
                - Binary directed   : +w at source row, -w at target row
                - Binary undirected : +w at both rows
                - Hyperedge directed: +w at head rows, -w at tail rows (stoich-aware)
                - Hyperedge undirected: +w at all member rows (stoich-aware)
                - Inter/coupling    : +w at (u, La) row, -w at (v, Lb) row (directed)

            Hyperedges MUST have a layer assignment in edge_layers (set via
            set_edge_kivela_role(eid, "intra", layer_tuple) after add_hyperedge).
            Hyperedges without a layer assignment are collected in the returned
            skipped list and excluded from the matrix — they do NOT silently corrupt
            the result.

            Parameters
            ----------
            layers : list[str] | list[tuple[str, ...]] | None
                Optional subset of layers. None = all layers in V_M.
                Single-aspect string ids are accepted.
            include_inter : bool
                Include inter-layer edges in the output columns. Default True.
            include_coupling : bool
                Include coupling edges in the output columns. Default True.

            Returns
            -------
            B : scipy.sparse.csr_matrix
                Shape (|V_M|, |E_selected|). Rows are vertex-layer pairs in the
                order given by self._row_to_nl after ensure_vertex_layer_index.
            edge_ids : list[str]
                Edge id for each column of B, in column order. Use this to map
                columns back to edges for interpretability.
            skipped : list[str]
                Edge ids that were excluded because their layer assignment could
                not be resolved. Inspect these if B looks sparse.

            Notes
            -----
            The hypergraph random-walk diffusion operator follows directly::

                B_csr = B  (this output)
                D_v = diag(|B| @ ones)          # vertex degree (sum of |entries| per row)
                D_e = diag(|B|.T @ ones)        # edge degree (sum of |entries| per col)
                Theta = D_v_inv @ B @ D_e_inv @ B.T

            Examples
            --------
        ```python
            B, eids, skipped = G.supra_incidence()
        ```
        """
        return self._G.supra_incidence(layers)

    def blocks(self, layers=None):
        """Return supra block matrices.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Layer subset. In single-aspect mode, string IDs are accepted.

        Returns
        -------
        dict
            ``{"intra": A_intra, "inter": A_inter, "coupling": A_coupling}``.
        """
        return {
            "intra": self._G.build_intra_block(layers),
            "inter": self._G.build_inter_block(layers),
            "coupling": self._G.build_coupling_block(layers),
        }

    def flatten(self):
        """Flatten the owner graph in-place and return it."""
        return self._G.flatten_layers()


class LayerClass:
    # Multilayers

    ## Aspects & layers

    def _placeholder_layer_referenced(self) -> bool:
        placeholder = self._placeholder_layer_coord()
        if any(ekey[1] == placeholder for ekey in self._entities):
            return True
        if any(
            (isinstance(key, tuple) and len(key) == 2 and key[1] == placeholder) for key in self._VM
        ):
            return True
        if any(key[1] == placeholder for key in self._state_attrs):
            return True
        for rec in self._edges.values():
            layers = rec.ml_layers
            if layers is None:
                continue
            if layers == placeholder:
                return True
            if (
                isinstance(layers, tuple)
                and len(layers) == 2
                and all(isinstance(x, tuple) for x in layers)
                and placeholder in layers
            ):
                return True
        return False

    def _drop_unused_placeholder_layers(self) -> None:
        if self._aspects == ("_",):
            return
        if self._placeholder_layer_referenced():
            return
        if not all(
            any(val != "_" for val in self._layers.get(aspect, set())) for aspect in self._aspects
        ):
            return
        for aspect in self._aspects:
            self._layers.get(aspect, set()).discard("_")
        self._rebuild_all_layers_cache()

    def set_aspects(self, aspects, elem_layers: dict[str, list[str]] | None = None):
        """Define multi-aspect structure.

        Parameters
        ----------
        aspects : list[str]
            Aspect identifiers (e.g., ``["time", "relation"]``).
        elem_layers : dict[str, list[str]]
            Elementary labels per aspect (e.g., ``{"time": ["t1","t2"]}``).

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``aspects`` is empty.

        Examples
        --------
        ```python
        G.set_aspects(["time", "relation"], {"time": ["t1", "t2"], "relation": ["F", "A"]})
        ```
        """
        if isinstance(aspects, dict):
            if elem_layers is not None:
                raise ValueError(
                    "Pass either aspects=list[...] or aspects={aspect: layers}, not both."
                )
            elem_layers = aspects
            aspects = list(aspects.keys())

        if not aspects:
            raise ValueError("aspects must be a non-empty list")
        aspects = list(aspects)
        elem_layers = elem_layers or {}
        old_aspects = tuple(getattr(self, "_aspects", ("_",)))
        old_placeholder = ("_",) if old_aspects == ("_",) else tuple("_" for _ in old_aspects)
        new_placeholder = tuple("_" for _ in aspects)

        had_existing_flat_entities = bool(self._entities) and old_aspects == ("_",)

        self._aspects = tuple(aspects)
        self._layers = {}
        for aspect in aspects:
            values = set(elem_layers.get(aspect, []))
            values.add("_")
            self._layers[aspect] = values

        if had_existing_flat_entities:
            self._entities = {
                (vid, new_placeholder if coord == old_placeholder else coord): rec
                for (vid, coord), rec in self._entities.items()
            }
            self._state_attrs = {
                (vid, new_placeholder if coord == old_placeholder else coord): attrs
                for (vid, coord), attrs in self._state_attrs.items()
            }
            self._VM = {
                (vid, new_placeholder if coord == old_placeholder else coord)
                for (vid, coord) in self._VM
            }
            self._nl_to_row = {
                (vid, new_placeholder if coord == old_placeholder else coord): row
                for (vid, coord), row in self._nl_to_row.items()
            }
            self._row_to_nl = [
                (vid, new_placeholder if coord == old_placeholder else coord)
                for (vid, coord) in self._row_to_nl
            ]
            self._rebuild_entity_indexes()
            warnings.warn(
                f"Declared aspects {tuple(aspects)!r}; existing flat vertices were reassigned "
                f"to placeholder layer {new_placeholder!r}. Set explicit layer coordinates if needed.",
                UserWarning,
                stacklevel=2,
            )
        else:
            self._rebuild_entity_indexes()

        self._rebuild_all_layers_cache()
        self._drop_unused_placeholder_layers()

    def set_elementary_layers(self, layers_by_aspect: dict[str, list[str]]):
        """Declare concrete elementary layer values for existing aspects."""
        if self._aspects == ("_",):
            raise ValueError("No aspects are configured; call layers.set_aspects(...) first.")
        for aspect, values in layers_by_aspect.items():
            if aspect not in self._aspects:
                raise KeyError(f"Unknown aspect {aspect!r}. Valid: {list(self._aspects)!r}")
            labels = {str(v) for v in values}
            if not labels:
                raise ValueError(
                    f"Aspect {aspect!r} must receive at least one elementary layer value."
                )
            self._layers.setdefault(aspect, set()).update(labels)
        self._rebuild_all_layers_cache()
        self._drop_unused_placeholder_layers()

    def _rebuild_all_layers_cache(self):
        if not self.aspects:
            self._all_layers = ()
            return
        # build cartesian product (tuple of tuples)
        spaces = [self.elem_layers.get(a, []) for a in self.aspects]
        if not all(spaces) and self.aspects:
            return  # No valid Cartesian product possible
        self._all_layers = tuple(itertools.product(*spaces)) if all(spaces) else ()

    def flatten_layers(self):
        """Remove multilayer structure in-place and project to a flat graph.

        Returns
        -------
        AnnNet
            The mutated graph itself.

        Notes
        -----
        This projects vertex identities from ``(vertex_id, layer_tuple)`` to bare
        ``vertex_id`` strings and drops multilayer-only metadata such as aspects,
        layer registries, supra-node attributes, and multilayer edge roles.
        """
        if self._aspects == ("_",):
            return self

        def _clone_table(df):
            if df is None:
                return None
            if hasattr(df, "clone"):
                return df.clone()
            if hasattr(df, "copy"):
                return df.copy()
            return df

        def _project_node(node):
            if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], tuple):
                return node[0]
            return node

        def _project_members(members):
            if members is None:
                return None
            if isinstance(members, (set, frozenset, list, tuple)):
                return sorted({_project_node(m) for m in members})
            return _project_node(members)

        history_flag = getattr(self, "_history_enabled", True)
        self._history_enabled = False
        try:
            flat = self.__class__(
                directed=self.directed,
                annotations_backend=getattr(self, "_annotations_backend", "polars"),
            )
            flat._history_enabled = False

            flat.graph_attributes = dict(self.graph_attributes)
            flat._vertex_key_fields = self._vertex_key_fields
            flat._vertex_key_index = dict(self._vertex_key_index)
            flat._legacy_single_aspect_enabled = self._legacy_single_aspect_enabled
            flat.vertex_aligned = False
            flat._default_slice = self._default_slice
            flat._current_slice = self._current_slice

            for slice_id, meta in self._slices.items():
                flat._slices[slice_id] = {
                    "vertices": {
                        v[0] if isinstance(v, tuple) and len(v) == 2 else v
                        for v in meta["vertices"]
                    },
                    "edges": set(meta["edges"]),
                    "attributes": dict(meta["attributes"]),
                }

            vertex_ids = sorted(
                {ekey[0] for ekey, rec in self._entities.items() if rec.kind == "vertex"}
            )
            for vid in vertex_ids:
                flat.add_vertex(vid)

            edge_entity_ids = sorted(
                {ekey[0] for ekey, rec in self._entities.items() if rec.kind == "edge_entity"}
            )
            edge_entity_id_set = set(edge_entity_ids)
            for eid in edge_entity_ids:
                if flat._resolve_entity_key(eid) not in flat._entities:
                    flat.add_edge(edge_id=eid, as_entity=True)

            edge_items = sorted(
                self._edges.items(),
                key=lambda item: (
                    item[1].col_idx < 0,
                    item[1].col_idx if item[1].col_idx >= 0 else 10**12,
                    item[0],
                ),
            )
            for eid, rec in edge_items:
                if rec.col_idx < 0 and rec.src is None and rec.tgt is None:
                    if eid not in flat._edges:
                        flat.add_edge(edge_id=eid, as_entity=True)
                    continue

                if rec.etype == "hyper":
                    src = _project_members(rec.src) or []
                    tgt = _project_members(rec.tgt) if rec.tgt is not None else None
                    flat.add_edge(
                        src=src,
                        tgt=tgt,
                        edge_id=eid,
                        weight=rec.weight,
                        directed=rec.directed,
                    )
                else:
                    flat.add_edge(
                        _project_node(rec.src),
                        _project_node(rec.tgt),
                        edge_id=eid,
                        weight=rec.weight,
                        directed=rec.directed,
                        as_entity=(eid in edge_entity_id_set),
                    )
                flat_rec = flat._edges.get(eid)
                if flat_rec is not None and rec.direction_policy is not None:
                    flat_rec.direction_policy = copy.deepcopy(rec.direction_policy)

            flat.slice_edge_weights = {
                lid: dict(weights) for lid, weights in self.slice_edge_weights.items()
            }
            flat.vertex_attributes = _clone_table(self.vertex_attributes)
            flat.edge_attributes = _clone_table(self.edge_attributes)
            flat.slice_attributes = _clone_table(self.slice_attributes)
            flat.edge_slice_attributes = _clone_table(self.edge_slice_attributes)

            # Keep the flat graph's empty layer table/schema and drop multilayer-only state.
            flat._aspect_attrs = {}
            flat._layer_attrs = {}
            flat._state_attrs = {}
            flat._all_layers = ()
            flat._rebuild_entity_indexes()

            self._aspects = flat._aspects
            self._layers = flat._layers
            self._entities = flat._entities
            self._row_to_entity = flat._row_to_entity
            self._vid_to_ekeys = flat._vid_to_ekeys
            self._edges = flat._edges
            self._col_to_edge = flat._col_to_edge
            self._adj = flat._adj
            self._src_to_edges = flat._src_to_edges
            self._tgt_to_edges = flat._tgt_to_edges
            self._matrix = flat._matrix
            self._csr_cache = None
            self._next_edge_id = flat._next_edge_id
            self._slices = flat._slices
            self._current_slice = flat._current_slice
            self._default_slice = flat._default_slice
            self.slice_edge_weights = flat.slice_edge_weights
            self.vertex_attributes = flat.vertex_attributes
            self.edge_attributes = flat.edge_attributes
            self.slice_attributes = flat.slice_attributes
            self.edge_slice_attributes = flat.edge_slice_attributes
            self.layer_attributes = flat.layer_attributes
            self._all_layers = flat._all_layers
            self.vertex_aligned = flat.vertex_aligned
            self._V = flat._V
            self._VM = flat._VM
            self._nl_to_row = flat._nl_to_row
            self._row_to_nl = flat._row_to_nl
            self._aspect_attrs = flat._aspect_attrs
            self._layer_attrs = flat._layer_attrs
            self._state_attrs = flat._state_attrs
            self._vertex_key_fields = flat._vertex_key_fields
            self._vertex_key_index = flat._vertex_key_index
            self._legacy_single_aspect_enabled = flat._legacy_single_aspect_enabled
        finally:
            self._history_enabled = history_flag
        return self

    def add_elementary_layer(self, aspect: str, label: str):
        """
        Register a new elementary layer label under an existing aspect.

        Parameters
        ----------
        aspect : str
            Existing aspect name.
        label : str
            New elementary layer label.

        Returns
        -------
        None
        """
        if aspect not in self._aspects:
            raise KeyError(f"Unknown aspect {aspect!r}")

        if label in self._layers[aspect]:
            return  # already exists

        self._layers[aspect].add(label)
        self._rebuild_all_layers_cache()
        self._drop_unused_placeholder_layers()

    ## Presence (V_M)

    def add_presence(self, u: str, layer_tuple: tuple[str, ...]):
        """Declare that a vertex is present in a layer tuple.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``u`` is not a vertex.
        ValueError
            If ``layer_tuple`` is invalid for the configured aspects.
        """
        self._validate_layer_tuple(layer_tuple)
        ekey = self._resolve_entity_key(u)
        rec = self._entities.get(ekey)
        if rec is None or rec.kind != "vertex":
            raise KeyError(f"'{u}' is not a vertex")
        self._V.add(u)
        self._VM.add((u, tuple(layer_tuple)))

    def remove_presence(self, u: str, layer_tuple: tuple[str, ...]):
        """Remove presence ``(u, aa)`` if it exists.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If ``layer_tuple`` is invalid for the configured aspects.
        """
        self._validate_layer_tuple(layer_tuple)
        try:
            self._VM.remove((u, tuple(layer_tuple)))
        except KeyError:
            pass

    def has_presence(self, u: str, layer_tuple: tuple[str, ...]) -> bool:
        """Check whether ``(u, aa)`` is in ``V_M``.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        bool
        """
        self._validate_layer_tuple(layer_tuple)
        return (u, tuple(layer_tuple)) in self._VM

    def iter_layers(self):
        """Iterate over all aspect-tuples (Cartesian product).

        Yields
        ------
        tuple[str, ...]
            Layer tuples in configured order.
        """
        return iter(self._all_layers)

    def iter_vertex_layers(self, u: str):
        """Iterate layer tuples where ``(u, aa)`` is in ``V_M``.

        Parameters
        ----------
        u : str
            Vertex identifier.

        Yields
        ------
        tuple[str, ...]
            Layer tuples for ``u``.
        """
        for uu, aa in self._VM:
            if uu == u:
                yield aa

    ## Index for supra rows

    def ensure_vertex_layer_index(self, restrict_layers: list[tuple[str, ...]] | None = None):
        """Build stable mapping between vertex–layer tuples and row indices.

        Parameters
        ----------
        restrict_layers : list[tuple[str, ...]] | None, optional
            If provided, index only these layers.

        Returns
        -------
        int
            Number of indexed vertex–layer pairs.

        Notes
        -----
        Ordering is lexicographic by vertex id, then by layer tuple.
        """
        if restrict_layers is not None:
            R = {tuple(x) for x in restrict_layers}
            vm = [(u, aa) for (u, aa) in self._VM if aa in R]
        else:
            vm = list(self._VM)
        vm.sort(key=lambda x: (x[0], x[1]))
        self._row_to_nl = vm
        self._nl_to_row = {nl: i for i, nl in enumerate(vm)}
        return len(vm)

    def nl_to_row(self, u: str, layer_tuple: tuple[str, ...]) -> int:
        """Map ``(u, aa)`` to row index.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        int

        Raises
        ------
        KeyError
            If the vertex–layer pair is not indexed.
        """
        key = (u, tuple(layer_tuple))
        if key not in self._nl_to_row:
            raise KeyError(f"vertex–layer {key!r} not indexed; call ensure_vertex_layer_index()")
        return self._nl_to_row[key]

    def row_to_nl(self, row: int) -> tuple[str, tuple[str, ...]]:
        """Map row index to ``(u, aa)``.

        Parameters
        ----------
        row : int
            Row index.

        Returns
        -------
        tuple[str, tuple[str, ...]]

        Raises
        ------
        KeyError
            If the row is not indexed.
        """
        try:
            return self._row_to_nl[row]
        except Exception:
            raise KeyError(f"row {row} not in vertex–layer index")

    ## Validation helpers

    def _validate_layer_tuple(self, aa: tuple[str, ...]):
        if self._aspects == ("_",):
            raise ValueError("no aspects are configured; call set_aspects(...) first")
        if len(aa) != len(self._aspects):
            raise ValueError(
                f"layer tuple rank mismatch: expected {len(self._aspects)}, got {len(aa)}"
            )
        for i, a in enumerate(self._aspects):
            allowed = self._layers.get(a, set())
            if aa[i] not in allowed:
                raise KeyError(f"unknown elementary layer {aa[i]!r} for aspect {a!r}")

    def layer_id_to_tuple(self, layer_id: str) -> tuple[str, ...]:
        """Map legacy string layer id to aspect tuple.

        Parameters
        ----------
        layer_id : str
            Layer identifier (single-aspect only).

        Returns
        -------
        tuple[str, ...]

        Raises
        ------
        ValueError
            If not in single-aspect mode.
        """
        if len(self._aspects) != 1:
            raise ValueError("layer_id_to_tuple is only valid when exactly 1 aspect is configured")
        return (layer_id,)

    def _layer_id_to_tuple_cached(self, layer: str):
        if not hasattr(self, "_layer_tuple_cache"):
            self._layer_tuple_cache = {}
        if layer not in self._layer_tuple_cache:
            self._layer_tuple_cache[layer] = self.layer_id_to_tuple(layer)
        return self._layer_tuple_cache[layer]

    def layer_tuple_to_id(self, aa: tuple[str, ...]) -> str:
        """Canonical string id for a layer tuple.

        Parameters
        ----------
        aa : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        str
            Canonical id (single label for 1 aspect, or ``"×"``-joined).
        """
        aa = tuple(aa)
        if len(self._aspects) == 1:
            return aa[0]
        return "×".join(aa)

    ## Aspect / layer / vertex–layer attributes

    def _elem_layer_id(self, aspect: str, label: str) -> str:
        """
        Canonical id for an *elementary* Kivela layer (aspect, label).

        This is the key used in `layer_attributes.layer_id`:
            layer_id = "{aspect}_{label}"
        """
        if aspect not in self._aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {list(self._aspects)!r}")
        allowed = self._layers.get(aspect, set())
        if label not in allowed:
            raise KeyError(
                f"unknown elementary layer {label!r} for aspect {aspect!r}; known: {sorted(allowed)!r}"
            )
        return f"{aspect}_{label}"

    def _upsert_layer_attribute_row(self, layer_id: str, attrs: dict):
        """
        Upsert a row in `self.layer_attributes` for `layer_id`.

        Strategy (simple & robust):
          - convert current DF to list[dict]
          - find existing row for this layer_id (if any)
          - merge attrs into that row (override keys)
          - rebuild DataFrame from the updated list of rows

        This avoids all schema/dtype headaches (Polars infers them).
        """
        df = self.layer_attributes

        # Convert existing DF to list of dict rows
        rows = df.to_dicts() if df.height > 0 else []

        # Find if we already have a row for this layer_id
        existing = None
        new_rows = []
        for r in rows:
            if r.get("layer_id") == layer_id:
                existing = r
                # don't append the old version
            else:
                new_rows.append(r)

        if existing is None:
            base = {"layer_id": layer_id}
        else:
            base = dict(existing)  # copy

        # Merge new attrs (override old keys)
        base.update(attrs)

        # Append updated row
        new_rows.append(base)

        # Rebuild DF; Polars will infer schema and fill missing values with nulls
        self.layer_attributes = build_dataframe_from_rows(new_rows)

    def set_elementary_layer_attrs(self, aspect: str, label: str, **attrs):
        """Attach attributes to an elementary Kivela layer.

        Parameters
        ----------
        aspect : str
            Aspect identifier.
        label : str
            Elementary layer label.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        lid = self._elem_layer_id(aspect, label)
        self._upsert_layer_attribute_row(lid, attrs)

    def get_elementary_layer_attrs(self, aspect: str, label: str) -> dict:
        """Get attributes for an elementary Kivela layer.

        Parameters
        ----------
        aspect : str
            Aspect identifier.
        label : str
            Elementary layer label.

        Returns
        -------
        dict
            Attributes dict; empty if not set.
        """
        lid = self._elem_layer_id(aspect, label)
        df = self.layer_attributes
        if df.height == 0 or "layer_id" not in df.columns:
            return {}

        if pl is not None and isinstance(df, pl.DataFrame):
            rows = df.filter(pl.col("layer_id") == lid)
            if rows.height == 0:
                return {}
        else:
            rows = nw.to_native(nw.from_native(df).filter(nw.col("layer_id") == lid))
            if (hasattr(rows, "__len__") and len(rows) == 0) or (
                getattr(rows, "height", None) == 0
            ):
                return {}

        # single row: drop 'layer_id' and convert to dict
        row = rows.drop("layer_id").to_dicts()[0]
        return row

    def set_aspect_attrs(self, aspect: str, **attrs):
        """Attach metadata to a Kivela aspect.

        Parameters
        ----------
        aspect : str
            Aspect identifier.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        if aspect not in self._aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {list(self._aspects)!r}")
        d = self._aspect_attrs.setdefault(aspect, {})
        d.update(attrs)

    def get_aspect_attrs(self, aspect: str) -> dict:
        """Return a shallow copy of metadata for a Kivela aspect.

        Parameters
        ----------
        aspect : str
            Aspect identifier.

        Returns
        -------
        dict
        """
        if aspect not in self._aspects:
            raise KeyError(f"unknown aspect {aspect!r}")
        return dict(self._aspect_attrs.get(aspect, {}))

    def set_layer_attrs(self, layer_tuple: tuple[str, ...], **attrs):
        """Attach metadata to a Kivela layer.

        Parameters
        ----------
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None
        """
        aa = tuple(layer_tuple)
        self._validate_layer_tuple(aa)
        d = self._layer_attrs.setdefault(aa, {})
        d.update(attrs)

    def get_layer_attrs(self, layer_tuple: tuple[str, ...]) -> dict:
        """Get metadata dict for a Kivela layer.

        Parameters
        ----------
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        dict
            Shallow copy; empty if not set.
        """
        aa = tuple(layer_tuple)
        self._validate_layer_tuple(aa)
        return dict(self._layer_attrs.get(aa, {}))

    def set_vertex_layer_attrs(self, u: str, layer_tuple: tuple[str, ...], **attrs):
        """Attach metadata to a vertex–layer pair.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.
        **attrs
            Key-value metadata to store.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``(u, layer_tuple)`` is not present in ``V_M``.
        """
        aa = tuple(layer_tuple)
        self._assert_presence(u, aa)  # enforce that (u,aa) exists in V_M
        key = (u, aa)
        d = self._state_attrs.setdefault(key, {})
        d.update(attrs)

    def get_vertex_layer_attrs(self, u: str, layer_tuple: tuple[str, ...]) -> dict:
        """Get metadata dict for a vertex–layer pair.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.

        Returns
        -------
        dict
            Shallow copy; empty if not set.
        """
        aa = tuple(layer_tuple)
        key = (u, aa)
        return dict(self._state_attrs.get(key, {}))

    ## explicit vertex–layer edge APIs

    def _layer_tuple_to_lid(self, aa: tuple[str, ...]) -> str:
        if not hasattr(self, "_lid_cache"):
            self._lid_cache = {}
        if aa not in self._lid_cache:
            self._lid_cache[aa] = aa[0] if len(self.aspects) == 1 else "×".join(aa)
        return self._lid_cache[aa]

    def set_edge_kivela_role(self, eid: str, role: str, layers):
        """Annotate an existing structural edge with Kivela semantics.

        Parameters
        ----------
        eid : str
            Edge identifier.
        role : str
            One of ``"intra"``, ``"inter"``, or ``"coupling"``.
        layers : tuple | list
            ``aa`` for ``"intra"``; ``(aa, bb)`` for ``"inter"``/``"coupling"``.

        Returns
        -------
        None

        Raises
        ------
        KeyError
            If ``eid`` does not exist.
        ValueError
            If ``role`` is invalid.
        """
        warnings.warn(
            "set_edge_kivela_role() is deprecated; multilayer role should be inferred from "
            "explicit supra-node endpoints in add_edge().",
            DeprecationWarning,
            stacklevel=2,
        )
        # Sanity: edge must exist in the structural registry
        rec = self._edges.get(eid)
        if rec is None:
            raise KeyError(
                f"Kivela annotation for unknown edge {eid!r}; "
                "create the edge via add_edge/add_hyperedge first."
            )

        role = str(role)
        if role == "intra":
            aa = tuple(layers)
            self._validate_layer_tuple(aa)
            # preserve "hyper" kind — hyperedges tagged as intra keep their topology
            if rec.etype != "hyper":
                rec.ml_kind = "intra"
            rec.ml_layers = aa
        elif role in {"inter", "coupling"}:
            La, Lb = layers
            La = tuple(La)
            Lb = tuple(Lb)
            self._validate_layer_tuple(La)
            self._validate_layer_tuple(Lb)
            rec.ml_kind = role
            rec.ml_layers = (La, Lb)
        else:
            raise ValueError(f"unknown Kivela role {role!r}")

    def add_intra_edge(
        self, u: str, v: str, layer: str, *, weight: float = 1.0, eid: str | None = None
    ):
        """Add an intra-layer edge in legacy single-aspect mode.

        Parameters
        ----------
        u : str
            Source vertex.
        v : str
            Target vertex.
        layer : str
            Layer id (single-aspect only).
        weight : float, optional
            Edge weight.
        eid : str | None, optional
            Edge id override.

        Returns
        -------
        str
            Edge id.
        """
        if len(getattr(self, "aspects", [])) == 1 and getattr(
            self, "_legacy_single_aspect_enabled", True
        ):
            aa = self._layer_id_to_tuple_cached(layer)
            return self.add_intra_edge_nl(u, v, aa, weight=weight, eid=eid)
        # Fallback when multi-aspect is not configured: let add_edge handle bookkeeping.
        eid = eid or f"{u}--{v}@{layer}"
        self.add_edge(u, v, layer=layer, weight=weight, edge_id=eid)
        # In legacy single-layer mode we don't have a full aspect tuple; store as ("layer",)
        if self.aspects:
            aa = self._layer_id_to_tuple_cached(layer) if len(self.aspects) == 1 else (layer,)
            self.set_edge_kivela_role(eid, "intra", aa)
        else:
            # no aspects configured -> treat as plain edge; leave edge_kind/edge_layers unset
            pass
        return eid

    def add_intra_edge_nl(
        self,
        u: str,
        v: str,
        layer_tuple: tuple[str, ...],
        *,
        weight: float = 1.0,
        eid: str | None = None,
    ):
        """Add an intra-layer edge for a multi-aspect layer.

        Parameters
        ----------
        u : str
            Source vertex.
        v : str
            Target vertex.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer ``aa``.
        weight : float, optional
            Edge weight.
        eid : str | None, optional
            Edge id override.

        Returns
        -------
        str
            Edge id.

        Raises
        ------
        KeyError
            If required vertex–layer presence is missing.
        ValueError
            If ``layer_tuple`` is invalid.

        Examples
        --------
        ```python
        G.add_intra_edge_nl("u1", "u2", ("t1", "F"))
        ```
        """
        warnings.warn(
            "add_intra_edge_nl() is deprecated; prefer add_edge((u, layer), (v, layer), ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        aa = layer_tuple if isinstance(layer_tuple, tuple) else tuple(layer_tuple)
        if not getattr(self, "_fast_mode", False):
            self._validate_layer_tuple(aa)
            self._assert_presence(u, aa)
            self._assert_presence(v, aa)
        Lid = self._layer_tuple_to_lid(aa)
        eid = eid or f"{u}>{v}@{Lid}"
        # Use a synthetic layer id for intra edges so existing slice bookkeeping runs.
        self.add_edge(u, v, layer=Lid, weight=weight, edge_id=eid)
        # Pure Kivela annotation:
        self.set_edge_kivela_role(eid, "intra", aa)
        return eid

    def add_intra_edges_bulk(self, edges, layer_tuple, weight=1.0, fast_mode=True):
        """Bulk add intra-layer edges with direct matrix updates.

        Parameters
        ----------
        edges : Iterable
            ``[(u, v), ...]`` or ``[(u, v, w), ...]``.
        layer_tuple : tuple[str, ...]
            Aspect tuple layer.
        weight : float, optional
            Default weight when ``w`` is not provided.
        fast_mode : bool, optional
            If True, skip validation/presence checks.

        Returns
        -------
        list[str]
            Edge ids added.
        """
        aa = tuple(layer_tuple)

        if not fast_mode:
            self._validate_layer_tuple(aa)
            for edge in edges:
                u = edge[0]
                v = edge[1]
                self._assert_presence(u, aa)
                self._assert_presence(v, aa)

        Lid = self._layer_tuple_to_lid(aa)
        _ents = self._entities
        _edges = self._edges
        _col_to_edge = self._col_to_edge
        slices = self._slices
        M = self._matrix

        # Intern hot strings + collect edge data
        import sys as _sys

        all_vertices: set[str] = set()
        edge_data: list[tuple[str, str, float]] = []
        for e in edges:
            if isinstance(e, tuple):
                u, v = e[0], e[1]
                w = float(e[2]) if len(e) > 2 else float(weight)
            else:
                u, v = e["u"], e["v"]
                w = float(e.get("weight", weight))
            if isinstance(u, str):
                u = _sys.intern(u)
            if isinstance(v, str):
                v = _sys.intern(v)
            all_vertices.update((u, v))
            edge_data.append((u, v, w))

        # Ensure all vertices exist in _entities — compute coord once
        coord = self._make_layer_coord(aa)
        for vid in all_vertices:
            ekey = (vid, coord)
            if ekey not in _ents:
                idx = len(_ents)
                _ents[ekey] = EntityRecord(row_idx=idx, kind="vertex")
                self._row_to_entity[idx] = ekey
                self._V.add(vid)
                self._VM.add(ekey)

        # Grow rows once if needed
        self._grow_rows_to(len(_ents))

        # Build vertex → row_idx cache (O(1) per lookup in inner loop)
        row_cache: dict[str, int] = {vid: _ents[(vid, coord)].row_idx for vid in all_vertices}

        # Pre-size columns for new edges (all edge_data are assumed new)
        num_new = len(edge_data)
        start_col = len(_col_to_edge)
        self._grow_cols_to(start_col + num_new)

        # Batch edge directedness
        is_dir = bool(self.directed) if self.directed is not None else False

        # scipy fast-set bypass
        _m_fast_set = getattr(M, "_set_intXint", None)

        edge_ids: list[str] = []
        col_idx = start_col

        for u, v, w in edge_data:
            base_eid = f"{u}>{v}@{Lid}"
            eid = base_eid
            k = 1
            while eid in _edges:
                eid = f"{base_eid}#{k}"
                k += 1

            if eid in _edges:
                # Update existing (rare in bulk creation) — just reuse column
                rec = _edges[eid]
                c = rec.col_idx
            else:
                # Create new column
                c = col_idx
                col_idx += 1
                rec = EdgeRecord(
                    src=u,
                    tgt=v,
                    weight=w,
                    directed=is_dir,
                    etype="binary",
                    col_idx=c,
                    ml_kind="intra",
                    ml_layers=aa,
                    direction_policy=None,
                )
                _edges[eid] = rec
                _col_to_edge[c] = eid
                self._adj.setdefault((u, v), []).append(eid)
                self._src_to_edges.setdefault(u, []).append(eid)
                self._tgt_to_edges.setdefault(v, []).append(eid)

            # Write matrix
            fw = float(w)
            src_row = row_cache[u]
            tgt_row = row_cache[v]
            if _m_fast_set is not None:
                _m_fast_set(src_row, c, fw)
                if u != v:
                    _m_fast_set(tgt_row, c, fw)
            else:
                M[src_row, c] = fw
                if u != v:
                    M[tgt_row, c] = fw

            edge_ids.append(eid)

        # Batch slice registration
        if Lid not in slices:
            slices[Lid] = {"vertices": set(), "edges": set(), "attributes": {}}
        slices[Lid]["edges"].update(edge_ids)
        slices[Lid]["vertices"].update(all_vertices)

        return edge_ids

    def add_inter_edge_nl(
        self,
        u: str,
        layer_a: tuple[str, ...],
        v: str,
        layer_b: tuple[str, ...],
        *,
        weight: float = 1.0,
        eid: str | None = None,
    ):
        """Add an inter-layer edge between two layer tuples.

        Parameters
        ----------
        u : str
            Vertex in layer ``aa``.
        layer_a : tuple[str, ...]
            Source layer tuple.
        v : str
            Vertex in layer ``bb``.
        layer_b : tuple[str, ...]
            Target layer tuple.
        weight : float, optional
            Edge weight.
        eid : str | None, optional
            Edge id override.

        Returns
        -------
        str
            Edge id.

        Raises
        ------
        KeyError
            If required vertex–layer presence is missing.
        ValueError
            If layer tuples are invalid.

        Examples
        --------
        ```python
        G.add_inter_edge_nl("u1", ("t1",), "u2", ("t2",))
        ```
        """
        warnings.warn(
            "add_inter_edge_nl() is deprecated; prefer add_edge((u, layer_a), (v, layer_b), ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        aa = layer_a if isinstance(layer_a, tuple) else tuple(layer_a)
        bb = layer_b if isinstance(layer_b, tuple) else tuple(layer_b)
        if not getattr(self, "_fast_mode", False):
            self._validate_layer_tuple(aa)
            self._validate_layer_tuple(bb)
            self._assert_presence(u, aa)
            self._assert_presence(v, bb)
        eid = eid or f"{u}>{v}@{self._layer_tuple_to_lid(aa)}~{self._layer_tuple_to_lid(bb)}"
        # No single slice applies; just register the edge structurally.
        self.add_edge(u, v, weight=weight, edge_id=eid)
        self.set_edge_kivela_role(eid, "inter", (aa, bb))
        return eid

    def add_coupling_edge_nl(
        self,
        u: str,
        layer_a: tuple[str, ...],
        layer_b: tuple[str, ...],
        *,
        weight: float = 1.0,
        eid: str | None = None,
    ):
        """Add a diagonal coupling edge between two layers for the same vertex.

        Parameters
        ----------
        u : str
            Vertex identifier.
        layer_a : tuple[str, ...]
            Layer tuple A.
        layer_b : tuple[str, ...]
            Layer tuple B.
        weight : float, optional
            Edge weight.
        eid : str | None, optional
            Edge id override.

        Returns
        -------
        str
            Edge id.
        """
        warnings.warn(
            "add_coupling_edge_nl() is deprecated; prefer add_edge((u, layer_a), (u, layer_b), ...).",
            DeprecationWarning,
            stacklevel=2,
        )
        eid2 = self.add_inter_edge_nl(u, layer_a, u, layer_b, weight=weight, eid=eid)
        # Re-label as coupling so supra_adjacency treats it as off-diagonal coupling
        aa = tuple(layer_a)
        bb = tuple(layer_b)
        self.set_edge_kivela_role(eid2, "coupling", (aa, bb))
        return eid2

    def layer_vertex_set(self, layer_tuple):
        """Vertices present in a Kivela layer.

        Parameters
        ----------
        layer_tuple : Iterable[str]
            Aspect tuple layer.

        Returns
        -------
        set[str]
        """
        aa = tuple(layer_tuple)
        return {u for (u, L) in self._VM if L == aa}

    def layer_edge_set(
        self,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """Edges associated with a Kivela layer.

        Parameters
        ----------
        layer_tuple : Iterable[str]
            Aspect tuple layer.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_tuple``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_tuple``.

        Returns
        -------
        set[str]
        """
        aa = tuple(layer_tuple)
        E = set()
        for eid, rec in self._edges.items():
            kind = "hyper" if rec.etype == "hyper" else rec.ml_kind
            layers = rec.ml_layers

            if kind in {"intra", "hyper"}:
                if layers == aa:
                    E.add(eid)

            elif kind == "inter" and include_inter:
                # layers expected to be (La, Lb)
                if isinstance(layers, tuple) and len(layers) == 2 and aa in layers:
                    E.add(eid)

            elif kind == "coupling" and include_coupling:
                if isinstance(layers, tuple) and len(layers) == 2 and aa in layers:
                    E.add(eid)

        return E

    ### Legacy inter-layer convenience (string layers) delegates when 1 aspect:

    def add_inter_edge(
        self,
        u: str,
        v: str,
        layer_a: str,
        layer_b: str,
        *,
        weight: float = 1.0,
        eid: str | None = None,
    ):
        """Add an inter-layer edge in legacy single-aspect mode.

        Parameters
        ----------
        u : str
            Source vertex.
        v : str
            Target vertex.
        layer_a : str
            Layer id (single-aspect only).
        layer_b : str
            Layer id (single-aspect only).
        weight : float, optional
            Edge weight.
        eid : str | None, optional
            Edge id override.

        Returns
        -------
        str
            Edge id.
        """
        if len(getattr(self, "aspects", [])) == 1 and getattr(
            self, "_legacy_single_aspect_enabled", True
        ):
            aa, bb = self.layer_id_to_tuple(layer_a), self.layer_id_to_tuple(layer_b)
            return self.add_inter_edge_nl(u, aa, v, bb, weight=weight, eid=eid)
        # Fallback in non-multi-aspect mode
        eid = eid or f"{u}--{v}=={layer_a}~{layer_b}"
        self.add_edge(u, v, weight=weight, edge_id=eid)
        rec = self._edges[eid]
        rec.ml_kind = "inter"
        rec.ml_layers = (layer_a, layer_b)
        return eid

    ## Layer algebra

    def layer_union(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """Union of several Kivela layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        Vs = []
        Es = []
        for aa in layer_tuples:
            Vs.append(self.layer_vertex_set(aa))
            Es.append(
                self.layer_edge_set(
                    aa,
                    include_inter=include_inter,
                    include_coupling=include_coupling,
                )
            )
        if not Vs:
            return {"vertices": set(), "edges": set()}
        V = set().union(*Vs)
        E = set().union(*Es)
        return {"vertices": V, "edges": E}

    def layer_intersection(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """Intersection of several Kivela layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        layer_tuples = list(layer_tuples)
        if not layer_tuples:
            return {"vertices": set(), "edges": set()}

        # start with first layer
        V = self.layer_vertex_set(layer_tuples[0])
        E = self.layer_edge_set(
            layer_tuples[0],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )

        for aa in layer_tuples[1:]:
            V &= self.layer_vertex_set(aa)
            E &= self.layer_edge_set(
                aa,
                include_inter=include_inter,
                include_coupling=include_coupling,
            )

        return {"vertices": V, "edges": E}

    def layer_difference(
        self,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ):
        """Set difference: elements in ``layer_a`` but not in ``layer_b``.

        Parameters
        ----------
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.

        Returns
        -------
        dict
            ``{"vertices": set[str], "edges": set[str]}``.
        """
        Va = self.layer_vertex_set(layer_a)
        Ea = self.layer_edge_set(
            layer_a,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        Vb = self.layer_vertex_set(layer_b)
        Eb = self.layer_edge_set(
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return {
            "vertices": Va - Vb,
            "edges": Ea - Eb,
        }

    ## Layer X Slice

    def create_slice_from_layer(
        self,
        slice_id: str,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """Create a slice induced by a single Kivela layer.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        layer_tuple : Iterable[str]
            Layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_tuple``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_tuple``.
        **attributes
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.

        Examples
        --------
        ```python
        G.create_slice_from_layer("t1_F", ("t1", "F"))
        ```
        """
        result = self.layer_union(
            [layer_tuple],
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer")
        attributes.setdefault("layer_tuple", tuple(layer_tuple))
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_union(
        self,
        slice_id: str,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """Create a slice as the union of several layers.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.
        **attributes
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
        """
        result = self.layer_union(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_union")
        attributes.setdefault("layers", [tuple(a) for a in layer_tuples])
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_intersection(
        self,
        slice_id: str,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """Create a slice as the intersection of several layers.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.
        **attributes
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
        """
        result = self.layer_intersection(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_intersection")
        attributes.setdefault("layers", [tuple(a) for a in layer_tuples])
        return self.create_slice_from_operation(slice_id, result, **attributes)

    def create_slice_from_layer_difference(
        self,
        slice_id: str,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
        **attributes,
    ):
        """Create a slice as the difference of two layers.

        Parameters
        ----------
        slice_id : str
            Slice identifier.
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.
        **attributes
            Slice attributes to store.

        Returns
        -------
        str
            The created slice id.
        """
        result = self.layer_difference(
            layer_a,
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        attributes.setdefault("source", "kivela_layer_difference")
        attributes.setdefault("layer_a", tuple(layer_a))
        attributes.setdefault("layer_b", tuple(layer_b))
        return self.create_slice_from_operation(slice_id, result, **attributes)

    ## Subgraph

    def subgraph_from_layer_tuple(
        self,
        layer_tuple,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> AnnNet:
        """Concrete subgraph induced by a single Kivela layer.

        Parameters
        ----------
        layer_tuple : Iterable[str]
            Layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_tuple``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_tuple``.

        Returns
        -------
        AnnNet
        """
        aa = tuple(layer_tuple)
        V = self.layer_vertex_set(aa)
        E = self.layer_edge_set(
            aa,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=V, edges=E)

    def subgraph_from_layer_union(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> AnnNet:
        """Concrete subgraph induced by the union of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to union.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the union.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the union.

        Returns
        -------
        AnnNet
        """
        res = self.layer_union(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    def subgraph_from_layer_intersection(
        self,
        layer_tuples,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> AnnNet:
        """Concrete subgraph induced by the intersection of several layers.

        Parameters
        ----------
        layer_tuples : Iterable[Iterable[str]]
            Layer tuples to intersect.
        include_inter : bool, optional
            Include inter-layer edges touching any layer in the intersection.
        include_coupling : bool, optional
            Include coupling edges touching any layer in the intersection.

        Returns
        -------
        AnnNet
        """
        res = self.layer_intersection(
            layer_tuples,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    def subgraph_from_layer_difference(
        self,
        layer_a,
        layer_b,
        *,
        include_inter: bool = False,
        include_coupling: bool = False,
    ) -> AnnNet:
        """Concrete subgraph induced by a set-difference of two layers.

        Parameters
        ----------
        layer_a : Iterable[str]
            Minuend layer tuple.
        layer_b : Iterable[str]
            Subtrahend layer tuple.
        include_inter : bool, optional
            Include inter-layer edges touching ``layer_a``.
        include_coupling : bool, optional
            Include coupling edges touching ``layer_a``.

        Returns
        -------
        AnnNet
        """
        res = self.layer_difference(
            layer_a,
            layer_b,
            include_inter=include_inter,
            include_coupling=include_coupling,
        )
        return self.extract_subgraph(vertices=res["vertices"], edges=res["edges"])

    ## helper

    def _assert_presence(self, u: str, aa: tuple[str, ...]):
        if (u, aa) not in self._VM:
            raise KeyError(
                f"presence missing: {(u, aa)} not in V_M; call add_presence(u, aa) first"
            )

    ## Supra_Adjacency

    def supra_adjacency(self, layers: list[str] | None = None):
        """Build the supra adjacency matrix.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers. In single-aspect mode, string ids are accepted.

        Returns
        -------
        scipy.sparse.csr_matrix
            Supra adjacency over the chosen vertex–layer index.

        Examples
        --------
        ```python
        A = G.supra_adjacency()
        ```
        """
        # Map optional legacy 'layers' (strings) to aspect tuples if needed
        if (
            layers is not None
            and len(getattr(self, "aspects", [])) == 1
            and getattr(self, "_legacy_single_aspect_enabled", True)
        ):
            layers_t = [self.layer_id_to_tuple(L) for L in layers]
        else:
            layers_t = None if layers is None else [tuple(L) for L in layers]
        self.ensure_vertex_layer_index(layers_t)

        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)

        # Fill diagonal blocks from intra-layer edges
        for eid, rec in self._edges.items():
            kind = rec.ml_kind
            if kind != "intra":
                continue
            L = rec.ml_layers
            # normalize L to tuple
            if not isinstance(L, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    L = self.layer_id_to_tuple(L)
                else:
                    raise ValueError(
                        "intra edge layer is not a tuple; configure aspects or use add_intra_edge_nl"
                    )
            if layers_t is not None and L not in layers_t:
                continue
            # resolve endpoints (u,v) in layer L
            u, v = rec.src, rec.tgt
            ru = self._nl_to_row.get((u, L))
            rv = self._nl_to_row.get((v, L))
            if ru is None or rv is None:
                continue
            w = rec.weight if rec.weight is not None else 1
            A[ru, rv] = A.get((ru, rv), 0.0) + w
            A[rv, ru] = A.get((rv, ru), 0.0) + w  # undirected; adapt if directed

        # Fill off-diagonal blocks from inter-layer/coupling edges
        for eid, rec in self._edges.items():
            kind = rec.ml_kind
            if kind not in {"inter", "coupling"}:
                continue
            La, Lb = rec.ml_layers
            # normalize La/Lb to tuples
            if not isinstance(La, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    La = self.layer_id_to_tuple(La)
                else:
                    raise ValueError(
                        "inter edge layer_a is not a tuple; configure aspects or use add_inter_edge_nl"
                    )
            if not isinstance(Lb, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    Lb = self.layer_id_to_tuple(Lb)
                else:
                    raise ValueError(
                        "inter edge layer_b is not a tuple; configure aspects or use add_inter_edge_nl"
                    )
            if layers_t is not None and (La not in layers_t or Lb not in layers_t):
                continue
            # endpoints
            u, v = rec.src, rec.tgt
            ru = self._nl_to_row.get((u, La))
            rv = self._nl_to_row.get((v, Lb))
            if ru is None or rv is None:
                continue
            w = rec.weight if rec.weight is not None else 1
            A[ru, rv] = A.get((ru, rv), 0.0) + w
            A[rv, ru] = A.get((rv, ru), 0.0) + w  # undirected; adapt if directed
        return A.tocsr()

    ## Supra_Incidence

    def supra_incidence(
        self,
        layers: list[str] | list[tuple[str, ...]] | None = None,
        include_inter: bool = True,
        include_coupling: bool = True,
    ) -> tuple[sp.csr_matrix, list[str]]:
        """Build the supra-incidence matrix over selected layers.

            Unlike supra_adjacency, this preserves the full hyperedge structure —
            a k-ary hyperedge becomes a single column with k nonzero entries, with
            stoichiometric coefficients intact. Binary intra, inter, coupling, and
            hyperedges are all handled in a unified column-oriented representation.

            Rows  : vertex-layer pairs (u, aa) — identical index to supra_adjacency,
                    built by ensure_vertex_layer_index.
            Cols  : one per selected edge, ordered as: intra edges (per layer, sorted
                    by eid), then inter/coupling edges, then unassigned hyperedges last.

            Column sign convention (matches _matrix):
                - Binary directed   : +w at source row, -w at target row
                - Binary undirected : +w at both rows
                - Hyperedge directed: +w at head rows, -w at tail rows (stoich-aware)
                - Hyperedge undirected: +w at all member rows (stoich-aware)
                - Inter/coupling    : +w at (u, La) row, -w at (v, Lb) row (directed)

            Hyperedges MUST have a layer assignment in edge_layers (set via
            set_edge_kivela_role(eid, "intra", layer_tuple) after add_hyperedge).
            Hyperedges without a layer assignment are collected in the returned
            skipped list and excluded from the matrix — they do NOT silently corrupt
            the result.

            Parameters
            ----------
            layers : list[str] | list[tuple[str, ...]] | None
                Optional subset of layers. None = all layers in V_M.
                Single-aspect string ids are accepted.
            include_inter : bool
                Include inter-layer edges in the output columns. Default True.
            include_coupling : bool
                Include coupling edges in the output columns. Default True.

            Returns
            -------
            B : scipy.sparse.csr_matrix
                Shape (|V_M|, |E_selected|). Rows are vertex-layer pairs in the
                order given by self._row_to_nl after ensure_vertex_layer_index.
            edge_ids : list[str]
                Edge id for each column of B, in column order. Use this to map
                columns back to edges for interpretability.
            skipped : list[str]
                Edge ids that were excluded because their layer assignment could
                not be resolved. Inspect these if B looks sparse.

            Notes
            -----
            The hypergraph random-walk diffusion operator follows directly::

                B_csr = B  (this output)
                D_v = diag(|B| @ ones)          # vertex degree (sum of |entries| per row)
                D_e = diag(|B|.T @ ones)        # edge degree (sum of |entries| per col)
                Theta = D_v_inv @ B @ D_e_inv @ B.T

            Examples
            --------
        ```python
            B, eids, skipped = G.supra_incidence()
        ```
        """
        # 0. Normalise layers argument — identical pattern to supra_adjacency

        if (
            layers is not None
            and len(getattr(self, "aspects", [])) == 1
            and getattr(self, "_legacy_single_aspect_enabled", True)
        ):
            layers_t = [self.layer_id_to_tuple(L) for L in layers]
        else:
            layers_t = None if layers is None else [tuple(L) for L in layers]

        self.ensure_vertex_layer_index(layers_t)
        n_rows = len(self._row_to_nl)

        # 1. Materialise _matrix as CSC (compressed sparse column) for fast column slicing — one call, reused for every hyperedge column.     #
        M_csc = self._matrix.tocsc()

        # 2. Helper: normalise a layer label to a tuple, same guard as supra_adjacency.                                                  #

        def _to_tuple(L):
            if isinstance(L, tuple):
                return L
            if len(getattr(self, "aspects", [])) == 1 and getattr(
                self, "_legacy_single_aspect_enabled", True
            ):
                return self.layer_id_to_tuple(L)
            return None  # unresolvable

        # 3. Collect columns — three passes matching supra_adjacency's logic.
        # We build lists of (row_indices, values) per column, then assemble into a single COO (coordinate format) matrix at the end.
        col_data: list[tuple[list[int], list[float]]] = []
        edge_ids: list[str] = []
        skipped: list[str] = []

        default_dir = True if self.directed is None else self.directed
        for eid, rec in self._edges.items():
            kind = "hyper" if rec.etype == "hyper" else rec.ml_kind
            # 3a. INTRA binary edge

            if kind == "intra":
                raw_L = rec.ml_layers
                if raw_L is None:
                    skipped.append(eid)
                    continue
                L = _to_tuple(raw_L)
                if L is None:
                    skipped.append(eid)
                    continue
                if layers_t is not None and L not in layers_t:
                    continue

                u, v = rec.src, rec.tgt
                if u is None or v is None:
                    skipped.append(eid)
                    continue

                ru = self._nl_to_row.get((u, L))
                rv = self._nl_to_row.get((v, L))
                if ru is None or rv is None:
                    skipped.append(eid)
                    continue

                w = float(rec.weight if rec.weight is not None else 1.0)
                is_dir = rec.directed if rec.directed is not None else default_dir

                if is_dir:
                    rows_out = [ru, rv]
                    vals_out = [w, -w]
                else:
                    rows_out = [ru, rv]
                    vals_out = [w, w]

                col_data.append((rows_out, vals_out))
                edge_ids.append(eid)

            # 3b. HYPER edge — read stoichiometry directly from _matrix

            elif kind == "hyper":
                raw_L = rec.ml_layers
                if raw_L is None:
                    # No layer assignment — skip and report
                    skipped.append(eid)
                    continue
                L = _to_tuple(raw_L)
                if L is None:
                    skipped.append(eid)
                    continue
                if layers_t is not None and L not in layers_t:
                    continue

                col_idx = rec.col_idx
                if col_idx < 0:
                    skipped.append(eid)
                    continue

                # Extract the full column from _matrix (sparse, so only nonzero entries — includes exact stoichiometric coefficients if set_edge_coeffs was called).
                col_vec = M_csc.getcol(col_idx)
                nz_rows_in_flat, _ = col_vec.nonzero()

                rows_out = []
                vals_out = []
                for flat_row in nz_rows_in_flat:
                    entity_key = self._row_to_entity.get(flat_row)
                    entity_id = entity_key[0] if isinstance(entity_key, tuple) else entity_key
                    if entity_id is None:
                        continue
                    supra_row = self._nl_to_row.get((entity_id, L))
                    if supra_row is None:
                        # Entity exists in _matrix but not in V_M for this layer — presence was never declared. Skip this entry rather than silently placing it in the wrong row.
                        continue
                    coeff = float(col_vec[flat_row, 0])
                    rows_out.append(supra_row)
                    vals_out.append(coeff)

                if not rows_out:
                    # Hyperedge has a layer but no members landed in V_M — almost certainly a missing add_presence call.
                    skipped.append(eid)
                    continue

                col_data.append((rows_out, vals_out))
                edge_ids.append(eid)

            # 3c. INTER / COUPLING edges

            elif kind in {"inter", "coupling"}:
                if kind == "inter" and not include_inter:
                    continue
                if kind == "coupling" and not include_coupling:
                    continue

                raw_layers = rec.ml_layers
                if raw_layers is None:
                    skipped.append(eid)
                    continue
                La_raw, Lb_raw = raw_layers
                La = _to_tuple(La_raw)
                Lb = _to_tuple(Lb_raw)
                if La is None or Lb is None:
                    skipped.append(eid)
                    continue
                if layers_t is not None and (La not in layers_t or Lb not in layers_t):
                    continue

                u, v = rec.src, rec.tgt
                if u is None or v is None:
                    skipped.append(eid)
                    continue

                ru = self._nl_to_row.get((u, La))
                rv = self._nl_to_row.get((v, Lb))
                if ru is None or rv is None:
                    skipped.append(eid)
                    continue

                w = float(rec.weight if rec.weight is not None else 1.0)
                is_dir = rec.directed if rec.directed is not None else default_dir

                if is_dir:
                    rows_out = [ru, rv]
                    vals_out = [w, -w]
                else:
                    rows_out = [ru, rv]
                    vals_out = [w, w]

                col_data.append((rows_out, vals_out))
                edge_ids.append(eid)

            # kind == "hyper" already handled; any other unknown kind: skip
            else:
                continue

        # 4. Assemble COO matrix then convert to CSR

        n_cols = len(col_data)

        if n_cols == 0:
            B = sp.csr_matrix((n_rows, 0), dtype=float)
            return B, edge_ids, skipped

        all_rows: list[int] = []
        all_cols: list[int] = []
        all_vals: list[float] = []

        for col_j, (rows_j, vals_j) in enumerate(col_data):
            all_rows.extend(rows_j)
            all_cols.extend([col_j] * len(rows_j))
            all_vals.extend(vals_j)

        B = sp.coo_matrix(
            (all_vals, (all_rows, all_cols)),
            shape=(n_rows, n_cols),
            dtype=float,
        ).tocsr()

        return B, edge_ids, skipped

    ##  Block partitions & Laplacians

    def _normalize_layers_arg(self, layers):
        """Normalize ``layers`` argument to aspect tuples or None."""
        if layers is None:
            return None
        if len(getattr(self, "aspects", [])) == 1 and getattr(
            self, "_legacy_single_aspect_enabled", True
        ):
            return [self.layer_id_to_tuple(L) for L in layers]
        return [tuple(L) for L in layers]

    def _build_block(self, include_kinds: set[str], layers: list[str] | list[tuple] | None = None):
        """Internal builder for supra block matrices."""

        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)

        # Intra-layer edges (diagonal blocks)
        if "intra" in include_kinds:
            for eid, rec in self._edges.items():
                kind = rec.ml_kind
                if kind != "intra":
                    continue
                L = rec.ml_layers
                if not isinstance(L, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(
                        self, "_legacy_single_aspect_enabled", True
                    ):
                        L = self.layer_id_to_tuple(L)
                    else:
                        continue
                if layers_t is not None and L not in layers_t:
                    continue
                u, v = rec.src, rec.tgt
                ru = self._nl_to_row.get((u, L))
                rv = self._nl_to_row.get((v, L))
                if ru is None or rv is None:
                    continue
                w = rec.weight if rec.weight is not None else 1.0
                A[ru, rv] = A.get((ru, rv), 0.0) + w
                A[rv, ru] = A.get((rv, ru), 0.0) + w

        # Inter/coupling edges (off-diagonal blocks)
        if include_kinds & {"inter", "coupling"}:
            for eid, rec in self._edges.items():
                kind = rec.ml_kind
                if kind not in include_kinds:
                    continue
                La, Lb = rec.ml_layers
                if not isinstance(La, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(
                        self, "_legacy_single_aspect_enabled", True
                    ):
                        La = self.layer_id_to_tuple(La)
                    else:
                        continue
                if not isinstance(Lb, tuple):
                    if len(getattr(self, "aspects", [])) == 1 and getattr(
                        self, "_legacy_single_aspect_enabled", True
                    ):
                        Lb = self.layer_id_to_tuple(Lb)
                    else:
                        continue
                if layers_t is not None and (La not in layers_t or Lb not in layers_t):
                    continue
                u, v = rec.src, rec.tgt
                ru = self._nl_to_row.get((u, La))
                rv = self._nl_to_row.get((v, Lb))
                if ru is None or rv is None:
                    continue
                w = rec.weight if rec.weight is not None else 1.0
                A[ru, rv] = A.get((ru, rv), 0.0) + w
                A[rv, ru] = A.get((rv, ru), 0.0) + w

        return A.tocsr()

    def build_intra_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only intra-layer edges (diagonal blocks).

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self._build_block({"intra"}, layers)

    def build_inter_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only inter-layer (non-diagonal) edges.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self._build_block({"inter"}, layers)

    def build_coupling_block(self, layers: list[str] | list[tuple] | None = None):
        """Supra matrix containing only coupling edges.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """
        return self._build_block({"coupling"}, layers)

    def supra_degree(self, layers: list[str] | list[tuple] | None = None):
        """Degree vector over the supra-graph.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        numpy.ndarray
        """

        A = self.supra_adjacency(layers)
        # sum over columns per row -> shape (n,1); flatten to 1D
        deg = np.asarray(A.sum(axis=1)).ravel()
        return deg

    def supra_laplacian(self, kind: str = "comb", layers: list[str] | list[tuple] | None = None):
        """Build supra-Laplacian.

        Parameters
        ----------
        kind : str, optional
            ``"comb"`` for combinatorial ``L = D - A`` or ``"norm"`` for
            normalized ``L = I - D^{-1/2} A D^{-1/2}``.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """

        A = self.supra_adjacency(layers)
        n = A.shape[0]
        deg = self.supra_degree(layers)
        if kind == "comb":
            D = sp.diags(deg, format="csr")
            return D - A
        elif kind == "norm":
            # D^{-1/2}; zero where deg==0
            invsqrt = np.zeros_like(deg, dtype=float)
            nz = deg > 0
            invsqrt[nz] = 1.0 / np.sqrt(deg[nz])
            Dm12 = sp.diags(invsqrt, format="csr")
            I = sp.eye(n, format="csr")
            return I - (Dm12 @ A @ Dm12)
        else:
            raise ValueError("kind must be 'comb' or 'norm'")

    ## Coupling generators (vertex-independent)

    def _aspect_index(self, aspect: str) -> int:
        if aspect not in self.aspects:
            raise KeyError(f"unknown aspect {aspect!r}; known: {self.aspects!r}")
        return self.aspects.index(aspect)

    def _layer_matches_filter(self, aa: tuple[str, ...], layer_filter: dict[str, set]) -> bool:
        """
        layer_filter: {aspect_name: {elem1, elem2, ...}}; a layer matches if aa[a] ∈ set for all keys.
        """
        if not layer_filter:
            return True
        for a_name, allowed in layer_filter.items():
            i = self._aspect_index(a_name)
            if aa[i] not in allowed:
                return False
        return True

    def add_layer_coupling_pairs(
        self, layer_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]], *, weight: float = 1.0
    ) -> int:
        """Add diagonal couplings for explicit layer pairs.

        Parameters
        ----------
        layer_pairs : list[tuple[tuple[str, ...], tuple[str, ...]]]
            Layer tuple pairs ``(aa, bb)``.
        weight : float, optional
            Edge weight.

        Returns
        -------
        int
            Number of edges added.
        """
        added = 0
        # normalize to tuples, validate once
        norm_pairs = []
        for La, Lb in layer_pairs:
            La = tuple(La)
            Lb = tuple(Lb)
            self._validate_layer_tuple(La)
            self._validate_layer_tuple(Lb)
            norm_pairs.append((La, Lb))
        # Build per-layer presence index to avoid O(|V_M|^2)
        layer_to_vertices = {}
        for u, aa in self._VM:
            layer_to_vertices.setdefault(aa, set()).add(u)
        for La, Lb in norm_pairs:
            Ua = layer_to_vertices.get(La, set())
            Ub = layer_to_vertices.get(Lb, set())
            for u in Ua & Ub:
                self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                added += 1
        return added

    def add_categorical_coupling(
        self, aspect: str, groups: list[list[str]], *, weight: float = 1.0
    ) -> int:
        """Add categorical couplings along one aspect.

        Parameters
        ----------
        aspect : str
            Aspect name to couple over.
        groups : list[list[str]]
            Groups of elementary labels to fully connect per vertex.
        weight : float, optional
            Edge weight.

        Returns
        -------
        int
            Number of edges added.
        """
        ai = self._aspect_index(aspect)
        added = 0
        # Map: (u, other_aspects_tuple) -> {elem_on_aspect: full_layer_tuple}
        buckets = {}
        for u, aa in self._VM:
            other = aa[:ai] + aa[ai + 1 :]
            buckets.setdefault((u, other), {}).setdefault(aa[ai], aa)
        for grp in groups:
            gset = set(grp)
            for (u, other), mapping in buckets.items():
                # pick only those aa whose aspect element is in this group
                layers = [mapping[e] for e in mapping.keys() if e in gset]
                if len(layers) < 2:
                    continue
                for La, Lb in itertools.combinations(sorted(layers), 2):
                    self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                    added += 1
        return added

    def add_diagonal_coupling_filter(
        self, layer_filter: dict[str, set], *, weight: float = 1.0
    ) -> int:
        """Add diagonal couplings within a filtered layer subspace.

        Parameters
        ----------
        layer_filter : dict[str, set]
            Aspect filters (e.g., ``{"time": {"t1","t2"}}``).
        weight : float, optional
            Edge weight.

        Returns
        -------
        int
            Number of edges added.
        """
        added = 0
        # collect per vertex the matching layers actually present
        per_u = {}
        for u, aa in self._VM:
            if self._layer_matches_filter(aa, layer_filter):
                per_u.setdefault(u, []).append(aa)
        for u, layers in per_u.items():
            if len(layers) < 2:
                continue
            for La, Lb in itertools.combinations(sorted(layers), 2):
                self.add_coupling_edge_nl(u, La, Lb, weight=weight)
                added += 1
        return added

    ## Tensor view & flattening map

    def tensor_index(self, layers: list[str] | list[tuple] | None = None):
        """Build indices for tensor view.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        tuple
            ``(vertices, layers_t, vertex_to_i, layer_to_i)``.

        Examples
        --------
        ```python
        vertices, layers_t, v2i, l2i = G.tensor_index()
        ```
        """
        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        # collect in the order they appear in the vertex–layer index: (vertex major, then layer)
        vertices = []
        layers_list = []
        seen_vertices = set()
        seen_layers = set()
        for u, aa in self._row_to_nl:
            if u not in seen_vertices:
                vertices.append(u)
                seen_vertices.add(u)
            if aa not in seen_layers:
                layers_list.append(aa)
                seen_layers.add(aa)
        vertex_to_i = {u: i for i, u in enumerate(vertices)}
        layer_to_i = {aa: i for i, aa in enumerate(layers_list)}
        return vertices, layers_list, vertex_to_i, layer_to_i

    def adjacency_tensor_view(self, layers: list[str] | list[tuple] | None = None):
        """Sparse 4-index adjacency view.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        dict
            ``{"vertices","layers","vertex_to_i","layer_to_i","ui","ai","vi","bi","w"}``.

        Notes
        -----
        Symmetric entries are emitted twice: ``(ui, ai, vi, bi)`` and ``(vi, bi, ui, ai)``.
        """

        vertices, layers_t, vertex_to_i, layer_to_i = self.tensor_index(layers)
        ui = []
        ai = []
        vi = []
        bi = []
        wv = []

        # Intra edges -> (u,aa)↔(v,aa)
        for eid, rec in self._edges.items():
            kind = rec.ml_kind
            if kind != "intra":
                continue
            L = rec.ml_layers
            if not isinstance(L, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    L = self.layer_id_to_tuple(L)
                else:
                    continue
            if layers is not None and L not in set(layers_t):
                continue
            u, v = rec.src, rec.tgt
            if (u, L) not in self._nl_to_row or (v, L) not in self._nl_to_row:
                continue
            w = rec.weight if rec.weight is not None else 1.0
            ui.extend((vertex_to_i[u], vertex_to_i[v]))
            vi.extend((vertex_to_i[v], vertex_to_i[u]))
            a = layer_to_i[L]
            ai.extend((a, a))
            bi.extend((a, a))
            wv.extend((w, w))

        # Inter / coupling -> (u,aa)↔(v,bb)
        for eid, rec in self._edges.items():
            kind = rec.ml_kind
            if kind not in {"inter", "coupling"}:
                continue
            La, Lb = rec.ml_layers
            if not isinstance(La, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    La = self.layer_id_to_tuple(La)
                else:
                    continue
            if not isinstance(Lb, tuple):
                if len(getattr(self, "aspects", [])) == 1 and getattr(
                    self, "_legacy_single_aspect_enabled", True
                ):
                    Lb = self.layer_id_to_tuple(Lb)
                else:
                    continue
            if layers is not None:
                S = set(layers_t)
                if La not in S or Lb not in S:
                    continue
            u, v = rec.src, rec.tgt
            if (u, La) not in self._nl_to_row or (v, Lb) not in self._nl_to_row:
                continue
            w = rec.weight if rec.weight is not None else 1.0
            ui.extend((vertex_to_i[u], vertex_to_i[v]))
            vi.extend((vertex_to_i[v], vertex_to_i[u]))
            ai.extend((layer_to_i[La], layer_to_i[Lb]))
            bi.extend((layer_to_i[Lb], layer_to_i[La]))
            wv.extend((w, w))

        return {
            "vertices": vertices,
            "layers": layers_t,
            "vertex_to_i": vertex_to_i,
            "layer_to_i": layer_to_i,
            "ui": np.asarray(ui, dtype=int),
            "ai": np.asarray(ai, dtype=int),
            "vi": np.asarray(vi, dtype=int),
            "bi": np.asarray(bi, dtype=int),
            "w": np.asarray(wv, dtype=float),
        }

    def flatten_to_supra(self, tensor_view: dict):
        """Flatten a tensor view into a supra adjacency matrix.

        Parameters
        ----------
        tensor_view : dict
            Output of :meth:`adjacency_tensor_view` or :meth:`unflatten_from_supra`.

        Returns
        -------
        scipy.sparse.csr_matrix
        """

        # Ensure index reflects current layers subset (tensor_view["layers"])
        layers_t = tensor_view["layers"] if tensor_view.get("layers", None) else None
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        A = sp.dok_matrix((n, n), dtype=float)
        vertices = tensor_view["vertices"]
        layers = tensor_view["layers"]
        ui, ai, vi, bi, w = (
            tensor_view["ui"],
            tensor_view["ai"],
            tensor_view["vi"],
            tensor_view["bi"],
            tensor_view["w"],
        )
        # Map back from indices to (u,aa) rows using current _nl_to_row
        for k in range(len(w)):
            u = vertices[int(ui[k])]
            aa = layers[int(ai[k])]
            v = vertices[int(vi[k])]
            bb = layers[int(bi[k])]
            ru = self._nl_to_row.get((u, aa))
            rv = self._nl_to_row.get((v, bb))
            if ru is None or rv is None:
                continue
            A[ru, rv] = A.get((ru, rv), 0.0) + float(w[k])
        return A.tocsr()

    def unflatten_from_supra(self, A, layers: list[str] | list[tuple] | None = None):
        """Unflatten a supra adjacency matrix into a tensor view.

        Parameters
        ----------
        A : scipy.sparse.spmatrix
            Supra adjacency matrix.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        dict
            Tensor view with the same schema as :meth:`adjacency_tensor_view`.
        """

        A = A.tocsr()
        vertices, layers_t, vertex_to_i, layer_to_i = self.tensor_index(layers)
        rows, cols = A.nonzero()
        data = A.data
        ui = np.empty_like(rows)
        vi = np.empty_like(cols)
        ai = np.empty_like(rows)
        bi = np.empty_like(cols)
        for k in range(len(rows)):
            (u, aa) = self._row_to_nl[int(rows[k])]
            (v, bb) = self._row_to_nl[int(cols[k])]
            ui[k] = vertex_to_i[u]
            vi[k] = vertex_to_i[v]
            ai[k] = layer_to_i[aa]
            bi[k] = layer_to_i[bb]
        return {
            "vertices": vertices,
            "layers": layers_t,
            "vertex_to_i": vertex_to_i,
            "layer_to_i": layer_to_i,
            "ui": ui,
            "ai": ai,
            "vi": vi,
            "bi": bi,
            "w": data.astype(float, copy=False),
        }

    ##  Dynamics & spectral probes

    def supra_adjacency_scaled(
        self,
        *,
        coupling_scale: float = 1.0,
        include_inter: bool = True,
        layers: list[str] | list[tuple] | None = None,
    ):
        """Build scaled supra adjacency.

        Parameters
        ----------
        coupling_scale : float, optional
            Scaling factor for coupling edges.
        include_inter : bool, optional
            Whether to include inter-layer edges.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """

        A_intra = self.build_intra_block(layers)
        A_coup = self.build_coupling_block(layers)
        A_inter = self.build_inter_block(layers) if include_inter else None
        A = A_intra.copy()
        if include_inter:
            A = A + A_inter
        if coupling_scale != 1.0:
            A = A + (A_coup * (coupling_scale - 1.0))
        else:
            A = A + A_coup
        return A.tocsr()

    def transition_matrix(self, layers: list[str] | list[tuple] | None = None):
        """Row-stochastic transition matrix ``P = D^{-1} A``.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        scipy.sparse.csr_matrix
        """

        A = self.supra_adjacency(layers).tocsr()
        deg = self.supra_degree(layers)
        invdeg = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        invdeg[nz] = 1.0 / deg[nz]
        Dinv = sp.diags(invdeg, format="csr")
        return Dinv @ A

    def random_walk_step(self, p, layers: list[str] | list[tuple] | None = None):
        """One random-walk step ``p' = p P``.

        Parameters
        ----------
        p : array-like
            Row vector of length ``|V_M|``.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        numpy.ndarray
        """

        P = self.transition_matrix(layers)
        p = np.asarray(p, dtype=float).reshape(1, -1)
        if p.shape[1] != P.shape[0]:
            raise ValueError(f"p has length {p.shape[1]} but supra has size {P.shape[0]}")
        return (p @ P).ravel()

    def diffusion_step(
        self, x, tau: float = 1.0, kind: str = "comb", layers: list[str] | list[tuple] | None = None
    ):
        """One explicit Euler step of diffusion on the supra-graph.

        Parameters
        ----------
        x : array-like
            State vector of length ``|V_M|``.
        tau : float, optional
            Time step.
        kind : str, optional
            ``"comb"`` or ``"norm"``.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        numpy.ndarray
        """

        L = self.supra_laplacian(kind=kind, layers=layers)
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.shape[0] != L.shape[0]:
            raise ValueError(f"x has length {x.shape[0]} but supra has size {L.shape[0]}")
        return x - tau * (L @ x)

    def algebraic_connectivity(self, layers: list[str] | list[tuple] | None = None):
        """Algebraic connectivity of the supra-graph.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        tuple[float, numpy.ndarray | None]
            ``(lambda_2, fiedler_vector)`` or ``(0.0, None)`` if too small.
        """

        from scipy.sparse.linalg import eigsh

        L = self.supra_laplacian(kind="comb", layers=layers).astype(float)
        n = L.shape[0]
        if n < 2:
            return 0.0, None
        # Compute k=2 smallest eigenvalues of symmetric PSD L
        vals, vecs = eigsh(L, k=2, which="SM", return_eigenvectors=True)
        # Sort just in case
        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        # lambda_0 ~ 0 (within numerical eps); lambda_1 is algebraic connectivity
        return float(vals[1]), vecs[:, 1]

    def k_smallest_laplacian_eigs(
        self, k: int = 6, kind: str = "comb", layers: list[str] | list[tuple] | None = None
    ):
        """Return k smallest eigenvalues/eigenvectors of the supra-Laplacian.

        Parameters
        ----------
        k : int, optional
            Number of eigenpairs to compute.
        kind : str, optional
            ``"comb"`` or ``"norm"``.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            ``(eigenvalues, eigenvectors)``.
        """

        from scipy.sparse.linalg import eigsh

        if k < 1:
            raise ValueError("k must be >= 1")
        L = self.supra_laplacian(kind=kind, layers=layers).astype(float)
        k = min(k, max(1, L.shape[0] - 1))
        vals, vecs = eigsh(L, k=k, which="SM", return_eigenvectors=True)
        order = np.argsort(vals)
        return vals[order], vecs[:, order]

    def dominant_rw_eigenpair(self, layers: list[str] | list[tuple] | None = None):
        """Dominant eigenpair of the random-walk operator.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        tuple[float, numpy.ndarray | None]
            ``(lambda_max, v)``.
        """

        from scipy.sparse.linalg import eigsh

        P = self.transition_matrix(layers).tocsr().astype(float)
        n = P.shape[0]
        if n == 0:
            return 0.0, None
        # Symmetrize for stable eigensolve; still informative about spectral radius.
        S = (P + P.T) * 0.5
        vals, vecs = eigsh(S, k=1, which="LA")
        return float(vals[0]), vecs[:, 0]

    def sweep_coupling_regime(
        self, scales, metric="algebraic_connectivity", layers: list[str] | list[tuple] | None = None
    ):
        """Scan coupling scales and evaluate a metric.

        Parameters
        ----------
        scales : Iterable[float]
            Coupling scales to evaluate.
        metric : str | callable, optional
            ``"algebraic_connectivity"`` or a callable ``metric(A)->float``.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        list[float]
            Metric values aligned with ``scales``.
        """
        results = []
        if isinstance(metric, str):
            metric = metric.strip().lower()
        for ω in scales:
            Aω = self.supra_adjacency_scaled(
                coupling_scale=float(ω), include_inter=True, layers=layers
            )
            if metric == "algebraic_connectivity":
                # Compute λ2 of L = D - Aω

                from scipy.sparse import diags

                deg = Aω.sum(axis=1).A.ravel()
                L = diags(deg) - Aω
                from scipy.sparse.linalg import eigsh

                if L.shape[0] < 2:
                    results.append(0.0)
                    continue
                vals, _ = eigsh(L.astype(float), k=2, which="SM")
                vals.sort()
                results.append(float(vals[1]))
            elif callable(metric):
                results.append(float(metric(Aω)))
            else:
                raise ValueError(
                    "Unknown metric; use 'algebraic_connectivity' or provide a callable(A)->float)"
                )
        return results

    ## Layer-aware descriptors

    def _rows_for_layer(self, L):
        """Return row indices in the supra index that belong to aspect-tuple layer L."""
        if not isinstance(L, tuple):
            if len(getattr(self, "aspects", [])) == 1 and getattr(
                self, "_legacy_single_aspect_enabled", True
            ):
                L = self.layer_id_to_tuple(L)
            else:
                raise ValueError("Layer id must be an aspect tuple")
        rows = []
        for i, (u, aa) in enumerate(self._row_to_nl):
            if aa == L:
                rows.append(i)
        return rows

    def layer_degree_vectors(self, layers: list[str] | list[tuple] | None = None):
        """Per-layer degree vectors (intra-layer only).

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        dict
            ``{layer_tuple: (rows_idx_list, deg_vector_np)}``.
        """

        A_intra = self.build_intra_block(layers).tocsr()
        out = {}
        # choose layers subset from current index
        chosen_layers = self._normalize_layers_arg(layers)
        if chosen_layers is None:
            # infer from current index
            chosen_layers = []
            seen = set()
            for _, aa in self._row_to_nl:
                if aa not in seen:
                    chosen_layers.append(aa)
                    seen.add(aa)
        for L in chosen_layers:
            rows = self._rows_for_layer(L)
            if not rows:
                continue
            sub = A_intra[rows][:, rows]
            deg = np.asarray(sub.sum(axis=1)).ravel()
            out[L] = (rows, deg)
        return out

    def participation_coefficient(self, layers: list[str] | list[tuple] | None = None):
        """Participation coefficient per vertex.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        dict[str, float]
        """

        # build per-layer deg vectors and aggregate per vertex
        layer_deg = self.layer_degree_vectors(layers)
        # aggregate k_u over layers
        per_vertex_total = {}
        per_vertex_by_layer = {}
        for L, (rows, deg) in layer_deg.items():
            for i, r in enumerate(rows):
                u, _ = self._row_to_nl[r]
                per_vertex_total[u] = per_vertex_total.get(u, 0.0) + float(deg[i])
                per_vertex_by_layer.setdefault(u, {})[L] = float(deg[i])
        P = {}
        for u, k in per_vertex_total.items():
            if k <= 0:
                P[u] = 0.0
                continue
            s = 0.0
            for L, kL in per_vertex_by_layer[u].items():
                x = kL / k
                s += x * x
            P[u] = 1.0 - s
        return P

    def versatility(self, layers: list[str] | list[tuple] | None = None):
        """Versatility proxy based on dominant eigenvector of supra adjacency.

        Parameters
        ----------
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers.

        Returns
        -------
        dict[str, float]
        """

        from scipy.sparse.linalg import eigsh

        A = self.supra_adjacency(layers).astype(float)
        n = A.shape[0]
        if n == 0:
            return {}
        # largest eigenpair of symmetric A
        vals, vecs = eigsh(A, k=1, which="LA")
        v = vecs[:, 0]
        per_vertex = {}
        for i, (u, _) in enumerate(self._row_to_nl):
            per_vertex[u] = per_vertex.get(u, 0.0) + float(abs(v[i]))
        # normalize
        m = max(per_vertex.values()) if per_vertex else 1.0
        if m > 0:
            for u in per_vertex:
                per_vertex[u] /= m
        return per_vertex

    ## Multislice modularity (scorer)

    def multislice_modularity(
        self,
        partition,
        *,
        gamma: float = 1.0,
        omega: float = 1.0,
        include_inter: bool = False,
        layers: list[str] | list[tuple] | None = None,
    ):
        """Mucha et al. multislice modularity (scorer only).

        Parameters
        ----------
        partition : array-like
            Community ids, length ``|V_M|`` in the current index.
        gamma : float, optional
            Resolution parameter.
        omega : float, optional
            Coupling strength (binary coupling structure scaled by ``omega``).
        include_inter : bool, optional
            Whether to include inter-layer (non-diagonal) edges.
        layers : list[str] | list[tuple[str, ...]] | None, optional
            Optional subset of layers to score on.

        Returns
        -------
        float
            Modularity score ``Q``.

        Examples
        --------
        ```python
        Q = G.multislice_modularity(partition)
        ```
        """

        # Ensure index over the right layers
        layers_t = self._normalize_layers_arg(layers)
        self.ensure_vertex_layer_index(layers_t)
        n = len(self._row_to_nl)
        part = np.asarray(partition)
        if part.shape[0] != n:
            raise ValueError(f"partition length {part.shape[0]} != |V_M| {n}")
        # Build A = A_intra + (include_inter ? A_inter : 0) + omega * (binary coupling structure)
        A_intra = self.build_intra_block(layers_t).tocsr()
        A_coup = self.build_coupling_block(layers_t).tocsr()
        # binarize coupling structure then scale by omega
        if A_coup.nnz:
            A_coup = (A_coup > 0).astype(float) * float(omega)
        A = A_intra.copy()
        if include_inter:
            A = A + self.build_inter_block(layers_t).tocsr()
        A = A + A_coup
        # 2μ = total edge weight in supra (sum of all entries)
        two_mu = float(A.sum())
        if two_mu <= 0:
            return 0.0
        # Build expected term (configuration null model) per layer block
        B = A.tolil()
        layer_deg = self.layer_degree_vectors(layers_t)  # degrees from intra only
        # map row -> layer to avoid repeated lookups
        row_layer = [aa for (_, aa) in self._row_to_nl]
        # For each layer L: subtract gamma * (k_i^L k_j^L)/(2 m_L) within its block
        for L, (rows, deg) in layer_deg.items():
            mL2 = float(deg.sum())  # = 2 m_L
            if mL2 <= 0:
                continue
            # outer product deg_i deg_j / (2 m_L)
            # Use sparse rank-1: for each i in rows, for each j in rows
            # But we only need to subtract for pairs that share a community later; to keep simple and general,
            # build a dense block if small, else sparse updates.
            if len(rows) <= 2048:
                expected = gamma * (np.outer(deg, deg) / mL2)
                # subtract into B
                for ii, ri in enumerate(rows):
                    # vectorized subtract on row
                    cols = rows
                    B[ri, cols] = (B[ri, cols]).toarray().ravel() - expected[ii, :]
            else:
                # large layer: fall back to sparse updates
                scale = gamma / mL2
                for a, ri in enumerate(rows):
                    if deg[a] == 0:
                        continue
                    valrow = (deg[a] * deg) * scale  # numpy broadcasting
                    B[ri, rows] = (B[ri, rows]).toarray().ravel() - valrow
        B = B.tocsr()
        # Q = (1 / (2μ)) * sum_{i,j} B_{ij} δ(g_i, g_j)
        # Compute by iterating over nonzeros in B and summing those within same community
        rows, cols = B.nonzero()
        data = B.data
        mask = part[rows] == part[cols]
        Q = float(data[mask].sum()) / two_mu
        return Q
