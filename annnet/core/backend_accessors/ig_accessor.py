from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from ._base import _BackendAccessorBase

if TYPE_CHECKING:
    from ..graph import AnnNet


class _IGBackendAccessor(_BackendAccessorBase):
    """igraph backend accessor attached to an AnnNet instance."""

    VERTEX_KEYS = {
        "source",
        "target",
        "u",
        "v",
        "vertex",
        "vertices",
        "vs",
        "to",
        "fr",
        "root",
        "roots",
        "neighbors",
        "nbunch",
        "path",
        "cut",
    }

    def __init__(self, owner: AnnNet):
        self._G = owner
        self._cache = {}
        self.cache_enabled = True

    def clear(self):
        self._cache.clear()

    def peek_vertices(self, k: int = 10):
        igG = self._get_or_make_ig(
            directed=True,
            hyperedge_mode="skip",
            slice=None,
            slices=None,
            needed_attrs=set(),
            simple=True,
            edge_aggs=None,
        )
        names = igG.vs["name"] if "name" in igG.vs.attributes() else None
        return [names[i] if names else i for i in range(min(max(0, int(k)), igG.vcount()))]

    def backend(
        self,
        *,
        directed: bool = True,
        hyperedge_mode: str = "skip",
        slice=None,
        slices=None,
        needed_attrs=None,
        simple: bool = False,
        edge_aggs: dict | None = None,
    ):
        return self._get_or_make_ig(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            needed_attrs=needed_attrs or set(),
            simple=simple,
            edge_aggs=edge_aggs,
        )

    def __getattr__(self, name: str):
        def wrapper(*args, **kwargs):
            import igraph as _ig

            directed = bool(kwargs.pop("_ig_directed", True))
            hyperedge_mode = kwargs.pop("_ig_hyperedge", "skip")
            slice = kwargs.pop("_ig_slice", None)
            slices = kwargs.pop("_ig_slices", None)
            label_field = kwargs.pop("_ig_label_field", None)
            guess_labels = kwargs.pop("_ig_guess_labels", True)
            simple = bool(kwargs.pop("_ig_simple", False))
            edge_aggs = kwargs.pop("_ig_edge_aggs", None)

            needed_edge_attrs = self._needed_edge_attrs_for_ig(name, kwargs)

            if str(hyperedge_mode).lower() == "reify":
                import warnings

                warnings.warn(
                    "igraph backend accessor does not support hyperedge_mode='reify'; falling back to 'skip'.",
                    category=RuntimeWarning,
                    stacklevel=3,
                )

            igG = self._get_or_make_ig(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_edge_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

            args = list(args)
            for idx, value in enumerate(args):
                if value is self._G:
                    args[idx] = igG
            for key, value in list(kwargs.items()):
                if value is self._G:
                    kwargs[key] = igG

            target = getattr(igG, name, None)
            if not callable(target):
                target = getattr(_ig, name, None)
            if not callable(target):
                raise AttributeError(f"igraph has no callable '{name}'")

            try:
                sig = inspect.signature(target)
                bound = sig.bind_partial(*args, **kwargs)
            except Exception:
                bound = None

            try:
                if label_field is None and guess_labels:
                    label_field = self._infer_label_field()
                if bound is not None:
                    self._coerce_vertices_in_bound(bound, igG, label_field)
                    bound.apply_defaults()
                    pargs, pkwargs = list(bound.args), dict(bound.kwargs)
                else:
                    self._coerce_vertices_in_kwargs(kwargs, igG, label_field)
                    pargs, pkwargs = list(args), dict(kwargs)
            except Exception:
                pargs, pkwargs = list(args), dict(kwargs)

            try:
                raw = target(*pargs, **pkwargs)
                return raw
            except (KeyError, ValueError) as exc:
                sample = self.peek_vertices(5)
                tip = (
                    f"{exc}. Vertices must match this graph's vertex IDs.\n"
                    f"- If you passed labels, set _ig_label_field=<vertex label column>.\n"
                    f"- Example: G.ig.distances(source='a', target='z', weights='weight', _ig_label_field='name')\n"
                    f"- A few vertex IDs igraph sees: {sample}"
                )
                raise type(exc)(tip) from exc

        return wrapper

    def __dir__(self):
        import igraph as _ig

        graph_obj = getattr(_ig, "Graph", None)
        return sorted(set(super().__dir__()) | self._callable_names(_ig, graph_obj))

    def _needed_edge_attrs_for_ig(self, func_name: str, kwargs: dict) -> set:
        needed = set()
        weight_name = kwargs.get("weights", kwargs.get("weight", None))
        if weight_name is not None:
            needed.add(str(weight_name))
        if "capacity" in kwargs and kwargs["capacity"] is not None:
            needed.add(str(kwargs["capacity"]))
        return needed

    def _convert_to_ig(
        self,
        *,
        directed: bool,
        hyperedge_mode: str,
        slice,
        slices,
        needed_attrs: set,
        simple: bool,
        edge_aggs: dict | None,
    ):
        from ...adapters import igraph_adapter as _gg_ig

        igG, _manifest = _gg_ig.to_igraph(
            self._G,
            directed=directed,
            hyperedge_mode="expand" if str(hyperedge_mode).lower() == "expand" else "skip",
            slice=slice,
            slices=slices,
            public_only=True,
        )
        self._prune_edge_attributes(igG, needed_attrs)
        if simple:
            igG = self._collapse_multiedges(
                igG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
            )
        return igG

    def _get_or_make_ig(
        self,
        *,
        directed: bool,
        hyperedge_mode: str,
        slice,
        slices,
        needed_attrs: set,
        simple: bool,
        edge_aggs: dict | None,
    ):
        key = (
            bool(directed),
            str(hyperedge_mode),
            self._freeze_cache_value(slices),
            str(slice) if slice is not None else None,
            self._freeze_cache_value(needed_attrs),
            bool(simple),
            self._freeze_cache_value(edge_aggs),
        )
        version = getattr(self._G, "_version", None)
        entry = self._cache.get(key)
        if (
            (not self.cache_enabled)
            or (entry is None)
            or (version is not None and entry.get("version") != version)
        ):
            igG = self._convert_to_ig(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )
            if self.cache_enabled:
                self._cache[key] = {"igG": igG, "version": version}
            return igG
        return entry["igG"]

    def _warn_on_loss(self, *, hyperedge_mode, slice, slices, manifest):
        import warnings

        msgs = []
        if (
            any(rec.etype == "hyper" for rec in self._G._edges.values())
            and hyperedge_mode != "expand"
        ):
            msgs.append("hyperedges dropped (hyperedge_mode='skip')")
        slices_dict = getattr(self._G, "_slices", None)
        if (
            isinstance(slices_dict, dict)
            and len(slices_dict) > 1
            and (slice is None and not slices)
        ):
            msgs.append("multiple slices flattened into single igraph graph")
        if manifest is None:
            msgs.append("no manifest provided; round-trip fidelity not guaranteed")
        if msgs:
            warnings.warn(
                "AnnNet-igraph conversion is lossy: " + "; ".join(msgs) + ".",
                category=RuntimeWarning,
                stacklevel=3,
            )

    def _name_to_index_map(self, igG):
        names = igG.vs["name"] if "name" in igG.vs.attributes() else None
        return {name: idx for idx, name in enumerate(names)} if names is not None else {}

    def _coerce_vertex(self, value, igG, label_field: str | None):
        if isinstance(value, int) and 0 <= value < igG.vcount():
            return value
        if label_field:
            candidate = self._lookup_vertex_id_by_label(label_field, value)
            if candidate is not None:
                value = candidate
        return self._name_to_index_map(igG).get(value, value)

    def _coerce_vertex_or_iter(self, obj, igG, label_field: str | None):
        if isinstance(obj, (list, tuple, set)):
            coerced = [self._coerce_vertex(value, igG, label_field) for value in obj]
            return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
        return self._coerce_vertex(obj, igG, label_field)

    def _coerce_vertices_in_kwargs(self, kwargs: dict, igG, label_field: str | None):
        for key in list(kwargs.keys()):
            if key in self.VERTEX_KEYS:
                kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], igG, label_field)

    def _coerce_vertices_in_bound(self, bound, igG, label_field: str | None):
        for key in list(bound.arguments.keys()):
            if key in self.VERTEX_KEYS:
                bound.arguments[key] = self._coerce_vertex_or_iter(
                    bound.arguments[key], igG, label_field
                )

    def _prune_edge_attributes(self, igG, needed_attrs: set):
        removable = set(igG.es.attributes()) - set(needed_attrs)
        for attr in removable:
            del igG.es[attr]

    def _collapse_multiedges(
        self, igG, *, directed: bool, aggregations: dict | None, needed_attrs: set
    ):
        import igraph as _ig

        H = _ig.Graph(directed=directed)
        H.add_vertices(igG.vcount())
        if "name" in igG.vs.attributes():
            H.vs["name"] = igG.vs["name"]

        aggregations = aggregations or {}

        def agg_for(key):
            agg = aggregations.get(key)
            if callable(agg):
                return agg
            if agg == "sum":
                return sum
            if agg == "min":
                return min
            if agg == "max":
                return max
            if agg == "mean":
                return lambda values: (sum(values) / len(values)) if values else None
            if key == "capacity":
                return sum
            if key == "weight":
                return min
            return lambda values: next(iter(values)) if values else None

        buckets = {}
        for edge in igG.es:
            u, v = edge.tuple
            edge_key = (u, v) if directed else tuple(sorted((u, v)))
            entry = buckets.setdefault(edge_key, {})
            for key, value in edge.attributes().items():
                if needed_attrs and key not in needed_attrs:
                    continue
                entry.setdefault(key, []).append(value)

        edges = list(buckets.keys())
        H.add_edges(edges)
        all_attrs = {key for attrs in buckets.values() for key in attrs.keys()}
        for key in all_attrs:
            agg = agg_for(key)
            H.es[key] = [agg(buckets[edge].get(key, [])) for edge in edges]
        return H
