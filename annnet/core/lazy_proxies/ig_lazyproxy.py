## Lazy iGraph proxy
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

if TYPE_CHECKING:
    from ..graph import AnnNet


class _LazyIGProxy:
    """Lazy, cached igraph adapter:
    - On-demand backend conversion (no persistent igraph graph).
    - Cache keyed by options until AnnNet._version changes.
    - Selective edge-attr exposure (keep only needed weights/capacity).
    - Clear warnings when conversion is lossy.
    - Auto label-ID mapping for vertex args (kwargs + positionals).
    - _ig_simple=True collapses parallel edges to simple (Di)AnnNet.
    - _ig_edge_aggs={"weight":"min","capacity":"sum"} for parallel-edge aggregation.
    """

    def __init__(self, owner: AnnNet):
        self._G = owner
        self._cache = {}  # key -> {"igG": ig.AnnNet, "version": int}
        self.cache_enabled = True

    #  public API
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
        out = []
        names = igG.vs["name"] if "name" in igG.vs.attributes() else None
        for i in range(min(max(0, int(k)), igG.vcount())):
            out.append(names[i] if names else i)
        return out

    # public helper so tests don’t touch private API
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
        needed_attrs = needed_attrs or set()
        return self._get_or_make_ig(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            needed_attrs=needed_attrs,
            simple=simple,
            edge_aggs=edge_aggs,
        )

    # - dynamic dispatch -
    def __getattr__(self, name: str):
        def wrapper(*args, **kwargs):
            import igraph as _ig

            # proxy-only knobs (consumed here)
            directed = bool(kwargs.pop("_ig_directed", True))
            hyperedge_mode = kwargs.pop("_ig_hyperedge", "skip")  # "skip" | "expand"
            slice = kwargs.pop("_ig_slice", None)
            slices = kwargs.pop("_ig_slices", None)
            label_field = kwargs.pop("_ig_label_field", None)
            guess_labels = kwargs.pop("_ig_guess_labels", True)
            simple = bool(kwargs.pop("_ig_simple", False))
            edge_aggs = kwargs.pop(
                "_ig_edge_aggs", None
            )  # {"weight":"min","capacity":"sum"} or callables

            # keep only attributes actually needed by the called function
            needed_edge_attrs = self._needed_edge_attrs_for_ig(name, kwargs)

            # build/reuse backend
            igG = self._get_or_make_ig(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_edge_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )

            # replace any AnnNet instance with igG
            args = list(args)
            for i, v in enumerate(args):
                if v is self._G:
                    args[i] = igG
            for k, v in list(kwargs.items()):
                if v is self._G:
                    kwargs[k] = igG

            # resolve target callable: prefer bound AnnNet method, else module-level
            target = getattr(igG, name, None)
            if not callable(target):
                target = getattr(_ig, name, None)
            if not callable(target):
                raise AttributeError(
                    f"igraph has no callable '{name}'. "
                    f"Use native igraph names, e.g. community_multilevel, pagerank, shortest_paths_dijkstra, components, etc."
                )

            # bind to signature (best effort) so we can coerce vertex args
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
                pargs, pkwargs = list(args), dict(kwargs)  # let igraph raise if invalid

            try:
                raw = target(*pargs, **pkwargs)
                return self._map_output_vertices(raw)
            except (KeyError, ValueError) as e:
                sample = self.peek_vertices(5)
                tip = (
                    f"{e}. Vertices must match this graph's vertex IDs.\n"
                    f"- If you passed labels, set _ig_label_field=<vertex label column> "
                    f"or rely on auto-guess ('name'/'label'/'title').\n"
                    f"- Example: G.ig.shortest_paths_dijkstra(G, source='a', target='z', weights='weight', _ig_label_field='name')\n"
                    f"- A few vertex IDs igraph sees: {sample}"
                )
                raise type(e)(tip) from e

        return wrapper

    # -- internals ---
    def _needed_edge_attrs_for_ig(self, func_name: str, kwargs: dict) -> set:
        """Heuristic: igraph uses `weights` (plural) for edge weights in most algos,
        some accept both; flows use 'capacity' if you forward them to adapters.
        """
        needed = set()
        # weight(s)
        w = kwargs.get("weights", kwargs.get("weight", None))
        if w is None:
            # sometimes user passes True to mean default "weight"
            if "weights" in kwargs and kwargs["weights"] is not None:
                needed.add(str(kwargs["weights"]))
        else:
            needed.add(str(w))
        # capacity (if you forward flow-like algos to ig backends)
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
        # try both adapter entry points: to_ig / to_igraph

        from ...adapters import igraph_adapter as _gg_ig  # annnet.adapters.igraph_adapter

        conv = None
        for cand in ("to_ig", "to_igraph"):
            conv = getattr(_gg_ig, cand, None) or conv
        if conv is None:
            raise RuntimeError(
                "igraph adapter missing: expected adapters.igraph_adapter.to_ig(...) or .to_igraph(...)."
            )

        igG, manifest = conv(
            self._G,
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            public_only=True,
        )

        # keep only requested edge attrs (or none at all)
        igG = self._prune_edge_attributes(igG, needed_attrs)

        # igraph lacks is_multigraph(); always collapse when simple=True
        if simple:
            igG = self._collapse_multiedges(
                igG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
            )

        self._warn_on_loss(
            hyperedge_mode=hyperedge_mode, slice=slice, slices=slices, manifest=manifest
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
            tuple(sorted(slices)) if slices else None,
            str(slice) if slice is not None else None,
            tuple(sorted(needed_attrs)) if needed_attrs else (),
            bool(simple),
            tuple(sorted(edge_aggs.items())) if isinstance(edge_aggs, dict) else None,
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

        has_hyper = False
        try:
            ek = getattr(self._G, "edge_kind", {})
            if hasattr(ek, "values"):
                has_hyper = any(str(v).lower() == "hyper" for v in ek.values())
        except Exception:
            pass
        msgs = []
        if has_hyper and hyperedge_mode != "expand":
            msgs.append("hyperedges dropped (hyperedge_mode='skip')")
        try:
            slices_dict = getattr(self._G, "_slices", None)
            if (
                isinstance(slices_dict, dict)
                and len(slices_dict) > 1
                and (slice is None and not slices)
            ):
                msgs.append("multiple slices flattened into single igraph graph")
        except Exception:
            pass
        if manifest is None:
            msgs.append("no manifest provided; round-trip fidelity not guaranteed")
        if msgs:
            warnings.warn(
                "AnnNet-igraph conversion is lossy: " + "; ".join(msgs) + ".",
                category=RuntimeWarning,
                stacklevel=3,
            )

    # -- label/ID mapping helpers
    def _infer_label_field(self) -> str | None:
        try:
            if hasattr(self._G, "default_label_field") and self._G.default_label_field:
                return self._G.default_label_field
            va = getattr(self._G, "vertex_attributes", None)
            cols = list(va.columns) if va is not None and hasattr(va, "columns") else []
            for c in ("name", "label", "title", "slug", "external_id", "string_id"):
                if c in cols:
                    return c
        except Exception:
            pass
        return None

    def _vertex_id_col(self) -> str:
        try:
            va = self._G.vertex_attributes
            cols = list(va.columns)
            for k in ("vertex_id", "id", "vid"):
                if k in cols:
                    return k
        except Exception:
            pass
        return "vertex_id"

    def _lookup_vertex_id_by_label(self, label_field: str, val):
        try:
            va = self._G.vertex_attributes
            if va is None or not hasattr(va, "columns") or label_field not in va.columns:
                return None
            id_col = self._vertex_id_col()
            try:
                # type: ignore

                matches = va.filter(pl.col(label_field) == val)
                if matches.height == 0:
                    return None
                try:
                    return matches.select(id_col).to_series().to_list()[0]
                except Exception:
                    return matches.select(id_col).item(0, 0)
            except Exception:
                for row in va.to_dicts():
                    if row.get(label_field) == val:
                        return row.get(id_col)
        except Exception:
            return None
        return None

    def _name_to_index_map(self, igG):
        names = igG.vs["name"] if "name" in igG.vs.attributes() else None
        return {n: i for i, n in enumerate(names)} if names is not None else {}

    def _coerce_vertex(self, x, igG, label_field: str | None):
        # already an index?
        if isinstance(x, int) and 0 <= x < igG.vcount():
            return x
        # graph-level mapping (label -> vertex_id)
        if label_field:
            cand = self._lookup_vertex_id_by_label(label_field, x)
            if cand is not None:
                x = cand
        # igraph name -> index
        name_to_idx = self._name_to_index_map(igG)
        if x in name_to_idx:
            return name_to_idx[x]
        # if user already passed internal vertex_id string, try treating it as name
        if isinstance(x, str) and x in name_to_idx:
            return name_to_idx[x]
        return x  # let igraph validate/raise

    def _coerce_vertex_or_iter(self, obj, igG, label_field: str | None):
        if isinstance(obj, (list, tuple, set)):
            coerced = [self._coerce_vertex(v, igG, label_field) for v in obj]
            return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
        return self._coerce_vertex(obj, igG, label_field)

    def _coerce_vertices_in_kwargs(self, kwargs: dict, igG, label_field: str | None):
        vertex_keys = {
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
        for key in list(kwargs.keys()):
            if key in vertex_keys:
                kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], igG, label_field)

    def _coerce_vertices_in_bound(self, bound, igG, label_field: str | None):
        vertex_keys = {
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
        for key in list(bound.arguments.keys()):
            if key in vertex_keys:
                bound.arguments[key] = self._coerce_vertex_or_iter(
                    bound.arguments[key], igG, label_field
                )

    def _map_output_vertices(self, obj):
        """Map igraph output structures (indices, set/list/tuple/dict) back to AnnNet row indices."""
        G = self._G
        id2row = G.entity_to_idx  # entity_id → row index
        idx2id = G.idx_to_entity  # row index → entity_id

        def map_idx(i):
            # if igraph returned an index, convert: index -> internal entity_id -> row index
            if isinstance(i, int) and i in idx2id:
                eid = idx2id[i]
                return id2row.get(eid, i)
            return i

        # Dict
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                out[map_idx(k)] = self._map_output_vertices(v)
            return out

        # List
        if isinstance(obj, list):
            return [self._map_output_vertices(x) for x in obj]

        # Tuple
        if isinstance(obj, tuple):
            return tuple(self._map_output_vertices(x) for x in obj)

        # Set
        if isinstance(obj, set):
            return {self._map_output_vertices(x) for x in obj}

        # Leaf
        return map_idx(obj)

    # -- edge-attr & multiedge helpers ---
    def _prune_edge_attributes(self, igG, needed_attrs: set):
        import igraph as _ig

        if not needed_attrs:
            # keep only 'name' on vertices, drop all edge attrs quickly by rebuild
            H = _ig.Graph(directed=igG.is_directed())
            H.add_vertices(igG.vcount())
            if "name" in igG.vs.attributes():
                H.vs["name"] = igG.vs["name"]
            H.add_edges([e.tuple for e in igG.es])
            return H
        # keep only specific attrs
        H = _ig.Graph(directed=igG.is_directed())
        H.add_vertices(igG.vcount())
        if "name" in igG.vs.attributes():
            H.vs["name"] = igG.vs["name"]
        edges = [e.tuple for e in igG.es]
        H.add_edges(edges)
        have = set(igG.es.attributes())
        for k in needed_attrs:
            if k in have:
                H.es[k] = igG.es[k]
        return H

    def _collapse_multiedges(
        self, igG, *, directed: bool, aggregations: dict | None, needed_attrs: set
    ):
        import igraph as _ig

        H = _ig.Graph(directed=directed)
        H.add_vertices(igG.vcount())
        if "name" in igG.vs.attributes():
            H.vs["name"] = igG.vs["name"]

        aggregations = aggregations or {}

        def _agg_for(key):
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
                return lambda vals: (sum(vals) / len(vals)) if vals else None
            if key == "capacity":
                return sum
            if key == "weight":
                return min
            return lambda vals: next(iter(vals)) if vals else None

        # bucket edges
        buckets = {}  # (u,v) or sorted(u,v) -> {attr: [vals]}
        for e in igG.es:
            u, v = e.tuple
            key = (u, v) if directed else tuple(sorted((u, v)))
            entry = buckets.setdefault(key, {})
            for k, val in e.attributes().items():
                if needed_attrs and k not in needed_attrs:
                    continue
                entry.setdefault(k, []).append(val)

        edges = list(buckets.keys())
        H.add_edges(edges)
        # aggregate per attribute
        all_attrs = set(k for _, attrs in buckets.items() for k in attrs.keys())
        for k in all_attrs:
            agg = _agg_for(k)
            H.es[k] = [agg(buckets[edge].get(k, [])) for edge in edges]
        return H
