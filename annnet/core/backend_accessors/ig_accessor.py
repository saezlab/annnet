from __future__ import annotations

from typing import TYPE_CHECKING
import inspect

from ._base import _BackendAccessorBase

if TYPE_CHECKING:
    from ..graph import AnnNet


class _IGBackendAccessor(_BackendAccessorBase):
    """igraph backend accessor attached to an AnnNet instance."""

    VERTEX_KEYS = {
        'source',
        'target',
        'u',
        'v',
        'vertex',
        'vertices',
        'vs',
        'to',
        'fr',
        'root',
        'roots',
        'neighbors',
        'nbunch',
        'path',
        'cut',
    }

    def __init__(self, owner: AnnNet):
        self._init_backend_accessor(owner, cache_attr='_ig_backend_cache')

    def peek_vertices(self, k: int = 10):
        igG = self._get_or_make_ig(
            directed=True,
            hyperedge_mode='skip',
            slice=None,
            slices=None,
            needed_attrs=set(),
            simple=True,
            edge_aggs=None,
        )
        names = igG.vs['name'] if 'name' in igG.vs.attributes() else None
        return [names[i] if names else i for i in range(min(max(0, int(k)), igG.vcount()))]

    def backend(
        self,
        *,
        directed: bool = True,
        hyperedge_mode: str = 'skip',
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

            directed = bool(kwargs.pop('_ig_directed', True))
            hyperedge_mode = kwargs.pop('_ig_hyperedge', 'skip')
            slice = kwargs.pop('_ig_slice', None)
            slices = kwargs.pop('_ig_slices', None)
            label_field = kwargs.pop('_ig_label_field', None)
            guess_labels = kwargs.pop('_ig_guess_labels', True)
            simple = bool(kwargs.pop('_ig_simple', False))
            edge_aggs = kwargs.pop('_ig_edge_aggs', None)

            needed_edge_attrs = self._needed_edge_attrs_for_ig(kwargs)

            if str(hyperedge_mode).lower() == 'reify':
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

            args, kwargs = self._replace_owner_graph(args, kwargs, igG)

            target = getattr(igG, name, None)
            if not callable(target):
                target = getattr(_ig, name, None)
            if not callable(target):
                raise AttributeError(f"igraph has no callable '{name}'")

            try:
                sig = inspect.signature(target)
                bound = sig.bind_partial(*args, **kwargs)
            except Exception:  # noqa: BLE001
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
            except Exception:  # noqa: BLE001
                pargs, pkwargs = list(args), dict(kwargs)

            try:
                raw = target(*pargs, **pkwargs)
                return raw
            except (KeyError, ValueError) as exc:
                sample = self.peek_vertices(5)
                tip = (
                    f"{exc}. Vertices must match this graph's vertex IDs.\n"
                    f'- If you passed labels, set _ig_label_field=<vertex label column>.\n'
                    f"- Example: G.ig.distances(source='a', target='z', weights='weight', _ig_label_field='name')\n"
                    f'- A few vertex IDs igraph sees: {sample}'
                )
                raise type(exc)(tip) from exc

        return wrapper

    def __dir__(self):
        import igraph as _ig

        graph_obj = getattr(_ig, 'Graph', None)
        return sorted(set(super().__dir__()) | self._callable_names(_ig, graph_obj))

    def _needed_edge_attrs_for_ig(self, kwargs: dict) -> set:
        needed = set()
        weight_name = kwargs.get('weights', kwargs.get('weight', None))
        if weight_name is not None:
            needed.add(str(weight_name))
        if 'capacity' in kwargs and kwargs['capacity'] is not None:
            needed.add(str(kwargs['capacity']))
        return needed

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
        key = self._cache_key(
            directed,
            hyperedge_mode,
            slices,
            str(slice) if slice is not None else None,
            needed_attrs,
            simple,
            edge_aggs,
        )

        def build():
            igG, manifest = self._adapter_export('ig')(
                self._G,
                directed=directed,
                hyperedge_mode='expand' if str(hyperedge_mode).lower() == 'expand' else 'skip',
                slice=slice,
                slices=slices,
                public_only=True,
            )
            self._prune_edge_attributes(igG, needed_attrs)
            if simple:
                igG = self._collapse_multiedges(
                    igG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
                )
            self._warn_on_lossy_conversion(
                backend_name='igraph',
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                manifest=manifest,
            )
            return {'igG': igG}

        entry, _rebuilt = self._get_or_make_cached(key, build)
        return entry['igG']

    def _name_to_index_map(self, igG):
        names = igG.vs['name'] if 'name' in igG.vs.attributes() else None
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
        return self._coerce_vertex_iterable(
            obj, lambda value: self._coerce_vertex(value, igG, label_field)
        )

    def _coerce_vertices_in_kwargs(self, kwargs: dict, igG, label_field: str | None):
        self._coerce_vertex_kwargs(
            kwargs, lambda obj: self._coerce_vertex_or_iter(obj, igG, label_field)
        )

    def _coerce_vertices_in_bound(self, bound, igG, label_field: str | None):
        self._coerce_vertex_bound(
            bound, lambda obj: self._coerce_vertex_or_iter(obj, igG, label_field)
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
        if 'name' in igG.vs.attributes():
            H.vs['name'] = igG.vs['name']

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
            agg = self._edge_attr_aggregator(key, aggregations)
            H.es[key] = [agg(buckets[edge].get(key, [])) for edge in edges]
        return H
