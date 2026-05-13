from __future__ import annotations

from typing import TYPE_CHECKING
import inspect

from ._base import _BackendAccessorBase

if TYPE_CHECKING:
    from ..graph import AnnNet


class _NXBackendAccessor(_BackendAccessorBase):
    """NetworkX backend accessor attached to an AnnNet instance."""

    VERTEX_KEYS = {'source', 'target', 'u', 'v', 'vertex', 'vertices', 'nbunch', 'center', 'path'}

    def __init__(self, owner: AnnNet):
        self._init_backend_accessor(owner, cache_attr='_nx_backend_cache')

    def peek_vertices(self, k: int = 10):
        nxG = self._get_or_make_nx(
            directed=True,
            hyperedge_mode='expand',
            slice=None,
            slices=None,
            needed_attrs=set(),
            simple=False,
            edge_aggs=None,
        )
        out = []
        it = iter(nxG.nodes())
        for _ in range(max(0, int(k))):
            try:
                out.append(next(it))
            except StopIteration:
                break
        return out

    def backend(
        self,
        *,
        directed: bool = True,
        hyperedge_mode: str = 'expand',
        slice=None,
        slices=None,
        needed_attrs=None,
        simple: bool = False,
        edge_aggs: dict | None = None,
    ):
        return self._get_or_make_nx(
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            needed_attrs=needed_attrs or set(),
            simple=simple,
            edge_aggs=edge_aggs,
        )

    def __getattr__(self, name: str):
        nx_callable = self._resolve_nx_callable(name)

        def wrapper(*args, **kwargs):
            import networkx as _nx

            directed = bool(kwargs.pop('_nx_directed', getattr(self, 'default_directed', True)))
            hyperedge_mode = kwargs.pop(
                '_nx_hyperedge', getattr(self, 'default_hyperedge_mode', 'expand')
            )
            slice = kwargs.pop('_nx_slice', None)
            slices = kwargs.pop('_nx_slices', None)
            label_field = kwargs.pop('_nx_label_field', None)
            guess_labels = kwargs.pop('_nx_guess_labels', True)
            simple = bool(kwargs.pop('_nx_simple', getattr(self, 'default_simple', False)))
            edge_aggs = kwargs.pop('_nx_edge_aggs', None)

            needed_edge_attrs = self._needed_edge_attrs(nx_callable, kwargs)

            args = list(args)
            has_owner_graph = any(arg is self._G for arg in args) or any(
                value is self._G for value in kwargs.values()
            )

            nxG = None
            if has_owner_graph:
                nxG = self._get_or_make_nx(
                    directed=directed,
                    hyperedge_mode=hyperedge_mode,
                    slice=slice,
                    slices=slices,
                    needed_attrs=needed_edge_attrs,
                    simple=simple,
                    edge_aggs=edge_aggs,
                )
                args, kwargs = self._replace_owner_graph(args, kwargs, nxG)

            try:
                sig = inspect.signature(nx_callable)
                bound = sig.bind_partial(*args, **kwargs)
            except Exception:  # noqa: BLE001
                bound = None

            try:
                if label_field is None and guess_labels:
                    label_field = self._infer_label_field()
                if bound is not None and nxG is not None:
                    self._coerce_vertices_in_bound(bound, nxG, label_field)
                    pargs, pkwargs = bound.args, bound.kwargs
                else:
                    if nxG is not None:
                        self._coerce_vertices_in_kwargs(kwargs, nxG, label_field)
                    pargs, pkwargs = tuple(args), kwargs
            except Exception:  # noqa: BLE001
                pargs, pkwargs = tuple(args), kwargs

            for key in list(pkwargs.keys()):
                if isinstance(key, str) and key.startswith('_nx_'):
                    pkwargs.pop(key, None)

            try:
                raw = nx_callable(*pargs, **pkwargs)
                return self._map_output_vertices(raw)
            except _nx.NodeNotFound as exc:
                sample = self.peek_vertices(5)
                tip = (
                    f"{exc}. vertices must be graph's vertex IDs.\n"
                    f'- If you passed labels, specify _nx_label_field=<vertex label column> '
                    f'or rely on auto-guess.\n'
                    f"- Example: G.nx.shortest_path_length(G, source='a', target='z', "
                    f"weight='weight', _nx_label_field='name')\n"
                    f'- A few vertex IDs NX sees: {sample}'
                )
                raise _nx.NodeNotFound(tip) from exc

        return wrapper

    def __dir__(self):
        return sorted(set(super().__dir__()) | self._callable_names(*self._nx_candidates()))

    def _resolve_nx_callable(self, name: str):
        for mod in self._nx_candidates():
            attr = getattr(mod, name, None)
            if callable(attr):
                return attr
        raise AttributeError(f"networkx has no callable '{name}'")

    def _nx_candidates(self):
        import networkx as _nx

        algorithms = getattr(_nx, 'algorithms', None)
        classes = getattr(_nx, 'classes', None)
        return tuple(
            candidate
            for candidate in (
                _nx,
                algorithms,
                getattr(algorithms, 'community', None),
                getattr(algorithms, 'approximation', None),
                getattr(algorithms, 'centrality', None),
                getattr(algorithms, 'shortest_paths', None),
                getattr(algorithms, 'flow', None),
                getattr(algorithms, 'components', None),
                getattr(algorithms, 'traversal', None),
                getattr(algorithms, 'bipartite', None),
                getattr(algorithms, 'link_analysis', None),
                classes,
                getattr(classes, 'function', None),
            )
            if candidate is not None
        )

    def _needed_edge_attrs(self, target, kwargs) -> set:
        needed = set()
        try:
            params = inspect.signature(target).parameters
        except Exception:  # noqa: BLE001
            params = {}
        if 'weight' in params:
            weight_name = kwargs.get('weight', 'weight')
            if weight_name is not None:
                needed.add(str(weight_name))
        elif 'weight' in kwargs and kwargs['weight'] is not None:
            needed.add(str(kwargs['weight']))
        if 'capacity' in params:
            capacity_name = kwargs.get('capacity', 'capacity')
            if capacity_name is not None:
                needed.add(str(capacity_name))
        elif 'capacity' in kwargs and kwargs['capacity'] is not None:
            needed.add(str(kwargs['capacity']))
        return needed

    def _get_or_make_nx(
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
            nxG, manifest = self._adapter_export('nx')(
                self._G,
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                public_only=True,
            )
            if needed_attrs:
                for _, _, _, data in nxG.edges(keys=True, data=True):
                    for name in list(data.keys()):
                        if name not in needed_attrs:
                            data.pop(name, None)
            elif simple:
                for _, _, _, data in nxG.edges(keys=True, data=True):
                    for name in list(data.keys()):
                        if name not in ('weight', 'capacity'):
                            data.pop(name, None)
            else:
                for _, _, _, data in nxG.edges(keys=True, data=True):
                    data.clear()

            if simple and nxG.is_multigraph():
                nxG = self._collapse_multiedges(
                    nxG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
                )

            self._warn_on_lossy_conversion(
                backend_name='NX',
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                manifest=manifest,
            )
            return {'nxG': nxG}

        entry, _rebuilt = self._get_or_make_cached(key, build)
        return entry['nxG']

    def _coerce_vertex_id(self, value, nxG, label_field: str | None):
        if isinstance(value, int):
            candidate = self._vertex_row_to_id(value)
            if candidate is not None:
                value = candidate
        if value in nxG:
            return value
        if label_field:
            candidate = self._lookup_vertex_id_by_label(label_field, value)
            if candidate is not None:
                return candidate
        return value

    def _coerce_vertex_or_iter(self, obj, nxG, label_field: str | None):
        return self._coerce_vertex_iterable(
            obj, lambda value: self._coerce_vertex_id(value, nxG, label_field)
        )

    def _coerce_vertices_in_kwargs(self, kwargs: dict, nxG, label_field: str | None):
        self._coerce_vertex_kwargs(
            kwargs, lambda obj: self._coerce_vertex_or_iter(obj, nxG, label_field)
        )

    def _coerce_vertices_in_bound(self, bound, nxG, label_field: str | None):
        self._coerce_vertex_bound(
            bound, lambda obj: self._coerce_vertex_or_iter(obj, nxG, label_field)
        )

    def _map_output_vertices(self, obj):
        _id_to_row, row_to_id = self._vertex_row_maps()

        def map_id(value):
            # NetworkX returns vertex IDs as the backend graph's node values.
            # Our backend nodes are vertex-ID strings, so strings pass through
            # unchanged. Integers that NX produced from a relabel-to-int pass
            # are mapped back to their vertex IDs.
            if isinstance(value, int) and not isinstance(value, bool) and value in row_to_id:
                return row_to_id[value]
            return value

        return self._map_nested_output(obj, map_id)

    def _collapse_multiedges(
        self, nxG, *, directed: bool, aggregations: dict | None, needed_attrs: set
    ):
        import networkx as _nx

        H = _nx.DiGraph() if directed else _nx.Graph()
        H.add_nodes_from(nxG.nodes(data=True))

        buckets = {}
        for u, v, _, data in nxG.edges(keys=True, data=True):
            edge_key = (u, v) if directed else tuple(sorted((u, v)))
            entry = buckets.setdefault(edge_key, {})
            for key, value in data.items():
                if needed_attrs and key not in needed_attrs:
                    continue
                entry.setdefault(key, []).append(value)

        for (u, v), attrs in buckets.items():
            H.add_edge(
                u,
                v,
                **{
                    key: self._edge_attr_aggregator(key, aggregations)(values)
                    for key, values in attrs.items()
                },
            )

        return H
