from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

from ._base import _BackendAccessorBase

if TYPE_CHECKING:
    from ..graph import AnnNet


class _NXBackendAccessor(_BackendAccessorBase):
    """NetworkX backend accessor attached to an AnnNet instance."""

    VERTEX_KEYS = {'source', 'target', 'u', 'v', 'vertex', 'vertices', 'nbunch', 'center', 'path'}

    def __init__(self, owner: AnnNet):
        self._G = owner
        self._cache = {}
        self.cache_enabled = True

    def clear(self):
        self._cache.clear()

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
                for idx, value in enumerate(args):
                    if value is self._G:
                        args[idx] = nxG
                for key, value in list(kwargs.items()):
                    if value is self._G:
                        kwargs[key] = nxG

            try:
                sig = inspect.signature(nx_callable)
                bound = sig.bind_partial(*args, **kwargs)
            except Exception:
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
            except Exception:
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
        import networkx as _nx

        candidates = [
            _nx,
            getattr(_nx, 'algorithms', None),
            getattr(_nx.algorithms, 'community', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'approximation', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'centrality', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'shortest_paths', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'flow', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'components', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'traversal', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'bipartite', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'link_analysis', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx, 'classes', None),
            getattr(_nx.classes, 'function', None) if hasattr(_nx, 'classes') else None,
        ]
        return sorted(set(super().__dir__()) | self._callable_names(*candidates))

    def _resolve_nx_callable(self, name: str):
        import networkx as _nx

        candidates = [
            _nx,
            getattr(_nx, 'algorithms', None),
            getattr(_nx.algorithms, 'community', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'approximation', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'centrality', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'shortest_paths', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'flow', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'components', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'traversal', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'bipartite', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx.algorithms, 'link_analysis', None) if hasattr(_nx, 'algorithms') else None,
            getattr(_nx, 'classes', None),
            getattr(_nx.classes, 'function', None) if hasattr(_nx, 'classes') else None,
        ]
        for mod in (candidate for candidate in candidates if candidate is not None):
            attr = getattr(mod, name, None)
            if callable(attr):
                return attr
        raise AttributeError(f"networkx has no callable '{name}'")

    def _needed_edge_attrs(self, target, kwargs) -> set:
        needed = set()
        try:
            params = inspect.signature(target).parameters
        except Exception:
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

    def _convert_to_nx(
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
        from ...adapters import networkx_adapter as _gg_nx

        nxG, manifest = _gg_nx.to_nx(
            self._G,
            directed=directed,
            hyperedge_mode=hyperedge_mode,
            slice=slice,
            slices=slices,
            public_only=True,
        )
        if needed_attrs:
            for _, _, _, data in nxG.edges(keys=True, data=True):
                for key in list(data.keys()):
                    if key not in needed_attrs:
                        data.pop(key, None)
        elif simple:
            for _, _, _, data in nxG.edges(keys=True, data=True):
                for key in list(data.keys()):
                    if key not in ('weight', 'capacity'):
                        data.pop(key, None)
        else:
            for _, _, _, data in nxG.edges(keys=True, data=True):
                data.clear()

        if simple and nxG.is_multigraph():
            nxG = self._collapse_multiedges(
                nxG, directed=directed, aggregations=edge_aggs, needed_attrs=needed_attrs
            )

        self._warn_on_loss(
            hyperedge_mode=hyperedge_mode, slice=slice, slices=slices, manifest=manifest
        )
        return nxG

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
        key = (
            bool(directed),
            str(hyperedge_mode),
            self._freeze_cache_value(slices),
            str(slice) if slice is not None else None,
            self._freeze_cache_value(needed_attrs),
            bool(simple),
            self._freeze_cache_value(edge_aggs),
        )
        version = getattr(self._G, '_version', None)
        entry = self._cache.get(key)
        if (
            (not self.cache_enabled)
            or (entry is None)
            or (version is not None and entry.get('version') != version)
        ):
            nxG = self._convert_to_nx(
                directed=directed,
                hyperedge_mode=hyperedge_mode,
                slice=slice,
                slices=slices,
                needed_attrs=needed_attrs,
                simple=simple,
                edge_aggs=edge_aggs,
            )
            if self.cache_enabled:
                self._cache[key] = {'nxG': nxG, 'version': version}
            return nxG
        return entry['nxG']

    def _warn_on_loss(self, *, hyperedge_mode, slice, slices, manifest):
        import warnings

        msgs = []
        if (
            any(rec.etype == 'hyper' for rec in self._G._edges.values())
            and hyperedge_mode != 'expand'
        ):
            msgs.append("hyperedges dropped (hyperedge_mode='skip')")
        slices_dict = getattr(self._G, '_slices', None)
        if (
            isinstance(slices_dict, dict)
            and len(slices_dict) > 1
            and (slice is None and not slices)
        ):
            msgs.append('multiple slices flattened into single NX graph')
        if manifest is None:
            msgs.append('no manifest provided; round-trip fidelity not guaranteed')
        if msgs:
            warnings.warn(
                'AnnNet-NX conversion is lossy: ' + '; '.join(msgs) + '.',
                category=RuntimeWarning,
                stacklevel=3,
            )

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
        if isinstance(obj, (list, tuple, set)):
            coerced = [self._coerce_vertex_id(value, nxG, label_field) for value in obj]
            return type(obj)(coerced) if not isinstance(obj, set) else set(coerced)
        return self._coerce_vertex_id(obj, nxG, label_field)

    def _coerce_vertices_in_kwargs(self, kwargs: dict, nxG, label_field: str | None):
        for key in list(kwargs.keys()):
            if key in self.VERTEX_KEYS:
                kwargs[key] = self._coerce_vertex_or_iter(kwargs[key], nxG, label_field)

    def _coerce_vertices_in_bound(self, bound, nxG, label_field: str | None):
        for key in list(bound.arguments.keys()):
            if key in self.VERTEX_KEYS:
                bound.arguments[key] = self._coerce_vertex_or_iter(
                    bound.arguments[key], nxG, label_field
                )

    def _map_output_vertices(self, obj):
        id_to_row, row_to_id = self._vertex_row_maps()

        def map_id(value):
            if isinstance(value, str):
                return id_to_row.get(value, value)
            if isinstance(value, int) and value in row_to_id:
                return value
            return value

        return self._map_nested_output(obj, map_id)

    def _collapse_multiedges(
        self, nxG, *, directed: bool, aggregations: dict | None, needed_attrs: set
    ):
        import networkx as _nx

        H = _nx.DiGraph() if directed else _nx.Graph()
        H.add_nodes_from(nxG.nodes(data=True))

        aggregations = aggregations or {}

        def agg_for(key):
            agg = aggregations.get(key)
            if callable(agg):
                return agg
            if agg == 'sum':
                return sum
            if agg == 'min':
                return min
            if agg == 'max':
                return max
            if key == 'capacity':
                return sum
            if key == 'weight':
                return min
            return lambda values: next(iter(values))

        buckets = {}
        for u, v, _, data in nxG.edges(keys=True, data=True):
            edge_key = (u, v) if directed else tuple(sorted((u, v)))
            entry = buckets.setdefault(edge_key, {})
            for key, value in data.items():
                if needed_attrs and key not in needed_attrs:
                    continue
                entry.setdefault(key, []).append(value)

        for (u, v), attrs in buckets.items():
            H.add_edge(u, v, **{key: agg_for(key)(values) for key, values in attrs.items()})

        return H
