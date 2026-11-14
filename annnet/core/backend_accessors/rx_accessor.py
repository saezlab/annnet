from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._base import _cached_signature, _BackendAccessorBase

if TYPE_CHECKING:
    from ..graph import AnnNet


class _RXBackendAccessor(_BackendAccessorBase):
    """rustworkx backend accessor attached to an AnnNet instance.

    ``anet.rx.<function>(...)`` calls ``rustworkx.<function>`` on a converted
    copy of the graph, which is cached until the graph changes. Node ids in the
    arguments become rustworkx indices, and node indices in the result become
    node ids again. A ``weight_fn`` or ``edge_cost_fn`` given as a string reads
    that key of the edge payload.

    Options, removed from the call before it reaches rustworkx:

    - ``_rx_directed`` (default True): a ``PyDiGraph`` or a ``PyGraph``.
    - ``_rx_hyperedge`` (default ``"skip"``): ``"skip"`` or ``"expand"``.
    - ``_rx_slice`` / ``_rx_slices``: convert only these slices.
    - ``_rx_raw`` (default False): return the rustworkx result untouched.
    """

    # The parameter names rustworkx gives a node argument.
    NODE_KEYS = {
        'source',
        'target',
        'node',
        'nodes',
        'origin',
        'to',
        'from_',
        'root',
        'start',
        'first_node',
        'second_node',
    }

    # Functions whose result is a list of node groups, given as lists of indices.
    _NODE_GROUP_RESULTS = {
        'strongly_connected_components',
        'weakly_connected_components',
        'connected_components',
        'simple_cycles',
        'cycle_basis',
        'all_simple_paths',
        'digraph_all_simple_paths',
        'graph_all_simple_paths',
        'bfs_layers',
        'topological_generations',
        'biconnected_components',
        'articulation_points',
        'dominating_set',
        'core_number',
        'maximal_independent_set',
    }

    def __init__(self, owner: AnnNet):
        self._init_backend_accessor(owner, cache_attr='_rx_backend_cache')

    def backend(
        self,
        *,
        directed: bool = True,
        hyperedge_mode: str = 'skip',
        slice=None,
        slices=None,
    ):
        """Return the converted rustworkx graph (cached until the graph changes)."""
        return self._get_or_make_rx(
            directed=directed, hyperedge_mode=hyperedge_mode, slice=slice, slices=slices
        )

    def peek_nodes(self, k: int = 10) -> list:
        rxG = self.backend()
        return [rxG[i]['node_id'] for i in list(rxG.node_indices())[: max(0, int(k))]]

    def __getattr__(self, name: str):
        if name.startswith('_'):
            raise AttributeError(name)

        def wrapper(*args, **kwargs):
            import rustworkx as _rx

            directed = bool(kwargs.pop('_rx_directed', True))
            hyperedge_mode = kwargs.pop('_rx_hyperedge', 'skip')
            slice = kwargs.pop('_rx_slice', None)
            slices = kwargs.pop('_rx_slices', None)
            raw = bool(kwargs.pop('_rx_raw', False))

            target = getattr(_rx, name, None)
            if not callable(target):
                raise AttributeError(f"rustworkx has no callable '{name}'")

            rxG = self._get_or_make_rx(
                directed=directed, hyperedge_mode=hyperedge_mode, slice=slice, slices=slices
            )
            args, kwargs = self._replace_owner_graph(args, kwargs, rxG)
            if not args or args[0] is not rxG:
                args = [rxG, *args]
            for key in ('weight_fn', 'edge_cost_fn'):
                if isinstance(kwargs.get(key), str):
                    kwargs[key] = _payload_getter(kwargs[key])

            index_of = rxG.attrs['index_of']
            coerce = lambda obj: self._coerce_node_iterable(
                obj, lambda value: index_of.get(value, value)
            )
            try:
                bound = _cached_signature(target).bind_partial(*args, **kwargs)
            except (TypeError, ValueError):
                bound = None
            if bound is not None:
                self._coerce_node_bound(bound, coerce)
                args, kwargs = list(bound.args), dict(bound.kwargs)
            else:
                self._coerce_node_kwargs(kwargs, coerce)

            try:
                result = target(*args, **kwargs)
            except (IndexError, KeyError, TypeError, OverflowError) as exc:
                raise type(exc)(
                    f'{exc}. Nodes are given by node id. A few node ids of this graph: '
                    f'{self.peek_nodes(5)}'
                ) from exc
            if raw:
                return result
            return self._to_ids(result, rxG, name)

        return wrapper

    def __dir__(self):
        import rustworkx as _rx

        return sorted(set(super().__dir__()) | self._callable_names(_rx))

    # -- results back to ids -------------------------------------------------

    def _to_ids(self, result, rxG, name: str):
        """Turn the node indices in a rustworkx result back into node ids."""
        import rustworkx as _rx

        node = lambda i: rxG[i]['node_id'] if isinstance(i, int) and rxG.has_node(i) else i
        edge = lambda i: rxG.get_edge_data_by_index(i).get('edge_id', i)
        type_name = type(result).__name__

        if isinstance(result, (_rx.NodeIndices,)):
            return [node(i) for i in result]
        if isinstance(result, _rx.EdgeIndices):
            return [edge(i) for i in result]
        if type_name == 'EdgeCentralityMapping':
            return {edge(k): v for k, v in result.items()}
        if type_name in {'EdgeIndexMap'}:
            return {edge(k): (node(u), node(v), w) for k, (u, v, w) in result.items()}
        if type_name in {'EdgeList', 'WeightedEdgeList'}:
            return [(node(e[0]), node(e[1]), *e[2:]) for e in result]
        if type_name in {'AllPairsPathMapping', 'AllPairsPathLengthMapping'}:
            return {node(k): self._to_ids(v, rxG, name) for k, v in result.items()}
        if type_name == 'PathMapping':
            return {node(k): [node(i) for i in v] for k, v in result.items()}
        if type_name == 'BFSSuccessors' or type_name == 'BFSPredecessors':
            return {node(k): [node(i) for i in v] for k, v in result}
        if type_name.endswith('Mapping') or isinstance(result, dict):
            return {node(k): v for k, v in result.items()}
        if isinstance(result, (set, frozenset)):
            # rustworkx returns a set only of node indices (descendants, ancestors, ...).
            return {node(i) for i in result}
        if name in self._NODE_GROUP_RESULTS:
            return self._map_nested_output(result, node)
        return result

    # -- the cached conversion ---------------------------------------------------

    def _get_or_make_rx(self, *, directed: bool, hyperedge_mode: str, slice, slices):
        mode = 'expand' if str(hyperedge_mode).lower() == 'expand' else 'skip'
        key = self._cache_key(directed, mode, slices, str(slice) if slice is not None else None)

        def build():
            rxG, manifest = self._adapter_export('rx')(
                self._G,
                directed=directed,
                hyperedge_mode=mode,
                slice=slice,
                slices=slices,
                public_only=True,
            )
            self._warn_on_lossy_conversion(
                backend_name='rustworkx',
                hyperedge_mode=mode,
                slice=slice,
                slices=slices,
                manifest=manifest,
            )
            return {'rxG': rxG}

        entry, _rebuilt = self._get_or_make_cached(key, build)
        return entry['rxG']


def _payload_getter(key: str):
    def get(payload: Any) -> float:
        value = payload.get(key, 1.0) if isinstance(payload, dict) else payload
        return float(value if value is not None else 1.0)

    return get
