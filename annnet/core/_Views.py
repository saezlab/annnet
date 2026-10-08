"""Live, read-only graph views.

GraphView stores its parent and filters and caches resolved membership against
state clocks. Callable filters are re-evaluated on each read. Each operation
uses a consistent resolution.

Membership resolution lives in ``_resolve``; scoped sequences in ``_select``;
attribute reads in ``_attribute_api.ScopedAttrs``; table rendering and summaries
in ``_tables`` and ``_summary``; graph construction in ``_materialize``.
Writes are rejected with a reference to ``materialize()``.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping

import numpy as np

from . import _structure
from ._select import EdgeSequence, NodeSequence, ReadOnlyViewError, refuse_write
from ._records import _external_entity_kind
from ._resolve import Resolved, resolve, normalize_filters
from ._stored_kinds import STORED_ENTITY_KIND
from ._attribute_api import ScopedAttrs

# ---------------------------------------------------------------------------
# Read-only wrappers
# ---------------------------------------------------------------------------


class _FrozenMapping(Mapping):
    """A read-only, nested view of a mapping; writes name ``.materialize()``."""

    __slots__ = ('_held',)

    def __init__(self, held):
        self._held = held

    def __getitem__(self, key):
        return _frozen(self._held[key])

    def __iter__(self):
        return iter(self._held)

    def __len__(self):
        return len(self._held)

    def __setitem__(self, key, value):
        raise refuse_write('the metadata of a view')

    def __delitem__(self, key):
        raise refuse_write('the metadata of a view')

    def update(self, *args, **kwargs):
        raise refuse_write('the metadata of a view')

    def pop(self, *args):
        raise refuse_write('the metadata of a view')

    def setdefault(self, *args):
        raise refuse_write('the metadata of a view')

    def clear(self):
        raise refuse_write('the metadata of a view')

    def __repr__(self):
        return f'{dict(self._held)!r} (read-only)'


def _frozen(value):
    if isinstance(value, MutableMapping):
        return _FrozenMapping(value)
    if isinstance(value, list):
        return tuple(_frozen(v) for v in value)
    if isinstance(value, set):
        return frozenset(value)
    if isinstance(value, np.ndarray):
        copy = value.copy()
        copy.flags.writeable = False
        return copy
    return value


class _ViewSlices:
    """The slice reads of a view, restricted to the slices and elements it holds."""

    __slots__ = ('_view',)

    def __init__(self, view):
        self._view = view

    def _resolved(self):
        return self._view._resolved()

    def list(self, include_default: bool = True) -> list:
        resolved = self._resolved()
        default = self._view._graph._default_slice
        return [sid for sid in resolved.slices if include_default or sid != default]

    def exists(self, slice_id) -> bool:
        return slice_id in self._resolved().slice_set

    def count(self) -> int:
        return len(self._resolved().slices)

    @property
    def active(self) -> str:
        return self._view._graph._current_slice

    def _record(self, slice_id):
        resolved = self._resolved()
        if slice_id not in resolved.slice_set:
            raise KeyError(f'slice {slice_id!r} is not in this view')
        record = self._view._graph._slices[slice_id]
        return set(record.nodes) & resolved.node_set, set(record.edges) & resolved.edge_set

    def nodes(self, slice_id) -> set:
        return self._record(slice_id)[0]

    def edges(self, slice_id) -> set:
        return self._record(slice_id)[1]

    def info(self, slice_id) -> dict:
        nodes, edges = self._record(slice_id)
        return {
            'nodes': nodes,
            'edges': edges,
            'attributes': self._view.attrs._row('slices', slice_id),
        }

    def union(self, slice_ids) -> dict:
        nodes: set = set()
        edges: set = set()
        for sid in slice_ids:
            held = self._record(sid)
            nodes |= held[0]
            edges |= held[1]
        return {'nodes': nodes, 'edges': edges}

    def intersect(self, slice_ids) -> dict:
        ids = list(slice_ids)
        if not ids:
            return {'nodes': set(), 'edges': set()}
        nodes, edges = self._record(ids[0])
        for sid in ids[1:]:
            held = self._record(sid)
            nodes &= held[0]
            edges &= held[1]
        return {'nodes': nodes, 'edges': edges}

    def difference(self, slice_a, slice_b) -> dict:
        left, right = self._record(slice_a), self._record(slice_b)
        return {'nodes': left[0] - right[0], 'edges': left[1] - right[1]}

    def compare(self, slice_a, slice_b, *, axis: str = 'edges', backend=None):
        from .._support.dataframe_backend import empty_dataframe, dataframe_from_rows

        if axis not in ('edges', 'nodes'):
            raise ValueError(f"axis must be 'edges' or 'nodes', got {axis!r}")
        index = 1 if axis == 'edges' else 0
        left, right = self._record(slice_a)[index], self._record(slice_b)[index]
        column = 'edge_id' if axis == 'edges' else 'node_id'
        backend = backend or self._view._graph._annotations_backend
        rows = []
        for element in sorted(left | right):
            in_left, in_right = element in left, element in right
            status = 'both' if in_left and in_right else ('a_only' if in_left else 'b_only')
            rows.append({column: element, 'status': status})
        if not rows:
            return empty_dataframe({column: 'text', 'status': 'text'}, backend=backend)
        return dataframe_from_rows(rows, backend=backend)

    def edge_frame(self, edges=None, slices=None, attrs=None, **kwargs):
        resolved = self._resolved()
        edge_list = (
            list(resolved.edge_ids)
            if edges is None
            else [e for e in edges if e in resolved.edge_set]
        )
        slice_list = (
            list(resolved.slices)
            if slices is None
            else [s for s in slices if s in resolved.slice_set]
        )
        return self._view._graph.slices.edge_frame(edge_list, slice_list, attrs, **kwargs)

    def __getattr__(self, name):
        raise AttributeError(
            f'a view has no slice operation {name!r}; the read operations are list, exists, '
            f'count, active, nodes, edges, info, union, intersect, difference, compare and '
            f'edge_frame. Call .materialize() to edit slices.'
        )

    def __repr__(self):
        return f'<view slices: {self.list()!r}>'


class _ViewLayers:
    """The layer reads of a view, restricted to its window and placements."""

    __slots__ = ('_view',)

    def __init__(self, view):
        self._view = view

    def _graph(self):
        return self._view._graph

    def list_aspects(self):
        return self._graph().layers.list_aspects()

    def aspect(self, name):
        return self._graph().layers.aspect(name)

    def list_layers(self, aspect=None, include_placeholder=False):
        return self._graph().layers.list_layers(aspect, include_placeholder)

    @property
    def window(self) -> tuple:
        """The layer coordinates in this view, in declaration order."""
        resolved = self._view._resolved()
        if resolved.layers is not None:
            return resolved.layers
        held = {key[1] for key in resolved.node_keys}
        return tuple(
            aa for aa in (tuple(x) for x in self._graph().layers._all_layers) if aa in held
        )

    def where(self, **predicates):
        from ._selection import LayerSelection

        selection = self._graph().layers.where(**predicates)
        inside = set(self.window)
        return LayerSelection(self._graph(), [aa for aa in selection.layers if aa in inside])

    def has_presence(self, u, layer_tuple) -> bool:
        return (u, tuple(layer_tuple)) in self._view._resolved().key_set

    def layer_node_set(self, layer_tuple) -> set:
        aa = tuple(layer_tuple)
        return {key[0] for key in self._view._resolved().node_keys if key[1] == aa}

    def layer_edge_set(self, layer_tuple, **kwargs) -> set:
        found = self._graph().layers.layer_edge_set(tuple(layer_tuple), **kwargs)
        return found & self._view._resolved().edge_set

    def values(self):
        return self._graph().layers.values()

    def matrix(self, name, *, nodes=None, layers=None, missing=np.nan):
        resolved = self._view._resolved()
        node_list = (
            list(resolved.node_ids)
            if nodes is None
            else [n for n in nodes if n in resolved.node_set]
        )
        window = set(self.window)
        layer_list = (
            list(self.window)
            if layers is None
            else [tuple(l) for l in layers if tuple(l) in window]
        )
        return self._graph().layers.matrix(
            name, nodes=node_list, layers=layer_list, missing=missing
        )

    def node_frame(self, nodes=None, layers=None, attrs=None, **kwargs):
        resolved = self._view._resolved()
        node_list = (
            list(resolved.node_ids)
            if nodes is None
            else [n for n in nodes if n in resolved.node_set]
        )
        window = set(self.window)
        layer_list = (
            list(self.window)
            if layers is None
            else [tuple(l) for l in layers if tuple(l) in window]
        )
        return self._graph().layers.node_frame(node_list, layer_list, attrs, **kwargs)

    def __getattr__(self, name):
        raise AttributeError(
            f'a view has no layer operation {name!r}; the read operations are list_aspects, '
            f'aspect, list_layers, window, where, has_presence, layer_node_set, '
            f'layer_edge_set, values, matrix and node_frame. Call .materialize() to edit.'
        )

    def __repr__(self):
        return f'<view layers: {list(self.window)!r}>'


# ---------------------------------------------------------------------------
# The view
# ---------------------------------------------------------------------------

_REMOVED_VIEW_NAMES = {
    'obs': "V.attrs.table('nodes')",
    'var': "V.attrs.table('edges')",
    'nodes_df': "V.attrs.table('nodes', derived=True)",
    'edges_df': "V.attrs.table('edges', derived=True)",
    'node_count': 'len(V.N)',
    'edge_count': 'len(V.E)',
    'node_ids': 'V.N.ids',
    'edge_ids': 'V.E.ids',
    'subview': 'V.view(...)',
    'X': 'V.B',
    'nx': 'V.materialize().nx',
    'ig': 'V.materialize().ig',
    'gt': 'V.materialize().gt',
    'cache': 'V.materialize().cache',
}

_MUTATORS = frozenset(
    {
        'add_nodes',
        'add_edges',
        'remove_nodes',
        'remove_edges',
        'ops',
        'history',
        'make_undirected',
        'write',
        'set_edge_coeffs',
        'remove_orphans',
        'validate',
    }
)


class GraphView:
    """A live, read-only, composable selection of a graph.

    Built by ``G.view(...)`` and narrowed by ``V.view(...)``. The membership
    rules are documented in :mod:`annnet.core._resolve`; the reading
    vocabulary is the graph's: ``N``, ``E``, ``attrs``, ``uns``, ``slices``,
    ``layers``, the named matrices, ``shape``/``supra_shape``, the traversal
    reads, ``view()``, ``materialize()`` and ``summary()``.
    """

    __slots__ = ('_graph', '_parent', '_filters', '_cache', '_snapshot')

    def __init__(
        self,
        parent,
        *,
        nodes=None,
        edges=None,
        layers=None,
        slices=None,
        predicate=None,
        boundary='closed',
    ):
        graph, _ = parent._selection_context()
        object.__setattr__(self, '_parent', parent)
        object.__setattr__(self, '_graph', graph)
        object.__setattr__(
            self,
            '_filters',
            normalize_filters(
                parent,
                nodes=nodes,
                edges=edges,
                layers=layers,
                slices=slices,
                predicate=predicate,
                boundary=boundary,
            ),
        )
        object.__setattr__(self, '_cache', None)
        object.__setattr__(self, '_snapshot', None)

    # -- resolution ---------------------------------------------------------

    def _is_live(self) -> bool:
        if self._filters.is_live():
            return True
        parent = self._parent
        return isinstance(parent, GraphView) and parent._is_live()

    def _parent_resolved(self):
        parent = self._parent
        return parent._resolved() if isinstance(parent, GraphView) else None

    def _resolved(self) -> Resolved:
        """The resolved state, cached against the graph's clocks."""
        if self._is_live():
            return resolve(self._graph, self._parent_resolved(), self._filters)
        clock = self._graph._state_clock()
        held = self._cache
        if held is not None and held[0] == clock:
            return held[1]
        resolved = resolve(self._graph, self._parent_resolved(), self._filters)
        object.__setattr__(self, '_cache', (clock, resolved))
        return resolved

    def _selection_context(self):
        """The scope protocol of :mod:`annnet.core._select`: ``(graph, resolved)``."""
        return self._graph, self._resolved()

    def _stable(self, build):
        """Run ``build(resolved)`` on one consistent resolution, or fail.

        A mutation of the parent between the resolution and the result would
        publish a table that describes no state the graph ever had, so the
        clock is compared before and after and the read is retried.
        """
        for _attempt in range(3):
            before = self._graph._state_clock()
            resolved = self._resolved()
            result = build(resolved)
            if self._graph._state_clock() == before:
                return result
        raise RuntimeError(
            'the graph changed while the view was being read; retry when it is quiet'
        )

    # -- identity ---------------------------------------------------------------

    @property
    def graph(self):
        """The root graph."""
        return self._graph

    @property
    def parent(self):
        """The graph or view this one narrows."""
        return self._parent

    @property
    def boundary(self) -> str:
        return self._filters.boundary

    @property
    def directed(self):
        return self._graph.directed

    @property
    def is_multilayer(self) -> bool:
        return self._graph.is_multilayer

    @property
    def aspects(self) -> list:
        return self._graph.aspects

    # -- the reading vocabulary -------------------------------------------------

    @property
    def N(self) -> NodeSequence:
        return NodeSequence(self)

    @property
    def E(self) -> EdgeSequence:
        return EdgeSequence(self)

    @property
    def attrs(self) -> ScopedAttrs:
        return ScopedAttrs(self)

    @property
    def uns(self):
        return _FrozenMapping(self._graph.graph_attributes)

    @property
    def slices(self) -> _ViewSlices:
        return _ViewSlices(self)

    @property
    def layers(self) -> _ViewLayers:
        return _ViewLayers(self)

    @property
    def shape(self) -> tuple[int, int]:
        resolved = self._resolved()
        return (len(resolved.node_ids), len(resolved.edge_ids))

    @property
    def supra_shape(self) -> tuple[int, int]:
        resolved = self._resolved()
        return (len(resolved.node_keys), len(resolved.edge_ids))

    @property
    def nv_supra(self) -> int:
        return len(self._resolved().node_keys)

    def supra_nodes(self) -> list:
        return list(self._resolved().node_keys)

    def __len__(self) -> int:
        return len(self._resolved().node_ids)

    def __iter__(self):
        return iter(self._resolved().node_ids)

    def __contains__(self, item) -> bool:
        resolved = self._resolved()
        if isinstance(item, str):
            return item in resolved.node_set
        if _structure.is_entity_key(item):
            return (item[0], tuple(item[1])) in resolved.key_set
        return False

    def __bool__(self) -> bool:
        return len(self._resolved().node_ids) > 0

    def has_node(self, node_id) -> bool:
        return node_id in self

    def has_edge(self, source=None, target=None, edge_id=None):
        resolved = self._resolved()
        if edge_id is not None and source is None and target is None:
            return edge_id in resolved.edge_set
        if source is not None and target is not None:
            found = [
                eid
                for eid in _structure.edges_between(self._graph, source, target)
                if eid in resolved.edge_set
            ]
            if edge_id is not None:
                return edge_id in found
            return (bool(found), found)
        raise ValueError('use has_edge(edge_id=...), has_edge(source, target) or all three')

    def at(self, node_id: str, **aspects) -> tuple:
        key = self._graph._node_layer_key(node_id, aspects)
        if key not in self._resolved().key_set:
            raise KeyError(f'{node_id!r} is not on layer {key[1]!r} in this view')
        return key

    def exists(self, node_id: str, **aspects) -> bool:
        key = self._graph._node_layer_key(node_id, aspects)
        return key in self._resolved().key_set

    def entity_kinds(self) -> dict:
        """The kind of every selected entity, by id (``'node'`` or ``'edge'``)."""
        graph = self._graph
        return {
            key[0]: _external_entity_kind(
                STORED_ENTITY_KIND[_structure.entity_ref(graph, key).kind]
            )
            for key in self._resolved().entity_keys
        }

    def degree(self, entity_id) -> int:
        """The number of selected edges touching one selected entity."""
        graph = self._graph
        resolved = self._resolved()
        try:
            key = graph._resolve_entity_key(entity_id)
        except (KeyError, ValueError, TypeError):
            return 0
        if key not in resolved.key_set:
            return 0
        return sum(1 for eid in _structure.entity_edges(graph, key) if eid in resolved.edge_set)

    def neighbors(self, entity_id) -> list:
        """The neighbours of one selected entity along selected edges."""
        return self._snapshot_graph().neighbors(entity_id)

    def out_neighbors(self, node_id) -> list:
        return self._snapshot_graph().out_neighbors(node_id)

    def in_neighbors(self, node_id) -> list:
        return self._snapshot_graph().in_neighbors(node_id)

    def successors(self, node_id) -> list:
        return self._snapshot_graph().successors(node_id)

    def predecessors(self, node_id) -> list:
        return self._snapshot_graph().predecessors(node_id)

    def incident_edges(self, nodes, direction: str = 'both') -> list:
        return self._snapshot_graph().incident_edges(nodes, direction)

    def edge_list(self) -> list:
        return self._snapshot_graph().edge_list()

    # -- matrices ----------------------------------------------------------------------

    def _snapshot_graph(self):
        """A materialized copy of the resolved state, kept against the clock.

        The named matrices and the traversal reads answer from it, so an
        algorithm asked about the view reads exactly the resolved graph and
        nothing of the parent. It is rebuilt when the parent changes.
        """
        clock = self._graph._state_clock()
        held = self._snapshot
        if held is not None and held[0] == clock and not self._is_live():
            return held[1]
        graph = self.materialize()
        object.__setattr__(self, '_snapshot', (clock, graph))
        return graph

    @property
    def B(self):
        return self._snapshot_graph().B

    @property
    def S(self):
        return self._snapshot_graph().S

    @property
    def H(self):
        return self._snapshot_graph().H

    @property
    def A(self):
        return self._snapshot_graph().A

    @property
    def L(self):
        return self._snapshot_graph().L

    @property
    def matrices(self):
        return self._snapshot_graph().matrices

    @property
    def idx(self):
        return self._snapshot_graph().idx

    # -- composition -------------------------------------------------------------------

    def view(
        self, nodes=None, edges=None, layers=None, slices=None, *, predicate=None, boundary='closed'
    ):
        """Narrow this view. Every argument intersects with what this view holds."""
        return GraphView(
            self,
            nodes=nodes,
            edges=edges,
            layers=layers,
            slices=slices,
            predicate=predicate,
            boundary=boundary,
        )

    def materialize(self, copy_attributes: bool = True):
        """Build an independent, editable graph holding exactly this selection.

        Parameters
        ----------
        copy_attributes : bool, default True
            Carry the attributes of every address over. ``False`` keeps the
            structure, the slice memberships and the aspects alone.

        Returns
        -------
        AnnNet
            A new graph; editing it changes nothing here, and editing the
            parent changes nothing there. ``uns['selection']`` records what
            was selected.
        """
        from . import _materialize

        return self._stable(
            lambda resolved: _materialize.materialize(resolved, copy_attributes=copy_attributes)
        )

    # -- inspection -----------------------------------------------------------------------

    def summary(self):
        from ._summary import summarize

        return summarize(self)

    def __repr__(self) -> str:
        resolved = self._resolved()
        filters = ', '.join(resolved.filters) or 'none'
        return (
            f'GraphView(nodes={len(resolved.node_ids)}, edges={len(resolved.edge_ids)}, '
            f'boundary={resolved.boundary!r}, filters={filters})'
        )

    # -- refusals ---------------------------------------------------------------------------

    def __setattr__(self, name, value):
        raise refuse_write('a view')

    def __getattr__(self, name):
        if name in _MUTATORS:
            raise AttributeError(
                f'a view has no {name!r}: it is read-only. Call .materialize() for an '
                f'independent graph, or apply the operation to the parent graph.'
            )
        if name in _REMOVED_VIEW_NAMES:
            raise AttributeError(f'GraphView has no {name!r}; use {_REMOVED_VIEW_NAMES[name]}')
        raise AttributeError(f'GraphView has no attribute {name!r}')

    def __dir__(self):
        return sorted(
            {
                'A',
                'B',
                'E',
                'H',
                'L',
                'N',
                'S',
                'aspects',
                'at',
                'attrs',
                'boundary',
                'degree',
                'directed',
                'edge_list',
                'entity_kinds',
                'exists',
                'graph',
                'has_edge',
                'has_node',
                'idx',
                'incident_edges',
                'in_neighbors',
                'is_multilayer',
                'layers',
                'materialize',
                'matrices',
                'neighbors',
                'nv_supra',
                'out_neighbors',
                'parent',
                'predecessors',
                'shape',
                'slices',
                'successors',
                'summary',
                'supra_nodes',
                'supra_shape',
                'uns',
                'view',
            }
        )


__all__ = ['GraphView', 'ReadOnlyViewError']
