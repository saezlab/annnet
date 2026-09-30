"""Membership and boundary resolution: what a set of filters selects of a graph.

This module owns the one resolver. Given a graph, an optional parent
resolution (a nested view narrows its parent) and a normalized set of
filters, :func:`resolve` answers with a :class:`Resolved`: the selected node
placements, distinct node ids, structural edges and entity rows, in the
graph's own order, plus the layer window and the slices in scope. Every
consumer — the view's sequences and tables, the matrices, materialization,
the subgraph operations — reads that one answer.

The rules:

- **closed** (the default): a node, placement, layer or slice filter keeps an
  edge only when its complete endpoint set is selected. A hyperedge is kept
  whole or dropped; a self-loop has one unique endpoint; a half-edge is judged
  by its nonempty side; a placeholder is not a structural edge. Selected
  isolated nodes stay. An edge-only filter keeps the edges and their endpoint
  entities, not every unrelated node. A slice is a membership set, not a
  request to induce edges.
- **open**: the edges touching the selected entities are kept and expanded to
  their full endpoints — one hop, not a neighbourhood — with edge and slice
  restrictions still enforced and never beyond the parent view.
- an **edge entity** endpoint needs its backing edge: an edge-only filter
  brings it along (referential closure), a closed node filter keeps the
  dependent edge only when the backing edge is kept, and an open filter brings
  the backing edge when the candidates hold it and drops the dependent edge
  otherwise. Nothing substitutes a node or leaves a dangling reference.

Nothing here knows about the view object: :mod:`annnet.core._Views` builds
the filters through :func:`normalize_filters` and calls :func:`resolve`.
"""

from __future__ import annotations

from typing import Any, NamedTuple
from collections.abc import Mapping

from . import _structure
from ._select import (
    EdgeSequence,
    NodeSequence,
    RowSelection,
    SliceSelection,
    as_mask,
    call_predicate,
)
from ._selection import LayerSelection

_EDGE_ENTITY_CODE = 1  # annnet.core._store.EDGE_ENTITY


class Resolved:
    """What one set of filters selects, as stable keys, in the graph's order.

    Attributes
    ----------
    node_keys : tuple
        The selected node placements, in row order.
    node_ids : tuple
        The distinct node ids, in ``N`` order.
    edge_ids : tuple
        The selected structural edges, in ``E`` order.
    entity_keys : tuple
        Every selected entity — placements and the edge entities of selected
        edges — in row order. The rows of ``V.B``.
    layers : tuple | None
        The layer window, or ``None`` for every layer.
    slices : tuple
        The slices in scope.
    boundary : str
        The boundary rule the filters were resolved under.
    filters : tuple[str, ...]
        The names of the filters that were active.
    expanded_keys, expanded_edges : frozenset
        What ``boundary='open'`` added beyond the predicate matches.
    """

    __slots__ = (
        'graph',
        'node_keys',
        'node_ids',
        'edge_ids',
        'entity_keys',
        'layers',
        'slices',
        'boundary',
        'filters',
        'expanded_keys',
        'expanded_edges',
        'node_set',
        'edge_set',
        'key_set',
        'layer_set',
        'slice_set',
        'elementary_set',
    )

    def __init__(
        self,
        graph,
        *,
        node_keys,
        edge_ids,
        entity_keys,
        layers,
        slices,
        boundary='closed',
        filters=(),
        expanded_keys=(),
        expanded_edges=(),
    ):
        self.graph = graph
        self.node_keys = tuple(node_keys)
        seen: dict = {}
        for node_id, _layer in self.node_keys:
            seen.setdefault(node_id, None)
        self.node_ids = tuple(seen)
        self.edge_ids = tuple(edge_ids)
        self.entity_keys = tuple(entity_keys)
        self.layers = None if layers is None else tuple(layers)
        self.slices = tuple(slices)
        self.boundary = boundary
        self.filters = tuple(filters)
        self.expanded_keys = frozenset(expanded_keys)
        self.expanded_edges = frozenset(expanded_edges)
        self.node_set = frozenset(self.node_ids)
        self.edge_set = frozenset(self.edge_ids)
        self.key_set = frozenset(self.entity_keys)
        self.layer_set = None if self.layers is None else frozenset(self.layers)
        self.slice_set = frozenset(self.slices)
        held = {key[1] for key in self.node_keys}
        self.elementary_set = frozenset(
            (aspect, label)
            for coordinate in held
            for aspect, label in zip(graph._aspects, coordinate, strict=False)
        )

    def holds(self, axis: str, key) -> bool:
        """Whether one key of one axis is inside this resolved state."""
        if axis == 'nodes':
            return key in self.node_set
        if axis == 'edges':
            return key in self.edge_set
        if axis == 'slices':
            return key in self.slice_set
        if axis == 'layers':
            coordinate = tuple(key)
            if self.layer_set is None:
                return any(k[1] == coordinate for k in self.node_keys)
            return coordinate in self.layer_set
        if axis == 'node_layers':
            return key in self.key_set
        if axis == 'elementary_layers':
            return self.layer_set is None or tuple(key) in self.elementary_set
        return False

    @property
    def supra_count(self) -> int:
        return len(self.node_keys)

    def __repr__(self) -> str:
        return (
            f'Resolved(nodes={len(self.node_ids)}, placements={len(self.node_keys)}, '
            f'edges={len(self.edge_ids)}, boundary={self.boundary!r})'
        )


class Filters(NamedTuple):
    """The normalized filters of one view. Built by :func:`normalize_filters`."""

    nodes: Any
    edges: Any
    layers: Any
    slices: Any
    predicate: Any
    boundary: str

    def is_live(self) -> bool:
        """Whether any part must be re-evaluated on every read."""
        for spec in (self.nodes, self.edges):
            if spec is None:
                continue
            if spec[0] == 'callable':
                return True
            if spec[0] == 'seq' and spec[1]._is_live():
                return True
        return self.predicate is not None

    def constrains_nodes(self) -> bool:
        return (
            self.nodes is not None
            or self.predicate is not None
            or self.layers is not None
            or self.slices is not None
        )

    def names(self) -> tuple:
        return tuple(
            name
            for name, spec in (
                ('nodes', self.nodes),
                ('edges', self.edges),
                ('layers', self.layers),
                ('slices', self.slices),
                ('predicate', self.predicate),
            )
            if spec is not None
        )


# ---------------------------------------------------------------------------
# Normalizing what a caller passed
# ---------------------------------------------------------------------------


def _root_of(owner):
    return owner._selection_context()[0]


def _normalize_nodes(owner, nodes):
    """Return the node filter in one of its normalized shapes, or None."""
    graph = _root_of(owner)
    if nodes is None:
        return None
    if isinstance(nodes, NodeSequence):
        if nodes.graph is not graph:
            raise ValueError('the node selection belongs to a different graph')
        return ('seq', nodes)
    if isinstance(nodes, EdgeSequence):
        raise TypeError('nodes= takes nodes; pass the edge selection as edges=')
    if isinstance(nodes, RowSelection):
        if nodes.graph is not graph:
            raise ValueError('the row selection belongs to a different graph')
        if nodes.address == 'node_layers':
            return ('rows', nodes)
        if nodes.address == 'nodes':
            return ('seq', nodes.project('nodes'))
        raise TypeError(
            f'{nodes.address!r} rows do not name nodes; project them with .project("nodes") '
            f'if they can be, or pass them to the argument they address'
        )
    if callable(nodes) and not isinstance(nodes, (str, bytes, tuple, list, set, frozenset)):
        return ('callable', nodes)
    mask = as_mask(nodes)
    if mask is not None:
        return ('ids', frozenset(NodeSequence(owner)._masked(mask).ids))
    if isinstance(nodes, str):
        items = [nodes]
    elif _structure.is_entity_key(nodes):
        items = [nodes]
    else:
        items = list(nodes)
    ids: set = set()
    keys: set = set()
    for item in items:
        if isinstance(item, str):
            if not graph.has_node(item):
                raise KeyError(f'unknown node id {item!r}')
            ids.add(item)
        elif _structure.is_entity_key(item):
            node_id, coordinate = item
            key = (node_id, tuple(coordinate))
            if not graph._has_node_layer(key):
                raise KeyError(f'{node_id!r} is not placed on layer {coordinate!r}')
            keys.add(key)
        else:
            raise TypeError(
                f'a node is named by an id or a (node_id, layer) key, not {type(item).__name__}'
            )
    return ('explicit', frozenset(ids), frozenset(keys))


def _normalize_edges(owner, edges):
    graph = _root_of(owner)
    if edges is None:
        return None
    if isinstance(edges, EdgeSequence):
        if edges.graph is not graph:
            raise ValueError('the edge selection belongs to a different graph')
        return ('seq', edges)
    if isinstance(edges, NodeSequence):
        raise TypeError('edges= takes edges; pass the node selection as nodes=')
    if isinstance(edges, RowSelection):
        if edges.graph is not graph:
            raise ValueError('the row selection belongs to a different graph')
        if edges.address == 'edges':
            return ('seq', edges.project('edges'))
        if edges.address == 'edge_slices':
            raise TypeError(
                'edge_slices rows carry a slice: project them explicitly with '
                ".project('edges') (or .project('slices')) so the slice meaning is not "
                'silently discarded'
            )
        raise TypeError(f'{edges.address!r} rows do not name edges')
    if callable(edges) and not isinstance(edges, (str, bytes, tuple, list, set, frozenset)):
        return ('callable', edges)
    mask = as_mask(edges)
    if mask is not None:
        return ('ids', frozenset(EdgeSequence(owner)._masked(mask).ids))
    items = [edges] if isinstance(edges, str) else list(edges)
    found = set()
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f'an edge is named by its id, not {type(item).__name__}')
        if not _structure.has_edge(graph, item):
            raise KeyError(f'unknown edge id {item!r}')
        found.add(item)
    return ('ids', frozenset(found))


def _normalize_layers(owner, layers):
    graph = _root_of(owner)
    if layers is None:
        return None
    if graph._aspects == ('_',):
        raise ValueError('no aspects are configured; layers= needs a multilayer graph')
    if isinstance(layers, LayerSelection):
        if layers._G is not graph:
            raise ValueError('the layer selection belongs to a different graph')
        return ('coords', frozenset(layers.layers))
    if isinstance(layers, RowSelection):
        if layers.graph is not graph:
            raise ValueError('the row selection belongs to a different graph')
        if layers.address != 'layers':
            raise TypeError(f'{layers.address!r} rows do not name layers')
        return ('rows', layers)
    if isinstance(layers, str):
        items = [(layers,)]
    elif isinstance(layers, tuple) and all(isinstance(part, str) for part in layers):
        items = [layers]
    else:
        items = [(item,) if isinstance(item, str) else tuple(item) for item in layers]
    found = set()
    for coordinate in items:
        graph.layers._validate_layer_tuple(coordinate)
        found.add(tuple(coordinate))
    return ('coords', frozenset(found))


def _normalize_slices(owner, slices):
    graph = _root_of(owner)
    if slices is None:
        return None
    if isinstance(slices, SliceSelection):
        return ('ids', tuple(slices.ids))
    if isinstance(slices, Mapping) and set(slices) >= {'nodes', 'edges'}:
        return ('membership', frozenset(slices['nodes']), frozenset(slices['edges']))
    if isinstance(slices, RowSelection):
        if slices.address == 'slices':
            return ('ids', tuple(slices.keys))
        raise TypeError(f'{slices.address!r} rows do not name slices')
    items = [slices] if isinstance(slices, str) else list(slices)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f'a slice is named by its id, not {type(item).__name__}')
        if item not in graph._slices:
            raise KeyError(f'unknown slice {item!r}')
    return ('ids', tuple(dict.fromkeys(items)))


def normalize_filters(
    owner, *, nodes=None, edges=None, layers=None, slices=None, predicate=None, boundary='closed'
) -> Filters:
    """Check and normalize what a caller passed to ``view(...)``.

    ``owner`` is the graph or view the filters narrow; explicit ids are
    validated against the root graph, and a mask binds to the owner's axis.
    """
    if boundary not in ('closed', 'open'):
        raise ValueError(f"boundary must be 'closed' or 'open', got {boundary!r}")
    if predicate is not None and not callable(predicate):
        raise TypeError('predicate= takes a callable over node ids')
    return Filters(
        nodes=_normalize_nodes(owner, nodes),
        edges=_normalize_edges(owner, edges),
        layers=_normalize_layers(owner, layers),
        slices=_normalize_slices(owner, slices),
        predicate=predicate,
        boundary=boundary,
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def endpoint_keys(graph, edge_id) -> tuple:
    """Every endpoint entity key of one edge, whether or not the graph is flat."""
    store = graph._store
    slot = store.edge_slot(edge_id)
    if slot is None:
        return ()
    sides = store.endpoints(slot)
    return tuple(sides.source | sides.target)


def resolve(graph, parent: Resolved | None, filters: Filters) -> Resolved:
    """Resolve one set of filters against a graph, inside ``parent`` when given.

    The cost is meant to follow the selection, not the graph: a filter that
    names nodes, placements or slice members starts from those and reaches the
    candidate edges through the store's incident-edge index, and only a filter
    that has to look at every element (a callable, or none at all) walks the
    universe. Row and column order is slot order, which is also the parent's
    order, so a set of keys is put in order by sorting on its slots.
    """
    store = graph._store
    entity_kind = store.entity_kind

    # The closure below asks the same two questions of the same edges and
    # keys several times over; each is answered once per resolution.
    entity_memo: dict = {}
    endpoint_memo: dict = {}
    member_start, member_len, member_ent = store.member_start, store.member_len, store.member_ent
    entity_key = store._entity_key

    def is_edge_entity(key) -> bool:
        verdict = entity_memo.get(key)
        if verdict is None:
            slot = store.entity_slot(key)
            verdict = entity_memo[key] = (
                slot is not None and int(entity_kind[slot]) == _EDGE_ENTITY_CODE
            )
        return verdict

    def endpoints_of(eid) -> tuple:
        keys = endpoint_memo.get(eid)
        if keys is None:
            slot = store.edge_slot(eid)
            if slot is None:
                keys = ()
            else:
                start = int(member_start[slot])
                stop = start + int(member_len[slot])
                keys = tuple(dict.fromkeys(entity_key[e] for e in member_ent[start:stop].tolist()))
            endpoint_memo[eid] = keys
        return keys

    # -- the universe: the parent, or everything the graph holds ---------------------
    if parent is not None:
        u_keys = parent.node_keys
        u_edges = parent.edge_ids
        u_slices = parent.slices
        u_layers = parent.layer_set
        u_key_set = parent.key_set
        u_edge_set = parent.edge_set
    else:
        u_keys = None
        u_edges = None
        u_slices = tuple(graph._slices)
        u_layers = None
        u_key_set = None
        u_edge_set = None

    def in_universe_key(key) -> bool:
        if u_key_set is not None:
            return key in u_key_set
        slot = store.entity_slot(key)
        return slot is not None and int(entity_kind[slot]) != _EDGE_ENTITY_CODE

    def in_universe_edge(eid) -> bool:
        if u_edge_set is not None:
            return eid in u_edge_set
        return _structure.carries_structure(graph, eid)

    def all_keys():
        return u_keys if u_keys is not None else _structure.node_keys(graph)

    def all_edges():
        return u_edges if u_edges is not None else _structure.edge_ids(graph)

    def keys_of_ids(ids) -> set:
        found = set()
        for node_id in ids:
            for key in store.entity_keys_of_id(node_id):
                if in_universe_key(key):
                    found.add(key)
        return found

    def ordered_keys(keys) -> list:
        return sorted(keys, key=store.entity_slot)

    def ordered_edges(ids) -> list:
        return sorted(ids, key=store.edge_slot)

    # -- layers ---------------------------------------------------------------
    layer_window = u_layers
    if filters.layers is not None:
        kind = filters.layers[0]
        wanted = frozenset(filters.layers[1].keys) if kind == 'rows' else filters.layers[1]
        layer_window = wanted if u_layers is None else (wanted & u_layers)

    # ``seed`` is the placement set a concrete filter names, or None while no
    # filter has named one; ``candidates`` is the edge set likewise.
    seed: set | None = None
    candidates: set | None = None

    def narrow_seed(keys: set) -> None:
        nonlocal seed
        seed = keys if seed is None else (seed & keys)

    def narrow_candidates(ids: set) -> None:
        nonlocal candidates
        ids = {eid for eid in ids if in_universe_edge(eid)}
        candidates = ids if candidates is None else (candidates & ids)

    # -- slices ------------------------------------------------------------------
    scope_slices = u_slices
    if filters.slices is not None:
        kind = filters.slices[0]
        if kind == 'ids':
            chosen = tuple(sid for sid in filters.slices[1] if sid in u_slices)
            member_nodes: set = set()
            member_edges: set = set()
            for sid in chosen:
                record = graph._slices.get(sid)
                if record is None:
                    continue
                member_nodes.update(record.nodes)
                member_edges.update(record.edges)
            narrow_candidates(member_edges)
            touched = set()
            for eid in candidates or ():
                for key in endpoints_of(eid):
                    touched.add(key[0])
            narrow_seed(keys_of_ids(member_nodes | touched))
            scope_slices = chosen
        else:
            _kind, member_nodes, member_edges = filters.slices
            narrow_candidates(member_edges)
            narrow_seed(keys_of_ids(member_nodes))

    # -- nodes -----------------------------------------------------------------------
    if filters.nodes is not None:
        spec = filters.nodes
        kind = spec[0]
        if kind == 'explicit':
            ids, keys = spec[1], spec[2]
            narrow_seed(keys_of_ids(ids) | {key for key in keys if in_universe_key(key)})
        elif kind == 'ids':
            narrow_seed(keys_of_ids(spec[1]))
        elif kind == 'seq':
            narrow_seed(keys_of_ids(spec[1].ids))
        elif kind == 'rows':
            narrow_seed({key for key in spec[1].keys if in_universe_key(key)})
        elif kind == 'callable':
            verdict: dict = {}
            fn = spec[1]
            pool = all_keys() if seed is None else seed
            narrow_seed(
                {key for key in pool if verdict.setdefault(key[0], call_predicate(fn, key[0]))}
            )
    if filters.predicate is not None:
        verdict = {}
        fn = filters.predicate
        pool = all_keys() if seed is None else seed
        narrow_seed({key for key in pool if verdict.setdefault(key[0], call_predicate(fn, key[0]))})

    if layer_window is not None:
        pool = all_keys() if seed is None else seed
        narrow_seed({key for key in pool if key[1] in layer_window})

    placements = list(all_keys()) if seed is None else ordered_keys(seed)

    # -- edges -------------------------------------------------------------------------
    if filters.edges is not None:
        spec = filters.edges
        kind = spec[0]
        if kind == 'ids':
            narrow_candidates(set(spec[1]))
        elif kind == 'seq':
            narrow_candidates(set(spec[1].ids))
        elif kind == 'callable':
            fn = spec[1]
            pool = all_edges() if candidates is None else candidates
            narrow_candidates({eid for eid in pool if call_predicate(fn, eid)})

    # -- closure ------------------------------------------------------------------------
    selected = set(placements)
    constrained = filters.constrains_nodes()
    expanded_keys: set = set()
    expanded_edges: set = set()

    # What the explicit filters allow, before the node constraint narrows it:
    # the open boundary reaches a backing edge through this.
    explicit = candidates

    def is_candidate(eid) -> bool:
        return in_universe_edge(eid) if explicit is None else eid in explicit

    if constrained:
        # Every edge that can be kept touches a selected placement, and the
        # store indexes those; nothing else needs to be looked at. An edge
        # entity endpoint is reached through its backing edge below.
        incident: set = set()
        for key in placements:
            slot = store.entity_slot(key)
            for edge_slot in store._entity_edges.get(slot, ()):
                incident.add(store.edge_id(edge_slot))
        narrow_candidates(incident)

    if candidates is None:
        edge_candidates = list(all_edges())
    else:
        edge_candidates = ordered_edges(candidates)

    if not constrained:
        # No node constraint: the edge filter (if any) decides, and the edges
        # bring their endpoints. An edge entity endpoint brings its backing edge.
        kept = set(edge_candidates)
        changed = True
        while changed:
            changed = False
            for eid in list(kept):
                for key in endpoints_of(eid):
                    if is_edge_entity(key) and key[0] not in kept and in_universe_edge(key[0]):
                        kept.add(key[0])
                        expanded_edges.add(key[0])
                        changed = True
        if filters.edges is None:
            node_keys = placements
        else:
            touched = set()
            for eid in kept:
                for key in endpoints_of(eid):
                    if not is_edge_entity(key):
                        touched.add(key)
            node_keys = [key for key in placements if key in touched]
    elif filters.boundary == 'closed':
        kept = set()
        pending = list(edge_candidates)
        changed = True
        while changed:
            changed = False
            still = []
            for eid in pending:
                ok = True
                for key in endpoints_of(eid):
                    if is_edge_entity(key):
                        if key[0] not in kept:
                            ok = False
                            break
                    elif key not in selected:
                        ok = False
                        break
                if ok:
                    kept.add(eid)
                    changed = True
                else:
                    still.append(eid)
            pending = still
        node_keys = placements
    else:  # open
        kept = set()
        for eid in edge_candidates:
            keys = endpoints_of(eid)
            if not keys:
                continue
            if any((not is_edge_entity(key)) and key in selected for key in keys):
                # Every endpoint must be inside the parent universe, or the
                # expansion would leave the parent view.
                if all(is_edge_entity(key) or in_universe_key(key) for key in keys):
                    kept.add(eid)
        # Referential closure for edge entities: a backing edge outside the
        # candidates cannot be represented, so the dependent edge is dropped.
        changed = True
        while changed:
            changed = False
            for eid in list(kept):
                for key in endpoints_of(eid):
                    if is_edge_entity(key) and key[0] not in kept:
                        if is_candidate(key[0]):
                            kept.add(key[0])
                            expanded_edges.add(key[0])
                        else:
                            kept.discard(eid)
                        changed = True
                        break
        touched = set()
        for eid in kept:
            for key in endpoints_of(eid):
                if not is_edge_entity(key):
                    touched.add(key)
        extra = touched - selected
        expanded_keys = extra
        node_keys = ordered_keys(selected | extra)

    if not expanded_edges and len(kept) == len(edge_candidates):
        edge_ids = edge_candidates
    else:
        edge_ids = ordered_edges(kept)
    node_key_set = set(node_keys)
    # The rows of B: the placements and the edge entities of the kept edges.
    entity_set = set(node_key_set)
    for eid in edge_ids:
        for key in store.entity_keys_of_id(eid):
            if is_edge_entity(key):
                entity_set.add(key)
    entity_keys = node_keys if len(entity_set) == len(node_key_set) else ordered_keys(entity_set)
    layers = (
        None
        if layer_window is None
        else tuple(aa for aa in (tuple(x) for x in graph.layers._all_layers) if aa in layer_window)
    )
    return Resolved(
        graph,
        node_keys=node_keys,
        edge_ids=edge_ids,
        entity_keys=entity_keys,
        layers=layers,
        slices=scope_slices,
        boundary=filters.boundary,
        filters=filters.names(),
        expanded_keys=expanded_keys,
        expanded_edges=expanded_edges,
    )


def resolve_explicit(graph, *, node_keys, edge_ids, boundary='open') -> Resolved:
    """Resolve an explicit placement set and edge set, as the layer algebra hands them over.

    ``boundary='open'`` brings the full endpoints of the named edges along,
    which is what a layer window with ``include_inter`` or ``include_coupling``
    reaches; ``'closed'`` keeps only the named edges whose endpoints are all
    named. Both go through :func:`resolve`, so the result is what a view of
    the same selection would give.
    """
    owner = graph
    filters = normalize_filters(
        owner,
        nodes=[(node_id, tuple(coordinate)) for node_id, coordinate in node_keys],
        edges=list(edge_ids),
        boundary=boundary,
    )
    return resolve(graph, None, filters)


__all__ = [
    'Filters',
    'Resolved',
    'endpoint_keys',
    'normalize_filters',
    'resolve',
    'resolve_explicit',
]
