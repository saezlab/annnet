"""The derived tables: everything a graph can say about one address, as a frame.

A *stored* table holds what was written at an address. A *derived* table adds
what the graph computes — the endpoints, kind, direction and weight of an edge;
the counts of a slice; the coordinate of a layer — and takes filters and joins
as explicit arguments. Both come through one entry point,
``G.attrs.table(address, derived=True, ...)``, so a caller never has to know a
second spelling for the richer frame.

Every builder here takes an optional *scope*, the resolved state of a view, and
restricts its rows to it **before** rendering: the rows are chosen first, then
the derived columns are computed for those rows alone, then the backend
converts. A view's table therefore costs the view, not the graph.

Canonical structural edge columns: ``kind`` is ``binary``, ``hyper`` or
``node_edge``; ``directed`` is a separate boolean; ``weight`` is the declared
weight; ``ml_kind`` is the multilayer role. ``source`` and ``target`` are single
node ids or null — a hyperedge's participants are in ``head``/``tail``/
``members`` as lists, never pipe-joined into a column that advertises an id.
The ``incidences`` layout gives one row per endpoint instead, with the
participant's identity, layer, role and coefficient, for joins that need them.
"""

from __future__ import annotations

import itertools

from . import _structure
from ._aspects import ORDERED_KEY
from ._records import as_endpoint
from ._stored_kinds import STORED_EDGE_KIND, STORED_ENTITY_KIND
from .._support.dataframe_backend import (
    empty_dataframe,
    dataframe_to_rows,
    dataframe_from_rows,
    dataframe_from_columns,
)

_ROLE_NAMES = {1: 'source', -1: 'target', 0: 'member'}

_EDGE_SCHEMA = {
    'edge_id': 'text',
    'kind': 'text',
    'directed': 'bool',
    'weight': 'float',
    'ml_kind': 'text',
    'source': 'text',
    'target': 'text',
    'src_layer': 'text',
    'dst_layer': 'text',
    'head': 'list_text',
    'tail': 'list_text',
    'members': 'list_text',
}

_INCIDENCE_SCHEMA = {
    'edge_id': 'text',
    'position': 'int',
    'entity_id': 'text',
    'entity_kind': 'text',
    'layer': 'list_text',
    'layer_id': 'text',
    'role': 'text',
    'coefficient': 'float',
    'kind': 'text',
    'directed': 'bool',
}


def _side_identity(side):
    """Return one side as ``(sorted node ids, the layer they share or None)``."""
    if not side:
        return [], None
    if len(side) == 1:
        endpoint = as_endpoint(next(iter(side)))
        return [endpoint.node_id], endpoint.layer
    endpoints = sorted(
        (as_endpoint(item) for item in side), key=lambda e: (e.node_id, e.layer or ())
    )
    layers = {endpoint.layer for endpoint in endpoints}
    one = layers.pop() if len(layers) == 1 else None
    return [endpoint.node_id for endpoint in endpoints], one


def _layer_display(graph, coordinate):
    if coordinate is None or graph._aspects == ('_',):
        return None
    return graph.layers.layer_tuple_to_id(coordinate)


def _scoped_edge_refs(graph, scope):
    """The edge records in column order, lazily, so a limit stops the walk."""
    refs = _structure.iter_edges(graph)
    if scope is None:
        return refs
    wanted = scope.edge_set
    return (ref for ref in refs if ref.id in wanted)


def _first(items, limit):
    """The first ``limit`` items, or all of them when ``limit`` is None."""
    if limit is None:
        return list(items)
    return list(itertools.islice(items, limit))


def _slice_override_rows(graph, slice_id, edge_ids):
    """``edge_id -> {slice_<name>: value}`` for one slice, from the contextual store."""
    held = graph._contextual.edge_slice_attrs
    wanted = set(edge_ids)
    out = {}
    for (sid, eid), attrs in held.items():
        if sid == slice_id and eid in wanted and attrs:
            out[eid] = {
                f'slice_{name}': value for name, value in attrs.items() if value is not None
            }
    return out


def edges_table(
    graph,
    *,
    scope=None,
    backend=None,
    slice=None,
    include_directed=True,
    include_weight=True,
    resolved_weight=True,
    layer=None,
    in_slice=None,
    include_hyper=True,
    include_binary=True,
    layout=None,
    limit=None,
):
    """One row per edge (``layout='edges'``) or per endpoint (``'incidences'``).

    Parameters
    ----------
    slice : str, optional
        Join this slice's per-edge overrides onto every row as ``slice_*``
        columns. This does not filter; see ``in_slice``.
    in_slice : str, optional
        Keep only the edges that are members of this slice.
    layer : tuple[str, ...], optional
        Keep only the edges of this layer coordinate.
    include_hyper, include_binary : bool
        Keep hyperedges / binary edges.
    include_directed, include_weight, resolved_weight : bool
        Whether to add the ``directed``, ``weight`` and ``effective_weight``
        columns. ``effective_weight`` is the ``slice`` override when one is
        joined and set, otherwise the declared weight.
    layout : {"edges", "incidences"}, optional
    limit : int, optional
        Build only the first ``limit`` rows that pass the filters. The walk
        stops there, so a preview costs its rows and not the graph.
    """
    layout = 'edges' if layout is None else layout
    if layout not in ('edges', 'incidences'):
        raise ValueError(f"layout must be 'edges' or 'incidences', got {layout!r}")
    if slice is not None and slice not in graph._slices:
        raise KeyError(f'unknown slice {slice!r}')
    if in_slice is not None and in_slice not in graph._slices:
        raise KeyError(f'unknown slice {in_slice!r}')
    if scope is not None:
        for sid in (slice, in_slice):
            if sid is not None and not scope.holds('slices', sid):
                raise KeyError(f'slice {sid!r} is outside this view')

    refs = _scoped_edge_refs(graph, scope)
    if not include_hyper:
        refs = (ref for ref in refs if ref.kind != _structure.HYPER)
    if not include_binary:
        refs = (ref for ref in refs if ref.kind == _structure.HYPER)
    if layer is not None:
        coordinate = tuple(layer) if not isinstance(layer, str) else (layer,)
        graph.layers._validate_layer_tuple(coordinate)
        keep_layer = graph.layers.layer_edge_set(coordinate)
        refs = (ref for ref in refs if ref.id in keep_layer)
    if in_slice is not None:
        keep_slice = set(graph._slices[in_slice].edges)
        refs = (ref for ref in refs if ref.id in keep_slice)
    refs = _first(refs, limit)

    if layout == 'incidences':
        return _incidence_rows(graph, refs, backend=backend)

    if not refs:
        schema = dict(_EDGE_SCHEMA)
        if not include_directed:
            schema.pop('directed')
        if not include_weight:
            schema.pop('weight')
        if resolved_weight:
            schema['effective_weight'] = 'float'
        return empty_dataframe(schema, backend=backend)

    eids = [str(ref.id) for ref in refs]
    kinds = [STORED_EDGE_KIND[ref.kind] for ref in refs]
    ml_kinds = [ref.ml_kind for ref in refs]
    weights = [ref.declared_weight for ref in refs]
    dirs = [bool(ref.directed) for ref in refs]

    src, tgt, head, tail, members = [], [], [], [], []
    src_layer, dst_layer = [], []
    for ref in refs:
        sides = _structure.edge_sides(graph, ref.id)
        source_ids, source_layer = _side_identity(sides.source)
        target_ids, target_layer = _side_identity(sides.target)
        src_layer.append(_layer_display(graph, source_layer))
        dst_layer.append(_layer_display(graph, target_layer))
        if ref.kind == _structure.HYPER:
            src.append(None)
            tgt.append(None)
            if target_ids:
                head.append(list(source_ids))
                tail.append(list(target_ids))
                members.append(None)
            else:
                head.append(None)
                tail.append(None)
                members.append(list(source_ids))
        else:
            src.append(source_ids[0] if source_ids else None)
            tgt.append(target_ids[0] if target_ids else None)
            head.append(None)
            tail.append(None)
            members.append(None)

    out: dict[str, list] = {'edge_id': eids, 'kind': kinds}
    if include_directed:
        out['directed'] = dirs
    if include_weight:
        out['weight'] = weights
    out.update(
        {
            'ml_kind': ml_kinds,
            'source': src,
            'target': tgt,
            'src_layer': src_layer,
            'dst_layer': dst_layer,
            'head': head,
            'tail': tail,
            'members': members,
        }
    )

    attrs_map = graph._attr_store.edge_attr_rows(eids)
    override_map = _slice_override_rows(graph, slice, eids) if slice is not None else {}
    for source_map in (attrs_map, override_map):
        names: list[str] = []
        seen: set[str] = set()
        for eid in eids:
            for name in source_map.get(eid, ()):
                if name not in seen and name not in out:
                    seen.add(name)
                    names.append(name)
        for name in names:
            out[name] = [source_map.get(eid, {}).get(name) for eid in eids]

    if resolved_weight:
        override = out.get('slice_weight')
        out['effective_weight'] = [
            weights[index] if override is None or override[index] is None else override[index]
            for index in range(len(eids))
        ]
    return dataframe_from_columns(out, backend=backend)


def _incidence_rows(graph, refs, *, backend=None):
    """One row per endpoint entry of every edge, in member order."""
    store = graph._store
    rows = []
    flat = graph._aspects == ('_',)
    entity_kinds = store.entity_kind
    for ref in refs:
        slot = store.edge_slot(ref.id)
        held = store.members(slot)
        kind = STORED_EDGE_KIND[ref.kind]
        directed = bool(ref.directed)
        for position, (entity_slot, coefficient, role) in enumerate(
            zip(
                held.entities.tolist(), held.coefficients.tolist(), held.roles.tolist(), strict=True
            )
        ):
            key = store.entity_key(int(entity_slot))
            if key is None:
                continue
            entity_kind = STORED_ENTITY_KIND[
                _structure._SLOT_ENTITY_KIND[int(entity_kinds[entity_slot])]
            ]
            rows.append(
                {
                    'edge_id': str(ref.id),
                    'position': position,
                    'entity_id': key[0],
                    'entity_kind': 'edge' if entity_kind == 'edge_entity' else entity_kind,
                    'layer': None if flat else list(key[1]),
                    'layer_id': None if flat else graph.layers.layer_tuple_to_id(key[1]),
                    'role': _ROLE_NAMES.get(int(role), 'member'),
                    'coefficient': float(coefficient),
                    'kind': kind,
                    'directed': directed,
                }
            )
    if not rows:
        return empty_dataframe(_INCIDENCE_SCHEMA, backend=backend)
    return dataframe_from_rows(rows, backend=backend)


def nodes_table(graph, *, scope=None, backend=None, limit=None):
    """One row per node: the id, its attributes, and its layers when layered."""
    ids = _first(scope.node_ids if scope is not None else _structure.node_ids(graph), limit)
    attrs = graph._attr_store.node_attr_rows(ids)
    layered = graph._aspects != ('_',)
    placements: dict = {}
    if layered:
        keys = scope.node_keys if scope is not None else _structure.node_keys(graph)
        wanted = set(ids)
        for node_id, coordinate in keys:
            if node_id in wanted:
                placements.setdefault(node_id, []).append(
                    graph.layers.layer_tuple_to_id(coordinate)
                )
    rows = []
    for node_id in ids:
        row = {'node_id': node_id, **attrs.get(node_id, {})}
        if layered:
            row['layers'] = placements.get(node_id, [])
        rows.append(row)
    if not rows:
        schema = {'node_id': 'text'}
        if layered:
            schema['layers'] = 'list_text'
        return empty_dataframe(schema, backend=backend)
    return dataframe_from_rows(rows, backend=backend)


def slices_table(graph, *, scope=None, backend=None, limit=None):
    """One row per slice: id, member counts and attributes."""
    ids = list(graph._slices)
    if scope is not None:
        ids = [sid for sid in ids if scope.holds('slices', sid)]
    ids = _first(ids, limit)
    rows = []
    for sid in ids:
        record = graph._slices[sid]
        nodes = record.nodes
        edges = record.edges
        if scope is not None:
            nodes = set(nodes) & scope.node_set
            edges = set(edges) & scope.edge_set
        attrs = graph._contextual.slice_attrs.get(sid, {})
        rows.append(
            {
                'slice_id': sid,
                'n_nodes': len(nodes),
                'n_edges': len(edges),
                **{k: v for k, v in attrs.items() if v is not None},
            }
        )
    if not rows:
        return empty_dataframe(
            {'slice_id': 'text', 'n_nodes': 'int', 'n_edges': 'int'}, backend=backend
        )
    return dataframe_from_rows(rows, backend=backend)


def aspects_table(graph, *, scope=None, backend=None, limit=None):
    """One row per aspect: name, elementary layers, whether ordered, attributes."""
    if graph._aspects == ('_',):
        return empty_dataframe(
            {'aspect': 'text', 'elementary_layers': 'list_text', 'ordered': 'bool'}, backend=backend
        )
    rows = []
    for aspect in _first(graph._aspects, limit):
        attrs = dict(graph._contextual.aspect_attrs.get(aspect, {}))
        ordered = bool(attrs.pop(ORDERED_KEY, False))
        rows.append(
            {
                'aspect': aspect,
                'elementary_layers': [
                    label for label in graph._layers.get(aspect, ()) if label != '_'
                ],
                'ordered': ordered,
                **{k: v for k, v in attrs.items() if v is not None},
            }
        )
    return dataframe_from_rows(rows, backend=backend)


def layers_table(graph, *, scope=None, backend=None, limit=None):
    """One row per declared layer coordinate, with its attributes and elementary attrs.

    ``layer`` is the coordinate and the join key; ``coordinate_id`` is a display
    id and never a key. The per-aspect columns spread the coordinate, and an
    elementary layer's attributes arrive prefixed ``{aspect}__{name}``.
    """
    if graph._aspects == ('_',) or not graph.layers._all_layers:
        return empty_dataframe({'layer': 'list_text', 'coordinate_id': 'text'}, backend=backend)
    elementary = graph._contextual.elementary_attrs
    layer_attrs = graph._contextual.layer_attrs
    coordinates = (tuple(aa) for aa in graph.layers._all_layers)
    if scope is not None:
        coordinates = (aa for aa in coordinates if scope.holds('layers', aa))
    rows = []
    for aa in _first(coordinates, limit):
        row = {'layer': list(aa), 'coordinate_id': graph.layers.layer_tuple_to_id(aa)}
        for index, aspect in enumerate(graph._aspects):
            row[aspect] = aa[index]
        row.update({k: v for k, v in layer_attrs.get(aa, {}).items() if v is not None})
        for index, aspect in enumerate(graph._aspects):
            for name, value in elementary.get((aspect, aa[index]), {}).items():
                if value is not None:
                    row[f'{aspect}__{name}'] = value
        rows.append(row)
    if not rows:
        return empty_dataframe({'layer': 'list_text', 'coordinate_id': 'text'}, backend=backend)
    return dataframe_from_rows(rows, backend=backend)


def node_layers_table(graph, *, scope=None, backend=None, limit=None):
    """One row per placement: node, layer, display id, attributes."""
    keys = _first(scope.node_keys if scope is not None else _structure.node_keys(graph), limit)
    flat = graph._aspects == ('_',)
    held = graph._contextual.node_layer_attrs
    rows = []
    for node_id, coordinate in keys:
        attrs = held.get((node_id, coordinate), {})
        rows.append(
            {
                'node_id': node_id,
                'layer': list(coordinate),
                'layer_id': None if flat else graph.layers.layer_tuple_to_id(coordinate),
                **{k: v for k, v in attrs.items() if v is not None},
            }
        )
    if not rows:
        return empty_dataframe(
            {'node_id': 'text', 'layer': 'list_text', 'layer_id': 'text'}, backend=backend
        )
    return dataframe_from_rows(rows, backend=backend)


def edge_slices_table(graph, *, scope=None, domain=(), backend=None, limit=None):
    """One row per (slice, edge) membership or override, with the effective weight.

    ``domain`` is the address's key list, as the attribute API computes it.
    """
    rows = []
    for slice_id, edge_id in _first(domain, limit):
        held = graph._contextual.edge_slice_attrs.get((slice_id, edge_id), {})
        member = edge_id in graph._slices[slice_id].edges
        weight = held.get('weight')
        rows.append(
            {
                'slice_id': slice_id,
                'edge_id': edge_id,
                'member': member,
                **{k: v for k, v in held.items() if v is not None},
                'effective_weight': float(weight)
                if weight is not None
                else float(_structure.edge_ref(graph, edge_id).weight),
            }
        )
    if not rows:
        return empty_dataframe(
            {'slice_id': 'text', 'edge_id': 'text', 'member': 'bool', 'effective_weight': 'float'},
            backend=backend,
        )
    return dataframe_from_rows(rows, backend=backend)


def elementary_layers_table(graph, *, scope=None, domain=(), backend=None, limit=None):
    """One row per elementary layer: aspect, label, legacy display id, attributes.

    ``domain`` is the address's key list, as the attribute API computes it.
    """
    held = graph._contextual.elementary_attrs
    rows = []
    for aspect, label in _first(domain, limit):
        rows.append(
            {
                'aspect': aspect,
                'elementary_layer': label,
                'layer_id': f'{aspect}_{label}',
                **{k: v for k, v in held.get((aspect, label), {}).items() if v is not None},
            }
        )
    if not rows:
        return empty_dataframe(
            {'aspect': 'text', 'elementary_layer': 'text', 'layer_id': 'text'}, backend=backend
        )
    return dataframe_from_rows(rows, backend=backend)


_BUILDERS = {
    'nodes': nodes_table,
    'edges': edges_table,
    'slices': slices_table,
    'aspects': aspects_table,
    'layers': layers_table,
    'node_layers': node_layers_table,
    'edge_slices': edge_slices_table,
    'elementary_layers': elementary_layers_table,
}


def derived_table(
    graph, address, *, scope=None, domain=None, layout=None, backend=None, limit=None, **query
):
    """Dispatch to the builder of one address, checking the query arguments.

    ``domain`` is the key list of the two addresses whose rows are not a
    structural registry (``edge_slices``, ``elementary_layers``); the
    attribute API computes it and hands it over. ``limit`` reaches the builder,
    which builds that many rows and no more.
    """
    builder = _BUILDERS[address]
    if address != 'edges':
        if layout is not None:
            raise TypeError(f'layout= applies to the edges table, not {address!r}')
        if query:
            raise TypeError(
                f'the derived {address} table takes no query arguments; got {sorted(query)!r}'
            )
        if address in ('edge_slices', 'elementary_layers'):
            return builder(graph, scope=scope, domain=domain or (), backend=backend, limit=limit)
        return builder(graph, scope=scope, backend=backend, limit=limit)
    allowed = {
        'slice',
        'include_directed',
        'include_weight',
        'resolved_weight',
        'layer',
        'in_slice',
        'include_hyper',
        'include_binary',
    }
    unknown = sorted(set(query) - allowed)
    if unknown:
        raise TypeError(f'unknown edge table argument(s) {unknown!r}; allowed: {sorted(allowed)!r}')
    return builder(graph, scope=scope, backend=backend, layout=layout, limit=limit, **query)


def derived_columns(graph, address) -> list[tuple[str, str]]:
    """The computed columns a derived table adds, as ``(name, type)`` pairs."""
    layered = graph._aspects != ('_',)
    if address == 'edges':
        return [
            ('kind', 'str'),
            ('directed', 'bool'),
            ('weight', 'float'),
            ('ml_kind', 'str'),
            ('source', 'str'),
            ('target', 'str'),
            ('src_layer', 'str'),
            ('dst_layer', 'str'),
            ('head', 'list'),
            ('tail', 'list'),
            ('members', 'list'),
            ('effective_weight', 'float'),
        ]
    if address == 'nodes':
        return [('layers', 'list')] if layered else []
    if address == 'slices':
        return [('n_nodes', 'int'), ('n_edges', 'int')]
    if address == 'aspects':
        return [('elementary_layers', 'list'), ('ordered', 'bool')]
    if address == 'layers':
        return [('coordinate_id', 'str')] + [(aspect, 'str') for aspect in graph.aspects]
    if address == 'node_layers':
        return [('layer_id', 'str')]
    if address == 'edge_slices':
        return [('member', 'bool'), ('effective_weight', 'float')]
    if address == 'elementary_layers':
        return [('layer_id', 'str')]
    return []


def rows_of(frame) -> list[dict]:
    """The rows of any frame, as dictionaries (a convenience for callers)."""
    return dataframe_to_rows(frame)


__all__ = ['derived_columns', 'derived_table', 'edges_table', 'nodes_table']
