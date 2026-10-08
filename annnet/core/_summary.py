"""A structured summary of a graph or a view, built without a frame or a matrix.

``G.summary()`` and ``V.summary()`` answer the first questions of an
exploration — how big is this, what aspects and slices does it declare, what
attributes are there to filter on, what kinds of edges does it hold — from the
counters and registries the graph already keeps. Nothing here builds a
dataframe, a matrix, a Cartesian coordinate product or reads a numerical
backing.

The edge-kind and direction tallies are the one computed part: one vectorized
pass over the edge arrays on a graph, one pass over the selected edges on a
view. They are marked as computed in the result.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import _structure
from ._stored_kinds import STORED_EDGE_KIND
from ._attribute_api import ADDRESSES


class Summary(Mapping):
    """The summary as a mapping, with a bounded readable repr.

    Keys: ``kind`` (``'graph'`` or ``'view'``), ``nodes``, ``supra_nodes``,
    ``edges``, ``edge_entities``, ``directed`` (the default), ``aspects``
    (``{name: {'labels': [...], 'ordered': bool}}``), ``layers`` (the number
    of occurring coordinates), ``slices`` (``{id: {'nodes': n, 'edges': n}}``),
    ``attrs`` (``{address: [field, ...]}``), ``edge_kinds`` (computed,
    ``{'binary': n, 'hyper': n, 'node_edge': n}``), ``directed_edges`` and
    ``undirected_edges`` (computed), ``computed`` (the keys that were tallied
    rather than read). A view adds ``boundary``, ``filters`` and ``expanded``.
    """

    __slots__ = ('_held',)

    def __init__(self, held: dict):
        self._held = held

    def __getitem__(self, key):
        return self._held[key]

    def __iter__(self):
        return iter(self._held)

    def __len__(self):
        return len(self._held)

    def __repr__(self) -> str:
        h = self._held
        lines = [
            f'{"AnnNet" if h["kind"] == "graph" else "GraphView"} summary: '
            f'{h["nodes"]} node(s), {h["edges"]} edge(s), {h["supra_nodes"]} placement(s)'
            f'{", " + str(h["edge_entities"]) + " edge entit" + ("y" if h["edge_entities"] == 1 else "ies") if h["edge_entities"] else ""}',
            f'  directed default: {h["directed"]}',
        ]
        if h['kind'] == 'view':
            lines.append(
                f'  boundary: {h["boundary"]}; filters: {", ".join(h["filters"]) or "none"}'
            )
            if h['expanded']['nodes'] or h['expanded']['edges']:
                lines.append(
                    f'  expanded by the open boundary: {h["expanded"]["nodes"]} node(s), '
                    f'{h["expanded"]["edges"]} edge(s)'
                )
        if h['aspects']:
            shown = []
            for name, spec in list(h['aspects'].items())[:6]:
                labels = spec['labels']
                text = ', '.join(labels[:6]) + (', …' if len(labels) > 6 else '')
                shown.append(f'{name}{" (ordered)" if spec["ordered"] else ""}: [{text}]')
            lines.append(f'  aspects ({len(h["aspects"])}): ' + '; '.join(shown))
            lines.append(f'  layers occurring: {h["layers"]}')
        if h['slices']:
            items = list(h['slices'].items())
            shown_slices = ', '.join(
                f'{sid} ({spec["nodes"]}n/{spec["edges"]}e)' for sid, spec in items[:6]
            )
            if len(items) > 6:
                shown_slices += f', … {len(items) - 6} more'
            lines.append(f'  slices ({len(items)}): {shown_slices}')
        fields = [
            f'{address}: {names[:8]}{"…" if len(names) > 8 else ""}'
            for address, names in h['attrs'].items()
            if names
        ]
        lines.append('  attrs: ' + ('; '.join(fields) if fields else 'none'))
        kinds = ', '.join(f'{kind}={count}' for kind, count in h['edge_kinds'].items() if count)
        lines.append(
            f'  edges (computed): {kinds or "none"}; directed={h["directed_edges"]}, '
            f'undirected={h["undirected_edges"]}'
        )
        return '\n'.join(lines)

    def to_dict(self) -> dict:
        return dict(self._held)


def _aspects(graph) -> dict:
    if graph._aspects == ('_',):
        return {}
    out = {}
    for name in graph._aspects:
        aspect = graph.layers.aspect(name)
        out[name] = {'labels': list(aspect.values), 'ordered': aspect.ordered}
    return out


def _edge_tallies_graph(graph) -> tuple[dict, int, int]:
    store = graph._store
    slots = store.live_edge_slots()
    names = tuple(
        STORED_EDGE_KIND[_structure._SLOT_EDGE_KIND[code]]
        for code in sorted(_structure._SLOT_EDGE_KIND)
    )
    kinds = {'binary': 0, 'hyper': 0, 'node_edge': 0}
    if not slots.size:
        return kinds, 0, 0
    kind_column = store.edge_kind_column(names)[slots]
    directed_column = store.edge_directed_column()[slots]
    structural = kind_column != 'edge_placeholder'
    for name in kinds:
        kinds[name] = int(np.count_nonzero(kind_column == name))
    directed = int(np.count_nonzero(directed_column[structural]))
    undirected = int(np.count_nonzero(structural)) - directed
    return kinds, directed, undirected


def _edge_tallies_view(graph, edge_ids) -> tuple[dict, int, int]:
    kinds = {'binary': 0, 'hyper': 0, 'node_edge': 0}
    directed = 0
    for edge_id in edge_ids:
        ref = _structure.edge_ref(graph, edge_id)
        kinds[STORED_EDGE_KIND[ref.kind]] = kinds.get(STORED_EDGE_KIND[ref.kind], 0) + 1
        if ref.directed:
            directed += 1
    return kinds, directed, len(edge_ids) - directed


def summarize(target) -> Summary:
    """Summarize a graph or a view. See :class:`Summary`.

    A view is anything that answers ``_selection_context()`` with a resolved
    membership; the graph answers with ``None``.
    """
    graph, resolved = target._selection_context()
    if resolved is not None:
        attrs = target.attrs
        edge_kinds, directed, undirected = _edge_tallies_view(graph, resolved.edge_ids)
        occurring = {key[1] for key in resolved.node_keys}
        slices = {}
        for sid in resolved.slices:
            record = graph._slices[sid]
            slices[sid] = {
                'nodes': len(set(record.nodes) & resolved.node_set),
                'edges': len(set(record.edges) & resolved.edge_set),
            }
        active = list(resolved.filters)
        held = {
            'kind': 'view',
            'nodes': len(resolved.node_ids),
            'supra_nodes': len(resolved.node_keys),
            'edges': len(resolved.edge_ids),
            'edge_entities': len(resolved.entity_keys) - len(resolved.node_keys),
            'directed': graph.directed,
            'aspects': _aspects(graph),
            'layers': len(occurring) if graph._aspects != ('_',) else 0,
            'slices': slices,
            'attrs': {address: attrs._fields(address) for address in ADDRESSES},
            'edge_kinds': edge_kinds,
            'directed_edges': directed,
            'undirected_edges': undirected,
            'boundary': resolved.boundary,
            'filters': active,
            'expanded': {
                'nodes': len({key[0] for key in resolved.expanded_keys}),
                'edges': len(resolved.expanded_edges),
            },
            'computed': ['edge_kinds', 'directed_edges', 'undirected_edges', 'layers', 'slices'],
        }
        return Summary(held)

    graph = target
    attrs = graph.attrs
    edge_kinds, directed, undirected = _edge_tallies_graph(graph)
    store = graph._store
    if graph._aspects != ('_',):
        layers_occurring = len({key[1] for key in _structure.node_keys(graph)})
    else:
        layers_occurring = 0
    held = {
        'kind': 'graph',
        'nodes': store.node_count,
        'supra_nodes': store.node_layer_count,
        'edges': _structure.edge_count(graph),
        'edge_entities': store.edge_entity_count,
        'directed': graph.directed,
        'aspects': _aspects(graph),
        'layers': layers_occurring,
        'slices': {
            sid: {'nodes': len(record.nodes), 'edges': len(record.edges)}
            for sid, record in graph._slices.items()
        },
        'attrs': {address: attrs._fields(address) for address in ADDRESSES},
        'edge_kinds': edge_kinds,
        'directed_edges': directed,
        'undirected_edges': undirected,
        'computed': ['edge_kinds', 'directed_edges', 'undirected_edges', 'layers'],
    }
    return Summary(held)


__all__ = ['Summary', 'summarize']
