"""The manifest an in-memory adapter returns beside the backend graph.

A backend graph holds nodes, binary edges and their attributes. The manifest
holds the rest: hyperedges with per-member coefficients, per-edge
directedness, slices, multilayer state, edge entities and the full attribute
tables. :func:`build_manifest` writes it and :func:`graph_from_manifest`
rebuilds the graph from it, whatever backend the graph went to.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._common import (
    _rows_like,
    _rows_to_df,
    binary_ends,
    slice_members,
    _attrs_to_dict,
    iter_edge_sides,
    requested_slices,
    restore_entities,
    dataframe_to_rows,
    edge_entity_record,
    endpoint_coeff_map,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    update_edge_attributes,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)

if TYPE_CHECKING:
    from ..core import AnnNet


def build_manifest(
    graph: AnnNet, *, public_only: bool = False, slice=None, slices=None
) -> dict[str, Any]:
    """Describe everything about ``graph`` that a backend graph cannot hold.

    With ``slice`` or ``slices``, the nodes and edges are those the slices hold.
    """
    requested = requested_slices(slice, slices)
    keep_nodes, keep_edges = slice_members(graph, requested)

    def _visible(attrs):
        return {k: v for k, v in attrs.items() if not public_only or not str(k).startswith('__')}

    raw_node_attrs = {
        row['node_id']: {k: v for k, v in row.items() if k != 'node_id'}
        for row in _rows_like(getattr(graph, '_node_table', None))
        if row.get('node_id') is not None
    }
    node_attrs = {
        v: _attrs_to_dict(_visible(raw_node_attrs.get(v, {})))
        for v in list(graph.N)
        if keep_nodes is None or v in keep_nodes
    }
    raw_edge_attrs = {
        row['edge_id']: {k: v for k, v in row.items() if k != 'edge_id'}
        for row in _rows_like(getattr(graph, '_edge_table', None))
        if row.get('edge_id') is not None
    }

    edge_attrs: dict = {}
    edges: dict = {}
    weights: dict = {}
    directed: dict = {}
    for eid, ref, sides in iter_edge_sides(graph):
        if keep_edges is not None and eid not in keep_edges:
            continue
        eattr = _attrs_to_dict(_visible(raw_edge_attrs.get(eid, {})))
        edge_attrs[eid] = eattr
        S, T = set(sides.source), set(sides.target)
        members = S | T
        if ref.kind != 'hyper' and len(members) == 1:
            u = next(iter(members))
            edges[eid] = (u, u, 'regular')
        elif ref.kind != 'hyper' and len(S) == 1 and len(T) == 1:
            # The order the store holds, for an undirected edge too, so the
            # round trip gives back the same source and target.
            edges[eid] = (next(iter(S)), next(iter(T)), 'regular')
        elif ref.kind != 'hyper' and len(members) == 2:
            u, v = binary_ends(S, T, ref.directed)
            edges[eid] = (u, v, 'regular')
        else:
            edges[eid] = (
                endpoint_coeff_map(eattr, '__source_attr', S),
                endpoint_coeff_map(eattr, '__target_attr', T),
                'hyper',
            )
        weights[eid] = float(ref.weight)
        directed[eid] = bool(ref.directed)

    slices_section, slice_weights = collect_slice_manifest(
        graph, requested_lids=(list(requested) if requested else None)
    )
    return {
        'edges': edges,
        'weights': weights,
        'slices': slices_section,
        'node_attrs': node_attrs,
        'edge_attrs': edge_attrs,
        'slice_weights': slice_weights,
        'edge_directed': directed,
        'edge_entities': edge_entity_record(graph),
        'manifest_version': 1,
        'multilayer': serialize_multilayer_manifest(
            graph,
            table_to_rows=dataframe_to_rows,
            serialize_edge_layers=serialize_edge_layers,
        ),
    }


def _bare(x):
    return x[0] if isinstance(x, tuple) else x


def graph_from_manifest(manifest: dict[str, Any]) -> AnnNet:
    """Rebuild a graph from a manifest that :func:`build_manifest` wrote."""
    from ..core import AnnNet

    H = AnnNet()
    # Aspects first, so multilayer endpoint keys resolve while edges are added.
    # Presence, attributes and edge layers come back with the multilayer state.
    multilayer = manifest.get('multilayer', {}) or {}
    aspects = multilayer.get('aspects')
    if aspects and tuple(H.aspects) != tuple(aspects):
        H.layers.set_aspects(list(aspects), multilayer.get('elem_layers') or None)

    edges = manifest.get('edges', {}) or {}
    weights = manifest.get('weights', {}) or {}
    node_ids = {_bare(v) for v in (manifest.get('node_attrs', {}) or {})}
    for definition in edges.values():
        if definition[-1] == 'regular':
            node_ids.update((_bare(definition[0]), _bare(definition[1])))
        elif definition[-1] == 'hyper':
            for side in definition[:2]:
                if isinstance(side, dict):
                    node_ids.update(_bare(u) for u in side)
    if node_ids:
        H._add_nodes_bulk([{'node_id': v} for v in node_ids])

    directed = manifest.get('edge_directed', {}) or {}
    binary: list = []
    hyper: list = []
    for eid, definition in edges.items():
        is_dir = bool(directed.get(eid, True))
        weight = weights.get(eid, 1.0)
        if definition[-1] == 'regular':
            binary.append(
                {
                    'source': definition[0],
                    'target': definition[1],
                    'edge_id': eid,
                    'edge_directed': is_dir,
                    'weight': weight,
                }
            )
            continue
        head_map, tail_map = definition[0], definition[1]
        if not (isinstance(head_map, dict) and isinstance(tail_map, dict)):
            continue
        # A coefficient map is stored only when it carries a non-unit value.
        attrs = {}
        if any(c != 1.0 for c in head_map.values()) or any(c != 1.0 for c in tail_map.values()):
            attrs = {
                '__source_attr': {u: {'__value': float(c)} for u, c in head_map.items()},
                '__target_attr': {v: {'__value': float(c)} for v, c in tail_map.items()},
            }
        spec = {'edge_id': eid, 'edge_directed': is_dir, 'weight': weight, 'attributes': attrs}
        if is_dir:
            spec.update(head=list(head_map), tail=list(tail_map))
        else:
            spec['members'] = list(set(head_map) | set(tail_map))
        hyper.append(spec)
    if binary:
        H._add_edges_bulk(binary, default_edge_directed=True)
    if hyper:
        H.add_hyperedges_bulk(hyper)

    restore_slice_manifest(
        H, manifest.get('slices', {}) or {}, manifest.get('slice_weights', {}) or {}
    )
    restore_multilayer_manifest(
        H,
        manifest.get('multilayer', {}),
        rows_to_table=_rows_to_df,
        deserialize_edge_layers=deserialize_edge_layers,
    )
    restore_entities(H, manifest.get('edge_entities'))

    node_updates = {v: a for v, a in (manifest.get('node_attrs', {}) or {}).items() if a}
    if node_updates:
        H.attrs.update('nodes', node_updates)
    edge_updates = {e: a for e, a in (manifest.get('edge_attrs', {}) or {}).items() if a}
    if edge_updates:
        update_edge_attributes(H, edge_updates)
    return H


__all__ = ['build_manifest', 'graph_from_manifest']
