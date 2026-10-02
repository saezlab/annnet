"""Build an independent graph from one resolved selection.

``V.materialize()``, every subgraph operation of ``G.ops`` and the layer
subgraphs of ``G.layers`` arrive here with a
:class:`~annnet.core._resolve.Resolved` state and leave with a new
:class:`~annnet.core.graph.AnnNet` that holds exactly the selected placements,
edge entities and structural edges, in the parent's order. All these paths use
the supplied resolution without recomputing membership.

The copy preserves:

- the topology: every kept edge with its full endpoint set, its declared
  direction and weight, its explicit coefficients, its multilayer kind and
  layer assignment, and its flexible-direction policy;
- the aspect declarations, their labels in declaration order and their
  ordered flags — an aspect is a declaration, so all of it comes along and no
  coordinate the parent accepts becomes invalid;
- the slice registry restricted to the slices in scope, with memberships cut
  to the selected elements; the default slice always exists;
- with ``copy_attributes=True`` (the default), the attributes at all eight
  addresses, cut to the selected keys: node and edge columns, slice, aspect,
  layer, elementary-layer, edge-slice and node-layer rows;
- the graph metadata (``uns``), deep-copied where it can be, which is where
  the provenance records live;
- the attached value backings, **by reference**: an attached array is a
  reader over the parent's measurements, and the copy reads the same array
  through the same index maps. Detach and re-attach to own one.

Placements and membership come from the resolved selection. Slice weight
overrides remain separate attribute rows unless ``edge_weights`` supplies
replacement stored weights.
"""

from __future__ import annotations

import copy as _copy
import time

from . import _build
from ._aspects import ORDERED_KEY
from ._records import SliceRecord


def _constructor_aspects(graph):
    if graph._aspects == ('_',):
        return None
    return {aspect: list(graph._layers.get(aspect, ())) for aspect in graph._aspects}


def materialize(resolved, *, copy_attributes: bool = True, edge_weights=None, record=True):
    """Return a new graph holding exactly ``resolved``.

    Parameters
    ----------
    resolved : Resolved
        The state one view resolved to.
    copy_attributes : bool, default True
        Carry the attributes at every address over.
    edge_weights : Mapping[str, float], optional
        Weights that replace the stored weight of the named edges in the copy
        (what ``ops.subgraph_from_slice(resolve_slice_weights=True)`` asks
        for). Explicit coefficients are kept as they are.
    record : bool, default True
        Record the selected state under ``uns['selection']``.
    """
    graph = resolved.graph
    graph_class = type(graph)
    new = graph_class(directed=graph.directed, aspects=_constructor_aspects(graph))

    # The topology, in the parent's order.
    _build.install_structure(
        new,
        store=graph._store.select(
            list(resolved.entity_keys), list(resolved.edge_ids), weights=dict(edge_weights or {})
        ),
    )
    new.node_aligned = graph.node_aligned
    new._next_edge_id = graph._next_edge_id
    new.layers._all_layers = tuple(tuple(aa) for aa in graph.layers._all_layers)

    # Slices in scope, memberships cut to the selection. A default slice is
    # part of every graph, so one is kept even when the scope left it out.
    node_set = resolved.node_set
    edge_set = resolved.edge_set
    records = {}
    for slice_id in resolved.slices:
        held = graph._slices.get(slice_id)
        if held is None:
            continue
        records[slice_id] = SliceRecord(
            set(held.nodes) & node_set, set(held.edges) & edge_set, dict(held.attributes)
        )
    default = graph._default_slice
    if default not in records:
        records[default] = SliceRecord()
    current = graph._current_slice if graph._current_slice in records else default
    _build.install_slices(new, records, default=default, current=current)

    # The ordered flags are declarations, so they travel with the aspects.
    contextual = graph._contextual
    aspect_rows = {}
    for aspect in new._aspects:
        held = contextual.aspect_attrs.get(aspect, {})
        if copy_attributes:
            row = dict(held)
        else:
            row = {ORDERED_KEY: held[ORDERED_KEY]} if ORDERED_KEY in held else {}
        if row:
            aspect_rows[aspect] = row
    new._contextual.replace('aspect_attrs', aspect_rows)

    if copy_attributes:
        _copy_attributes(graph, new, resolved)

    # Metadata, and the provenance records inside it.
    try:
        new.graph_attributes = _copy.deepcopy(graph.graph_attributes)
    except Exception:  # noqa: BLE001 - an uncopyable value is carried by reference
        new.graph_attributes = dict(graph.graph_attributes)
    if record:
        new.graph_attributes['selection'] = _selected_state(resolved)

    # Attached value backings are referenced, never copied (see the module docstring).
    backings = getattr(graph, '_node_layer_backings', None)
    if backings:
        new._node_layer_backings = list(backings)
    passthrough = getattr(graph, '_layer_table_passthrough', None)
    if passthrough is not None and copy_attributes:
        new._layer_table_passthrough = passthrough

    new._history_enabled = graph._history_enabled
    new._history = []
    new._version = 0
    new._snapshots = []
    new._history_clock0 = time.perf_counter_ns()
    new._install_history_hooks()
    return new


def _copy_attributes(graph, new, resolved) -> None:
    """Cut every address to the selection and install it on the copy."""
    node_ids = list(resolved.node_ids)
    edge_ids = list(resolved.edge_ids)
    contextual = graph._contextual
    target = new._contextual

    new._attr_store.load_node_rows(
        {'node_id': node_id, **attrs}
        for node_id, attrs in graph._attr_store.node_attr_rows(node_ids).items()
    )
    new._attr_store.load_edge_rows(
        {'edge_id': edge_id, **attrs}
        for edge_id, attrs in graph._attr_store.edge_attr_rows(edge_ids).items()
    )

    slice_set = set(new._slices)
    target.replace(
        'slice_attrs',
        {sid: attrs for sid, attrs in contextual.slice_attrs.items() if sid in slice_set},
    )
    edge_set = resolved.edge_set
    target.replace(
        'edge_slice_attrs',
        {
            key: attrs
            for key, attrs in contextual.edge_slice_attrs.items()
            if key[0] in slice_set and key[1] in edge_set
        },
    )
    key_set = resolved.key_set
    target.replace(
        'node_layer_attrs',
        {key: attrs for key, attrs in contextual.node_layer_attrs.items() if key in key_set},
    )
    if resolved.layers is None:
        occurring = {key[1] for key in resolved.node_keys}
        layer_rows = {
            aa: attrs
            for aa, attrs in contextual.layer_attrs.items()
            if aa in occurring or not occurring
        }
    else:
        window = set(resolved.layers)
        layer_rows = {aa: attrs for aa, attrs in contextual.layer_attrs.items() if aa in window}
    target.replace('layer_attrs', layer_rows)
    # Elementary layers are declarations of the aspect registry, which the copy
    # carries whole, so their attributes come whole as well.
    target.replace('elementary_attrs', dict(contextual.elementary_attrs))


def _selected_state(resolved) -> dict:
    """A small, serializable record of what was selected."""
    return {
        'nodes': len(resolved.node_ids),
        'supra_nodes': len(resolved.node_keys),
        'edges': len(resolved.edge_ids),
        'layers': None if resolved.layers is None else [list(aa) for aa in resolved.layers],
        'slices': list(resolved.slices),
        'boundary': resolved.boundary,
        'filters': list(resolved.filters),
        'expanded_nodes': sorted({key[0] for key in resolved.expanded_keys}),
        'expanded_edges': sorted(resolved.expanded_edges),
        'structure_version': resolved.graph._store.structure_version,
    }


__all__ = ['materialize']
