"""The one import path every reader ends in.

A reader parses a file. Where the result *goes* is not the reader's business,
and it used to be: five of the thirteen readers accepted a ``graph=`` to merge
into, three took a slice, one took a layer, and the rest could only ever build a
new graph. Which one you got depended on who wrote the adapter.

This module holds that decision once. A reader builds a graph the way it always
did and calls :func:`deliver`, which either hands it back, or merges it into a
graph the caller already has, placing what it carries in a named slice or on a
named layer.

Merging reads the source through the structural facade and writes through the
public surface, so it works for any pair of graphs and knows nothing about the
format that produced either one.
"""

from __future__ import annotations

from typing import Any, Literal
import inspect
from functools import wraps

from ...core import _structure as S

OnConflict = Literal['error', 'skip', 'replace']

_ACTIONS = ('error', 'skip', 'replace')

# The four the contract adds to every reader. Declared once so that the wrapper,
# the synthesized signature and the documentation cannot drift apart.
_DELIVERY = ('into', 'slice', 'layer', 'on_conflict')

_DELIVERY_DOC = """
    into : AnnNet, optional
        Merge what was read into this graph and return it, instead of building
        a new one.
    slice : str, optional
        Register everything imported under this slice, creating it if needed.
    layer : tuple[str, ...] | str, optional
        Place imported entities on this layer coordinate.
    on_conflict : {'error', 'skip', 'replace'}, default 'error'
        What to do when an imported id is one the destination already holds.
"""


def delivers(reader):
    """Give one reader the shared destination contract.

    The reader keeps building a graph the way it always did and stays unaware of
    where the result goes. This adds ``into``, ``slice``, ``layer`` and
    ``on_conflict`` to its signature, takes them off the call, and routes what it
    returns through :func:`deliver`.

    A reader that already had its own ``graph=`` keeps it: that path writes into
    the destination while parsing, which is cheaper than building and merging, so
    ``into=`` is forwarded to it rather than handled here.
    """
    signature = inspect.signature(reader)
    native_graph = 'graph' in signature.parameters
    # A reader that already declares one of these means its own thing by it —
    # ``from_sbml`` places species in its ``slice`` and ``layer`` as it parses.
    # Taking those off the call would silently disable the reader's own handling,
    # so only the names it does not declare are ours.
    ours = tuple(name for name in _DELIVERY if name not in signature.parameters)

    @wraps(reader)
    def wrapper(*args, **kwargs):
        delivery = {name: kwargs.pop(name, None) for name in ours}
        on_conflict = delivery.pop('on_conflict', None) or 'error'
        into = delivery.pop('into', None)
        placement = {name: delivery.get(name) for name in ('slice', 'layer')}
        if native_graph and into is not None and kwargs.get('graph') is None:
            # The reader can write into the destination itself, which is cheaper
            # than building a second graph and merging it.
            kwargs['graph'] = into
            built = reader(*args, **kwargs)
            if placement['slice'] is None and placement['layer'] is None:
                return built
            return deliver(built, on_conflict=on_conflict, **placement)
        built = reader(*args, **kwargs)
        return deliver(built, into=into, on_conflict=on_conflict, **placement)

    _ANNOTATIONS = {
        'into': 'AnnNet | None',
        'slice': 'str | None',
        'layer': 'tuple[str, ...] | str | None',
        'on_conflict': "Literal['error', 'skip', 'replace']",
    }
    extra = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=_ANNOTATIONS[name],
        )
        for name in ours
    ]
    parameters = list(signature.parameters.values())
    variadic = [p for p in parameters if p.kind is inspect.Parameter.VAR_KEYWORD]
    fixed = [p for p in parameters if p.kind is not inspect.Parameter.VAR_KEYWORD]
    wrapper.__signature__ = signature.replace(parameters=fixed + extra + variadic)
    if wrapper.__doc__ and 'Parameters' in wrapper.__doc__ and 'into :' not in wrapper.__doc__:
        wrapper.__doc__ = wrapper.__doc__.rstrip() + '\n' + _DELIVERY_DOC
    return wrapper


class ImportConflict(ValueError):
    """An imported id collided with one the destination already holds."""


def deliver(
    built,
    *,
    into=None,
    slice: str | None = None,
    layer: tuple[str, ...] | str | None = None,
    on_conflict: OnConflict = 'error',
):
    """Return the graph a read produced, or merge it into the caller's.

    Every reader ends here, so every reader offers the same three choices.

    Parameters
    ----------
    built : AnnNet
        What the reader parsed.
    into : AnnNet, optional
        Merge into this graph and return it. When omitted, ``built`` is returned.
    slice : str, optional
        Register everything imported under this slice, creating it if needed.
    layer : tuple[str, ...] | str, optional
        Place imported entities on this layer coordinate.
    on_conflict : {"error", "skip", "replace"}
        What to do when an imported id is one the destination already holds.

    Returns
    -------
    AnnNet
    """
    if on_conflict not in _ACTIONS:
        raise ValueError(f'on_conflict must be one of {_ACTIONS}, got {on_conflict!r}')
    if into is None:
        if slice is not None or layer is not None:
            _place_in_self(built, slice=slice, layer=layer)
        return built
    return merge_into(built, into, slice=slice, layer=layer, on_conflict=on_conflict)


def _coord(graph, layer) -> tuple | None:
    """Normalize a layer argument against the aspects a graph declares."""
    if layer is None:
        return None
    if isinstance(layer, str):
        return (layer,)
    return tuple(layer)


def _place_in_self(graph, *, slice, layer) -> None:
    """Apply slice and layer placement to a freshly built graph."""
    if layer is not None:
        _relayer(graph, _coord(graph, layer))
    if slice is not None:
        _register_slice(graph, slice, S.node_ids(graph), S.edge_ids(graph))


def _relayer(graph, coord: tuple) -> None:
    """Move every entity of a graph onto one layer coordinate."""
    moves = {key: (key[0], coord) for key in S.entity_keys(graph) if key[1] != coord}
    if moves:
        graph._remap_entity_keys(moves)


def _register_slice(graph, slice_id: str, node_ids, edge_ids) -> None:
    """Create the slice if it is new and give it these members."""
    if not graph.slices.exists(slice_id):
        graph.slices.add(slice_id)
    if node_ids:
        graph.slices.add_nodes(slice_id, node_ids)
    if edge_ids:
        graph.slices.attach_edges(slice_id, [e for e in edge_ids if S.has_edge(graph, e)])


def merge_into(
    source,
    target,
    *,
    slice: str | None = None,
    layer: tuple[str, ...] | str | None = None,
    on_conflict: OnConflict = 'error',
):
    """Copy everything one graph holds into another, and return the destination.

    Entities come first, because an edge may name only entities that exist. An
    entity already present under the same key is left alone rather than
    rewritten: a merge adds, it does not overwrite what the destination knew.
    """
    coord = _coord(target, layer)
    added_nodes, added_edges = [], []

    for ref in S.iter_entities(source):
        key = (ref.id, coord) if coord is not None else ref.key
        if S.has_entity(target, key):
            continue
        target.add_nodes([ref.id], layer=key[1] if key[1] != ('_',) else None)
        added_nodes.append(ref.id)

    _entities, definitions = S.definitions_of(source)
    for definition in definitions:
        if S.has_edge(target, definition.id):
            if on_conflict == 'error':
                raise ImportConflict(
                    f'edge {definition.id!r} is already in the destination; '
                    f"pass on_conflict='skip' or 'replace'"
                )
            if on_conflict == 'skip':
                continue
            target.remove_edge(definition.id)
        _add_definition(target, definition, coord)
        added_edges.append(definition.id)

    _copy_attributes(source, target)
    if slice is not None:
        _register_slice(target, slice, added_nodes, added_edges)
    return target


def _endpoint(endpoint, coord):
    """Re-express one endpoint on the destination's layer coordinate."""
    if coord is None:
        return endpoint
    if isinstance(endpoint, tuple) and len(endpoint) == 2 and isinstance(endpoint[1], tuple):
        return (endpoint[0], coord)
    return (endpoint, coord)


def _add_definition(target, definition, coord) -> None:
    """Write one edge definition into the destination."""
    source_side = [_endpoint(e, coord) for e in sorted(definition.source, key=repr)]
    target_side = [_endpoint(e, coord) for e in sorted(definition.target, key=repr)]
    common: dict[str, Any] = {
        'edge_id': definition.id,
        'weight': definition.weight,
    }
    if definition.directed is not None:
        common['directed'] = definition.directed
    if definition.kind == S.HYPER:
        target.add_edges([{'head': source_side, 'tail': target_side, **common}])
    else:
        target.add_edges(
            [
                {
                    'source': source_side[0] if source_side else None,
                    'target': target_side[0] if target_side else None,
                    **common,
                }
            ]
        )
    if definition.coefficients:
        target.set_edge_coeffs(definition.id, dict(definition.coefficients))


def _copy_attributes(source, target) -> None:
    """Carry every attribute level across, generic and contextual alike.

    The contextual levels are one store of dicts, so this is a mapping update
    rather than six different merges into six different shapes.
    """
    for node_id, attrs in source._attr_store.node_attr_rows().items():
        clean = {k: v for k, v in attrs.items() if v is not None and k != 'node_id'}
        if clean and S.has_entity_id(target, node_id):
            target.attrs.set_node_attrs(node_id, **clean)
    for edge_id, attrs in source._attr_store.edge_attr_rows().items():
        clean = {k: v for k, v in attrs.items() if v is not None and k != 'edge_id'}
        if clean and S.has_edge(target, edge_id):
            target.attrs.set_edge_attrs(edge_id, **clean)

    for level, contents in (
        (name, getattr(source._contextual, name)) for name in _CONTEXTUAL_LEVELS
    ):
        for key, attrs in contents.items():
            target._contextual.set(level, key, dict(attrs))

    if source.graph_attributes:
        target.graph_attributes.update(dict(source.graph_attributes))


_CONTEXTUAL_LEVELS = (
    'slice_attrs',
    'edge_slice_attrs',
    'node_layer_attrs',
    'aspect_attrs',
    'layer_attrs',
    'elementary_attrs',
)
