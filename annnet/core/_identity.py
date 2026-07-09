"""The one place identity is resolved."""

from __future__ import annotations

import warnings


def is_explicit_entity_key(value) -> bool:
    """Return True when ``value`` is an explicit ``(vertex_id, layer_coord)`` key."""
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], tuple)
    )


def placeholder_layer_coord(g) -> tuple:
    """Canonical placeholder coordinate for the current aspect rank."""
    if g._aspects == ('_',):
        return ('_',)
    return tuple('_' for _ in g._aspects)


def ensure_placeholder_layers_declared(g) -> tuple:
    """Ensure the placeholder value ``'_'`` exists for every declared aspect."""
    coord = placeholder_layer_coord(g)
    if g._aspects == ('_',):
        g._layers.setdefault('_', {'_'})
        return coord
    for aspect in g._aspects:
        g._layers.setdefault(aspect, set()).add('_')
    g._rebuild_all_layers_cache()
    return coord


def warn_placeholder_vertex_assignment(g, vertex_ids, *, context: str) -> None:
    """Warn that vertices were assigned to the placeholder layer tuple."""
    coord = placeholder_layer_coord(g)
    if isinstance(vertex_ids, str):
        warnings.warn(
            f'{context}: vertex {vertex_ids!r} was assigned to placeholder layer '
            f'{coord!r}. Pass layer= to place it explicitly.',
            UserWarning,
            stacklevel=3,
        )
        return
    vids = list(vertex_ids)
    if not vids:
        return
    sample = ', '.join(repr(v) for v in vids[:3])
    suffix = '' if len(vids) <= 3 else ', ...'
    warnings.warn(
        f'{context}: {len(vids)} vertices were assigned to placeholder layer '
        f'{coord!r} ({sample}{suffix}). Pass layer= to place them explicitly.',
        UserWarning,
        stacklevel=3,
    )


def make_layer_coord(g, layer_spec) -> tuple:
    """Normalize any layer specification to a canonical ``layer_coord`` tuple."""
    is_flat = g._aspects == ('_',)

    if layer_spec is None:
        if is_flat:
            return ('_',)
        raise ValueError(
            f'layer= must be specified for multilayer graphs with aspects {g._aspects}. '
            'Pass a dict {aspect: value}, a tuple of values, or a bare string (single-aspect).'
        )

    if isinstance(layer_spec, str):
        if len(g._aspects) != 1:
            raise ValueError(
                f'String layer spec {layer_spec!r} is only valid for single-aspect graphs. '
                f'Got aspects {g._aspects}. Pass a dict or tuple.'
            )
        coord = (layer_spec,)
    elif isinstance(layer_spec, dict):
        missing = [asp for asp in g._aspects if asp not in layer_spec]
        if missing:
            raise ValueError(
                f'Missing aspects {missing} in layer spec. Required: {list(g._aspects)}'
            )
        coord = tuple(layer_spec[asp] for asp in g._aspects)
    elif isinstance(layer_spec, tuple):
        coord = layer_spec
    else:
        raise TypeError(
            f'layer_spec must be None, str, dict, or tuple; got {type(layer_spec).__name__!r}'
        )

    if len(coord) != len(g._aspects):
        raise ValueError(f'Layer coord length {len(coord)} != number of aspects {len(g._aspects)}')
    for asp, val in zip(g._aspects, coord, strict=False):
        if val not in g._layers[asp]:
            raise ValueError(
                f'Layer value {val!r} not declared for aspect {asp!r}. '
                f'Valid: {sorted(g._layers[asp])}'
            )
    return coord


def resolve_ekey(g, vid_or_key) -> tuple:
    """Resolve a vertex identifier to an internal ``(vid, layer_coord)`` key."""
    if isinstance(vid_or_key, str):
        if g._aspects == ('_',):
            return (vid_or_key, ('_',))
        matches = []
        for ekey in g._vid_to_ekeys.get(vid_or_key, ()):
            rec = g._entities.get(ekey)
            if rec is None:
                continue
            matches.append((rec.row_idx, ekey))
        if len(matches) == 1:
            return matches[0][1]
        if len(matches) > 1:
            choices = [ekey for _, ekey in sorted(matches)]
            raise ValueError(
                f'Ambiguous bare vertex_id {vid_or_key!r} in multilayer graph; '
                f'use an explicit (vertex_id, layer_coord) tuple. Choices: {choices!r}'
            )
        return (vid_or_key, placeholder_layer_coord(g))
    if is_explicit_entity_key(vid_or_key):
        vid, layer_coord = vid_or_key
        return (vid, make_layer_coord(g, layer_coord))
    raise TypeError(
        f'vertex_id must be str or (str, tuple[str,...]), got {type(vid_or_key).__name__!r}'
    )


def resolve_vertex_insert_coord(g, layer_spec, *, vertex_ids=None, context='add_vertex') -> tuple:
    """Resolve layer placement for vertex insertion, with placeholder fallback."""
    if layer_spec is not None:
        return make_layer_coord(g, layer_spec)
    if g._aspects == ('_',):
        return ('_',)
    coord = ensure_placeholder_layers_declared(g)
    if vertex_ids is not None:
        warn_placeholder_vertex_assignment(g, vertex_ids, context=context)
    return coord


def entity_row(g, vid) -> int:
    """Return incidence-matrix row index for a vertex (resolves bare vid)."""
    return g._entities[resolve_ekey(g, vid)].row_idx


# ---------------------------------------------------------------------------
# Slice membership translation (slices store bare vids)
# ---------------------------------------------------------------------------


def endpoint_slice_vertex_ids(g, endpoint) -> set[str]:
    """Map an endpoint identity to the bare vertex ids used by slice membership."""
    if endpoint is None:
        return set()
    if isinstance(endpoint, str):
        return {endpoint}
    if is_explicit_entity_key(endpoint):
        return {endpoint[0]}
    if isinstance(endpoint, (set, frozenset, list, tuple)):
        out: set[str] = set()
        for member in endpoint:
            out.update(endpoint_slice_vertex_ids(g, member))
        return out
    return set()


def slice_contains_endpoint(g, slice_vertices, endpoint) -> bool:
    """Return True when the slice contains every bare vertex id for an endpoint."""
    vids = endpoint_slice_vertex_ids(g, endpoint)
    return bool(vids) and vids <= slice_vertices


def add_endpoint_to_slice_vertices(g, slice_vertices, endpoint) -> None:
    """Add the bare vertex ids represented by an endpoint to a slice membership set."""
    slice_vertices.update(endpoint_slice_vertex_ids(g, endpoint))
