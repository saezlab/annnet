"""Non-overlapping local benchmark dimensions."""

from __future__ import annotations

from .backends import backend_operation_dimensions
from .accessors import accessor_dimensions
from .primitives import primitive_dimensions
from .annotations import annotation_update_dimensions

GROUPS = ('primitives', 'annotation_updates', 'backend_operations', 'accessors')


def extra_dimensions(
    scale,
    *,
    backend: str = 'auto',
    samples: int = 3,
    groups: tuple[str, ...] | None = None,
    max_vertices: int = 2_500,
    max_edges: int = 10_000,
    max_accessor_repeats: int = 5,
) -> list[dict]:
    """Run local non-overlapping benchmark dimensions as flat records."""
    selected = groups or GROUPS
    recs: list[dict] = []
    kwargs = {
        'backend': backend,
        'samples': samples,
        'max_vertices': max_vertices,
        'max_edges': max_edges,
        'max_accessor_repeats': max_accessor_repeats,
    }
    if 'primitives' in selected:
        recs += primitive_dimensions(scale, **kwargs)
    if 'annotation_updates' in selected:
        recs += annotation_update_dimensions(scale, **kwargs)
    if 'backend_operations' in selected:
        recs += backend_operation_dimensions(scale, **kwargs)
    if 'accessors' in selected:
        recs += accessor_dimensions(scale, **kwargs)
    return recs
