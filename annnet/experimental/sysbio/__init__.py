"""Joining measurements to a network, and projecting results back out.

Becomes its own package, on top of ``vocabulary``. The loop it serves:

    AnnData -> attach -> AnnNet -> method -> AnnNet (annotated) -> AnnData
"""

from __future__ import annotations

from ._attach import (
    REDUCERS,
    PLACEMENTS,
    KIND_POLICIES,
    ENTITY_POLICIES,
    PARTIAL_POLICIES,
    AttachReport,
    attach,
)
from ._support import obs_layers
from ._write_back import TARGETS, COMBINERS, connected, write_back, measurements

__all__ = [
    'COMBINERS',
    'ENTITY_POLICIES',
    'KIND_POLICIES',
    'PARTIAL_POLICIES',
    'PLACEMENTS',
    'REDUCERS',
    'TARGETS',
    'AttachReport',
    'attach',
    'connected',
    'measurements',
    'obs_layers',
    'write_back',
]
