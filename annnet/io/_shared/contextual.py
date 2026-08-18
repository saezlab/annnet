"""Contextual attributes as a JSON document, and back.

The contextual levels are keyed by tuples — ``(slice_id, edge_id)``,
``(node_id, layer)`` — and JSON keys are strings. This is the one place that
translation lives, so a sidecar, and anything else that has to write contextual
state to a text format, agree on the encoding.
"""

from __future__ import annotations

import json
from typing import Any


def _encode_key(key) -> str:
    """A level key as one JSON string. A tuple keeps its shape through JSON."""
    return json.dumps(list(key) if isinstance(key, tuple) else key, sort_keys=True)


def _decode_key(text: str):
    value = json.loads(text)
    return tuple(_retuple(part) for part in value) if isinstance(value, list) else value


def _retuple(part):
    """A layer coordinate is a tuple inside a key, and JSON made it a list."""
    return tuple(part) if isinstance(part, list) else part


def contextual_payload(graph) -> dict[str, dict[str, Any]]:
    """Every contextual level of one graph, JSON-safe."""
    store = graph._contextual
    return {
        level: {_encode_key(key): dict(attrs) for key, attrs in getattr(store, level).items()}
        for level in store.levels
        if getattr(store, level)
    }


def restore_contextual(graph, payload: dict[str, dict[str, Any]]) -> None:
    """Put every level back, merging into whatever the graph already holds."""
    store = graph._contextual
    for level, contents in (payload or {}).items():
        if level not in store.levels:
            continue
        for text, attrs in contents.items():
            store.set(level, _decode_key(text), dict(attrs))
