"""
GraphML/GEXF import and export helpers for AnnNet.

Provides:
    to_graphml(G, path, ...)   -> None
    from_graphml(path, ...)    -> AnnNet
    to_gexf(G, path, ...)      -> None
    from_gexf(path, ...)       -> AnnNet

GraphML and GEXF support are implemented through NetworkX because NetworkX
already owns mature readers and writers for these formats. This module is
therefore an intentional IO-to-adapter bridge: AnnNet is first projected to a
NetworkX graph, then NetworkX handles the file format.

This is an accepted boundary exception.
"""

from __future__ import annotations

import os
import re
import json
import math
from typing import TYPE_CHECKING

import networkx as nx

from .._support.serialization import serialize_endpoint, deserialize_endpoint
from ..adapters.networkx_adapter import (
    to_nx,
    from_nx,
    from_nx_without_manifest,
)

if TYPE_CHECKING:
    from ..core import AnnNet


_BOOL = {'true': True, 'false': False}
_NUM_RE = re.compile(r'^[+-]?(?:\d+|\d*\.\d+)(?:[eE][+-]?\d+)?$')
_SUPRA_PREFIX = '{"kind": "supra"'


def _encode_node(n):
    """GraphML/GEXF node keys must be strings.

    Encode multilayer supra-node tuples as JSON; leave plain string ids untouched.
    """
    return json.dumps(serialize_endpoint(n)) if isinstance(n, tuple) else n


def _decode_node(n):
    """Reverse ``_encode_node``.

    Restore a JSON-encoded supra-node key to its ``(vid, layer_coord)`` tuple;
    leave every other id unchanged.
    """
    if isinstance(n, str) and n.startswith(_SUPRA_PREFIX):
        try:
            return deserialize_endpoint(json.loads(n))
        except (json.JSONDecodeError, ValueError):
            return n
    return n


def _relabel_nx(G, fn):
    mapping = {n: fn(n) for n in G.nodes}
    if any(k != v for k, v in mapping.items()):
        return nx.relabel_nodes(G, mapping, copy=True)
    return G


def _encode_manifest(o):
    """Make the to_nx manifest JSON-safe.

    Multilayer supra-node keys appear both as dict keys (hyperedge head/tail
    coeff maps) and values (edge endpoints); JSON forbids tuple keys, so encode
    them.
    """
    if isinstance(o, dict):
        return {
            (json.dumps(serialize_endpoint(k)) if isinstance(k, tuple) else k): _encode_manifest(v)
            for k, v in o.items()
        }
    if isinstance(o, tuple):
        se = serialize_endpoint(o)
        return se if se is not o else [_encode_manifest(x) for x in o]
    if isinstance(o, list):
        return [_encode_manifest(x) for x in o]
    return o


def _decode_manifest(o):
    """Reverse ``_encode_manifest``."""
    if isinstance(o, dict):
        if o.get('kind') == 'supra':
            return deserialize_endpoint(o)
        return {_decode_node(k): _decode_manifest(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_decode_manifest(x) for x in o]
    return o


def _sanitize_graphml_inplace(G):
    """Remove values GraphML can't encode and normalize types before write."""

    def clean_dict(d: dict):
        for k in list(d.keys()):
            v = d[k]
            if v is None or (isinstance(v, float) and math.isnan(v)):
                del d[k]
                continue
            if isinstance(v, (str, int, float, bool)):
                continue
            try:
                d[k] = json.dumps(v, ensure_ascii=False)
            except (TypeError, ValueError):
                d[k] = str(v)

    clean_dict(G.graph)
    for _, data in G.nodes(data=True):
        clean_dict(data)
    for _, _, data in G.edges(data=True):
        clean_dict(data)


def _restore_types_graphml_inplace(G):
    """Heuristically restore types after read_graphml."""

    def coerce(s):
        if not isinstance(s, str):
            return s
        t = s.strip()
        low = t.lower()
        if low in _BOOL:
            return _BOOL[low]
        if (t.startswith('{') and t.endswith('}')) or (t.startswith('[') and t.endswith(']')):
            try:
                return json.loads(t)
            except json.JSONDecodeError:
                return t
        if _NUM_RE.match(t):
            try:
                if '.' not in t and 'e' not in low:
                    return int(t)
                return float(t)
            except ValueError:
                return t
        return t

    def fix_dict(d: dict):
        for k in list(d.keys()):
            d[k] = coerce(d[k])

    fix_dict(G.graph)
    for _, data in G.nodes(data=True):
        fix_dict(data)
    for _, _, data in G.edges(data=True):
        fix_dict(data)


def to_graphml(graph, path, *, directed=True, hyperedge_mode='reify', public_only=False):
    """Export via NetworkX with reified hyperedges; sanitize attrs for GraphML.

    Also writes a sidecar manifest for lossless re-import.
    """
    G, manifest = to_nx(
        graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only
    )
    G = _relabel_nx(G, _encode_node)
    _sanitize_graphml_inplace(G)
    nx.write_graphml(G, path)
    # sidecar manifest
    mpath = str(path) + '.manifest.json'
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(_encode_manifest(manifest), f, ensure_ascii=False)


def from_graphml(path, *, hyperedge='reified'):
    """Import via NetworkX; if a sidecar manifest is present, use it as SSOT.

    Otherwise, fall back to a best-effort no-manifest import with type restoration.
    """
    G = nx.read_graphml(path)
    _restore_types_graphml_inplace(G)
    G = _relabel_nx(G, _decode_node)
    mpath = str(path) + '.manifest.json'
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as f:
            manifest = _decode_manifest(json.load(f))
        # Rebuild exactly from the manifest (lossless), ignoring GraphML-added noise
        return from_nx(G, manifest, hyperedge=('reified' if hyperedge == 'reified' else 'none'))
    # Fallback (no manifest available)
    return from_nx_without_manifest(G, hyperedge=('reified' if hyperedge == 'reified' else 'none'))


def to_gexf(graph: AnnNet, path, *, directed=True, hyperedge_mode='reify', public_only=False):
    """Export an AnnNet graph to GEXF via NetworkX.

    GEXF (like GraphML) cannot encode ``None`` attribute values; the same
    sanitiser used for GraphML is applied here. A sidecar manifest is also
    written so ``from_gexf`` can round-trip losslessly.
    """
    G, manifest = to_nx(
        graph, directed=directed, hyperedge_mode=hyperedge_mode, public_only=public_only
    )
    G = _relabel_nx(G, _encode_node)
    _sanitize_graphml_inplace(G)
    nx.write_gexf(G, path)
    mpath = str(path) + '.manifest.json'
    with open(mpath, 'w', encoding='utf-8') as f:
        json.dump(_encode_manifest(manifest), f, ensure_ascii=False)


def from_gexf(path, *, hyperedge='reified') -> AnnNet:
    """Import a GEXF graph through NetworkX.

    Uses the sidecar manifest written by ``to_gexf`` when present (lossless
    round-trip); otherwise falls back to a best-effort import.
    """
    G = nx.read_gexf(path)
    _restore_types_graphml_inplace(G)
    G = _relabel_nx(G, _decode_node)
    mpath = str(path) + '.manifest.json'
    if os.path.exists(mpath):
        with open(mpath, encoding='utf-8') as f:
            manifest = _decode_manifest(json.load(f))
        return from_nx(G, manifest, hyperedge=('reified' if hyperedge == 'reified' else 'none'))
    return from_nx_without_manifest(G, hyperedge=('reified' if hyperedge == 'reified' else 'none'))
