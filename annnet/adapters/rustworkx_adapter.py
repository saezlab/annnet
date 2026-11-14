"""
AnnNet-rustworkx adapter.

Provides:
    to_rustworkx(anet)                 -> (rustworkx.PyDiGraph | rustworkx.PyGraph, manifest)
    from_rustworkx(rxG, manifest=None) -> AnnNet

A rustworkx graph addresses nodes and edges by integer index and carries one
Python payload per node and per edge. The export gives every node the payload
``{'node_id': ..., **attributes}`` and every edge
``{'edge_id': ..., 'weight': ..., 'directed': ..., **attributes}``, so a weight
function reads ``lambda e: e['weight']``. ``rxG.attrs`` holds
``{'index_of': {node_id: index}}``.

Hyperedges, per-edge directedness, slices, multilayer state and the attribute
tables travel in the manifest, as for the other in-memory adapters, so
``from_rustworkx(*to_rustworkx(anet))`` is the round trip.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._common import (
    _rows_like,
    slice_members,
    iter_edge_sides,
    _serialize_value,
    requested_slices,
)
from ._manifest import build_manifest, graph_from_manifest

if TYPE_CHECKING:
    from ..core import AnnNet
from ..core import _structure
from .._support import projection as _projection


def _rx():
    try:
        import rustworkx
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            'rustworkx is not installed; install it with `pip install "annnet[rustworkx]"`'
        ) from exc
    return rustworkx


def _payloads(table, key: str, public_only: bool) -> dict:
    out = {}
    for row in _rows_like(table):
        ident = row.get(key)
        if ident is None:
            continue
        out[ident] = {
            k: _serialize_value(v)
            for k, v in row.items()
            if k != key and v is not None and not (public_only and str(k).startswith('__'))
        }
    return out


def to_rustworkx(
    graph: AnnNet,
    directed: bool = True,
    hyperedge_mode: str = 'skip',
    hyperedges: str | None = None,
    coefficients: str = 'error',
    slice: str | None = None,
    slices: list[str] | None = None,
    public_only: bool = False,
    reify_prefix: str = 'he::',
) -> tuple[Any, dict]:
    """Export AnnNet → ``(rustworkx graph, manifest)``.

    Parameters
    ----------
    graph : AnnNet
    directed : bool, default True
        Build a ``PyDiGraph``; else a ``PyGraph``. An undirected edge in a
        directed export becomes two edges, one each way.
    hyperedge_mode : {"skip", "expand", "reify"}, default "skip"
        ``"skip"`` leaves hyperedges to the manifest. ``"expand"`` adds head ×
        tail edges (directed) or a clique (undirected). ``"reify"`` adds a node
        per hyperedge and a membership edge per member, carrying ``role`` and
        ``coeff``.
    hyperedges : str, optional
        The same choice under the name every exporter shares; ``"star"`` and
        ``"clique"`` stand for ``"reify"`` and ``"expand"``.
    coefficients : {"error", "drop"}, default "error"
        What ``"expand"`` does with per-member coefficients it cannot keep.
    slice, slices : optional
        Export only the nodes and edges these slices hold.
    public_only : bool, default False
        Leave out attributes whose name starts with ``"__"``.
    reify_prefix : str, default "he::"
        The ``node_id`` prefix of a reified hyperedge node.

    Returns
    -------
    tuple[rustworkx.PyDiGraph | rustworkx.PyGraph, dict]
    """
    rx = _rx()
    hyperedge_mode = _projection.normalise(hyperedges if hyperedges is not None else hyperedge_mode)
    _projection.check_coefficients(
        _structure.hyperedges_with_coefficients(graph), hyperedge_mode, coefficients
    )
    requested = requested_slices(slice, slices)
    keep_nodes, keep_edges = slice_members(graph, requested)

    rxG = rx.PyDiGraph(multigraph=True) if directed else rx.PyGraph(multigraph=True)
    node_payloads = _payloads(getattr(graph, '_node_table', None), 'node_id', public_only)
    edge_payloads = _payloads(getattr(graph, '_edge_table', None), 'edge_id', public_only)

    index_of: dict = {}

    def ensure(node_id, payload=None) -> int:
        idx = index_of.get(node_id)
        if idx is None:
            idx = rxG.add_node({**(payload or {}), 'node_id': node_id})
            index_of[node_id] = idx
        return idx

    for v in graph.N:
        if keep_nodes is None or v in keep_nodes:
            ensure(v, node_payloads.get(v))

    def endpoint_id(x):
        return x[0] if _structure.is_entity_key(x) else x

    edges: list = []
    reified: list = []
    for eid, ref, sides in iter_edge_sides(graph):
        if keep_edges is not None and eid not in keep_edges:
            continue
        S = [endpoint_id(x) for x in sides.source]
        T = [endpoint_id(x) for x in sides.target]
        payload = {
            **edge_payloads.get(eid, {}),
            'edge_id': eid,
            'weight': float(ref.weight),
            'directed': bool(ref.directed),
        }
        members = list(dict.fromkeys(S + T))
        if ref.kind != 'hyper' and len(members) <= 2:
            if len(members) == 1:
                u = ensure(members[0])
                edges.append((u, u, payload))
            elif ref.directed:
                edges.append((ensure(S[0]), ensure(T[0]), payload))
            else:
                u, v = ensure(members[0]), ensure(members[1])
                edges.append((u, v, payload))
                if directed:
                    edges.append((v, u, dict(payload)))
            continue
        if hyperedge_mode == 'skip':
            continue
        if hyperedge_mode == 'reify':
            reified.append((eid, ref))
            continue
        # expand
        if ref.directed:
            pairs = [(u, v) for u in S for v in T]
        elif directed:
            pairs = [(u, v) for u in members for v in members if u != v]
        else:
            pairs = [(u, v) for i, u in enumerate(members) for v in members[i + 1 :]]
        for u, v in pairs:
            edges.append((ensure(u), ensure(v), dict(payload)))

    if reified:
        manifest_edges = build_manifest(graph, slice=slice, slices=slices)['edges']
        for eid, ref in reified:
            head_map, tail_map = manifest_edges[eid][0], manifest_edges[eid][1]
            he = ensure(
                f'{reify_prefix}{eid}',
                {
                    **edge_payloads.get(eid, {}),
                    'is_hyperedge': True,
                    'edge_id': eid,
                    'directed': bool(ref.directed),
                    'hyper_weight': float(ref.weight),
                },
            )
            if ref.directed:
                for u, c in tail_map.items():
                    edges.append((ensure(endpoint_id(u)), he, _member('tail', c, eid)))
                for v, c in head_map.items():
                    edges.append((he, ensure(endpoint_id(v)), _member('head', c, eid)))
            else:
                for u, c in {**tail_map, **head_map}.items():
                    ui = ensure(endpoint_id(u))
                    edges.append((ui, he, _member('member', c, eid)))
                    if directed:
                        edges.append((he, ui, _member('member', c, eid)))

    rxG.add_edges_from(edges)
    rxG.attrs = {'index_of': index_of}
    manifest = build_manifest(graph, public_only=public_only, slice=slice, slices=slices)
    return rxG, manifest


def _member(role: str, coeff, eid) -> dict:
    return {'role': role, 'coeff': float(coeff), 'membership_of': eid}


def from_rustworkx(rxG: Any, manifest: dict[str, Any] | None = None, *, hyperedge: str = 'none'):
    """Build an AnnNet from a rustworkx graph.

    Parameters
    ----------
    rxG : rustworkx.PyDiGraph | rustworkx.PyGraph
    manifest : dict, optional
        The manifest :func:`to_rustworkx` returned. With it, the graph comes
        back exactly; without it, the graph is read from the payloads.
    hyperedge : {"none", "reified"}, default "none"
        ``"reified"`` turns reified hyperedge nodes back into hyperedges.

    Returns
    -------
    AnnNet
    """
    if manifest is not None:
        H = graph_from_manifest(manifest)
        if hyperedge == 'reified':
            known = set((manifest.get('edges') or {}).keys())
            _add_reified(H, rxG, skip=known)
        return H
    return _from_payloads(rxG, hyperedge=hyperedge)


def _node_id(rxG, idx):
    payload = rxG[idx]
    if isinstance(payload, dict) and 'node_id' in payload:
        return payload['node_id']
    return payload if isinstance(payload, str) else str(idx)


def _reified_nodes(rxG) -> list[int]:
    return [
        i for i in rxG.node_indices() if isinstance(rxG[i], dict) and rxG[i].get('is_hyperedge')
    ]


def _add_reified(H, rxG, *, skip=()) -> set:
    """Rebuild the hyperedges reified in ``rxG``; return the membership edge indices."""
    membership: set = set()
    for hi in _reified_nodes(rxG):
        node = rxG[hi]
        eid = node.get('edge_id') or str(_node_id(rxG, hi))
        head, tail, members = {}, {}, {}
        if isinstance(rxG, _rx().PyDiGraph):
            incident = rxG.incident_edge_index_map(hi, all_edges=True)
        else:
            incident = rxG.incident_edge_index_map(hi)
        for eidx, (u, v, payload) in incident.items():
            membership.add(eidx)
            other = _node_id(rxG, v if u == hi else u)
            role = payload.get('role') if isinstance(payload, dict) else None
            coeff = float(payload.get('coeff', 1.0)) if isinstance(payload, dict) else 1.0
            {'head': head, 'tail': tail}.get(role, members)[other] = coeff
        if eid in skip:
            continue
        extra = {
            k: v
            for k, v in node.items()
            if k not in {'is_hyperedge', 'edge_id', 'directed', 'hyper_weight', 'node_id'}
        }
        weight = float(node.get('hyper_weight', 1.0))
        for x in {*head, *tail, *members}:
            if not H.has_node(x):
                H.add_nodes(x)
        if head or tail:
            H.add_edges(
                [{'head': list(head), 'tail': list(tail), 'edge_id': eid, 'weight': weight}]
            )
            coeffs = {'__source_attr': head, '__target_attr': tail}
        else:
            H.add_edges([{'members': list(members), 'edge_id': eid, 'weight': weight}])
            coeffs = {'__source_attr': members, '__target_attr': members}
        if any(c != 1.0 for side in coeffs.values() for c in side.values()):
            extra.update(
                {k: {u: {'__value': c} for u, c in side.items()} for k, side in coeffs.items()}
            )
        if extra:
            H.attrs.update('edges', {eid: extra})
    return membership


def _from_payloads(rxG, *, hyperedge: str) -> AnnNet:
    from ..core import AnnNet

    is_dir = isinstance(rxG, _rx().PyDiGraph)
    H = AnnNet(directed=is_dir)
    reified = set(_reified_nodes(rxG)) if hyperedge == 'reified' else set()
    nodes, node_attrs = [], {}
    for i in rxG.node_indices():
        if i in reified:
            continue
        vid = _node_id(rxG, i)
        nodes.append({'node_id': vid})
        payload = rxG[i]
        if isinstance(payload, dict):
            extra = {k: v for k, v in payload.items() if k != 'node_id' and v is not None}
            if extra:
                node_attrs[vid] = extra
    if nodes:
        H._add_nodes_bulk(nodes)
    membership = _add_reified(H, rxG) if reified else set()

    seen_ids: set = set()
    binary, edge_attrs = [], {}
    for eidx, (u, v, payload) in rxG.edge_index_map().items():
        if eidx in membership:
            continue
        data = payload if isinstance(payload, dict) else {'weight': payload}
        eid = data.get('edge_id', f'rx::e#{eidx}')
        if eid in seen_ids:
            # The second half of an undirected edge exported as two arcs.
            continue
        seen_ids.add(eid)
        weight = data.get('weight', 1.0)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        binary.append(
            {
                'source': _node_id(rxG, u),
                'target': _node_id(rxG, v),
                'edge_id': eid,
                'edge_directed': bool(data.get('directed', is_dir)),
                'weight': weight,
            }
        )
        extra = {
            k: val
            for k, val in data.items()
            if k not in {'edge_id', 'weight', 'directed'} and val is not None
        }
        if extra:
            edge_attrs[eid] = extra
    if binary:
        H._add_edges_bulk(binary, default_edge_directed=is_dir)
    if node_attrs:
        H.attrs.update('nodes', node_attrs)
    if edge_attrs:
        H.attrs.update('edges', edge_attrs)
    return H


__all__ = ['from_rustworkx', 'to_rustworkx']
