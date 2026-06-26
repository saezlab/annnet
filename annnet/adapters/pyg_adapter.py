"""
AnnNet-PyTorch Geometric adapter for AnnNet.

Provides:
    to_pyg(G) -> torch_geometric.data.HeteroData

PyTorch Geometric represents graph data as tensors. This adapter exports AnnNet
vertices and edges into a heterogeneous graph structure suitable for downstream
GNN workflows.

AnnNet-specific structures such as slices, multilayer metadata, hyperedge
semantics, and rich attribute tables are only exported where they can be mapped
to tensor-compatible node, edge, or graph-level fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ._common import (
    _iter_vertex_ids,
    dataframe_to_rows,
)

if TYPE_CHECKING:
    from ..core import AnnNet


def _rows_to_tensor(
    rows: list[dict],
    cols: list[str],
    *,
    device: str,
) -> torch.Tensor:
    """Convert list of dicts to tensor, extracting specified columns."""
    if not rows:
        return torch.empty((0, len(cols)), dtype=torch.float32, device=device)

    # Explicitly handle None values - convert to 0.0
    data = []
    for row in rows:
        row_data = []
        for c in cols:
            val = row.get(c)
            if val is None:
                row_data.append(0.0)
            else:
                try:
                    row_data.append(float(val))
                except (TypeError, ValueError):
                    row_data.append(0.0)
        data.append(row_data)

    arr = np.array(data, dtype=np.float32)
    # Extra safety: replace any remaining NaN with 0.0
    arr = np.nan_to_num(arr, nan=0.0)

    return torch.from_numpy(arr).to(device=device)


def _validate_numeric(rows: list[dict], cols: list[str], context: str):
    """Validate that specified columns contain numeric data."""
    for row in rows:
        for col in cols:
            val = row.get(col)
            if val is None:
                continue
            try:
                float(val)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{context}: column '{col}' must be numeric, got {type(val).__name__}"
                ) from None


def _edge_weight(graph: AnnNet, edge_id: str) -> float:
    weight = graph.edge_weights.get(edge_id)
    return 1.0 if weight is None else float(weight)


def _endpoint_coeff(graph: AnnNet, edge_id: str, key: str, endpoint) -> float:
    coeff_map = graph.attrs.get_attr_edge(edge_id, key) or {}
    return float(coeff_map.get(endpoint, {}).get('__value', 1.0))


def _edge_weight_lookup(weights: dict, edge_id: str) -> float:
    """Cached-dict variant of :func:`_edge_weight` for tight loops."""
    w = weights.get(edge_id)
    return 1.0 if w is None else float(w)


def _endpoint_coeff_from_map(coeff_map, endpoint) -> float:
    """Cached-map variant of :func:`_endpoint_coeff` for tight loops."""
    if not coeff_map:
        return 1.0
    return float(coeff_map.get(endpoint, {}).get('__value', 1.0))


def _flush_edge_buckets(data, buckets, device):
    """Write batched edges out to HeteroData in one tensor build per etype."""
    for etype, bucket in buckets.items():
        src_list = bucket['src']
        if src_list:
            edge_index = torch.tensor([src_list, bucket['tgt']], dtype=torch.long, device=device)
            edge_weight = torch.tensor(bucket['w'], dtype=torch.float32, device=device)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            edge_weight = torch.empty((0,), dtype=torch.float32, device=device)
        data[etype].edge_index = edge_index
        if bucket.get('emit_weight', True):
            data[etype].edge_weight = edge_weight


def to_pyg(
    graph: AnnNet,
    node_features: dict[str, list[str]] | None = None,
    edge_features: dict[tuple[str, str, str], list[str]] | None = None,
    slice_id: str | None = None,
    hyperedge_mode: Literal['skip', 'reify', 'expand'] = 'reify',
    device: str = 'cpu',
) -> HeteroData:
    """
    Direct AnnNet -> PyTorch Geometric adapter.

    - Respects AnnNet architecture (uses entity_types, edge_definitions)
    - Narwhals-compatible dataframe input via shared row conversion
    - Heterogeneous (vertex kinds)
    - Hypergraph-safe (reification)
    - Slice-aware (boolean masks)
    """

    if slice_id is None:
        slice_id = graph.slices.active

    data = HeteroData()
    manifest = {
        'node_index': {},
        'edge_index': {},
    }

    # Build attribute lookup maps
    vert_rows = dataframe_to_rows(getattr(graph, 'vertex_attributes', None))
    edge_rows = dataframe_to_rows(getattr(graph, 'edge_attributes', None))

    v_attrs_map: dict[str, dict] = {}
    if vert_rows:
        id_col = None
        if vert_rows[0]:
            if 'vertex_id' in vert_rows[0]:
                id_col = 'vertex_id'
            elif 'id' in vert_rows[0]:
                id_col = 'id'

        if id_col:
            for r in vert_rows:
                vid = str(r.get(id_col))
                v_attrs_map[vid] = r

    e_attrs_map: dict[str, dict] = {}
    if edge_rows:
        id_col = None
        if edge_rows[0]:
            if 'edge_id' in edge_rows[0]:
                id_col = 'edge_id'
            elif 'id' in edge_rows[0]:
                id_col = 'id'

        if id_col:
            for r in edge_rows:
                eid = str(r.get(id_col))
                e_attrs_map[eid] = r

    # Group vertices by kind
    kind_to_vertices: dict[str, list[str]] = {}
    for uid in _iter_vertex_ids(graph):
        row = v_attrs_map.get(str(uid), {})
        kind = row.get('kind') or 'default'

        if kind not in kind_to_vertices:
            kind_to_vertices[kind] = []
        kind_to_vertices[kind].append(str(uid))

    # Process nodes by kind
    for kind, vids in kind_to_vertices.items():
        n = len(vids)

        idx_map = dict(zip(vids, range(n), strict=False))
        manifest['node_index'][kind] = idx_map

        if node_features and kind in node_features:
            cols = node_features[kind]
            kind_rows = [v_attrs_map.get(vid, {}) for vid in vids]
            _validate_numeric(kind_rows, cols, f'node[{kind}]')
            data[kind].x = _rows_to_tensor(kind_rows, cols, device=device)
        else:
            data[kind].num_nodes = n

        # Slice mask
        if slice_id is not None:
            try:
                members = set(graph.slices.vertices(slice_id))
            except Exception:  # noqa: BLE001
                members = set()

            mask = np.array([v in members for v in vids], dtype=bool)
            data[kind][f'{slice_id}_mask'] = torch.from_numpy(mask).to(device)

    # Cache hot-path lookups once per to_pyg call (each was an O(E)
    # rebuild on every previous access).
    edge_weights_cache = dict(graph.edge_weights)
    edge_attr_cols: set[str] = set()
    try:
        from .._support.dataframe_backend import dataframe_columns  # local to avoid hard dep

        edge_attr_cols = set(dataframe_columns(graph.edge_attributes) or ())
    except Exception:  # noqa: BLE001
        edge_attr_cols = set()
    has_stoich = '__source_attr' in edge_attr_cols or '__target_attr' in edge_attr_cols

    node_index = manifest['node_index']

    # Batch binary edges into per-etype Python lists, then build tensors
    # ONCE per etype. The previous per-edge torch.cat made this O(E^2)
    # in tensor-copy traffic.
    edge_buckets: dict[tuple, dict] = {}

    def _bare(endpoint):
        # Multilayer endpoints are (vid, layer_coord) tuples; collapse to
        # bare vid for kind / attribute lookups (kind is per vertex_id,
        # not per supra-node).
        if isinstance(endpoint, tuple) and len(endpoint) == 2 and isinstance(endpoint[0], str):
            return endpoint[0]
        return str(endpoint)

    for eid, (src, tgt, _etype) in graph.edge_definitions.items():
        u_str = _bare(src)
        v_str = _bare(tgt)

        u_row = v_attrs_map.get(u_str, {})
        v_row = v_attrs_map.get(v_str, {})

        uk = u_row.get('kind') or 'default'
        vk = v_row.get('kind') or 'default'

        if uk not in node_index or vk not in node_index:
            continue

        ui = node_index[uk].get(u_str)
        vi = node_index[vk].get(v_str)
        if ui is None or vi is None:
            continue

        etype = (uk, 'edge', vk)
        bucket = edge_buckets.get(etype)
        if bucket is None:
            bucket = {'src': [], 'tgt': [], 'w': [], 'emit_weight': True}
            edge_buckets[etype] = bucket

        bucket['src'].append(ui)
        bucket['tgt'].append(vi)

        w = _edge_weight_lookup(edge_weights_cache, eid)
        if has_stoich:
            w *= _endpoint_coeff(graph, eid, '__source_attr', src)
            w *= _endpoint_coeff(graph, eid, '__target_attr', tgt)
        bucket['w'].append(w)

    # Hyperedges share the same bucket dict so expand-mode contributions
    # (which can hit the same etype as binary edges) are concatenated, not
    # overwritten. Reify-mode hits a disjoint 'member_of' etype.
    if hyperedge_mode != 'skip':
        if hyperedge_mode == 'reify':
            for eid, spec in graph.hyperedge_definitions.items():
                _process_hyperedge_reify(
                    graph, eid, spec, data, manifest, device, v_attrs_map, edge_buckets
                )
        elif hyperedge_mode == 'expand':
            for eid, spec in graph.hyperedge_definitions.items():
                _process_hyperedge_expand(
                    graph,
                    eid,
                    spec,
                    data,
                    manifest,
                    device,
                    v_attrs_map,
                    e_attrs_map,
                    edge_buckets,
                    edge_weights_cache,
                )

    _flush_edge_buckets(data, edge_buckets, device)

    # Edge features
    if edge_features:
        for etype, cols in edge_features.items():
            if etype not in data.edge_types:
                continue

            _validate_numeric(list(e_attrs_map.values()), cols, f'edge{etype}')

            feat_rows = list(e_attrs_map.values())
            if feat_rows:
                feats = _rows_to_tensor(feat_rows, cols, device=device)
                data[etype].edge_attr = feats

    data.manifest = manifest
    return data


def _process_hyperedge_reify(
    graph: AnnNet,
    eid: str,
    spec: dict,
    data: HeteroData,
    manifest: dict,
    device: str,
    v_attrs_map: dict,
    buckets: dict[tuple, dict],
):
    """Reify hyperedge as virtual hypernode, batching emitted edges into ``buckets``."""
    if 'hypernode' not in data.node_types:
        data['hypernode'].num_nodes = 0

    he_idx = data['hypernode'].num_nodes
    data['hypernode'].num_nodes += 1

    directed = bool(spec.get('directed', False))

    if directed:
        members = set(spec.get('head', [])) | set(spec.get('tail', []))
    else:
        members = set(spec.get('members', []))

    node_index = manifest['node_index']
    for u in members:
        u_str = str(u)
        u_row = v_attrs_map.get(u_str, {})
        uk = u_row.get('kind') or 'default'

        if uk not in node_index:
            continue
        ui = node_index[uk].get(u_str)
        if ui is None:
            continue

        etype = (uk, 'member_of', 'hypernode')
        bucket = buckets.get(etype)
        if bucket is None:
            bucket = {'src': [], 'tgt': [], 'w': [], 'emit_weight': False}
            buckets[etype] = bucket
        bucket['src'].append(ui)
        bucket['tgt'].append(he_idx)


def _process_hyperedge_expand(
    graph: AnnNet,
    eid: str,
    spec: dict,
    data: HeteroData,
    manifest: dict,
    device: str,
    v_attrs_map: dict,
    e_attrs_map: dict,
    buckets: dict[tuple, dict],
    edge_weights_cache: dict,
):
    """Expand hyperedge into pairwise edges, batched into ``buckets``."""
    directed = bool(spec.get('directed', False))

    if directed:
        S = set(spec.get('head', []))
        T = set(spec.get('tail', []))
        for t in T:
            for s in S:
                _add_expanded_edge(
                    eid,
                    str(t),
                    str(s),
                    manifest,
                    v_attrs_map,
                    buckets,
                    edge_weights_cache,
                )
    else:
        members = list(spec.get('members', []))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                _add_expanded_edge(
                    eid,
                    str(members[i]),
                    str(members[j]),
                    manifest,
                    v_attrs_map,
                    buckets,
                    edge_weights_cache,
                )


def _add_expanded_edge(
    eid: str,
    u_str: str,
    v_str: str,
    manifest: dict,
    v_attrs_map: dict,
    buckets: dict[tuple, dict],
    edge_weights_cache: dict,
):
    """Batch a single expanded pairwise edge from a hyperedge."""
    u_row = v_attrs_map.get(u_str, {})
    v_row = v_attrs_map.get(v_str, {})

    uk = u_row.get('kind') or 'default'
    vk = v_row.get('kind') or 'default'

    node_index = manifest['node_index']
    if uk not in node_index or vk not in node_index:
        return

    ui = node_index[uk].get(u_str)
    vi = node_index[vk].get(v_str)
    if ui is None or vi is None:
        return

    etype = (uk, 'edge', vk)
    bucket = buckets.get(etype)
    if bucket is None:
        bucket = {'src': [], 'tgt': [], 'w': [], 'emit_weight': True}
        buckets[etype] = bucket
    bucket['src'].append(ui)
    bucket['tgt'].append(vi)
    bucket['w'].append(_edge_weight_lookup(edge_weights_cache, eid))
