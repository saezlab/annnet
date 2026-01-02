from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from torch_geometric.data import HeteroData

from ._utils import _df_to_rows, _safe_df_to_rows

if TYPE_CHECKING:
    from annnet.core.graph import AnnNet

try:
    import polars as pl
except Exception:
    pl = None

try:
    import pandas as pd
except Exception:
    pd = None


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
            if val is not None and not isinstance(val, (int, float, np.number)):
                raise ValueError(
                    f"{context}: column '{col}' must be numeric, got {type(val).__name__}"
                )


def to_pyg(
    graph: AnnNet,
    node_features: dict[str, list[str]] | None = None,
    edge_features: dict[tuple[str, str, str], list[str]] | None = None,
    slice_id: str | None = None,
    hyperedge_mode: Literal["skip", "reify", "expand"] = "reify",
    device: str = "cpu",
) -> HeteroData:
    """
    Direct AnnNet -> PyTorch Geometric adapter.

    - Respects AnnNet architecture (uses entity_types, edge_definitions)
    - Narwhals-compatible (polars/pandas via _df_to_rows)
    - Heterogeneous (vertex kinds)
    - Hypergraph-safe (reification)
    - Slice-aware (boolean masks)
    """

    if slice_id is None:
        slice_id = graph.slices.active

    data = HeteroData()
    manifest = {
        "node_index": {},
        "edge_index": {},
    }

    # Build attribute lookup maps
    vert_rows = _safe_df_to_rows(getattr(graph, "vertex_attributes", None))
    edge_rows = _safe_df_to_rows(getattr(graph, "edge_attributes", None))

    v_attrs_map: dict[str, dict] = {}
    if vert_rows:
        id_col = None
        if vert_rows[0]:
            if "vertex_id" in vert_rows[0]:
                id_col = "vertex_id"
            elif "id" in vert_rows[0]:
                id_col = "id"

        if id_col:
            for r in vert_rows:
                vid = str(r.get(id_col))
                v_attrs_map[vid] = r

    e_attrs_map: dict[str, dict] = {}
    if edge_rows:
        id_col = None
        if edge_rows[0]:
            if "edge_id" in edge_rows[0]:
                id_col = "edge_id"
            elif "id" in edge_rows[0]:
                id_col = "id"

        if id_col:
            for r in edge_rows:
                eid = str(r.get(id_col))
                e_attrs_map[eid] = r

    # Group vertices by kind
    kind_to_vertices: dict[str, list[str]] = {}
    for uid, utype in graph.entity_types.items():
        if utype != "vertex":
            continue

        row = v_attrs_map.get(str(uid), {})
        kind = row.get("kind", "default")

        if kind not in kind_to_vertices:
            kind_to_vertices[kind] = []
        kind_to_vertices[kind].append(str(uid))

    # Process nodes by kind
    for kind, vids in kind_to_vertices.items():
        n = len(vids)

        idx_map = dict(zip(vids, range(n)))
        manifest["node_index"][kind] = idx_map

        if node_features and kind in node_features:
            cols = node_features[kind]
            kind_rows = [v_attrs_map.get(vid, {}) for vid in vids]
            _validate_numeric(kind_rows, cols, f"node[{kind}]")
            data[kind].x = _rows_to_tensor(kind_rows, cols, device=device)
        else:
            data[kind].num_nodes = n

        # Slice mask
        if slice_id is not None:
            try:
                members = set(graph.get_slice_vertices(slice_id))
            except Exception:
                members = set()

            mask = np.array([v in members for v in vids], dtype=bool)
            data[kind][f"{slice_id}_mask"] = torch.from_numpy(mask).to(device)

    # Process edges using edge_definitions (source of truth)
    for eid, defn in graph.edge_definitions.items():
        is_hyper = eid in getattr(graph, "hyperedge_definitions", {})

        if is_hyper:
            if hyperedge_mode == "skip":
                continue

            if hyperedge_mode == "reify":
                _process_hyperedge_reify(graph, eid, data, manifest, device, v_attrs_map)
            elif hyperedge_mode == "expand":
                _process_hyperedge_expand(
                    graph, eid, data, manifest, device, v_attrs_map, e_attrs_map
                )
            continue

        # Regular binary edge
        try:
            u, v, _ = defn
        except Exception:
            continue

        u_str = str(u)
        v_str = str(v)

        u_row = v_attrs_map.get(u_str, {})
        v_row = v_attrs_map.get(v_str, {})

        uk = u_row.get("kind", "default")
        vk = v_row.get("kind", "default")

        if uk not in manifest["node_index"] or vk not in manifest["node_index"]:
            continue

        ui = manifest["node_index"][uk].get(u_str)
        vi = manifest["node_index"][vk].get(v_str)

        if ui is None or vi is None:
            continue

        etype = (uk, "edge", vk)

        if etype not in data.edge_types:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            data[etype].edge_weight = torch.empty((0,), dtype=torch.float32, device=device)

        edge_idx = torch.tensor([[ui], [vi]], dtype=torch.long, device=device)
        data[etype].edge_index = torch.cat([data[etype].edge_index, edge_idx], dim=1)

        w = float(graph.edge_weights.get(eid, 1.0))

        src_map = graph.get_edge_attribute(eid, "__source_attr") or {}
        tgt_map = graph.get_edge_attribute(eid, "__target_attr") or {}

        w *= src_map.get(u, {}).get("__value", 1.0)
        w *= tgt_map.get(v, {}).get("__value", 1.0)

        ew = torch.tensor([w], dtype=torch.float32, device=device)
        data[etype].edge_weight = torch.cat([data[etype].edge_weight, ew])

    # Edge features
    if edge_features:
        for etype, cols in edge_features.items():
            if etype not in data.edge_types:
                continue

            _validate_numeric(list(e_attrs_map.values()), cols, f"edge{etype}")

            feat_rows = list(e_attrs_map.values())
            if feat_rows:
                feats = _rows_to_tensor(feat_rows, cols, device=device)
                data[etype].edge_attr = feats

    data.manifest = manifest
    return data


def _process_hyperedge_reify(
    graph: AnnNet,
    eid: str,
    data: HeteroData,
    manifest: dict,
    device: str,
    v_attrs_map: dict,
):
    """Reify hyperedge as virtual hypernode."""
    if "hypernode" not in data.node_types:
        data["hypernode"].num_nodes = 0

    he_idx = data["hypernode"].num_nodes
    data["hypernode"].num_nodes += 1

    hdef = graph.hyperedge_definitions[eid]
    directed = hdef.get("directed", True)

    if directed:
        S = set(hdef.get("head", []))
        T = set(hdef.get("tail", []))
        members = S | T
    else:
        members = set(hdef.get("members", []))

    for u in members:
        u_str = str(u)
        u_row = v_attrs_map.get(u_str, {})
        uk = u_row.get("kind", "default")

        if uk not in manifest["node_index"]:
            continue

        ui = manifest["node_index"][uk].get(u_str)
        if ui is None:
            continue

        etype = (uk, "member_of", "hypernode")

        if etype not in data.edge_types:
            data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

        edge_idx = torch.tensor([[ui], [he_idx]], dtype=torch.long, device=device)
        data[etype].edge_index = torch.cat([data[etype].edge_index, edge_idx], dim=1)


def _process_hyperedge_expand(
    graph: AnnNet,
    eid: str,
    data: HeteroData,
    manifest: dict,
    device: str,
    v_attrs_map: dict,
    e_attrs_map: dict,
):
    """Expand hyperedge into pairwise edges."""
    hdef = graph.hyperedge_definitions[eid]
    directed = hdef.get("directed", True)

    if directed:
        S = set(hdef.get("head", []))
        T = set(hdef.get("tail", []))

        for t in T:
            for s in S:
                _add_expanded_edge(
                    graph, eid, str(t), str(s), data, manifest, device, v_attrs_map, e_attrs_map
                )
    else:
        members = list(hdef.get("members", []))
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                _add_expanded_edge(
                    graph,
                    eid,
                    str(members[i]),
                    str(members[j]),
                    data,
                    manifest,
                    device,
                    v_attrs_map,
                    e_attrs_map,
                )


def _add_expanded_edge(
    graph: AnnNet,
    eid: str,
    u_str: str,
    v_str: str,
    data: HeteroData,
    manifest: dict,
    device: str,
    v_attrs_map: dict,
    e_attrs_map: dict,
):
    """Add single expanded edge from hyperedge."""
    u_row = v_attrs_map.get(u_str, {})
    v_row = v_attrs_map.get(v_str, {})

    uk = u_row.get("kind", "default")
    vk = v_row.get("kind", "default")

    if uk not in manifest["node_index"] or vk not in manifest["node_index"]:
        return

    ui = manifest["node_index"][uk].get(u_str)
    vi = manifest["node_index"][vk].get(v_str)

    if ui is None or vi is None:
        return

    etype = (uk, "edge", vk)

    if etype not in data.edge_types:
        data[etype].edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        data[etype].edge_weight = torch.empty((0,), dtype=torch.float32, device=device)

    edge_idx = torch.tensor([[ui], [vi]], dtype=torch.long, device=device)
    data[etype].edge_index = torch.cat([data[etype].edge_index, edge_idx], dim=1)

    w = float(graph.edge_weights.get(eid, 1.0))
    ew = torch.tensor([w], dtype=torch.float32, device=device)
    data[etype].edge_weight = torch.cat([data[etype].edge_weight, ew])
