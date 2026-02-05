from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.graph import AnnNet

from ..adapters._utils import (
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_VM,
    _safe_df_to_rows,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_VM,
)
from .io_annnet import _iter_rows, _read_parquet, _write_parquet_df


def build_dataframe_from_rows(rows):
    """Build a DataFrame from a list of row records using available backends.

    Parameters
    ----------
    rows : list[dict] | list[list] | list[tuple]
        Row records to load into a DataFrame.

    Returns
    -------
    DataFrame-like
        Polars DataFrame if available, otherwise pandas DataFrame.

    Raises
    ------
    RuntimeError
        If neither polars nor pandas is available.

    Notes
    -----
    Uses Polars when installed for performance; otherwise falls back to pandas.
    """
    try:
        import polars as pl

        if not rows:
            return pl.DataFrame()

        # Peek at first row to detect list columns
        schema_overrides = {
            "head": pl.List(pl.Utf8),
            "tail": pl.List(pl.Utf8),
            "members": pl.List(pl.Utf8),
        }


        for key, value in first_row.items():
            if isinstance(value, (list, tuple)):
                # Force list columns to be List(Utf8) type
                schema_overrides[key] = pl.List(pl.Utf8)

        if schema_overrides:
            return pl.DataFrame(rows, schema_overrides=schema_overrides)
        else:
            return pl.DataFrame(rows)

    except Exception:
        try:
            import pandas as pd

            return pd.DataFrame.from_records(rows)
        except Exception:
            raise RuntimeError(
                "No dataframe backend available. Install polars (recommended) or pandas."
            )


def pd_build_dataframe_from_rows(rows):
    """Build a DataFrame from a list of row records using available backends."""
    # Force pandas temporarily to bypass Polars list handling bug
    import pandas as pd

    return pd.DataFrame.from_records(rows)


def _strip_nulls(d: dict):
    # remove keys whose value is None or NaN
    clean = {}
    for k, v in list(d.items()):
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        clean[k] = v
    return clean


def _is_directed_eid(graph, eid):
    """Get edge directedness. Default False for hyperedges, True for binary."""
    kind = getattr(graph, "edge_kind", {}).get(eid)

    # Check edge_directed dict first
    try:
        ed = getattr(graph, "edge_directed", {})
        if eid in ed:
            return bool(ed[eid])
    except Exception:
        pass

    # Check attribute
    try:
        val = graph.get_edge_attribute(eid, "directed")
        if val is not None:
            return bool(val)
    except Exception:
        pass

    # For hyperedges, check if S and T are identical (undirected)
    if kind == "hyper":
        try:
            eidx = graph.edge_to_idx[eid]
            S, T = graph.get_edge(eidx)
            # If S == T, it's undirected (same member set in both)
            if S == T:
                return False
        except Exception:
            pass

    # Default: True for binary, False for hyper
    return kind != "hyper"


def _coerce_coeff_mapping(val):
    """Normalize various serialized forms into {vertex: {__value: float}|float}.
    Accepts dict | list | list-of-dicts | list-of-pairs | JSON string.
    """
    if val is None:
        return {}
    if isinstance(val, str):
        try:
            return _coerce_coeff_mapping(json.loads(val))
        except Exception:
            return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, (list, tuple)):
        out = {}
        for item in val:
            if isinstance(item, dict):
                if "vertex" in item and "__value" in item:
                    out[item["vertex"]] = {"__value": item["__value"]}
                else:
                    for k, v in item.items():
                        out[k] = v
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                k, v = item
                out[k] = v
        return out
    return {}


def _is_nullish(val) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and math.isnan(val):
            return True
    except Exception:
        pass
    try:
        import pandas as pd

        if pd.isna(val):
            return True
    except Exception:
        pass
    return False


def _as_list_or_empty(val):
    if _is_nullish(val):
        return []

    # Polars Series / Array
    try:
        import polars as pl
        if isinstance(val, pl.Series):
            return val.to_list()
    except Exception:
        pass

    # numpy array
    try:
        import numpy as np
        if isinstance(val, np.ndarray):
            return val.tolist()
    except Exception:
        pass

    # already list / tuple
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)

    # scalar -> singleton
    return [val]


def _endpoint_coeff_map(edge_attrs, private_key, endpoint_set):
    """Return {vertex: float_coeff} for the given endpoint_set.
    Reads from edge_attrs[private_key] which may be serialized in multiple shapes.
    Missing endpoints default to 1.0.
    """
    raw_mapping = (edge_attrs or {}).get(private_key, {})
    mapping = _coerce_coeff_mapping(raw_mapping)
    endpoints = list(endpoint_set or mapping.keys())
    out = {}
    for u in endpoints:
        val = mapping.get(u, 1.0)
        if isinstance(val, dict):
            val = val.get("__value", 1.0)
        try:
            out[u] = float(val)
        except Exception:
            out[u] = 1.0
    return out


def _build_attr_map(df, key_col: str) -> dict:
    """Build {key: attrs} mapping from a dataframe-like table."""
    out = {}
    for rec in _iter_rows(df):
        if not isinstance(rec, dict):
            try:
                rec = dict(rec)
            except Exception:
                continue
        if key_col not in rec:
            continue
        key = rec.get(key_col)
        if key is None:
            continue
        rec = dict(rec)
        rec.pop(key_col, None)
        if key not in out:
            out[key] = rec
    return out


def to_parquet(graph: AnnNet, path):
    """Write lossless GraphDir:
      vertices.parquet, edges.parquet, slices.parquet, edge_slices.parquet, manifest.json
    Wide tables (attrs as columns). Hyperedges stored with 'kind' and head/tail/members lists.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # vertices
    v_attr_map = _build_attr_map(getattr(graph, "vertex_attributes", None), "vertex_id")
    v_rows = []
    for v in graph.vertices():
        row = {"vertex_id": v}
        attrs = v_attr_map.get(v)
        if attrs:
            row.update(attrs)
        v_rows.append(row)
    _write_parquet_df(
        build_dataframe_from_rows(v_rows),
        path / "vertices.parquet",
        compression="zstd",
    )

    # edges
    e_attr_map = _build_attr_map(getattr(graph, "edge_attributes", None), "edge_id")
    e_rows = []
    for eidx in range(graph.number_of_edges()):
        eid = graph.idx_to_edge[eidx]
        S, T = graph.get_edge(eidx)
        kind = graph.edge_kind.get(eid)
        row = {
            "edge_id": eid,
            "kind": ("hyper" if kind == "hyper" else "binary"),
            "directed": bool(_is_directed_eid(graph, eid)),
            "weight": float(getattr(graph, "edge_weights", {}).get(eid, 1.0)),
        }
        attrs = e_attr_map.get(eid)
        if attrs:
            # Filter out structural columns to prevent contamination
            attrs = {
                k: v
                for k, v in attrs.items()
                if k
                not in (
                    "head",
                    "tail",
                    "members",
                    "source",
                    "target",
                    "kind",
                    "directed",
                    "weight",
                    "edge_id",
                )
            }
            row.update(attrs)

        if row["kind"] == "binary":
            members = S | T
            if len(members) == 1:
                u = next(iter(members))
                v = u
            else:
                u, v = sorted(members)
            row.update({"source": u, "target": v})
        else:
            head_map = _endpoint_coeff_map(row, "__source_attr", S) or dict.fromkeys(S or [], 1.0)
            tail_map = _endpoint_coeff_map(row, "__target_attr", T) or dict.fromkeys(T or [], 1.0)
            # DEBUG
            #print(f"Exporting hyperedge {eid}: S={S}, T={T}, directed={row['directed']}")
            #print(f"  head_map={head_map}, tail_map={tail_map}")

            row.update(
                {
                    "head": list(head_map.keys()),
                    "tail": list(tail_map.keys()),
                    "members": list({*head_map.keys(), *tail_map.keys()})
                    if not row["directed"]
                    else None,
                }
            )
        #if row["kind"] == "hyper":
            #print(f"Row to append: {row}")

        e_rows.append(row)

    #print(f"Sample e_rows[0] if hyper: {[r for r in e_rows if r['kind'] == 'hyper'][0]}")

    _write_parquet_df(
        build_dataframe_from_rows(e_rows),
        path / "edges.parquet",
        compression="zstd",
    )

    # slices
    L = []
    try:
        for lid in graph.list_slices(include_default=True):
            L.append({"slice_id": lid})
    except Exception:
        pass
    _write_parquet_df(
        build_dataframe_from_rows(L),
        path / "slices.parquet",
        compression="zstd",
    )

    # edge_slices
    EL = []
    try:
        for lid in graph.list_slices(include_default=True):
            for eid in graph.get_slice_edges(lid):
                rec = {"slice_id": lid, "edge_id": eid}
                try:
                    w = graph.get_edge_slice_attr(lid, eid, "weight", default=None)
                except Exception:
                    try:
                        w = graph.get_edge_slice_attr(lid, eid, "weight")
                    except Exception:
                        w = None
                if w is not None:
                    rec["weight"] = float(w)
                EL.append(rec)
    except Exception:
        pass
    _write_parquet_df(
        build_dataframe_from_rows(EL),
        path / "edge_slices.parquet",
        compression="zstd",
    )

    # manifest.json (tiny)
    manifest = {
        "format_version": 1,
        "counts": {"V": len(v_rows), "E": len(e_rows), "slices": len(L)},
        "schema": {"edges.kind": ["binary", "hyper"]},
        "provenance": {"package": "annnet"},
        "multilayer": {
            "aspects": list(getattr(graph, "aspects", [])),
            "aspect_attrs": dict(getattr(graph, "_aspect_attrs", {})),
            "elem_layers": dict(getattr(graph, "elem_layers", {})),
            "VM": _serialize_VM(getattr(graph, "_VM", set())),
            "edge_kind": dict(getattr(graph, "edge_kind", {})),
            "edge_layers": _serialize_edge_layers(getattr(graph, "edge_layers", {})),
            "node_layer_attrs": _serialize_node_layer_attrs(
                getattr(graph, "_vertex_layer_attrs", {})
            ),
            "layer_tuple_attrs": _serialize_layer_tuple_attrs(getattr(graph, "_layer_attrs", {})),
            "layer_attributes": _safe_df_to_rows(getattr(graph, "layer_attributes", None)),
        },
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2))


def from_parquet(path) -> AnnNet:
    """Read GraphDir (lossless) using bulk ops for speed."""
    from ..core.graph import AnnNet

    path = Path(path)
    V = _read_parquet(path / "vertices.parquet")
    E = _read_parquet(path / "edges.parquet")
    L = _read_parquet(path / "slices.parquet") if (path / "slices.parquet").exists() else None
    EL = (
        _read_parquet(path / "edge_slices.parquet")
        if (path / "edge_slices.parquet").exists()
        else None
    )

    H = AnnNet()

    # -------------------------
    # Vertices (bulk)
    # -------------------------
    # Convert vertices DF to dict rows once and bulk add
    v_rows = []
    for rec in _iter_rows(V):
        v_rows.append(dict(rec))
    if v_rows:
        H.add_vertices_bulk(v_rows)

    # -------------------------
    # Edges (bulk, columnar)
    # -------------------------
    # Split binary vs hyper first (avoid row-wise graph ops)
    try:
        # Polars / Narwhals fast path
        binary = E.filter(E["kind"] == "binary")
        hyper = E.filter(E["kind"] == "hyper")
        is_polars_like = True
    except Exception:
        # Fallback: materialize rows and split
        rows = list(_iter_rows(E))
        binary = [r for r in rows if r.get("kind") == "binary"]
        hyper = [r for r in rows if r.get("kind") == "hyper"]
        is_polars_like = False

    # ---- Binary edges ----
    if is_polars_like:
        # Columnar extraction
        src = binary.get_column("source").to_list() if "source" in binary.columns else []
        dst = binary.get_column("target").to_list() if "target" in binary.columns else []
        eids = binary.get_column("edge_id").to_list()
        directed = binary.get_column("directed").to_list() if "directed" in binary.columns else [True] * len(eids)
        weights = binary.get_column("weight").to_list() if "weight" in binary.columns else [1.0] * len(eids)

        # Build minimal dicts for bulk add
        edge_rows = (
            {"source": u, "target": v, "edge_id": eid, "edge_directed": bool(d)}
            for u, v, eid, d in zip(src, dst, eids, directed)
            if u is not None and v is not None
        )
        H.add_edges_bulk(edge_rows)

        # Apply weights in batch
        for eid, w in zip(eids, weights):
            H.edge_weights[eid] = float(w)

        # Remaining edge attrs (vectorized -> rows, but small)
        # Drop structural columns before attaching attrs
        drop_cols = {"edge_id", "kind", "directed", "weight", "source", "target", "head", "tail", "members"}
        for rec in _iter_rows(binary):
            eid = rec.get("edge_id")
            attrs = {k: v for k, v in rec.items() if k not in drop_cols}
            attrs = _strip_nulls(attrs)
            if attrs:
                H.set_edge_attrs(eid, **attrs)

    else:
        # Fallback path (still bulk, but from Python rows)
        edge_rows = []
        weights = {}
        extra_attrs = {}
        for rec in binary:
            rec = dict(rec)
            eid = rec.pop("edge_id")
            u = rec.pop("source", None)
            v = rec.pop("target", None)
            d = bool(rec.pop("directed", True))
            w = float(rec.pop("weight", 1.0))
            if u is None or v is None:
                continue
            edge_rows.append({"source": u, "target": v, "edge_id": eid, "edge_directed": d})
            weights[eid] = w
            attrs = _strip_nulls({k: v for k, v in rec.items() if k not in ("kind", "head", "tail", "members")})
            if attrs:
                extra_attrs[eid] = attrs

        if edge_rows:
            H.add_edges_bulk(edge_rows)
            for eid, w in weights.items():
                H.edge_weights[eid] = w
            if extra_attrs:
                H.set_edge_attrs_bulk(extra_attrs)

    # ---- Hyperedges ----
    if is_polars_like:
        eids = hyper.get_column("edge_id").to_list()
        directed = hyper.get_column("directed").to_list() if "directed" in hyper.columns else [False] * len(eids)
        weights = hyper.get_column("weight").to_list() if "weight" in hyper.columns else [1.0] * len(eids)

        heads = hyper.get_column("head").to_list() if "head" in hyper.columns else [None] * len(eids)
        tails = hyper.get_column("tail").to_list() if "tail" in hyper.columns else [None] * len(eids)
        members = hyper.get_column("members").to_list() if "members" in hyper.columns else [None] * len(eids)

        hyper_rows = []
        for eid, d, h, t, m in zip(eids, directed, heads, tails, members):
            d = bool(d)
            if d:
                hh = _as_list_or_empty(h)
                tt = _as_list_or_empty(t)
                if hh and tt:
                    hyper_rows.append({"head": list(hh), "tail": list(tt), "edge_id": eid, "edge_directed": True})
            else:
                mm = _as_list_or_empty(m)
                if not mm:
                    mm = list(set(_as_list_or_empty(h)) | set(_as_list_or_empty(t)))
                if len(mm) >= 2:
                    hyper_rows.append({"members": list(mm), "edge_id": eid, "edge_directed": False})

        if hyper_rows:
            H.add_hyperedges_bulk(hyper_rows)

        for eid, w in zip(eids, weights):
            H.edge_weights[eid] = float(w)

        # Extra attrs
        drop_cols = {"edge_id", "kind", "directed", "weight", "source", "target", "head", "tail", "members"}
        extra = {}
        for rec in _iter_rows(hyper):
            eid = rec.get("edge_id")
            attrs = {k: v for k, v in rec.items() if k not in drop_cols}
            attrs = _strip_nulls(attrs)
            if attrs:
                extra[eid] = attrs
        if extra:
            H.set_edge_attrs_bulk(extra)

    else:
        hyper_rows = []
        weights = {}
        extra_attrs = {}
        for rec in hyper:
            rec = dict(rec)
            eid = rec.pop("edge_id")
            d = bool(rec.pop("directed", False))
            w = float(rec.pop("weight", 1.0))
            h = _as_list_or_empty(rec.pop("head", None))
            t = _as_list_or_empty(rec.pop("tail", None))
            m = _as_list_or_empty(rec.pop("members", None))

            if d:
                if h and t:
                    hyper_rows.append({"head": list(h), "tail": list(t), "edge_id": eid, "edge_directed": True})
            else:
                if not m:
                    m = list(set(h) | set(t))
                if len(m) >= 2:
                    hyper_rows.append({"members": list(m), "edge_id": eid, "edge_directed": False})

            weights[eid] = w
            attrs = _strip_nulls({k: v for k, v in rec.items() if k != "kind"})
            if attrs:
                extra_attrs[eid] = attrs

        if hyper_rows:
            H.add_hyperedges_bulk(hyper_rows)
            for eid, w in weights.items():
                H.edge_weights[eid] = w
            if extra_attrs:
                H.set_edge_attrs_bulk(extra_attrs)

    # -------------------------
    # Slices
    # -------------------------
    if L is not None:
        for rec in _iter_rows(L):
            lid = rec.get("slice_id")
            try:
                if lid not in set(H.list_slices(include_default=True)):
                    H.add_slice(lid)
            except Exception:
                pass

    # -------------------------
    # Edge slices (bulk add edges to slice)
    # -------------------------
    if EL is not None:
        by_slice = {}
        slice_weights = {}
        for rec in _iter_rows(EL):
            lid = rec.get("slice_id")
            eid = rec.get("edge_id")
            if lid is None or eid is None:
                continue
            by_slice.setdefault(lid, []).append(eid)
            if "weight" in rec and rec["weight"] is not None:
                slice_weights.setdefault(lid, {})[eid] = float(rec["weight"])

        for lid, eids in by_slice.items():
            try:
                H.add_edges_to_slice_bulk(lid, eids)
            except Exception:
                for eid in eids:
                    try:
                        H.add_edge_to_slice(lid, eid)
                    except Exception:
                        pass

        for lid, mp in slice_weights.items():
            for eid, w in mp.items():
                try:
                    H.set_edge_slice_attrs(lid, eid, weight=w)
                except Exception:
                    try:
                        H.set_edge_slice_attr(lid, eid, "weight", w)
                    except Exception:
                        pass

    # -------------------------
    # Manifest (unchanged)
    # -------------------------
    manifest_path = path / "manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            mm = manifest.get("multilayer", {})

            aspects = mm.get("aspects", [])
            elem_layers = mm.get("elem_layers", {})

            if aspects:
                H.aspects = list(aspects)
                H.elem_layers = dict(elem_layers or {})
                H._rebuild_all_layers_cache()

            aspect_attrs = mm.get("aspect_attrs", {})
            if aspect_attrs:
                H._aspect_attrs.update(aspect_attrs)

            VM_data = mm.get("VM", [])
            if VM_data:
                H._VM = _deserialize_VM(VM_data)

            ek = mm.get("edge_kind", {})
            el_ser = mm.get("edge_layers", {})
            if ek:
                H.edge_kind.update(ek)
            if el_ser:
                H.edge_layers.update(_deserialize_edge_layers(el_ser))

            nl_attrs_ser = mm.get("node_layer_attrs", [])
            if nl_attrs_ser:
                H._vertex_layer_attrs = _deserialize_node_layer_attrs(nl_attrs_ser)

            layer_tuple_attrs_ser = mm.get("layer_tuple_attrs", [])
            if layer_tuple_attrs_ser:
                H._layer_attrs = _deserialize_layer_tuple_attrs(layer_tuple_attrs_ser)

            layer_attr_rows = mm.get("layer_attributes", [])
            if layer_attr_rows:
                H.layer_attributes = build_dataframe_from_rows(layer_attr_rows)

        except Exception:
            pass

    return H
