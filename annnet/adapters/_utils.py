# ---- robust helpers (keep in sync across adapters) ----
import ast as _ast
import json as _json
from enum import Enum
from typing import Any


def _is_directed_eid(graph, eid):
    """Best-effort directedness probe; default True."""
    try:
        rec = getattr(graph, "_edges", {}).get(eid)
        if rec is not None and rec.directed is not None:
            return bool(rec.directed)
    except Exception:
        pass
    try:
        v = graph.get_attr_edge(eid, "directed")
        return bool(v) if v is not None else True
    except Exception:
        return True


def _coerce_coeff_mapping(val):
    """Normalize endpoint-coeff containers into {vertex: {__value: float}|float}.
    Accepts dict | list | list-of-dicts | list-of-pairs | JSON string.
    """
    if val is None:
        return {}
    if isinstance(val, str):
        try:
            return _coerce_coeff_mapping(_json.loads(val))
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


def _serialize_value(val: Any) -> Any:
    if isinstance(val, Enum):
        return val.name
    if hasattr(val, "items"):
        return dict(val)
    return val


def _attrs_to_dict(attrs_dict: dict) -> dict:
    out = {}
    for key, val in attrs_dict.items():
        if isinstance(val, Enum):
            out[key] = val.name
        elif hasattr(val, "items"):
            out[key] = {
                inner_key: (inner_val.name if isinstance(inner_val, Enum) else inner_val)
                for inner_key, inner_val in dict(val).items()
            }
        else:
            out[key] = val
    return out


def _serialize_endpoint(endpoint: Any) -> Any:
    """Convert an endpoint into a JSON-safe representation."""
    if isinstance(endpoint, tuple) and len(endpoint) == 2 and isinstance(endpoint[1], tuple):
        return {"kind": "supra", "vertex": endpoint[0], "layer": list(endpoint[1])}
    return endpoint


def _deserialize_endpoint(value: Any) -> Any:
    """Restore a structural endpoint from JSON-safe or legacy serialized forms."""
    if isinstance(value, dict) and value.get("kind") == "supra":
        return (value.get("vertex"), tuple(value.get("layer") or []))
    if isinstance(value, list) and len(value) == 2 and isinstance(value[1], list):
        return (value[0], tuple(value[1]))
    if isinstance(value, str):
        try:
            parsed = _json.loads(value)
        except Exception:
            parsed = None
        if parsed is not None and parsed != value:
            return _deserialize_endpoint(parsed)
        if value.startswith("(") and value.endswith(")"):
            try:
                parsed = _ast.literal_eval(value)
            except Exception:
                parsed = None
            if (
                isinstance(parsed, tuple)
                and len(parsed) == 2
                and isinstance(parsed[1], (tuple, list))
            ):
                return (parsed[0], tuple(parsed[1]))
    return value


def _rows_like(table):
    if table is None:
        return []
    if hasattr(table, "to_dicts"):
        try:
            return list(table.to_dicts())
        except Exception:
            pass
    if hasattr(table, "to_dict"):
        try:
            recs = table.to_dict(orient="records")
            if isinstance(recs, list):
                return recs
        except Exception:
            pass
    if hasattr(table, "to_pylist"):
        try:
            return list(table.to_pylist())
        except Exception:
            pass
    if hasattr(table, "fetchall") and hasattr(table, "columns"):
        try:
            cols = list(table.columns)
            return [dict(zip(cols, row)) for row in table.fetchall()]
        except Exception:
            pass
    if isinstance(table, dict):
        keys = list(table.keys())
        if keys and isinstance(table[keys[0]], list):
            n = len(table[keys[0]])
            return [{k: table[k][i] for k in keys} for i in range(n)]
    if isinstance(table, list) and table and isinstance(table[0], dict):
        return list(table)
    return []


def save_manifest(manifest: dict, path: str):
    with open(path, "w") as f:
        _json.dump(manifest, f, indent=2)


def load_manifest(path: str) -> dict:
    with open(path) as f:
        return _json.load(f)


def _endpoint_coeff_map(edge_attrs, private_key, endpoint_set):
    """Return {vertex: float_coeff} for the given endpoint_set.
    Reads from edge_attrs[private_key], which may be serialized in multiple shapes.
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


# Serialization helpers moved from graphtool_adapter.py

from typing import Any, Optional

from .._dataframe_backend import dataframe_from_rows, dataframe_height, dataframe_to_rows


def _serialize_edge_layers(edge_layers: dict[str, Any]) -> dict[str, Any]:
    """
    Convert edge_layers[eid] (aa or (aa, bb)) into JSON-safe form.

    - intra:  aa -> {"kind": "single", "layers": [list(aa)]}
    - inter/coupling: (aa, bb) -> {"kind": "pair", "layers": [list(aa), list(bb)]}
    """
    out = {}
    for eid, L in edge_layers.items():
        if L is None:
            continue
        # e.g. intra: L == aa (tuple[str,...])
        if isinstance(L, tuple) and (len(L) == 0 or isinstance(L[0], str)):
            out[eid] = {"kind": "single", "layers": [list(L)]}
        # inter/coupling: L == (aa, bb)
        elif (
            isinstance(L, tuple)
            and len(L) == 2
            and isinstance(L[0], tuple)
            and isinstance(L[1], tuple)
        ):
            out[eid] = {"kind": "pair", "layers": [list(L[0]), list(L[1])]}
        else:
            # fallback: just repr it
            out[eid] = {"kind": "raw", "value": repr(L)}
    return out


def _deserialize_edge_layers(data: dict[str, Any]) -> dict[str, Any]:
    """
    Inverse of _serialize_edge_layers.

    Returns eid -> aa or (aa, bb) (tuples).
    """
    out = {}
    for eid, rec in data.items():
        kind = rec.get("kind")
        if kind == "single":
            aa = tuple(rec["layers"][0])
            out[eid] = aa
        elif kind == "pair":
            La = tuple(rec["layers"][0])
            Lb = tuple(rec["layers"][1])
            out[eid] = (La, Lb)
        else:
            # unknown / raw -> ignore or store as is
            # here we just skip, user can handle it manually if needed
            continue
    return out


def _serialize_VM(VM: set[tuple[str, tuple[str, ...]]]) -> list[dict]:
    """
    Serialize V_M = {(u, aa)} to JSON-safe list of dicts.
    """
    return [{"node": u, "layer": list(aa)} for (u, aa) in VM]


def _deserialize_VM(data: list[dict]) -> set[tuple[str, tuple[str, ...]]]:
    """
    Inverse of _serialize_VM.
    """
    return {(rec["node"], tuple(rec["layer"])) for rec in data}


def _serialize_node_layer_attrs(nl_attrs: dict[tuple[str, tuple[str, ...]], dict]) -> list[dict]:
    """
    (u, aa) -> {attrs}  ->  [{"node": u, "layer": list(aa), "attrs": {...}}, ...]
    """
    out = []
    for (u, aa), attrs in nl_attrs.items():
        out.append(
            {
                "node": u,
                "layer": list(aa),
                "attrs": dict(attrs),
            }
        )
    return out


def _deserialize_node_layer_attrs(data: list[dict]) -> dict[tuple[str, tuple[str, ...]], dict]:
    """
    Inverse of _serialize_node_layer_attrs.
    """
    out: dict[tuple[str, tuple[str, ...]], dict] = {}
    for rec in data:
        key = (rec["node"], tuple(rec["layer"]))
        out[key] = dict(rec.get("attrs", {}))
    return out


def _serialize_slices(slices: dict[str, dict]) -> dict[str, dict]:
    """
    _slices is {slice_id: {"vertices": set, "edges": set, "attributes": dict}}
    Convert sets to lists for JSON.
    """
    out = {}
    for sid, rec in slices.items():
        out[sid] = {
            "vertices": list(rec.get("vertices", [])),
            "edges": list(rec.get("edges", [])),
            "attributes": dict(rec.get("attributes", {})),
        }
    return out


def _deserialize_slices(data: dict[str, dict]) -> dict[str, dict]:
    """
    Inverse of _serialize_slices.
    """
    out = {}
    for sid, rec in data.items():
        out[sid] = {
            "vertices": set(rec.get("vertices", [])),
            "edges": set(rec.get("edges", [])),
            "attributes": dict(rec.get("attributes", {})),
        }
    return out


def _df_to_rows(df) -> list[dict]:
    """
    Convert a dataframe-like table to list-of-dicts in a stable way.
    """
    if df is None or dataframe_height(df) == 0:
        return []
    return dataframe_to_rows(df)


def _rows_to_df(rows: list[dict]):
    """
    Build a dataframe from list-of-dicts with the configured backend.
    """
    return dataframe_from_rows(rows)


def _serialize_layer_tuple_attrs(layer_attrs: dict[tuple[str, ...], dict]) -> list[dict]:
    """
    _layer_attrs: {aa_tuple -> {attr_name: value}}
    -> JSON-safe: [{"layer": list(aa), "attrs": {...}}, ...]
    """
    out = []
    for aa, attrs in layer_attrs.items():
        out.append({"layer": list(aa), "attrs": dict(attrs)})
    return out


def _deserialize_layer_tuple_attrs(data: list[dict]) -> dict[tuple[str, ...], dict]:
    """
    Inverse of _serialize_layer_tuple_attrs.
    """
    out: dict[tuple[str, ...], dict] = {}
    for rec in data:
        aa = tuple(rec["layer"])
        out[aa] = dict(rec.get("attrs", {}))
    return out


def _safe_df_to_rows(df):
    """Return list[dict] rows from dataframe-like input; return [] on None."""
    if df is None:
        return []
    return _df_to_rows(df)

