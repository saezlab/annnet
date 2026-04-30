from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING
from pathlib import Path

import narwhals as nw

if TYPE_CHECKING:
    from ..core.graph import AnnNet

from ..adapters._utils import _safe_df_to_rows
from .._support.serialization import (
    endpoint_coeff_map,
    serialize_endpoint,
    deserialize_endpoint,
    serialize_edge_layers,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)
from .._support.dataframe_backend import (
    dataframe_from_rows,
    _dataframe_read_parquet,
    _dataframe_write_parquet,
)


def _build_dataframe_from_rows(rows):
    """Build a dataframe/table using AnnNet's configured backend selection."""
    if not rows:
        return dataframe_from_rows(rows)
    order = []
    for row in rows:
        for key in row.keys():
            if key not in order:
                order.append(key)
    df = dataframe_from_rows(rows)
    try:
        return nw.from_native(df, eager_only=True).select(order).to_native()
    except (AttributeError, TypeError, ValueError):
        return df


def _empty_table(columns: list[str]):
    df = dataframe_from_rows([dict.fromkeys(columns)])
    try:
        return nw.from_native(df, eager_only=True).head(0).to_native()
    except (AttributeError, TypeError, ValueError):
        if hasattr(df, 'head'):
            return df.head(0)
        return df


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
    kind = 'hyper' if eid in graph.hyperedge_definitions else 'binary'

    if eid in graph.edge_directed:
        return bool(graph.edge_directed[eid])

    # Check attribute
    val = graph.attrs.get_attr_edge(eid, 'directed', None)
    if val is not None:
        return bool(val)

    # For hyperedges, check if S and T are identical (undirected)
    if kind == 'hyper':
        info = graph.hyperedge_definitions.get(eid, {})
        S = set(info.get('head', info.get('members', [])))
        T = set(info.get('tail', info.get('members', [])))
        if S == T:
            return False

    # Default: True for binary, False for hyper
    return kind != 'hyper'


def _edge_weight(graph, eid: str) -> float:
    weight = graph.edge_weights.get(eid)
    return 1.0 if weight is None else float(weight)


def _is_nullish(val) -> bool:
    if val is None:
        return True
    try:
        if isinstance(val, float) and math.isnan(val):
            return True
    except TypeError:
        pass
    try:
        # Covers numpy scalar NaN without importing a concrete dataframe backend.
        maybe_nan = val != val
        if isinstance(maybe_nan, bool) and maybe_nan:
            return True
    except (TypeError, ValueError):
        pass
    return False


def _as_list_or_empty(val):
    if _is_nullish(val):
        return []

    # already list / tuple
    if isinstance(val, list):
        return val
    if isinstance(val, tuple):
        return list(val)
    if hasattr(val, 'to_list') and callable(val.to_list):
        return val.to_list()
    if hasattr(val, 'tolist') and callable(val.tolist):
        return val.tolist()

    # scalar -> singleton
    return [val]


def _build_attr_map(df, key_col: str) -> dict:
    """Build {key: attrs} mapping from a dataframe-like table."""
    out = {}
    for rec in _safe_df_to_rows(df):
        if not isinstance(rec, dict):
            try:
                rec = dict(rec)
            except (TypeError, ValueError):
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
    v_attr_map = _build_attr_map(getattr(graph, 'vertex_attributes', None), 'vertex_id')
    v_rows = []
    for v in graph.vertices():
        row = {'vertex_id': v}
        attrs = v_attr_map.get(v)
        if attrs:
            row.update(attrs)
        v_rows.append(row)
    vertex_df = _build_dataframe_from_rows(v_rows) if v_rows else _empty_table(['vertex_id'])
    _dataframe_write_parquet(vertex_df, path / 'vertices.parquet')

    # edges
    e_attr_map = _build_attr_map(getattr(graph, 'edge_attributes', None), 'edge_id')
    e_rows = []
    for eid, info in graph.hyperedge_definitions.items():
        S = set(info.get('head', info.get('members', [])))
        T = set(info.get('tail', info.get('members', [])))
        row = {
            'edge_id': eid,
            'kind': 'hyper',
            'directed': bool(_is_directed_eid(graph, eid)),
            'weight': _edge_weight(graph, eid),
        }
        attrs = e_attr_map.get(eid)
        if attrs:
            # Filter out structural columns to prevent contamination
            attrs = {
                k: v
                for k, v in attrs.items()
                if k
                not in (
                    'head',
                    'tail',
                    'members',
                    'source',
                    'target',
                    'kind',
                    'directed',
                    'weight',
                    'edge_id',
                )
            }
            row.update(attrs)

        head_map = endpoint_coeff_map(row, '__source_attr', S) or dict.fromkeys(S or [], 1.0)
        tail_map = endpoint_coeff_map(row, '__target_attr', T) or dict.fromkeys(T or [], 1.0)
        row.update(
            {
                'head': list(head_map.keys()),
                'tail': list(tail_map.keys()),
                'members': list({*head_map.keys(), *tail_map.keys()})
                if not row['directed']
                else None,
            }
        )
        e_rows.append(row)

    for eid, (src, tgt, _etype) in graph.edge_definitions.items():
        S = set() if src is None else {src}
        T = set() if tgt is None else {tgt}
        row = {
            'edge_id': eid,
            'kind': 'binary',
            'directed': bool(_is_directed_eid(graph, eid)),
            'weight': _edge_weight(graph, eid),
            'source': json.dumps(serialize_endpoint(src), ensure_ascii=False),
            'target': json.dumps(serialize_endpoint(tgt), ensure_ascii=False),
        }
        attrs = e_attr_map.get(eid)
        if attrs:
            attrs = {
                k: v
                for k, v in attrs.items()
                if k
                not in (
                    'head',
                    'tail',
                    'members',
                    'source',
                    'target',
                    'kind',
                    'directed',
                    'weight',
                    'edge_id',
                )
            }
            row.update(attrs)
        e_rows.append(row)

    edge_df = (
        _build_dataframe_from_rows(e_rows)
        if e_rows
        else _empty_table(
            ['edge_id', 'kind', 'directed', 'weight', 'source', 'target', 'head', 'tail', 'members']
        )
    )
    _dataframe_write_parquet(edge_df, path / 'edges.parquet')

    # slices
    L = [{'slice_id': lid} for lid in graph.slices.list_slices(include_default=True)]
    slices_df = _build_dataframe_from_rows(L) if L else _empty_table(['slice_id'])
    _dataframe_write_parquet(slices_df, path / 'slices.parquet')

    # edge_slices
    EL = []
    for lid in graph.slices.list_slices(include_default=True):
        for eid in graph.slices.get_slice_edges(lid):
            rec = {'slice_id': lid, 'edge_id': eid}
            try:
                w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight', default=None)
            except TypeError:
                w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight')
            if w is not None:
                rec['weight'] = float(w)
            EL.append(rec)
    edge_slices_df = (
        _build_dataframe_from_rows(EL) if EL else _empty_table(['slice_id', 'edge_id', 'weight'])
    )
    _dataframe_write_parquet(edge_slices_df, path / 'edge_slices.parquet')

    # manifest.json (tiny)
    manifest = {
        'format_version': 1,
        'counts': {'V': len(v_rows), 'E': len(e_rows), 'slices': len(L)},
        'schema': {'edges.kind': ['binary', 'hyper']},
        'provenance': {'package': 'annnet'},
        'multilayer': serialize_multilayer_manifest(
            graph,
            table_to_rows=_safe_df_to_rows,
            serialize_edge_layers=serialize_edge_layers,
        ),
    }
    (path / 'manifest.json').write_text(json.dumps(manifest, indent=2))


def from_parquet(path) -> AnnNet:
    """Read GraphDir (lossless) using bulk ops for speed."""
    from ..core.graph import AnnNet

    path = Path(path)
    V = _dataframe_read_parquet(path / 'vertices.parquet')
    E = _dataframe_read_parquet(path / 'edges.parquet')
    L = (
        _dataframe_read_parquet(path / 'slices.parquet')
        if (path / 'slices.parquet').exists()
        else None
    )
    EL = (
        _dataframe_read_parquet(path / 'edge_slices.parquet')
        if (path / 'edge_slices.parquet').exists()
        else None
    )

    H = AnnNet()

    # -------------------------
    # Vertices (bulk)
    # -------------------------
    # Convert vertices DF to dict rows once and bulk add
    v_rows = []
    for rec in _safe_df_to_rows(V):
        v_rows.append(dict(rec))
    if v_rows:
        H.add_vertices_bulk(v_rows)

    # -------------------------
    # Edges (bulk, columnar)
    # -------------------------
    # Split binary vs hyper first (avoid row-wise graph ops)
    rows = list(_safe_df_to_rows(E))
    if not rows:
        binary = []
        hyper = []
        is_polars_like = False
    else:
        try:
            # Polars / Narwhals fast path
            binary = E.filter(E['kind'] == 'binary')
            hyper = E.filter(E['kind'] == 'hyper')
            is_polars_like = True
        except (AttributeError, TypeError, NotImplementedError):
            # Fallback: materialize rows and split
            binary = [r for r in rows if r.get('kind') == 'binary']
            hyper = [r for r in rows if r.get('kind') == 'hyper']
            is_polars_like = False

    # ---- Binary edges ----
    if is_polars_like:
        # Columnar extraction
        src = binary.get_column('source').to_list() if 'source' in binary.columns else []
        dst = binary.get_column('target').to_list() if 'target' in binary.columns else []
        eids = binary.get_column('edge_id').to_list()
        directed = (
            binary.get_column('directed').to_list()
            if 'directed' in binary.columns
            else [True] * len(eids)
        )
        weights = (
            binary.get_column('weight').to_list()
            if 'weight' in binary.columns
            else [1.0] * len(eids)
        )

        # Build minimal dicts for bulk add
        edge_rows = (
            {
                'source': deserialize_endpoint(u),
                'target': deserialize_endpoint(v),
                'edge_id': eid,
                'edge_directed': bool(d),
                'weight': float(w),
            }
            for u, v, eid, d, w in zip(src, dst, eids, directed, weights, strict=False)
            if u is not None and v is not None
        )
        H.add_edges_bulk(edge_rows)

        # Remaining edge attrs (vectorized -> rows, but small)
        # Drop structural columns before attaching attrs
        drop_cols = {
            'edge_id',
            'kind',
            'directed',
            'weight',
            'source',
            'target',
            'head',
            'tail',
            'members',
        }
        for rec in _safe_df_to_rows(binary):
            eid = rec.get('edge_id')
            attrs = {k: v for k, v in rec.items() if k not in drop_cols}
            attrs = _strip_nulls(attrs)
            if attrs:
                H.attrs.set_edge_attrs(eid, **attrs)

    else:
        # Fallback path (still bulk, but from Python rows)
        edge_rows = []
        extra_attrs = {}
        for rec in binary:
            rec = dict(rec)
            eid = rec.pop('edge_id')
            u = deserialize_endpoint(rec.pop('source', None))
            v = deserialize_endpoint(rec.pop('target', None))
            d = bool(rec.pop('directed', True))
            w = float(rec.pop('weight', 1.0))
            if u is None or v is None:
                continue
            edge_rows.append(
                {'source': u, 'target': v, 'edge_id': eid, 'edge_directed': d, 'weight': w}
            )
            attrs = _strip_nulls(
                {k: v for k, v in rec.items() if k not in ('kind', 'head', 'tail', 'members')}
            )
            if attrs:
                extra_attrs[eid] = attrs

        if edge_rows:
            H.add_edges_bulk(edge_rows)
            if extra_attrs:
                H.attrs.set_edge_attrs_bulk(extra_attrs)

    # ---- Hyperedges ----
    if is_polars_like:
        eids = hyper.get_column('edge_id').to_list()
        directed = (
            hyper.get_column('directed').to_list()
            if 'directed' in hyper.columns
            else [False] * len(eids)
        )
        weights = (
            hyper.get_column('weight').to_list() if 'weight' in hyper.columns else [1.0] * len(eids)
        )

        heads = (
            hyper.get_column('head').to_list() if 'head' in hyper.columns else [None] * len(eids)
        )
        tails = (
            hyper.get_column('tail').to_list() if 'tail' in hyper.columns else [None] * len(eids)
        )
        members = (
            hyper.get_column('members').to_list()
            if 'members' in hyper.columns
            else [None] * len(eids)
        )

        hyper_rows = []
        for eid, d, h, t, m, w in zip(eids, directed, heads, tails, members, weights, strict=False):
            d = bool(d)
            if d:
                hh = _as_list_or_empty(h)
                tt = _as_list_or_empty(t)
                if hh and tt:
                    hyper_rows.append(
                        {
                            'head': list(hh),
                            'tail': list(tt),
                            'edge_id': eid,
                            'edge_directed': True,
                            'weight': float(w),
                        }
                    )
            else:
                mm = _as_list_or_empty(m)
                if not mm:
                    mm = list(set(_as_list_or_empty(h)) | set(_as_list_or_empty(t)))
                if len(mm) >= 2:
                    hyper_rows.append(
                        {
                            'members': list(mm),
                            'edge_id': eid,
                            'edge_directed': False,
                            'weight': float(w),
                        }
                    )

        if hyper_rows:
            H.add_hyperedges_bulk(hyper_rows)

        # Extra attrs
        drop_cols = {
            'edge_id',
            'kind',
            'directed',
            'weight',
            'source',
            'target',
            'head',
            'tail',
            'members',
        }
        extra = {}
        for rec in _safe_df_to_rows(hyper):
            eid = rec.get('edge_id')
            attrs = {k: v for k, v in rec.items() if k not in drop_cols}
            attrs = _strip_nulls(attrs)
            if attrs:
                extra[eid] = attrs
        if extra:
            H.attrs.set_edge_attrs_bulk(extra)

    else:
        hyper_rows = []
        extra_attrs = {}
        for rec in hyper:
            rec = dict(rec)
            eid = rec.pop('edge_id')
            d = bool(rec.pop('directed', False))
            w = float(rec.pop('weight', 1.0))
            h = _as_list_or_empty(rec.pop('head', None))
            t = _as_list_or_empty(rec.pop('tail', None))
            m = _as_list_or_empty(rec.pop('members', None))

            if d:
                if h and t:
                    hyper_rows.append(
                        {
                            'head': list(h),
                            'tail': list(t),
                            'edge_id': eid,
                            'edge_directed': True,
                            'weight': w,
                        }
                    )
            else:
                if not m:
                    m = list(set(h) | set(t))
                if len(m) >= 2:
                    hyper_rows.append(
                        {'members': list(m), 'edge_id': eid, 'edge_directed': False, 'weight': w}
                    )
            attrs = _strip_nulls({k: v for k, v in rec.items() if k != 'kind'})
            if attrs:
                extra_attrs[eid] = attrs

        if hyper_rows:
            H.add_hyperedges_bulk(hyper_rows)
            if extra_attrs:
                H.attrs.set_edge_attrs_bulk(extra_attrs)

    # -------------------------
    # Slices
    # -------------------------
    if L is not None:
        existing_slices = set(H.slices.list_slices(include_default=True))
        for rec in _safe_df_to_rows(L):
            lid = rec.get('slice_id')
            if lid is not None and lid not in existing_slices:
                H.slices.add_slice(lid)
                existing_slices.add(lid)

    # -------------------------
    # Edge slices (bulk add edges to slice)
    # -------------------------
    if EL is not None:
        by_slice = {}
        slice_weights = {}
        for rec in _safe_df_to_rows(EL):
            lid = rec.get('slice_id')
            eid = rec.get('edge_id')
            if lid is None or eid is None:
                continue
            by_slice.setdefault(lid, []).append(eid)
            if 'weight' in rec and rec['weight'] is not None:
                slice_weights.setdefault(lid, {})[eid] = float(rec['weight'])

        for lid, eids in by_slice.items():
            H.slices.add_edges(lid, eids)

        for lid, mp in slice_weights.items():
            for eid, w in mp.items():
                H.attrs.set_edge_slice_attrs(lid, eid, weight=w)

    # -------------------------
    # Manifest (unchanged)
    # -------------------------
    manifest_path = path / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        restore_multilayer_manifest(
            H,
            manifest.get('multilayer', {}),
            rows_to_table=_build_dataframe_from_rows,
            deserialize_edge_layers=deserialize_edge_layers,
        )

    return H
