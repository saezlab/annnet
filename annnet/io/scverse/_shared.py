"""Shared helpers for scverse bridges."""

from __future__ import annotations

import copy
import json
import math
from typing import Any

from scipy import sparse
import pandas as pd

from .._common import (
    dataframe_to_rows,
    serialize_endpoint,
    dataframe_from_rows,
    deserialize_endpoint,
    serialize_edge_layers,
    collect_slice_manifest,
    restore_slice_manifest,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)

ANNNET_UNS_KEY = '__annnet__'
ANNNET_ENCODING = 'annnet-scverse'
ANNNET_VERSION = 1

_OBS_VERTEX_ID_COL = 'annnet_vertex_id'
_OBS_LAYER_PREFIX = 'annnet_layer_'
_STRUCTURAL_VAR_COLUMNS = {
    'source',
    'target',
    'head',
    'tail',
    'members',
    'weight',
    'directed',
    'edge_type',
    'multilayer_kind',
    'edge_layers',
}


def require_dependency(module_name: str, install_hint: str):
    """Import an optional dependency or raise a clear ImportError."""
    try:
        module = __import__(module_name, fromlist=['__name__'])
    except ImportError as exc:
        raise ImportError(
            f'{module_name} is required for this scverse bridge. Install {install_hint}.'
        ) from exc
    return module


def is_nullish(value: Any) -> bool:
    """Return whether a scalar-like value should be treated as missing."""
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
    except TypeError:
        pass
    try:
        result = pd.isna(value)
    except Exception:  # noqa: BLE001
        return False
    return bool(result) if isinstance(result, (bool, type(pd.NA))) else False


def graph_table_rows(df: Any) -> list[dict[str, Any]]:
    """Materialize a dataframe-like table as plain Python row dicts."""
    return [dict(row) for row in dataframe_to_rows(df)] if df is not None else []


def rows_to_backend_table(rows: list[dict[str, Any]], *, backend: str) -> Any:
    """Build a backend-native table from row dictionaries."""
    if not rows:
        return dataframe_from_rows([], backend=backend)
    return dataframe_from_rows(rows, backend=backend)


def copy_graph_uns(graph_uns: dict[str, Any]) -> dict[str, Any]:
    """Deep-copy graph-level metadata for embedding in AnnData.uns."""
    try:
        return copy.deepcopy(dict(graph_uns or {}))
    except Exception:  # noqa: BLE001
        return dict(graph_uns or {})


def _vertex_entities(graph) -> list[tuple[str, tuple[str, ...], int]]:
    """Return vertex entities in row order as (vertex_id, layer_coord, row_idx)."""
    entities: list[tuple[str, tuple[str, ...], int]] = []
    for row_idx in range(len(graph._row_to_entity)):
        ekey = graph._row_to_entity.get(row_idx)
        if ekey is None:
            continue
        rec = graph._entities.get(ekey)
        if rec is None or rec.kind != 'vertex':
            continue
        entities.append((ekey[0], ekey[1], rec.row_idx))
    return entities


def _vertex_obs_name(vertex_id: str, layer_coord: tuple[str, ...], *, is_multilayer: bool) -> str:
    """Build a stable AnnData obs name for a vertex entity."""
    if not is_multilayer:
        return str(vertex_id)
    payload = serialize_endpoint((vertex_id, layer_coord))
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _edge_endpoint_text(value: Any) -> Any:
    """Serialize structural endpoint values into AnnData-friendly cells."""
    if value is None:
        return None
    if isinstance(value, (list, tuple, set, frozenset)):
        seq = [serialize_endpoint(item) for item in list(value)]
        return json.dumps(seq, ensure_ascii=False, sort_keys=True)
    encoded = serialize_endpoint(value)
    if isinstance(encoded, dict):
        return json.dumps(encoded, ensure_ascii=False, sort_keys=True)
    return encoded


def _decode_endpoint_cell(value: Any) -> Any:
    """Inverse of `_edge_endpoint_text` for a scalar endpoint."""
    if is_nullish(value):
        return None
    return deserialize_endpoint(value)


def _decode_endpoint_seq(value: Any) -> list[Any]:
    """Inverse of `_edge_endpoint_text` for endpoint sequences."""
    if is_nullish(value):
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
    else:
        parsed = value
    if isinstance(parsed, (list, tuple)):
        return [deserialize_endpoint(item) for item in parsed]
    return [deserialize_endpoint(parsed)]


def _attr_map(
    rows: list[dict[str, Any]], key_col: str, *, include_private: bool
) -> dict[str, dict[str, Any]]:
    """Return key -> attrs mapping from raw row dictionaries."""
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = row.get(key_col)
        if key is None:
            continue
        attrs = {}
        for col, value in row.items():
            if col == key_col:
                continue
            if not include_private and str(col).startswith('__'):
                continue
            if is_nullish(value):
                continue
            attrs[col] = value
        out[key] = attrs
    return out


def build_obs_dataframe(graph, *, include_private: bool) -> pd.DataFrame:
    """Materialize AnnNet vertex entities into an AnnData obs dataframe."""
    vertex_attrs = _attr_map(
        graph_table_rows(graph.vertex_attributes), 'vertex_id', include_private=include_private
    )
    rows: list[dict[str, Any]] = []
    index: list[str] = []
    is_multilayer = bool(graph.aspects)
    for vertex_id, layer_coord, _row_idx in _vertex_entities(graph):
        index.append(_vertex_obs_name(vertex_id, layer_coord, is_multilayer=is_multilayer))
        row = dict(vertex_attrs.get(vertex_id, {}))
        row[_OBS_VERTEX_ID_COL] = vertex_id
        if is_multilayer:
            for aspect, value in zip(graph.aspects, layer_coord, strict=False):
                row[f'{_OBS_LAYER_PREFIX}{aspect}'] = value
        rows.append(row)
    obs = pd.DataFrame(rows, index=pd.Index(index, dtype='object'))
    obs.index.name = 'obs_id'
    return obs


def build_var_dataframe(graph, *, include_private: bool) -> pd.DataFrame:
    """Materialize AnnNet structural edges into an AnnData var dataframe."""
    edge_attrs = _attr_map(
        graph_table_rows(graph.edge_attributes), 'edge_id', include_private=include_private
    )
    rows: list[dict[str, Any]] = []
    index: list[str] = []
    default_directed = True if graph.directed is None else bool(graph.directed)

    for edge_id in graph.edges():
        rec = graph._edges[edge_id]
        row = {
            'weight': float(rec.weight if rec.weight is not None else 1.0),
            'directed': bool(default_directed if rec.directed is None else rec.directed),
            'edge_type': rec.etype,
            'multilayer_kind': rec.ml_kind,
            'edge_layers': _edge_endpoint_text(rec.ml_layers),
        }
        if rec.etype == 'hyper':
            if rec.tgt is None:
                row['members'] = _edge_endpoint_text(sorted(rec.src, key=repr))
            else:
                row['head'] = _edge_endpoint_text(sorted(rec.src, key=repr))
                row['tail'] = _edge_endpoint_text(sorted(rec.tgt, key=repr))
        else:
            row['source'] = _edge_endpoint_text(rec.src)
            row['target'] = _edge_endpoint_text(rec.tgt)
        row.update(edge_attrs.get(edge_id, {}))
        rows.append(row)
        index.append(edge_id)

    var = pd.DataFrame(rows, index=pd.Index(index, dtype='object'))
    var.index.name = 'edge_id'
    return var


def build_vertex_incidence(graph) -> sparse.csr_matrix:
    """Return the vertex-only incidence matrix aligned to the exported obs rows."""
    row_indexes = [row_idx for _vertex_id, _layer, row_idx in _vertex_entities(graph)]
    col_indexes = [graph._edges[edge_id].col_idx for edge_id in graph.edges()]
    matrix = graph.X().tocsr()
    if not row_indexes:
        return sparse.csr_matrix((0, len(col_indexes)), dtype=matrix.dtype)
    if not col_indexes:
        return sparse.csr_matrix((len(row_indexes), 0), dtype=matrix.dtype)
    return matrix[row_indexes, :][:, col_indexes]


def build_annnet_manifest(graph) -> dict[str, Any]:
    """Collect the AnnNet-only structural state needed for lossless restoration."""
    slices_section, slice_weights = collect_slice_manifest(graph)
    multilayer = serialize_multilayer_manifest(
        graph,
        table_to_rows=graph_table_rows,
        serialize_edge_layers=serialize_edge_layers,
    )
    return {
        'encoding': ANNNET_ENCODING,
        'version': ANNNET_VERSION,
        'directed': graph.directed,
        'active_slice': graph.slices.active,
        'graph_uns': copy_graph_uns(graph.uns),
        'vertex_attrs': graph_table_rows(graph.vertex_attributes),
        'edge_attrs': graph_table_rows(graph.edge_attributes),
        'slice_attrs': graph_table_rows(graph.slice_attributes),
        'edge_slice_attrs': graph_table_rows(graph.edge_slice_attributes),
        'slices': slices_section,
        'slice_weights': slice_weights,
        'multilayer': multilayer,
    }


def add_vertices_from_obs(graph, obs: pd.DataFrame) -> None:
    """Restore vertices from AnnData.obs using the AnnNet structural columns when present."""
    layer_cols = [col for col in obs.columns if str(col).startswith(_OBS_LAYER_PREFIX)]
    if layer_cols:
        aspects = [str(col)[len(_OBS_LAYER_PREFIX) :] for col in layer_cols]
        elem_layers = {
            aspect: sorted(
                {
                    str(val)
                    for val in obs[f'{_OBS_LAYER_PREFIX}{aspect}'].tolist()
                    if not is_nullish(val)
                }
            )
            for aspect in aspects
        }
        graph.layers.set_aspects(aspects, elem_layers)
        # Bucket vertices by layer tuple, then bulk-add per group —
        # per-row add_vertices is O(rows) graph mutations.
        by_layer: dict[tuple, list[str]] = {}
        for obs_name, row in obs.iterrows():
            vertex_id = row.get(_OBS_VERTEX_ID_COL, obs_name)
            layer = tuple(str(row[f'{_OBS_LAYER_PREFIX}{aspect}']) for aspect in aspects)
            by_layer.setdefault(layer, []).append(str(vertex_id))
        for layer, vids in by_layer.items():
            graph.add_vertices(vids, layer=layer)
        return

    # Flat graph: one bulk call.
    vids = [str(row.get(_OBS_VERTEX_ID_COL, obs_name)) for obs_name, row in obs.iterrows()]
    if vids:
        graph.add_vertices(vids)


def restore_multilayer(graph, manifest: dict[str, Any]) -> None:
    """Restore AnnNet multilayer metadata from an exported manifest."""
    if not manifest:
        return
    restore_multilayer_manifest(
        graph,
        manifest,
        rows_to_table=lambda rows: rows_to_backend_table(
            rows, backend=getattr(graph, '_annotations_backend', 'auto')
        ),
        deserialize_edge_layers=deserialize_edge_layers,
    )


def restore_attrs_from_manifest(graph, manifest: dict[str, Any]) -> None:
    """Restore graph, vertex, edge, slice, and edge-slice attributes from the manifest."""
    graph.uns.update(copy_graph_uns(manifest.get('graph_uns', {})))

    vertex_updates = {
        row['vertex_id']: {k: v for k, v in row.items() if k != 'vertex_id' and not is_nullish(v)}
        for row in manifest.get('vertex_attrs', [])
        if row.get('vertex_id') is not None
    }
    if vertex_updates:
        graph.attrs.set_vertex_attrs_bulk(vertex_updates)

    edge_updates = {
        row['edge_id']: {k: v for k, v in row.items() if k != 'edge_id' and not is_nullish(v)}
        for row in manifest.get('edge_attrs', [])
        if row.get('edge_id') is not None
    }
    if edge_updates:
        graph.attrs.set_edge_attrs_bulk(edge_updates)

    restore_slice_manifest(
        graph,
        manifest.get('slices', {}) or {},
        manifest.get('slice_weights', {}) or {},
    )

    for row in manifest.get('slice_attrs', []):
        slice_id = row.get('slice_id')
        if slice_id is None:
            continue
        attrs = {k: v for k, v in row.items() if k != 'slice_id' and not is_nullish(v)}
        if attrs:
            graph.attrs.set_slice_attrs(slice_id, **attrs)

    grouped_edge_slice: dict[str, dict[str, dict[str, Any]]] = {}
    for row in manifest.get('edge_slice_attrs', []):
        slice_id = row.get('slice_id')
        edge_id = row.get('edge_id')
        if slice_id is None or edge_id is None:
            continue
        attrs = {
            k: v for k, v in row.items() if k not in {'slice_id', 'edge_id'} and not is_nullish(v)
        }
        if attrs:
            grouped_edge_slice.setdefault(slice_id, {})[edge_id] = attrs
    for slice_id, updates in grouped_edge_slice.items():
        graph.attrs.set_edge_slice_attrs_bulk(slice_id, updates)

    active_slice = manifest.get('active_slice')
    if active_slice is not None and graph.slices.exists(active_slice):
        graph.slices.active = active_slice


def infer_directed_from_var(var: pd.DataFrame) -> bool | None:
    """Infer graph-level directedness from AnnData.var when no manifest is present."""
    if 'directed' not in var.columns:
        return None
    values = [bool(v) for v in var['directed'].tolist() if not is_nullish(v)]
    if not values:
        return None
    return values[0] if all(v == values[0] for v in values) else None


def restore_vertices_from_obs_attrs(graph, obs: pd.DataFrame) -> None:
    """Restore vertex attributes from AnnData.obs columns in the generic path."""
    layer_cols = {col for col in obs.columns if str(col).startswith(_OBS_LAYER_PREFIX)}
    updates: dict[str, dict[str, Any]] = {}
    for obs_name, row in obs.iterrows():
        vertex_id = str(row.get(_OBS_VERTEX_ID_COL, obs_name))
        attrs = {}
        for key, value in row.items():
            if key == _OBS_VERTEX_ID_COL or key in layer_cols or is_nullish(value):
                continue
            attrs[str(key)] = value
        if attrs:
            updates.setdefault(vertex_id, {}).update(attrs)
    if updates:
        graph.attrs.set_vertex_attrs_bulk(updates)


def add_edges_from_var(graph, var: pd.DataFrame) -> None:
    """Restore AnnNet edges from AnnData.var structural columns.

    Buckets rows by edge kind (binary, undirected-hyper, directed-hyper)
    and bulk-adds each bucket — per-row ``add_edges`` is O(rows) graph
    mutations.
    """
    binary_rows: list = []
    undirected_hyper_rows: list = []
    directed_hyper_rows: list = []

    for edge_id, row in var.iterrows():
        weight = row.get('weight', 1.0)
        weight = 1.0 if is_nullish(weight) else float(weight)
        directed = row.get('directed', graph.directed)
        directed = None if is_nullish(directed) else bool(directed)

        if not is_nullish(row.get('members')):
            members = _decode_endpoint_seq(row.get('members'))
            undirected_hyper_rows.append(
                {
                    'members': members,
                    'edge_id': str(edge_id),
                    'edge_directed': False,
                    'weight': weight,
                }
            )
        elif not is_nullish(row.get('head')) or not is_nullish(row.get('tail')):
            head = _decode_endpoint_seq(row.get('head'))
            tail = _decode_endpoint_seq(row.get('tail'))
            directed_hyper_rows.append(
                {
                    'head': head,
                    'tail': tail,
                    'edge_id': str(edge_id),
                    'edge_directed': True if directed is None else directed,
                    'weight': weight,
                }
            )
        else:
            source = _decode_endpoint_cell(row.get('source'))
            target = _decode_endpoint_cell(row.get('target'))
            if source is None or target is None:
                raise ValueError(
                    f'Cannot restore edge {edge_id!r} from AnnData.var: missing source/target.'
                )
            entry = {
                'source': source,
                'target': target,
                'edge_id': str(edge_id),
                'weight': weight,
            }
            if directed is not None:
                entry['edge_directed'] = directed
            binary_rows.append(entry)

    if binary_rows:
        graph.add_edges_bulk(binary_rows)
    if undirected_hyper_rows:
        graph.add_hyperedges_bulk(undirected_hyper_rows)
    if directed_hyper_rows:
        graph.add_hyperedges_bulk(directed_hyper_rows)


def restore_edge_attrs_from_var(graph, var: pd.DataFrame) -> None:
    """Restore edge attributes from the generic AnnData.var path."""
    updates: dict[str, dict[str, Any]] = {}
    for edge_id, row in var.iterrows():
        attrs = {}
        for key, value in row.items():
            if key in _STRUCTURAL_VAR_COLUMNS or is_nullish(value):
                continue
            attrs[str(key)] = value
        if attrs:
            updates[str(edge_id)] = attrs
    if updates:
        graph.attrs.set_edge_attrs_bulk(updates)


def obs_spatial_matrix(
    obs: pd.DataFrame,
    *,
    spatial_columns: tuple[str, str] | None = None,
) -> tuple[str, Any] | None:
    """Extract a 2D spatial coordinate matrix from obs if requested or detectable."""
    candidates = []
    if spatial_columns is not None:
        candidates.append(tuple(spatial_columns))
    candidates.extend(
        [
            ('spatial_x', 'spatial_y'),
            ('x', 'y'),
            ('x_coord', 'y_coord'),
            ('pxl_col_in_fullres', 'pxl_row_in_fullres'),
        ]
    )
    for x_col, y_col in candidates:
        if x_col in obs.columns and y_col in obs.columns:
            coords = obs[[x_col, y_col]].to_numpy()
            return 'spatial', coords
    return None
