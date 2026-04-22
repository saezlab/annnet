"""
CX2 Adapter for AnnNet AnnNet.

Provides:
    to_cx2(G)        -> List[Dict[str, Any]] (CX2 JSON object)
    from_cx2(cx2_data) -> AnnNet

This adapter maps AnnNet Graphs to the Cytoscape Exchange (CX2) format.
It creates a lossless representation by serializing complex features
(hyperedges, slices, multilayer) into a 'manifest' stored within the
CX2 'networkAttributes'.
"""

from __future__ import annotations

import gzip
import json
import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..core.graph import AnnNet
from ..core._helpers import EntityRecord
from ..adapters._utils import (
    _df_to_rows,
    _serialize_VM,
    _safe_df_to_rows,
    _serialize_slices,
    _serialize_edge_layers,
    _serialize_node_layer_attrs,
    _serialize_layer_tuple_attrs,
)
from .._dataframe_backend import dataframe_columns, dataframe_from_rows, rename_dataframe_columns

# --- Helpers ---
CX_STYLE_KEY = '__cx_style__'


def _cx2_collect_reified(aspects):
    """
    Detect reified hyperedges from CX2 nodes + edges.

    Returns
    -------
      hyperdefs: list of (eid, directed, head_map, tail_map, attrs, he_node_id)
      membership_edges: set of edge-ids in CX2 that belong to hyperedge membership structure.
    """
    nodes = aspects.get('nodes', [])
    edges = aspects.get('edges', [])

    he_nodes = {}
    for n in nodes:
        v = n.get('v', {})
        if v.get('is_hyperedge', False):
            eid = v.get('eid')
            if eid is None:
                continue
            he_nodes[n['id']] = (eid, v)

    if not he_nodes:
        return [], set()

    # Build adjacency around hyperedge nodes
    hyperdefs = []
    membership_edges = set()

    for he_id, (eid, attrs) in he_nodes.items():
        head_map = {}
        tail_map = {}

        for e in edges:
            u = e['s']
            v = e['t']
            vid = e['id']
            ev = e.get('v', {})

            if u != he_id and v != he_id:
                continue

            membership_edges.add(vid)
            other = v if u == he_id else u
            role = ev.get('role', None)
            coeff = float(ev.get('weight', ev.get('coeff', 1.0)))

            if role == 'head':
                head_map[other] = coeff
            elif role == 'tail':
                tail_map[other] = coeff
            else:
                # undirected membership
                head_map[other] = coeff
                tail_map[other] = coeff

        # Determine directedness
        if any(k for k in (head_map or {})) or any(k for k in (tail_map or {})):
            directed = (
                True if any(ev.get('role') in ('head', 'tail') for ev in attrs.values()) else False
            )
        else:
            directed = False

        hyperdefs.append((eid, directed, head_map, tail_map, attrs, he_id))

    return hyperdefs, membership_edges


def _rows_to_df(rows):
    # --- 1) Normalize all rows to full schema ---
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    norm = [{k: r.get(k, None) for k in keys} for r in rows]
    return dataframe_from_rows(norm)


def _infer_cx2_type(values: list[Any]) -> str:
    """Infer a CX2 attribute data type string from Python row values."""
    nonnull = [value for value in values if value is not None]
    if not nonnull:
        return 'string'
    first = nonnull[0]
    if isinstance(first, bool):
        return 'boolean'
    if isinstance(first, int) and not isinstance(first, bool):
        return 'long'
    if isinstance(first, float):
        return 'double'
    if isinstance(first, (list, tuple)):
        flattened = [
            item for value in nonnull if isinstance(value, (list, tuple)) for item in value
        ]
        return f'list_of_{_infer_cx2_type(flattened)}'
    return 'string'


def _infer_cx2_types(rows: list[dict[str, Any]], *, id_col: str | None = None) -> dict[str, str]:
    keys = sorted({key for row in rows for key in row})
    return {key: _infer_cx2_type([row.get(key) for row in rows]) for key in keys if key != id_col}


def _jsonify(obj):
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    else:
        return obj


# --- Core Adapter: to_cx2 ---


def to_cx2(
    G: AnnNet,
    *,
    export_name='annnet export',
    layer=None,
    include_inter=False,
    include_coupling=False,
    hyperedges='skip',
) -> list[dict[str, Any]]:
    """
    Convert an AnnNet graph to CX2 compliant JSON format.

    The output is a list of aspect dictionaries (CX2 format).
    Complex AnnNet features (hyperedges, slices, multilayer structure) are
    serialized into a JSON string stored in networkAttributes under '__AnnNet_Manifest__'.

    Parameters
    ----------
    G : AnnNet
        The graph to export.
    export_name : str, default "annnet export"
        Name of the exported network (stored in networkAttributes).
    layer : tuple of str, optional
        Elementary layer tuple specifying which layer to export. If provided,
        only the subgraph for that layer is exported. Useful for multilayer graphs
        where flattening creates unreadable visualizations in Cytoscape (e.g.,
        coupling edges become self-loops). If None, exports the entire graph.
        Example: layer=("social", "2020") for a 2-aspect multilayer network.
    hyperedges : {"skip", "expand", "reify"}, default "skip"
        How to handle hyperedges in the export:
        - "skip": Omit hyperedges entirely
        - "expand": Convert to cartesian product of pairwise edges
        - "reify": Create explicit hyperedge nodes with membership edges

    Returns
    -------
    list of dict
        CX2-compliant JSON structure (list of aspect dictionaries).

    Notes
    -----
    - Cytoscape does not natively support multilayer networks. When exporting
      multilayer graphs without specifying a layer, coupling edges may appear
      as self-loops and the visualization becomes cluttered.
    - Use the `layer` parameter to export individual elementary layers for
      clean, interpretable Cytoscape visualizations.
    - The full multilayer structure is preserved in the manifest regardless
      of the `layer` parameter, enabling lossless round-trip via from_cx2().
    """

    if layer is not None:
        if not isinstance(layer, tuple):
            raise TypeError(f'layer must be a tuple, got {type(layer).__name__}')
        G = G.subgraph_from_layer_tuple(
            layer, include_coupling=include_coupling, include_inter=include_inter
        )

    # 1. Prepare Manifest (Lossless storage of complex features)
    vert_rows = _safe_df_to_rows(getattr(G, 'vertex_attributes', None))
    edge_rows = _safe_df_to_rows(getattr(G, 'edge_attributes', None))
    slice_rows = _safe_df_to_rows(getattr(G, 'slice_attributes', None))
    edge_slice_rows = _safe_df_to_rows(getattr(G, 'edge_slice_attributes', None))
    layer_attr_rows = _safe_df_to_rows(getattr(G, 'layer_attributes', None))

    # strip CX-specific style from what we embed into the manifest
    g_attrs = dict(getattr(G, 'graph_attributes', {}))
    g_attrs.pop(CX_STYLE_KEY, None)

    manifest = {
        'version': 1,
        'graph': {
            'directed': bool(G.directed) if G.directed is not None else True,
            'attributes': g_attrs,
        },
        'vertices': {
            'types': {ekey[0]: ent.kind for ekey, ent in G._entities.items()},
            'attributes': vert_rows,
        },
        'edges': {
            'definitions': {
                eid: (rec.src, rec.tgt, rec.etype)
                for eid, rec in G._edges.items()
                if rec.etype != 'hyper'
            },
            'weights': {eid: rec.weight for eid, rec in G._edges.items() if rec.weight is not None},
            'directed': {
                eid: bool(rec.directed) for eid, rec in G._edges.items() if rec.directed is not None
            },
            'direction_policy': dict(getattr(G, 'edge_direction_policy', {})),
            'hyperedges': {
                eid: (
                    {'directed': True, 'head': list(rec.src or []), 'tail': list(rec.tgt or [])}
                    if rec.tgt is not None
                    else {'directed': False, 'members': list(rec.src or [])}
                )
                for eid, rec in G._edges.items()
                if rec.etype == 'hyper'
            },
            'attributes': edge_rows,
            'kivela': {
                'edge_kind': {
                    eid: ('hyper' if rec.etype == 'hyper' else rec.ml_kind)
                    for eid, rec in G._edges.items()
                    if rec.etype == 'hyper' or rec.ml_kind is not None
                },
                'edge_layers': _serialize_edge_layers(getattr(G, 'edge_layers', {})),
            },
        },
        'slices': {
            'data': _serialize_slices(getattr(G, '_slices', {})),
            'slice_attributes': slice_rows,
            'edge_slice_attributes': edge_slice_rows,
        },
        'multilayer': {
            'aspects': list(getattr(G, 'aspects', [])),
            'aspect_attrs': dict(getattr(G, '_aspect_attrs', {})),
            'elem_layers': dict(getattr(G, 'elem_layers', {})),
            'VM': _serialize_VM(getattr(G, '_VM', set())),
            'edge_kind': {
                eid: ('hyper' if rec.etype == 'hyper' else rec.ml_kind)
                for eid, rec in G._edges.items()
                if rec.etype == 'hyper' or rec.ml_kind is not None
            },
            'edge_layers': _serialize_edge_layers(getattr(G, 'edge_layers', {})),
            'node_layer_attrs': _serialize_node_layer_attrs(getattr(G, '_state_attrs', {})),
            'layer_tuple_attrs': _serialize_layer_tuple_attrs(getattr(G, '_layer_attrs', {})),
            'layer_attributes': layer_attr_rows,
        },
        'tables': {
            'vertex_attributes': vert_rows,
            'edge_attributes': edge_rows,
            'slice_attributes': slice_rows,
            'edge_slice_attributes': edge_slice_rows,
            'layer_attributes': layer_attr_rows,
        },
    }
    manifest['edges']['expanded'] = {}

    # 2. Build Core CX2 Aspects

    # ID Mapping: AnnNet ID (str/int) -> CX2 ID (int)
    node_map: dict[Any, int] = {}
    cx_nodes: list[dict[str, Any]] = []
    cx_edges: list[dict[str, Any]] = []

    def _clean_cx2_attrs(attrs, string_cols, list_cols=None):
        """Clean attributes: remove nulls, convert non-JSON types."""
        if not attrs:
            return {}
        if list_cols is None:
            list_cols = set()
        out = {}
        for k, v in attrs.items():
            if v is None:
                if k in string_cols:
                    out[k] = ''
                elif k in list_cols:
                    out[k] = []
                # else: drop null entirely (CX2 doesn't accept null)
            elif isinstance(v, (set, frozenset)):
                out[k] = list(v)
            elif isinstance(v, (str, int, float, bool)):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = [
                    str(x) if not isinstance(x, (str, int, float, bool, type(None))) else x
                    for x in v
                ]
            else:
                out[k] = str(v)  # fallback: stringify unknown types
        return out

    # -- Nodes --
    # We only map entities of type 'vertex' to visual nodes.
    current_node_id = 0

    # Pre-fetch attribute data for fast lookup
    v_attrs_df = getattr(G, 'vertex_attributes', None)

    # Identify which vertex columns are string vs numeric (for cleaning None)
    v_inferred_types = _infer_cx2_types(vert_rows, id_col='vertex_id')
    v_string_cols = {col for col, dtype in v_inferred_types.items() if dtype == 'string'}
    v_numeric_cols = {
        col for col, dtype in v_inferred_types.items() if dtype in {'integer', 'long', 'double'}
    }

    # Build map: vertex_id -> attribute row dict
    v_attrs_map: dict[str, dict[str, Any]] = {}
    if v_attrs_df is not None and vert_rows:
        row_keys = set(vert_rows[0])
        id_col = 'vertex_id' if 'vertex_id' in row_keys else 'id'
        for r in vert_rows:  # vert_rows is _df_to_rows(vertex_attributes)
            key = r.get(id_col)
            if key is not None:
                v_attrs_map[str(key)] = r

    def _clean_vertex_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
        """Replace None in strings with '', drop None for numeric/others."""
        out: dict[str, Any] = {}
        for k, v in attrs.items():
            if v is None:
                if k in v_string_cols:
                    out[k] = ''  # string: None -> ""
                elif k in v_numeric_cols:
                    # numeric: drop the key entirely (no null numbers in CX2)
                    continue
                else:
                    # unknown type: safest is to drop null
                    continue
            else:
                out[k] = v
        return out

    for uid, utype in G.entity_types.items():
        if utype != 'vertex':
            continue

        cx_id = current_node_id
        current_node_id += 1
        node_map[uid] = cx_id

        # base node object
        n_obj: dict[str, Any] = {
            'id': cx_id,
            'v': {'name': str(uid)},  # 'name' is standard in Cytoscape
        }

        coords: dict[str, float] = {}

        # Attach attributes if present
        row = v_attrs_map.get(str(uid))
        if row is not None:
            attrs = dict(row)  # copy
            # get rid of id column from attributes
            attrs.pop('id', None)
            attrs.pop('vertex_id', None)

            # pull layout_* into x/y/z for Cytoscape layout
            for src, dst in (('layout_x', 'x'), ('layout_y', 'y'), ('layout_z', 'z')):
                val = attrs.get(src)
                if val is not None:
                    try:
                        coords[dst] = float(val)
                    except (TypeError, ValueError):
                        pass
                # never expose layout_* as regular attributes
                attrs.pop(src, None)

            # clean None values: "" for strings, drop for numeric/others
            attrs = _clean_vertex_attrs(attrs)
            n_obj['v'].update(attrs)

        # put node coordinates at top-level (where Cytoscape expects them)
        if coords:
            n_obj.update(coords)

        cx_nodes.append(n_obj)

    # -- Edges --
    # Only binary edges between mapped vertices
    current_edge_id = 0
    e_attrs_df = getattr(G, 'edge_attributes', None)

    # string columns in edge_attributes (for None -> "")
    e_inferred_types = _infer_cx2_types(edge_rows, id_col='edge_id')
    e_string_cols = {col for col, dtype in e_inferred_types.items() if dtype == 'string'}

    # Create lookup for edge attributes (handle both 'edge_id' and 'id')
    e_attrs_map = {}
    if e_attrs_df is not None and edge_rows:
        row_keys = set(edge_rows[0])
        id_col = 'edge_id' if 'edge_id' in row_keys else 'id'
        for r in edge_rows:
            if id_col in r:
                e_attrs_map[str(r[id_col])] = r

    for eid, rec in G._edges.items():
        is_hyper = rec.etype == 'hyper'

        # --- Hyperedge handling ---
        if is_hyper:
            if hyperedges == 'skip':
                continue

            directed = rec.tgt is not None
            if directed:
                S = set(rec.src or [])
                T = set(rec.tgt or [])
            else:
                members = set(rec.src or [])
                S = members
                T = members

            if hyperedges == 'expand':
                exp_entry = {
                    'mode': 'directed' if directed else 'undirected',
                    'tail': list(T) if directed else None,
                    'head': list(S) if directed else None,
                    'members': list(S | T) if not directed else None,
                    'expanded_edges': [],
                }
                members = S | T
                if directed:
                    # tail -> head cartesian
                    for u in T:
                        for v in S:
                            exp_entry['expanded_edges'].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            raw_attrs = e_attrs_map.get(str(eid), {})
                            clean_attrs = _clean_cx2_attrs(raw_attrs, e_string_cols)
                            cx_edges.append(
                                {
                                    'id': cx_eid,
                                    's': node_map[u],
                                    't': node_map[v],
                                    'v': {
                                        'interaction': str(eid),
                                        'weight': float(1.0 if rec.weight is None else rec.weight),
                                        **clean_attrs,
                                    },
                                }
                            )

                else:
                    # undirected clique
                    mem = list(members)
                    for i in range(len(mem)):
                        for j in range(i + 1, len(mem)):
                            u, v = mem[i], mem[j]
                            exp_entry['expanded_edges'].append([u, v])
                            cx_eid = current_edge_id
                            current_edge_id += 1
                            raw_attrs = e_attrs_map.get(str(eid), {})
                            clean_attrs = _clean_cx2_attrs(raw_attrs, e_string_cols)
                            cx_edges.append(
                                {
                                    'id': cx_eid,
                                    's': node_map[u],
                                    't': node_map[v],
                                    'v': {
                                        'interaction': str(eid),
                                        'weight': float(1.0 if rec.weight is None else rec.weight),
                                        **clean_attrs,
                                    },
                                }
                            )
                manifest['edges']['expanded'][str(eid)] = exp_entry
                continue

            if hyperedges == 'reify':
                he_cx_id = current_node_id
                current_node_id += 1

                # filter edge attrs so we don't leak edge_id/id onto the node
                he_attrs = e_attrs_map.get(str(eid), {}).copy()
                he_attrs.pop('edge_id', None)
                he_attrs.pop('id', None)
                he_attrs = _clean_cx2_attrs(he_attrs, e_string_cols)

                he_node = {
                    'id': he_cx_id,
                    'v': {
                        'name': f'hyperedge::{eid}',
                        'is_hyperedge': True,
                        'eid': str(eid),
                        **he_attrs,
                    },
                }
                cx_nodes.append(he_node)

                weight = float(1.0 if rec.weight is None else rec.weight)

                if directed:
                    # tail -> HE
                    for u in T:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                'id': cx_eid,
                                's': node_map[u],
                                't': he_cx_id,
                                'v': {
                                    'interaction': f'{eid}::tail',
                                    'role': 'tail',
                                    'weight': weight,
                                },
                            }
                        )
                    # HE -> head
                    for v in S:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                'id': cx_eid,
                                's': he_cx_id,
                                't': node_map[v],
                                'v': {
                                    'interaction': f'{eid}::head',
                                    'role': 'head',
                                    'weight': weight,
                                },
                            }
                        )

                else:
                    # undirected: HE - members
                    members = S | T
                    for u in members:
                        cx_eid = current_edge_id
                        current_edge_id += 1
                        cx_edges.append(
                            {
                                'id': cx_eid,
                                's': node_map[u],
                                't': he_cx_id,
                                'v': {
                                    'interaction': f'{eid}::member',
                                    'role': 'member',
                                    'weight': weight,
                                },
                            }
                        )
                continue

            # unknown hyperedge mode - skip
            continue

        u, v = rec.src, rec.tgt

        if u not in node_map or v not in node_map:
            continue

        cx_u = node_map[u]
        cx_v = node_map[v]
        cx_eid = current_edge_id
        current_edge_id += 1

        e_obj = {
            'id': cx_eid,
            's': cx_u,
            't': cx_v,
            'v': {
                'interaction': str(eid),
                'weight': float(1.0 if rec.weight is None else rec.weight),
            },
        }

        # Attach attributes
        if str(eid) in e_attrs_map:
            attrs = e_attrs_map[str(eid)].copy()
            # Remove redundant keys
            attrs.pop('edge_id', None)
            attrs.pop('id', None)
            attrs = _clean_cx2_attrs(attrs, e_string_cols)
            e_obj['v'].update(attrs)

        cx_edges.append(e_obj)

    # 3. Attribute Declarations
    attr_decls = {'nodes': {}, 'edges': {}, 'networkAttributes': {}}

    # Define Node Attributes
    for col, dtype in v_inferred_types.items():
        if col == 'id':
            continue
        attr_decls['nodes'][col] = {'d': dtype}

    attr_decls['nodes']['name'] = {'d': 'string'}  # Always added
    attr_decls['nodes']['is_hyperedge'] = {'d': 'boolean'}
    attr_decls['nodes']['eid'] = {'d': 'string'}
    attr_decls['nodes']['tag'] = {'d': 'string'}
    attr_decls['nodes']['reaction'] = {'d': 'string'}

    # Define Edge Attributes
    id_col = 'edge_id' if edge_rows and 'edge_id' in edge_rows[0] else 'id'
    for col, dtype in e_inferred_types.items():
        if col == id_col:
            continue
        attr_decls['edges'][col] = {'d': dtype}
    attr_decls['edges']['interaction'] = {'d': 'string'}
    attr_decls['edges']['weight'] = {'d': 'double'}
    attr_decls['edges']['edge_id'] = {'d': 'string'}
    attr_decls['edges']['role'] = {'d': 'string'}

    # Define Network Attributes
    # We store the manifest as a JSON string to ensure compatibility
    attr_decls['networkAttributes']['__AnnNet_Manifest__'] = {'d': 'string'}
    attr_decls['networkAttributes']['name'] = {'d': 'string'}
    attr_decls['networkAttributes']['directed'] = {'d': 'boolean'}

    # 4. Construct Final CX2 List

    # Start with basic metadata
    meta = [
        {'name': 'attributeDeclarations', 'elementCount': 1},
        {'name': 'networkAttributes', 'elementCount': 1},
        {'name': 'nodes', 'elementCount': len(cx_nodes)},
        {'name': 'edges', 'elementCount': len(cx_edges)},
    ]

    cx2: list[dict[str, Any]] = [
        {'CXVersion': '2.0', 'hasFragments': False},
        {'metaData': meta},
        {'attributeDeclarations': [attr_decls]},
        {
            'networkAttributes': [
                {
                    'name': export_name,
                    'directed': bool(G.directed) if G.directed is not None else True,
                    '__AnnNet_Manifest__': base64.b64encode(
                        gzip.compress(json.dumps(_jsonify(manifest)).encode())
                    ).decode(),
                }
            ]
        },
        {'nodes': cx_nodes},
        {'edges': cx_edges},
    ]

    # Re-emit Cytoscape visual style if we have it
    style = dict(getattr(G, 'graph_attributes', {})).get(CX_STYLE_KEY, {}) or {}

    vp = style.get('visualProperties')
    if vp:
        meta.append({'name': 'visualProperties', 'elementCount': 1})
        cx2.append({'visualProperties': [vp]})

    nb = style.get('nodeBypasses')
    if nb:
        meta.append({'name': 'nodeBypasses', 'elementCount': len(nb)})
        cx2.append({'nodeBypasses': nb})

    eb = style.get('edgeBypasses')
    if eb:
        meta.append({'name': 'edgeBypasses', 'elementCount': len(eb)})
        cx2.append({'edgeBypasses': eb})

    vep = style.get('visualEditorProperties')
    if vep:
        meta.append({'name': 'visualEditorProperties', 'elementCount': 1})
        cx2.append({'visualEditorProperties': [vep]})

    # Status goes last
    cx2.append({'status': [{'success': True}]})

    return cx2


# --- Core Adapter: from_cx2 ---


def from_cx2(cx2_data, *, hyperedges='manifest'):
    """
    Fully robust CX2 - AnnNet importer.

    Supports:
      - manifest reconstruction (full fidelity)
      - reified hyperedges
      - expanded hyperedges
      - Cytoscape-edited files
      - sparse attribute tables
      - arbitrary attribute modifications
    """
    # Small helper: normalize rows so Polars won't crash

    def _normalize_rows(rows):
        if not rows:
            return []
        keys = set().union(*(r.keys() for r in rows))
        return [{k: r.get(k, None) for k in keys} for r in rows]

    # Load file or JSON string

    import os
    import json

    if isinstance(cx2_data, str):
        if os.path.exists(cx2_data):
            with open(cx2_data) as f:
                cx2_data = json.load(f)
        else:
            try:
                cx2_data = json.loads(cx2_data)
            except Exception:  # noqa: BLE001
                raise ValueError('Invalid CX2 string or file') from None

    # Parse aspects into a dict

    aspects = {}
    for item in cx2_data:
        if not item:
            continue
        key = list(item.keys())[0]

        if key in ('CXVersion', 'metaData', 'status'):
            aspects[key] = item[key]
        else:
            aspects.setdefault(key, []).extend(item[key])

    # Extract networkAttributes + manifest JSON

    net_attrs = {}
    for na in aspects.get('networkAttributes', []):
        net_attrs.update(na)

    manifest_str = net_attrs.get('__AnnNet_Manifest__')
    manifest = None
    if manifest_str:
        try:
            # Support both compressed (gzip+base64) and legacy plain-JSON manifests
            try:
                manifest = json.loads(gzip.decompress(base64.b64decode(manifest_str)).decode())
            except Exception:  # noqa: BLE001
                manifest = json.loads(manifest_str)
        except Exception:  # noqa: BLE001
            manifest = None
    visual_props = aspects.get('visualProperties', [])

    # Extract Cytoscape visual style aspects (kept opaque but preserved)
    style_aspects: dict[str, Any] = {}

    vp = aspects.get('visualProperties')
    if vp:
        style_aspects['visualProperties'] = vp[0] if isinstance(vp, list) else vp

    nb = aspects.get('nodeBypasses')
    if nb:
        style_aspects['nodeBypasses'] = nb

    eb = aspects.get('edgeBypasses')
    if eb:
        style_aspects['edgeBypasses'] = eb

    vep = aspects.get('visualEditorProperties')
    if vep:
        style_aspects['visualEditorProperties'] = vep[0] if isinstance(vep, list) else vep

    # Construct AnnNet

    from annnet.core.graph import AnnNet

    G = AnnNet()

    # PATH A: MANIFEST RECONSTRUCTION

    if manifest and hyperedges in ('manifest', 'reified'):
        # --- Base graph attrs ---
        gmeta = manifest.get('graph', {})
        G.directed = gmeta.get('directed', True)
        G.graph_attributes = dict(gmeta.get('attributes', {}))

        if visual_props:
            G.graph_attributes['__cx_visualProperties__'] = visual_props

        # --- Vertices ---
        vmeta = manifest.get('vertices', {})
        v_rows = _normalize_rows(vmeta.get('attributes', []))
        if v_rows:
            G.vertex_attributes = _rows_to_df(v_rows)
        if vmeta.get('types'):
            for vid, kind in vmeta['types'].items():
                try:
                    ekey = G._resolve_entity_key(vid)
                except Exception:  # noqa: BLE001
                    continue
                if ekey in G._entities:
                    G._entities[ekey].kind = kind
                else:
                    row_idx = max(G._row_to_entity.keys(), default=-1) + 1
                    G._register_entity_record(ekey, EntityRecord(row_idx=row_idx, kind=kind))

        # --- Edges + hyperedges ---
        emeta = manifest.get('edges', {})

        # edge_attributes
        e_rows = _normalize_rows(emeta.get('attributes', []))
        if e_rows:
            G.edge_attributes = _rows_to_df(e_rows)

        # weights, directed flags, definitions
        if emeta.get('weights'):
            for eid, w in emeta['weights'].items():
                rec = G._edges.get(eid)
                if rec is not None:
                    rec.weight = float(w)
        if emeta.get('directed'):
            for eid, val in emeta['directed'].items():
                rec = G._edges.get(eid)
                if rec is not None:
                    rec.directed = bool(val)
        if emeta.get('definitions'):
            for eid, defn in emeta['definitions'].items():
                rec = G._edges.get(eid)
                if rec is None:
                    continue
                rec.src, rec.tgt, rec.etype = defn
        if emeta.get('direction_policy'):
            G.edge_direction_policy.update(emeta['direction_policy'])

        # hyperedge definitions
        if emeta.get('hyperedges'):
            fixed = {}
            for eid, info in emeta['hyperedges'].items():
                # Older / simple manifests store hyperedges as a plain list of members
                # e.g. "he1": ["n1", "n2", "n3"]
                if isinstance(info, list):
                    fixed[eid] = {
                        'directed': False,
                        'members': set(info),
                    }
                    continue

                # Newer manifests: dict form with keys like "directed", "members" or "head"/"tail"
                directed = bool(info.get('directed', False))
                if directed:
                    fixed[eid] = {
                        'directed': True,
                        'head': set(info.get('head', [])),
                        'tail': set(info.get('tail', [])),
                    }
                else:
                    fixed[eid] = {
                        'directed': False,
                        'members': set(info.get('members', [])),
                    }
            for eid, info in fixed.items():
                rec = G._edges.get(eid)
                if rec is None:
                    continue
                rec.etype = 'hyper'
                if info['directed']:
                    rec.src = list(info.get('head', []))
                    rec.tgt = list(info.get('tail', []))
                    rec.directed = True
                else:
                    rec.src = list(info.get('members', []))
                    rec.tgt = None
                    rec.directed = False

        # --- Expanded hyperedges (if present) ---
        exp = emeta.get('expanded', {})
        if exp:
            hyperedge_bulk_data = []
            for eid, info in exp.items():
                directed = info.get('mode') == 'directed'
                if directed:
                    hyperedge_bulk_data.append(
                        {
                            'head': info.get('head', []),
                            'tail': info.get('tail', []),
                            'edge_id': eid,
                            'edge_directed': True,
                        }
                    )
                else:
                    hyperedge_bulk_data.append(
                        {
                            'members': info.get('members', []),
                            'edge_id': eid,
                            'edge_directed': False,
                        }
                    )

            if hyperedge_bulk_data:
                G.add_hyperedges_bulk(hyperedge_bulk_data)

        # --- Layers (Kivela)---
        kiv = emeta.get('kivela', {})
        if kiv.get('edge_kind'):
            for eid, kind in kiv['edge_kind'].items():
                rec = G._edges.get(eid)
                if rec is None:
                    continue
                if kind == 'hyper':
                    rec.etype = 'hyper'
                else:
                    rec.ml_kind = kind
        if kiv.get('edge_layers'):
            G.edge_layers.update(kiv['edge_layers'])

        # --- Slices ---
        smeta = manifest.get('slices', {})
        if smeta.get('data'):
            for sname, sdata in smeta['data'].items():
                verts = set(sdata.get('vertices', []))
                edgs = set(sdata.get('edges', []))
                attrs = dict(sdata.get('attributes', {}))

                G._slices[sname] = {
                    'vertices': verts,
                    'edges': edgs,
                    'attributes': attrs,
                }
        if smeta.get('slice_attributes'):
            G.slice_attributes = _rows_to_df(_normalize_rows(smeta['slice_attributes']))
        if smeta.get('edge_slice_attributes'):
            G.edge_slice_attributes = _rows_to_df(_normalize_rows(smeta['edge_slice_attributes']))

        # --- Multilayer ---
        mm = manifest.get('multilayer', {})
        if mm.get('aspects'):
            G.aspects = mm['aspects']
        if mm.get('elem_layers'):
            G.elem_layers = dict(mm['elem_layers'])
        if mm.get('aspect_attrs'):
            G._aspect_attrs = mm['aspect_attrs']
        if mm.get('node_layer_attrs'):
            G._state_attrs = mm['node_layer_attrs']
        if mm.get('layer_tuple_attrs'):
            G._layer_attrs = mm['layer_tuple_attrs']
        if mm.get('layer_attributes'):
            G.layer_attributes = _rows_to_df(_normalize_rows(mm['layer_attributes']))

        # --- OPTIONAL: overlay reified hyperedges ---
        if hyperedges == 'reified':
            _cx2_collect_reified(aspects, G)

    # PATH B: NO MANIFEST

    else:
        directed = net_attrs.get('directed', True)
        G = AnnNet(directed=directed)

        G.entity_types = {}
        if visual_props:
            # make sure we have a dict
            if not hasattr(G, 'graph_attributes') or G.graph_attributes is None:
                G.graph_attributes = {}
            G.graph_attributes['__cx_visualProperties__'] = visual_props

    # Overlay Cytoscape edits: nodes + edges from CX2

    # Track whether PATH A (manifest) was used — affects overlay attr handling
    _manifest_mode = bool(manifest and hyperedges in ('manifest', 'reified'))

    # Map CX numeric ids - AnnNet string ids
    cx2node = {}
    node_aspects = aspects.get('nodes', [])

    # --- build a row map of existing vertex attributes ---
    # In manifest mode the full attr table is already set; skip reading it back (expensive).
    # We only need vmap in PATH B where G starts empty.
    if _manifest_mode:
        vmap = {}
    else:
        vmap = {}
        existing = _df_to_rows(getattr(G, 'vertex_attributes', None))
        for r in existing:
            vid = str(r.get('vertex_id', r.get('id')))
            vmap[vid] = dict(r)

    # --- update vertex attrs ---
    vertex_bulk_data = []
    for n in node_aspects:
        cx_id = n['id']
        attrs = dict(n.get('v', {}))
        ann_id = str(attrs.get('name', cx_id))
        cx2node[cx_id] = ann_id

        row = vmap.get(ann_id, {'vertex_id': ann_id})

        # Merge attributes from Cytoscape (except display name)
        for k, v in attrs.items():
            if k != 'name':
                row[k] = v

        # Layout coordinates live on the node, not in v
        if 'x' in n and n['x'] is not None:
            row['layout_x'] = float(n['x'])
        if 'y' in n and n['y'] is not None:
            row['layout_y'] = float(n['y'])
        if 'z' in n and n['z'] is not None:
            row['layout_z'] = float(n['z'])

        vertex_bulk_data.append(row)

    # Single bulk vertex insert (registers entities; also upserts layout coords in manifest mode)
    if vertex_bulk_data:
        G.add_vertices_bulk(vertex_bulk_data)

    if _manifest_mode:
        # Manifest already set the full vertex_attributes; add_vertices_bulk has upserted any
        # new layout coords into it.  Just fix the column name if needed.
        cols = set(dataframe_columns(G.vertex_attributes))
        if 'vertex_id' not in cols and 'id' in cols:
            G.vertex_attributes = rename_dataframe_columns(G.vertex_attributes, {'id': 'vertex_id'})
    else:
        # rebuild vertex table
        if vertex_bulk_data:
            G.vertex_attributes = _rows_to_df(_normalize_rows(vertex_bulk_data))
        else:
            G.vertex_attributes = _rows_to_df([])

        # Normalise ID column name: prefer 'vertex_id' consistently
        cols = set(dataframe_columns(G.vertex_attributes))
        if 'vertex_id' not in cols and 'id' in cols:
            G.vertex_attributes = rename_dataframe_columns(G.vertex_attributes, {'id': 'vertex_id'})

    # --- edges ---
    # In manifest mode: edge_attributes is already set from the manifest — skip reading it back.
    if _manifest_mode:
        emap = {}
    else:
        emap = {}
        existing = _df_to_rows(getattr(G, 'edge_attributes', None))
        for r in existing:
            eid = str(r.get('edge_id', r.get('id')))
            emap[eid] = dict(r)

    edge_bulk_data = []
    for e in aspects.get('edges', []):
        s = cx2node.get(e['s'])
        t = cx2node.get(e['t'])
        if not s or not t:
            continue

        attrs = e.get('v', {})
        eid = str(attrs.get('edge_id', attrs.get('interaction', e['id'])))
        w = float(attrs.get('weight', 1.0))

        # Collect edge data for bulk insert
        edge_dict = {
            'source': s,
            'target': t,
            'edge_id': eid,
            'weight': w,
        }

        # Collect additional attributes (excluding interaction and weight)
        extra_attrs = {k: v for k, v in attrs.items() if k not in ('interaction', 'weight')}
        if extra_attrs:
            edge_dict['attributes'] = extra_attrs
            if not _manifest_mode:
                emap.setdefault(eid, {'edge_id': eid}).update(extra_attrs)

        edge_bulk_data.append(edge_dict)

    # Single bulk edge insert
    if edge_bulk_data:
        G.add_edges_bulk(edge_bulk_data)

    if not _manifest_mode:
        enorm = _normalize_rows(list(emap.values()))
        G.edge_attributes = _rows_to_df(enorm)

    # Attach Cytoscape style blob if we captured any
    if style_aspects:
        # make sure graph_attributes exists and is a dict
        G.graph_attributes = dict(getattr(G, 'graph_attributes', {}))
        G.graph_attributes[CX_STYLE_KEY] = style_aspects

    return G


# -------------------------------------------------- Browser visualzaion Cytoscape.js


def show_cx2(
    G: AnnNet,
    *,
    export_name='annnet export',
    layer=None,
    include_inter=False,
    include_coupling=False,
    hyperedges='skip',
    port=None,
    auto_open=True,
) -> str:
    """
    Visualize graph in web browser using Cytoscape.js.

    Parameters
    ----------
    G : AnnNet
        The graph to visualize.
    export_name : str, default "annnet export"
        Name of the network display.
    layer : tuple of str, optional
        Elementary layer tuple specifying which layer to visualize. If None,
        shows the entire graph (may include coupling edges as self-loops).
        Example: layer=("social", "2020") for a 2-aspect multilayer network.
    include_inter : bool, default False
        When layer is specified, whether to include interlayer edges (edges
        between nodes in different layers but same aspect). Only relevant if
        layer is not None.
    include_coupling : bool, default False
        When layer is specified, whether to include coupling edges (edges
        connecting the same node across layers). Only relevant if layer is not None.
        Warning: Including coupling creates self-loops in the visualization.
    hyperedges : {"skip", "expand", "reify"}, default "skip"
        How to handle hyperedges:
        - "skip": Omit hyperedges entirely
        - "expand": Convert to cartesian product of pairwise edges
        - "reify": Create explicit hyperedge nodes with membership edges
    port : int, optional
        Port for local web server. If None, finds available port automatically.
    auto_open : bool, default True
        If True, automatically open browser.

    Returns
    -------
    str
        Local URL to the visualization.

    Notes
    -----
    For multilayer graphs:
    - Without layer parameter: shows entire flattened graph with coupling edges
      appearing as self-loops (messy but shows full structure)
    - With layer parameter: shows clean single-layer view without coupling edges
      (recommended for visualization)

    Press Ctrl+C in terminal to stop the web server.

    Examples
    --------
    >>> show_cx2(G)  # Show entire graph (all layers flattened)
    >>> show_cx2(G, layer=('social', '2020'))  # Clean single layer view
    >>> show_cx2(G, layer=('social', '2020'), include_coupling=True)  # With self-loops
    """
    import json
    import socket
    from pathlib import Path
    import tempfile
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    if port is None:
        port = find_free_port()

    # to_cx2 already handles layer extraction internally
    cx2_data = to_cx2(
        G,
        export_name=export_name,
        layer=layer,
        include_inter=include_inter,
        include_coupling=include_coupling,
        hyperedges=hyperedges,
    )

    cytoscape_json = _cx2_to_cytoscapejs(cx2_data)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{export_name}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            #cy {{ width: 100vw; height: 100vh; }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255,255,255,0.9);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div id="info">
            <strong>{export_name}</strong><br>
            Nodes: <span id="nodeCount">0</span> | Edges: <span id="edgeCount">0</span>
        </div>
        <div id="cy"></div>
        <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {json.dumps(cytoscape_json)},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': '#0074D9',
                            'label': 'data(label)',
                            'width': 40,
                            'height': 40,
                            'font-size': '12px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#000',
                            'text-outline-width': 2,
                            'text-outline-color': '#fff'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.6
                        }}
                    }},
                    {{
                        selector: 'edge[source = target]',
                        style: {{
                            'curve-style': 'loop',
                            'loop-direction': '0deg',
                            'loop-sweep': '45deg',
                            'line-color': '#ff6b6b',
                            'target-arrow-color': '#ff6b6b',
                            'width': 1.5
                        }}
                    }}
                ],
                layout: {{
                    name: 'cose',
                    animate: true,
                    nodeRepulsion: 8000,
                    idealEdgeLength: 100,
                    numIter: 1000
                }}
            }});

            document.getElementById('nodeCount').textContent = cy.nodes().length;
            document.getElementById('edgeCount').textContent = cy.edges().length;
        </script>
    </body>
    </html>
    """

    temp_dir = tempfile.mkdtemp()
    html_path = Path(temp_dir) / 'index.html'
    html_path.write_text(html_content)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=temp_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    try:
        server = HTTPServer(('localhost', port), Handler)
    except OSError as e:
        if e.errno == 98:
            port = find_free_port()
            server = HTTPServer(('localhost', port), Handler)
        else:
            raise

    url = f'http://localhost:{port}'

    def run_server():
        print(f'Serving visualization at {url}')
        print('Press Ctrl+C to stop')
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    if auto_open:
        webbrowser.open(url)

    return url


def _cx2_to_cytoscapejs(cx2_data: list[dict]) -> dict:
    """Convert CX2 format to Cytoscape.js format."""
    nodes = []
    edges = []

    node_map = {}

    for aspect in cx2_data:
        if 'nodes' in aspect:
            for node in aspect['nodes']:
                node_id = str(node['id'])
                node_map[node['id']] = node_id
                label = node.get('v', {}).get('name', node_id)
                nodes.append({'data': {'id': node_id, 'label': label, **node.get('v', {})}})

        if 'edges' in aspect:
            for edge in aspect['edges']:
                source = str(edge['s'])
                target = str(edge['t'])
                edges.append(
                    {
                        'data': {
                            'id': str(edge['id']),
                            'source': source,
                            'target': target,
                            **edge.get('v', {}),
                        }
                    }
                )

    return {'nodes': nodes, 'edges': edges}
