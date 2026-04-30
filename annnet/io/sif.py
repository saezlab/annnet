from __future__ import annotations

import json
from collections.abc import Iterable

from ..core.graph import AnnNet
from .._support.graph_records import _rows_to_df
from .._support.serialization import (
    serialize_endpoint,
    deserialize_endpoint,
    serialize_edge_layers,
    deserialize_edge_layers,
    restore_multilayer_manifest,
    serialize_multilayer_manifest,
)
from .._support.dataframe_backend import dataframe_to_rows


def _split_sif_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is not None:
        return [t for t in line.rstrip('\n\r').split(delimiter) if t != '']
    if '\t' in line:
        return [t for t in line.rstrip('\n\r').split('\t') if t != '']
    return line.strip().split()


def _safe_vertex_attr_rows(graph: AnnNet):
    va = getattr(graph, 'vertex_attributes', None)
    if va is None:
        return []
    return dataframe_to_rows(va)


def _get_all_edge_attrs(graph: AnnNet, edge_id: str):
    ea = getattr(graph, 'edge_attributes', None)
    if ea is not None:
        for row in dataframe_to_rows(ea):
            if row.get('edge_id') == edge_id:
                attrs = dict(row)
                attrs.pop('edge_id', None)
                return {k: v for k, v in attrs.items() if v is not None}
    return {}


def _get_edge_weight(graph: AnnNet, edge_id: str, default=1.0):
    edge_weights = getattr(graph, 'edge_weights', None)
    if edge_weights is not None:
        weight = edge_weights.get(edge_id)
        if weight is not None:
            try:
                return float(weight)
            except (TypeError, ValueError):
                return default
    return default


def _get_edge_directed(graph: AnnNet, edge_id: str) -> bool:
    if edge_id in graph.edge_directed:
        return bool(graph.edge_directed[edge_id])
    value = graph.attrs.get_attr_edge(edge_id, 'directed', None)
    if value is not None:
        return bool(value)
    return True if graph.directed is None else bool(graph.directed)


def _build_edge_attr_map(graph: AnnNet):
    ea = getattr(graph, 'edge_attributes', None)
    if ea is None:
        return None
    rows = dataframe_to_rows(ea)
    if not rows:
        return None
    out = {}
    for row in rows:
        eid = row.get('edge_id', None)
        if eid is None:
            continue
        attrs = {k: v for k, v in row.items() if k != 'edge_id' and v is not None}
        if attrs:
            out[eid] = attrs
    return out if out else None


def to_sif(
    graph: AnnNet,
    path: str | None = None,
    *,
    relation_attr: str = 'relation',
    default_relation: str = 'interacts_with',
    write_nodes: bool = True,
    nodes_path: str | None = None,
    lossless: bool = False,
    manifest_path: str | None = None,
) -> None | tuple[None, dict]:
    """Export graph to SIF format.

    Standard mode (lossless=False):
        - Writes only binary edges to SIF file
        - Hyperedges, weights, attrs, IDs are lost
        - Returns None

    Lossless mode (lossless=True):
        - Writes binary edges to SIF file
        - Returns (None, manifest) where manifest contains all lost info
        - Manifest can be saved separately and used with from_sif()

    Args:
        path: Output SIF file path (if None, only manifest is returned in lossless mode)
        relation_attr: Edge attribute key for relation type
        default_relation: Default relation if attr missing
        write_nodes: Whether to write .nodes sidecar with vertex attrs
        nodes_path: Custom path for nodes sidecar (default: path + ".nodes")
        lossless: If True, return manifest with all non-SIF data
        manifest_path: If provided, write manifest to this path (only when lossless=True)

    Returns
    -------
        None (standard mode) or (None, manifest_dict) (lossless mode)

    """

    manifest = (
        {
            'version': '1.0',
            'binary_edges': {},
            'hyperedges': {},
            'vertex_attrs': {},
            'edge_metadata': {},
            'slices': {},
            'multilayer': serialize_multilayer_manifest(
                graph,
                table_to_rows=dataframe_to_rows,
                serialize_edge_layers=serialize_edge_layers,
            ),
        }
        if lossless
        else None
    )

    if path:
        edge_attr_map = _build_edge_attr_map(graph)

        with open(path, 'w', encoding='utf-8') as f:
            for eid, (src, tgt, _etype) in graph.edge_definitions.items():
                if src is None or tgt is None:
                    continue
                src_str = str(src).strip()
                tgt_str = str(tgt).strip()
                if (
                    not src_str
                    or not tgt_str
                    or src_str.lower() == 'none'
                    or tgt_str.lower() == 'none'
                ):
                    continue

                if edge_attr_map is not None:
                    all_attrs = edge_attr_map.get(eid, {})
                else:
                    all_attrs = _get_all_edge_attrs(graph, eid)
                rel = all_attrs.get(relation_attr, default_relation)

                f.write(f'{src_str}\t{rel}\t{tgt_str}\n')

                if lossless:
                    directed = _get_edge_directed(graph, eid)
                    weight = _get_edge_weight(graph, eid, 1.0)

                    manifest['binary_edges'][eid] = {
                        'source': src_str,
                        'target': tgt_str,
                        'source_endpoint': serialize_endpoint(src),
                        'target_endpoint': serialize_endpoint(tgt),
                        'directed': directed,
                    }

                    if weight != 1.0 or all_attrs or eid != f'edge_{len(manifest["binary_edges"])}':
                        manifest['edge_metadata'][eid] = {
                            'weight': weight,
                            'attrs': all_attrs,
                        }

        if write_nodes:
            sidecar = nodes_path if nodes_path is not None else (str(path) + '.nodes')
            with open(sidecar, 'w', encoding='utf-8') as nf:
                nf.write('# nodes sidecar for SIF; format: <vertex_id>\tkey=value ...\n')

                vrows = _safe_vertex_attr_rows(graph)
                vmap: dict[str, dict[str, object]] = {}

                if vrows:
                    for row in vrows:
                        if not isinstance(row, dict):
                            continue
                        vid_raw = row.get('vertex_id', None)
                        if vid_raw is None:
                            continue
                        vid = str(vid_raw).strip()
                        if not vid or vid.lower() == 'none':
                            continue
                        attrs = {k: v for k, v in row.items() if k != 'vertex_id' and v is not None}
                        vmap[vid] = attrs

                if not vmap:
                    getter = getattr(graph, 'get_vertex_attrs', None)
                    if callable(getter):
                        try:
                            for vid in graph.vertices():
                                if vid is None:
                                    continue
                                svid = str(vid).strip()
                                if not svid or svid.lower() == 'none':
                                    continue
                                attrs = getter(vid) or {}
                                vmap[svid] = {k: v for k, v in attrs.items() if v is not None}
                        except AttributeError:
                            pass

                for vid in graph.vertices():
                    if vid is None:
                        continue
                    svid = str(vid).strip()
                    if not svid or svid.lower() == 'none':
                        continue
                    attrs = vmap.get(svid, {})
                    if attrs:
                        kv = '\t'.join(f'{k}={v}' for k, v in attrs.items())
                        nf.write(f'{svid}\t{kv}\n')
                    else:
                        nf.write(f'{svid}\n')

                    if lossless and attrs:
                        manifest['vertex_attrs'][svid] = attrs

    if lossless:
        for eid, info in graph.hyperedge_definitions.items():
            directed = bool(info.get('directed', False))
            head = list(info.get('head', [])) if directed else []
            tail = list(info.get('tail', [])) if directed else []
            members = list(info.get('members', [])) if not directed else []

            weight = _get_edge_weight(graph, eid, 1.0)
            attrs = _get_all_edge_attrs(graph, eid)

            manifest['hyperedges'][eid] = {
                'directed': directed,
                'head': head,
                'tail': tail,
                'members': members,
                'weight': weight,
                'attrs': attrs,
            }

        for lid in graph.slices.list_slices(include_default=True):
            edge_ids = list(graph.slices.get_slice_edges(lid))
            if not edge_ids:
                continue

            slice_info = {'edges': edge_ids, 'weights': {}}

            for eid in edge_ids:
                try:
                    w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight', default=None)
                except TypeError:
                    w = graph.attrs.get_edge_slice_attr(lid, eid, 'weight')
                if w is not None:
                    slice_info['weights'][eid] = float(w)

            manifest['slices'][str(lid)] = slice_info

        if manifest_path:
            with open(manifest_path, 'w', encoding='utf-8') as mf:
                json.dump(manifest, mf, indent=2)

        return None, manifest

    return None


def from_sif(
    path: str,
    *,
    manifest: str | dict | None = None,
    directed: bool = True,
    relation_attr: str = 'relation',
    default_relation: str = 'interacts_with',
    read_nodes_sidecar: bool = True,
    nodes_path: str | None = None,
    encoding: str = 'utf-8',
    delimiter: str | None = None,
    comment_prefixes: Iterable[str] = ('#', '!'),
) -> AnnNet:
    """Import graph from SIF (Simple Interaction Format).

    Standard mode (manifest=None):
        - Reads binary edges from SIF file (source, relation, target)
        - Auto-generates edge IDs (edge_0, edge_1, ...)
        - All edges inherit the `directed` parameter
        - Vertex attributes loaded from optional .nodes sidecar
        - Hyperedges, per-edge directedness, and complex metadata are lost

    Lossless mode (manifest provided):
        - Reads binary edges from SIF file
        - Restores original edge IDs, weights, and attributes from manifest
        - Reconstructs hyperedges from manifest
        - Restores per-edge directedness from manifest
        - Restores slice memberships and slice-specific weights from manifest
        - Full round-trip fidelity when paired with to_sif(lossless=True)

    SIF Format:
        - Three columns: source<TAB>relation<TAB>target
        - Lines starting with comment_prefixes are ignored
        - Vertices referenced in edges are created automatically

    Sidecar .nodes file format (optional):
        - One vertex per line: vertex_id<TAB>key=value<TAB>key=value...
        - Boolean values: true/false (case-insensitive)
        - Numeric values: auto-detected floats
        - String values: everything else

    Args:
        path: Input SIF file path
        manifest: Manifest dict or path to manifest JSON (for lossless reconstruction)
        directed: Default directedness for edges (overridden by manifest if provided)
        relation_attr: Edge attribute key for storing relation type
        default_relation: Default relation if not specified in file
        read_nodes_sidecar: Whether to read .nodes sidecar file with vertex attributes
        nodes_path: Custom path for nodes sidecar (default: path + ".nodes")
        encoding: File encoding (default: utf-8)
        delimiter: Custom delimiter (default: auto-detect TAB or whitespace)
        comment_prefixes: Line prefixes to skip (default: # and !)

    Returns
    -------
        AnnNet: Reconstructed graph object

    Notes
    -----
        - SIF format only supports binary edges natively
        - For full graph reconstruction (hyperedges, slices, metadata), use manifest
        - Manifest files are created by to_sif(lossless=True)
        - Edge IDs are auto-generated in standard mode, preserved in lossless mode
        - Vertex attributes require .nodes sidecar file or manifest

    """
    if manifest is not None and not isinstance(manifest, dict):
        with open(str(manifest), encoding='utf-8') as mf:
            manifest = json.load(mf)

    H = AnnNet(directed=None if manifest and 'binary_edges' in manifest else directed)

    # Single-key hashing with separator
    SEP = '\x00'  # NULL byte - impossible in text files
    binary_edge_index = None
    if manifest and 'binary_edges' in manifest:
        em = manifest.get('edge_metadata', {})
        binary_edge_index = {
            info['source']
            + SEP
            + info['target']
            + SEP
            + em.get(eid, {}).get('attrs', {}).get(relation_attr, default_relation): (eid, info)
            for eid, info in manifest['binary_edges'].items()
        }

    def _parse_node_kv(tok: str):
        if '=' not in tok:
            return None, None
        k, _, v = tok.partition('=')
        k, v = k.strip(), v.strip()
        if not k:
            return None, None
        lv = v.lower()
        if lv == 'true':
            return k, True
        if lv == 'false':
            return k, False
        if lv in ('nan', 'inf', '-inf'):
            return k, v
        try:
            return k, float(v)
        except ValueError:
            return k, v

    # ===== NODES SIDECAR WITH PRE-DETECT DELIMITER =====
    vertex_data = {}

    if read_nodes_sidecar:
        sidecar = nodes_path if nodes_path is not None else (str(path) + '.nodes')
        import os

        if os.path.exists(sidecar):
            # Detect delimiter once
            use_tab = None
            vd = vertex_data  # OPT 7: Localize

            with open(sidecar, encoding=encoding) as nf:
                for raw in nf:
                    s = raw.rstrip('\n\r')
                    if not s or any(s.lstrip().startswith(pfx) for pfx in comment_prefixes):
                        continue

                    # Auto-detect on first data line
                    if use_tab is None:
                        use_tab = '\t' in s

                    toks = s.split('\t') if use_tab else [s]
                    vid = toks[0].strip()
                    if not vid or vid.lower() == 'none':
                        continue

                    attrs = {}
                    if len(toks) > 1:
                        for t in toks[1:]:
                            k, v = _parse_node_kv(t)
                            if k is not None:
                                attrs[k] = v
                    vd[vid] = attrs

    # Merge manifest vertex attrs
    if manifest and 'vertex_attrs' in manifest:
        for vid, attrs in manifest['vertex_attrs'].items():
            vertex_data.setdefault(vid, {}).update(attrs)

    # ===== EDGES FILE WITH: INLINE + LOCALS + FAST COLLECTIONS =====
    edges_raw = []

    # Resolve delimiter once
    use_tab = delimiter is None
    actual_delim = delimiter if delimiter is not None else '\t'

    # Localize lookups
    vd = vertex_data
    append_edge = edges_raw.append
    comment_tuple = tuple(comment_prefixes)

    with open(path, encoding=encoding) as f:
        for raw in f:
            # Inline fast path for comments
            if raw.startswith(comment_tuple):
                continue

            line = raw.rstrip('\n\r')
            if not line:
                continue

            # Inline split (no function call)
            if use_tab:
                if '\t' not in line:
                    continue
                toks = line.split('\t')
            else:
                toks = line.split(actual_delim)

            if len(toks) < 3:
                continue

            src, rel, tgt = toks[0].strip(), toks[1].strip(), toks[2].strip()
            if not src or src.lower() == 'none' or not tgt or tgt.lower() == 'none':
                continue

            # Avoid setdefault allocation on hit
            if src not in vd:
                vd[src] = {}
            if tgt not in vd:
                vd[tgt] = {}

            append_edge((src, tgt, rel))

    # ===== BULK ADD VERTICES =====
    if vertex_data:
        vertices_bulk = list(vertex_data.items())
        H.add_vertices_bulk(vertices_bulk)

    # ===== BULK ADD EDGES WITH FAST HASHING + DELAYED EXPANSION =====
    if manifest and 'binary_edges' in manifest:
        # Lossless mode with single-key lookup
        edges_bulk = []
        get_edge = binary_edge_index.get  # Localize
        append = edges_bulk.append  # Localize

        for src, tgt, rel in edges_raw:
            # Single string key (no tuple allocation)
            key = src + SEP + tgt + SEP + rel
            hit = get_edge(key)

            if hit:
                orig_eid, info = hit
                edge_directed_val = info.get('directed', directed)
                meta = manifest.get('edge_metadata', {}).get(orig_eid, {})
                weight = meta.get('weight', 1.0)
                attrs = meta.get('attrs', {})

                append(
                    {
                        'source': deserialize_endpoint(info.get('source_endpoint', src)),
                        'target': deserialize_endpoint(info.get('target_endpoint', tgt)),
                        'weight': weight,
                        'edge_id': orig_eid,
                        'edge_directed': edge_directed_val,
                        'attributes': attrs if attrs else {},
                    }
                )
            else:
                append(
                    {
                        'source': src,
                        'target': tgt,
                        'weight': 1.0,
                        'edge_directed': directed,
                        'attributes': {relation_attr: rel},
                    }
                )
    else:
        # Standard mode - delayed dict expansion via generator
        edges_bulk = (
            {
                'source': src,
                'target': tgt,
                'weight': 1.0,
                'edge_directed': directed,
                'attributes': {relation_attr: rel},
            }
            for src, tgt, rel in edges_raw
        )

    if edges_raw:  # Check original list, not generator
        H.add_edges_bulk(edges_bulk, default_weight=1.0, default_edge_directed=directed)

    # ===== HYPEREDGES =====
    if manifest and 'hyperedges' in manifest:
        hyperedges_bulk = []
        for eid, info in manifest['hyperedges'].items():
            directed_he = info.get('directed', False)
            he_dict = {
                'edge_id': eid,
                'weight': info.get('weight', 1.0),
                'edge_directed': directed_he,
                'attributes': info.get('attrs', {}),
            }

            if directed_he:
                he_dict['head'] = info.get('head', [])
                he_dict['tail'] = info.get('tail', [])
            else:
                he_dict['members'] = info.get('members', [])

            hyperedges_bulk.append(he_dict)

        if hyperedges_bulk:
            H.add_hyperedges_bulk(hyperedges_bulk, default_weight=1.0, default_edge_directed=False)

    # ===== SLICES WITH NO EXCEPTIONS + CACHED SET =====
    if manifest and 'slices' in manifest:
        # Build set once
        existing_slices = set(H.slices.list_slices(include_default=True))

        for lid, slice_info in manifest['slices'].items():
            # Guard instead of exception
            if lid not in existing_slices:
                H.slices.add_slice(lid)
                existing_slices.add(lid)  # Keep cached set in sync

            edge_ids = slice_info.get('edges', [])
            if edge_ids:
                H.slices.add_edges(lid, edge_ids)

            weights = slice_info.get('weights', {})
            if weights:
                H.attrs.set_edge_slice_attrs_bulk(
                    lid, [{'edge_id': eid, 'weight': w} for eid, w in weights.items()]
                )

    # ===== MULTILAYER =====
    if manifest and 'multilayer' in manifest:
        restore_multilayer_manifest(
            H,
            manifest['multilayer'],
            rows_to_table=_rows_to_df,
            deserialize_edge_layers=deserialize_edge_layers,
        )

    return H
