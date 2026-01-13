from __future__ import annotations

import json
from collections.abc import Iterable

try:
    from ..core.graph import AnnNet
except Exception:
    from annnet.core.graph import AnnNet

try:
    import polars as pl  # optional
except Exception:  # ModuleNotFoundError, etc.
    pl = None

from ..adapters._utils import (
    _deserialize_edge_layers,
    _deserialize_layer_tuple_attrs,
    _deserialize_node_layer_attrs,
    _deserialize_VM,
    _df_to_rows,
    _rows_to_df,
    _serialize_edge_layers,
    _serialize_layer_tuple_attrs,
    _serialize_node_layer_attrs,
    _serialize_VM,
)


def _split_sif_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is not None:
        return [t for t in line.rstrip("\n\r").split(delimiter) if t != ""]
    if "\t" in line:
        return [t for t in line.rstrip("\n\r").split("\t") if t != ""]
    return line.strip().split()


def _safe_vertex_attr_table(graph: AnnNet):
    va = getattr(graph, "vertex_attributes", None)
    if va is None:
        return None
    return va if hasattr(va, "columns") and hasattr(va, "to_dicts") else None


def _get_all_edge_attrs(graph: AnnNet, edge_id: str):
    ea = getattr(graph, "edge_attributes", None)
    if (
        ea is not None
        and hasattr(ea, "columns")
        and hasattr(ea, "filter")
        and hasattr(ea, "to_dicts")
    ):
        try:
            if "edge_id" in ea.columns:
                rows = ea.filter(ea["edge_id"] == edge_id).to_dicts()
                if rows:
                    attrs = dict(rows[0])
                    attrs.pop("edge_id", None)
                    return {k: v for k, v in attrs.items() if v is not None}
        except Exception:
            pass
    return {}


def _get_edge_weight(graph: AnnNet, edge_id: str, default=1.0):
    ew = getattr(graph, "edge_weights", None)
    if ew is not None and hasattr(ew, "get"):
        try:
            w = ew.get(edge_id, default)
            return float(w) if w is not None else default
        except Exception:
            pass
    return default


def to_sif(
    graph: AnnNet,
    path: str | None = None,
    *,
    relation_attr: str = "relation",
    default_relation: str = "interacts_with",
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

    Returns:
        None (standard mode) or (None, manifest_dict) (lossless mode)

    """

    manifest = (
        {
            "version": "1.0",
            "binary_edges": {},
            "hyperedges": {},
            "vertex_attrs": {},
            "edge_metadata": {},
            "edge_metadata": {},
            "slices": {},
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
                "layer_tuple_attrs": _serialize_layer_tuple_attrs(
                    getattr(graph, "_layer_attrs", {})
                ),
                "layer_attributes": _df_to_rows(getattr(graph, "layer_attributes", pl.DataFrame())),
            },
        }
        if lossless
        else None
    )

    if path:
        with open(path, "w", encoding="utf-8") as f:
            edge_defs = getattr(graph, "edge_definitions", {}) or {}

            for eid, (src, tgt, _etype) in edge_defs.items():
                if src is None or tgt is None:
                    continue
                src_str = str(src).strip()
                tgt_str = str(tgt).strip()
                if (
                    not src_str
                    or not tgt_str
                    or src_str.lower() == "none"
                    or tgt_str.lower() == "none"
                ):
                    continue

                all_attrs = _get_all_edge_attrs(graph, eid)
                rel = all_attrs.get(relation_attr, default_relation)

                f.write(f"{src_str}\t{rel}\t{tgt_str}\n")

                if lossless:
                    directed = getattr(graph, "edge_directed", {}).get(eid, True)
                    weight = _get_edge_weight(graph, eid, 1.0)

                    manifest["binary_edges"][eid] = {
                        "source": src_str,
                        "target": tgt_str,
                        "directed": directed,
                    }

                    if weight != 1.0 or all_attrs or eid != f"edge_{len(manifest['binary_edges'])}":
                        manifest["edge_metadata"][eid] = {
                            "weight": weight,
                            "attrs": all_attrs,
                        }

        if write_nodes:
            sidecar = nodes_path if nodes_path is not None else (str(path) + ".nodes")
            with open(sidecar, "w", encoding="utf-8") as nf:
                nf.write("# nodes sidecar for SIF; format: <vertex_id>\tkey=value ...\n")

                vtable = _safe_vertex_attr_table(graph)
                vmap: dict[str, dict[str, object]] = {}

                if vtable is not None:
                    try:
                        for row in vtable.to_dicts():
                            vid_raw = row.get("vertex_id", None)
                            if vid_raw is None:
                                continue
                            vid = str(vid_raw).strip()
                            if not vid or vid.lower() == "none":
                                continue
                            attrs = {
                                k: v for k, v in row.items() if k != "vertex_id" and v is not None
                            }
                            vmap[vid] = attrs
                    except Exception:
                        vmap = {}

                if not vmap:
                    getter = getattr(graph, "get_vertex_attrs", None)
                    if callable(getter):
                        try:
                            for vid in graph.vertices():
                                if vid is None:
                                    continue
                                svid = str(vid).strip()
                                if not svid or svid.lower() == "none":
                                    continue
                                attrs = getter(vid) or {}
                                vmap[svid] = {k: v for k, v in attrs.items() if v is not None}
                        except Exception:
                            pass

                for vid in graph.vertices():
                    if vid is None:
                        continue
                    svid = str(vid).strip()
                    if not svid or svid.lower() == "none":
                        continue
                    attrs = vmap.get(svid, {})
                    if attrs:
                        kv = "\t".join(f"{k}={v}" for k, v in attrs.items())
                        nf.write(f"{svid}\t{kv}\n")
                    else:
                        nf.write(f"{svid}\n")

                    if lossless and attrs:
                        manifest["vertex_attrs"][svid] = attrs

    if lossless:
        edge_kind = getattr(graph, "edge_kind", {}) or {}
        hyp_defs = getattr(graph, "hyperedge_definitions", {}) or {}

        for eid, kind in edge_kind.items():
            if kind != "hyper":
                continue
            meta = hyp_defs.get(eid)
            if not meta:
                continue

            directed = bool(meta.get("directed", False))
            head = list(meta.get("head", [])) if directed else []
            tail = list(meta.get("tail", [])) if directed else []
            members = list(meta.get("members", [])) if not directed else []

            weight = _get_edge_weight(graph, eid, 1.0)
            attrs = _get_all_edge_attrs(graph, eid)

            manifest["hyperedges"][eid] = {
                "directed": directed,
                "head": head,
                "tail": tail,
                "members": members,
                "weight": weight,
                "attrs": attrs,
            }

        try:
            slice_ids = list(graph.list_slices(include_default=True))
            for lid in slice_ids:
                try:
                    edge_ids = list(graph.get_slice_edges(lid))
                    if not edge_ids:
                        continue

                    slice_info = {"edges": edge_ids, "weights": {}}

                    for eid in edge_ids:
                        try:
                            w = graph.get_edge_slice_attr(lid, eid, "weight", default=None)
                            if w is not None:
                                slice_info["weights"][eid] = float(w)
                        except Exception:
                            pass

                    manifest["slices"][str(lid)] = slice_info
                except Exception:
                    pass
        except Exception:
            pass

        if manifest_path:
            with open(manifest_path, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

        return None, manifest

    return None


def from_sif(
    path: str,
    *,
    manifest: str | dict | None = None,
    directed: bool = True,
    relation_attr: str = "relation",
    default_relation: str = "interacts_with",
    read_nodes_sidecar: bool = True,
    nodes_path: str | None = None,
    encoding: str = "utf-8",
    delimiter: str | None = None,
    comment_prefixes: Iterable[str] = ("#", "!"),
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

    Returns:
        AnnNet: Reconstructed graph object

    Notes:
        - SIF format only supports binary edges natively
        - For full graph reconstruction (hyperedges, slices, metadata), use manifest
        - Manifest files are created by to_sif(lossless=True)
        - Edge IDs are auto-generated in standard mode, preserved in lossless mode
        - Vertex attributes require .nodes sidecar file or manifest

    """
    if manifest is not None and not isinstance(manifest, dict):
        with open(str(manifest), encoding="utf-8") as mf:
            manifest = json.load(mf)

    H = AnnNet(directed=None if manifest and "binary_edges" in manifest else directed)

    # Single-key hashing with separator
    SEP = "\x00"  # NULL byte - impossible in text files
    binary_edge_index = None
    if manifest and "binary_edges" in manifest:
        em = manifest.get("edge_metadata", {})
        binary_edge_index = {
            info["source"] + SEP + info["target"] + SEP + 
            em.get(eid, {}).get("attrs", {}).get(relation_attr, default_relation): (eid, info)
            for eid, info in manifest["binary_edges"].items()
        }

    def _parse_node_kv(tok: str):
        if "=" not in tok:
            return None, None
        k, _, v = tok.partition("=")
        k, v = k.strip(), v.strip()
        if not k:
            return None, None
        lv = v.lower()
        if lv == "true":
            return k, True
        if lv == "false":
            return k, False
        if lv in ("nan", "inf", "-inf"):
            return k, v
        try:
            return k, float(v)
        except:
            return k, v

    # ===== NODES SIDECAR WITH PRE-DETECT DELIMITER =====
    vertex_data = {}
    
    if read_nodes_sidecar:
        sidecar = nodes_path if nodes_path is not None else (str(path) + ".nodes")
        import os
        if os.path.exists(sidecar):
            # Detect delimiter once
            use_tab = None
            vd = vertex_data  # OPT 7: Localize
            
            with open(sidecar, encoding=encoding) as nf:
                for raw in nf:
                    s = raw.rstrip("\n\r")
                    if not s or any(s.lstrip().startswith(pfx) for pfx in comment_prefixes):
                        continue
                    
                    # Auto-detect on first data line
                    if use_tab is None:
                        use_tab = "\t" in s
                    
                    toks = s.split("\t") if use_tab else [s]
                    vid = toks[0].strip()
                    if not vid or vid.lower() == "none":
                        continue
                    
                    attrs = {}
                    if len(toks) > 1:
                        for t in toks[1:]:
                            k, v = _parse_node_kv(t)
                            if k is not None:
                                attrs[k] = v
                    vd[vid] = attrs

    # Merge manifest vertex attrs
    if manifest and "vertex_attrs" in manifest:
        for vid, attrs in manifest["vertex_attrs"].items():
            vertex_data.setdefault(vid, {}).update(attrs)

    # ===== EDGES FILE WITH: INLINE + LOCALS + FAST COLLECTIONS =====
    edges_raw = []
    
    # Resolve delimiter once
    use_tab = delimiter is None
    actual_delim = delimiter if delimiter is not None else "\t"
    
    # Localize lookups
    vd = vertex_data
    append_edge = edges_raw.append
    comment_tuple = tuple(comment_prefixes)
    
    with open(path, encoding=encoding) as f:
        for raw in f:
            # Inline fast path for comments
            if raw.startswith(comment_tuple):
                continue
            
            line = raw.rstrip("\n\r")
            if not line:
                continue
            
            # Inline split (no function call)
            if use_tab:
                if "\t" not in line:
                    continue
                toks = line.split("\t")
            else:
                toks = line.split(actual_delim)
            
            if len(toks) < 3:
                continue
            
            src, rel, tgt = toks[0].strip(), toks[1].strip(), toks[2].strip()
            if not src or src.lower() == "none" or not tgt or tgt.lower() == "none":
                continue
            
            # Avoid setdefault allocation on hit
            if src not in vd:
                vd[src] = {}
            if tgt not in vd:
                vd[tgt] = {}
            
            append_edge((src, tgt, rel))

    # ===== BULK ADD VERTICES =====
    if vertex_data:
        vertices_bulk = [(vid, attrs) for vid, attrs in vertex_data.items()]
        H.add_vertices_bulk(vertices_bulk)

    # ===== BULK ADD EDGES WITH FAST HASHING + DELAYED EXPANSION =====
    if manifest and "binary_edges" in manifest:
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
                edge_directed_val = info.get("directed", directed)
                meta = manifest.get("edge_metadata", {}).get(orig_eid, {})
                weight = meta.get("weight", 1.0)
                attrs = meta.get("attrs", {})
                
                append({
                    'source': src,
                    'target': tgt,
                    'weight': weight,
                    'edge_id': orig_eid,
                    'edge_directed': edge_directed_val,
                    'attributes': attrs if attrs else {}
                })
            else:
                append({
                    'source': src,
                    'target': tgt,
                    'weight': 1.0,
                    'edge_directed': directed,
                    'attributes': {relation_attr: rel}
                })
    else:
        # Standard mode - delayed dict expansion via generator
        edges_bulk = (
            {
                'source': src,
                'target': tgt,
                'weight': 1.0,
                'edge_directed': directed,
                'attributes': {relation_attr: rel}
            }
            for src, tgt, rel in edges_raw
        )
    
    if edges_raw:  # Check original list, not generator
        H.add_edges_bulk(edges_bulk, default_weight=1.0, default_edge_directed=directed)

    # ===== HYPEREDGES =====
    if manifest and "hyperedges" in manifest:
        hyperedges_bulk = []
        for eid, info in manifest["hyperedges"].items():
            directed_he = info.get("directed", False)
            he_dict = {
                'edge_id': eid,
                'weight': info.get("weight", 1.0),
                'edge_directed': directed_he,
                'attributes': info.get("attrs", {})
            }
            
            if directed_he:
                he_dict['head'] = info.get("head", [])
                he_dict['tail'] = info.get("tail", [])
            else:
                he_dict['members'] = info.get("members", [])
            
            hyperedges_bulk.append(he_dict)
        
        if hyperedges_bulk:
            H.add_hyperedges_bulk(hyperedges_bulk, default_weight=1.0, default_edge_directed=False)

    # ===== SLICES WITH NO EXCEPTIONS + CACHED SET =====
    if manifest and "slices" in manifest:
        # Build set once
        existing_slices = set(H.list_slices(include_default=True))
        
        for lid, slice_info in manifest["slices"].items():
            # Guard instead of exception
            if lid not in existing_slices:
                H.add_slice(lid)
                existing_slices.add(lid)  # Keep cached set in sync
            
            edge_ids = slice_info.get("edges", [])
            if edge_ids:
                H.add_edges_to_slice_bulk(lid, edge_ids)
            
            weights = slice_info.get("weights", {})
            if weights:
                H.set_edge_slice_attrs_bulk(lid, [
                    {'edge_id': eid, 'weight': w} for eid, w in weights.items()
                ])

    # ===== MULTILAYER =====
    if manifest and "multilayer" in manifest:
        try:
            mm = manifest["multilayer"]
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
                H.layer_attributes = _rows_to_df(layer_attr_rows)
        except:
            pass

    return H