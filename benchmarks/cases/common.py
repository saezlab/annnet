from __future__ import annotations

from typing import Any
from dataclasses import replace
import importlib.util
from collections.abc import Callable, Sequence

from .. import harness


def capped_scale(
    scale,
    *,
    max_vertices: int,
    max_edges: int,
    max_accessor_repeats: int,
):
    vertices = min(scale.vertices, max_vertices)
    edges = min(scale.edges, max_edges)
    hyperedges = min(scale.hyperedges, max(1, edges // 4))
    return replace(
        scale,
        vertices=vertices,
        edges=edges,
        hyperedges=hyperedges,
        accessor_repeats=min(scale.accessor_repeats, max_accessor_repeats),
    )


def scale_note(original, effective) -> str:
    if original.vertices == effective.vertices and original.edges == effective.edges:
        return ''
    return f'capped from {original.vertices:,} vertices/{original.edges:,} edges'


def time_record(
    engine: str,
    group: str,
    op: str,
    scale,
    fn: Callable[[], object],
    backend: str | None,
    samples: int,
    note: str,
    *,
    optional_module: str | None = None,
) -> dict:
    if optional_module is not None and importlib.util.find_spec(optional_module) is None:
        rec = record(engine, group, op, scale, backend=backend, note=note)
        rec['status'] = 'skipped'
        rec['reason'] = f'{optional_module} is not installed'
        return rec
    try:
        rec = record(
            engine,
            group,
            op,
            scale,
            backend=backend,
            time=harness.time_oneshot(fn, samples=max(1, samples), warmup=0),
            note=note,
        )
        rec['status'] = 'ok'
        return rec
    except Exception as exc:
        rec = record(engine, group, op, scale, backend=backend, note=note)
        rec['status'] = 'error'
        rec['error'] = f'{type(exc).__name__}: {exc}'
        return rec


def record(
    engine: str,
    group: str,
    op: str,
    scale,
    *,
    backend=None,
    time=None,
    memory=None,
    note: str = '',
) -> dict:
    return {
        'engine': engine,
        'backend': backend,
        'scale': scale.name,
        'n_vertices': scale.vertices,
        'n_edges': scale.edges,
        'group': group,
        'op': op,
        'time': time.as_dict() if time is not None else None,
        'memory': memory.as_dict() if memory is not None else None,
        'note': note,
    }


def annnet():
    from annnet.core.graph import AnnNet

    return AnnNet


def build_annnet_graph(
    scale,
    *,
    backend: str = 'auto',
    vertex_attrs: bool = False,
    edge_attrs: bool = False,
):
    AnnNet = annnet()
    vertices = make_vertices(scale.vertices)
    pairs = make_edge_pairs(scale.vertices, scale.edges)
    vertex_records = make_vertex_records(
        vertices,
        with_attrs=vertex_attrs,
        attr_count=scale.node_attrs,
        sparse_every=scale.sparse_every,
        annotation_density=scale.annotation_density,
    )
    edge_records = make_edge_records(
        pairs,
        with_attrs=edge_attrs,
        attr_count=scale.edge_attrs,
        sparse_every=scale.sparse_every,
        annotation_density=scale.annotation_density,
    )
    graph = AnnNet(directed=True, annotations_backend=backend)
    graph.add_vertices(vertex_records, slice='base')
    edge_ids = graph.add_edges(edge_records, slice='base')
    return graph, vertices, pairs, list(edge_ids)


def make_vertices(n_vertices: int) -> list[str]:
    return [f'v{i}' for i in range(n_vertices)]


def make_edge_pairs(
    n_vertices: int,
    n_edges: int,
    *,
    directed: bool = True,
) -> list[tuple[str, str]]:
    if n_vertices < 2 or n_edges <= 0:
        return []
    max_edges = n_vertices * (n_vertices - 1)
    if not directed:
        max_edges //= 2
    target = min(n_edges, max_edges)
    vertices = make_vertices(n_vertices)
    out: list[tuple[str, str]] = []
    seen: set[tuple[int, int]] = set()
    offset = 1
    while len(out) < target and offset < n_vertices:
        for src_idx in range(n_vertices):
            tgt_idx = (src_idx + offset) % n_vertices
            if src_idx == tgt_idx:
                continue
            key = (src_idx, tgt_idx) if directed else tuple(sorted((src_idx, tgt_idx)))
            if key in seen:
                continue
            seen.add(key)
            out.append((vertices[src_idx], vertices[tgt_idx]))
            if len(out) >= target:
                break
        offset += 1
    return out


def make_numeric_edge_pairs(
    n_vertices: int,
    n_edges: int,
    *,
    directed: bool = True,
) -> list[tuple[int, int]]:
    pairs = make_edge_pairs(n_vertices, n_edges, directed=directed)
    return [(int(src[1:]), int(tgt[1:])) for src, tgt in pairs]


def make_edge_records(
    pairs: Sequence[tuple[str, str]],
    *,
    with_attrs: bool = False,
    attr_count: int = 0,
    sparse_every: int = 0,
    annotation_density: float | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, (source, target) in enumerate(pairs):
        row: dict[str, Any] = {
            'edge_id': f'e{idx}',
            'source': source,
            'target': target,
            'weight': 1.0,
        }
        if with_attrs:
            row.update(
                attr_values(
                    idx,
                    attr_count,
                    sparse_every,
                    annotation_density=annotation_density,
                    prefix='edge_attr',
                )
            )
        records.append(row)
    return records


def make_vertex_records(
    vertices: Sequence[str],
    *,
    with_attrs: bool = False,
    attr_count: int = 0,
    sparse_every: int = 0,
    annotation_density: float | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, vertex_id in enumerate(vertices):
        row: dict[str, Any] = {'vertex_id': vertex_id}
        if with_attrs:
            row.update(
                attr_values(
                    idx,
                    attr_count,
                    sparse_every,
                    annotation_density=annotation_density,
                    prefix='node_attr',
                )
            )
        records.append(row)
    return records


def make_vertex_attr_updates(
    vertices: Sequence[str],
    *,
    attr_count: int,
    sparse_every: int,
    annotation_density: float | None = None,
    prefix: str = 'node_attr',
) -> dict[str, dict[str, Any]]:
    return {
        vertex_id: attr_values(
            idx,
            attr_count,
            sparse_every,
            annotation_density=annotation_density,
            prefix=prefix,
        )
        for idx, vertex_id in enumerate(vertices)
    }


def make_edge_attr_updates(
    edge_ids: Sequence[str],
    *,
    attr_count: int,
    sparse_every: int,
    annotation_density: float | None = None,
    prefix: str = 'edge_attr',
) -> dict[str, dict[str, Any]]:
    return {
        edge_id: attr_values(
            idx,
            attr_count,
            sparse_every,
            annotation_density=annotation_density,
            prefix=prefix,
        )
        for idx, edge_id in enumerate(edge_ids)
    }


def attr_values(
    idx: int,
    attr_count: int,
    sparse_every: int,
    *,
    annotation_density: float | None = None,
    prefix: str,
) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for attr_idx in range(attr_count):
        if not include_attr(idx, attr_idx, sparse_every, annotation_density):
            continue
        key = f'{prefix}_{attr_idx}'
        kind = attr_idx % 4
        if kind == 0:
            attrs[key] = idx + attr_idx
        elif kind == 1:
            attrs[key] = float((idx + attr_idx) % 100) / 10.0
        elif kind == 2:
            attrs[key] = f'value_{attr_idx % 17}'
        else:
            attrs[key] = ((idx + attr_idx) % 2) == 0
    return attrs


def include_attr(
    row_idx: int,
    attr_idx: int,
    sparse_every: int,
    annotation_density: float | None,
) -> bool:
    if annotation_density is not None:
        density = max(0.0, min(1.0, annotation_density))
        threshold = int(density * 10_000)
        score = ((row_idx + 1) * 1_315_423_911 + (attr_idx + 1) * 2_654_435_761) % 10_000
        return score < threshold
    return not (sparse_every and (row_idx + attr_idx) % sparse_every == 0)


def optional_module_for_engine(engine: str) -> str | None:
    return {
        'networkx': 'networkx',
        'igraph': 'igraph',
        'graph-tool': 'graph_tool',
    }.get(engine)


def graphtool_property_type(attr_idx: int) -> str:
    kind = attr_idx % 4
    if kind in {0, 1}:
        return 'double'
    if kind == 2:
        return 'string'
    return 'bool'


def graphtool_property_value(value, attr_idx: int):
    kind = attr_idx % 4
    if kind in {0, 1}:
        return float(value or 0.0)
    if kind == 2:
        return '' if value is None else str(value)
    return bool(value)
