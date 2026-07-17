"""Shared benchmark reporting constants and record helpers."""

from __future__ import annotations

from pathlib import Path

COMPARABLE_OPS = ['build', 'degree', 'neighbors', 'has_edge', 'enumerate_edges']
COMPARABLE_ENGINES = ('annnet', 'networkx', 'igraph', 'graph-tool')
BASELINE_ENGINES = ('networkx', 'igraph', 'graph-tool')
ENGINE_LABELS = {
    'annnet': 'AnnNet',
    'networkx': 'NetworkX',
    'igraph': 'igraph',
    'graph-tool': 'graph-tool',
}

ANNNET_INTERNAL_ORDER = [
    'build_hyperedges',
    'build_multilayer',
    'materialize_incidence_csr',
    'materialize_adjacency',
    'copy',
    'subgraph_half',
    'annotations_vertex_bulk',
    'io_write',
    'io_read',
]

EXTRA_GROUPS = [
    (
        'primitives',
        'AnnNet mutation primitives',
        'Bulk and repeated public mutation calls carried over from the local benchmark suite.',
    ),
    (
        'annotation_updates',
        'Annotation update variants',
        'Initial writes versus updates for vertex, edge, slice, and edge-slice annotations.',
    ),
    (
        'backend_operations',
        'Backend mutation and annotation operations',
        'Mutation and annotation operations across AnnNet, NetworkX, igraph, and graph-tool when installed.',
    ),
    (
        'accessors',
        'Lazy adapter accessors',
        'Cold means the AnnNet accessor cache is cleared and the first read pays the adapter '
        'conversion cost. Warm means one conversion has already populated the cache and the '
        'timed loop reads the cached backend object. Explicit reconvert bypasses the accessor '
        'cache; round-trip also rebuilds AnnNet; after-mutation primes the cache, mutates '
        'AnnNet, and times the refreshed view.',
    ),
]

SCALE_ORDER = {'tiny': 0, 'xsmall': 1, 'small': 2, 'medium': 3, 'large': 4, 'xlarge': 5}
SCALE_COLORS = {
    'tiny': '#4C78A8',
    'xsmall': '#F58518',
    'small': '#54A24B',
    'medium': '#B279A2',
    'large': '#E45756',
    'xlarge': '#72B7B2',
}
ENGINE_COLORS = {
    'annnet': '#4C78A8',
    'networkx': '#F58518',
    'igraph': '#54A24B',
    'graph-tool': '#B279A2',
}
PHASE_COLORS = {'write': '#4C78A8', 'read': '#F58518', 'export': '#4C78A8', 'import': '#F58518'}

OP_LABELS = {
    'add_edges_bulk': 'add edges\nbulk',
    'add_edges_repeated': 'add edges\nrepeated',
    'add_vertices_bulk': 'add vertices\nbulk',
    'add_vertices_repeated': 'add vertices\nrepeated',
    'annotations_vertex_bulk': 'vertex attrs\nbulk',
    'build_hyperedges': 'build\nhyperedges',
    'build_multilayer': 'build\nmultilayer',
    'build_with_history': 'build with\nhistory',
    'copy_with_history': 'copy with\nhistory',
    'create_empty': 'empty\ngraph',
    'enumerate_edges': 'enumerate\nedges',
    'gt_accessor_after_mutation_components': 'cache after\nmutation',
    'gt_accessor_cold_components': 'cold\naccessor',
    'gt_accessor_warm_repeated_components': 'warm\ncached',
    'gt_explicit_reconvert_repeated_components': 'explicit\nconvert',
    'gt_explicit_roundtrip_repeated_components': 'explicit\nroundtrip',
    'ig_accessor_after_mutation_vcount': 'cache after\nmutation',
    'ig_accessor_cold_vcount': 'cold\naccessor',
    'ig_accessor_warm_repeated_vcount': 'warm\ncached',
    'ig_explicit_reconvert_repeated_vcount': 'explicit\nconvert',
    'ig_explicit_roundtrip_repeated_vcount': 'explicit\nroundtrip',
    'io_read': 'io\nread',
    'io_write': 'io\nwrite',
    'materialize_adjacency': 'materialize\nadjacency',
    'materialize_incidence_csr': 'incidence\ncsr',
    'nx_accessor_after_mutation_number_of_nodes': 'cache after\nmutation',
    'nx_accessor_cold_number_of_nodes': 'cold\naccessor',
    'nx_accessor_warm_repeated_number_of_nodes': 'warm\ncached',
    'nx_explicit_reconvert_repeated_number_of_nodes': 'explicit\nconvert',
    'nx_explicit_roundtrip_repeated_number_of_nodes': 'explicit\nroundtrip',
    'remove_edges_fraction': 'remove\nedges',
    'remove_vertices_fraction': 'remove\nvertices',
    'set_edge_attrs_bulk': 'edge attrs\nbulk',
    'set_edge_attrs_bulk_initial': 'edge attrs\ninitial',
    'set_edge_attrs_bulk_update': 'edge attrs\nupdate',
    'set_edge_slice_attrs_bulk': 'edge-slice\nattrs',
    'set_slice_attrs_repeated': 'slice attrs\nrepeated',
    'set_vertex_attrs_bulk': 'vertex attrs\nbulk',
    'set_vertex_attrs_bulk_initial': 'vertex attrs\ninitial',
    'set_vertex_attrs_bulk_update': 'vertex attrs\nupdate',
    'subgraph_half': 'subgraph\nhalf',
    'traversal_sweep': 'traversal\nsweep',
}


def median_ms(rec: dict | None) -> float | None:
    if rec is None:
        return None
    t = rec.get('time')
    return t['median_s'] * 1e3 if t else None


def median_s(rec: dict | None) -> float | None:
    if rec is None:
        return None
    t = rec.get('time')
    return t['median_s'] if t else None


def fmt_ms(v: float | None) -> str:
    if v is None:
        return '-'
    if v < 1:
        return f'{v * 1000:.1f} us'
    if v < 1000:
        return f'{v:.3g} ms'
    return f'{v / 1000:.3g} s'


def fmt_bytes(n: float | None) -> str:
    if n is None:
        return '-'
    for unit in ('B', 'KB', 'MB', 'GB'):
        if abs(n) < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} TB'


def index_records(records: list[dict]) -> dict[tuple[str, str, str], dict]:
    idx = {}
    for r in records:
        idx.setdefault((r['engine'], r['scale'], r['op']), r)
    return idx


def scales_in(records: list[dict]) -> list[str]:
    return sorted({r['scale'] for r in records}, key=lambda s: SCALE_ORDER.get(s, 99))


def pick(idx: dict, engine: str, scale: str, op: str) -> dict | None:
    return idx.get((engine, scale, op))


def slug(value: object) -> str:
    return ''.join(ch if ch.isalnum() else '_' for ch in str(value)).strip('_').lower()


def rel(path: Path, base: Path) -> Path:
    return path.relative_to(base)


def op_label(op: str) -> str:
    return OP_LABELS.get(op, op.replace('_', '\n'))


def one_line_op(op: str) -> str:
    return op_label(op).replace('\n', ' ')


def ordered(values, order_map=None):
    order_map = order_map or {}
    return sorted(values, key=lambda value: (order_map.get(value, 999), str(value)))


def engines_in(
    records: list[dict],
    *,
    ops: list[str] | tuple[str, ...] | set[str] | None = None,
    require_metric: bool = False,
    include: tuple[str, ...] = COMPARABLE_ENGINES,
) -> list[str]:
    ops = set(ops) if ops is not None else None
    present = set()
    for r in records:
        if r.get('engine') not in include:
            continue
        if ops is not None and r.get('op') not in ops:
            continue
        if require_metric and not (r.get('time') or r.get('memory')):
            continue
        present.add(r['engine'])
    return [engine for engine in include if engine in present]


def engine_label(engine: str) -> str:
    return ENGINE_LABELS.get(engine, engine)


def ratio_label(baseline: str) -> str:
    return f'AnnNet / {engine_label(baseline)}'


def ms(tstat: dict | None) -> float | None:
    """Median milliseconds from a bare TimeStat dict."""
    return tstat['median_s'] * 1e3 if tstat else None
