from __future__ import annotations

import json

from benchmarks import cases
from benchmarks.reporting import render as render_report
from benchmarks.scales import Scale


def _tiny_scale() -> Scale:
    return Scale(
        name='unit',
        vertices=6,
        edges=10,
        hyperedges=2,
        slices=2,
        node_attrs=2,
        edge_attrs=2,
        sparse_every=3,
        remove_fraction=0.2,
        accessor_repeats=2,
    )


def test_extra_dimensions_run_selected_primitive_group() -> None:
    rows = cases.extra_dimensions(_tiny_scale(), groups=('primitives',), samples=1)

    assert {row['group'] for row in rows} == {'primitives'}
    assert {row['op'] for row in rows} >= {
        'create_empty',
        'add_vertices_bulk',
        'add_edges_bulk',
        'remove_edges_fraction',
    }
    assert all(row['status'] == 'ok' for row in rows)
    assert all(row['time'] is not None for row in rows)


def test_annotation_update_dimensions_run_on_tiny_scale() -> None:
    rows = cases.extra_dimensions(_tiny_scale(), groups=('annotation_updates',), samples=1)

    assert {row['group'] for row in rows} == {'annotation_updates'}
    assert {row['op'] for row in rows} >= {
        'set_vertex_attrs_bulk_initial',
        'set_vertex_attrs_bulk_update',
        'set_edge_attrs_bulk_initial',
        'set_edge_slice_attrs_bulk',
    }
    assert all(row['status'] == 'ok' for row in rows)


def test_report_renders_extra_dimensions(tmp_path) -> None:
    rows = cases.extra_dimensions(_tiny_scale(), groups=('primitives',), samples=1)
    payload = {
        'environment': {'libraries': {}},
        'config': {'tier': 'quick', 'scales': ['unit'], 'backends': ['auto']},
        'records': rows,
        'io_formats': [],
        'adapters': [],
    }
    path = render_report(payload, tmp_path / 'REPORT.md', plots_dir=tmp_path / 'plots')
    text = path.read_text()

    assert 'AnnNet mutation primitives' in text
    assert 'add_edges_bulk' in text
    json.dumps(payload)


def test_report_emits_networkx_ratio_heatmap(tmp_path) -> None:
    rows = []
    for scale, n_edges, ann_s, nx_s in (
        ('tiny', 400, 0.004, 0.002),
        ('small', 4_000, 0.030, 0.010),
    ):
        for engine, median_s in (('annnet', ann_s), ('networkx', nx_s)):
            rows.append(
                {
                    'engine': engine,
                    'scale': scale,
                    'op': 'build',
                    'group': 'comparable',
                    'backend': None,
                    'n_vertices': n_edges // 4,
                    'n_edges': n_edges,
                    'time': {
                        'min_s': median_s,
                        'median_s': median_s,
                        'mean_s': median_s,
                        'stdev_s': 0.0,
                        'p95_s': median_s,
                        'samples': 1,
                        'inner': 1,
                        'total_calls': 1,
                    },
                }
            )
    payload = {
        'environment': {'libraries': {}},
        'config': {'tier': 'quick', 'scales': ['tiny', 'small'], 'backends': ['auto']},
        'records': rows,
        'io_formats': [],
        'adapters': [],
    }

    path = render_report(payload, tmp_path / 'REPORT.md', plots_dir=tmp_path / 'plots')
    text = path.read_text()

    assert 'AnnNet / NetworkX ratio heatmap' in text
    assert (tmp_path / 'plots' / 'annnet_vs_networkx_ratio_heatmap.png').exists()
    assert not (tmp_path / 'plots' / 'annnet_vs_graph_tool_ratio_heatmap.png').exists()
