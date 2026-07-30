from __future__ import annotations

import json

import pytest

from benchmarks import cases, engines, workloads
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


# ---------------------------------------------------------------------------
# The operation set the cycle has to measure
# ---------------------------------------------------------------------------


def _ok(rows):
    """Return the records that actually ran."""
    return [row for row in rows if row.get('status') != 'error']


def _ops(rows):
    return {row['op'] for row in _ok(rows)}


def test_the_suite_declares_the_required_operation_set() -> None:
    """The suite states which operations it must cover, so a gap is visible."""
    assert set(workloads.REQUIRED_OPERATIONS) == {
        'single_element_add',
        'single_element_remove',
        'bulk_build',
        'traversal',
        'subgraph',
        'matrix_materialization',
        'read_after_mutate',
    }


def test_every_required_operation_maps_to_an_emitted_op_name() -> None:
    """Each required operation names the record op that measures it."""
    for name, op in workloads.REQUIRED_OPERATIONS.items():
        assert isinstance(op, str) and op, name


def test_mutation_workload_emits_the_single_element_operations() -> None:
    rows = workloads.mutations(engines.annnet_engine(), _tiny_scale(), samples=1)
    assert _ops(rows) >= {'single_element_add', 'single_element_remove', 'read_after_mutate'}
    assert all(row['engine'] == 'annnet' for row in rows)
    assert all(row['time'] is not None for row in _ok(rows))


@pytest.mark.parametrize('engine_name', ['networkx', 'igraph'])
def test_mutation_workload_runs_on_the_reference_engines(engine_name) -> None:
    engine = engines.engine_by_name(engine_name)
    if not engine.available():
        pytest.skip(f'{engine_name} is not installed')
    rows = workloads.mutations(engine, _tiny_scale(), samples=1)
    assert _ops(rows) >= {'single_element_add', 'single_element_remove'}


def test_matrix_growth_workload_sweeps_the_interleaved_case() -> None:
    """N appends, each followed by one matrix read, over a sweep of N."""
    rows = workloads.matrix_growth(_tiny_scale(), samples=1)
    assert _ops(rows) == {'append_then_read', 'append_only'}
    sizes = sorted({row['n_edges'] for row in _ok(rows)})
    assert len(sizes) >= 3, 'a growth curve needs at least three points'


def test_matrix_cache_probe_compares_extending_against_remapping() -> None:
    rows = workloads.matrix_cache_probe(_tiny_scale(), samples=1)
    assert _ops(rows) >= {'cache_extend', 'cache_remap'}


def test_attribute_workload_compares_a_column_against_a_dataframe_column() -> None:
    rows = workloads.attribute_ops(_tiny_scale(), samples=1)
    assert _ops(rows) >= {'attr_column_op', 'dataframe_column_op'}


def test_attribute_option_workload_covers_both_storage_options() -> None:
    rows = workloads.attribute_storage_options(_tiny_scale(), samples=1)
    assert {row['note'].split(';')[0] for row in _ok(rows)} >= {'columnar', 'journal'}
    assert _ops(rows) >= {'attr_write_single', 'attr_read_frame'}


def test_report_renders_the_new_sections(tmp_path) -> None:
    """The report must show the ratios and the attribute options side by side."""
    scale = _tiny_scale()
    rows = workloads.mutations(engines.annnet_engine(), scale, samples=1)
    nx_engine = engines.engine_by_name('networkx')
    if nx_engine.available():
        rows += workloads.mutations(nx_engine, scale, samples=1)
    rows += workloads.matrix_growth(scale, samples=1)
    rows += workloads.matrix_cache_probe(scale, samples=1)
    rows += workloads.attribute_ops(scale, samples=1)
    rows += workloads.attribute_storage_options(scale, samples=1)

    payload = {
        'environment': {'libraries': {}},
        'config': {'tier': 'quick', 'scales': ['unit'], 'backends': ['auto']},
        'records': rows,
        'io_formats': [],
        'adapters': [],
    }
    text = render_report(payload, tmp_path / 'REPORT.md', plots_dir=tmp_path / 'plots').read_text()

    assert 'Single-element writes' in text
    assert 'Reading the matrix after every write' in text
    assert 'The append-only matrix cache' in text
    assert 'Attribute storage' in text
    assert 'single_element_add' in text
    json.dumps(payload)
