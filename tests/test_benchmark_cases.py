from __future__ import annotations

import json

import pytest

from benchmarks import cases, engines, workloads
from benchmarks.reporting import render as render_report
from benchmarks.scales import Scale


def _tiny_scale() -> Scale:
    return Scale(
        name='unit',
        nodes=6,
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
        'add_nodes_bulk',
        'add_edges_bulk',
        'remove_edges_fraction',
    }
    assert all(row['status'] == 'ok' for row in rows)
    assert all(row['time'] is not None for row in rows)


def test_annotation_update_dimensions_run_on_tiny_scale() -> None:
    rows = cases.extra_dimensions(_tiny_scale(), groups=('annotation_updates',), samples=1)

    assert {row['group'] for row in rows} == {'annotation_updates'}
    assert {row['op'] for row in rows} >= {
        'set_node_attrs_bulk_initial',
        'set_node_attrs_bulk_update',
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
                    'n_nodes': n_edges // 4,
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
    """One record per baseline per operation, and every one names its baseline."""
    rows = workloads.attribute_ops(_tiny_scale(), samples=1)
    assert _ops(rows) >= {'column_read', 'column_op', 'column_read_and_op'}
    assert {row['baseline'] for row in rows} >= {'annnet', 'numpy'}


class TestTheAttributeBenchmarkNamesWhatItCompares:
    """`FR-001` to `FR-005`: a record says what it measured and against what.

    The record this replaces was called ``dataframe_column_op`` and summed a
    bare numpy array, with no dataframe of any kind involved. The number it
    produced reached the preprint report.
    """

    @staticmethod
    def _rows():
        return workloads.attribute_ops(_tiny_scale(), samples=1)

    def test_every_record_names_its_baseline(self) -> None:
        rows = self._rows()
        assert rows
        for row in rows:
            assert row.get('baseline'), row
            assert row['baseline'] in workloads.ATTRIBUTE_BASELINES

    def test_nothing_is_called_a_dataframe_unless_a_dataframe_produced_it(self) -> None:
        """`FR-001`. The name of a record identifies the library behind it."""
        for row in self._rows():
            named = f'{row["op"]} {row.get("note", "")} {row["baseline"]}'.lower()
            if 'dataframe' in named:
                assert row['baseline'] in ('polars', 'pandas', 'pyarrow'), row

    def test_the_read_and_the_operation_are_separate_records(self) -> None:
        """`FR-004`. A ratio says which of the two it comes from."""
        rows = self._rows()
        for baseline in {row['baseline'] for row in _ok(rows)}:
            ops = {row['op'] for row in _ok(rows) if row['baseline'] == baseline}
            assert {'column_read', 'column_op', 'column_read_and_op'} <= ops, baseline

    def test_both_sides_of_a_pair_do_the_same_work(self) -> None:
        """`FR-002`. One op name means one starting state and one operation.

        A note is ``<the work>; <the baseline it ran on>``. The work half must
        be word for word the same across the baselines of one op, which is what
        the pair this replaces got wrong: one side included the column read and
        the other summed an array it already held.
        """
        rows = _ok(self._rows())
        for op in ('column_read', 'column_op', 'column_read_and_op'):
            work = {row['note'].split(';')[0] for row in rows if row['op'] == op}
            assert len(work) == 1, (op, work)

    def test_an_absent_backend_is_reported_as_skipped_and_named(self) -> None:
        """`FR-005`. A smaller row count would otherwise hide a missing library."""
        rows = workloads.attribute_ops(_tiny_scale(), samples=1)
        emitted = {(row['baseline'], row['op']) for row in rows}
        expected = {
            (baseline, op)
            for baseline in workloads.ATTRIBUTE_BASELINES
            for op in ('column_read', 'column_op', 'column_read_and_op')
        }
        assert emitted == expected
        for row in rows:
            assert row['status'] in ('ok', 'skipped')
            if row['status'] == 'skipped':
                assert row['baseline'] in row['note']
                assert row['time'] is None

    def test_the_annnet_record_measures_the_public_column_read(self) -> None:
        rows = _ok(self._rows())
        annnet_rows = [row for row in rows if row['baseline'] == 'annnet']
        assert annnet_rows
        assert all(row['engine'] == 'annnet' for row in annnet_rows)


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


def test_the_annnet_feature_workload_records_every_group_it_names() -> None:
    """The suite runs on this package, and a call it gets wrong skips the lot.

    The runner catches an error from one workload and skips the engine it came
    from, so a single wrong keyword empties the AnnNet column of the whole
    report and nothing says so. This walks the feature workload for real.
    """
    rows = workloads.annnet_features(_tiny_scale(), samples=1)
    assert rows, 'the feature workload recorded nothing'
    assert all(row['engine'] == 'annnet' for row in rows)
    assert not [row for row in rows if row.get('status') == 'error']
    groups = {row['group'] for row in _ok(rows)}
    assert {'layers', 'slices'} <= groups, groups


def test_the_annnet_only_workload_reports_no_error() -> None:
    rows = workloads.annnet_only(_tiny_scale(), samples=1)
    assert rows, 'the AnnNet-only workload recorded nothing'
    assert not [row for row in rows if row.get('status') == 'error']
