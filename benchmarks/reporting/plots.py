"""Plot and artifact generation for benchmark reports."""

from __future__ import annotations

import csv
import math
from pathlib import Path

from .common import (
    SCALE_ORDER,
    EXTRA_GROUPS,
    PHASE_COLORS,
    SCALE_COLORS,
    ENGINE_COLORS,
    COMPARABLE_OPS,
    BASELINE_ENGINES,
    ms,
    rel,
    slug,
    fmt_ms,
    ordered,
    median_s,
    op_label,
    scales_in,
    engines_in,
    one_line_op,
    ratio_label,
    engine_label,
    index_records,
)

PNG_DPI = 180


def _matplotlib():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'axes.edgecolor': '#2F3437',
            'axes.labelcolor': '#1F2326',
            'axes.titleweight': 'semibold',
            'axes.titlesize': 12,
            'font.size': 10,
            'legend.fontsize': 9,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white',
            'xtick.color': '#1F2326',
            'ytick.color': '#1F2326',
        }
    )
    return plt


def _style_axis(ax, *, grid_axis='both'):
    ax.grid(True, axis=grid_axis, which='major', ls='-', lw=0.6, alpha=0.18)
    ax.grid(True, axis=grid_axis, which='minor', ls=':', lw=0.5, alpha=0.22)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#D0D5D8')


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=PNG_DPI)
    return path


def render_comparable_plots(records: list[dict], plots_dir: Path) -> list[str]:
    scales = scales_in(records)
    if len(scales) < 2:
        return []
    try:
        _matplotlib()
    except Exception:
        return []

    plots_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        '## Plots',
        '',
        'The plots use log-scaled time axes. Ratio heatmaps are centered at 1x: '
        'blue means AnnNet is faster than the baseline, red means slower.',
        '',
    ]

    artifacts = [
        ('Comparable operation scaling', _plot_scaling_dashboard(records, plots_dir / 'scaling_wall_time_overview.png')),
        ('Retained memory scaling', _plot_memory_scaling(records, plots_dir / 'memory_retained_scaling.png')),
        ('Memory density', _plot_bytes_per_edge(records, plots_dir / 'memory_bytes_per_edge.png')),
    ]
    for baseline in BASELINE_ENGINES:
        artifacts.insert(
            1,
            (
                f'{ratio_label(baseline)} ratio heatmap',
                _plot_ratio_heatmap(
                    records,
                    plots_dir / f'annnet_vs_{slug(baseline)}_ratio_heatmap.png',
                    baseline,
                ),
            ),
        )
    artifacts += _plot_individual_comparable(records, plots_dir)

    for label, png in artifacts:
        if png:
            lines += [f'### {label}', '', f'![{label}]({rel(png, plots_dir.parent)})', '']
    return lines


def _plot_individual_comparable(records: list[dict], plots_dir: Path) -> list[tuple[str, Path | None]]:
    plt = _matplotlib()
    idx = index_records(records)
    scales = scales_in(records)
    artifacts = []
    for op in COMPARABLE_OPS:
        series = {}
        for engine in engines_in(records, ops=[op], require_metric=True):
            xs, ys = [], []
            for scale in scales:
                r = idx.get((engine, scale, op))
                if r and r.get('time'):
                    xs.append(r['n_edges'])
                    ys.append(r['time']['median_s'])
            if len(xs) >= 2:
                series[engine] = (xs, ys)
        if not series:
            continue
        fig, ax = plt.subplots(figsize=(7.2, 4.5))
        for engine, (xs, ys) in series.items():
            ax.plot(
                xs,
                ys,
                marker='o',
                linewidth=2.2,
                markersize=5,
                label=engine_label(engine),
                color=ENGINE_COLORS.get(engine),
            )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('edges')
        ax.set_ylabel('median time (s)')
        ax.set_title(one_line_op(op))
        _style_axis(ax)
        ax.legend(frameon=False)
        fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.15)
        png = plots_dir / f'{op}.png'
        _save(fig, png)
        plt.close(fig)
        artifacts.append((f'{one_line_op(op)} scaling', png))
    return artifacts


def render_extra_plots(records: list[dict], plots_dir: Path) -> list[str]:
    timed = [r for r in records if median_s(r) is not None]
    extra_groups = {group for group, _title, _blurb in EXTRA_GROUPS}
    extra = [r for r in timed if r.get('group') in extra_groups]
    lines = []
    try:
        _matplotlib()
    except Exception:
        return lines

    artifacts = []
    overview = _barh(
        timed,
        plots_dir / 'overview_wall_time_top18.png',
        'Slowest timed operations',
        limit=18,
    )
    if overview:
        artifacts.append(('Slowest operations', overview))

    for group, title, _blurb in EXTRA_GROUPS:
        group_recs = [r for r in extra if r.get('group') == group]
        if group == 'backend_operations':
            png = _plot_backend_grouped(
                group_recs,
                plots_dir / f'extra_{slug(group)}_wall_time.png',
                title,
            )
        elif group == 'accessors':
            png = _plot_accessors(
                group_recs,
                plots_dir / f'extra_{slug(group)}_wall_time.png',
                title,
            )
        else:
            png = _plot_scale_grouped(
                group_recs,
                plots_dir / f'extra_{slug(group)}_wall_time.png',
                title,
            )
        if png:
            artifacts.append((title, png))

    if not artifacts:
        return lines
    lines += ['## Extra Dimension Plots', '', 'Wall-time summaries for the local dimensions.', '']
    for label, png in artifacts:
        lines.append(f'### {label}')
        lines.append('')
        lines.append(f'![{label}]({rel(png, plots_dir.parent)})')
        lines.append('')
    return lines


def render_io_adapter_plots(payload: dict, plots_dir: Path) -> list[str]:
    lines = []
    try:
        _matplotlib()
    except Exception:
        return lines

    artifacts = []
    io_rows = []
    for rec in payload.get('io_formats') or []:
        if rec.get('error'):
            continue
        write_ms = ms(rec.get('write'))
        read_ms = ms(rec.get('read'))
        if write_ms is not None:
            io_rows.append({'format': rec.get('format'), 'phase': 'write', 'time': write_ms / 1e3})
        if read_ms is not None:
            io_rows.append({'format': rec.get('format'), 'phase': 'read', 'time': read_ms / 1e3})
    io_png = _plot_phase_grouped(
        io_rows,
        plots_dir / 'io_formats_wall_time.png',
        'IO formats: read/write median time',
        category_key='format',
        phase_key='phase',
        value_key='time',
    )
    if io_png:
        artifacts.append(('IO formats', io_png))

    adapter_rows = []
    for rec in payload.get('adapters') or []:
        if rec.get('skipped') or rec.get('error'):
            continue
        export_ms = ms(rec.get('export'))
        import_ms = ms(rec.get('import'))
        if export_ms is not None:
            adapter_rows.append(
                {'adapter': rec.get('adapter'), 'phase': 'export', 'time': export_ms / 1e3}
            )
        if import_ms is not None:
            adapter_rows.append(
                {'adapter': rec.get('adapter'), 'phase': 'import', 'time': import_ms / 1e3}
            )
    adapter_png = _plot_phase_grouped(
        adapter_rows,
        plots_dir / 'adapter_conversions_wall_time.png',
        'Adapter conversions: export/import median time',
        category_key='adapter',
        phase_key='phase',
        value_key='time',
    )
    if adapter_png:
        artifacts.append(('Adapter conversions', adapter_png))

    if not artifacts:
        return lines
    lines += ['## IO And Adapter Plots', '', 'Focused plots for serialization and bridge costs.', '']
    for label, png in artifacts:
        lines.append(f'### {label}')
        lines.append('')
        lines.append(f'![{label}]({rel(png, plots_dir.parent)})')
        lines.append('')
    return lines


def artifact_index(records: list[dict], plots_dir: Path) -> list[str]:
    records_csv = write_records_csv(records, plots_dir)
    status_csv = write_status_csv(records, plots_dir)
    pngs = sorted(plots_dir.glob('*.png'))
    lines = [
        '## Artifact Index',
        '',
        '| artifact | path |',
        '|---|---|',
        f'| records CSV | `{rel(records_csv, plots_dir.parent)}` |',
        f'| status CSV | `{rel(status_csv, plots_dir.parent)}` |',
    ]
    for png in pngs:
        lines.append(f'| plot | `{rel(png, plots_dir.parent)}` |')
    lines.append('')
    return lines


def write_records_csv(records: list[dict], plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / 'benchmark_records.csv'
    fields = [
        'scale',
        'group',
        'engine',
        'backend',
        'op',
        'status',
        'n_vertices',
        'n_edges',
        'median_s',
        'mean_s',
        'p95_s',
        'note',
        'reason',
        'error',
    ]
    with path.open('w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in records:
            t = r.get('time') or {}
            writer.writerow(
                {
                    'scale': r.get('scale'),
                    'group': r.get('group'),
                    'engine': r.get('engine'),
                    'backend': r.get('backend'),
                    'op': r.get('op'),
                    'status': r.get('status', 'ok' if (r.get('time') or r.get('memory')) else ''),
                    'n_vertices': r.get('n_vertices'),
                    'n_edges': r.get('n_edges'),
                    'median_s': t.get('median_s'),
                    'mean_s': t.get('mean_s'),
                    'p95_s': t.get('p95_s'),
                    'note': r.get('note'),
                    'reason': r.get('reason'),
                    'error': r.get('error'),
                }
            )
    return path


def write_status_csv(records: list[dict], plots_dir: Path) -> Path:
    counts = {}
    for r in records:
        default_status = 'ok' if (r.get('time') or r.get('memory')) else 'unknown'
        key = (r.get('group'), r.get('status', default_status))
        counts[key] = counts.get(key, 0) + 1
    path = plots_dir / 'status_counts.csv'
    with path.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['group', 'status', 'count'])
        for (group, status), count in sorted(counts.items()):
            writer.writerow([group, status, count])
    return path


def _barh(records: list[dict], path: Path, title: str, *, limit: int = 24) -> Path | None:
    timed = [r for r in records if median_s(r) is not None]
    if not timed:
        return None
    timed = sorted(timed, key=lambda r: median_s(r) or 0)[-limit:]
    labels = [f'{r["scale"]} | {r["engine"]} | {one_line_op(r["op"])}' for r in timed]
    values = [median_s(r) for r in timed]
    colors = [ENGINE_COLORS.get(r.get('engine'), '#4C78A8') for r in timed]

    plt = _matplotlib()
    height = max(5.8, 0.48 * len(timed) + 1.5)
    fig, ax = plt.subplots(figsize=(13.5, height))
    bars = ax.barh(labels, values, color=colors)
    max_value = max(values) if values else 1.0
    ax.set_xlim(0, max_value * 1.18)
    ax.set_xlabel('median time (s)')
    ax.set_title(title)
    _style_axis(ax, grid_axis='x')
    ax.tick_params(axis='y', labelsize=8)
    for bar, value in zip(bars, values, strict=False):
        ax.text(
            value + max_value * 0.018,
            bar.get_y() + bar.get_height() / 2,
            fmt_ms(value * 1000),
            va='center',
            fontsize=8,
            color='#1F2326',
        )
    fig.subplots_adjust(left=0.34, right=0.94, top=0.92, bottom=0.10)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_scale_grouped(records: list[dict], path: Path, title: str, *, op_order=None) -> Path | None:
    records = [r for r in records if median_s(r) is not None]
    if not records:
        return None
    scales = ordered({r['scale'] for r in records}, SCALE_ORDER)
    ops = ordered({r['op'] for r in records}, op_order)

    plt = _matplotlib()
    width = 0.74 / max(1, len(scales))
    x = list(range(len(ops)))
    fig, ax = plt.subplots(figsize=(max(9.0, len(ops) * 1.15), 5.4))
    for idx, scale in enumerate(scales):
        offsets = [pos + (idx - (len(scales) - 1) / 2) * width for pos in x]
        vals = []
        for op in ops:
            rec = next((r for r in records if r['scale'] == scale and r['op'] == op), None)
            vals.append(max(median_s(rec) or 0, 1e-9))
        ax.bar(offsets, vals, width=width, label=scale, color=SCALE_COLORS.get(scale))
    ax.set_yscale('log')
    ax.set_ylabel('median time (s, log scale)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([op_label(op) for op in ops], fontsize=8)
    _style_axis(ax, grid_axis='y')
    ax.legend(title='scale', frameon=False, ncol=min(4, len(scales)))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.30)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_backend_grouped(records: list[dict], path: Path, title: str) -> Path | None:
    records = [r for r in records if median_s(r) is not None]
    if not records:
        return None
    scales = ordered({r['scale'] for r in records}, SCALE_ORDER)
    engines = [
        engine
        for engine in ('annnet', 'networkx', 'igraph', 'graph-tool')
        if any(r['engine'] == engine for r in records)
    ]
    ops = ordered(
        {r['op'] for r in records},
        {
            'add_vertices_bulk': 0,
            'add_edges_bulk': 1,
            'remove_edges_fraction': 2,
            'remove_vertices_fraction': 3,
            'set_vertex_attrs_bulk': 4,
            'set_edge_attrs_bulk': 5,
        },
    )

    plt = _matplotlib()
    ncols = min(2, len(scales))
    nrows = math.ceil(len(scales) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(10.5, ncols * 6.8), nrows * 4.9),
        sharey=True,
        squeeze=False,
    )
    width = 0.74 / max(1, len(engines))
    x = list(range(len(ops)))
    axes_flat = axes.ravel()
    for ax, scale in zip(axes_flat, scales, strict=False):
        for idx, engine in enumerate(engines):
            offsets = [pos + (idx - (len(engines) - 1) / 2) * width for pos in x]
            vals = []
            for op in ops:
                rec = next(
                    (
                        r
                        for r in records
                        if r['scale'] == scale and r['engine'] == engine and r['op'] == op
                    ),
                    None,
                )
                vals.append(max(median_s(rec) or 0, 1e-9))
            ax.bar(offsets, vals, width=width, label=engine_label(engine), color=ENGINE_COLORS.get(engine))
        ax.set_yscale('log')
        ax.set_title(scale)
        ax.set_xticks(x)
        ax.set_xticklabels([op_label(op) for op in ops], fontsize=8)
        _style_axis(ax, grid_axis='y')
    for ax in axes_flat[len(scales):]:
        ax.axis('off')
    for row in axes:
        row[0].set_ylabel('median time (s, log scale)')
    axes_flat[min(len(scales), len(axes_flat)) - 1].legend(
        title='engine', frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1)
    )
    fig.suptitle(title, fontsize=14, fontweight='semibold')
    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.15, wspace=0.10, hspace=0.48)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_accessors(records: list[dict], path: Path, title: str) -> Path | None:
    records = [r for r in records if median_s(r) is not None]
    if not records:
        return None
    engines = [
        engine
        for engine in ('networkx', 'igraph', 'graph-tool')
        if any(r['engine'] == engine for r in records)
    ]
    if not engines:
        return None
    scales = ordered({r['scale'] for r in records}, SCALE_ORDER)
    methods = [
        ('cold', 'cold\naccessor'),
        ('warm', 'warm\ncached'),
        ('reconvert', 'explicit\nconvert'),
        ('roundtrip', 'explicit\nroundtrip'),
        ('after_mutation', 'cache after\nmutation'),
    ]

    def method_of(op):
        if 'cold' in op:
            return 'cold'
        if 'warm' in op:
            return 'warm'
        if 'reconvert' in op:
            return 'reconvert'
        if 'roundtrip' in op:
            return 'roundtrip'
        if 'after_mutation' in op:
            return 'after_mutation'
        return op

    plt = _matplotlib()
    fig, axes = plt.subplots(
        1,
        len(engines),
        figsize=(max(9.5, len(engines) * 5.4), 5.4),
        sharey=True,
        squeeze=False,
    )
    width = 0.74 / max(1, len(scales))
    x = list(range(len(methods)))
    for ax, engine in zip(axes[0], engines, strict=False):
        for idx, scale in enumerate(scales):
            offsets = [pos + (idx - (len(scales) - 1) / 2) * width for pos in x]
            vals = []
            for method, _label in methods:
                rec = next(
                    (
                        r
                        for r in records
                        if r['engine'] == engine
                        and r['scale'] == scale
                        and method_of(r['op']) == method
                    ),
                    None,
                )
                vals.append(max(median_s(rec) or 0, 1e-9))
            ax.bar(offsets, vals, width=width, label=scale, color=SCALE_COLORS.get(scale))
        ax.set_yscale('log')
        ax.set_title(engine_label(engine))
        ax.set_xticks(x)
        ax.set_xticklabels([label for _method, label in methods], fontsize=8)
        _style_axis(ax, grid_axis='y')
    axes[0][0].set_ylabel('median time (s, log scale)')
    axes[0][-1].legend(title='scale', frameon=False, loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.suptitle(title, fontsize=14, fontweight='semibold')
    fig.subplots_adjust(left=0.08, right=0.86, top=0.82, bottom=0.30, wspace=0.08)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_phase_grouped(rows, path, title, *, category_key, phase_key, value_key) -> Path | None:
    rows = [row for row in rows if row.get(value_key) is not None]
    if not rows:
        return None
    categories = sorted({row[category_key] for row in rows})
    phases = [
        phase
        for phase in ('write', 'read', 'export', 'import')
        if any(row[phase_key] == phase for row in rows)
    ]

    plt = _matplotlib()
    width = 0.74 / max(1, len(phases))
    x = list(range(len(categories)))
    fig, ax = plt.subplots(figsize=(max(8.0, len(categories) * 1.3), 5.0))
    for idx, phase in enumerate(phases):
        offsets = [pos + (idx - (len(phases) - 1) / 2) * width for pos in x]
        vals = []
        for category in categories:
            row = next(
                (
                    row
                    for row in rows
                    if row[category_key] == category and row[phase_key] == phase
                ),
                None,
            )
            vals.append(max(row[value_key] if row else 0, 1e-9))
        ax.bar(offsets, vals, width=width, label=phase, color=PHASE_COLORS.get(phase))
    ax.set_yscale('log')
    ax.set_ylabel('median time (s, log scale)')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    _style_axis(ax, grid_axis='y')
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.88, bottom=0.18)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_scaling_dashboard(records: list[dict], path: Path) -> Path | None:
    scales = scales_in(records)
    if len(scales) < 2:
        return None
    idx = index_records(records)
    engines = engines_in(records, ops=COMPARABLE_OPS, require_metric=True)
    if not engines:
        return None

    plt = _matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), squeeze=False)
    axes_flat = axes.ravel()
    for ax, op in zip(axes_flat, COMPARABLE_OPS, strict=False):
        for engine in engines:
            xs, ys = [], []
            for scale in scales:
                r = idx.get((engine, scale, op))
                if r and r.get('time'):
                    xs.append(r['n_edges'])
                    ys.append(r['time']['median_s'])
            if xs:
                ax.plot(
                    xs,
                    ys,
                    marker='o',
                    markersize=5,
                    linewidth=2.2,
                    label=engine_label(engine),
                    color=ENGINE_COLORS.get(engine),
                )
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(one_line_op(op))
        ax.set_xlabel('edges')
        ax.set_ylabel('median time (s)')
        _style_axis(ax)
    axes_flat[-1].axis('off')
    handles, labels = [], []
    seen = set()
    for ax in axes_flat:
        for handle, label in zip(*ax.get_legend_handles_labels(), strict=False):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    axes_flat[-1].legend(handles, labels, loc='center', frameon=False, title='engine')
    fig.suptitle('Comparable operations scaling', fontsize=16, fontweight='semibold')
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.09, wspace=0.26, hspace=0.38)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_ratio_heatmap(records: list[dict], path: Path, baseline_engine: str) -> Path | None:
    scales = scales_in(records)
    if not scales:
        return None
    idx = index_records(records)
    values = []
    have_value = False
    for op in COMPARABLE_OPS:
        row = []
        for scale in scales:
            ann = idx.get(('annnet', scale, op))
            baseline = idx.get((baseline_engine, scale, op))
            ann_s = median_s(ann)
            baseline_s = median_s(baseline)
            ratio = ann_s / baseline_s if ann_s and baseline_s else None
            row.append(ratio)
            have_value = have_value or ratio is not None
        values.append(row)
    if not have_value:
        return None

    plt = _matplotlib()
    from matplotlib.colors import TwoSlopeNorm

    logs = [math.log10(v) for row in values for v in row if v and v > 0]
    vmax = max(1.0, min(4.0, max(abs(v) for v in logs)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    matrix = [[math.log10(v) if v and v > 0 else float('nan') for v in row] for row in values]
    fig, ax = plt.subplots(figsize=(max(8.8, len(scales) * 1.25), 5.2))
    im = ax.imshow(matrix, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_xticks(range(len(scales)))
    ax.set_xticklabels(scales)
    ax.set_yticks(range(len(COMPARABLE_OPS)))
    ax.set_yticklabels([one_line_op(op) for op in COMPARABLE_OPS])
    ax.set_title(f'{ratio_label(baseline_engine)} median-time ratio')
    ax.set_xticks([x - 0.5 for x in range(1, len(scales))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(COMPARABLE_OPS))], minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    for y, row in enumerate(values):
        for x, value in enumerate(row):
            if value is None:
                text = '-'
                color = '#555555'
            elif value >= 1000:
                text = f'{value:.0f}x'
                color = 'white'
            elif value >= 10:
                text = f'{value:.1f}x'
                color = 'white' if value >= 100 else '#111111'
            else:
                text = f'{value:.2f}x'
                color = 'white' if value <= 0.03 else '#111111'
            ax.text(x, y, text, ha='center', va='center', fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('log10 ratio; red = AnnNet slower, blue = faster')
    fig.subplots_adjust(left=0.20, right=0.92, top=0.88, bottom=0.15)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_memory_scaling(records: list[dict], path: Path) -> Path | None:
    memory_records = [r for r in records if r.get('op') == 'footprint' and r.get('memory')]
    if not memory_records:
        return None
    engines = engines_in(memory_records, ops=['footprint'], require_metric=True)

    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for engine in engines:
        xs, ys = [], []
        for r in sorted(
            [r for r in memory_records if r['engine'] == engine],
            key=lambda r: r['n_edges'],
        ):
            xs.append(r['n_edges'])
            ys.append(r['memory']['retained_bytes'] / 1024**2)
        if xs:
            ax.plot(
                xs,
                ys,
                marker='o',
                linewidth=2.2,
                label=engine_label(engine),
                color=ENGINE_COLORS.get(engine),
            )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('edges')
    ax.set_ylabel('retained memory (MB)')
    ax.set_title('Retained memory scaling')
    _style_axis(ax)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
    _save(fig, path)
    plt.close(fig)
    return path


def _plot_bytes_per_edge(records: list[dict], path: Path) -> Path | None:
    memory_records = [r for r in records if r.get('op') == 'footprint' and r.get('memory')]
    if not memory_records:
        return None
    scales = ordered({r['scale'] for r in memory_records}, SCALE_ORDER)
    engines = engines_in(memory_records, ops=['footprint'], require_metric=True)
    if not engines:
        return None

    plt = _matplotlib()
    width = 0.74 / max(1, len(engines))
    x = list(range(len(scales)))
    fig, ax = plt.subplots(figsize=(max(8.8, len(scales) * 1.25), 5.0))
    for idx, engine in enumerate(engines):
        offsets = [pos + (idx - (len(engines) - 1) / 2) * width for pos in x]
        vals = []
        for scale in scales:
            r = next((r for r in memory_records if r['scale'] == scale and r['engine'] == engine), None)
            vals.append((r['memory']['retained_bytes'] / max(1, r['n_edges'])) if r else 0)
        ax.bar(offsets, vals, width=width, label=engine_label(engine), color=ENGINE_COLORS.get(engine))
    ax.set_yscale('log')
    ax.set_ylabel('retained bytes / edge')
    ax.set_title('Memory density by scale')
    ax.set_xticks(x)
    ax.set_xticklabels(scales)
    _style_axis(ax, grid_axis='y')
    ax.legend(frameon=False, ncol=min(4, len(engines)))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.88, bottom=0.14)
    _save(fig, path)
    plt.close(fig)
    return path
