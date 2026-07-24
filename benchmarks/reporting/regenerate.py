"""Regenerate benchmark reports and plots from saved results JSON files."""

from __future__ import annotations

import json
from pathlib import Path
import argparse

from .markdown import render


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'roots',
        nargs='*',
        help='result directories or results.json files; defaults to benchmarks/results/*/results.json',
    )
    args = parser.parse_args(argv)

    if args.roots:
        result_paths = [_as_results_path(Path(root)) for root in args.roots]
    else:
        result_paths = sorted(Path('benchmarks/results').glob('*/results.json'))

    for results_path in result_paths:
        root = results_path.parent
        payload = json.loads(results_path.read_text())
        report_path = render(payload, root / 'REPORT.md', plots_dir=root / 'plots')
        print(report_path)
    return 0


def _as_results_path(path: Path) -> Path:
    if path.is_dir():
        return path / 'results.json'
    return path


if __name__ == '__main__':
    raise SystemExit(main())
