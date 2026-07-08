"""Benchmark scale definitions shared by the suite."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scale:
    name: str
    vertices: int
    edges: int
    hyperedges: int
    slices: int


# One source of truth for scale, shared by run.py (comparison suite) and
# specsheet.py (headline PDF). Edge counts are ~4x vertices throughout so the
# curve shape is consistent across the whole ladder, up to 1M V / 4M E.
SCALES = {
    'small': Scale('small', 1_000, 4_000, 200, 3),
    'medium': Scale('medium', 10_000, 40_000, 4_000, 5),
    'large': Scale('large', 100_000, 400_000, 20_000, 10),
    'xlarge': Scale('xlarge', 1_000_000, 4_000_000, 200_000, 20),
}
