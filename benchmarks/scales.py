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
    node_attrs: int = 8
    edge_attrs: int = 4
    sparse_every: int = 5
    remove_fraction: float = 0.05
    accessor_repeats: int = 10
    annotation_density: float | None = None

    @property
    def remove_vertices(self) -> int:
        return max(1, int(self.vertices * self.remove_fraction))

    @property
    def remove_edges(self) -> int:
        return max(1, int(self.edges * self.remove_fraction))


# One source of truth for scale, shared by run.py (comparison suite) and
# reporting/specsheet.py (headline PDF). Edge counts are ~4x vertices throughout
# so the curve shape is consistent across the whole ladder, up to 1M V / 4M E.
SCALES = {
    'tiny': Scale('tiny', 100, 400, 40, 2, node_attrs=4, edge_attrs=2, accessor_repeats=3),
    'xsmall': Scale('xsmall', 300, 1_200, 80, 2, node_attrs=6, edge_attrs=3, accessor_repeats=5),
    'small': Scale('small', 1_000, 4_000, 200, 3),
    'medium': Scale('medium', 10_000, 40_000, 4_000, 5),
    'large': Scale('large', 100_000, 400_000, 20_000, 10),
    'xlarge': Scale('xlarge', 1_000_000, 4_000_000, 200_000, 20),
}
