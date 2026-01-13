from dataclasses import dataclass


@dataclass
class Scale:
    name: str
    vertices: int
    edges: int
    hyperedges: int
    slices: int

SCALES = {
    "small":  Scale("small", 1_000, 5_000, 200, 3),
    "medium": Scale("medium", 10_000, 50_000, 4_000, 5),
    "large":  Scale("large", 100_000, 500_000, 20_000, 5),
}