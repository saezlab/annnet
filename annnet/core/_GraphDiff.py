class GraphDiff:
    """Represents the difference between two graph states.

    Attributes
    --
    vertices_added : set
        Vertices in b but not in a
    vertices_removed : set
        Vertices in a but not in b
    edges_added : set
        Edges in b but not in a
    edges_removed : set
        Edges in a but not in b
    slices_added : set
        slices in b but not in a
    slices_removed : set
        slices in a but not in b

    """

    def __init__(self, snapshot_a, snapshot_b):
        self.snapshot_a = snapshot_a
        self.snapshot_b = snapshot_b

        # Compute differences
        self.vertices_added = snapshot_b["vertex_ids"] - snapshot_a["vertex_ids"]
        self.vertices_removed = snapshot_a["vertex_ids"] - snapshot_b["vertex_ids"]
        self.edges_added = snapshot_b["edge_ids"] - snapshot_a["edge_ids"]
        self.edges_removed = snapshot_a["edge_ids"] - snapshot_b["edge_ids"]
        self.slices_added = snapshot_b["slice_ids"] - snapshot_a["slice_ids"]
        self.slices_removed = snapshot_a["slice_ids"] - snapshot_b["slice_ids"]

    def summary(self):
        """Human-readable summary of differences."""
        lines = [
            f"Diff: {self.snapshot_a['label']} - {self.snapshot_b['label']}",
            "",
            f"Vertices: {len(self.vertices_added):+d} added, {len(self.vertices_removed)} removed",
            f"Edges: {len(self.edges_added):+d} added, {len(self.edges_removed)} removed",
            f"slices: {len(self.slices_added):+d} added, {len(self.slices_removed)} removed",
        ]
        return "\n".join(lines)

    def is_empty(self):
        """Check if there are no differences."""
        return (
            not self.vertices_added
            and not self.vertices_removed
            and not self.edges_added
            and not self.edges_removed
            and not self.slices_added
            and not self.slices_removed
        )

    def __repr__(self):
        return self.summary()

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "snapshot_a": self.snapshot_a["label"],
            "snapshot_b": self.snapshot_b["label"],
            "vertices_added": list(self.vertices_added),
            "vertices_removed": list(self.vertices_removed),
            "edges_added": list(self.edges_added),
            "edges_removed": list(self.edges_removed),
            "slices_added": list(self.slices_added),
            "slices_removed": list(self.slices_removed),
        }
