"""Unit tests for annnet/core/_History.py — History mixin and GraphDiff."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from annnet.core._History import GraphDiff
from annnet.core.graph import AnnNet


def _make_snapshot(label, vertex_ids, edge_ids, slice_ids=None):
    """Minimal snapshot dict for GraphDiff construction."""
    return {
        "label": label,
        "version": 0,
        "vertex_ids": set(vertex_ids),
        "edge_ids": set(edge_ids),
        "slice_ids": set(slice_ids or []),
    }


class TestHistoryLogging(unittest.TestCase):
    """History is enabled by default; mutations are logged."""

    def test_history_enabled_by_default(self):
        G = AnnNet()
        self.assertTrue(G._history_enabled)
        self.assertTrue(callable(G.history))

    def test_add_vertex_logged(self):
        G = AnnNet()
        G.add_vertex("A")
        events = G.history()
        ops = [e["op"] for e in events]
        self.assertIn("add_vertex", ops)

    def test_add_edge_logged(self):
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        G.history.clear()
        G.add_edge("A", "B")
        ops = [e["op"] for e in G.history()]
        self.assertIn("add_edge", ops)

    def test_version_increments(self):
        G = AnnNet()
        v0 = G._version
        G.add_vertex("A")
        G.add_vertex("B")
        self.assertGreater(G._version, v0)
        self.assertEqual(G._version, v0 + 2)

    def test_event_fields_present(self):
        G = AnnNet()
        G.add_vertex("X")
        evt = G.history()[-1]
        self.assertIn("version", evt)
        self.assertIn("ts_utc", evt)
        self.assertIn("mono_ns", evt)
        self.assertIn("op", evt)

    def test_disable_and_reenable_history(self):
        G = AnnNet()
        G.history.enable(False)
        G.add_vertex("A")
        before = len(G.history())
        G.history.enable(True)
        G.add_vertex("B")
        after = len(G.history())
        self.assertEqual(after, before + 1)

    def test_version_increments_even_when_history_disabled(self):
        G = AnnNet()
        v0 = G._version
        G.history.enable(False)
        G.add_vertex("A")
        G.add_vertex("B")
        self.assertEqual(G._version, v0 + 2)
        self.assertEqual(len(G.history()), 0)

    def test_clear_history(self):
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        self.assertGreater(len(G.history()), 0)
        G.history.clear()
        self.assertEqual(len(G.history()), 0)

    def test_history_as_dataframe(self):
        G = AnnNet()
        G.add_vertex("A")
        df = G.history(as_df=True)
        # Should have at least one row and the standard columns
        self.assertGreater(len(df), 0)
        cols = list(df.columns) if hasattr(df, "columns") else []
        self.assertIn("op", cols)
        self.assertIn("version", cols)

    def test_history_namespace_methods_work(self):
        G = AnnNet()
        G.history.clear()
        G.add_vertex("A")
        self.assertGreater(len(G.history()), 0)
        G.history.enable(False)
        before = len(G.history())
        G.add_vertex("B")
        self.assertEqual(len(G.history()), before)
        G.history.enable(True)


class TestMark(unittest.TestCase):
    """mark() inserts a labelled event."""

    def test_mark_appears_in_history(self):
        G = AnnNet()
        G.history.clear()
        G.history.mark("checkpoint_1")
        events = G.history()
        ops = [e["op"] for e in events]
        self.assertIn("mark", ops)

    def test_mark_label_stored(self):
        G = AnnNet()
        G.history.clear()
        G.history.mark("my_label")
        evt = next(e for e in G.history() if e["op"] == "mark")
        self.assertEqual(evt["label"], "my_label")

    def test_mark_ignored_when_disabled(self):
        G = AnnNet()
        G.history.enable(False)
        G.history.clear()
        G.history.mark("should_not_appear")
        self.assertEqual(len(G.history()), 0)
        G.history.enable(True)


class TestExportHistory(unittest.TestCase):
    """export_history() writes events to disk."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil

        shutil.rmtree(self.tmpdir)

    def test_export_returns_zero_when_empty(self):
        G = AnnNet()
        G.history.clear()
        path = os.path.join(self.tmpdir, "empty.parquet")
        n = G.history.export(path)
        self.assertEqual(n, 0)

    def test_export_json(self):
        G = AnnNet()
        G.add_vertex("A")
        path = os.path.join(self.tmpdir, "hist.json")
        n = G.history.export(path)
        self.assertGreater(n, 0)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_export_ndjson(self):
        G = AnnNet()
        G.add_vertex("A")
        path = os.path.join(self.tmpdir, "hist.ndjson")
        n = G.history.export(path)
        self.assertGreater(n, 0)
        self.assertTrue(os.path.exists(path))
        lines = open(path).readlines()
        self.assertGreater(len(lines), 0)
        json.loads(lines[0])  # valid JSON per line

    def test_export_returns_event_count(self):
        """export_history() returns the number of events written (JSON path)."""
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        path = os.path.join(self.tmpdir, "hist2.json")
        n = G.history.export(path)
        # The JSON export should succeed and match len(G.history())
        self.assertEqual(n, len(G.history()))

    def test_export_csv_serializes_nested_payloads(self):
        G = AnnNet(aspects={"condition": ["healthy", "treated"], "time": ["t0", "t1"]})
        G.add_vertex("A", layer=("healthy", "t0"))
        G.add_vertex("A", layer=("treated", "t1"))
        path = os.path.join(self.tmpdir, "hist.csv")
        n = G.history.export(path)
        self.assertGreater(n, 0)
        self.assertTrue(os.path.exists(path))
        text = open(path, encoding="utf-8").read()
        self.assertIn("add_vertex", text)


class TestSnapshot(unittest.TestCase):
    """snapshot(), diff(), list_snapshots()."""

    def test_snapshot_returns_dict(self):
        G = AnnNet()
        G.add_vertex("A")
        snap = G.history.snapshot(label="s1")
        self.assertIsInstance(snap, dict)
        self.assertEqual(snap["label"], "s1")
        self.assertIn("vertex_ids", snap)
        self.assertIn("edge_ids", snap)

    def test_snapshot_captures_current_state(self):
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        snap = G.history.snapshot()
        self.assertIn("A", snap["vertex_ids"])
        self.assertIn("B", snap["vertex_ids"])

    def test_list_snapshots(self):
        G = AnnNet()
        G.history.snapshot(label="a")
        G.history.snapshot(label="b")
        labels = [s["label"] for s in G.history.list_snapshots()]
        self.assertIn("a", labels)
        self.assertIn("b", labels)

    def test_diff_detects_added_vertex(self):
        G = AnnNet()
        G.add_vertex("A")
        snap1 = G.history.snapshot(label="before")
        G.add_vertex("B")
        d = G.history.diff("before")
        self.assertIn("B", d.vertices_added)
        self.assertNotIn("A", d.vertices_added)

    def test_history_namespace_exposes_snapshot_ops(self):
        G = AnnNet()
        G.add_vertex("A")
        snap = G.history.snapshot("s1")
        self.assertEqual(snap["label"], "s1")
        self.assertEqual(len(G.history.list_snapshots()), 1)

    def test_diff_detects_removed_vertex(self):
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        snap1 = G.history.snapshot(label="before")
        G.remove_vertex("B")
        d = G.history.diff("before")
        self.assertIn("B", d.vertices_removed)

    def test_diff_detects_added_edge(self):
        G = AnnNet()
        G.add_vertex("A")
        G.add_vertex("B")
        snap1 = G.history.snapshot(label="before")
        eid = G.add_edge("A", "B", edge_id="e1")
        d = G.history.diff("before")
        self.assertIn("e1", d.edges_added)

    def test_diff_between_two_snapshots(self):
        G = AnnNet()
        G.add_vertex("A")
        snap1 = G.history.snapshot(label="s1")
        G.add_vertex("B")
        snap2 = G.history.snapshot(label="s2")
        d = G.history.diff("s1", "s2")
        self.assertIn("B", d.vertices_added)


class TestGraphDiff(unittest.TestCase):
    """GraphDiff directly — summary(), is_empty(), to_dict()."""

    def test_is_empty_when_identical(self):
        snap = _make_snapshot("s", ["A", "B"], ["e1"])
        d = GraphDiff(snap, snap)
        self.assertTrue(d.is_empty())

    def test_is_not_empty_after_vertex_added(self):
        s1 = _make_snapshot("s1", ["A"], ["e1"])
        s2 = _make_snapshot("s2", ["A", "B"], ["e1"])
        d = GraphDiff(s1, s2)
        self.assertFalse(d.is_empty())
        self.assertIn("B", d.vertices_added)

    def test_vertices_removed(self):
        s1 = _make_snapshot("s1", ["A", "B"], [])
        s2 = _make_snapshot("s2", ["A"], [])
        d = GraphDiff(s1, s2)
        self.assertIn("B", d.vertices_removed)

    def test_edges_added_and_removed(self):
        s1 = _make_snapshot("s1", ["A", "B"], ["e1"])
        s2 = _make_snapshot("s2", ["A", "B"], ["e2"])
        d = GraphDiff(s1, s2)
        self.assertIn("e2", d.edges_added)
        self.assertIn("e1", d.edges_removed)

    def test_slices_added(self):
        s1 = _make_snapshot("s1", [], [], slice_ids=["default"])
        s2 = _make_snapshot("s2", [], [], slice_ids=["default", "new_slice"])
        d = GraphDiff(s1, s2)
        self.assertIn("new_slice", d.slices_added)

    def test_summary_returns_string(self):
        s1 = _make_snapshot("s1", ["A"], [])
        s2 = _make_snapshot("s2", ["A", "B"], [])
        d = GraphDiff(s1, s2)
        txt = d.summary()
        self.assertIsInstance(txt, str)
        self.assertIn("s1", txt)
        self.assertIn("s2", txt)

    def test_repr_calls_summary(self):
        s1 = _make_snapshot("s1", [], [])
        s2 = _make_snapshot("s2", ["X"], [])
        d = GraphDiff(s1, s2)
        self.assertEqual(repr(d), d.summary())

    def test_to_dict_serializable(self):
        s1 = _make_snapshot("s1", ["A"], ["e1"])
        s2 = _make_snapshot("s2", ["A", "B"], ["e2"])
        d = GraphDiff(s1, s2)
        data = d.to_dict()
        self.assertEqual(data["snapshot_a"], "s1")
        self.assertEqual(data["snapshot_b"], "s2")
        self.assertIn("B", data["vertices_added"])
        self.assertIn("e1", data["edges_removed"])
        self.assertIn("e2", data["edges_added"])
        # Ensure all values are JSON-serializable
        json.dumps(data)


if __name__ == "__main__":
    unittest.main()
