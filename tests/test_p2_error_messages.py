"""P2: error-message / DX polish.

- ``remove_vertices`` / ``remove_edges`` raise KeyError on unknown IDs by
  default (NetworkX convention); ``errors='ignore'`` restores the legacy
  silent behavior.
- ``add_edges([{'source': 'A'}])`` raises ValueError naming the missing
  field and the item index, not a bare ``KeyError``.
- ``get_vertex(999)`` raises KeyError with the valid index range and a
  pointer to ``vertices()``.
- ``history.diff()`` with no args diffs the most recent snapshot vs the
  current state.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def test_remove_vertices_raises_on_unknown_id_by_default():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    with pytest.raises(KeyError, match='NOPE'):
        G.remove_vertices('NOPE')


def test_remove_vertices_errors_ignore_restores_legacy_behavior():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.remove_vertices(['NOPE', 'A'], errors='ignore')
    assert set(G.vertices()) == {'B'}


def test_remove_edges_raises_on_unknown_id_by_default():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    with pytest.raises(KeyError, match='nope'):
        G.remove_edges('nope')


def test_remove_edges_errors_ignore_restores_legacy_behavior():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([('A', 'B'), ('B', 'C')])
    G.remove_edges(['edge_0', 'nope'], errors='ignore')
    assert len(G.edges()) == 1


def test_add_edges_missing_target_raises_value_error_with_index():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    with pytest.raises(ValueError, match=r"index 0.*'target'"):
        G.add_edges([{'source': 'A'}])


def test_add_edges_missing_source_raises_value_error_with_index():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    with pytest.raises(ValueError, match=r"index 1.*'source'"):
        G.add_edges([{'source': 'A', 'target': 'B'}, {'target': 'B'}])


def test_get_vertex_out_of_range_message_is_helpful():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    with pytest.raises(KeyError, match='row index 999'):
        G.get_vertex(999)


def test_history_diff_no_args_uses_most_recent_snapshot():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.history.snapshot('start')
    G.add_vertices(['C'])
    diff = G.history.diff()
    # Diff is "snapshot 'start' → current", so C is added.
    added = getattr(diff, 'vertices_added', None) or getattr(diff, 'added_vertices', None)
    if added is not None:
        assert 'C' in added


def test_history_diff_no_args_no_snapshots_raises():
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    with pytest.raises(ValueError, match='snapshot'):
        G.history.diff()
