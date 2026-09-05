"""What a graph was built from.

The history log records the *edits*, so reloading a file tells you what was done
and not what it was done to. A graph assembled from four downloaded tables knows
everything about its own structure and nothing about where it came from — and a
release that changed under you leaves no trace either way.
"""

from __future__ import annotations

import annnet as an
from annnet.core._provenance import KEY, CHECKSUM_LIMIT, checksum_of


def _graph():
    G = an.Graph(directed=True)
    G.add_nodes(['A', 'B'])
    return G


class TestRecording:
    def test_a_record_comes_back(self):
        G = _graph()
        G.provenance.record('OmniPath', version='2024.1')
        assert [entry['name'] for entry in G.provenance.records()] == ['OmniPath']

    def test_records_keep_the_order_they_were_read_in(self):
        G = _graph()
        for name in ('first', 'second', 'third'):
            G.provenance.record(name)
        assert [entry['name'] for entry in G.provenance.records()] == [
            'first',
            'second',
            'third',
        ]

    def test_a_retrieval_time_is_always_stamped(self):
        G = _graph()
        entry = G.provenance.record('OmniPath')
        assert entry['retrieved']

    def test_extra_fields_are_kept(self):
        G = _graph()
        entry = G.provenance.record('OmniPath', reader='from_edge_frame', format='tsv')
        assert entry['reader'] == 'from_edge_frame'
        assert entry['format'] == 'tsv'

    def test_a_none_field_is_not_kept(self):
        G = _graph()
        assert 'version' not in G.provenance.record('OmniPath', version=None)

    def test_it_lives_in_uns_under_a_reserved_key(self):
        G = _graph()
        G.provenance.record('OmniPath')
        assert KEY in G.uns

    def test_the_accessor_is_sized_and_iterable(self):
        G = _graph()
        G.provenance.record('one')
        G.provenance.record('two')
        assert len(G.provenance) == 2
        assert [entry['name'] for entry in G.provenance] == ['one', 'two']

    def test_a_graph_that_records_nothing_reports_nothing(self):
        assert len(_graph().provenance) == 0


class TestChecksum:
    def test_a_file_is_hashed(self, tmp_path):
        path = tmp_path / 'resource.tsv'
        path.write_text('source\ttarget\nA\tB\n')
        G = _graph()
        entry = G.provenance.record('local', uri=path)
        assert entry['checksum'] == checksum_of(path)

    def test_the_same_bytes_hash_the_same(self, tmp_path):
        left, right = tmp_path / 'a', tmp_path / 'b'
        left.write_text('same')
        right.write_text('same')
        assert checksum_of(left) == checksum_of(right)

    def test_different_bytes_hash_differently(self, tmp_path):
        left, right = tmp_path / 'a', tmp_path / 'b'
        left.write_text('one')
        right.write_text('two')
        assert checksum_of(left) != checksum_of(right)

    def test_a_missing_file_hashes_to_nothing_rather_than_raising(self, tmp_path):
        assert checksum_of(tmp_path / 'absent') is None

    def test_a_file_past_the_limit_records_no_checksum(self, tmp_path):
        """Not taken, rather than taken slowly — and said so by its absence."""
        path = tmp_path / 'big.tsv'
        path.write_text('x' * 64)
        assert checksum_of(path, limit=8) is None

    def test_the_limit_is_declared(self):
        assert CHECKSUM_LIMIT > 0

    def test_checksum_false_records_none(self, tmp_path):
        path = tmp_path / 'resource.tsv'
        path.write_text('data')
        G = _graph()
        assert 'checksum' not in G.provenance.record('local', uri=path, checksum=False)

    def test_a_source_may_state_its_own(self):
        G = _graph()
        entry = G.provenance.record('remote', uri='https://example.org/x', checksum='abc123')
        assert entry['checksum'] == 'abc123'

    def test_a_uri_that_is_not_a_file_records_no_checksum(self):
        G = _graph()
        assert 'checksum' not in G.provenance.record('remote', uri='https://example.org/x')


class TestFrame:
    def test_it_is_callable_and_gives_a_table(self):
        G = _graph()
        G.provenance.record('OmniPath', version='2024.1')
        rows = _rows(G.provenance())
        assert len(rows) == 1
        assert rows[0]['name'] == 'OmniPath'

    def test_the_leading_columns_are_the_declared_ones(self):
        G = _graph()
        G.provenance.record('OmniPath')
        columns = _columns(G.provenance())
        assert columns[:3] == ['name', 'format', 'uri']

    def test_an_extra_field_becomes_a_column(self):
        G = _graph()
        G.provenance.record('OmniPath', organism='human')
        assert 'organism' in _columns(G.provenance())

    def test_a_graph_with_no_records_gives_an_empty_table(self):
        frame = _graph().provenance()
        assert _rows(frame) == []
        assert 'name' in _columns(frame)


def _rows(frame):
    from annnet._support.dataframe_backend import dataframe_to_rows

    return dataframe_to_rows(frame)


def _columns(frame):
    from annnet._support.dataframe_backend import dataframe_columns

    return list(dataframe_columns(frame))


class TestReadersRecordThemselves:
    """An accessor nothing writes to is a feature that does not exist."""

    def test_from_edge_frame_records_the_table(self):
        G = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        assert [entry['name'] for entry in G.provenance.records()] == ['edge frame']

    def test_the_record_names_the_reader(self):
        G = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        assert G.provenance.records()[0]['reader'] == 'from_edge_frame'

    def test_from_cx2_records_the_file(self, tmp_path):
        G = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        path = tmp_path / 'g.cx2'
        an.to_cx2(G, path)
        back = an.from_cx2(path)
        assert 'CX2' in [entry['name'] for entry in back.provenance.records()]

    def test_a_chain_of_readers_keeps_every_step(self, tmp_path):
        """Where the data came from, then what it passed through."""
        G = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        path = tmp_path / 'g.cx2'
        an.to_cx2(G, path)
        back = an.from_cx2(path)
        assert [entry['name'] for entry in back.provenance.records()] == ['edge frame', 'CX2']

    def test_reading_the_native_format_adds_nothing(self, tmp_path):
        """`read` restores records; adding one would grow them on every round trip.

        The path of a file you just named is the one fact the caller already has,
        and what the file's own records say is where the data came from.
        """
        G = an.from_edge_frame([{'source': 'A', 'target': 'B'}])
        path = tmp_path / 'g.annnet'
        G.write(path)
        back = an.read(path)
        assert [entry['name'] for entry in back.provenance.records()] == ['edge frame']
