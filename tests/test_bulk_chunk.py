"""A bulk write is chunked by the store, not by the convention of its callers.

``BULK_CHUNK`` says how many edges a caller should hold back before it calls the
bulk write. The number is not arbitrary: a batch of specs and their member
entries are tracked containers, and one that survives a collection of the
youngest generation is promoted, so every collection after it scans the batch
again. A load of 25 600 edges written in one call spent 50 milliseconds in the
collector, which is the whole of what the bulk write saves.

Three call sites respected that by hand and nothing made them. `FR-020`: a caller
who submits work in any shape gets the result a caller who respects the limit
gets — the same slots, the same arrays, the same clock and the same append log.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet.core import _store as ST


def _specs(count: int, *, nodes: int = 64, first: int = 0) -> list[tuple]:
    return [
        (
            f'e{first + i}',
            [
                ((f'v{(first + i) % nodes}', ('_',)), 1.0, ST.SOURCE),
                ((f'v{(first + i + 1) % nodes}', ('_',)), -1.0, ST.TARGET),
            ],
            ST.BINARY,
            True,
            float(first + i),
            False,
            None,
            None,
            None,
        )
        for i in range(count)
    ]


def _store_with_nodes(nodes: int = 64) -> ST.CoreState:
    store = ST.CoreState(directed=True)
    for i in range(nodes):
        store.add_entity((f'v{i}', ('_',)))
    return store


def _fingerprint(store: ST.CoreState) -> dict:
    return {
        'edge_ids': list(store._edge_id),
        'edge_slot': dict(store._edge_slot),
        'kind': store.edge_kind[: len(store._edge_id)].tolist(),
        'directed': store.edge_directed[: len(store._edge_id)].tolist(),
        'weight': store.edge_weight[: len(store._edge_id)].tolist(),
        'member_start': store.member_start[: len(store._edge_id)].tolist(),
        'member_len': store.member_len[: len(store._edge_id)].tolist(),
        'member_ent': store.member_ent[: store._member_used].tolist(),
        'member_coef': store.member_coef[: store._member_used].tolist(),
        'member_role': store.member_role[: store._member_used].tolist(),
        'version': store.structure_version,
        'append_log': list(store.append_log),
        'append_log_from': store.append_log_from_version,
        'free': list(store.edge_free),
    }


def _chunked(store: ST.CoreState, specs: list) -> list[int]:
    """What a caller that respects the limit does."""
    slots = []
    for start in range(0, len(specs), ST.BULK_CHUNK):
        slots.extend(store.add_edges(specs[start : start + ST.BULK_CHUNK]))
    return slots


class TestAnyShapeGivesTheSameResult:
    @pytest.mark.parametrize('count', [ST.BULK_CHUNK + 1, 3 * ST.BULK_CHUNK, 1000])
    def test_one_call_matches_a_caller_that_chunks(self, count):
        specs = _specs(count)
        one, many = _store_with_nodes(), _store_with_nodes()
        assert one.add_edges(list(specs)) == _chunked(many, list(specs))
        assert _fingerprint(one) == _fingerprint(many)

    def test_it_holds_when_freed_slots_are_reused(self):
        specs = _specs(3 * ST.BULK_CHUNK)

        def prepare():
            store = _store_with_nodes()
            store.add_edges(_specs(40, first=10_000))
            for i in range(10_000, 10_020):
                store.remove_edge(f'e{i}')
            return store

        one, many = prepare(), prepare()
        assert one.add_edges(list(specs)) == _chunked(many, list(specs))
        assert _fingerprint(one) == _fingerprint(many)

    def test_a_small_batch_is_untouched(self):
        specs = _specs(ST.BULK_CHUNK)
        one, many = _store_with_nodes(), _store_with_nodes()
        assert one.add_edges(list(specs)) == _chunked(many, list(specs))
        assert _fingerprint(one) == _fingerprint(many)

    def test_a_generator_is_accepted_and_chunked(self):
        specs = _specs(3 * ST.BULK_CHUNK)
        one, many = _store_with_nodes(), _store_with_nodes()
        assert one.add_edges(iter(list(specs))) == _chunked(many, list(specs))
        assert _fingerprint(one) == _fingerprint(many)


class TestABadBatchStillLeavesTheStoreAsItWas:
    """Chunking must not turn one refusal into a partial write."""

    def test_an_unknown_entity_late_in_the_batch_writes_nothing(self):
        specs = _specs(3 * ST.BULK_CHUNK)
        specs[-1] = (
            specs[-1][0],
            [(('nobody', ('_',)), 1.0, ST.SOURCE)],
            *specs[-1][2:],
        )
        store = _store_with_nodes()
        before = _fingerprint(store)
        with pytest.raises(KeyError, match='does not hold'):
            store.add_edges(specs)
        assert _fingerprint(store) == before

    def test_a_duplicate_across_two_chunks_writes_nothing(self):
        specs = _specs(3 * ST.BULK_CHUNK)
        specs[-1] = ('e0', *specs[-1][1:])
        store = _store_with_nodes()
        before = _fingerprint(store)
        with pytest.raises(KeyError, match='Duplicate edge id'):
            store.add_edges(specs)
        assert _fingerprint(store) == before

    def test_an_id_the_store_already_holds_writes_nothing(self):
        store = _store_with_nodes()
        store.add_edges(_specs(10, first=500))
        before = _fingerprint(store)
        specs = _specs(3 * ST.BULK_CHUNK)
        specs[-1] = ('e500', *specs[-1][1:])
        with pytest.raises(KeyError, match='Duplicate edge id'):
            store.add_edges(specs)
        assert _fingerprint(store) == before


def test_no_single_call_reaches_the_store_with_more_than_the_limit(monkeypatch):
    """The enforcement is in the store, so it holds for every caller."""
    store = _store_with_nodes()
    widths = []
    original = ST.CoreState._add_edges

    def watched(self, specs):
        widths.append(len(specs))
        return original(self, specs)

    monkeypatch.setattr(ST.CoreState, '_add_edges', watched)
    store.add_edges(_specs(1000))
    assert widths
    assert max(widths) <= ST.BULK_CHUNK


def test_the_graph_gives_the_same_answer_whatever_the_batch_size():
    from annnet import AnnNet

    def built(chunk):
        graph = AnnNet(directed=True)
        payload = [
            {'source': f'v{i % 64}', 'target': f'v{(i + 1) % 64}', 'edge_id': f'e{i}'}
            for i in range(1000)
        ]
        for start in range(0, len(payload), chunk):
            graph.add_edges(payload[start : start + chunk])
        return graph

    whole, chunked = built(1000), built(ST.BULK_CHUNK)
    assert whole.E.ids == chunked.E.ids
    assert np.array_equal(np.asarray(whole.B.todense()), np.asarray(chunked.B.todense()))
