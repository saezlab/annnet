"""The node table and the edge table materialize on every dataframe backend.

The store keeps its own typed columns, so the backend matters only when a table
materializes. Narwhals is the layer that makes one materialization serve every
backend, and that is what this checks.
"""

from __future__ import annotations

import numpy as np
import pytest

from annnet._support.dataframe_backend import available_dataframe_backends
from annnet.core import _attrs as A, _store as ST

FLAT = ('_',)
BACKENDS = ('polars', 'pandas', 'pyarrow')


def key(node_id):
    return (node_id, FLAT)


def graph_with_attributes():
    store = ST.CoreState(directed=True)
    attrs = A.AttributeStore(store)
    for node_id in ('A', 'B', 'C'):
        store.add_entity(key(node_id))
    store.add_edge(
        'e0',
        [(key('A'), 1.0, ST.SOURCE), (key('B'), -1.0, ST.TARGET)],
        kind=ST.BINARY,
        directed=True,
        weight=1.5,
    )
    attrs.set_node_column('score', np.array([0.1, 0.2, 0.3]))
    attrs.set_node(key('A'), 'label', 'first')
    attrs.set_edge('e0', 'confidence', 0.9)
    return store, attrs


def _available(name: str) -> bool:
    return bool(available_dataframe_backends().get(name))


@pytest.mark.parametrize('backend', BACKENDS)
def test_the_node_table_materializes_on_every_backend(backend):
    if not _available(backend):
        pytest.skip(f'{backend} is not installed')
    _store, attrs = graph_with_attributes()
    table = attrs.obs(backend=backend)
    assert table is not None
    rows = attrs.obs_rows()
    assert {row['node_id'] for row in rows} == {'A', 'B', 'C'}


@pytest.mark.parametrize('backend', BACKENDS)
def test_the_edge_table_materializes_on_every_backend(backend):
    if not _available(backend):
        pytest.skip(f'{backend} is not installed')
    _store, attrs = graph_with_attributes()
    assert attrs.var(backend=backend) is not None


def test_every_backend_reports_the_same_columns_and_values():
    backends = [name for name in BACKENDS if _available(name)]
    if len(backends) < 2:
        pytest.skip('at least two dataframe backends are needed to compare them')
    _store, attrs = graph_with_attributes()
    seen = {}
    for backend in backends:
        attrs.drop_tables()
        table = attrs.obs(backend=backend)
        import narwhals as nw

        frame = nw.from_native(table, eager_only=True)
        seen[backend] = (
            tuple(sorted(frame.columns)),
            {row['node_id']: row.get('score') for row in frame.to_native().to_dicts()}
            if hasattr(frame.to_native(), 'to_dicts')
            else None,
        )
    columns = {value[0] for value in seen.values()}
    assert len(columns) == 1, f'the backends disagree on the columns: {seen}'
