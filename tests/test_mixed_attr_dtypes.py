"""An attribute written as an int stays writable as a float.

Dataframe backends infer a column's dtype from its first value, so a second
write of a wider Python type used to blow up inside the backend while building
the table. The merged dtype was already known; it just was not applied when the
column was built.
"""

from __future__ import annotations

import pytest

from annnet._support.dataframe_backend import (
    available_dataframe_backends,
    dataframe_from_rows,
    dataframe_to_rows,
)
from annnet.core.graph import AnnNet

BACKENDS = ('polars', 'pandas', 'pyarrow')


def two_nodes() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_nodes(['A', 'B'])
    return G


def test_a_float_widens_an_int_node_attribute() -> None:
    G = two_nodes()
    G.attrs.update('nodes', {'A': {'val': 45}})
    G.attrs.update('nodes', {'B': {'val': 45.6}})
    assert dict(G.attrs.row('nodes', 'A'))['val'] == 45.0
    assert dict(G.attrs.row('nodes', 'B'))['val'] == 45.6


def test_a_string_and_an_int_sit_in_one_attribute() -> None:
    """The store keeps each value as it was written; only the table widens.

    A column of a dataframe carries one type, so a table that holds both has to
    widen to the one that holds either. The store holds a cell per element, so
    nothing is converted to make room for its neighbour.
    """
    G = two_nodes()
    G.attrs.update('nodes', {'A': {'val': 45}})
    G.attrs.update('nodes', {'B': {'val': 'x'}})
    assert dict(G.attrs.row('nodes', 'A'))['val'] == 45
    assert dict(G.attrs.row('nodes', 'B'))['val'] == 'x'
    assert [row['val'] for row in dataframe_to_rows(G.attrs.nodes)] == ['45', 'x']


def test_a_float_widens_an_int_edge_attribute() -> None:
    G = two_nodes()
    G.add_edges('A', 'B', edge_id='e0')
    G.add_edges('B', 'A', edge_id='e1')
    G.attrs.update('edges', {'e0': {'val': 45}})
    G.attrs.update('edges', {'e1': {'val': 45.6}})
    assert dict(G.attrs.row('edges', 'e0'))['val'] == 45.0
    assert dict(G.attrs.row('edges', 'e1'))['val'] == 45.6


def test_the_widened_column_survives_a_third_write() -> None:
    G = two_nodes()
    G.attrs.update('nodes', {'A': {'val': 45}})
    G.attrs.update('nodes', {'B': {'val': 45.6}})
    G.attrs.update('nodes', {'A': {'val': 1}})
    assert dict(G.attrs.row('nodes', 'A'))['val'] == 1.0


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_mixed_column_builds_on_every_backend(backend: str) -> None:
    if not available_dataframe_backends().get(backend):
        pytest.skip(f'{backend} is not installed')
    rows = [{'id': 'A', 'val': 45}, {'id': 'B', 'val': 45.6}]
    values = [row['val'] for row in dataframe_to_rows(dataframe_from_rows(rows, backend=backend))]
    assert values == [45.0, 45.6]


@pytest.mark.parametrize('backend', BACKENDS)
def test_a_declared_schema_still_wins_over_the_values(backend: str) -> None:
    if not available_dataframe_backends().get(backend):
        pytest.skip(f'{backend} is not installed')
    rows = [{'id': 'A', 'val': 45}, {'id': 'B', 'val': 45.6}]
    df = dataframe_from_rows(rows, schema={'id': 'text', 'val': 'text'}, backend=backend)
    values = [row['val'] for row in dataframe_to_rows(df)]
    assert all(isinstance(value, str) for value in values), values
    assert [float(value) for value in values] == [45.0, 45.6]
