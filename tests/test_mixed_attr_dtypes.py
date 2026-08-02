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


def two_vertices() -> AnnNet:
    G = AnnNet(directed=True)
    G.add_vertices(['A', 'B'])
    return G


def test_a_float_widens_an_int_vertex_attribute() -> None:
    G = two_vertices()
    G.attrs.set_vertex_attrs('A', val=45)
    G.attrs.set_vertex_attrs('B', val=45.6)
    assert G.attrs.get_vertex_attrs('A')['val'] == 45.0
    assert G.attrs.get_vertex_attrs('B')['val'] == 45.6


def test_a_string_widens_an_int_vertex_attribute() -> None:
    G = two_vertices()
    G.attrs.set_vertex_attrs('A', val=45)
    G.attrs.set_vertex_attrs('B', val='x')
    assert G.attrs.get_vertex_attrs('A')['val'] == '45'
    assert G.attrs.get_vertex_attrs('B')['val'] == 'x'


def test_a_float_widens_an_int_edge_attribute() -> None:
    G = two_vertices()
    G.add_edges('A', 'B', edge_id='e0')
    G.add_edges('B', 'A', edge_id='e1')
    G.attrs.set_edge_attrs('e0', val=45)
    G.attrs.set_edge_attrs('e1', val=45.6)
    assert G.attrs.get_edge_attrs('e0')['val'] == 45.0
    assert G.attrs.get_edge_attrs('e1')['val'] == 45.6


def test_the_widened_column_survives_a_third_write() -> None:
    G = two_vertices()
    G.attrs.set_vertex_attrs('A', val=45)
    G.attrs.set_vertex_attrs('B', val=45.6)
    G.attrs.set_vertex_attrs('A', val=1)
    assert G.attrs.get_vertex_attrs('A')['val'] == 1.0


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
