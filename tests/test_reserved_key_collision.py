"""Setting a reserved key as a user attribute must raise, not silently drop.

Reserved names exist for clear structural reasons: they're either column
names of the underlying tables (``vertex_id``, ``edge_id``, ``slice_id``,
``weight``), structural-edge fields (``source``, ``target``, ``directed``,
``edge_type``, ``slice``), or input-shape signal keys for ``add_edges``
(``members``, ``head``, ``tail``, ``flexible``, ``slice_weight``, ``kind``).

If a caller tries to set one as a *user attribute*, the right behavior is
to raise — silently swallowing the value loses data invisibly.
"""

from __future__ import annotations

import pytest

from annnet.core.graph import AnnNet


def _graph_with_edge():
    G = AnnNet(directed=False)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e1')
    return G


def test_set_edge_attrs_raises_on_reserved_key():
    G = _graph_with_edge()
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_edge_attrs('e1', kind='complex')


def test_set_edge_attrs_bulk_raises_on_reserved_key():
    G = _graph_with_edge()
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_edge_attrs_bulk({'e1': {'members': ['x']}})


def test_set_vertex_attrs_bulk_raises_on_reserved_key():
    """The single-form for vertex/slice can't collide on the only reserved
    key (``vertex_id``/``slice_id``) — those collide with the positional
    parameter. The bulk form is the real surface for collisions."""
    G = AnnNet(directed=False)
    G.add_vertices(['A'])
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_vertex_attrs_bulk({'A': {'vertex_id': 'B'}})


def test_set_edge_slice_attrs_allows_weight_but_raises_on_others():
    G = _graph_with_edge()
    G.slices.add_slice('s1')
    # 'weight' is allowed even though it's reserved structurally
    G.attrs.set_edge_slice_attrs('s1', 'e1', weight=2.0)
    with pytest.raises(ValueError, match='reserved'):
        G.attrs.set_edge_slice_attrs('s1', 'e1', members=['x'])
