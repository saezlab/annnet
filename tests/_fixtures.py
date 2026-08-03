"""The operation-matrix fixtures.

The operation matrix is the set of graph shapes that the core must hold. Every
story in the core refactor runs its checks over this same set, so a change that
breaks one shape shows up in one place.

Each builder returns a small graph that isolates one shape. Use
``OPERATION_MATRIX`` to look a builder up by name, and ``CASE_NAMES`` to
parametrize a test over the whole set.

The graphs carry no biological meaning. The names are plain letters.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

from annnet.core.graph import AnnNet


# Which canonical store the builders give their graphs. ``build_case`` sets it,
# so one set of builders serves both stores and the shapes cannot drift apart.
_STORE = 'records'


@contextmanager
def _store_mode(store: str):
    global _STORE
    previous, _STORE = _STORE, store
    try:
        yield
    finally:
        _STORE = previous


def _quiet(build):
    """Run a builder with the placeholder-layer warnings suppressed."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return build()


# ---------------------------------------------------------------------------
# Binary edges
# ---------------------------------------------------------------------------


def binary_directed() -> AnnNet:
    """Three nodes in a directed chain."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e_ab', weight=1.5)
    G.add_edges('B', 'C', edge_id='e_bc', weight=2.0)
    return G


def binary_undirected() -> AnnNet:
    """Three nodes joined by undirected edges."""
    G = AnnNet(store=_STORE, directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e_ab', weight=1.0)
    G.add_edges('B', 'C', edge_id='e_bc', weight=3.0)
    return G


def self_loop() -> AnnNet:
    """One node with an edge to itself, next to a plain edge."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'A', edge_id='e_loop', weight=0.5)
    G.add_edges('A', 'B', edge_id='e_ab')
    return G


def parallel_edge() -> AnnNet:
    """Two separate edges over the same pair of endpoints."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e_first', weight=1.0)
    G.add_edges('A', 'B', edge_id='e_second', weight=2.0, parallel='parallel')
    return G


# ---------------------------------------------------------------------------
# Hyperedges
# ---------------------------------------------------------------------------


def hyper_undirected() -> AnnNet:
    """One hyperedge over three nodes, with no direction."""
    G = AnnNet(store=_STORE, directed=False)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'h_abc'}])
    return G


def hyper_directed() -> AnnNet:
    """One hyperedge that runs from two source nodes to one target node.

    The source side carries the positive coefficient, exactly as it does for a
    binary edge.
    """
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'source': ['A', 'B'], 'target': ['C'], 'edge_id': 'h_ab_c'}])
    return G


# ---------------------------------------------------------------------------
# Edge-entity and boundary edge
# ---------------------------------------------------------------------------


def edge_entity() -> AnnNet:
    """One edge that is also a node, and one edge that points at it."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='ee_ab', as_entity=True)
    G.add_edges('ee_ab', 'C', edge_id='e_meta')
    return G


def placeholder_edge() -> AnnNet:
    """An edge entity declared before the edge it stands for exists.

    The placeholder holds no members and occupies no column, so it carries no
    structure. It is the one shape that separates "the graph knows this edge id"
    from "this edge is part of the topology".
    """
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges('A', 'B', edge_id='e_ab')
    G._ensure_edge_entity_placeholder('e_later')
    return G


def boundary_edge() -> AnnNet:
    """Two one-sided edges. One drains a node and one feeds it.

    A boundary edge holds a single member. The sign of the coefficient carries
    the direction, so the graph needs no placeholder node on the open side.
    """
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B'])
    G.add_edges([{'members': ['A'], 'edge_id': 'b_out'}])
    G.add_edges([{'members': ['A'], 'edge_id': 'b_in'}])
    G.add_edges('A', 'B', edge_id='e_ab')
    G.set_edge_coeffs('b_out', {'A': -1.0})
    G.set_edge_coeffs('b_in', {'A': 1.0})
    return G


def coefficient_edge() -> AnnNet:
    """One hyperedge whose members carry coefficients other than plus or minus one."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges([{'members': ['A', 'B', 'C'], 'edge_id': 'r_1'}])
    G.set_edge_coeffs('r_1', {'A': -2.0, 'B': -1.0, 'C': 3.0})
    return G


# ---------------------------------------------------------------------------
# Multilayer and slices
# ---------------------------------------------------------------------------


def multilayer() -> AnnNet:
    """Two nodes in two layers, with an intra-layer edge and a coupling edge."""

    def build():
        G = AnnNet(store=_STORE, directed=True)
        G.layers.set_aspects(['phase'], {'phase': ['t0', 't1']})
        G.add_vertices(['A', 'B'], layer=('t0',))
        G.add_vertices(['A', 'B'], layer=('t1',))
        G.add_edges(
            [
                {'source': ('A', ('t0',)), 'target': ('B', ('t0',)), 'edge_id': 'e_t0'},
                {'source': ('A', ('t1',)), 'target': ('B', ('t1',)), 'edge_id': 'e_t1'},
                {'source': ('A', ('t0',)), 'target': ('A', ('t1',)), 'edge_id': 'e_couple'},
            ]
        )
        return G

    return _quiet(build)


def sliced() -> AnnNet:
    """One graph whose edges split over two named slices."""
    G = AnnNet(store=_STORE, directed=True)
    G.add_vertices(['A', 'B', 'C'])
    G.add_edges('A', 'B', edge_id='e_ab')
    G.add_edges('B', 'C', edge_id='e_bc')
    G.slices.add('left')
    G.slices.add('right')
    G.slices.add_edge_to_slice('left', 'e_ab')
    G.slices.add_edge_to_slice('right', 'e_bc')
    return G


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

OPERATION_MATRIX = {
    'binary_directed': binary_directed,
    'binary_undirected': binary_undirected,
    'self_loop': self_loop,
    'parallel_edge': parallel_edge,
    'hyper_undirected': hyper_undirected,
    'hyper_directed': hyper_directed,
    'edge_entity': edge_entity,
    'boundary_edge': boundary_edge,
    'placeholder_edge': placeholder_edge,
    'coefficient_edge': coefficient_edge,
    'multilayer': multilayer,
    'sliced': sliced,
}

CASE_NAMES = tuple(OPERATION_MATRIX)


def build_case(name: str, store: str = 'records') -> AnnNet:
    """Build one operation-matrix graph by name, on either canonical store."""
    try:
        builder = OPERATION_MATRIX[name]
    except KeyError:
        raise KeyError(
            f'Unknown operation-matrix case {name!r}. Known: {list(CASE_NAMES)}'
        ) from None
    with _store_mode(store):
        return builder()


def all_cases():
    """Yield every operation-matrix case as a ``(name, graph)`` pair."""
    for name in CASE_NAMES:
        yield name, build_case(name)
