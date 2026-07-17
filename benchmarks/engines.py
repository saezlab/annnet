"""Engine adapters over identical graph input for fair backend comparison.

Build the *same* graph in AnnNet, NetworkX, igraph and graph-tool, and expose
the semantically-comparable operations behind one interface.

Fairness rules baked in here:

* All engines are fed an identical vertex list and ``(src, tgt, weight)``
  edge list from :func:`make_data`, so no engine gets an easier input.
* Query closures perform a single representative call (one ``degree``, one
  ``neighbors``, one ``has_edge``); the harness repeats them. This isolates
  per-call overhead — the thing that actually differs between a dict-of-dicts
  (NetworkX), C adjacency libraries (igraph / graph-tool) and an incidence
  matrix (AnnNet) — from the degree of any particular vertex (the ring input
  keeps degree constant across scale on purpose).
* Comparisons are only ever drawn between these adapters. AnnNet's unique
  capabilities live in ``workloads.py`` as an AnnNet-only section, never as a
  "win" against engines that cannot express them.

Imports of ``annnet`` are deferred into the closures so each worker process
imports the stack fresh in a clean interpreter.
"""

from __future__ import annotations

from collections.abc import Callable


# ---------------------------------------------------------------------------
# Shared, deterministic graph data
# ---------------------------------------------------------------------------
def make_data(n_vertices: int, n_edges: int) -> tuple[list[str], list[tuple[str, str, float]]]:
    """A deterministic ring-like directed graph shared by every engine.

    Constant per-vertex degree (~2) is intentional: it lets query benchmarks
    measure per-call overhead rather than the cost of a fat adjacency list.
    """
    names = [f'v{i}' for i in range(n_vertices)]
    edges = [(f'v{i % n_vertices}', f'v{(i + 1) % n_vertices}', 1.0) for i in range(n_edges)]
    return names, edges


# ---------------------------------------------------------------------------
# Engine interface
# ---------------------------------------------------------------------------
class Engine:
    """One benchmarkable graph backend.

    Subclasses provide a fresh-build factory and, given a built handle, a dict of
    no-argument query closures keyed by a canonical operation name so the report
    can line them up across engines.
    """

    name: str = '?'

    def available(self) -> bool:
        try:
            self._import()
            return True
        except Exception:
            return False

    def _import(self):  # pragma: no cover - trivial
        raise NotImplementedError

    def build_factory(self, data) -> Callable[[], object]:
        """Return a no-arg callable that constructs a fresh graph (for timing/memory)."""
        raise NotImplementedError

    def query_ops(self, handle, data) -> dict[str, Callable[[], object]]:
        """Return canonical-name -> repeatable read closure over a built handle."""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# AnnNet
# ---------------------------------------------------------------------------
class AnnNetEngine(Engine):
    def __init__(self, backend: str = 'auto', directed: bool = True):
        self.backend = backend
        self.directed = directed
        self.name = 'annnet'

    def _import(self):
        from annnet.core.graph import AnnNet

        return AnnNet

    def build_factory(self, data):
        names, edges = data
        AnnNet = self._import()
        backend, directed = self.backend, self.directed

        def build():
            G = AnnNet(directed=directed, annotations_backend=backend)
            G.add_vertices(({'vertex_id': v} for v in names), slice='base')
            G.add_edges({'source': u, 'target': v, 'weight': w} for (u, v, w) in edges)
            return G

        return build

    def query_ops(self, G, data):
        names, edges = data
        v_lo, v_hi = names[0], names[1 % len(names)]
        return {
            'degree': lambda: G.degree(v_lo),
            'neighbors': lambda: G.neighbors(v_lo),
            'has_edge': lambda: G.has_edge(v_lo, v_hi),
            'enumerate_edges': lambda: list(G.edges()),
        }


# ---------------------------------------------------------------------------
# NetworkX
# ---------------------------------------------------------------------------
class NetworkXEngine(Engine):
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.name = 'networkx'

    def _import(self):
        import networkx as nx

        return nx

    def build_factory(self, data):
        names, edges = data
        nx = self._import()
        directed = self.directed

        def build():
            G = nx.DiGraph() if directed else nx.Graph()
            G.add_nodes_from(names)
            G.add_weighted_edges_from(edges)
            return G

        return build

    def query_ops(self, G, data):
        names, edges = data
        v_lo, v_hi = names[0], names[1 % len(names)]
        return {
            'degree': lambda: G.degree(v_lo),
            'neighbors': lambda: list(G.neighbors(v_lo)),
            'has_edge': lambda: G.has_edge(v_lo, v_hi),
            'enumerate_edges': lambda: list(G.edges()),
        }


# ---------------------------------------------------------------------------
# igraph
# ---------------------------------------------------------------------------
class IGraphEngine(Engine):
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.name = 'igraph'

    def _import(self):
        import igraph as ig

        return ig

    def build_factory(self, data):
        names, edges = data
        ig = self._import()
        directed = self.directed
        idx = {n: i for i, n in enumerate(names)}
        edge_pairs = [(idx[u], idx[v]) for (u, v, _) in edges]
        weights = [w for (_, _, w) in edges]

        def build():
            g = ig.Graph(directed=directed)
            g.add_vertices(len(names))
            g.vs['name'] = names
            g.add_edges(edge_pairs)
            g.es['weight'] = weights
            return g

        return build

    def query_ops(self, g, data):
        names, edges = data
        # igraph queries are index-based; resolve the representative ids once.
        i_lo, i_hi = 0, 1 % len(names)
        return {
            'degree': lambda: g.degree(i_lo),
            'neighbors': lambda: g.neighbors(i_lo),
            'has_edge': lambda: g.are_connected(i_lo, i_hi),
            'enumerate_edges': lambda: g.get_edgelist(),
        }


# ---------------------------------------------------------------------------
# graph-tool
# ---------------------------------------------------------------------------
class GraphToolEngine(Engine):
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.name = 'graph-tool'

    def _import(self):
        import graph_tool.all as gt

        return gt

    def build_factory(self, data):
        names, edges = data
        gt = self._import()
        directed = self.directed
        idx = {n: i for i, n in enumerate(names)}
        edge_rows = [(idx[u], idx[v], w) for (u, v, w) in edges]

        def build():
            g = gt.Graph(directed=directed, fast_edge_lookup=True)
            g.add_vertex(len(names))
            weight = g.new_edge_property('double')
            if edge_rows:
                g.add_edge_list(edge_rows, eprops=[weight])
            g.ep['weight'] = weight
            return g

        return build

    def query_ops(self, g, data):
        names, edges = data
        i_lo, i_hi = 0, 1 % len(names)
        v_lo, v_hi = g.vertex(i_lo), g.vertex(i_hi)

        def degree():
            if g.is_directed():
                return v_lo.in_degree() + v_lo.out_degree()
            return v_lo.out_degree()

        def neighbors():
            if g.is_directed():
                return list(g.iter_out_neighbors(v_lo))
            return list(g.iter_all_neighbors(v_lo))

        return {
            'degree': degree,
            'neighbors': neighbors,
            'has_edge': lambda: g.edge(v_lo, v_hi) is not None,
            'enumerate_edges': lambda: list(g.iter_edges()),
        }


ENGINE_CLASSES = {
    'annnet': AnnNetEngine,
    'networkx': NetworkXEngine,
    'igraph': IGraphEngine,
    'graph-tool': GraphToolEngine,
}
BASELINE_ENGINE_NAMES = ('networkx', 'igraph', 'graph-tool')


def engine_by_name(name: str, **kwargs) -> Engine:
    return ENGINE_CLASSES[name](**kwargs)


def annnet_engine(backend: str = 'auto', directed: bool = True) -> AnnNetEngine:
    return AnnNetEngine(backend=backend, directed=directed)


def baseline_engines(directed: bool = True) -> list[Engine]:
    return [engine_by_name(name, directed=directed) for name in BASELINE_ENGINE_NAMES]
