from __future__ import annotations

import io
import math
from typing import Any, Literal
import contextlib
from collections.abc import Iterable

import numpy as np

from .._support.plotting_backend import select_plot_backend

# The plotting helpers read a graph through its public surface alone —
# ``G.N``, ``G.E.at``, ``G.E.effective_weight`` and ``G.attrs[address, key]`` —
# so a duck-typed stand-in with the same four names can be drawn too.


def _node_attr_getter(graph):
    def read(node_id, key, default=None):
        value = graph.attrs.row('nodes', node_id).get(key)
        return default if value is None else value

    return read


def _edge_attr_getter(graph):
    def read(edge_id, key, default=None):
        value = graph.attrs.row('edges', edge_id).get(key)
        return default if value is None else value

    return read


def _edge_weight_getter(graph):
    return graph.E.effective_weight


# Small helpers


def _normalize(
    values: Iterable[float], lo: float | None = None, hi: float | None = None, eps: float = 1e-12
):
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    if lo is None:
        lo = np.nanmin(arr)
    if hi is None:
        hi = np.nanmax(arr)
    if not math.isfinite(lo):
        lo = 0.0
    if not math.isfinite(hi):
        hi = 1.0
    denom = max(hi - lo, eps)
    return (arr - lo) / denom


def _greyscale(v: float) -> str:
    v = float(np.clip(v, 0.0, 1.0))
    c = int(round(v * 255))
    return f'#{c:02x}{c:02x}{c:02x}'


def _suppress_repr_warnings(g: Any) -> None:
    """Monkey-patch _repr_* methods to hide stderr noise from visualization libs."""
    repr_methods = [m for m in dir(g) if m.startswith('_repr_') and callable(getattr(g, m))]
    for method_name in repr_methods:
        original = getattr(g, method_name)

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                with contextlib.redirect_stderr(io.StringIO()):
                    return orig_func(*args, **kwargs)

            return wrapper

        setattr(g, method_name, make_wrapper(original))


# Label builders


def build_node_labels(graph, key: str | None = None) -> dict[str, str]:
    """Build display labels for graph nodes."""
    labels: dict[str, str] = {}
    for vid in list(graph.N):
        if key is None:
            labels[vid] = str(vid)
        else:
            labels[vid] = str(_node_attr_getter(graph)(vid, key, default=vid))
    return labels


def _edge_ids_in_order(graph) -> list[str]:
    """Return the edge ids in the order the graph holds them.

    A drawing indexes its edges by position, because a layout does. The graph
    answers by id, so this is the one place the two meet.
    """
    return list(graph.E)


def build_edge_labels(
    graph,
    *,
    use_weight: bool = True,
    extra_keys: list[str] | None = None,
    layer: str | None = None,
) -> dict[int, str]:
    """Build display labels for graph edges."""
    extra_keys = extra_keys or []
    labels: dict[int, str] = {}
    for j, eid in enumerate(_edge_ids_in_order(graph)):
        parts: list[str] = []
        if use_weight:
            try:
                w = _edge_weight_getter(graph)(eid, slice=layer)
                parts.append(f'{w:.3g}')
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
        for k in extra_keys:
            v = _edge_attr_getter(graph)(eid, k, default=None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                parts.append(f'{k}={v}')
        if parts:
            labels[j] = '\\n'.join(parts)
    return labels


# Edge style from weights


def edge_style_from_weights(
    graph,
    *,
    layer: str | None = None,
    min_width: float = 1.15,
    max_width: float = 4.25,
    color_mode: Literal['greys', 'signed'] = 'greys',
) -> dict[int, dict[str, str]]:
    """Compute visual edge styles (pen width and color) from effective weights.

    Parameters
    ----------
    graph : object
        AnnNet-like object exposing `edges()` and
        `get_effective_edge_weight(eid, layer)` methods.
    layer : str, optional
        Layer name for retrieving edge weights. Defaults to `None`, which uses global weights.
    min_width : float, optional
        Minimum line width for edges. Default is 0.5.
    max_width : float, optional
        Maximum line width for edges. Default is 5.0.
    color_mode : {'greys', 'signed'}, optional
        Edge coloring mode:
        - ``'greys'`` : map absolute weight to grayscale (darker = heavier)
        - ``'signed'`` : use red for positive, blue for negative, black for zero

    Returns
    -------
    dict[int, dict[str, str]]
        A mapping from edge index to a style dict with keys:
        - ``penwidth`` : stroke width (stringified float)
        - ``color`` : color name or hex code

    Notes
    -----
    - Invalid or missing weights default to 1.0.
    - Normalization is performed across all edges in the graph.

    """
    eids = _edge_ids_in_order(graph)
    eidxs = list(range(len(eids)))
    raw_vals: list[float] = []
    for eid in eids:
        try:
            raw_vals.append(abs(float(_edge_weight_getter(graph)(eid, slice=layer))))
        except (AttributeError, KeyError, TypeError, ValueError):
            raw_vals.append(1.0)

    x = _normalize(raw_vals)
    if np.all(x == x[0]):
        x = np.full_like(x, 0.5)
    styles: dict[int, dict[str, str]] = {}
    for j, xv in zip(eidxs, x, strict=False):
        pen = min_width + float(xv) * (max_width - min_width)
        if color_mode == 'signed':
            eid = eids[j]
            try:
                w = float(_edge_weight_getter(graph)(eid, slice=layer))
            except (AttributeError, KeyError, TypeError, ValueError):
                w = 0.0
            color = 'firebrick4' if w > 0 else ('dodgerblue4' if w < 0 else 'black')
        else:
            color = _greyscale(0.42 - 0.32 * float(xv))  # heavier => darker
        styles[j] = {'penwidth': f'{pen:.3f}', 'color': color}
    return styles


# Backends


def _add_nodes_graphviz(
    Gv, node_names: Iterable[str], custom_node_attr: dict[str, dict[str, str]] | None = None
):
    custom_node_attr = custom_node_attr or {}
    for v in node_names:
        # An ellipse grows with its label, so long names stay inside the node
        attrs = {'shape': 'ellipse'}
        attrs.update(custom_node_attr.get(v, {}))
        Gv.node(v, **attrs)


def _add_nodes_pydot(
    Gd, node_names: Iterable[str], custom_node_attr: dict[str, dict[str, str]] | None = None
):
    import pydot

    custom_node_attr = custom_node_attr or {}
    for v in node_names:
        # An ellipse grows with its label, so long names stay inside the node
        attrs = {'shape': 'ellipse'}
        attrs.update(custom_node_attr.get(v, {}))
        Gd.add_node(pydot.Node(v, **attrs))


def _is_true_hyperedge(S: frozenset, T: frozenset) -> bool:
    """Return True only for genuine hyperedges, not undirected binary edges.

    Undirected binary edges are represented as (frozenset({u,v}), frozenset({u,v}))
    and must NOT be treated as hyperedges.
    """
    if len(S) <= 1 and len(T) <= 1:
        return False
    # Undirected binary edge: S == T and exactly 2 members
    if S == T and len(S) == 2:
        return False
    return True


def to_graphviz(
    graph,
    *,
    layout: str = 'dot',
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_node_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
):
    """Convert a graph to a Graphviz Digraph."""
    import graphviz

    Gv = graphviz.Digraph(
        engine=layout, graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
    )

    # nodes to materialize (union of all endpoints)
    all_nodes: set[str] = set()
    eids = _edge_ids_in_order(graph)
    edges_iter = range(len(eids)) if edge_indexes is None else edge_indexes

    # First pass: collect nodes
    for j in edges_iter:
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_graphviz(Gv, sorted(all_nodes), custom_node_attr)

    # Second pass: add edges
    for j in range(len(eids)):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        # pick styling overrides
        e_attr = {}
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        if _is_true_hyperedge(S, T):
            center = f'e_{j}_center'
            Gv.node(center, shape='square', width='0.1', height='0.1', fixedsize='true', label='')
            label_attr = {}
            hyper_label = e_attr.pop('label', None)
            if hyper_label is not None:
                label_attr = {'label': hyper_label}
                for key in ('fontcolor', 'fontsize', 'fontname'):
                    if key in e_attr:
                        label_attr[key] = e_attr[key]
            labeled = False
            for u in sorted(S, key=str):
                a = {'arrowtail': 'none', 'arrowhead': 'none', 'dir': 'both'}
                a.update(e_attr)
                if not T and label_attr and not labeled:
                    a.update(label_attr)
                    labeled = True
                Gv.edge(str(u), center, **a)
            for v in sorted(T, key=str):
                a = dict(e_attr)
                if label_attr and not labeled:
                    a.update(label_attr)
                    labeled = True
                Gv.edge(center, str(v), **a)
        else:
            # binary edge (directed or undirected)
            if S == T:
                # undirected: S == T; extract the two distinct endpoints (or self-loop)
                nodes = list(S)
                uu, vv = (nodes[0], nodes[1]) if len(nodes) == 2 else (nodes[0], nodes[0])
                a = {'arrowhead': 'none', 'dir': 'none'}
                a.update(e_attr)
                Gv.edge(str(uu), str(vv), **a)
            else:
                # directed binary edge
                u = next(iter(S))
                v = next(iter(T))
                head = 'normal'
                inter = _edge_attr_getter(graph)(eids[j], 'interaction', default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = 'tee'
                a = {'arrowhead': head}
                a.update(e_attr)
                Gv.edge(str(u), str(v), **a)

    if suppress_warnings:
        _suppress_repr_warnings(Gv)
    if (
        any(_is_true_hyperedge(*graph.E.at(eids[j])) for j in range(len(eids)))
        and graph_attr is None
    ):
        Gv.graph_attr['splines'] = 'true'
    return Gv


def to_pydot(
    graph,
    *,
    layout: str = 'dot',  # kept for API parity; pydot doesn't use engine here
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_node_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
):
    """Convert a graph to a pydot Dot graph."""
    import pydot

    Gd = pydot.Dot(graph_type='digraph', **(graph_attr or {}))
    if node_attr:
        Gd.set_node_defaults(**node_attr)
    if edge_attr:
        Gd.set_edge_defaults(**edge_attr)

    all_nodes: set[str] = set()
    eids = _edge_ids_in_order(graph)
    edges_iter = range(len(eids)) if edge_indexes is None else edge_indexes
    for j in edges_iter:
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_pydot(Gd, sorted(all_nodes), custom_node_attr)

    for j in range(len(eids)):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        e_attr = {}
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        if _is_true_hyperedge(S, T):
            center = f'e_{j}_center'
            Gd.add_node(
                pydot.Node(
                    center, shape='square', width='0.1', height='0.1', fixedsize='true', label=''
                )
            )
            label_attr = {}
            hyper_label = e_attr.pop('label', None)
            if hyper_label is not None:
                label_attr = {'label': hyper_label}
                for key in ('fontcolor', 'fontsize', 'fontname'):
                    if key in e_attr:
                        label_attr[key] = e_attr[key]
            labeled = False
            for u in sorted(S, key=str):
                a = {'arrowtail': 'none', 'arrowhead': 'none', 'dir': 'both'}
                a.update(e_attr)
                if not T and label_attr and not labeled:
                    a.update(label_attr)
                    labeled = True
                Gd.add_edge(pydot.Edge(str(u), center, **a))
            for v in sorted(T, key=str):
                a = dict(e_attr)
                if label_attr and not labeled:
                    a.update(label_attr)
                    labeled = True
                Gd.add_edge(pydot.Edge(center, str(v), **a))
        else:
            # binary edge (directed or undirected)
            if S == T:
                # undirected: extract the two distinct endpoints (or self-loop)
                nodes = list(S)
                uu, vv = (nodes[0], nodes[1]) if len(nodes) == 2 else (nodes[0], nodes[0])
                a = {'arrowhead': 'none', 'dir': 'none'}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(uu), str(vv), **a))
            else:
                u = next(iter(S))
                v = next(iter(T))
                head = 'normal'
                inter = _edge_attr_getter(graph)(eids[j], 'interaction', default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = 'tee'
                a = {'arrowhead': head}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(u), str(v), **a))

    if (
        any(_is_true_hyperedge(*graph.E.at(eids[j])) for j in range(len(eids)))
        and graph_attr is None
    ):
        Gd.set_splines('true')
    return Gd


def to_matplotlib(
    graph,
    *,
    ax=None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
    show_node_labels: bool = True,
    node_label_key: str | None = None,
    show_edge_labels: bool = False,
    edge_label_keys: list[str] | None = None,
    layer: str | None = None,
    node_size: float = 900.0,
    node_color: str = '#f5f5f5',
    edge_color: str = '#333333',
    hyperedge_color: str = '#777777',
):
    """Draw an AnnNet graph with matplotlib and return ``(figure, axes)``.

    This minimal fallback renderer does not require Graphviz, pydot, or
    NetworkX. It places nodes on a circle, draws binary directed edges as
    arrows, undirected binary edges as plain segments, and hyperedges via a
    small center marker connected to incident nodes.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    eids = _edge_ids_in_order(graph)
    edges = list(range(len(eids))) if edge_indexes is None else list(edge_indexes)
    nodes: set[str] = set(map(str, list(graph.N)))
    for j in edges:
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        nodes.update(map(str, S | T))

    ordered_nodes = sorted(nodes)
    n = max(1, len(ordered_nodes))
    positions = {
        node: (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
        for i, node in enumerate(ordered_nodes)
    }

    if ordered_nodes:
        xs = [positions[v][0] for v in ordered_nodes]
        ys = [positions[v][1] for v in ordered_nodes]
        ax.scatter(
            xs,
            ys,
            s=node_size,
            c=node_color,
            edgecolors='#222222',
            linewidths=1.0,
            zorder=3,
        )

    if show_node_labels:
        labels = build_node_labels(graph, key=node_label_key)
        for node in ordered_nodes:
            x, y = positions[node]
            ax.text(x, y, str(labels.get(node, node)), ha='center', va='center', zorder=4)

    edge_labels = (
        build_edge_labels(graph, use_weight=True, extra_keys=edge_label_keys, layer=layer)
        if show_edge_labels
        else {}
    )
    for j in edges:
        S, T = graph.E.at(eids[j])
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        if _is_true_hyperedge(S, T):
            center = (0.0, 0.0)
            incident = sorted(map(str, S | T))
            if incident:
                center = (
                    float(np.mean([positions[v][0] for v in incident])),
                    float(np.mean([positions[v][1] for v in incident])),
                )
            ax.scatter([center[0]], [center[1]], s=80.0, c=hyperedge_color, marker='s', zorder=2)
            for node in incident:
                x, y = positions[node]
                ax.plot([center[0], x], [center[1], y], color=hyperedge_color, linewidth=1.0)
            if j in edge_labels:
                ax.text(
                    center[0],
                    center[1],
                    edge_labels[j],
                    ha='left',
                    va='bottom',
                    fontsize=7,
                    color='#777777',
                )
            continue

        if S == T:
            nodes = sorted(map(str, S))
            if len(nodes) == 1:
                u = v = nodes[0]
            elif len(nodes) >= 2:
                u, v = nodes[0], nodes[1]
            else:
                continue
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.plot([x0, x1], [y0, y1], color=edge_color, linewidth=1.2, zorder=1)
        else:
            if not S or not T:
                continue
            u = str(next(iter(S)))
            v = str(next(iter(T)))
            x0, y0 = positions[u]
            x1, y1 = positions[v]
            ax.annotate(
                '',
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={'arrowstyle': '->', 'color': edge_color, 'lw': 1.2},
                zorder=1,
            )

        if j in edge_labels:
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                edge_labels[j],
                fontsize=7,
                color='#777777',
            )

    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


# One-call plotting API


def plot(
    graph,
    *,
    backend: Literal['auto', 'graphviz', 'pydot', 'matplotlib'] | None = None,
    layout: str = 'dot',
    layer: str | None = None,
    show_edge_labels: bool = False,
    edge_label_keys: list[str] | None = None,
    show_node_labels: bool = True,
    node_label_key: str | None = None,
    use_weight_style: bool = True,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
    **kwargs,
):
    """Build a styled graph object to display or to write with :func:`save`.

    Parameters
    ----------
    graph : AnnNet or GraphView
        The graph or view to draw.
    backend : {'auto', 'graphviz', 'pydot', 'matplotlib'} or None, optional
        Visualization backend to use. ``None`` uses AnnNet's configured
        plotting default. ``'auto'`` prefers Graphviz, then pydot, then
        matplotlib.
    layout : str, optional
        Layout engine (e.g. ``'dot'``, ``'neato'``). Default is ``'dot'``.
    layer : str, optional
        Layer name for weight or label extraction. Default is `None`.
    show_edge_labels : bool, optional
        Whether to include weight and attribute labels on edges. Default is ``False``.
    edge_label_keys : list of str, optional
        Extra edge attribute keys to display if `show_edge_labels=True`.
    show_node_labels : bool, optional
        Whether to label nodes with IDs or attributes. Default is ``True``.
    node_label_key : str, optional
        Attribute key for node labels. If `None`, uses node IDs.
    use_weight_style : bool, optional
        Whether to style edges based on weights. Default is ``True``.
    orphan_edges : bool, optional
        Whether to render edges with missing endpoints. Default is ``True``.
    suppress_warnings : bool, optional
        Suppress backend rendering warnings (stderr). Default is ``True``.
    **kwargs
        Additional keyword arguments forwarded to `to_graphviz()` or `to_pydot()`.

    Returns
    -------
    graphviz.Digraph, pydot.Dot, or tuple
        A styled, backend-specific graph object. Matplotlib returns
        ``(figure, axes)``.

    Raises
    ------
    ValueError
        If an invalid `backend` name is provided.

    Notes
    -----
    - Edge styling and labels are applied before backend construction.
    - If `show_edge_labels=True`, edges are regenerated with label overrides.

    """
    backend = select_plot_backend(backend)

    # edge styles
    custom_edge_attr: dict[int, dict[str, str]] = {}
    if use_weight_style:
        custom_edge_attr = edge_style_from_weights(graph, layer=layer)

    # node labels (set via custom_node_attr)
    custom_node_attr: dict[str, dict[str, str]] | None = None
    if show_node_labels:
        vlabels = build_node_labels(graph, key=node_label_key)
        custom_node_attr = {k: {'label': v} for k, v in vlabels.items()}

    if backend == 'graphviz':
        G = to_graphviz(
            graph,
            layout=layout,
            graph_attr=kwargs.get('graph_attr'),
            node_attr=kwargs.get('node_attr'),
            edge_attr=kwargs.get('edge_attr'),
            custom_edge_attr=custom_edge_attr,
            custom_node_attr=custom_node_attr,
            edge_indexes=kwargs.get('edge_indexes'),
            orphan_edges=orphan_edges,
            suppress_warnings=suppress_warnings,
        )
        if show_edge_labels:
            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            # reapply labels by regenerating with label overrides
            for j, txt in elabels.items():
                custom_edge_attr.setdefault(j, {})['label'] = txt
                custom_edge_attr[j].setdefault('fontcolor', '#8a8a8a')
                custom_edge_attr[j].setdefault('fontsize', '8')
            G = to_graphviz(
                graph,
                layout=layout,
                graph_attr=kwargs.get('graph_attr'),
                node_attr=kwargs.get('node_attr'),
                edge_attr=kwargs.get('edge_attr'),
                custom_edge_attr=custom_edge_attr,
                custom_node_attr=custom_node_attr,
                edge_indexes=kwargs.get('edge_indexes'),
                orphan_edges=orphan_edges,
                suppress_warnings=suppress_warnings,
            )
        return G

    elif backend == 'pydot':
        G = to_pydot(
            graph,
            layout=layout,
            graph_attr=kwargs.get('graph_attr'),
            node_attr=kwargs.get('node_attr'),
            edge_attr=kwargs.get('edge_attr'),
            custom_edge_attr=custom_edge_attr,
            custom_node_attr=custom_node_attr,
            edge_indexes=kwargs.get('edge_indexes'),
            orphan_edges=orphan_edges,
        )
        if show_edge_labels:
            # mutate by adding parallel labeled edges
            import pydot

            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            eids = _edge_ids_in_order(graph)
            for j, txt in elabels.items():
                S, T = graph.E.at(eids[j])
                sv = next(iter(S)) if len(S) else f'e_{j}_source'
                tv = next(iter(T)) if len(T) else f'e_{j}_target'
                G.add_edge(
                    pydot.Edge(str(sv), str(tv), label=txt, fontcolor='#8a8a8a', fontsize='8')
                )
        return G

    elif backend == 'matplotlib':
        return to_matplotlib(
            graph,
            ax=kwargs.get('ax'),
            edge_indexes=kwargs.get('edge_indexes'),
            orphan_edges=orphan_edges,
            show_node_labels=show_node_labels,
            node_label_key=node_label_key,
            show_edge_labels=show_edge_labels,
            edge_label_keys=edge_label_keys,
            layer=layer,
            node_size=kwargs.get('node_size', 900.0),
            node_color=kwargs.get('node_color', '#f5f5f5'),
            edge_color=kwargs.get('edge_color', '#333333'),
            hyperedge_color=kwargs.get('hyperedge_color', '#777777'),
        )

    else:
        raise ValueError("backend must be 'auto', 'graphviz', 'pydot', or 'matplotlib'")


# Write to a file


def save(obj: Any, path: str, format: str = 'svg') -> str:
    """Write a Graphviz, pydot or matplotlib graph object to a file.

    Parameters
    ----------
    obj : graphviz.Digraph, pydot.Dot, matplotlib Figure/Axes, or (Figure, Axes)
        The graph object returned by `plot()`.
    path : str
        Destination file path (without extension if `format` is specified).
    format : {'svg', 'png', 'raw'}, optional
        Output format. Default is ``'svg'``. ``'png'`` and ``'raw'`` are supported for Pydot.

    Returns
    -------
    str
        Full path to the written output file.

    Raises
    ------
    TypeError
        If `obj` is not a supported graph type.

    Notes
    -----
    - Graphviz objects use the built-in `.render()` API.
    - Pydot objects write directly via `.write_svg()`, `.write_png()`, or `.write_raw()`.
    - Matplotlib figures use `.savefig()`.
    - The file extension is appended automatically if not present.

    """
    kind = obj.__class__.__module__
    fmt = format.lower()
    if isinstance(obj, tuple) and obj:
        obj = obj[0]
        kind = obj.__class__.__module__
    if 'graphviz' in kind:
        return obj.render(path, format=fmt, cleanup=True)
    elif 'pydot' in kind:
        if fmt == 'png':
            obj.write_png(path if path.lower().endswith('.png') else f'{path}.png')
            return path if path.lower().endswith('.png') else f'{path}.png'
        elif fmt in ('svg',):
            obj.write_svg(path if path.lower().endswith('.svg') else f'{path}.svg')
            return path if path.lower().endswith('.svg') else f'{path}.svg'
        else:
            obj.write_raw(path)
            return path
    elif 'matplotlib' in kind:
        fig = obj.figure if hasattr(obj, 'figure') and not hasattr(obj, 'savefig') else obj
        out = path if path.lower().endswith(f'.{fmt}') else f'{path}.{fmt}'
        fig.savefig(out, format=fmt, bbox_inches='tight')
        return out
    else:
        raise TypeError(
            'Unknown graph object; expected graphviz.Digraph, pydot.Dot, or matplotlib Figure/Axes'
        )


# Interactive view with Cytoscape.js


def show_cytoscape(  # pragma: no cover  — local web server, not covered in unit tests
    graph,
    *,
    export_name='annnet export',
    layer=None,
    include_inter=False,
    include_coupling=False,
    hyperedges='skip',
    port=None,
    auto_open=True,
    inline=False,
    height='520px',
):
    """Show the graph as an interactive Cytoscape.js network.

    The graph goes through :func:`annnet.io.to_cx2` first, so the layer and
    hyperedge options are those of the CX2 export. In a Jupyter notebook,
    ``inline=True`` shows the network in the cell output. Otherwise the
    function starts a local web server and opens the view in the browser.

    Parameters
    ----------
    graph : AnnNet
        The graph to show.
    export_name : str, default "annnet export"
        Name of the network display.
    layer : tuple of str, optional
        Elementary layer tuple specifying which layer to visualize. If None,
        shows the entire graph (may include coupling edges as self-loops).
        Example: layer=("social", "2020") for a 2-aspect multilayer network.
    include_inter : bool, default False
        When layer is specified, whether to include interlayer edges (edges
        between nodes in different layers but same aspect). Only relevant if
        layer is not None.
    include_coupling : bool, default False
        When layer is specified, whether to include coupling edges (edges
        connecting the same node across layers). Only relevant if layer is not None.
        Warning: Including coupling creates self-loops in the visualization.
    hyperedges : {"skip", "expand", "reify"}, default "skip"
        How to handle hyperedges:
        - "skip": Omit hyperedges entirely
        - "expand": Convert to cartesian product of pairwise edges
        - "reify": Create explicit hyperedge nodes with membership edges
    port : int, optional
        Port for local web server. If None, finds available port automatically.
    auto_open : bool, default True
        If True, automatically open browser.
    inline : bool, default False
        If True, return the view as notebook output instead of starting a
        web server.
    height : str, default "520px"
        Height of the inline view.

    Returns
    -------
    str or IPython.display.HTML
        The local URL of the view, or the inline view if ``inline=True``.

    Notes
    -----
    For multilayer graphs:
    - Without layer parameter: shows entire flattened graph with coupling edges
      appearing as self-loops (messy but shows full structure)
    - With layer parameter: shows clean single-layer view without coupling edges
      (recommended for visualization)

    Press Ctrl+C in terminal to stop the web server.

    Examples
    --------
    >>> show_cytoscape(anet, inline=True)  # in a notebook
    >>> show_cytoscape(anet)  # all layers flattened, in the browser
    >>> show_cytoscape(anet, layer=('social', '2020'))  # one layer
    """
    import json
    import socket
    from pathlib import Path
    import tempfile
    import threading
    import webbrowser
    from http.server import HTTPServer, SimpleHTTPRequestHandler

    def find_free_port():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port

    if port is None:
        port = find_free_port()

    from annnet.io.cx2 import to_cx2

    # to_cx2 already handles layer extraction internally
    cx2_data = to_cx2(
        graph,
        export_name=export_name,
        layer=layer,
        include_inter=include_inter,
        include_coupling=include_coupling,
        hyperedges=hyperedges,
    )

    cytoscape_json = _cx2_to_cytoscapejs(cx2_data)

    if inline:
        # Render directly in the notebook cell; no server, no browser.
        from IPython.display import HTML

        # Canonical Cytoscape.js element form is a flat array; nodes first, then
        # edges (so an edge never references a not-yet-registered node).
        elements = list(cytoscape_json.get('nodes', [])) + list(cytoscape_json.get('edges', []))
        cid = f'cy_{abs(hash((export_name, len(elements)))) % 10**8}'
        inline_html = f"""
<div id="{cid}" style="width:100%;height:{height};border:1px solid #ddd;position:relative;"></div>
<div style="font:12px sans-serif;color:#555;margin-top:4px">{export_name} —
{len(cytoscape_json.get('nodes', []))} nodes, {len(cytoscape_json.get('edges', []))} edges</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
<script>
(function() {{
  function draw() {{
    cytoscape({{
      container: document.getElementById("{cid}"),
      elements: {json.dumps(elements)},
      style: [
        {{selector: 'node', style: {{'background-color': '#0074D9', 'label': 'data(label)',
          'width': 22, 'height': 22, 'font-size': '9px', 'color': '#222',
          'text-outline-width': 2, 'text-outline-color': '#fff'}}}},
        {{selector: 'node[?is_hyperedge]', style: {{'background-color': '#ff851b', 'shape': 'diamond'}}}},
        {{selector: 'edge', style: {{'width': 2.5, 'line-color': '#555',
          'target-arrow-color': '#555', 'target-arrow-shape': 'triangle',
          'arrow-scale': 0.9, 'curve-style': 'bezier', 'opacity': 1}}}}
      ],
      layout: {{name: 'cose', animate: false, numIter: 800, nodeRepulsion: 8000, idealEdgeLength: 70}}
    }});
  }}
  if (window.cytoscape) {{ draw(); }}
  else {{ var iv = setInterval(function() {{ if (window.cytoscape) {{ clearInterval(iv); draw(); }} }}, 100); }}
}})();
</script>"""
        return HTML(inline_html)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{export_name}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.26.0/cytoscape.min.js"></script>
        <style>
            body {{ margin: 0; padding: 0; font-family: sans-serif; }}
            #cy {{ width: 100vw; height: 100vh; }}
            #info {{
                position: absolute;
                top: 10px;
                left: 10px;
                background: rgba(255,255,255,0.9);
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
            }}
        </style>
    </head>
    <body>
        <div id="info">
            <strong>{export_name}</strong><br>
            Nodes: <span id="nodeCount">0</span> | Edges: <span id="edgeCount">0</span>
        </div>
        <div id="cy"></div>
        <script>
            var cy = cytoscape({{
                container: document.getElementById('cy'),
                elements: {json.dumps(cytoscape_json)},
                style: [
                    {{
                        selector: 'node',
                        style: {{
                            'background-color': '#0074D9',
                            'label': 'data(label)',
                            'width': 40,
                            'height': 40,
                            'font-size': '12px',
                            'text-valign': 'center',
                            'text-halign': 'center',
                            'color': '#000',
                            'text-outline-width': 2,
                            'text-outline-color': '#fff'
                        }}
                    }},
                    {{
                        selector: 'edge',
                        style: {{
                            'width': 2,
                            'line-color': '#ccc',
                            'target-arrow-color': '#ccc',
                            'target-arrow-shape': 'triangle',
                            'curve-style': 'bezier',
                            'opacity': 0.6
                        }}
                    }},
                    {{
                        selector: 'edge[source = target]',
                        style: {{
                            'curve-style': 'loop',
                            'loop-direction': '0deg',
                            'loop-sweep': '45deg',
                            'line-color': '#ff6b6b',
                            'target-arrow-color': '#ff6b6b',
                            'width': 1.5
                        }}
                    }}
                ],
                layout: {{
                    name: 'cose',
                    animate: true,
                    nodeRepulsion: 8000,
                    idealEdgeLength: 100,
                    numIter: 1000
                }}
            }});

            document.getElementById('nodeCount').textContent = cy.nodes().length;
            document.getElementById('edgeCount').textContent = cy.edges().length;
        </script>
    </body>
    </html>
    """

    temp_dir = tempfile.mkdtemp()
    html_path = Path(temp_dir) / 'index.html'
    html_path.write_text(html_content)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=temp_dir, **kwargs)

        def log_message(self, format, *args):
            pass

    try:
        server = HTTPServer(('localhost', port), Handler)
    except OSError as e:
        if e.errno == 98:
            port = find_free_port()
            server = HTTPServer(('localhost', port), Handler)
        else:
            raise

    url = f'http://localhost:{port}'

    def run_server():
        print(f'Serving visualization at {url}')
        print('Press Ctrl+C to stop')
        server.serve_forever()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    if auto_open:
        webbrowser.open(url)

    return url


def _cx2_to_cytoscapejs(cx2_data: list[dict]) -> dict:
    """Convert CX2 format to Cytoscape.js format."""
    nodes = []
    edges = []

    node_map = {}

    for aspect in cx2_data:
        if 'nodes' in aspect:
            for node in aspect['nodes']:
                node_id = str(node['id'])
                node_map[node['id']] = node_id
                label = node.get('v', {}).get('name', node_id)
                nodes.append({'data': {'id': node_id, 'label': label, **node.get('v', {})}})

        if 'edges' in aspect:
            for edge in aspect['edges']:
                source = str(edge['s'])
                target = str(edge['t'])
                edges.append(
                    {
                        'data': {
                            # Cytoscape.js requires ids unique across nodes AND
                            # edges; CX2 numbers them in separate spaces, so
                            # namespace the edge id to avoid colliding with a node.
                            'id': f'e{edge["id"]}',
                            'source': source,
                            'target': target,
                            **edge.get('v', {}),
                        }
                    }
                )

    return {'nodes': nodes, 'edges': edges}
