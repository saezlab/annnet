from __future__ import annotations

import contextlib
import io
import math
from collections.abc import Iterable
from typing import Any, Literal

import numpy as np

from .._plotting_backend import select_plot_backend

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
    return f"#{c:02x}{c:02x}{c:02x}"


def _suppress_repr_warnings(g: Any) -> None:
    """Monkey-patch _repr_* methods to hide stderr noise from visualization libs."""
    repr_methods = [m for m in dir(g) if m.startswith("_repr_") and callable(getattr(g, m))]
    for method_name in repr_methods:
        original = getattr(g, method_name)

        def make_wrapper(orig_func):
            def wrapper(*args, **kwargs):
                with contextlib.redirect_stderr(io.StringIO()):
                    return orig_func(*args, **kwargs)

            return wrapper

        setattr(g, method_name, make_wrapper(original))


# Label builders


def build_vertex_labels(graph, key: str | None = None) -> dict[str, str]:
    labels: dict[str, str] = {}
    for vid in graph.vertices():
        if key is None:
            labels[vid] = str(vid)
        else:
            labels[vid] = str(graph.get_attr_vertex(vid, key, default=vid))
    return labels


def build_edge_labels(
    graph,
    *,
    use_weight: bool = True,
    extra_keys: list[str] | None = None,
    layer: str | None = None,
) -> dict[int, str]:
    extra_keys = extra_keys or []
    labels: dict[int, str] = {}
    for j in range(graph.ne):
        eid = graph.idx_to_edge[j]
        parts: list[str] = []
        if use_weight:
            try:
                w = graph.get_effective_edge_weight(eid, slice=layer)
                parts.append(f"w={w:.3g}")
            except Exception:
                pass
        for k in extra_keys:
            v = graph.get_attr_edge(eid, k, default=None)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                parts.append(f"{k}={v}")
        if parts:
            labels[j] = "\\n".join(parts)
    return labels


# Edge style from weights


def edge_style_from_weights(
    graph,
    *,
    layer: str | None = None,
    min_width: float = 0.5,
    max_width: float = 5.0,
    color_mode: Literal["greys", "signed"] = "greys",
) -> dict[int, dict[str, str]]:
    """Compute visual edge styles (pen width and color) from effective weights.

    Parameters
    ----------
    graph : object
        AnnNet-like object exposing `ne`, `idx_to_edge`, and
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
    eidxs = list(range(graph.ne))
    raw_vals: list[float] = []
    for j in eidxs:
        eid = graph.idx_to_edge[j]
        try:
            raw_vals.append(abs(float(graph.get_effective_edge_weight(eid, slice=layer))))
        except Exception:
            raw_vals.append(1.0)

    x = _normalize(raw_vals)
    if np.all(x == x[0]):
        x = np.full_like(x, 0.5)
    styles: dict[int, dict[str, str]] = {}
    for j, xv in zip(eidxs, x):
        pen = min_width + float(xv) * (max_width - min_width)
        if color_mode == "signed":
            eid = graph.idx_to_edge[j]
            try:
                w = float(graph.get_effective_edge_weight(eid, slice=layer))
            except Exception:
                w = 0.0
            color = "firebrick4" if w > 0 else ("dodgerblue4" if w < 0 else "black")
        else:
            color = _greyscale(1.0 - float(xv))  # heavier => darker
        styles[j] = {"penwidth": f"{pen:.3f}", "color": color}
    return styles


# Backends


def _add_nodes_graphviz(
    Gv, node_names: Iterable[str], custom_vertex_attr: dict[str, dict[str, str]] | None = None
):
    custom_vertex_attr = custom_vertex_attr or {}
    for v in node_names:
        attrs = {"shape": "circle"}
        attrs.update(custom_vertex_attr.get(v, {}))
        Gv.node(v, **attrs)


def _add_nodes_pydot(
    Gd, node_names: Iterable[str], custom_vertex_attr: dict[str, dict[str, str]] | None = None
):
    import pydot

    custom_vertex_attr = custom_vertex_attr or {}
    for v in node_names:
        attrs = {"shape": "circle"}
        attrs.update(custom_vertex_attr.get(v, {}))
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
    layout: str = "dot",
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_vertex_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
):
    import graphviz

    Gv = graphviz.Digraph(
        engine=layout, graph_attr=graph_attr, node_attr=node_attr, edge_attr=edge_attr
    )

    # vertices to materialize (union of all endpoints)
    all_nodes: set[str] = set()
    edges_iter = range(graph.ne) if edge_indexes is None else edge_indexes

    # First pass: collect nodes
    for j in edges_iter:
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_graphviz(Gv, sorted(all_nodes), custom_vertex_attr)

    # Second pass: add edges
    for j in range(graph.ne):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        # pick styling overrides
        e_attr = dict()
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        if _is_true_hyperedge(S, T):
            center = f"e_{j}_center"
            Gv.node(center, shape="square", width="0.1", height="0.1", label="")
            for u in S:
                a = {"arrowtail": "none", "arrowhead": "none", "dir": "both"}
                a.update(e_attr)
                Gv.edge(str(u), center, **a)
            for v in T:
                Gv.edge(center, str(v), **e_attr)
        else:
            # binary edge (directed or undirected)
            if S == T:
                # undirected: S == T; extract the two distinct endpoints (or self-loop)
                nodes = list(S)
                uu, vv = (nodes[0], nodes[1]) if len(nodes) == 2 else (nodes[0], nodes[0])
                a = {"arrowhead": "none", "dir": "none"}
                a.update(e_attr)
                Gv.edge(str(uu), str(vv), **a)
            else:
                # directed binary edge
                u = next(iter(S))
                v = next(iter(T))
                head = "normal"
                inter = graph.get_attr_edge(graph.idx_to_edge[j], "interaction", default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = "tee"
                a = {"arrowhead": head}
                a.update(e_attr)
                Gv.edge(str(u), str(v), **a)

    if suppress_warnings:
        _suppress_repr_warnings(Gv)
    if any(_is_true_hyperedge(*graph.get_edge(j)) for j in range(graph.ne)) and graph_attr is None:
        Gv.graph_attr["splines"] = "true"
    return Gv


def to_pydot(
    graph,
    *,
    layout: str = "dot",  # kept for API parity; pydot doesn't use engine here
    graph_attr: dict[str, str] | None = None,
    node_attr: dict[str, str] | None = None,
    edge_attr: dict[str, str] | None = None,
    custom_edge_attr: dict[int, dict[str, str]] | None = None,
    custom_vertex_attr: dict[str, dict[str, str]] | None = None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
):
    import pydot

    Gd = pydot.Dot(graph_type="digraph", **(graph_attr or {}))
    if node_attr:
        Gd.set_node_defaults(**node_attr)
    if edge_attr:
        Gd.set_edge_defaults(**edge_attr)

    all_nodes: set[str] = set()
    edges_iter = range(graph.ne) if edge_indexes is None else edge_indexes
    for j in edges_iter:
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        all_nodes.update(map(str, S | T))

    _add_nodes_pydot(Gd, sorted(all_nodes), custom_vertex_attr)

    for j in range(graph.ne):
        if edge_indexes is not None and j not in edge_indexes:
            continue
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue

        e_attr = dict()
        if custom_edge_attr and j in custom_edge_attr:
            e_attr.update(custom_edge_attr[j])

        if _is_true_hyperedge(S, T):
            center = f"e_{j}_center"
            Gd.add_node(pydot.Node(center, shape="square", width="0.1", height="0.1", label=""))
            for u in S:
                a = {"arrowtail": "none", "arrowhead": "none", "dir": "both"}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(u), center, **a))
            for v in T:
                Gd.add_edge(pydot.Edge(center, str(v), **e_attr))
        else:
            # binary edge (directed or undirected)
            if S == T:
                # undirected: extract the two distinct endpoints (or self-loop)
                nodes = list(S)
                uu, vv = (nodes[0], nodes[1]) if len(nodes) == 2 else (nodes[0], nodes[0])
                a = {"arrowhead": "none", "dir": "none"}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(uu), str(vv), **a))
            else:
                u = next(iter(S))
                v = next(iter(T))
                head = "normal"
                inter = graph.get_attr_edge(graph.idx_to_edge[j], "interaction", default=None)
                if isinstance(inter, (int, float)) and inter < 0:
                    head = "tee"
                a = {"arrowhead": head}
                a.update(e_attr)
                Gd.add_edge(pydot.Edge(str(u), str(v), **a))

    if any(_is_true_hyperedge(*graph.get_edge(j)) for j in range(graph.ne)) and graph_attr is None:
        Gd.set_splines("true")
    return Gd


def to_matplotlib(
    graph,
    *,
    ax=None,
    edge_indexes: list[int] | None = None,
    orphan_edges: bool = True,
    show_vertex_labels: bool = True,
    vertex_label_key: str | None = None,
    show_edge_labels: bool = False,
    edge_label_keys: list[str] | None = None,
    layer: str | None = None,
    node_size: float = 900.0,
    node_color: str = "#f5f5f5",
    edge_color: str = "#333333",
    hyperedge_color: str = "#777777",
):
    """Draw an AnnNet graph with matplotlib and return ``(figure, axes)``.

    This minimal fallback renderer does not require Graphviz, pydot, or
    NetworkX. It places vertices on a circle, draws binary directed edges as
    arrows, undirected binary edges as plain segments, and hyperedges via a
    small center marker connected to incident vertices.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    edges = list(range(graph.ne)) if edge_indexes is None else list(edge_indexes)
    vertices: set[str] = set(map(str, graph.vertices()))
    for j in edges:
        S, T = graph.get_edge(j)
        if not orphan_edges and (len(S) == 0 or len(T) == 0):
            continue
        vertices.update(map(str, S | T))

    ordered_vertices = sorted(vertices)
    n = max(1, len(ordered_vertices))
    positions = {
        vertex: (math.cos(2.0 * math.pi * i / n), math.sin(2.0 * math.pi * i / n))
        for i, vertex in enumerate(ordered_vertices)
    }

    if ordered_vertices:
        xs = [positions[v][0] for v in ordered_vertices]
        ys = [positions[v][1] for v in ordered_vertices]
        ax.scatter(
            xs,
            ys,
            s=node_size,
            c=node_color,
            edgecolors="#222222",
            linewidths=1.0,
            zorder=3,
        )

    if show_vertex_labels:
        labels = build_vertex_labels(graph, key=vertex_label_key)
        for vertex in ordered_vertices:
            x, y = positions[vertex]
            ax.text(x, y, str(labels.get(vertex, vertex)), ha="center", va="center", zorder=4)

    edge_labels = (
        build_edge_labels(graph, use_weight=True, extra_keys=edge_label_keys, layer=layer)
        if show_edge_labels
        else {}
    )
    for j in edges:
        S, T = graph.get_edge(j)
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
            ax.scatter([center[0]], [center[1]], s=80.0, c=hyperedge_color, marker="s", zorder=2)
            for vertex in incident:
                x, y = positions[vertex]
                ax.plot([center[0], x], [center[1], y], color=hyperedge_color, linewidth=1.0)
            if j in edge_labels:
                ax.text(center[0], center[1], edge_labels[j], ha="left", va="bottom", fontsize=8)
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
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": edge_color, "lw": 1.2},
                zorder=1,
            )

        if j in edge_labels:
            ax.text((x0 + x1) / 2.0, (y0 + y1) / 2.0, edge_labels[j], fontsize=8)

    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


# One-call plotting API


def plot(
    graph,
    *,
    backend: Literal["auto", "graphviz", "pydot", "matplotlib"] | None = None,
    layout: str = "dot",
    layer: str | None = None,
    show_edge_labels: bool = False,
    edge_label_keys: list[str] | None = None,
    show_vertex_labels: bool = True,
    vertex_label_key: str | None = None,
    use_weight_style: bool = True,
    orphan_edges: bool = True,
    suppress_warnings: bool = True,
    **kwargs,
):
    """Build a fully styled graph object ready for rendering.

    Parameters
    ----------
    graph : object
        AnnNet-like object with `vertices()`, `get_edge()`, `get_attr_edge()`, etc.
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
    show_vertex_labels : bool, optional
        Whether to label vertices with IDs or attributes. Default is ``True``.
    vertex_label_key : str, optional
        Attribute key for vertex labels. If `None`, uses vertex IDs.
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

    # vertex labels (set via custom_vertex_attr)
    custom_vertex_attr: dict[str, dict[str, str]] | None = None
    if show_vertex_labels:
        vlabels = build_vertex_labels(graph, key=vertex_label_key)
        custom_vertex_attr = {k: {"label": v} for k, v in vlabels.items()}

    if backend == "graphviz":
        G = to_graphviz(
            graph,
            layout=layout,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr", dict(fixedsize="true")),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            orphan_edges=orphan_edges,
            suppress_warnings=suppress_warnings,
        )
        if show_edge_labels:
            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            # reapply labels by regenerating with label overrides
            for j, txt in elabels.items():
                custom_edge_attr.setdefault(j, {})["label"] = txt
            G = to_graphviz(
                graph,
                layout=layout,
                graph_attr=kwargs.get("graph_attr"),
                node_attr=kwargs.get("node_attr", dict(fixedsize="true")),
                edge_attr=kwargs.get("edge_attr"),
                custom_edge_attr=custom_edge_attr,
                custom_vertex_attr=custom_vertex_attr,
                edge_indexes=kwargs.get("edge_indexes"),
                orphan_edges=orphan_edges,
                suppress_warnings=suppress_warnings,
            )
        return G

    elif backend == "pydot":
        G = to_pydot(
            graph,
            layout=layout,
            graph_attr=kwargs.get("graph_attr"),
            node_attr=kwargs.get("node_attr"),
            edge_attr=kwargs.get("edge_attr"),
            custom_edge_attr=custom_edge_attr,
            custom_vertex_attr=custom_vertex_attr,
            edge_indexes=kwargs.get("edge_indexes"),
            orphan_edges=orphan_edges,
        )
        if show_edge_labels:
            # mutate by adding parallel labeled edges
            import pydot

            elabels = build_edge_labels(
                graph, use_weight=True, extra_keys=edge_label_keys, layer=layer
            )
            for j, txt in elabels.items():
                S, T = graph.get_edge(j)
                sv = next(iter(S)) if len(S) else f"e_{j}_source"
                tv = next(iter(T)) if len(T) else f"e_{j}_target"
                G.add_edge(pydot.Edge(str(sv), str(tv), label=txt))
        return G

    elif backend == "matplotlib":
        return to_matplotlib(
            graph,
            ax=kwargs.get("ax"),
            edge_indexes=kwargs.get("edge_indexes"),
            orphan_edges=orphan_edges,
            show_vertex_labels=show_vertex_labels,
            vertex_label_key=vertex_label_key,
            show_edge_labels=show_edge_labels,
            edge_label_keys=edge_label_keys,
            layer=layer,
            node_size=kwargs.get("node_size", 900.0),
            node_color=kwargs.get("node_color", "#f5f5f5"),
            edge_color=kwargs.get("edge_color", "#333333"),
            hyperedge_color=kwargs.get("hyperedge_color", "#777777"),
        )

    else:
        raise ValueError("backend must be 'auto', 'graphviz', 'pydot', or 'matplotlib'")


# Renderer


def render(obj: Any, path: str, format: str = "svg") -> str:
    """Render a Graphviz, Pydot, or matplotlib graph object to disk.

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
    if "graphviz" in kind:
        return obj.render(path, format=fmt, cleanup=True)
    elif "pydot" in kind:
        if fmt == "png":
            obj.write_png(path if path.lower().endswith(".png") else f"{path}.png")
            return path if path.lower().endswith(".png") else f"{path}.png"
        elif fmt in ("svg",):
            obj.write_svg(path if path.lower().endswith(".svg") else f"{path}.svg")
            return path if path.lower().endswith(".svg") else f"{path}.svg"
        else:
            obj.write_raw(path)
            return path
    elif "matplotlib" in kind:
        fig = obj.figure if hasattr(obj, "figure") and not hasattr(obj, "savefig") else obj
        out = path if path.lower().endswith(f".{fmt}") else f"{path}.{fmt}"
        fig.savefig(out, format=fmt, bbox_inches="tight")
        return out
    else:
        raise TypeError(
            "Unknown graph object; expected graphviz.Digraph, pydot.Dot, or matplotlib Figure/Axes"
        )
