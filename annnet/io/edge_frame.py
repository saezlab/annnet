"""Build a graph from a table of edges, in one call and with stable ids.

An edge list — one row per relation, a column for each endpoint, a column for
whatever the relation carries — is the most common way a network is handed
around. Turning one into a graph took a loop per row, two calls inside it, an id
minted from the row's *position*, and a list kept on the side so a later
comparison could name the edges again::

    prior_edge_ids = []
    for index, (source, effect, target) in enumerate(rows):
        edge_id = f'prior_{index:02d}'
        G.add_edges(source, target, edge_id=edge_id, slice='prior')
        G.attrs.set_edge_attrs(edge_id, interaction=effect)
        prior_edge_ids.append(edge_id)

Two of those are defects rather than verbosity. A **positional id** means
re-sorting the input renames every edge, so two runs over the same data disagree
about what anything is called. And the **side list** exists only because nothing
on the graph could answer *which edges came from this table* — which slices do,
and did.

An id here is derived from what the edge **is**: its endpoints, where it sits,
and everything the row says about it. Re-sort the table and the ids are the same;
change a recorded value and the id changes with it.
"""

from __future__ import annotations

from typing import Any
from hashlib import blake2b

from ..core import AnnNet
from ._conflict import ON_CONFLICT, resolve_conflicts
from .._support.dataframe_backend import dataframe_columns, dataframe_to_rows

#: How many hex characters of the digest an id carries. Sixty-four bits is far
#: past the point where a collision is likelier than a bug elsewhere, and it
#: keeps an id short enough to read in a table.
ID_WIDTH = 16


def edge_id_for(payload: dict, *, prefix: str = 'e') -> str:
    """A deterministic id for one edge, derived from what it is.

    The payload is serialized in sorted key order, so two rows describing the
    same edge give the same id however the table was sorted, and two rows
    differing in any recorded value do not.

    Parameters
    ----------
    payload : dict
        Everything that identifies the edge — its endpoints, its placement, and
        the values it carries.
    prefix : str, default "e"
        What the id begins with, so a reader can tell where it came from.

    Returns
    -------
    str

    Examples
    --------
    >>> edge_id_for({'source': 'A', 'target': 'B'}) == edge_id_for({'target': 'B', 'source': 'A'})
    True
    """
    material = '\x1f'.join(f'{key}\x1e{payload[key]!r}' for key in sorted(payload))
    digest = blake2b(material.encode('utf-8'), digest_size=ID_WIDTH // 2).hexdigest()
    return f'{prefix}:{digest}'


def _rows_of(frame) -> list[dict]:
    if isinstance(frame, list):
        return [dict(row) for row in frame]
    return dataframe_to_rows(frame)


def _columns_of(frame, rows) -> list[str]:
    if isinstance(frame, list):
        seen: dict[str, None] = {}
        for row in rows:
            seen.update(dict.fromkeys(row))
        return list(seen)
    return list(dataframe_columns(frame))


def add_edges_from_frame(
    graph: AnnNet,
    frame: Any,
    *,
    source: str = 'source',
    target: str = 'target',
    edge_id: str | None = None,
    weight: str | None = None,
    sign: str | None = None,
    attrs: list[str] | None = None,
    directed: bool | None = None,
    slice: str | None = None,
    layer: tuple[str, ...] | None = None,
    id_prefix: str = 'e',
    on_conflict: str = 'error',
) -> list[str]:
    """Add every row of an edge table to a graph.

    Parameters
    ----------
    graph : AnnNet
    frame : DataFrame-like | list[dict]
        One row per edge. Any backend Narwhals reads, or plain dicts.
    source, target : str
        The columns naming the endpoints.
    edge_id : str, optional
        A column holding the id. Without it, an id is **derived** from the row —
        see :func:`edge_id_for`.
    weight : str, optional
        A column holding the incidence weight.
    sign : str, optional
        A column holding a direction of effect, written to an attribute named
        ``"sign"`` whatever the column is called. The *name* is a convention this
        module carries and does not interpret; what it means belongs to whatever
        declares the vocabulary.
    attrs : list[str], optional
        Which other columns to carry as edge attributes. Default: all of them.
    directed : bool, optional
        Directedness for these edges. Defaults to the graph's.
    slice : str, optional
        The slice these edges land in — which is also what answers *which edges
        came from this table*, so no side list is needed.
    layer : tuple[str, ...], optional
        The layer these edges land in.
    id_prefix : str, default "e"
        The prefix derived ids carry.
    on_conflict : str, default "error"
        What to do about an id the graph already holds. See
        :data:`annnet.io.edge_frame.ON_CONFLICT`.

    Returns
    -------
    list[str]
        The edge ids, in row order — which is the mapping back to the rows.

    Raises
    ------
    KeyError
        If a named column is not in the table.
    ValueError
        If ``on_conflict`` is unknown.
    EdgeIdConflict
        If ``on_conflict='error'`` and an id is already taken. Nothing lands.

    Examples
    --------
    >>> ids = add_edges_from_frame(G, table, sign='effect', slice='prior')  # doctest: +SKIP
    """
    if on_conflict not in ON_CONFLICT:
        raise ValueError(f'on_conflict must be one of {ON_CONFLICT}, got {on_conflict!r}')

    rows = _rows_of(frame)
    if not rows:
        return []
    columns = _columns_of(frame, rows)
    for named in (source, target, edge_id, weight, sign):
        if named is not None and named not in columns:
            raise KeyError(f'{named!r} is not a column of this table; it has {columns!r}')

    structural = {source, target, edge_id, weight, sign} - {None}
    carried = list(attrs) if attrs is not None else [c for c in columns if c not in structural]

    specs: list[dict] = []
    ids: list[str] = []
    for row in rows:
        payload: dict[str, Any] = {
            'source': row[source],
            'target': row[target],
        }
        if layer is not None:
            payload['layer'] = tuple(layer)
        if slice is not None:
            payload['slice'] = slice
        for name in carried:
            if row.get(name) is not None:
                payload[name] = row[name]
        if weight is not None:
            payload['weight'] = row[weight]
        if sign is not None:
            payload['sign'] = row[sign]

        this_id = row[edge_id] if edge_id is not None else edge_id_for(payload, prefix=id_prefix)
        ids.append(str(this_id))

        spec: dict[str, Any] = {
            'source': row[source],
            'target': row[target],
            'edge_id': str(this_id),
        }
        if directed is not None:
            spec['directed'] = directed
        if weight is not None:
            spec['weight'] = row[weight]
        if sign is not None:
            spec['sign'] = row[sign]
        for name in carried:
            spec[name] = row.get(name)
        specs.append(spec)

    kept = resolve_conflicts(graph, specs, ids, on_conflict=on_conflict, prefix=id_prefix)
    if kept:
        graph.add_edges(kept, slice=slice, layer=layer)
    return ids


def from_edge_frame(
    frame: Any,
    *,
    directed: bool = True,
    aspects: dict | None = None,
    node_attrs: Any = None,
    **kwargs: Any,
) -> AnnNet:
    """Build a graph from an edge table.

    Parameters
    ----------
    frame : DataFrame-like | list[dict]
        One row per edge.
    directed : bool, default True
        The graph's default direction.
    aspects : dict, optional
        Aspect declarations, as :meth:`LayerAccessor.set_aspects` takes them.
        Declared before the edges land, so a layer named on a row already exists.
    node_attrs : Mapping[str, dict], optional
        Attributes to set on nodes once they exist.
    **kwargs
        Passed to :func:`add_edges_from_frame`.

    Returns
    -------
    AnnNet

    Examples
    --------
    >>> G = from_edge_frame(table, sign='effect', slice='prior')  # doctest: +SKIP
    """
    graph = AnnNet(directed=directed)
    if aspects:
        graph.layers.set_aspects(aspects)
    add_edges_from_frame(graph, frame, directed=directed, **kwargs)
    if node_attrs:
        graph.attrs.set_node_attrs_bulk(node_attrs)
    # A table has no uri to hash, so what is recorded is that one was read and
    # how many rows it had — which is what tells two runs apart.
    graph.provenance.record(
        'edge frame', format='table', reader='from_edge_frame', rows=len(graph.edges())
    )
    return graph
