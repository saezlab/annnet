"""Project a graph's results back into the object a workflow reads next.

The last step of the loop: a method annotates the graph, and the answer has to
reach the AnnData the rest of an analysis is written against.

**This is a projection, not a round trip.** Hyperedges, per-member coefficients
and topology do not fit into ``obs``/``var``/``layers`` and never will. What
comes back is the part of the answer that is a number per entity. Anyone
expecting symmetry should learn it here rather than discover it in public.

Three targets, because a result has three shapes: a number per entity per
condition is a matrix (``layers``), one per entity is a column (``var``), and one
per *condition* belongs to the network rather than to any entity — that is
``obs``, and it is the one a single-cell workflow reads next.
"""

from __future__ import annotations

from typing import Any
from collections.abc import Callable

import numpy as np

from ...core import AnnNet
from ._attach import _match
from ._support import column as _column, obs_layers
from ..vocabulary import as_mapper
from ..scverse._shared import require_dependency

#: Where a result may land, and what shape each one is.
TARGETS = ('layers', 'var', 'obs')

#: How several nodes joined to one measured entity become one number.
COMBINERS: dict[str, Callable[[list], Any]] = {
    'mean': lambda found: sum(found) / len(found),
    'sum': sum,
    'first': lambda found: found[0],
    'max': max,
}


def write_back(
    graph: AnnNet,
    adata: Any,
    *,
    key: str,
    aspect: str | None = None,
    layer_key: str | None = None,
    on: str = 'node_id',
    var_key: str | None = None,
    mapper: Any = None,
    into: str = 'layers',
    name: str | None = None,
    within: dict | None = None,
    missing: Any = np.nan,
    combine: str = 'mean',
) -> Any:
    """Write a graph attribute back into an AnnData.

    Parameters
    ----------
    graph : AnnNet
    adata : anndata.AnnData
        Written in place.
    key : str
        The attribute to read off the graph.
    aspect : str, optional
        Which aspect the ``obs`` rows are values of. Required for
        ``into="layers"`` and ``into="obs"``.
    layer_key, on, var_key, mapper, within :
        As :func:`attach`, and they should match the call that attached.
    into : {"layers", "var", "obs"}, default "layers"
        ``"layers"`` writes a conditions-by-entities matrix from node-layer
        values; ``"var"`` one value per entity from node attributes; ``"obs"``
        one value per condition from *layer* attributes, through the same
        condition-to-layer map :func:`attach` built.
    name : str, optional
        What to call it in the AnnData. Default: ``key``.
    missing : Any, default ``numpy.nan``
        What a cell the graph has no value for holds.
    combine : {"mean", "first", "sum", "max"}, default "mean"
        How to reduce when several nodes joined to one measured entity.

    Returns
    -------
    numpy.ndarray
        What was written.

    Raises
    ------
    KeyError
        If a named column is missing, or ``aspect`` is not declared.
    ValueError
        If a policy value is unknown, or ``aspect`` is not given for
        ``into="layers"`` or ``into="obs"``.

    Examples
    --------
    >>> write_back(G, adata, key='activity', aspect='condition')  # doctest: +SKIP
    >>> write_back(G, adata, key='n_active', aspect='condition', into='obs')
    """
    require_dependency('anndata', 'annnet[scverse] or pip install anndata')
    if into not in TARGETS:
        raise ValueError(f'into must be one of {TARGETS}, got {into!r}')
    if combine not in COMBINERS:
        raise ValueError(f'combine must be one of {sorted(COMBINERS)}, got {combine!r}')
    reduce = COMBINERS[combine]
    target = name or key

    # obs is per condition: a layer attribute, and no entity join at all.
    if into == 'obs':
        coordinates = obs_layers(
            graph,
            adata,
            aspect=aspect,
            layer_key=layer_key,
            within=within,
            called=f'into={into!r}',
        )
        found = [graph.layers.attrs(coordinate).get(key, missing) for coordinate in coordinates]
        values = np.asarray(found)
        adata.obs[target] = values
        return values

    entities = _column(adata.var, var_key, adata.var_names, 'var')
    # A permissive policy on purpose. Writing *out* only needs node -> entity,
    # so how a composite's members combine — the question attach had to be told
    # the answer to — does not arise, and refusing here would refuse a graph the
    # attach already accepted.
    matched, _unmapped, _multiplicity, _composites = _match(
        graph, entities, on, {'default': 'first'}, as_mapper(mapper), 'reduce'
    )

    if into == 'var':
        attributes = graph._attr_store.node_attr_rows()
        column = []
        for entity in entities:
            found = [
                attributes[node_id][key]
                for node_id in matched.get(entity, ())
                if attributes.get(node_id, {}).get(key) is not None
            ]
            column.append(reduce(found) if found else missing)
        values = np.asarray(column)
        adata.var[target] = values
        return values

    coordinates = obs_layers(
        graph,
        adata,
        aspect=aspect,
        layer_key=layer_key,
        within=within,
        called=f'into={into!r}',
    )

    # One block per distinct node set rather than one resolver call per cell:
    # the graph-side read is the expensive half, and it is a rectangle.
    wanted = sorted({node_id for ids in matched.values() for node_id in ids})
    matrix = np.full((len(coordinates), len(entities)), missing, dtype=float)
    if wanted:
        block = graph.layers.matrix(key, nodes=wanted, layers=coordinates, missing=np.nan)
        at = {node_id: position for position, node_id in enumerate(block.nodes)}
        for position, entity in enumerate(entities):
            taken = [at[node_id] for node_id in matched.get(entity, ()) if node_id in at]
            if not taken:
                continue
            values = block.values[:, taken]
            for row in range(len(coordinates)):
                present = [v for v in values[row] if not np.isnan(v)]
                if present:
                    matrix[row, position] = reduce(present)
    adata.layers[target] = matrix
    return matrix


def connected(graph: AnnNet) -> bool:
    """Whether this graph holds attached measurements.

    Two lines, and they turn *a connected AnnNet* from a phrase in a docstring
    into something a downstream method can assert on.
    """
    return bool(getattr(graph, '_node_layer_backings', ()) or ())


def measurements(graph: AnnNet) -> list[str]:
    """The names of the attributes an attached array answers for."""
    found: set[str] = set()
    for backing in getattr(graph, '_node_layer_backings', ()) or ():
        found |= backing.names()
    return sorted(found)
