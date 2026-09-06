"""Score regulator activity from a signed regulon the graph holds.

decoupler scores, per sample, how active each regulator is, given a signed set of
regulator→target edges and a matrix of measurements.

**The arithmetic is decoupler's, unchanged.** ``tests/test_decoupler_adapter.py``
pins numeric parity against calling ``dc.mt.ulm`` directly on the equivalent
DataFrame. What this adapter contributes is the three things around it: the
declaration before, reading the regulon off the graph, and writing the answer
back **additively**.

Why the spec does not say ``bipartite``
---------------------------------------

A regulon looks bipartite — regulators on one side, targets on the other — and
declaring it would be free rigour. It is not: measured on CollecTRI, **921 of
1,185 regulators are themselves targets**. A spec that declared ``bipartite``
would refuse the resource the method exists to score, and what a user learns from
that is to skip the check.
"""

from __future__ import annotations

from typing import Any
from dataclasses import field, dataclass

import numpy as np

from ....core import AnnNet
from .._support import obs_layers
from ...vocabulary import MethodSpec, check
from ...scverse._shared import require_dependency
from ...._support.dataframe_backend import dataframe_to_rows

#: What decoupler needs a graph to be.
#:
#: Not ``bipartite`` — see the module docstring. Not ``acyclic`` either: a
#: regulator regulating itself is a fact about biology, not a malformed input.
SPEC = MethodSpec(
    name='decoupler',
    requires_edge=('sign',),
    directed=True,
    hyperedges='none',
)

#: The scoring functions this adapter passes through. Any other name decoupler
#: exposes under ``dc.mt`` works too; these are the ones the tests cover.
METHODS = ('ulm', 'mlm', 'ora', 'gsea', 'udt', 'waggr', 'zscore')


@dataclass(slots=True)
class DecouplerResult:
    """What one run produced, and where each part of it went.

    Attributes
    ----------
    method : str
        The ``dc.mt`` function that ran.
    scores : Any
        Samples by regulators, as decoupler returned it.
    padj : Any
        The adjusted p-values, when the method produces them; otherwise ``None``.
    sources : list[str]
        The regulators that were scored.
    dropped : list[str]
        The regulators ``tmin`` left out — returned rather than passed over,
        because one missing for want of measured targets looks exactly like one
        that scored zero.
    slice : str | None
        The slice the scored edges went into.
    key : str
        The node-layer attribute the per-condition scores went into.
    summary_key : str | None
        The layer attribute the per-condition count went into.
    placed : int
        Node-layers created so the scores had somewhere to sit.
    """

    method: str = 'ulm'
    scores: Any = None
    padj: Any = None
    sources: list = field(default_factory=list)
    dropped: list = field(default_factory=list)
    slice: str | None = None
    key: str = 'score'
    summary_key: str | None = None
    placed: int = 0

    def __repr__(self) -> str:
        return (
            f'DecouplerResult(method={self.method!r}, sources={len(self.sources)}, '
            f'dropped={len(self.dropped)}, slice={self.slice!r}, key={self.key!r})'
        )


def _labels(graph, source_attr, target_attr) -> tuple[dict, dict]:
    """Node id -> the name to report each endpoint under."""
    rows = graph._attr_store.node_attr_rows()
    source = (
        {n: a[source_attr] for n, a in rows.items() if a.get(source_attr)} if source_attr else {}
    )
    target = (
        {n: a[target_attr] for n, a in rows.items() if a.get(target_attr)} if target_attr else {}
    )
    return source, target


def regulon(
    graph: AnnNet,
    *,
    slice: str | None = None,
    source_attr: str | None = None,
    target_attr: str | None = None,
    weight: str = 'sign',
) -> Any:
    """The signed regulon this graph holds, in the long frame decoupler reads.

    Parameters
    ----------
    graph : AnnNet
    slice : str, optional
        Read only the edges in this slice.
    source_attr, target_attr : str, optional
        A node attribute to report the endpoint under instead of its node id —
        useful when the graph is keyed by accession and the assay by symbol.
    weight : str, default "sign"
        The edge attribute that becomes decoupler's ``weight`` column.

    Returns
    -------
    pandas.DataFrame
        Columns ``source``, ``target``, ``weight``.

    Raises
    ------
    KeyError
        If ``weight`` is not a column the edge view produces.

    Examples
    --------
    >>> net = decoupler.regulon(G, slice='regulon')  # doctest: +SKIP
    """
    pd = require_dependency('pandas', 'pandas')
    # in_slice filters rows; slice would join attributes onto every row instead.
    frame = graph.views.edges(in_slice=slice, include_hyper=False)
    columns = list(frame.columns)
    if weight not in columns:
        raise KeyError(
            f'{weight!r} is not a column of the edge view; it has {columns!r}. '
            f'An edge attribute appears there once some edge carries it.'
        )
    source_labels, target_labels = _labels(graph, source_attr, target_attr)
    rows = dataframe_to_rows(frame)
    long = pd.DataFrame(
        {
            'source': [source_labels.get(r['source'], r['source']) for r in rows],
            'target': [target_labels.get(r['target'], r['target']) for r in rows],
            'weight': [r[weight] for r in rows],
        }
    )
    return _one_row_per_pair(long)


def _one_row_per_pair(long):
    """Collapse duplicate (source, target) rows, which decoupler refuses."""
    duplicated = long.duplicated(subset=['source', 'target'], keep=False)
    if not duplicated.any():
        return long
    return (
        long.groupby(['source', 'target'], as_index=False, sort=False)['weight']
        .first()
        .reset_index(drop=True)
    )


def run(
    graph: AnnNet,
    adata: Any,
    *,
    method: str = 'ulm',
    aspect: str | None = None,
    layer_key: str | None = None,
    within: dict | None = None,
    slice: str | None = None,
    into_slice: str | None = None,
    source_attr: str | None = None,
    target_attr: str | None = None,
    weight: str = 'sign',
    key: str = 'score',
    summary: str | None = 'n_active',
    active: float = 2.0,
    tmin: int = 5,
    contract: bool = True,
    **kwargs: Any,
) -> DecouplerResult:
    """Score regulator activity, and write the answer onto the graph.

    Parameters
    ----------
    graph : AnnNet
        Holds the regulon.
    adata : anndata.AnnData
        Holds the measurements. ``obs`` are the conditions.
    method : str, default "ulm"
        A function name under ``dc.mt``. See :data:`METHODS`.
    aspect : str, optional
        Which aspect the ``obs`` rows are values of. Required to write scores
        onto the graph; without it only the returned frames are produced.
    layer_key, within :
        As :func:`annnet.experimental.sysbio.attach`.
    slice : str, optional
        Read the regulon from this slice only.
    into_slice : str, optional
        Put the scored edges in a slice of this name. Default: no slice.
    source_attr, target_attr, weight :
        As :func:`regulon`.
    key : str, default "score"
        The node-layer attribute the per-condition scores go into.
    summary : str | None, default "n_active"
        A layer attribute holding how many regulators scored past ``active`` in
        each condition. ``None`` writes none.
    active : float, default 2.0
        The absolute score a regulator counts as active at, for ``summary``.
    tmin : int, default 5
        decoupler's minimum targets per regulator.
    contract : bool, default True
        Check :data:`SPEC` before running, and refuse rather than score
        something that will look plausible.
    **kwargs
        Passed to the decoupler function.

    Returns
    -------
    DecouplerResult

    Raises
    ------
    ValueError
        If ``method`` is not a function decoupler exposes.
    ContractViolation
        If ``contract`` and the graph does not meet :data:`SPEC`.

    Examples
    --------
    >>> result = decoupler.run(G, pdata, aspect='condition', slice='regulon')
    """
    dc = require_dependency('decoupler', "annnet[decoupler] or pip install 'decoupler>=2'")
    if contract:
        check(graph, method=SPEC, strict=True)
    scorer = getattr(dc.mt, method, None)
    if scorer is None:
        raise ValueError(
            f'{method!r} is not a function decoupler exposes under dc.mt; the ones '
            f'this adapter is tested against are {list(METHODS)!r}'
        )

    net = regulon(
        graph, slice=slice, source_attr=source_attr, target_attr=target_attr, weight=weight
    )
    asked = sorted(set(net['source']))

    # decoupler writes into obsm in place, so score a shallow copy: annotating
    # the caller's object is not this adapter's business.
    scored = adata.copy()
    scorer(scored, net, tmin=tmin, **kwargs)
    scores = scored.obsm.get(f'score_{method}')
    padj = scored.obsm.get(f'padj_{method}')
    if scores is None:  # pragma: no cover - a method that names its output otherwise
        found = [k for k in scored.obsm if k.startswith('score_')]
        scores = scored.obsm[found[0]] if found else None

    got = list(scores.columns) if scores is not None else []
    dropped = [source for source in asked if source not in got]

    result = DecouplerResult(
        method=method, scores=scores, padj=padj, sources=got, dropped=dropped, key=key
    )
    if scores is None or aspect is None:
        return result

    coordinates = obs_layers(
        graph, adata, aspect=aspect, layer_key=layer_key, within=within, called='run'
    )
    # A regulator whose own gene was never measured has no node-layer to be
    # scored onto, so the identity comes first and the values after.
    result.placed = graph.layers.place(got, coordinates)
    graph.layers.set_node_attrs_bulk(
        {
            (source, coordinate): float(scores.iloc[row][source])
            for row, coordinate in enumerate(coordinates)
            for source in got
        },
        key=key,
    )
    if summary:
        for row, coordinate in enumerate(coordinates):
            counted = int((np.abs(np.asarray(scores.iloc[row], dtype=float)) > active).sum())
            graph.layers.set_attrs(coordinate, **{summary: counted})
        result.summary_key = summary
    if into_slice:
        scored_edges = _edges_of(graph, slice, got, source_attr)
        graph.slices.add(into_slice, edges=scored_edges, role='scored', method=method)
        result.slice = into_slice
    return result


def _edges_of(graph, slice, sources, source_attr) -> list[str]:
    """The edges whose regulator was scored."""
    labels, _ = _labels(graph, source_attr, None)
    wanted = set(sources)
    found = []
    for row in dataframe_to_rows(graph.views.edges(in_slice=slice, include_hyper=False)):
        name = labels.get(row['source'], row['source'])
        if name in wanted:
            found.append(row['edge_id'])
    return found
