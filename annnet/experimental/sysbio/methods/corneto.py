"""Fit CORNETO's CARNIVAL on a signed signalling network the graph holds.

CARNIVAL takes a signed directed prior, a set of perturbed receptors and a set of
measured downstream activities, and finds the sub-network that best explains the
second from the first.

Run per condition, the answer is one sub-network each — and each goes into its
own slice, so the result is a value per **edge, per condition**::

    G.slices.edge_frame(slices=fit.slices, attrs=['activity'])

``inputs=`` and ``outputs=`` name **node-layer attributes**, so the activities a
previous step wrote are what this fits: no intermediate dictionary, and no way
for the two steps to disagree about which condition is which.

The optimisation is CORNETO's. The solver is HiGHS, which ships with SciPy.

Mapping the answer back
-----------------------

CORNETO preprocesses the network — it adds synthetic edges and drops unreachable
ones — so **position is not a mapping**. Each edge therefore carries its AnnNet
edge id as an attribute, and the synthetic ones carry ``None``, which is what
tells them apart on the way out.
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

#: What CARNIVAL needs a graph to be.
#:
#: Not ``bipartite`` — a cascade is what the method is for. Not ``acyclic``
#: either: it takes a cyclic prior and constrains the *solution*, which is a
#: different demand and one the input does not have to meet.
SPEC = MethodSpec(
    name='corneto.carnival',
    requires_edge=('sign',),
    directed=True,
    hyperedges='none',
)

#: The attribute a selected edge carries inside its condition's slice: ``±1`` for
#: the sign the signal had. An edge absent from a slice carried none.
ACTIVITY = 'activity'

#: Open, and ships with SciPy. Named rather than left to CORNETO's default, so a
#: run cannot silently start depending on a commercial licence.
SOLVER = 'HIGHS'

#: The edge attribute carrying the AnnNet id through CORNETO's preprocessing.
ID_ATTR = 'annnet_id'


@dataclass(slots=True)
class CarnivalResult:
    """What one fit produced, per condition.

    Attributes
    ----------
    conditions : list[str]
        The conditions fitted, in the order the slices are named.
    slices : list[str]
        One slice per condition, holding the edges that carried signal.
    selected : dict[str, int]
        Condition -> how many edges it selected.
    objective : list[float]
        CORNETO's objective values, one list per condition.
    absent : dict[str, list[str]]
        Condition -> nodes carrying a perturbation or a measurement that this
        network does not contain — typically a regulator scored off a *regulon*
        that the *signalling* prior never names.
    key : str
        The edge-in-slice attribute the signs went into.
    """

    conditions: list = field(default_factory=list)
    slices: list = field(default_factory=list)
    selected: dict = field(default_factory=dict)
    objective: list = field(default_factory=list)
    absent: dict = field(default_factory=dict)
    key: str = ACTIVITY

    def __repr__(self) -> str:
        counts = ', '.join(f'{name}={self.selected.get(name, 0)}' for name in self.conditions)
        return f'CarnivalResult({counts or "no conditions"}, key={self.key!r})'


def pkn(
    graph: AnnNet,
    *,
    slice: str | None = None,
    source_attr: str | None = None,
    target_attr: str | None = None,
) -> Any:
    """The signed directed network this graph holds, as a CORNETO graph.

    Each edge carries its AnnNet edge id, because CORNETO's preprocessing adds
    edges and drops others — so position cannot map the answer back.

    Parameters
    ----------
    graph : AnnNet
    slice : str, optional
        Read only the edges in this slice.
    source_attr, target_attr : str, optional
        A node attribute to name the endpoint by instead of its node id.

    Returns
    -------
    corneto.Graph

    Raises
    ------
    ValueError
        If some edge in scope carries no ``sign``, which would silently become
        an activation.
    """
    cn = require_dependency('corneto', "annnet[corneto] or pip install 'corneto>=1.0.0rc5'")
    rows = graph._attr_store.node_attr_rows()
    source_labels = (
        {n: a[source_attr] for n, a in rows.items() if a.get(source_attr)} if source_attr else {}
    )
    target_labels = (
        {n: a[target_attr] for n, a in rows.items() if a.get(target_attr)} if target_attr else {}
    )

    out = cn.Graph()
    unsigned = []
    for row in dataframe_to_rows(graph.views.edges(in_slice=slice, include_hyper=False)):
        sign = row.get('sign')
        if sign is None:
            unsigned.append(row['edge_id'])
            continue
        out.add_edge(
            source_labels.get(row['source'], row['source']),
            target_labels.get(row['target'], row['target']),
            interaction=int(np.sign(float(sign))),
            **{ID_ATTR: row['edge_id']},
        )
    if unsigned:
        raise ValueError(
            f'{len(unsigned)} edge(s) in scope carry no sign (e.g. {unsigned[:5]!r}). '
            f'CARNIVAL reads a sign per edge, and an absent one would become an '
            f'activation without saying so.'
        )
    return out


def _per_condition(graph, coordinate, key, top=None) -> dict:
    """The node-layer attribute ``key`` reads on one layer, as ``{node: ±1}``."""
    resolver = graph.layers.values()
    found = {}
    for node_id in graph.nodes():
        value = resolver.get(node_id, coordinate, key, None)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        found[node_id] = float(value)
    if top is not None and len(found) > top:
        # The perf knob: CARNIVAL's cost grows with the number of measurements it
        # has to explain, and the weakest are the least informative.
        kept = sorted(found, key=lambda n: abs(found[n]), reverse=True)[:top]
        found = {n: found[n] for n in kept}
    return {n: int(np.sign(v)) for n, v in found.items() if v != 0}


def run(
    graph: AnnNet,
    *,
    inputs: str,
    outputs: str,
    aspect: str,
    conditions: list | None = None,
    adata: Any = None,
    layer_key: str | None = None,
    within: dict | None = None,
    slice: str | None = None,
    into_slice: str = 'carnival',
    key: str = ACTIVITY,
    top: int | None = None,
    lambda_reg: float = 0.0,
    solver: str = SOLVER,
    contract: bool = True,
    verbose: bool = False,
) -> CarnivalResult:
    """Fit CARNIVAL once per condition, each into its own slice.

    Parameters
    ----------
    graph : AnnNet
        Holds the signed signalling prior, and the per-condition attributes.
    inputs : str
        The node-layer attribute naming perturbations, as ``±1``.
    outputs : str
        The node-layer attribute naming measured downstream activity. This is
        what a previous scoring step wrote, so the two cannot disagree about
        which condition is which.
    aspect : str
        Which aspect the conditions are values of.
    conditions : list[str], optional
        Which conditions to fit. Default: every value of ``aspect`` that carries
        both an input and an output.
    adata : anndata.AnnData, optional
        Used only to take the condition order from ``obs``, when given.
    layer_key, within :
        As :func:`annnet.experimental.sysbio.attach`.
    slice : str, optional
        Read the prior from this slice only.
    into_slice : str, default "carnival"
        The stem of the per-condition slice names — ``"{into_slice}__{condition}"``.
    key : str, default "activity"
        The edge-in-slice attribute the selected signs go into.
    top : int, optional
        Fit only the strongest ``top`` measurements per condition. CARNIVAL's
        cost grows with the number it has to explain.
    lambda_reg : float, default 0.0
        CORNETO's cross-sample edge regularisation. Higher is sparser.
    solver : str, default "HIGHS"
    contract : bool, default True
        Check :data:`SPEC` before running.
    verbose : bool, default False

    Returns
    -------
    CarnivalResult

    Raises
    ------
    ContractViolation
        If ``contract`` and the graph does not meet :data:`SPEC`.

    Examples
    --------
    >>> fit = corneto.run(  # doctest: +SKIP
    ...     G, inputs='perturbation', outputs=activity.key, aspect='condition'
    ... )
    >>> G.slices.edge_frame(slices=fit.slices, attrs=['activity'])  # doctest: +SKIP
    """
    require_dependency('corneto', "annnet[corneto] or pip install 'corneto>=1.0.0rc5'")
    from corneto.methods.carnival import CarnivalFlow

    if contract:
        check(graph, method=SPEC, strict=True)

    if adata is not None:
        coordinates = obs_layers(
            graph, adata, aspect=aspect, layer_key=layer_key, within=within, called='run'
        )
    else:
        wanted = conditions if conditions is not None else graph.layers.aspect(aspect).values
        declared = tuple(graph.aspects or ())
        from .._support import coordinate as _coordinate

        coordinates = [_coordinate(declared, aspect, value, within) for value in wanted]
    if conditions is not None:
        keep = {str(c) for c in conditions}
        coordinates = [c for c in coordinates if c[declared_index(graph, aspect)] in keep]

    network = pkn(graph, slice=slice)
    held = set(network.V)

    result = CarnivalResult(key=key)
    for coordinate in coordinates:
        name = coordinate[declared_index(graph, aspect)]
        perturbations = _per_condition(graph, coordinate, inputs)
        measurements = _per_condition(graph, coordinate, outputs, top=top)
        missing = sorted((set(perturbations) | set(measurements)) - held)
        perturbations = {k: v for k, v in perturbations.items() if k in held}
        measurements = {k: v for k, v in measurements.items() if k in held}
        if not perturbations or not measurements:
            result.absent[name] = missing
            continue

        model = CarnivalFlow(lambda_reg=lambda_reg)
        problem = model.build(
            network, perturbations=perturbations, transcription_factors=measurements
        )
        problem.solve(solver=solver, verbosity=1 if verbose else 0)

        values = np.asarray(problem.expr['edge_value'].value).reshape(-1)
        processed = model.processed_graph
        selected: dict[str, float] = {}
        for index in range(processed.num_edges):
            edge_id = processed.get_attr_edge(index).get(ID_ATTR)
            if edge_id is None or index >= len(values):
                continue
            signal = float(values[index])
            if signal != 0.0:
                selected[edge_id] = float(np.sign(signal))

        slice_name = f'{into_slice}__{name}'
        graph.slices.add(slice_name, edges=sorted(selected), role='fitted', condition=name)
        for edge_id, signal in selected.items():
            graph.attrs.set_edge_slice_attrs(slice_name, edge_id, **{key: signal})

        result.conditions.append(name)
        result.slices.append(slice_name)
        result.selected[name] = len(selected)
        result.objective.append([float(o.value) for o in problem.objectives])
        result.absent[name] = missing
    return result


def declared_index(graph, aspect: str) -> int:
    """Which position in a layer coordinate one aspect occupies."""
    return tuple(graph.aspects or ()).index(aspect)
