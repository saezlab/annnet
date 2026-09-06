"""The whole loop, in one call each.

Two tiers, deliberately. :mod:`annnet.experimental.sysbio.methods` is the thin
adapter, for someone building a pipeline who wants each step visible. This is the
composed workflow, for someone reading the documentation who wants to see what
the steps add up to.

Each function here is a **composition and nothing else** — attach, check, run,
write back, all of which exist on their own. There is no arithmetic in this
module, and there is no third code path: change the adapter and these change with
it.
"""

from __future__ import annotations

from typing import Any

from ...core import AnnNet
from ._attach import attach
from ._write_back import write_back


def tf_activity(
    graph: AnnNet,
    adata: Any,
    *,
    aspect: str,
    on: str = 'node_id',
    slice: str | None = None,
    method: str = 'ulm',
    key: str = 'score',
    into_obs: str | None = 'n_active',
    values: str = 'expression',
    attach_first: bool = True,
    **kwargs: Any,
) -> Any:
    """Score regulator activity from a regulon, and land the summary in ``obs``.

    The composed loop: join the measurements onto the graph, check what the
    method needs, score, write the per-regulator answer onto the graph and the
    per-condition summary back into the ``AnnData``.

    Parameters
    ----------
    graph : AnnNet
        Holds the regulon.
    adata : anndata.AnnData
        Holds the measurements; written in place.
    aspect : str
        Which aspect the ``obs`` rows are values of.
    on : str, default "node_id"
        What to match a measured entity against.
    slice : str, optional
        Read the regulon from this slice only.
    method : str, default "ulm"
        A function name under ``dc.mt``.
    key : str, default "score"
        The node-layer attribute the scores go into.
    into_obs : str | None, default "n_active"
        The ``obs`` column the per-condition summary goes into. ``None`` writes
        none.
    values : str, default "expression"
        The attribute name the attached measurements take.
    attach_first : bool, default True
        Join the measurements on first. ``False`` when they are already attached.
    **kwargs
        Passed to :func:`annnet.experimental.sysbio.methods.decoupler.run`.

    Returns
    -------
    DecouplerResult

    Examples
    --------
    >>> result = usecases.tf_activity(G, pdata, aspect='condition', on='symbol')
    >>> pdata.obs['n_active'].head()  # doctest: +SKIP
    """
    from .methods import decoupler

    if attach_first:
        attach(
            graph,
            adata,
            aspect=aspect,
            on=on,
            values={values: None},
            multiplicity=kwargs.pop('multiplicity', 'allow'),
            mapper=kwargs.pop('mapper', None),
        )
    result = decoupler.run(
        graph,
        adata,
        method=method,
        aspect=aspect,
        slice=slice,
        key=key,
        summary=into_obs,
        **kwargs,
    )
    if into_obs and result.summary_key:
        write_back(graph, adata, key=result.summary_key, aspect=aspect, into='obs')
    return result


def causal_subnetwork(
    graph: AnnNet,
    *,
    inputs: str,
    outputs: str,
    aspect: str,
    slice: str | None = None,
    into_slice: str = 'carnival',
    **kwargs: Any,
) -> Any:
    """Fit one signalling sub-network per condition, each into its own slice.

    Parameters
    ----------
    graph : AnnNet
        Holds the signed prior and the per-condition attributes.
    inputs, outputs : str
        The node-layer attributes naming perturbations and measured activity.
        ``outputs`` is usually what :func:`tf_activity` just wrote, which is why
        no dictionary passes between them.
    aspect : str
        Which aspect the conditions are values of.
    slice : str, optional
        Read the prior from this slice only.
    into_slice : str, default "carnival"
        The stem of the per-condition slice names.
    **kwargs
        Passed to :func:`annnet.experimental.sysbio.methods.corneto.run`.

    Returns
    -------
    CarnivalResult

    Examples
    --------
    >>> fit = usecases.causal_subnetwork(  # doctest: +SKIP
    ...     G, inputs='perturbation', outputs='score', aspect='condition'
    ... )
    >>> G.slices.edge_frame(slices=fit.slices, attrs=['activity'])  # doctest: +SKIP
    """
    from .methods import corneto

    return corneto.run(
        graph,
        inputs=inputs,
        outputs=outputs,
        aspect=aspect,
        slice=slice,
        into_slice=into_slice,
        **kwargs,
    )


def activity_to_obs(graph: AnnNet, adata: Any, *, key: str, aspect: str, **kwargs: Any) -> Any:
    """Put one per-condition number from the graph into ``adata.obs``.

    The last step on its own, for a workflow that did the middle differently.

    Parameters
    ----------
    graph : AnnNet
    adata : anndata.AnnData
        Written in place.
    key : str
        The *layer* attribute to read.
    aspect : str
        Which aspect the ``obs`` rows are values of.
    **kwargs
        Passed to :func:`annnet.experimental.sysbio.write_back`.

    Returns
    -------
    numpy.ndarray
    """
    return write_back(graph, adata, key=key, aspect=aspect, into='obs', **kwargs)
