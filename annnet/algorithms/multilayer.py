"""Measures and dynamics on the supra-graph of a multilayer network.

Each function takes the graph first. The supra-graph is the one
``anet.layers.supra()`` builds: one row per node-layer placement.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    'diffusion_step',
    'layer_degree_vectors',
    'multislice_modularity',
    'participation_coefficient',
    'random_walk_step',
    'sweep_coupling_regime',
    'versatility',
]


def layer_degree_vectors(anet, layers=None) -> dict:
    """Intra-layer degree of every node, one vector per layer.

    Returns
    -------
    dict[tuple, tuple[list[int], numpy.ndarray]]
        For each layer coordinate, the supra rows of its placements and their
        degrees in that layer.
    """
    return anet.layers._layer_degree_vectors(layers)


def participation_coefficient(anet, layers=None) -> dict:
    """How evenly each node's degree spreads over the layers.

    Returns
    -------
    dict[str, float]
        ``1 - sum_a (k_a / k)^2`` per node; 0 when all edges sit in one layer.
    """
    return anet.layers._participation_coefficient(layers)


def versatility(anet, layers=None) -> dict:
    """Eigenvector versatility: the dominant eigenvector of the supra-adjacency, summed per node.

    Returns
    -------
    dict[str, float]
    """
    return anet.layers._versatility(layers)


def multislice_modularity(
    anet,
    partition,
    *,
    gamma: float = 1.0,
    omega: float = 1.0,
    include_inter: bool = False,
    layers=None,
) -> float:
    """Multislice modularity of a partition of the placements (Mucha et al., 2010).

    Parameters
    ----------
    partition : Sequence[int]
        One community label per supra row.
    gamma : float, default 1.0
        Resolution.
    omega : float, default 1.0
        Coupling strength.
    include_inter : bool, default False
        Count inter-layer edges as intra-slice ones.
    layers : list, optional
        Restrict to these layers.
    """
    return anet.layers._multislice_modularity(
        partition, gamma=gamma, omega=omega, include_inter=include_inter, layers=layers
    )


def random_walk_step(anet, p, layers=None) -> Any:
    """One step ``p' = p P`` of a random walk on the supra-graph."""
    return anet.layers._random_walk_step(p, layers)


def diffusion_step(anet, x, tau: float = 1.0, laplacian: str = 'comb', layers=None) -> Any:
    """One explicit Euler step ``x' = x - tau L x`` of diffusion on the supra-graph."""
    return anet.layers._diffusion_step(x, tau, laplacian, layers)


def sweep_coupling_regime(anet, scales, metric='algebraic_connectivity', layers=None) -> list:
    """Evaluate a metric of the scaled supra-adjacency at each coupling scale.

    Parameters
    ----------
    scales : Iterable[float]
        Coupling scales.
    metric : str | callable, default "algebraic_connectivity"
        ``"algebraic_connectivity"``, or a callable ``metric(A) -> float``.
    """
    return anet.layers._sweep_coupling_regime(scales, metric, layers)
