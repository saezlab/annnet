"""SBML and COBRA model ingestion helpers for AnnNet.

Provides:
    from_sbml(path, graph=None, slice="default", preserve_stoichiometry=True) -> AnnNet
    from_cobra_model(model, graph=None, slice="default", preserve_stoichiometry=True) -> AnnNet

This module converts stoichiometric models into directed AnnNet hyperedges.
Reactants and products are represented as edge endpoint sets, with optional
stoichiometric coefficients stored on the edge.

Boundary reactions are represented as one-sided (half) edges —
the real metabolites form the single populated endpoint set, with no placeholder
sink/source vertex — and flagged with an ``is_boundary`` edge attribute.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np

from ..core import AnnNet

try:
    from cobra.io import read_sbml_model  # type: ignore
    from cobra.util.array import create_stoichiometric_matrix  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'COBRApy not installed; install cobra or use the libsbml-based IO module instead.'
    ) from e


warnings.filterwarnings('ignore', message='Signature .*numpy.longdouble.*')


BOUNDARY_SOURCE = '__BOUNDARY_SOURCE__'
BOUNDARY_SINK = '__BOUNDARY_SINK__'


def _ensure_vertices(G, vertices: Iterable[str], slice: str | None) -> None:
    # Internal bulk vertex insertion handles missing vertices efficiently.
    G._add_vertices_bulk(list(vertices), slice=slice)


def _graph_from_stoich(
    S: np.ndarray,
    metabolite_ids: Sequence[str],
    reaction_ids: Sequence[str],
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    G = AnnNet(directed=True) if graph is None else graph

    # Ensure all species exist. Boundary reactions are modelled as one-sided
    # (half) edges rather than routed through placeholder sink/source vertices.
    G._add_vertices_bulk(list(metabolite_ids), slice=slice)

    m, n = S.shape
    assert m == len(metabolite_ids)
    assert n == len(reaction_ids)

    for j, eid in enumerate(reaction_ids):
        col = S[:, j]
        head = [metabolite_ids[i] for i, v in enumerate(col) if v > 0]  # products
        tail = [metabolite_ids[i] for i, v in enumerate(col) if v < 0]  # reactants

        if not head and not tail:
            # Truly empty column; ignore
            continue

        # Boundary reactions keep one side empty (a single real endpoint), so the
        # incidence column carries exactly the real metabolite coefficients — no
        # placeholder vertex, no phantom cross-metabolite coupling in adjacency.
        boundary_kind = None
        coeffs = {metabolite_ids[i]: float(v) for i, v in enumerate(col) if v != 0.0}

        if not head:
            boundary_kind = 'sink'  # products empty → metabolites consumed by the environment
        elif not tail:
            boundary_kind = 'source'  # reactants empty → metabolites drawn from the environment

        if boundary_kind:
            # One-sided (half) edge: the real metabolites become the single populated
            # endpoint set (no placeholder partner). Signs live in ``coeffs``; the
            # sink/source distinction is preserved on the is_boundary edge attribute.
            eid_added = G.add_edges(
                src=list(head or tail),
                tgt=None,
                slice=slice,
                edge_id=eid,
                directed=True,
                weight=1.0,
            )
        else:
            eid_added = G.add_edges(
                src=head,
                tgt=tail,
                slice=slice,
                edge_id=eid,
                directed=True,
                weight=1.0,
            )

        if preserve_stoichiometry:
            G.set_edge_coeffs(eid_added, coeffs)
        else:
            G.attrs.set_edge_attrs(eid_added, stoich=coeffs)

        # mark boundary reactions for easy filtering
        if boundary_kind:
            G.attrs.set_edge_attrs(eid_added, is_boundary=True, boundary_kind=boundary_kind)

    return G


# ---------------- COBRA-based import ----------------


def from_cobra_model(
    model,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Convert a COBRApy model to AnnNet. Requires cobra.util.array.create_stoichiometric_matrix.

    Edge attributes added: name, default_lb, default_ub, gpr (Gene-Protein-Reaction rule [GPR]).
    """

    S = create_stoichiometric_matrix(model)
    rxn_ids = [rxn.id for rxn in model.reactions]
    met_ids = [met.id for met in model.metabolites]

    G = _graph_from_stoich(
        S, met_ids, rxn_ids, graph=graph, slice=slice, preserve_stoichiometry=preserve_stoichiometry
    )

    # Attach per-reaction metadata via set_edge_attrs (AnnNet API)
    for rxn in model.reactions:
        eid = rxn.id
        attrs = {
            'name': getattr(rxn, 'name', None),
            'default_lb': getattr(rxn, 'lower_bound', None),
            'default_ub': getattr(rxn, 'upper_bound', None),
            'gpr': getattr(rxn, 'gene_reaction_rule', None),
        }
        # drop Nones
        clean = {k: v for k, v in attrs.items() if v is not None}
        if clean:
            G.attrs.set_edge_attrs(eid, **clean)

    return G


def from_sbml(
    path: str,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Read SBML using COBRApy if available."""

    model = read_sbml_model(path)
    return from_cobra_model(
        model, graph=graph, slice=slice, preserve_stoichiometry=preserve_stoichiometry
    )
