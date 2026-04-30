"""SBML and COBRA model ingestion helpers for AnnNet.

Provides:
    from_sbml(path, graph=None, slice="default", preserve_stoichiometry=True) -> AnnNet
    from_cobra_model(model, graph=None, slice="default", preserve_stoichiometry=True) -> AnnNet

This module converts stoichiometric models into directed AnnNet hyperedges.
Reactants and products are represented as edge endpoint sets, with optional
stoichiometric coefficients stored on the edge.

Boundary reactions are represented using dedicated placeholder vertices.
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


def _ensure_vertices(G, vertices: Iterable[str], slice: str | None) -> None:
    # `add_vertices_bulk` exists and handles missing vertices efficiently.
    G.add_vertices_bulk(list(vertices), slice=slice)


BOUNDARY_SOURCE = '__BOUNDARY_SOURCE__'
BOUNDARY_SINK = '__BOUNDARY_SINK__'


def _ensure_boundary_vertices(G, slice: str):
    # idempotent – AnnNet.add_vertices_bulk ignores existing ids
    G.add_vertices_bulk([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)


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

    # Ensure all species + boundary placeholders exist
    G.add_vertices_bulk(list(metabolite_ids), slice=slice)
    _ensure_boundary_vertices(G, slice)

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

        boundary = None
        coeffs = {metabolite_ids[i]: float(v) for i, v in enumerate(col) if v != 0.0}

        if not head:
            # sink: products empty → route to SINK on head side
            head = [BOUNDARY_SINK]
            boundary = ('sink', BOUNDARY_SINK)
            # keep column balanced if we write per-vertex coefficients
            sink_coeff = float(sum(-v for v in col if v < 0))  # sum of absolute reactants
            coeffs[BOUNDARY_SINK] = sink_coeff

        elif not tail:
            # source: reactants empty → route from SOURCE on tail side
            tail = [BOUNDARY_SOURCE]
            boundary = ('source', BOUNDARY_SOURCE)
            source_coeff = float(-sum(v for v in col if v > 0))  # negative sum of products
            coeffs[BOUNDARY_SOURCE] = source_coeff

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
        if boundary:
            kind, bnode = boundary
            G.attrs.set_edge_attrs(
                eid_added, is_boundary=True, boundary_kind=kind, boundary_node=bnode
            )

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
