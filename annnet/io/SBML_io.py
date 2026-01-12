from __future__ import annotations

import types
import warnings
from collections.abc import Iterable

import numpy as np

warnings.filterwarnings("ignore", message="Signature .*numpy.longdouble.*")

from ..core.graph import AnnNet

try:
    import libsbml  # python-libsbml
except ImportError:  # pragma: no cover
    libsbml = None

# ----------------------- utilities -----------------------


def _monkeypatch_set_hyperedge_coeffs(G) -> bool:
    """Add set_hyperedge_coeffs(edge_id, coeffs) to AnnNet instance if missing.
    Writes per-vertex coefficients into the incidence column.
    Returns True if patch was applied, False if already available.
    """
    if hasattr(G, "set_hyperedge_coeffs"):
        return False  # already there

    def set_hyperedge_coeffs(self, edge_id: str, coeffs: dict[str, float]) -> None:
        col = self.edge_to_idx[edge_id]
        for vid, coeff in coeffs.items():
            row = self.entity_to_idx[vid]
            self._matrix[row, col] = float(coeff)

    G.set_hyperedge_coeffs = types.MethodType(set_hyperedge_coeffs, G)  # type: ignore
    return True


def _ensure_vertices(G, vertices: Iterable[str], slice: str | None) -> None:
    # `add_vertices_bulk` exists and handles missing vertices efficiently.
    G.add_vertices_bulk(list(vertices), slice=slice)


BOUNDARY_SOURCE = "__BOUNDARY_SOURCE__"
BOUNDARY_SINK = "__BOUNDARY_SINK__"


def _ensure_boundary_vertices(G, slice: str) -> None:
    # idempotent – AnnNet.add_vertices_bulk ignores existing ids
    G.add_vertices_bulk([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)


# ---------------- SBML / libSBML-based import ----------------


def _read_sbml_model(path: str):
    """Read SBML with libSBML and return the Model object."""
    if libsbml is None:
        raise ImportError(
            "python-libsbml is required for SBML import without COBRApy. "
            "Install it e.g. with `pip install python-libsbml`."
        )

    doc = libsbml.readSBML(path)
    if doc is None:
        raise ValueError(f"libSBML failed to read file: {path}")

    # Fail on serious SBML errors
    if doc.getNumErrors() > 0:
        # Collect only errors with severity >= ERROR
        msgs = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                msgs.append(f"[{err.getSeverity()}] {err.getMessage()}")
        if msgs:
            raise ValueError("SBML document contains errors:\n" + "\n".join(msgs))

    model = doc.getModel()
    if model is None:
        raise ValueError("SBML file has no Model element.")

    return model


def _graph_from_sbml_model(
    model,
    graph: AnnNet | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Build a AnnNet from an SBML model using only libSBML.

    - Vertices: SBML species ids (plus global boundary source/sink nodes).
    - Hyperedges: reactions, tail = reactants, head = products.
    - Stoichiometry: signed coefficients (reactants negative, products positive).
    """
    if graph is None:
        if AnnNet is None:
            raise RuntimeError("AnnNet class not importable; pass `graph=` explicitly.")
        G = AnnNet(directed=True)
    else:
        G = graph
    # Ensure all species + boundary placeholders exist
    species_ids: list[str] = [s.getId() for s in model.getListOfSpecies()]
    _ensure_vertices(G, species_ids, slice)
    _ensure_boundary_vertices(G, slice)

    # Try to enable per-vertex coefficients
    if preserve_stoichiometry:
        _monkeypatch_set_hyperedge_coeffs(G)

    # Iterate reactions
    for rxn in model.getListOfReactions():
        rid = rxn.getId() or rxn.getName()
        if not rid:
            # skip nameless reactions; they cannot be indexed reliably
            continue

        coeffs: dict[str, float] = {}
        tail: list[str] = []  # reactants
        head: list[str] = []  # products

        # Reactants: negative stoichiometry
        for sr in rxn.getListOfReactants():
            sid = sr.getSpecies()
            if not sid:
                continue
            sto = sr.getStoichiometry()
            if sto == 0.0:
                sto = 1.0  # SBML default if not set
            coeffs[sid] = coeffs.get(sid, 0.0) - float(sto)
            tail.append(sid)

        # Products: positive stoichiometry
        for sr in rxn.getListOfProducts():
            sid = sr.getSpecies()
            if not sid:
                continue
            sto = sr.getStoichiometry()
            if sto == 0.0:
                sto = 1.0
            coeffs[sid] = coeffs.get(sid, 0.0) + float(sto)
            head.append(sid)

        # Deduplicate while preserving order
        tail = list(dict.fromkeys(tail))
        head = list(dict.fromkeys(head))

        if not head and not tail:
            # Ignore truly empty reaction
            continue

        boundary: tuple[str, str] | None = None

        if not head:
            # Sink: products empty -> route to SINK on head side
            head = [BOUNDARY_SINK]
            boundary = ("sink", BOUNDARY_SINK)
            # Sum of absolute reactant stoichiometries
            sink_coeff = float(sum(-v for v in coeffs.values() if v < 0.0))
            coeffs[BOUNDARY_SINK] = sink_coeff

        elif not tail:
            # Source: reactants empty -> route from SOURCE on tail side
            tail = [BOUNDARY_SOURCE]
            boundary = ("source", BOUNDARY_SOURCE)
            source_coeff = float(-sum(v for v in coeffs.values() if v > 0.0))
            coeffs[BOUNDARY_SOURCE] = source_coeff

        eid_added = G.add_hyperedge(
            head=head,
            tail=tail,
            slice=slice,
            edge_id=rid,
            edge_directed=True,
            weight=1.0,
        )

        # Write exact coefficients if supported; else stash as attribute
        if preserve_stoichiometry and hasattr(G, "set_hyperedge_coeffs"):
            G.set_hyperedge_coeffs(eid_added, coeffs)
        else:
            G.set_edge_attrs(eid_added, stoich=coeffs)

        # Basic reaction metadata
        attrs = {
            "name": rxn.getName() or None,
            "reversible": bool(rxn.getReversible()),
        }
        clean = {k: v for k, v in attrs.items() if v is not None}
        if clean:
            G.set_edge_attrs(eid_added, **clean)

        # Mark boundary reactions for easy filtering
        if boundary:
            kind, bnode = boundary
            G.set_edge_attrs(
                eid_added,
                is_boundary=True,
                boundary_kind=kind,
                boundary_node=bnode,
            )

    return G


def from_sbml(
    path: str,
    graph: AnnNet | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Read SBML using python-libsbml.

    Parameters
    ----------
    path:
        Path to the SBML file.
    graph:
        Optional existing AnnNet to add entities/edges to.
    slice:
        AnnNet slice name.
    preserve_stoichiometry:
        If True, store per-vertex stoichiometric coefficients
        (either via `set_hyperedge_coeffs` if available, or as
        an edge attribute `stoich`).
    """
    model = _read_sbml_model(path)
    return _graph_from_sbml_model(
        model,
        graph=graph,
        slice=slice,
        preserve_stoichiometry=preserve_stoichiometry,
    )
