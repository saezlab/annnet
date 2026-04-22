from __future__ import annotations

import warnings
from collections.abc import Iterable

warnings.filterwarnings("ignore", message="Signature .*numpy.longdouble.*")

from ..core.graph import AnnNet

try:
    import libsbml
except ImportError:
    libsbml = None

BOUNDARY_SOURCE = "__BOUNDARY_SOURCE__"
BOUNDARY_SINK = "__BOUNDARY_SINK__"


# ── tiny helpers ──────────────────────────────────────────────────────────────


def _call(obj, method, default=None):
    """Call obj.method() if it exists, else return default."""
    fn = getattr(obj, method, None)
    return fn() if fn is not None else default


def _isset(obj, setter_name: str) -> bool:
    """Return True if obj has an isSet* method that returns True."""
    fn = getattr(obj, setter_name, None)
    return bool(fn()) if fn is not None else False


def _ensure_boundary_vertices(G, slice: str) -> None:
    add_vertices = getattr(G, "add_vertices", None)
    if add_vertices is not None:
        add_vertices([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)
        return

    add_vertices_bulk = getattr(G, "add_vertices_bulk", None)
    if add_vertices_bulk is not None:
        add_vertices_bulk([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)
        return

    add_vertex = getattr(G, "add_vertex", None)
    if add_vertex is not None:
        add_vertex(BOUNDARY_SOURCE, slice=slice)
        add_vertex(BOUNDARY_SINK, slice=slice)
        return

    raise AttributeError(
        "graph must provide add_vertices(...), add_vertices_bulk(...), or add_vertex(...)"
    )


# ── SBML reader ───────────────────────────────────────────────────────────────


def _read_sbml_model(path: str):
    if libsbml is None:
        raise ImportError(
            "python-libsbml is required for SBML import. Install with `pip install python-libsbml`."
        )

    doc = libsbml.readSBML(path)
    if doc is None:
        raise ValueError(f"libSBML failed to read file: {path}")

    if doc.getNumErrors() > 0:
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


# ── compartment → slice ───────────────────────────────────────────────────────


def _register_compartments(G, model, default_slice: str) -> None:
    """Create one AnnNet slice per SBML compartment, carrying compartment metadata.
    No-ops gracefully if the graph or model does not support the required API.
    """
    get_compartments = getattr(model, "getListOfCompartments", None)
    if get_compartments is None:
        return
    has_slice = getattr(G, "has_slice", None)
    add_slice = getattr(G, "add_slice", None)
    if add_slice is None:
        return

    for c in get_compartments():
        cid = _call(c, "getId")
        if not cid or cid == default_slice:
            continue
        if has_slice is not None and has_slice(cid):
            continue
        attrs = {}
        name = _call(c, "getName")
        if name:
            attrs["name"] = name
        if _isset(c, "isSetSize"):
            attrs["size"] = _call(c, "getSize")
        if _isset(c, "isSetSpatialDimensions"):
            attrs["spatial_dimensions"] = _call(c, "getSpatialDimensions")
        if _isset(c, "isSetUnits"):
            attrs["units"] = _call(c, "getUnits")
        sbo = _call(c, "getSBOTermID")
        if sbo:
            attrs["sbo_term"] = sbo
        constant = _call(c, "getConstant")
        if constant is not None:
            attrs["constant"] = bool(constant)
        outside = _call(c, "getOutside")  # L2 parent compartment
        if outside:
            attrs["outside"] = outside
        add_slice(cid, **attrs)


# ── species → vertices ────────────────────────────────────────────────────────


def _register_species(G, model, default_slice: str) -> dict[str, str]:
    """Add all species as vertices into their compartment slice.
    Returns a mapping sid -> compartment_id for later use by reactions.
    """
    sid_to_compartment: dict[str, str] = {}
    by_slice: dict[str, list] = {}

    for s in model.getListOfSpecies():
        sid = _call(s, "getId")
        if not sid:
            continue

        compartment = _call(s, "getCompartment") or default_slice
        sid_to_compartment[sid] = compartment

        attrs: dict = {}
        name = _call(s, "getName")
        if name:
            attrs["name"] = name
        if compartment != default_slice:
            attrs["compartment"] = compartment
        sbo = _call(s, "getSBOTermID")
        if sbo:
            attrs["sbo_term"] = sbo
        meta_id = _call(s, "getMetaId")
        if meta_id:
            attrs["meta_id"] = meta_id
        if _isset(s, "isSetInitialAmount"):
            attrs["initial_amount"] = _call(s, "getInitialAmount")
        if _isset(s, "isSetInitialConcentration"):
            attrs["initial_concentration"] = _call(s, "getInitialConcentration")
        has_only = _call(s, "getHasOnlySubstanceUnits")
        if has_only is not None:
            attrs["has_only_substance_units"] = bool(has_only)
        bc = _call(s, "getBoundaryCondition")
        if bc is not None:
            attrs["boundary_condition"] = bool(bc)
        const = _call(s, "getConstant")
        if const is not None:
            attrs["constant"] = bool(const)

        target = compartment if compartment else default_slice
        by_slice.setdefault(target, []).append((sid, attrs) if attrs else sid)

    add_vertices = getattr(G, "add_vertices", None)
    add_vertices_bulk = getattr(G, "add_vertices_bulk", None)
    add_vertex = getattr(G, "add_vertex", None)

    for target_slice, items in by_slice.items():
        if add_vertices is not None:
            add_vertices(items, slice=target_slice)
            continue
        if add_vertices_bulk is not None:
            payload = []
            for item in items:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    sid, attrs = item
                    payload.append({"vertex_id": sid, **attrs})
                else:
                    payload.append(item)
            try:
                add_vertices_bulk(payload, slice=target_slice)
            except TypeError:
                # Test doubles may only accept bare ids.
                fallback_ids = [
                    item[0] if isinstance(item, tuple) and len(item) == 2 else item
                    for item in items
                ]
                add_vertices_bulk(fallback_ids, slice=target_slice)
            continue
        if add_vertex is not None:
            for item in items:
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    sid, attrs = item
                    add_vertex(sid, slice=target_slice, **attrs)
                else:
                    add_vertex(item, slice=target_slice)
            continue
        raise AttributeError(
            "graph must provide add_vertices(...), add_vertices_bulk(...), or add_vertex(...)"
        )

    return sid_to_compartment


# ── main builder ──────────────────────────────────────────────────────────────


def _graph_from_sbml_model(
    model,
    graph: AnnNet | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Build an AnnNet from an SBML model using only libSBML.

    Vertices  : SBML species ids assigned to their compartment slice.
                Boundary placeholders go into the default slice.
    Slices    : One AnnNet slice per SBML compartment (with metadata).
    Hyperedges: reactions — tail = reactants + modifiers, head = products.
    Stoichiometry:
        reactants  : -stoich  (consumed)
        products   : +stoich  (produced)
        modifiers  : -1.0     (regulatory input, not consumed)
    Edge attrs: reversible, name, sbo_term, kinetic_law, local_params,
                modifier_roles, compartment, meta_id, is_boundary, …
    """
    if graph is None:
        if AnnNet is None:
            raise RuntimeError("AnnNet class not importable; pass `graph=` explicitly.")
        G = AnnNet(directed=True)
    else:
        G = graph

    _register_compartments(G, model, slice)
    sid_to_compartment = _register_species(G, model, slice)
    _ensure_boundary_vertices(G, slice)

    hyperedges: list[dict] = []
    coeffs_map: dict[str, dict[str, float]] = {}
    edge_attrs_map: dict[str, dict] = {}
    rxn_slices_map: dict[str, set[str]] = {}

    for rxn in model.getListOfReactions():
        rid = _call(rxn, "getId") or _call(rxn, "getName")
        if not rid:
            continue

        coeffs: dict[str, float] = {}
        tail: list[str] = []
        head: list[str] = []

        for sr in _call(rxn, "getListOfReactants") or []:
            sid = _call(sr, "getSpecies")
            if not sid:
                continue
            sto = _call(sr, "getStoichiometry") or 1.0
            coeffs[sid] = coeffs.get(sid, 0.0) - float(sto)
            tail.append(sid)

        for sr in _call(rxn, "getListOfProducts") or []:
            sid = _call(sr, "getSpecies")
            if not sid:
                continue
            sto = _call(sr, "getStoichiometry") or 1.0
            coeffs[sid] = coeffs.get(sid, 0.0) + float(sto)
            head.append(sid)

        # modifiers: regulatory inputs → tail, coefficient -1
        modifier_roles: dict[str, str] = {}
        for sr in _call(rxn, "getListOfModifiers") or []:
            sid = _call(sr, "getSpecies")
            if not sid:
                continue
            if sid not in coeffs:
                coeffs[sid] = -1.0
            if sid not in tail:
                tail.append(sid)
            sbo = _call(sr, "getSBOTermID")
            if sbo:
                modifier_roles[sid] = sbo

        tail = list(dict.fromkeys(tail))
        head = list(dict.fromkeys(head))

        if not head and not tail:
            continue

        is_boundary = False
        boundary_kind = None
        boundary_node = None

        if not head:
            head = [BOUNDARY_SINK]
            is_boundary = True
            boundary_kind = "sink"
            boundary_node = BOUNDARY_SINK
            coeffs[BOUNDARY_SINK] = float(sum(-v for v in coeffs.values() if v < 0.0))

        elif not tail:
            tail = [BOUNDARY_SOURCE]
            is_boundary = True
            boundary_kind = "source"
            boundary_node = BOUNDARY_SOURCE
            coeffs[BOUNDARY_SOURCE] = float(-sum(v for v in coeffs.values() if v > 0.0))

        hyperedges.append(
            {
                "edge_id": rid,
                "head": head,
                "tail": tail,
                "slice": slice,
                "weight": 1.0,
                "edge_directed": True,
            }
        )
        coeffs_map[rid] = coeffs

        # ── edge attributes ──────────────────────────────────────────────────
        attrs: dict = {"reversible": bool(_call(rxn, "getReversible", False))}

        name = _call(rxn, "getName")
        if name:
            attrs["name"] = name
        sbo = _call(rxn, "getSBOTermID")
        if sbo:
            attrs["sbo_term"] = sbo
        meta_id = _call(rxn, "getMetaId")
        if meta_id:
            attrs["meta_id"] = meta_id
        rxn_comp = _call(rxn, "getCompartment")  # L3 only
        if rxn_comp:
            attrs["compartment"] = rxn_comp
        if modifier_roles:
            attrs["modifier_roles"] = modifier_roles

        if _isset(rxn, "isSetKineticLaw"):
            kl = _call(rxn, "getKineticLaw")
            if kl is not None:
                formula = _call(kl, "getFormula")
                if formula:
                    attrs["kinetic_law"] = formula
                local_params = {}
                for lp in _call(kl, "getListOfLocalParameters") or []:
                    pid = _call(lp, "getId")
                    if pid and _isset(lp, "isSetValue"):
                        local_params[pid] = _call(lp, "getValue")
                if local_params:
                    attrs["local_params"] = local_params

        if is_boundary:
            attrs["is_boundary"] = True
            attrs["boundary_kind"] = boundary_kind
            attrs["boundary_node"] = boundary_node

        edge_attrs_map[rid] = attrs

        # ── compartment slices this reaction touches ─────────────────────────
        rxn_slices: set[str] = set()
        for sid in list(tail) + list(head):
            if sid in (BOUNDARY_SOURCE, BOUNDARY_SINK):
                continue
            comp = sid_to_compartment.get(sid)
            if comp and comp != slice:
                rxn_slices.add(comp)
        rxn_slices_map[rid] = rxn_slices

    # ── bulk insert ───────────────────────────────────────────────────────────
    add_edges = getattr(G, "add_edges", None)
    add_hyperedges_bulk = getattr(G, "add_hyperedges_bulk", None)
    add_hyperedge = getattr(G, "add_hyperedge", None)

    if add_edges is not None:
        add_edges(hyperedges, slice=slice)
    elif add_hyperedges_bulk is not None:
        add_hyperedges_bulk(hyperedges, slice=slice)
    elif add_hyperedge is not None:
        for h in hyperedges:
            add_hyperedge(
                head=h["head"],
                tail=h["tail"],
                slice=h.get("slice", slice),
                edge_id=h["edge_id"],
                directed=h.get("edge_directed", True),
                weight=h.get("weight", 1.0),
            )
    else:
        raise AttributeError(
            "graph must provide add_edges(...), add_hyperedges_bulk(...), or add_hyperedge(...)"
        )

    # ── stoichiometry ─────────────────────────────────────────────────────────
    set_edge_attrs = getattr(G, "set_edge_attrs", None)
    attrs_ns = getattr(G, "attrs", None)

    for rid, coeffs in coeffs_map.items():
        if preserve_stoichiometry:
            G.set_edge_coeffs(rid, coeffs)
        else:
            if attrs_ns is not None:
                attrs_ns.set_edge_attrs(rid, stoich=coeffs)
            elif set_edge_attrs is not None:
                set_edge_attrs(rid, stoich=coeffs)
            else:
                raise AttributeError(
                    "graph must provide attrs.set_edge_attrs(...) or set_edge_attrs(...)"
                )

    # ── edge attributes ───────────────────────────────────────────────────────
    if edge_attrs_map:
        set_edge_attrs_bulk = getattr(G, "set_edge_attrs_bulk", None)
        if set_edge_attrs_bulk is not None:
            set_edge_attrs_bulk(edge_attrs_map)
        else:
            for rid, attrs in edge_attrs_map.items():
                if attrs_ns is not None:
                    attrs_ns.set_edge_attrs(rid, **attrs)
                elif set_edge_attrs is not None:
                    set_edge_attrs(rid, **attrs)
                else:
                    raise AttributeError(
                        "graph must provide attrs.set_edge_attrs(...), set_edge_attrs_bulk(...), or set_edge_attrs(...)"
                    )

    # ── assign reactions to their compartment slices ──────────────────────────
    add_edges_to_slice_bulk = getattr(G, "add_edges_to_slice_bulk", None)
    if add_edges_to_slice_bulk is not None:
        by_slice: dict[str, list[str]] = {}
        for rid, rxn_slices in rxn_slices_map.items():
            for cid in rxn_slices:
                by_slice.setdefault(cid, []).append(rid)
        for cid, rids in by_slice.items():
            try:
                add_edges_to_slice_bulk(cid, rids)
            except Exception:
                pass
    else:
        for rid, rxn_slices in rxn_slices_map.items():
            for cid in rxn_slices:
                try:
                    G.slices.add_edge_to_slice(cid, rid)
                except Exception:
                    pass

    return G


# ── public entry point ────────────────────────────────────────────────────────


def from_sbml(
    path: str,
    graph: AnnNet | None = None,
    *,
    slice: str = "default",
    preserve_stoichiometry: bool = True,
) -> AnnNet:
    """Read an SBML file into an AnnNet hypergraph.

    Parameters
    ----------
    path:
        Path to the .xml / .sbml file.
    graph:
        Optional existing AnnNet to merge into.
    slice:
        AnnNet slice name for default placement. Compartment-specific
        species will additionally appear in their compartment slice.
    preserve_stoichiometry:
        Write signed stoichiometric coefficients into the incidence matrix
        via set_edge_coeffs (reactants negative, products positive,
        modifiers -1). Default True.
    """
    model = _read_sbml_model(path)
    return _graph_from_sbml_model(
        model,
        graph=graph,
        slice=slice,
        preserve_stoichiometry=preserve_stoichiometry,
    )
