"""SBML ingestion helper for AnnNet.

Provides:
    from_sbml(path, graph=None, slice="default", preserve_stoichiometry=True) -> AnnNet

This module reads SBML through python-libsbml when available and converts
stoichiometric reactions into directed AnnNet hyperedges. Reactants and products
are represented as edge endpoint sets, with optional stoichiometric coefficients
stored on the edge.

Boundary reactions are represented as one-sided (half) edges —
the real species form the single populated endpoint set, with no placeholder
sink/source vertex — and flagged with an ``is_boundary`` edge attribute.
"""

from __future__ import annotations

import warnings

from ..core import AnnNet

warnings.filterwarnings('ignore', message='Signature .*numpy.longdouble.*')

try:
    import libsbml
except ImportError as exc:
    raise ImportError(
        'python-libsbml is required for SBML import. Install with `pip install python-libsbml`.'
    ) from exc


BOUNDARY_SOURCE = '__BOUNDARY_SOURCE__'
BOUNDARY_SINK = '__BOUNDARY_SINK__'


# ── tiny helpers ──────────────────────────────────────────────────────────────


def _call(obj, method, default=None):
    """Call obj.method() if it exists, else return default."""
    fn = getattr(obj, method, None)
    return fn() if fn is not None else default


def _isset(obj, setter_name: str) -> bool:
    """Return True if obj has an isSet* method that returns True."""
    fn = getattr(obj, setter_name, None)
    return bool(fn()) if fn is not None else False


# ── SBML reader ───────────────────────────────────────────────────────────────


def _read_sbml_model(path: str):
    doc = libsbml.readSBML(path)
    if doc is None:
        raise ValueError(f'libSBML failed to read file: {path}')

    if doc.getNumErrors() > 0:
        msgs = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            if err.getSeverity() >= libsbml.LIBSBML_SEV_ERROR:
                msgs.append(f'[{err.getSeverity()}] {err.getMessage()}')
        if msgs:
            raise ValueError('SBML document contains errors:\n' + '\n'.join(msgs))

    model = doc.getModel()
    if model is None:
        raise ValueError('SBML file has no Model element.')

    return model


# ── compartment → slice ───────────────────────────────────────────────────────


def _register_compartments(G, model, default_slice: str) -> None:
    """Create one AnnNet slice per SBML compartment, carrying compartment metadata."""
    for c in model.getListOfCompartments():
        cid = _call(c, 'getId')
        if not cid or cid == default_slice:
            continue
        if G.slices.exists(cid):
            continue
        attrs = {}
        name = _call(c, 'getName')
        if name:
            attrs['name'] = name
        if _isset(c, 'isSetSize'):
            attrs['size'] = _call(c, 'getSize')
        if _isset(c, 'isSetSpatialDimensions'):
            attrs['spatial_dimensions'] = _call(c, 'getSpatialDimensions')
        if _isset(c, 'isSetUnits'):
            attrs['units'] = _call(c, 'getUnits')
        sbo = _call(c, 'getSBOTermID')
        if sbo:
            attrs['sbo_term'] = sbo
        constant = _call(c, 'getConstant')
        if constant is not None:
            attrs['constant'] = bool(constant)
        outside = _call(c, 'getOutside')  # L2 parent compartment
        if outside:
            attrs['outside'] = outside
        G.slices.add(cid, **attrs)


# ── species → vertices ────────────────────────────────────────────────────────


def _register_species(G, model, default_slice: str, layer: str | None = None) -> dict[str, str]:
    """Add all species as vertices into their compartment slice.

    Returns a mapping sid -> compartment_id for later use by reactions.
    """
    sid_to_compartment: dict[str, str] = {}
    by_slice: dict[str, list] = {}

    for s in model.getListOfSpecies():
        sid = _call(s, 'getId')
        if not sid:
            continue

        compartment = _call(s, 'getCompartment') or default_slice
        sid_to_compartment[sid] = compartment

        attrs: dict = {}
        name = _call(s, 'getName')
        if name:
            attrs['name'] = name
        if compartment != default_slice:
            attrs['compartment'] = compartment
        sbo = _call(s, 'getSBOTermID')
        if sbo:
            attrs['sbo_term'] = sbo
        meta_id = _call(s, 'getMetaId')
        if meta_id:
            attrs['meta_id'] = meta_id
        if _isset(s, 'isSetInitialAmount'):
            attrs['initial_amount'] = _call(s, 'getInitialAmount')
        if _isset(s, 'isSetInitialConcentration'):
            attrs['initial_concentration'] = _call(s, 'getInitialConcentration')
        has_only = _call(s, 'getHasOnlySubstanceUnits')
        if has_only is not None:
            attrs['has_only_substance_units'] = bool(has_only)
        bc = _call(s, 'getBoundaryCondition')
        if bc is not None:
            attrs['boundary_condition'] = bool(bc)
        const = _call(s, 'getConstant')
        if const is not None:
            attrs['constant'] = bool(const)

        target = compartment if compartment else default_slice
        by_slice.setdefault(target, []).append((sid, attrs) if attrs else sid)

    for target_slice, items in by_slice.items():
        kw = {'slice': target_slice}
        if layer is not None:
            kw['layer'] = layer
        G._add_vertices_bulk(items, **kw)

    return sid_to_compartment


# ── main builder ──────────────────────────────────────────────────────────────


def _graph_from_sbml_model(
    model,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
    preserve_stoichiometry: bool = True,
    layer: str | None = None,
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
    G = AnnNet(directed=True) if graph is None else graph

    _register_compartments(G, model, slice)
    sid_to_compartment = _register_species(G, model, slice, layer=layer)
    # Boundary reactions are modelled as one-sided (half) hyperedges, so no
    # placeholder sink/source vertices are created.

    hyperedges: list[dict] = []
    coeffs_map: dict[str, dict[str, float]] = {}
    edge_attrs_map: dict[str, dict] = {}
    rxn_slices_map: dict[str, set[str]] = {}

    for rxn in model.getListOfReactions():
        rid = _call(rxn, 'getId') or _call(rxn, 'getName')
        if not rid:
            continue

        coeffs: dict[str, float] = {}
        tail: list[str] = []
        head: list[str] = []

        for sr in _call(rxn, 'getListOfReactants') or []:
            sid = _call(sr, 'getSpecies')
            if not sid:
                continue
            sto = _call(sr, 'getStoichiometry') or 1.0
            coeffs[sid] = coeffs.get(sid, 0.0) - float(sto)
            tail.append(sid)

        for sr in _call(rxn, 'getListOfProducts') or []:
            sid = _call(sr, 'getSpecies')
            if not sid:
                continue
            sto = _call(sr, 'getStoichiometry') or 1.0
            coeffs[sid] = coeffs.get(sid, 0.0) + float(sto)
            head.append(sid)

        # modifiers: regulatory inputs → tail, coefficient -1
        modifier_roles: dict[str, str] = {}
        for sr in _call(rxn, 'getListOfModifiers') or []:
            sid = _call(sr, 'getSpecies')
            if not sid:
                continue
            if sid not in coeffs:
                coeffs[sid] = -1.0
            if sid not in tail:
                tail.append(sid)
            sbo = _call(sr, 'getSBOTermID')
            if sbo:
                modifier_roles[sid] = sbo

        tail = list(dict.fromkeys(tail))
        head = list(dict.fromkeys(head))

        if not head and not tail:
            continue

        # Boundary reactions keep one side empty (a single real endpoint set), so the
        # incidence column carries exactly the real species coefficients — no placeholder
        # vertex, no phantom cross-species coupling through a shared sink/source in adjacency.
        is_boundary = False
        boundary_kind = None

        if not head:
            is_boundary = True
            boundary_kind = 'sink'  # products empty → species consumed by the environment

        elif not tail:
            is_boundary = True
            boundary_kind = 'source'  # reactants empty → species drawn from the environment

        if is_boundary:
            # One-sided (half) edge: real species become the single populated endpoint
            # set (head), no placeholder partner. Signs live in ``coeffs``; the
            # sink/source distinction is preserved on the is_boundary edge attribute.
            head, tail = list(head or tail), []

        hyperedges.append(
            {
                'edge_id': rid,
                'head': head,
                'tail': tail,
                'slice': slice,
                'weight': 1.0,
                'edge_directed': True,
            }
        )
        coeffs_map[rid] = coeffs

        # ── edge attributes ──────────────────────────────────────────────────
        attrs: dict = {'reversible': bool(_call(rxn, 'getReversible', False))}

        name = _call(rxn, 'getName')
        if name:
            attrs['name'] = name
        sbo = _call(rxn, 'getSBOTermID')
        if sbo:
            attrs['sbo_term'] = sbo
        meta_id = _call(rxn, 'getMetaId')
        if meta_id:
            attrs['meta_id'] = meta_id
        rxn_comp = _call(rxn, 'getCompartment')  # L3 only
        if rxn_comp:
            attrs['compartment'] = rxn_comp
        if modifier_roles:
            attrs['modifier_roles'] = modifier_roles

        if _isset(rxn, 'isSetKineticLaw'):
            kl = _call(rxn, 'getKineticLaw')
            if kl is not None:
                formula = _call(kl, 'getFormula')
                if formula:
                    attrs['kinetic_law'] = formula
                local_params = {}
                for lp in _call(kl, 'getListOfLocalParameters') or []:
                    pid = _call(lp, 'getId')
                    if pid and _isset(lp, 'isSetValue'):
                        local_params[pid] = _call(lp, 'getValue')
                if local_params:
                    attrs['local_params'] = local_params

        if is_boundary:
            attrs['is_boundary'] = True
            attrs['boundary_kind'] = boundary_kind

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
    kw = {'slice': slice}
    if layer is not None:
        kw['layer'] = layer
    G.add_hyperedges_bulk(hyperedges, **kw)

    # ── stoichiometry ─────────────────────────────────────────────────────────
    for rid, coeffs in coeffs_map.items():
        if preserve_stoichiometry:
            G.set_edge_coeffs(rid, coeffs)
        else:
            G.attrs.set_edge_attrs(rid, stoich=coeffs)

    # ── edge attributes ───────────────────────────────────────────────────────
    if edge_attrs_map:
        G.attrs.set_edge_attrs_bulk(edge_attrs_map)

    # ── assign reactions to their compartment slices ──────────────────────────
    by_slice: dict[str, list[str]] = {}
    existing_slices = set(G.slices.list(include_default=True))
    for rid, rxn_slices in rxn_slices_map.items():
        for cid in rxn_slices:
            if cid in existing_slices:
                by_slice.setdefault(cid, []).append(rid)
    for cid, rids in by_slice.items():
        G.slices.add_edges(cid, rids)

    return G


# ── public entry point ────────────────────────────────────────────────────────


def from_sbml(
    path: str,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
    preserve_stoichiometry: bool = True,
    layer: str | None = None,
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
        layer=layer,
    )
