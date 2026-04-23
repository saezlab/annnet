from __future__ import annotations

import warnings

from ..core.graph import AnnNet

warnings.filterwarnings('ignore', message='Signature .*numpy.longdouble.*')

try:
    import libsbml
except ImportError:
    libsbml = None

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


def _ensure_boundary_vertices(G, slice: str) -> None:
    G.add_vertices_bulk([BOUNDARY_SOURCE, BOUNDARY_SINK], slice=slice)


def _set_edge_attrs(G, edge_id, **attrs):
    if hasattr(G, 'attrs'):
        return G.attrs.set_edge_attrs(edge_id, **attrs)
    setter = getattr(G, 'set_edge_attrs', None)
    if setter is None:
        raise AttributeError('graph does not expose edge-attribute setters')
    return setter(edge_id, **attrs)


# ── SBML reader ───────────────────────────────────────────────────────────────


def _read_sbml_model(path: str):
    if libsbml is None:
        raise ImportError(
            'python-libsbml is required for SBML import. Install with `pip install python-libsbml`.'
        )

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
    """Create one AnnNet slice per SBML compartment, carrying compartment metadata.

    No-ops gracefully if the graph or model does not support the required API.
    """
    get_compartments = getattr(model, 'getListOfCompartments', None)
    if get_compartments is None:
        return
    has_slice = getattr(getattr(G, 'slices', None), 'has_slice', None) or getattr(
        G, 'has_slice', None
    )
    add_slice = getattr(getattr(G, 'slices', None), 'add_slice', None) or getattr(
        G, 'add_slice', None
    )
    if add_slice is None:
        return

    for c in get_compartments():
        cid = _call(c, 'getId')
        if not cid or cid == default_slice:
            continue
        if has_slice is not None and has_slice(cid):
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
        add_slice(cid, **attrs)


# ── species → vertices ────────────────────────────────────────────────────────


def _register_species(G, model, default_slice: str) -> dict[str, str]:
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
        G.add_vertices_bulk(items, slice=target_slice)

    return sid_to_compartment


# ── main builder ──────────────────────────────────────────────────────────────


def _graph_from_sbml_model(
    model,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
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
            raise RuntimeError('AnnNet class not importable; pass `graph=` explicitly.')
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

        is_boundary = False
        boundary_kind = None
        boundary_node = None

        if not head:
            head = [BOUNDARY_SINK]
            is_boundary = True
            boundary_kind = 'sink'
            boundary_node = BOUNDARY_SINK
            coeffs[BOUNDARY_SINK] = float(sum(-v for v in coeffs.values() if v < 0.0))

        elif not tail:
            tail = [BOUNDARY_SOURCE]
            is_boundary = True
            boundary_kind = 'source'
            boundary_node = BOUNDARY_SOURCE
            coeffs[BOUNDARY_SOURCE] = float(-sum(v for v in coeffs.values() if v > 0.0))

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
            attrs['boundary_node'] = boundary_node

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
    G.add_hyperedges_bulk(hyperedges, slice=slice)

    # ── stoichiometry ─────────────────────────────────────────────────────────
    for rid, coeffs in coeffs_map.items():
        if preserve_stoichiometry:
            G.set_edge_coeffs(rid, coeffs)
        else:
            _set_edge_attrs(G, rid, stoich=coeffs)

    # ── edge attributes ───────────────────────────────────────────────────────
    if edge_attrs_map:
        set_edge_attrs_bulk = getattr(G, 'set_edge_attrs_bulk', None)
        if set_edge_attrs_bulk is not None:
            set_edge_attrs_bulk(edge_attrs_map)
        else:
            for rid, attrs in edge_attrs_map.items():
                _set_edge_attrs(G, rid, **attrs)

    # ── assign reactions to their compartment slices ──────────────────────────
    add_edges_to_slice_bulk = getattr(G, 'add_edges_to_slice_bulk', None)
    add_edge_to_slice = getattr(G, 'add_edge_to_slice', None)
    if add_edges_to_slice_bulk is not None:
        by_slice: dict[str, list[str]] = {}
        for rid, rxn_slices in rxn_slices_map.items():
            for cid in rxn_slices:
                by_slice.setdefault(cid, []).append(rid)
        for cid, rids in by_slice.items():
            try:
                add_edges_to_slice_bulk(cid, rids)
            except Exception:  # noqa: BLE001
                pass
    elif add_edge_to_slice is not None:
        for rid, rxn_slices in rxn_slices_map.items():
            for cid in rxn_slices:
                try:
                    add_edge_to_slice(cid, rid)
                except Exception:  # noqa: BLE001
                    pass

    return G


# ── public entry point ────────────────────────────────────────────────────────


def from_sbml(
    path: str,
    graph: AnnNet | None = None,
    *,
    slice: str = 'default',
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
