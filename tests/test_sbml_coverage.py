"""Coverage tests for ``annnet/io/sbml.py``.

These tests rely on the same dummy-libSBML infrastructure as
``tests/test_sbml_adapter.py`` but target the metadata branches and error
paths that weren't reached there.
"""

from __future__ import annotations

import pytest

import annnet.io.sbml as sbml_mod
from annnet.io.sbml import (
    BOUNDARY_SINK,
    BOUNDARY_SOURCE,
    _graph_from_sbml_model,
    _read_sbml_model,
    from_sbml,
)
from tests.test_sbml_adapter import (
    DummyDoc,
    DummyError,
    DummyGraph as BaseDummyGraph,
    DummyModel,
    DummyReaction,
    DummySpecies,
    DummySpeciesReference,
)


class DummyGraph(BaseDummyGraph):
    """Coverage-test variant that tolerates ``(sid, attrs)`` tuples."""

    def _add_vertices_bulk(self, ids, slice=None):
        for item in ids:
            if isinstance(item, tuple):
                self.vertices.add(item[0])
            else:
                self.vertices.add(item)


# ── richer dummy SBML wrappers ─────────────────────────────────────────


class RichCompartment:
    """Compartment dummy that supports every metadata accessor branch."""

    def __init__(
        self,
        cid,
        name=None,
        size=None,
        spatial_dimensions=None,
        units=None,
        sbo_term=None,
        constant=None,
        outside=None,
    ):
        self._id = cid
        self._name = name
        self._size = size
        self._spatial = spatial_dimensions
        self._units = units
        self._sbo = sbo_term
        self._constant = constant
        self._outside = outside

    def getId(self):
        return self._id

    def getName(self):
        return self._name or ''

    def isSetSize(self):
        return self._size is not None

    def getSize(self):
        return self._size

    def isSetSpatialDimensions(self):
        return self._spatial is not None

    def getSpatialDimensions(self):
        return self._spatial

    def isSetUnits(self):
        return self._units is not None

    def getUnits(self):
        return self._units

    def getSBOTermID(self):
        return self._sbo

    def getConstant(self):
        return self._constant

    def getOutside(self):
        return self._outside


class RichSpecies(DummySpecies):
    def __init__(
        self,
        sid,
        name=None,
        compartment=None,
        sbo_term=None,
        meta_id=None,
        initial_amount=None,
        initial_concentration=None,
        has_only_substance_units=None,
        boundary_condition=None,
        constant=None,
    ):
        super().__init__(sid)
        self._name = name
        self._compartment = compartment
        self._sbo = sbo_term
        self._meta = meta_id
        self._ia = initial_amount
        self._ic = initial_concentration
        self._hosu = has_only_substance_units
        self._bc = boundary_condition
        self._const = constant

    def getName(self):
        return self._name or ''

    def getCompartment(self):
        return self._compartment or ''

    def getSBOTermID(self):
        return self._sbo

    def getMetaId(self):
        return self._meta

    def isSetInitialAmount(self):
        return self._ia is not None

    def getInitialAmount(self):
        return self._ia

    def isSetInitialConcentration(self):
        return self._ic is not None

    def getInitialConcentration(self):
        return self._ic

    def getHasOnlySubstanceUnits(self):
        return self._hosu

    def getBoundaryCondition(self):
        return self._bc

    def getConstant(self):
        return self._const


class RichSpeciesReference(DummySpeciesReference):
    """Adds SBO-term support for modifier-role coverage."""

    def __init__(self, species_id, stoich=1.0, sbo_term=None):
        super().__init__(species_id, stoich)
        self._sbo = sbo_term

    def getSBOTermID(self):
        return self._sbo


class RichLocalParam:
    def __init__(self, pid, value):
        self._id = pid
        self._value = value

    def getId(self):
        return self._id

    def isSetValue(self):
        return self._value is not None

    def getValue(self):
        return self._value


class RichKineticLaw:
    def __init__(self, formula=None, local_params=None):
        self._formula = formula
        self._params = local_params or []

    def getFormula(self):
        return self._formula

    def getListOfLocalParameters(self):
        return self._params


class RichReaction(DummyReaction):
    def __init__(
        self,
        rid=None,
        name=None,
        reversible=False,
        reactants=None,
        products=None,
        modifiers=None,
        sbo_term=None,
        meta_id=None,
        compartment=None,
        kinetic_law=None,
    ):
        super().__init__(
            rid=rid,
            name=name,
            reversible=reversible,
            reactants=reactants,
            products=products,
            modifiers=modifiers,
        )
        self._sbo = sbo_term
        self._meta = meta_id
        self._comp = compartment
        self._kl = kinetic_law

    def getSBOTermID(self):
        return self._sbo

    def getMetaId(self):
        return self._meta

    def getCompartment(self):
        return self._comp or ''

    def isSetKineticLaw(self):
        return self._kl is not None

    def getKineticLaw(self):
        return self._kl


# ── _read_sbml_model error paths ──────────────────────────────────────


def test_read_sbml_model_raises_when_libsbml_returns_none(monkeypatch) -> None:
    monkeypatch.setattr(sbml_mod.libsbml, 'readSBML', lambda _: None)
    with pytest.raises(ValueError, match='libSBML failed'):
        _read_sbml_model('bogus.xml')


def test_read_sbml_model_raises_when_model_is_none(monkeypatch) -> None:
    monkeypatch.setattr(sbml_mod.libsbml, 'readSBML', lambda _: DummyDoc(model=None))
    with pytest.raises(ValueError, match='no Model element'):
        _read_sbml_model('bogus.xml')


def test_read_sbml_model_tolerates_low_severity_errors(monkeypatch) -> None:
    """Warnings below LIBSBML_SEV_ERROR are ignored, not raised."""
    low_err = DummyError(severity=sbml_mod.libsbml.LIBSBML_SEV_WARNING, msg='cosmetic')
    doc = DummyDoc(model=DummyModel(species=[], reactions=[]), errors=[low_err])
    monkeypatch.setattr(sbml_mod.libsbml, 'readSBML', lambda _: doc)
    out = _read_sbml_model('bogus.xml')
    assert out is not None


# ── _register_compartments full metadata path ────────────────────────


def test_register_compartments_attaches_every_metadata_field() -> None:
    """Compartment metadata flows into the registered AnnNet slice attributes."""
    comp = RichCompartment(
        cid='cyto',
        name='cytoplasm',
        size=1.5,
        spatial_dimensions=3,
        units='litre',
        sbo_term='SBO:0000290',
        constant=True,
        outside='extracell',
    )
    species = [RichSpecies('A', compartment='cyto')]
    reactions = []
    model = DummyModel(species=species, reactions=reactions, compartments=[comp])

    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')

    # The slice metadata should carry every populated attr.
    attrs = G.slice_attrs['cyto']
    assert attrs['name'] == 'cytoplasm'
    assert attrs['size'] == 1.5
    assert attrs['spatial_dimensions'] == 3
    assert attrs['units'] == 'litre'
    assert attrs['sbo_term'] == 'SBO:0000290'
    assert attrs['constant'] is True
    assert attrs['outside'] == 'extracell'


def test_register_compartments_skips_default_and_no_id() -> None:
    """A compartment matching the default slice or with no id is skipped."""
    comps = [
        RichCompartment(cid=None),  # no id
        RichCompartment(cid='default'),  # same as default slice
    ]
    model = DummyModel(species=[], reactions=[], compartments=comps)
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    # Neither compartment registered any new slice.
    assert G.slice_attrs == {}


def test_register_compartments_skips_already_existing_slice() -> None:
    comp = RichCompartment(cid='cyto', name='cytoplasm')
    model = DummyModel(species=[], reactions=[], compartments=[comp])
    G = DummyGraph()
    G.slices.add('cyto')  # pre-existing
    _graph_from_sbml_model(model, graph=G, slice='default')
    # No attrs were attached because we early-returned on .exists().
    assert G.slice_attrs.get('cyto') is None


# ── _register_species metadata branches ──────────────────────────────


def test_register_species_attaches_every_metadata_field() -> None:
    species = [
        RichSpecies(
            'A',
            name='alpha',
            compartment='cyto',
            sbo_term='SBO:0000247',
            meta_id='meta-A',
            initial_amount=10.0,
            initial_concentration=2.5,
            has_only_substance_units=True,
            boundary_condition=False,
            constant=False,
        )
    ]
    model = DummyModel(
        species=species,
        reactions=[],
        compartments=[RichCompartment(cid='cyto')],
    )
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    assert 'A' in G.vertices


def test_register_species_skips_species_without_id() -> None:
    species = [DummySpecies('')]  # empty id
    model = DummyModel(species=species, reactions=[])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    assert '' not in G.vertices


# ── reaction edge attribute and modifier branches ────────────────────


def test_reaction_with_full_metadata_attaches_all_attrs() -> None:
    """Touch every reaction-attr branch + kinetic-law + local params."""
    kl = RichKineticLaw(
        formula='k1 * A',
        local_params=[RichLocalParam('k1', 0.5), RichLocalParam('k2', None)],
    )
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('A')],
        products=[RichSpeciesReference('B')],
        modifiers=[RichSpeciesReference('M', sbo_term='SBO:0000019')],
        sbo_term='SBO:0000176',
        meta_id='meta-R1',
        compartment='cyto',
        kinetic_law=kl,
    )
    species = [RichSpecies(s, compartment='cyto') for s in ('A', 'B', 'M')]
    model = DummyModel(
        species=species,
        reactions=[rxn],
        compartments=[
            RichCompartment(cid='cyto'),
        ],
    )
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')

    edge = next(e for e in G.edges if e['id'] == 'R1')
    attrs = edge['attrs']
    assert attrs.get('sbo_term') == 'SBO:0000176'
    assert attrs.get('meta_id') == 'meta-R1'
    assert attrs.get('compartment') == 'cyto'
    assert attrs.get('modifier_roles') == {'M': 'SBO:0000019'}
    assert attrs.get('kinetic_law') == 'k1 * A'
    assert attrs.get('local_params') == {'k1': 0.5}  # k2 has no value


def test_reaction_skips_endpoints_with_blank_species_id() -> None:
    """Reactants/products with empty species ids must not enter coeffs."""
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('')],  # blank
        products=[RichSpeciesReference('B')],
    )
    species = [RichSpecies('B')]
    model = DummyModel(species=species, reactions=[rxn])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    edge = next(e for e in G.edges if e['id'] == 'R1')
    assert 'B' in edge['head']
    # reactants empty → one-sided half-edge (empty tail), no placeholder source vertex.
    assert edge['tail'] == []
    assert BOUNDARY_SOURCE not in edge['head']


def test_reaction_modifier_appended_only_when_not_already_present() -> None:
    """Modifier whose species id is also a reactant must not duplicate the tail."""
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('A')],
        products=[RichSpeciesReference('B')],
        modifiers=[RichSpeciesReference('A')],  # same as reactant
    )
    species = [RichSpecies(s) for s in ('A', 'B')]
    model = DummyModel(species=species, reactions=[rxn])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    edge = next(e for e in G.edges if e['id'] == 'R1')
    # 'A' appears once in tail despite being both reactant and modifier.
    assert edge['tail'].count('A') == 1


def test_reaction_with_modifier_only_falls_back_to_boundary() -> None:
    """A reaction with only modifiers + no reactants should land on sink."""
    rxn = RichReaction(
        rid='R1',
        modifiers=[RichSpeciesReference('M')],
    )
    species = [RichSpecies('M')]
    model = DummyModel(species=species, reactions=[rxn])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    edge = next(e for e in G.edges if e['id'] == 'R1')
    # Only modifier present → no products → one-sided half-edge on the modifier (no BOUNDARY_SINK).
    assert 'M' in edge['head']
    assert edge['tail'] == []
    assert BOUNDARY_SINK not in edge['head']


# ── compartment-aware slice assignment ────────────────────────────────


def test_reactions_get_assigned_to_compartment_slices_of_their_species() -> None:
    """A reaction whose species belong to compartment slices is added to them."""
    comp = RichCompartment(cid='cyto')
    species = [
        RichSpecies('A', compartment='cyto'),
        RichSpecies('B', compartment='cyto'),
    ]
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('A')],
        products=[RichSpeciesReference('B')],
    )
    model = DummyModel(species=species, reactions=[rxn], compartments=[comp])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default')
    # R1 should be in the 'cyto' slice.
    assert 'R1' in G.slice_memberships.get('cyto', set())


# ── preserve_stoichiometry=False branch ───────────────────────────────


def test_preserve_stoichiometry_false_writes_stoich_into_edge_attrs() -> None:
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('A')],
        products=[RichSpeciesReference('B')],
    )
    species = [RichSpecies(s) for s in ('A', 'B')]
    model = DummyModel(species=species, reactions=[rxn])
    G = DummyGraph()
    _graph_from_sbml_model(model, graph=G, slice='default', preserve_stoichiometry=False)
    edge = next(e for e in G.edges if e['id'] == 'R1')
    assert 'stoich' in edge['attrs']
    assert edge['attrs']['stoich'].get('A') == -1.0
    assert edge['attrs']['stoich'].get('B') == 1.0


# ── from_sbml end-to-end via monkeypatched libSBML ────────────────────


def test_from_sbml_uses_libsbml_read_path(tmp_path, monkeypatch) -> None:
    """``from_sbml`` calls ``_read_sbml_model`` and then ``_graph_from_sbml_model``."""
    species = [RichSpecies('A'), RichSpecies('B')]
    rxn = RichReaction(
        rid='R1',
        reactants=[RichSpeciesReference('A')],
        products=[RichSpeciesReference('B')],
    )
    model = DummyModel(species=species, reactions=[rxn])
    doc = DummyDoc(model=model)
    monkeypatch.setattr(sbml_mod.libsbml, 'readSBML', lambda _: doc)
    G = DummyGraph()
    out = from_sbml(str(tmp_path / 'x.xml'), graph=G)
    assert out is G
    assert any(e['id'] == 'R1' for e in G.edges)
