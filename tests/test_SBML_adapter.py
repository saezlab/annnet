# test_sbml_adapter_unittest.py
import types
import unittest
from unittest.mock import patch
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))

import annnet.adapters.SBML_adapter as sa

from annnet.adapters.SBML_adapter import (
    _graph_from_sbml_model,
    BOUNDARY_SOURCE,
    BOUNDARY_SINK,
)



# -----------------------
# Fake SBML/libSBML layer
# -----------------------


class DummySpecies:
    def __init__(self, sid: str):
        self._id = sid

    def getId(self):
        return self._id


class DummySpeciesReference:
    def __init__(self, species_id: str, stoich: float = 1.0):
        self._species = species_id
        self._stoich = stoich

    def getSpecies(self):
        return self._species

    def getStoichiometry(self):
        # mimics libSBML: 0.0 may mean "not explicitly set"
        return self._stoich


class DummyReaction:
    def __init__(
        self,
        rid: str | None = None,
        name: str | None = None,
        reversible: bool = False,
        reactants=None,
        products=None,
    ):
        self._id = rid
        self._name = name
        self._reversible = reversible
        self._reactants = reactants or []
        self._products = products or []

    def getId(self):
        return self._id or ""

    def getName(self):
        return self._name or ""

    def getReversible(self):
        return self._reversible

    def getListOfReactants(self):
        return self._reactants

    def getListOfProducts(self):
        return self._products


class DummyModel:
    def __init__(self, species, reactions):
        self._species = species
        self._reactions = reactions

    def getListOfSpecies(self):
        return self._species

    def getListOfReactions(self):
        return self._reactions


class DummyError:
    def __init__(self, severity: int, msg: str):
        self._sev = severity
        self._msg = msg

    def getSeverity(self):
        return self._sev

    def getMessage(self):
        return self._msg


class DummyDoc:
    def __init__(self, model, errors=None):
        self._model = model
        self._errors = errors or []

    def getNumErrors(self):
        return len(self._errors)

    def getError(self, i):
        return self._errors[i]

    def getModel(self):
        return self._model


# ---------------
# Dummy Graph API
# ---------------


class DummyGraph:
    """Minimal Graph stand-in to test adapter logic."""

    def __init__(self, directed: bool = True):
        self.directed = directed
        self.vertices = set()
        self.edges = []

    def add_vertices_bulk(self, ids, slice=None):
        self.vertices.update(ids)

    def add_hyperedge(
        self,
        *,
        head,
        tail,
        slice,
        edge_id,
        edge_directed,
        weight,
    ):
        edge = {
            "id": edge_id,
            "head": list(head),
            "tail": list(tail),
            "slice": slice,
            "directed": edge_directed,
            "weight": weight,
            "attrs": {},
        }
        self.edges.append(edge)
        return edge_id

    def set_edge_attrs(self, edge_id, **attrs):
        for e in self.edges:
            if e["id"] == edge_id:
                e["attrs"].update(attrs)
                return
        raise KeyError(f"Edge {edge_id!r} not found")

    def set_hyperedge_coeffs(self, edge_id, coeffs: dict[str, float]) -> None:
        """Test-only implementation: just stash coeffs on the edge attrs."""
        for e in self.edges:
            if e["id"] == edge_id:
                # same place tests already expect
                e["attrs"]["stoich"] = dict(coeffs)
                return
        raise KeyError(f"Edge {edge_id!r} not found")

# -----------------------
# Tests for _read_sbml_model
# -----------------------


class TestReadSbmlModel(unittest.TestCase):
    def test_read_sbml_model_success(self):
        """_read_sbml_model returns the model if no libSBML errors."""
        model = DummyModel([], [])
        doc = DummyDoc(model=model, errors=[])

        fake_libsbml = types.SimpleNamespace(
            LIBSBML_SEV_ERROR=2,
            readSBML=lambda path: doc,
        )

        with patch.object(sa, "libsbml", fake_libsbml):
            out = sa._read_sbml_model("dummy.xml")

        self.assertIs(out, model)

    def test_read_sbml_model_with_errors_raises(self):
        """_read_sbml_model raises ValueError on libSBML errors."""
        model = DummyModel([], [])
        err = DummyError(severity=2, msg="serious error")
        doc = DummyDoc(model=model, errors=[err])

        fake_libsbml = types.SimpleNamespace(
            LIBSBML_SEV_ERROR=2,
            readSBML=lambda path: doc,
        )

        with patch.object(sa, "libsbml", fake_libsbml):
            with self.assertRaises(ValueError) as ctx:
                sa._read_sbml_model("dummy.xml")

        msg = str(ctx.exception)
        self.assertIn("SBML document contains errors", msg)
        self.assertIn("serious error", msg)


# -----------------------------
# Tests for _graph_from_sbml_model
# -----------------------------


class TestGraphFromSbmlModel(unittest.TestCase):
    def test_basic_stoichiometry(self):
        """Single reaction A -> 2 B gives signed stoichiometry {-1, +2}."""
        species = [DummySpecies("A"), DummySpecies("B")]
        r = DummyReaction(
            rid="R1",
            name="A_to_2B",
            reactants=[DummySpeciesReference("A", 1.0)],
            products=[DummySpeciesReference("B", 2.0)],
        )
        model = DummyModel(species, [r])

        G = DummyGraph()
        G_out = _graph_from_sbml_model(
            model,
            graph=G,
            slice="s",
            preserve_stoichiometry=True,
        )

        self.assertIs(G_out, G)
        self.assertTrue({"A", "B", BOUNDARY_SOURCE, BOUNDARY_SINK}.issubset(G.vertices))

        self.assertEqual(len(G.edges), 1)
        edge = G.edges[0]
        self.assertEqual(edge["id"], "R1")
        self.assertEqual(edge["head"], ["B"])
        self.assertEqual(edge["tail"], ["A"])

        stoich = edge["attrs"]["stoich"]
        self.assertAlmostEqual(stoich["A"], -1.0)
        self.assertAlmostEqual(stoich["B"], 2.0)

        self.assertEqual(edge["attrs"]["name"], "A_to_2B")
        self.assertFalse(edge["attrs"]["reversible"])

    def test_default_stoichiometry_is_one(self):
        """Stoichiometry 0.0 (unset) defaults to 1.0."""
        species = [DummySpecies("X"), DummySpecies("Y")]
        r = DummyReaction(
            rid="R_def",
            reactants=[DummySpeciesReference("X", 0.0)],
            products=[DummySpeciesReference("Y", 0.0)],
        )
        model = DummyModel(species, [r])

        G = DummyGraph()
        _graph_from_sbml_model(model, graph=G, slice="s", preserve_stoichiometry=True)

        edge = G.edges[0]
        stoich = edge["attrs"]["stoich"]
        self.assertAlmostEqual(stoich["X"], -1.0)
        self.assertAlmostEqual(stoich["Y"], 1.0)

    def test_sink_reaction_uses_boundary_sink(self):
        """Reaction A -> ∅ → A -> BOUNDARY_SINK with positive sink coefficient."""
        species = [DummySpecies("A")]
        r = DummyReaction(
            rid="R_sink",
            reactants=[DummySpeciesReference("A", 3.0)],
            products=[],
        )
        model = DummyModel(species, [r])

        G = DummyGraph()
        _graph_from_sbml_model(model, graph=G, slice="slice", preserve_stoichiometry=True)

        self.assertEqual(len(G.edges), 1)
        edge = G.edges[0]

        self.assertEqual(edge["id"], "R_sink")
        self.assertEqual(edge["tail"], ["A"])
        self.assertEqual(edge["head"], [BOUNDARY_SINK])
        self.assertIn(BOUNDARY_SINK, G.vertices)

        stoich = edge["attrs"]["stoich"]
        self.assertAlmostEqual(stoich["A"], -3.0)
        self.assertAlmostEqual(stoich[BOUNDARY_SINK], 3.0)

        self.assertTrue(edge["attrs"]["is_boundary"])
        self.assertEqual(edge["attrs"]["boundary_kind"], "sink")
        self.assertEqual(edge["attrs"]["boundary_node"], BOUNDARY_SINK)

    def test_source_reaction_uses_boundary_source(self):
        """Reaction ∅ -> B → BOUNDARY_SOURCE -> B with negative source coefficient."""
        species = [DummySpecies("B")]
        r = DummyReaction(
            rid="R_source",
            reactants=[],
            products=[DummySpeciesReference("B", 4.0)],
        )
        model = DummyModel(species, [r])

        G = DummyGraph()
        _graph_from_sbml_model(model, graph=G, slice="slice", preserve_stoichiometry=True)

        self.assertEqual(len(G.edges), 1)
        edge = G.edges[0]

        self.assertEqual(edge["id"], "R_source")
        self.assertEqual(edge["tail"], [BOUNDARY_SOURCE])
        self.assertEqual(edge["head"], ["B"])
        self.assertIn(BOUNDARY_SOURCE, G.vertices)

        stoich = edge["attrs"]["stoich"]
        self.assertAlmostEqual(stoich["B"], 4.0)
        self.assertAlmostEqual(stoich[BOUNDARY_SOURCE], -4.0)

        self.assertTrue(edge["attrs"]["is_boundary"])
        self.assertEqual(edge["attrs"]["boundary_kind"], "source")
        self.assertEqual(edge["attrs"]["boundary_node"], BOUNDARY_SOURCE)

    def test_ignores_empty_reaction(self):
        """Reactions with no reactants and no products are ignored."""
        species = [DummySpecies("A")]
        empty_rxn = DummyReaction(rid="R_empty", reactants=[], products=[])
        model = DummyModel(species, [empty_rxn])

        G = DummyGraph()
        _graph_from_sbml_model(model, graph=G, slice="slice", preserve_stoichiometry=True)

        self.assertEqual(len(G.edges), 0)

    def test_skips_reactions_without_id_and_name(self):
        """Reactions without both id and name are skipped."""
        species = [DummySpecies("A"), DummySpecies("B")]
        nameless_rxn = DummyReaction(
            rid=None,
            name=None,
            reactants=[DummySpeciesReference("A", 1.0)],
            products=[DummySpeciesReference("B", 1.0)],
        )
        model = DummyModel(species, [nameless_rxn])

        G = DummyGraph()
        _graph_from_sbml_model(model, graph=G, slice="slice", preserve_stoichiometry=True)

        self.assertEqual(len(G.edges), 0)


# -----------------------
# from_sbml integration
# -----------------------


class TestFromSbmlIntegration(unittest.TestCase):
    def test_from_sbml_integration_with_dummy_graph(self):
        """from_sbml uses libSBML → model → graph_from_sbml_model end-to-end."""
        species = [DummySpecies("A"), DummySpecies("B")]
        rxn = DummyReaction(
            rid="RXN1",
            reactants=[DummySpeciesReference("A", 1.0)],
            products=[DummySpeciesReference("B", 1.0)],
        )
        model = DummyModel(species, [rxn])
        doc = DummyDoc(model=model, errors=[])

        fake_libsbml = types.SimpleNamespace(
            LIBSBML_SEV_ERROR=2,
            readSBML=lambda path: doc,
        )

        G = DummyGraph()
        with patch.object(sa, "libsbml", fake_libsbml):
            G_out = sa.from_sbml("dummy.xml", graph=G, slice="myslice", preserve_stoichiometry=True)

        self.assertIs(G_out, G)
        self.assertEqual(len(G.edges), 1)
        edge = G.edges[0]
        self.assertEqual(edge["id"], "RXN1")
        self.assertEqual(edge["head"], ["B"])
        self.assertEqual(edge["tail"], ["A"])
        stoich = edge["attrs"]["stoich"]
        self.assertAlmostEqual(stoich["A"], -1.0)
        self.assertAlmostEqual(stoich["B"], 1.0)

    def test_from_sbml_raises_if_libsbml_missing(self):
        """from_sbml raises ImportError if libsbml is not available."""
        G = DummyGraph()
        with patch.object(sa, "libsbml", None):
            with self.assertRaises(ImportError):
                sa.from_sbml("dummy.xml", graph=G)


if __name__ == "__main__":
    unittest.main()

