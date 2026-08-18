import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]  # project root
sys.path.insert(0, str(ROOT))
from annnet.io.graphml import (
    from_graphml,
    to_graphml,
)  # GraphML (AnnNet Markup Language)

from .conftest import assert_graphs_equal


class TestGraphMLAdapter:
    """Tests for GraphML adapter (NetworkX wrapper)."""

    def test_simple_round_trip(self, simple_graph, tmpdir_fixture):
        G = simple_graph
        to_graphml(G, tmpdir_fixture / 'graph.graphml', hyperedge_mode='skip')
        G2 = from_graphml(tmpdir_fixture / 'graph.graphml', hyperedge='none')
        assert_graphs_equal(G, G2, check_slices=False, check_hyperedges=False)

    def test_reified_hyperedges(self, complex_graph, tmpdir_fixture):
        G = complex_graph
        to_graphml(G, tmpdir_fixture / 'graph.graphml', hyperedge_mode='reify')
        G2 = from_graphml(tmpdir_fixture / 'graph.graphml', hyperedge='reified')
        assert 'h1' in G2.hyperedge_definitions
        assert 'h2' in G2.hyperedge_definitions

    def test_the_companion_sidecar_carries_what_graphml_cannot(self, complex_graph, tmpdir_fixture):
        """GraphML keeps no edge id, so the mapping that does lives beside it.

        That used to be a ``.manifest.json`` of this adapter's own invention, one
        of three such conventions in the package. It is the one shared sidecar
        now, and there is exactly one file beside the export.
        """
        G = complex_graph
        to_graphml(G, tmpdir_fixture / 'graph.graphml', hyperedge_mode='reify')
        assert (tmpdir_fixture / 'graph.graphml.annnet-sidecar').exists()
        assert not (tmpdir_fixture / 'graph.graphml.manifest.json').exists()
        G2 = from_graphml(tmpdir_fixture / 'graph.graphml', hyperedge='reified')
        assert_graphs_equal(G, G2, check_slices=True, check_hyperedges=True)

    def test_attribute_type_preservation(self, tmpdir_fixture):
        from annnet.core.graph import AnnNet

        G = AnnNet()
        G.add_nodes('A')
        G.attrs.set_node_attrs('A', string_val='text', int_val=42, float_val=3.14, bool_val=True)
        to_graphml(G, tmpdir_fixture / 'graph.graphml')
        G2 = from_graphml(tmpdir_fixture / 'graph.graphml')
        attrs = G2.attrs.get_node_attrs('A')
        assert isinstance(attrs.get('string_val'), str)
        assert isinstance(attrs.get('int_val'), int)
        assert isinstance(attrs.get('float_val'), float)
        assert isinstance(attrs.get('bool_val'), bool)
